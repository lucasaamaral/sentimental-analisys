"""Stage 2: fine-tune FinBERT-PT-BR with validation-based model selection.

Workflow:
    1. Read labeled_samples/labeled_samples.csv filtering rows where human_label is filled in
    2. Create fixed stratified train/validation/test splits
    3. Search hyperparameters using validation metrics only
    4. Train the selected configuration as a three-seed ensemble
    5. Evaluate the selected ensemble once on the independent test split
    6. Save independent test metrics and deployment metadata
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from classification_workflow.config.paths import (
    BASE_MODEL_NAME as MODEL_NAME,
    ENSEMBLE_MODEL_DIRS,
    FINETUNED_MODEL_DIR as OUTPUT_MODEL,
    LABELED_SAMPLES_PATH as SAMPLE_PATH,
    MODELS_DIR,
    TRAINING_METADATA_PATH,
)
from classification_workflow.data.data_splits import (
    HOLDOUT_SEED,
    HOLDOUT_TEST_SIZE,
    VALIDATION_SEED,
    VALIDATION_TEST_SIZE,
    cleanup_temporary_split_files,
    create_fixed_train_validation_test_split,
    load_labelled_dataframe,
)
from classification_workflow.ml.evaluation import evaluate_trainer_on_dataframe
from classification_workflow.ml.inference import (
    attach_base_model_predictions,
    describe_device,
    load_sequence_classifier,
)
from classification_workflow.ml.trainer import (
    CANDIDATE_CONFIGS,
    ENSEMBLE_SEEDS,
    TrainingConfig,
    _setup_training_prerequisites,
    average_seed_predictions,
    classification_metrics,
    labels_from_dataframe,
    predictions_to_labels,
    train_single_seed,
)

warnings.filterwarnings("ignore")

SEARCH_MODEL_DIR = MODELS_DIR / "hyperparameter-search"


# =========================================================================
# Metadata Persistence
# =========================================================================


def save_training_metadata(
    full_dataframe: pd.DataFrame,
    model_train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    selected_config: TrainingConfig,
    search_results: list[dict],
    holdout_summary: dict,
) -> None:
    training_metadata = {
        "selected_strategy": asdict(selected_config),
        "selection_method": "validation_macro_f1_grid_search",
        "search_results": search_results,
        "final_training_epochs": selected_config.epochs,
        "selected_training_seed": selected_config.training_seed,
        "ensemble_mode": True,
        "ensemble_seeds": list(ENSEMBLE_SEEDS),
        "ensemble_model_dirs": [str(path) for path in ENSEMBLE_MODEL_DIRS],
        "decision_method": "identity_argmax",
        "n_total_labelled_examples": int(len(full_dataframe)),
        "n_model_selection_examples": int(len(model_train_df) + len(validation_df)),
        "n_model_training_examples": int(len(model_train_df)),
        "n_validation_examples": int(len(validation_df)),
        "n_holdout_examples": int(len(holdout_df)),
        "sample_path": str(SAMPLE_PATH),
        "holdout_test_size": HOLDOUT_TEST_SIZE,
        "holdout_seed": HOLDOUT_SEED,
        "validation_size_of_model_selection": VALIDATION_TEST_SIZE,
        "validation_seed": VALIDATION_SEED,
        "temporary_split_files_removed": True,
        "independent_holdout_metrics": {
            "base_accuracy": holdout_summary["base_model"]["accuracy"],
            "base_macro_f1": holdout_summary["base_model"]["macro_f1"],
            "finetuned_accuracy": holdout_summary["finetuned_model"]["accuracy"],
            "finetuned_macro_f1": holdout_summary["finetuned_model"]["macro_f1"],
        },
    }

    with open(TRAINING_METADATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(training_metadata, handle, ensure_ascii=False, indent=2)


# =========================================================================
# Orchestration Phases
# =========================================================================


def prepare_scored_splits() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Score samples with the base model and create train/validation/test splits."""
    print(f"\nLoading base model for baseline metrics: {MODEL_NAME}")
    base_tokenizer, base_model, base_device = load_sequence_classifier(MODEL_NAME)
    print(f"Base model device: {describe_device(base_device)}")

    full_dataframe = load_labelled_dataframe(SAMPLE_PATH)
    full_dataframe = attach_base_model_predictions(
        full_dataframe,
        base_tokenizer,
        base_model,
        base_device,
    )

    model_train_df, validation_df, holdout_df = create_fixed_train_validation_test_split(
        full_dataframe
    )

    del base_tokenizer, base_model
    if base_device.type == "cuda":
        torch.cuda.empty_cache()

    return full_dataframe, model_train_df, validation_df, holdout_df


def evaluate_validation_logits(
    validation_df: pd.DataFrame,
    validation_logits: np.ndarray,
) -> dict[str, float]:
    return classification_metrics(
        labels_from_dataframe(validation_df),
        predictions_to_labels(validation_logits),
    )


def select_hyperparameters(
    model_train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    tokenizer,
) -> tuple[TrainingConfig, list[dict]]:
    """Train candidate configurations and select the best by validation macro-F1."""
    train_dataset, validation_dataset, data_collator, class_weights = (
        _setup_training_prerequisites(model_train_df, validation_df, tokenizer)
    )

    search_results: list[dict] = []
    for index, training_config in enumerate(CANDIDATE_CONFIGS, start=1):
        print(
            f"\n=== Hyperparameter search {index}/{len(CANDIDATE_CONFIGS)}: "
            f"{training_config.name} ==="
        )
        predictions = train_single_seed(
            model_dir=SEARCH_MODEL_DIR / training_config.name,
            seed=training_config.training_seed,
            training_config=training_config,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            data_collator=data_collator,
            class_weights=class_weights,
            reference_df=model_train_df,
            evaluation_df=validation_df,
            tokenizer=tokenizer,
            seed_label=f"validation-search seed={training_config.training_seed}",
        )
        metrics = evaluate_validation_logits(validation_df, predictions.evaluation)
        result = {
            "rank_input_order": index,
            "config": asdict(training_config),
            "validation_accuracy": metrics["accuracy"],
            "validation_macro_f1": metrics["macro_f1"],
        }
        search_results.append(result)
        print(
            f"Validation result: accuracy={metrics['accuracy']:.1%} | "
            f"macro_f1={metrics['macro_f1']:.3f}"
        )

    ranked_results = sorted(
        search_results,
        key=lambda item: (
            item["validation_macro_f1"],
            item["validation_accuracy"],
            -item["rank_input_order"],
        ),
        reverse=True,
    )
    best_name = ranked_results[0]["config"]["name"]
    selected_config = next(config for config in CANDIDATE_CONFIGS if config.name == best_name)

    print("\nValidation hyperparameter ranking:")
    for rank, result in enumerate(ranked_results, start=1):
        print(
            f"{rank}. {result['config']['name']} | "
            f"accuracy={result['validation_accuracy']:.1%} | "
            f"macro_f1={result['validation_macro_f1']:.3f}"
        )
    print(f"\nSelected by validation macro-F1: {selected_config.name}")
    return selected_config, ranked_results


def train_selected_ensemble(
    selected_config: TrainingConfig,
    model_train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    tokenizer,
) -> np.ndarray:
    """Train the selected ensemble and return independent-test logits."""
    print("\nSelected configuration for final ensemble:")
    print(
        f"name={selected_config.name}, freeze={selected_config.freeze_layers}, "
        f"epochs={selected_config.epochs}, lr={selected_config.learning_rate}, "
        f"wd={selected_config.weight_decay}, "
        f"label_smoothing={selected_config.label_smoothing}, "
        f"seeds={list(ENSEMBLE_SEEDS)}"
    )

    train_dataset, validation_dataset, data_collator, class_weights = (
        _setup_training_prerequisites(model_train_df, validation_df, tokenizer)
    )

    seed_predictions = []
    for seed, model_dir in zip(ENSEMBLE_SEEDS, ENSEMBLE_MODEL_DIRS):
        seed_predictions.append(
            train_single_seed(
                model_dir=model_dir,
                seed=seed,
                training_config=selected_config,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                data_collator=data_collator,
                class_weights=class_weights,
                reference_df=model_train_df,
                evaluation_df=holdout_df,
                tokenizer=tokenizer,
                seed_label=f"final seed={seed}",
            )
        )

    print("\n--- Averaging logits from selected ensemble seeds ---")
    _, blended_holdout = average_seed_predictions(seed_predictions)
    return blended_holdout


def main() -> None:
    cleanup_temporary_split_files()

    # Phase 1: score samples and split data. No test metric is computed here.
    full_dataframe, model_train_df, validation_df, holdout_df = prepare_scored_splits()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Phase 2: validation-only hyperparameter selection.
    selected_config, search_results = select_hyperparameters(
        model_train_df,
        validation_df,
        tokenizer,
    )

    # Phase 3: train selected ensemble and evaluate the independent test once.
    blended_holdout_logits = train_selected_ensemble(
        selected_config,
        model_train_df,
        validation_df,
        holdout_df,
        tokenizer,
    )

    holdout_summary = evaluate_trainer_on_dataframe(
        None,
        holdout_df,
        tokenizer,
        split_name="independent_holdout_test",
        precomputed_logits=blended_holdout_logits,
        sample_path=str(SAMPLE_PATH),
        model_path=str(OUTPUT_MODEL),
    )

    print("\nDeployment metrics on independent_holdout_test:")
    print(
        f"accuracy={holdout_summary['finetuned_model']['accuracy']:.1%} | "
        f"macro_f1={holdout_summary['finetuned_model']['macro_f1']:.3f}"
    )

    save_training_metadata(
        full_dataframe,
        model_train_df,
        validation_df,
        holdout_df,
        selected_config,
        search_results,
        holdout_summary,
    )
    cleanup_temporary_split_files()

    print(f"\nFinal production model saved to: {OUTPUT_MODEL}")
    print(f"Ensemble models saved to: {[str(path) for path in ENSEMBLE_MODEL_DIRS]}")
    print(f"Training metadata saved to: {TRAINING_METADATA_PATH}")


if __name__ == "__main__":
    main()