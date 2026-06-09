"""Stage 2: fine-tune the selected FinBERT-PT-BR configuration.

Workflow:
    1. Read labeled_samples/labeled_samples.csv filtering rows where human_label is filled in
    2. Create a fixed stratified holdout test split
    3. Train the winning configuration on all model-selection rows
    4. Persist the final identity prediction-decision parameters
    5. Save holdout metrics and deployment metadata
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
    TRAINING_METADATA_PATH,
)
from classification_workflow.data.data_splits import (
    HOLDOUT_SEED,
    HOLDOUT_TEST_SIZE,
    cleanup_temporary_split_files,
    create_fixed_holdout_split,
    load_labelled_dataframe,
)
from classification_workflow.ml.evaluation import (
    evaluate_trainer_on_dataframe,
    print_baselines,
)
from classification_workflow.ml.inference import (
    attach_base_model_predictions,
    describe_device,
    load_sequence_classifier,
)
from classification_workflow.ml.trainer import (
    ENSEMBLE_SEEDS,
    WINNING_CONFIG,
    _setup_training_prerequisites,
    average_seed_predictions,
    train_single_seed,
)

warnings.filterwarnings("ignore")

WINNING_STRATEGY = "smooth_ce_s010_lr1e5_3seed"


# =========================================================================
# Metadata Persistence
# =========================================================================


def save_training_metadata(
    full_dataframe: pd.DataFrame,
    model_selection_df: pd.DataFrame,
    model_train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    baseline_summary: dict,
    holdout_summary: dict,
) -> None:
    training_metadata = {
        "selected_strategy": {
            **asdict(WINNING_CONFIG),
            "winning_strategy": WINNING_STRATEGY,
        },
        "final_training_epochs": WINNING_CONFIG.epochs,
        "selected_training_seed": WINNING_CONFIG.training_seed,
        "ensemble_mode": True,
        "ensemble_seeds": list(ENSEMBLE_SEEDS),
        "ensemble_model_dirs": [str(path) for path in ENSEMBLE_MODEL_DIRS],
        "decision_method": "identity_argmax",
        "n_total_labelled_examples": int(len(full_dataframe)),
        "n_model_selection_examples": int(len(model_selection_df)),
        "n_model_training_examples": int(len(model_train_df)),
        "n_holdout_examples": int(len(holdout_df)),
        "sample_path": str(SAMPLE_PATH),
        "holdout_test_size": HOLDOUT_TEST_SIZE,
        "holdout_seed": HOLDOUT_SEED,
        "temporary_split_files_removed": True,
        "baseline_full_sample": baseline_summary["full_sample"],
        "holdout_metrics": {
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


def prepare_base_model_baselines() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Phase 1: load base model, score all samples, create holdout split, print baselines."""
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

    model_selection_df, holdout_df = create_fixed_holdout_split(full_dataframe)
    holdout_df = attach_base_model_predictions(
        holdout_df,
        base_tokenizer,
        base_model,
        base_device,
    )

    del base_tokenizer, base_model
    if base_device.type == "cuda":
        torch.cuda.empty_cache()

    baseline_summary = print_baselines(full_dataframe, holdout_df)
    return full_dataframe, holdout_df, baseline_summary


def train_ensemble(
    model_train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    tokenizer,
) -> np.ndarray:
    """Phase 2: train the selected ensemble and return holdout logits."""
    print("\nSelected winning configuration:")
    print(f"strategy={WINNING_STRATEGY}")
    print(
        f"freeze={WINNING_CONFIG.freeze_layers}, epochs={WINNING_CONFIG.epochs}, "
        f"lr={WINNING_CONFIG.learning_rate}, wd={WINNING_CONFIG.weight_decay}, "
        f"label_smoothing={WINNING_CONFIG.label_smoothing}, "
        f"seeds={list(ENSEMBLE_SEEDS)}"
    )

    train_dataset, data_collator, class_weights = _setup_training_prerequisites(
        model_train_df, tokenizer
    )

    seed_predictions = []
    for seed, model_dir in zip(ENSEMBLE_SEEDS, ENSEMBLE_MODEL_DIRS):
        seed_predictions.append(
            train_single_seed(
                model_dir=model_dir,
                seed=seed,
                train_dataset=train_dataset,
                data_collator=data_collator,
                class_weights=class_weights,
                reference_df=model_train_df,
                holdout_df=holdout_df,
                tokenizer=tokenizer,
                seed_label=f"seed={seed}",
            )
        )

    print("\n--- Averaging logits from ensemble seeds ---")
    _, blended_holdout = average_seed_predictions(seed_predictions)

    return blended_holdout


def main() -> None:
    cleanup_temporary_split_files()

    # Phase 1: Load base model, score samples, create holdout split
    full_dataframe, holdout_df, baseline_summary = prepare_base_model_baselines()

    model_selection_df, _ = create_fixed_holdout_split(full_dataframe)
    model_train_df = model_selection_df

    # Phase 2: Train ensemble
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    blended_holdout_logits = train_ensemble(
        model_train_df, holdout_df, tokenizer
    )

    # Phase 3: Evaluate direct argmax on averaged logits
    holdout_summary = evaluate_trainer_on_dataframe(
        None,
        holdout_df,
        tokenizer,
        split_name="fixed_holdout_test",
        precomputed_logits=blended_holdout_logits,
        sample_path=str(SAMPLE_PATH),
        model_path=str(OUTPUT_MODEL),
    )

    print("\nDeployment metrics on fixed_holdout_test:")
    print(
        f"accuracy={holdout_summary['finetuned_model']['accuracy']:.1%} | "
        f"macro_f1={holdout_summary['finetuned_model']['macro_f1']:.3f}"
    )

    # Phase 4: Save metadata and cleanup
    save_training_metadata(
        full_dataframe,
        model_selection_df,
        model_train_df,
        holdout_df,
        baseline_summary,
        holdout_summary,
    )
    cleanup_temporary_split_files()

    print(f"\nFinal production model saved to: {OUTPUT_MODEL}")
    print(f"Ensemble models saved to: {[str(path) for path in ENSEMBLE_MODEL_DIRS]}")
    print(f"Training metadata saved to: {TRAINING_METADATA_PATH}")


if __name__ == "__main__":
    main()
