"""Stage 2: fine-tune the selected FinBERT-PT-BR configuration.

Workflow:
    1. Read labeled_samples/labeled_samples.csv filtering rows where human_label is filled in
    2. Create a fixed stratified holdout test split
    3. Reserve a calibration-only split from the model-selection rows
    4. Train the winning configuration on the remaining model-training rows
    5. Learn the best calibration transform on the exclusive calibration split
    6. Save calibrated holdout metrics and persist the calibrated deployment parameters
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from sentiment_workflow.config.labels import LABEL2ID
from sentiment_workflow.config.paths import (
    BASE_MODEL_NAME as MODEL_NAME,
    FINETUNED_MODEL_DIR as OUTPUT_MODEL,
    LABELED_SAMPLES_PATH as SAMPLE_PATH,
    SECONDARY_MODEL_DIR,
    TEST_SPLIT_PATH,
    TRAIN_SPLIT_PATH,
    TRAINING_METADATA_PATH,
)
from sentiment_workflow.data.data_splits import (
    CALIBRATION_SEED,
    CALIBRATION_SIZE,
    HOLDOUT_SEED,
    HOLDOUT_TEST_SIZE,
    cleanup_temporary_split_files,
    create_calibration_split,
    create_fixed_holdout_split,
    load_labelled_dataframe,
)
from sentiment_workflow.ml.calibration import learn_calibration_biases
from sentiment_workflow.ml.evaluation import (
    evaluate_trainer_on_dataframe,
    print_baselines,
)
from sentiment_workflow.ml.inference import (
    attach_base_model_predictions,
    describe_device,
    load_sequence_classifier,
)
from sentiment_workflow.ml.trainer import (
    ENSEMBLE_BLEND_WEIGHT,
    ENSEMBLE_SECONDARY_SEED,
    WINNING_CONFIG,
    _get_trainer_predictions,
    _setup_training_prerequisites,
    blend_ensemble_predictions,
    train_single_seed,
)

warnings.filterwarnings("ignore")

WINNING_STRATEGY = "smooth_ce_s01_ensemble_789_123"


# =========================================================================
# Metadata Persistence
# =========================================================================


def save_training_metadata(
    full_dataframe: pd.DataFrame,
    model_selection_df: pd.DataFrame,
    model_train_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    baseline_summary: dict,
    calibration_summary: dict,
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
        "ensemble_primary_seed": WINNING_CONFIG.training_seed,
        "ensemble_secondary_seed": ENSEMBLE_SECONDARY_SEED,
        "secondary_model_dir": str(SECONDARY_MODEL_DIR),
        "ensemble_blend_weight": ENSEMBLE_BLEND_WEIGHT,
        "n_total_labelled_examples": int(len(full_dataframe)),
        "n_model_selection_examples": int(len(model_selection_df)),
        "n_model_training_examples": int(len(model_train_df)),
        "n_calibration_examples": int(len(calibration_df)),
        "n_holdout_examples": int(len(holdout_df)),
        "sample_path": str(SAMPLE_PATH),
        "train_split_path": str(TRAIN_SPLIT_PATH),
        "holdout_test_path": str(TEST_SPLIT_PATH),
        "holdout_test_size": HOLDOUT_TEST_SIZE,
        "holdout_seed": HOLDOUT_SEED,
        "calibration_size": CALIBRATION_SIZE,
        "calibration_seed": CALIBRATION_SEED,
        "temporary_split_files_removed": True,
        "baseline_full_sample": baseline_summary["full_sample"],
        "calibration": {
            "enabled": True,
            "selected_method": calibration_summary["selected_method"],
            "selected_scales": calibration_summary["selected_scales"],
            "selected_biases": calibration_summary["selected_biases"],
            "positive_recall_floor": calibration_summary.get("positive_recall_floor"),
            "baseline": calibration_summary["baseline"],
            "calibrated": calibration_summary["calibrated"],
            "search_result": calibration_summary["search_result"],
            "selected_candidate": calibration_summary["selected_candidate"],
            "candidates": calibration_summary["candidates"],
        },
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
    calibration_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    tokenizer,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Phase 2: train primary + secondary seeds and return blended calibration/holdout logits."""
    print("\nSelected winning configuration:")
    print(f"strategy={WINNING_STRATEGY}")
    print(
        f"freeze={WINNING_CONFIG.freeze_layers}, epochs={WINNING_CONFIG.epochs}, "
        f"lr={WINNING_CONFIG.learning_rate}, wd={WINNING_CONFIG.weight_decay}, "
        f"weights={WINNING_CONFIG.class_weight_mode}, "
        f"focal_gamma={WINNING_CONFIG.focal_gamma}, "
        f"seed={WINNING_CONFIG.training_seed}"
    )

    train_dataset, data_collator, class_weights = _setup_training_prerequisites(
        model_train_df, tokenizer
    )

    primary_preds = train_single_seed(
        model_dir=OUTPUT_MODEL,
        seed=WINNING_CONFIG.training_seed,
        train_dataset=train_dataset,
        data_collator=data_collator,
        class_weights=class_weights,
        calibration_df=calibration_df,
        holdout_df=holdout_df,
        tokenizer=tokenizer,
        seed_label=f"primary (seed={WINNING_CONFIG.training_seed})",
    )

    secondary_preds = train_single_seed(
        model_dir=SECONDARY_MODEL_DIR,
        seed=ENSEMBLE_SECONDARY_SEED,
        train_dataset=train_dataset,
        data_collator=data_collator,
        class_weights=class_weights,
        calibration_df=calibration_df,
        holdout_df=holdout_df,
        tokenizer=tokenizer,
        seed_label=f"secondary (seed={ENSEMBLE_SECONDARY_SEED})",
    )

    print("\n--- Blending primary + secondary logits ---")
    blended_cal, blended_holdout = blend_ensemble_predictions(
        primary_preds, secondary_preds
    )

    cal_labels = np.array(
        [LABEL2ID[label] for label in calibration_df["human_label"].tolist()]
    )

    return blended_cal, blended_holdout, cal_labels


def main() -> None:
    cleanup_temporary_split_files()

    # Phase 1: Load base model, score samples, create holdout split
    full_dataframe, holdout_df, baseline_summary = prepare_base_model_baselines()

    model_selection_df, _ = create_fixed_holdout_split(full_dataframe)
    model_train_df, calibration_df = create_calibration_split(model_selection_df)

    # Phase 2: Train ensemble
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    blended_cal_logits, blended_holdout_logits, cal_labels = train_ensemble(
        model_train_df, calibration_df, holdout_df, tokenizer
    )

    # Phase 3: Calibrate and evaluate
    calibration_summary = learn_calibration_biases(blended_cal_logits, cal_labels)
    holdout_summary = evaluate_trainer_on_dataframe(
        None,
        holdout_df,
        tokenizer,
        split_name="fixed_holdout_test",
        calibration_summary=calibration_summary,
        precomputed_logits=blended_holdout_logits,
        sample_path=str(SAMPLE_PATH),
        model_path=str(OUTPUT_MODEL),
    )

    print("\nCalibrated deployment metrics on fixed_holdout_test:")
    print(
        f"accuracy={holdout_summary['finetuned_model']['accuracy']:.1%} | "
        f"macro_f1={holdout_summary['finetuned_model']['macro_f1']:.3f}"
    )

    # Phase 4: Save metadata and cleanup
    save_training_metadata(
        full_dataframe,
        model_selection_df,
        model_train_df,
        calibration_df,
        holdout_df,
        baseline_summary,
        calibration_summary,
        holdout_summary,
    )
    cleanup_temporary_split_files()

    print(f"\nFinal production model saved to: {OUTPUT_MODEL}")
    print(f"Secondary model saved to: {SECONDARY_MODEL_DIR}")
    print(f"Training metadata saved to: {TRAINING_METADATA_PATH}")


if __name__ == "__main__":
    main()
