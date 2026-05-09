"""Data splitting and preparation utilities for model training workflows.

Handles:
- Loading and validating labeled datasets
- Creating stratified train/test splits
- Creating calibration splits
- Temporary file management
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from sentiment_workflow.config.labels import LABEL2ID
from sentiment_workflow.config.paths import (
    CALIBRATION_SPLIT_PATH,
    TEST_SPLIT_PATH,
    TRAIN_MODEL_SPLIT_PATH,
    TRAIN_SPLIT_PATH,
)
from sentiment_workflow.data.csv_utils import write_normalized_csv


# =========================================================================
# Data Split Configuration
# =========================================================================

# Holdout (evaluation) split parameters
HOLDOUT_TEST_SIZE = 0.2
HOLDOUT_SEED = 2026

# Calibration split parameters (exclusive calibration set)
CALIBRATION_SIZE = 0.2
CALIBRATION_SEED = 2027


# =========================================================================
# Data Loading & Preparation
# =========================================================================


def load_labelled_dataframe(sample_path: Path) -> pd.DataFrame:
    """Load and validate labeled samples from CSV.
    
    Filters for:
    - Non-null human_label entries
    - Valid labels (POSITIVE, NEGATIVE, NEUTRAL)
    - Uppercase normalization
    
    Args:
        sample_path: Path to labeled_samples.csv
        
    Returns:
        Validated DataFrame with human_label column populated
        
    Raises:
        ValueError: If no valid labeled examples found
    """
    print(f"\nReading: {sample_path}")
    dataframe = pd.read_csv(sample_path, encoding="utf-8")

    valid_labels = set(LABEL2ID.keys())
    dataframe = dataframe[dataframe["human_label"].notna()]
    dataframe = dataframe[
        dataframe["human_label"].str.strip().str.upper().isin(valid_labels)
    ].copy()
    dataframe["human_label"] = dataframe["human_label"].str.strip().str.upper()

    if dataframe.empty:
        raise ValueError(
            "No labelled examples found. Fill the 'human_label' column in "
            "labeled_samples/labeled_samples.csv with POSITIVE, NEGATIVE or NEUTRAL before "
            "running fine-tuning."
        )

    dataframe = dataframe.reset_index(drop=True)
    label_counts = dataframe["human_label"].value_counts().reindex(
        list(LABEL2ID.keys()), fill_value=0
    )
    imbalance_ratio = float(label_counts.max() / label_counts.min())

    print(f"Total labelled examples: {len(dataframe)}")
    print(label_counts.to_string())
    print(f"Overall label imbalance ratio (max/min): {imbalance_ratio:.2f}")
    return dataframe


def _summarise_split(dataframe: pd.DataFrame, title: str) -> None:
    """Print summary statistics for a data split.
    
    Args:
        dataframe: DataFrame with human_label column
        title: Title for the summary output
    """
    label_counts = dataframe["human_label"].value_counts().reindex(
        list(LABEL2ID.keys()), fill_value=0
    )
    imbalance_ratio = float(label_counts.max() / label_counts.min())
    print(title)
    print(label_counts.to_string())
    print(f"imbalance_ratio={imbalance_ratio:.2f}")


def create_fixed_holdout_split(
    full_dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create fixed stratified holdout test split.
    
    Stratified split ensures label distribution is preserved across train/test.
    Uses fixed seed (HOLDOUT_SEED=2026) for reproducibility.
    
    Args:
        full_dataframe: Full labeled dataset
        
    Returns:
        (model_selection_df, holdout_df) tuple
        - model_selection_df: 80% for model selection/training
        - holdout_df: 20% for final evaluation
        
    Side Effects:
        Writes splits to TRAIN_SPLIT_PATH and TEST_SPLIT_PATH
    """
    model_selection_df, holdout_df = train_test_split(
        full_dataframe,
        test_size=HOLDOUT_TEST_SIZE,
        random_state=HOLDOUT_SEED,
        shuffle=True,
        stratify=full_dataframe["human_label"],
    )
    model_selection_df = model_selection_df.reset_index(drop=True)
    holdout_df = holdout_df.reset_index(drop=True)
    write_normalized_csv(model_selection_df, TRAIN_SPLIT_PATH, index=False, encoding="utf-8")
    write_normalized_csv(holdout_df, TEST_SPLIT_PATH, index=False, encoding="utf-8")

    print(
        f"\nFixed split: model_selection={len(model_selection_df)} rows | "
        f"holdout_test={len(holdout_df)} rows"
    )
    _summarise_split(model_selection_df, "Model-selection label distribution:")
    _summarise_split(holdout_df, "Holdout test label distribution:")
    print(f"Saved model-selection split to: {TRAIN_SPLIT_PATH}")
    print(f"Saved holdout test split to: {TEST_SPLIT_PATH}")
    return model_selection_df, holdout_df


def create_calibration_split(
    model_selection_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create exclusive calibration split from model-selection data.
    
    Splits model-selection data into:
    - model_train: For training ensemble
    - calibration: Exclusive split for learning calibration transform
    
    Uses fixed seed (CALIBRATION_SEED=2027) for reproducibility.
    Ensures calibration set is completely separate from training.
    
    Args:
        model_selection_df: Data from create_fixed_holdout_split()
        
    Returns:
        (model_train_df, calibration_df) tuple
        - model_train_df: 80% for ensemble training
        - calibration_df: 20% exclusive for calibration (not seen during training)
        
    Side Effects:
        Creates but does NOT write splits (handled by cleanup_temporary_split_files)
    """
    model_train_df, calibration_df = train_test_split(
        model_selection_df,
        test_size=CALIBRATION_SIZE,
        random_state=CALIBRATION_SEED,
        shuffle=True,
        stratify=model_selection_df["human_label"],
    )
    model_train_df = model_train_df.reset_index(drop=True)
    calibration_df = calibration_df.reset_index(drop=True)

    print(
        f"\nExclusive calibration split: model_train={len(model_train_df)} rows | "
        f"calibration={len(calibration_df)} rows"
    )
    _summarise_split(model_train_df, "Model-training label distribution:")
    _summarise_split(calibration_df, "Calibration label distribution:")
    return model_train_df, calibration_df


def cleanup_temporary_split_files() -> None:
    """Remove temporary split files created during training.
    
    Removes:
    - TRAIN_MODEL_SPLIT_PATH (temporary training split)
    - CALIBRATION_SPLIT_PATH (temporary calibration split)
    
    Note: TRAIN_SPLIT_PATH and TEST_SPLIT_PATH are kept for reference.
    """
    for path in (TRAIN_MODEL_SPLIT_PATH, CALIBRATION_SPLIT_PATH):
        if path.exists():
            path.unlink()
