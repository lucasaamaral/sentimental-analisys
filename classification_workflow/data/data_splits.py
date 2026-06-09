"""Data splitting and preparation utilities for model training workflows.

Handles:
- Loading and validating labeled datasets
- Creating stratified train/test splits
- Temporary file management
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from classification_workflow.config.labels import LABEL2ID
from classification_workflow.config.paths import (
    TEST_SPLIT_PATH,
    TRAIN_MODEL_SPLIT_PATH,
    TRAIN_SPLIT_PATH,
)


# =========================================================================
# Data Split Configuration
# =========================================================================

# Holdout (evaluation) split parameters
HOLDOUT_TEST_SIZE = 0.2
HOLDOUT_SEED = 2026

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
    print(
        f"\nFixed split: model_selection={len(model_selection_df)} rows | "
        f"holdout_test={len(holdout_df)} rows"
    )
    _summarise_split(model_selection_df, "Model-selection label distribution:")
    _summarise_split(holdout_df, "Holdout test label distribution:")
    return model_selection_df, holdout_df


def cleanup_temporary_split_files() -> None:
    """Remove temporary split files created by earlier training workflows.

    Removes:
    - TRAIN_SPLIT_PATH
    - TEST_SPLIT_PATH
    - TRAIN_MODEL_SPLIT_PATH (temporary training split)
    """
    for path in (
        TRAIN_SPLIT_PATH,
        TEST_SPLIT_PATH,
        TRAIN_MODEL_SPLIT_PATH,
    ):
        if path.exists():
            path.unlink()
