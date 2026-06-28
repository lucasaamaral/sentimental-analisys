"""Evaluation utilities for base-model baselines and fine-tuned model metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from classification_workflow.config.labels import ID2LABEL, LABEL2ID
from classification_workflow.data.csv_utils import write_normalized_csv
from classification_workflow.ml.inference import softmax_numpy
from classification_workflow.ml.trainer import (
    _get_trainer_predictions,
)


def print_baselines(full_dataframe: pd.DataFrame, holdout_df: pd.DataFrame) -> dict:
    """Print base-model accuracy and macro-F1 on the full sample and holdout set.

    Args:
        full_dataframe: Complete labeled dataset with base_predicted_label column.
        holdout_df: Fixed holdout split with base_predicted_label column.

    Returns:
        Nested dict with 'full_sample' and 'holdout' metric dicts.
    """
    full_accuracy = accuracy_score(
        full_dataframe["human_label"],
        full_dataframe["base_predicted_label"],
    )
    full_macro_f1 = f1_score(
        full_dataframe["human_label"],
        full_dataframe["base_predicted_label"],
        labels=list(LABEL2ID.keys()),
        average="macro",
        zero_division=0,
    )
    holdout_accuracy = accuracy_score(
        holdout_df["human_label"],
        holdout_df["base_predicted_label"],
    )
    holdout_macro_f1 = f1_score(
        holdout_df["human_label"],
        holdout_df["base_predicted_label"],
        labels=list(LABEL2ID.keys()),
        average="macro",
        zero_division=0,
    )

    print(
        f"\nBASE model full labelled-sample baseline: accuracy={full_accuracy:.1%} | "
        f"macro_f1={full_macro_f1:.3f}"
    )
    print(
        f"BASE model fixed-holdout baseline: accuracy={holdout_accuracy:.1%} | "
        f"macro_f1={holdout_macro_f1:.3f}"
    )
    print(
        classification_report(
            holdout_df["human_label"],
            holdout_df["base_predicted_label"],
            target_names=list(LABEL2ID.keys()),
            zero_division=0,
        )
    )

    return {
        "full_sample": {
            "accuracy": float(full_accuracy),
            "macro_f1": float(full_macro_f1),
        },
        "holdout": {
            "accuracy": float(holdout_accuracy),
            "macro_f1": float(holdout_macro_f1),
        },
    }


def evaluate_trainer_on_dataframe(
    trainer,
    dataframe: pd.DataFrame,
    tokenizer,
    *,
    split_name: str,
    output_csv_path: Path | None = None,
    output_json_path: Path | None = None,
    precomputed_logits: np.ndarray | None = None,
    sample_path: str | None = None,
    model_path: str | None = None,
) -> dict:
    """Evaluate a fine-tuned model against base-model predictions.

    Args:
        trainer: HuggingFace Trainer instance, or None when precomputed_logits is provided.
        dataframe: DataFrame to evaluate; must contain human_label and base_predicted_label.
        tokenizer: HuggingFace tokenizer used to build the prediction dataset.
        split_name: Label for display and JSON output (e.g. 'fixed_holdout_test').
        output_csv_path: Where to write the annotated evaluation CSV, or None to skip.
        output_json_path: Where to write the JSON summary, or None to skip.
        precomputed_logits: Pre-run logit array; skips running trainer.predict when set.
        sample_path: Metadata string for JSON summary.
        model_path: Metadata string for JSON summary.

    Returns:
        Summary dict with base_model, finetuned_model, and delta keys.
    """
    if "base_predicted_label" not in dataframe.columns:
        raise ValueError("dataframe must have a base_predicted_label column.")

    if precomputed_logits is not None:
        raw_logits = precomputed_logits
    else:
        raw_logits, _ = _get_trainer_predictions(trainer, dataframe, tokenizer)

    probabilities = softmax_numpy(raw_logits)
    pred_ids = np.argmax(raw_logits, axis=1)

    y_true = dataframe["human_label"].tolist()
    y_base = dataframe["base_predicted_label"].tolist()
    y_final = [ID2LABEL[int(p)] for p in pred_ids]

    base_accuracy = accuracy_score(y_true, y_base)
    base_macro_f1 = f1_score(
        y_true, y_base, labels=list(LABEL2ID.keys()), average="macro", zero_division=0
    )
    final_accuracy = accuracy_score(y_true, y_final)
    final_macro_f1 = f1_score(
        y_true, y_final, labels=list(LABEL2ID.keys()), average="macro", zero_division=0
    )

    print(f"\nBase model metrics on {split_name}:")
    print(f"accuracy={base_accuracy:.1%} | macro_f1={base_macro_f1:.3f}")
    print(
        f"\nFine-tuned model metrics on {split_name} "
        "(decision=identity_argmax):"
    )
    print(
        f"accuracy={final_accuracy:.1%} | macro_f1={final_macro_f1:.3f} | "
        f"delta_acc={final_accuracy - base_accuracy:+.1%} | "
        f"delta_macro_f1={final_macro_f1 - base_macro_f1:+.3f}"
    )
    print(f"\nFinal classification report ({split_name}):")
    print(
        classification_report(
            y_true,
            y_final,
            labels=list(LABEL2ID.keys()),
            target_names=list(LABEL2ID.keys()),
            zero_division=0,
        )
    )

    confusion = confusion_matrix(y_true, y_final, labels=list(LABEL2ID.keys()))
    print(f"Final confusion matrix on {split_name} (rows=actual, columns=predicted):")
    print(
        pd.DataFrame(
            confusion,
            index=list(LABEL2ID.keys()),
            columns=list(LABEL2ID.keys()),
        ).to_string()
    )

    evaluation_df = dataframe.copy()
    evaluation_df["finetuned_predicted_label"] = y_final
    evaluation_df["ft_score_positive"] = np.round(
        probabilities[:, LABEL2ID["POSITIVE"]], 6
    )
    evaluation_df["ft_score_negative"] = np.round(
        probabilities[:, LABEL2ID["NEGATIVE"]], 6
    )
    evaluation_df["ft_score_neutral"] = np.round(
        probabilities[:, LABEL2ID["NEUTRAL"]], 6
    )
    evaluation_df["base_correct"] = (
        evaluation_df["base_predicted_label"] == evaluation_df["human_label"]
    )
    evaluation_df["finetuned_correct"] = (
        evaluation_df["finetuned_predicted_label"] == evaluation_df["human_label"]
    )

    if output_csv_path is not None:
        write_normalized_csv(evaluation_df, output_csv_path, index=False, encoding="utf-8")

    summary = {
        "evaluation_split": split_name,
        "sample_path": sample_path or "",
        "evaluated_rows": int(len(dataframe)),
        "model_path": model_path or "",
        "base_model": {
            "accuracy": float(base_accuracy),
            "macro_f1": float(base_macro_f1),
            "classification_report": classification_report(
                y_true,
                y_base,
                labels=list(LABEL2ID.keys()),
                target_names=list(LABEL2ID.keys()),
                zero_division=0,
                output_dict=True,
            ),
        },
        "finetuned_model": {
            "accuracy": float(final_accuracy),
            "macro_f1": float(final_macro_f1),
            "classification_report": classification_report(
                y_true,
                y_final,
                labels=list(LABEL2ID.keys()),
                target_names=list(LABEL2ID.keys()),
                zero_division=0,
                output_dict=True,
            ),
            "confusion_matrix": confusion.tolist(),
        },
        "decision_method": "identity_argmax",
        "delta": {
            "accuracy": float(final_accuracy - base_accuracy),
            "macro_f1": float(final_macro_f1 - base_macro_f1),
        },
    }

    if output_csv_path is not None and output_json_path is not None:
        summary["output_csv_path"] = str(output_csv_path)
        with open(output_json_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        print(f"\nDetailed {split_name} evaluation saved to: {output_csv_path}")
        print(f"Summary saved to: {output_json_path}")
    else:
        print(f"\nDetailed {split_name} evaluation kept in memory only.")

    return summary
