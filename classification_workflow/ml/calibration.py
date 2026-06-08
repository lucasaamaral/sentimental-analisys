"""Calibration for the fine-tuned FinBERT model.

Exposes two entry points:
- ``learn_calibration_biases(logits, labels)``: applies the fixed bias override and
  returns a calibration summary dict compatible with the training pipeline.
- ``evaluate_predictions(true_ids, predicted_ids)``: computes accuracy, macro-F1,
  per-class report, and confusion matrix as a serialisable dict.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from classification_workflow.config.labels import ID2LABEL, LABEL2ID
from classification_workflow.ml.inference import apply_biases


# Fixed calibration biases found via grid search on the holdout set.
# P_recall=0.72, N_recall=0.69, U_recall=0.52 → macro_f1=0.643
CALIBRATION_BIAS: dict[str, float] = {"POSITIVE": -0.65, "NEGATIVE": -0.20, "NEUTRAL": 0.0}


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def evaluate_predictions(true_ids: np.ndarray, predicted_ids: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(true_ids, predicted_ids)),
        "macro_f1": float(f1_score(true_ids, predicted_ids, average="macro", zero_division=0)),
        "classification_report": classification_report(
            true_ids,
            predicted_ids,
            labels=list(ID2LABEL.keys()),
            target_names=list(LABEL2ID.keys()),
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(
            true_ids,
            predicted_ids,
            labels=list(ID2LABEL.keys()),
        ).tolist(),
    }


def values_by_label(values: np.ndarray) -> dict[str, float]:
    return {label: float(values[index]) for label, index in LABEL2ID.items()}


def label_value_dict_to_array(
    values: dict[str, float] | None,
    *,
    default: float,
) -> np.ndarray:
    if not isinstance(values, dict):
        return np.full(len(LABEL2ID), default, dtype=np.float32)
    return np.array(
        [float(values.get(label, default)) for label in LABEL2ID],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def learn_calibration_biases(logits: np.ndarray, labels: np.ndarray) -> dict:
    """Apply CALIBRATION_BIAS and return a calibration summary dict."""
    baseline_predictions = np.argmax(logits, axis=1)
    baseline_metrics = evaluate_predictions(labels, baseline_predictions)

    override_biases = np.array(
        [CALIBRATION_BIAS[label] for label in LABEL2ID], dtype=np.float32
    )
    override_biases = override_biases - np.max(override_biases)
    cal_biases_dict = values_by_label(override_biases)
    cal_predictions = np.argmax(apply_biases(logits, override_biases), axis=1)
    calibrated_metrics = evaluate_predictions(labels, cal_predictions)

    bias_text = ", ".join(f"{lbl}={cal_biases_dict[lbl]:+.2f}" for lbl in LABEL2ID)
    print(
        f"\nCalibration OVERRIDE: {CALIBRATION_BIAS}\n"
        f"- bias_override: macro_f1={calibrated_metrics['macro_f1']:.3f} | "
        f"accuracy={calibrated_metrics['accuracy']:.1%} | biases={{ {bias_text} }}"
    )

    candidate = {
        "method": "bias_override",
        "selected_scales": values_by_label(np.ones(len(LABEL2ID), dtype=np.float32)),
        "selected_biases": cal_biases_dict,
        "metrics": calibrated_metrics,
        "search_result": {"note": "hardcoded from CALIBRATION_BIAS"},
    }
    return {
        "selected_method": "bias_override",
        "selected_scales": candidate["selected_scales"],
        "selected_biases": candidate["selected_biases"],
        "positive_recall_floor": None,
        "baseline": baseline_metrics,
        "calibrated": calibrated_metrics,
        "search_result": candidate["search_result"],
        "selected_candidate": candidate,
        "candidates": {"bias_override": candidate},
    }
