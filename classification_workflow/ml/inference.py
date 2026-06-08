from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, BertForSequenceClassification

from classification_workflow.config.labels import (
    BASE_MODEL_LABEL_MAP,
    LABELS,
    normalise_label,
)
from classification_workflow.config.paths import TRAINING_METADATA_PATH


@dataclass
class InferenceOutput:
    logits: np.ndarray
    probabilities: np.ndarray
    score_dicts: list[dict[str, float]]
    predicted_labels: list[str]


@dataclass(frozen=True)
class CalibrationParameters:
    scales: np.ndarray
    biases: np.ndarray
    method: str | None = None


def describe_device(device: torch.device) -> str:
    return "GPU" if device.type == "cuda" else "CPU"


def load_training_metadata(
    metadata_path: Path = TRAINING_METADATA_PATH,
    hf_repo: str | None = None,
) -> dict | None:
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as handle:
            return json.load(handle)

    if hf_repo is not None:
        from huggingface_hub import hf_hub_download

        local_path = hf_hub_download(repo_id=hf_repo, filename="training_strategy.json")
        with open(local_path, encoding="utf-8") as handle:
            return json.load(handle)

    return None


def extract_calibration_parameters(
    metadata: dict | None,
    labels: tuple[str, ...] = LABELS,
) -> CalibrationParameters | None:
    if metadata is None:
        return None

    calibration = metadata.get("calibration")
    if not isinstance(calibration, dict) or not calibration.get("enabled"):
        return None

    selected_biases = calibration.get("selected_biases")
    if not isinstance(selected_biases, dict):
        return None

    selected_scales = calibration.get("selected_scales")
    if isinstance(selected_scales, dict):
        scales = np.array(
            [float(selected_scales.get(label, 1.0)) for label in labels],
            dtype=np.float32,
        )
    else:
        scales = np.ones(len(labels), dtype=np.float32)

    biases = np.array(
        [float(selected_biases.get(label, 0.0)) for label in labels],
        dtype=np.float32,
    )

    return CalibrationParameters(
        scales=scales,
        biases=biases,
        method=calibration.get("selected_method"),
    )


def format_calibration_parameters(
    calibration: CalibrationParameters,
    labels: tuple[str, ...] = LABELS,
) -> str:
    scale_text = ", ".join(
        f"{label}={calibration.scales[index]:.2f}" for index, label in enumerate(labels)
    )
    bias_text = ", ".join(
        f"{label}={calibration.biases[index]:+.2f}"
        for index, label in enumerate(labels)
    )
    return (
        f"method={calibration.method or 'bias_only'} | "
        f"scales=[{scale_text}] | biases=[{bias_text}]"
    )


def load_sequence_classifier(
    model_name_or_path: str | Path,
) -> tuple[AutoTokenizer, BertForSequenceClassification, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = BertForSequenceClassification.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    return torch.softmax(torch.tensor(logits), dim=-1).cpu().numpy()


def apply_calibration(
    logits: np.ndarray,
    scales: np.ndarray | None = None,
    biases: np.ndarray | None = None,
) -> np.ndarray:
    calibrated_logits = logits
    if scales is not None:
        calibrated_logits = calibrated_logits * scales.reshape(1, -1)
    if biases is not None:
        calibrated_logits = calibrated_logits + biases.reshape(1, -1)
    return calibrated_logits


def apply_biases(logits: np.ndarray, biases: np.ndarray) -> np.ndarray:
    return apply_calibration(logits, biases=biases)


def resolve_model_labels(
    model: BertForSequenceClassification,
    label_aliases: dict[str, str] | None = None,
) -> list[str]:
    id2label = model.config.id2label or {}
    resolved_labels: list[str] = []

    for index in range(model.config.num_labels):
        raw_label = id2label.get(index, id2label.get(str(index), str(index)))
        label_key = normalise_label(raw_label)
        if label_aliases is not None:
            label_key = label_aliases.get(label_key, label_key)
        resolved_labels.append(label_key)

    return resolved_labels


def predict_scores(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: BertForSequenceClassification,
    device: torch.device,
    *,
    label_aliases: dict[str, str] | None = None,
    calibration_parameters: CalibrationParameters | None = None,
    batch_size: int = 32,
) -> InferenceOutput:
    model_labels = resolve_model_labels(model, label_aliases)
    if not texts:
        empty_logits = np.empty((0, len(model_labels)), dtype=np.float32)
        return InferenceOutput(
            logits=empty_logits,
            probabilities=empty_logits.copy(),
            score_dicts=[],
            predicted_labels=[],
        )

    all_logits: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            batch_logits = model(**encoded).logits.detach().cpu().numpy()
        all_logits.append(batch_logits)

    logits = np.concatenate(all_logits, axis=0)
    if calibration_parameters is not None:
        logits = apply_calibration(
            logits,
            scales=calibration_parameters.scales,
            biases=calibration_parameters.biases,
        )

    probabilities = softmax_numpy(logits)
    score_dicts: list[dict[str, float]] = []
    predicted_labels: list[str] = []

    for row in probabilities:
        scores = {
            model_labels[index]: round(float(row[index]), 6)
            for index in range(len(model_labels))
        }
        score_dicts.append(scores)
        predicted_labels.append(max(scores, key=scores.get))

    return InferenceOutput(
        logits=logits,
        probabilities=probabilities,
        score_dicts=score_dicts,
        predicted_labels=predicted_labels,
    )


def predict_scores_blended(
    texts: list[str],
    models: list[tuple],
    *,
    label_aliases: dict[str, str] | None = None,
    calibration_parameters: CalibrationParameters | None = None,
    batch_size: int = 32,
) -> InferenceOutput:
    """Ensemble inference: average raw logits from each model, then calibrate.

    ``models`` is a list of ``(tokenizer, model, device)`` tuples.
    """
    raw_logits_list: list[np.ndarray] = []
    for tokenizer_i, model_i, device_i in models:
        result = predict_scores(
            texts,
            tokenizer_i,
            model_i,
            device_i,
            label_aliases=label_aliases,
            calibration_parameters=None,
            batch_size=batch_size,
        )
        raw_logits_list.append(result.logits)

    blended = np.mean(raw_logits_list, axis=0)

    if calibration_parameters is not None:
        blended = apply_calibration(
            blended,
            scales=calibration_parameters.scales,
            biases=calibration_parameters.biases,
        )

    if not texts:
        return InferenceOutput(
            logits=blended,
            probabilities=blended.copy(),
            score_dicts=[],
            predicted_labels=[],
        )

    # Re-use label order from the first model
    first_tokenizer, first_model, _ = models[0]
    model_labels = resolve_model_labels(first_model, label_aliases)

    probabilities = softmax_numpy(blended)
    score_dicts: list[dict[str, float]] = []
    predicted_labels: list[str] = []
    for row in probabilities:
        scores = {
            model_labels[index]: round(float(row[index]), 6)
            for index in range(len(model_labels))
        }
        score_dicts.append(scores)
        predicted_labels.append(max(scores, key=scores.get))

    return InferenceOutput(
        logits=blended,
        probabilities=probabilities,
        score_dicts=score_dicts,
        predicted_labels=predicted_labels,
    )


def attach_base_model_predictions(
    dataframe: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: BertForSequenceClassification,
    device: torch.device,
    batch_size: int = 8,
) -> pd.DataFrame:
    """Run the base model on dataframe texts and attach predictions as a new column.

    Args:
        dataframe: Input DataFrame with a 'sentiment_text' column.
        tokenizer: Tokenizer for the base model.
        model: Base BertForSequenceClassification model.
        device: Device the model lives on.
        batch_size: Inference batch size (default 8).

    Returns:
        Copy of dataframe with 'base_predicted_label' column added.
    """
    augmented_df = dataframe.copy()
    texts = augmented_df["sentiment_text"].fillna("").astype(str).tolist()
    predictions = predict_scores(
        texts,
        tokenizer,
        model,
        device,
        label_aliases=BASE_MODEL_LABEL_MAP,
        batch_size=batch_size,
    )
    augmented_df["base_predicted_label"] = predictions.predicted_labels
    return augmented_df
