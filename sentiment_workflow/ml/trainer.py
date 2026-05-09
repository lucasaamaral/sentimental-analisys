"""Training infrastructure: config, dataset, focal-loss trainer, model factory."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import (
    BertForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from sentiment_workflow.config.labels import ID2LABEL, LABEL2ID
from sentiment_workflow.config.paths import BASE_MODEL_NAME as MODEL_NAME


MAX_LEN = 256
BATCH_SIZE = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class TrainingConfig:
    name: str
    description: str
    freeze_layers: int
    epochs: int
    learning_rate: float
    weight_decay: float
    class_weight_mode: str
    focal_gamma: float
    training_seed: int
    loss_type: str = "focal"        # "focal" | "smooth_ce"
    label_smoothing: float = 0.0    # only used when loss_type == "smooth_ce"
    warmup_ratio: float = 0.1
    max_grad_norm: float = 0.5


WINNING_CONFIG = TrainingConfig(
    name="smooth_ce_s01_s789",
    description=(
        "smooth_ce (label_smoothing=0.10, primary_seed=789, secondary_seed=123): "
        "probe_multiseed winner on 515-row dataset (412 train / 103 holdout). "
        "2-seed ensemble macro_f1=0.605, P.f1=0.650, N.f1=0.712, U.f1=0.453. "
        "Improvement over focal_g10_s123: +0.017 macro_f1, +0.023 NEUTRAL F1."
    ),
    freeze_layers=0,
    epochs=4,
    learning_rate=7e-6,
    weight_decay=0.03,
    class_weight_mode="sqrt_balanced",
    focal_gamma=1.0,
    training_seed=789,
    loss_type="smooth_ce",
    label_smoothing=0.10,
)

# Second seed trained alongside primary for logit blending at inference time.
ENSEMBLE_SECONDARY_SEED = 123


class NewsDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_len: int):
        self.texts = dataframe["sentiment_text"].astype(str).tolist()
        self.labels = [LABEL2ID[label] for label in dataframe["human_label"].tolist()]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoded = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            truncation=True,
        )
        encoded["labels"] = self.labels[idx]
        return encoded


class ChampionTrainer(Trainer):
    """Unified trainer supporting focal loss and label-smoothed cross-entropy."""

    def __init__(
        self,
        class_weights: torch.Tensor,
        focal_gamma: float,
        loss_type: str = "focal",
        label_smoothing: float = 0.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self.class_weights.to(logits.device)

        if self.loss_type == "smooth_ce":
            loss = F.cross_entropy(
                logits, labels, weight=weights, label_smoothing=self.label_smoothing
            )
        else:
            log_prob = F.log_softmax(logits, dim=-1)
            nll = F.nll_loss(log_prob, labels, weight=weights, reduction="none")
            pt = torch.exp(-nll)
            loss = ((1 - pt) ** self.focal_gamma * nll).mean()

        return (loss, outputs) if return_outputs else loss


# Backward-compatible alias
FocalTrainer = ChampionTrainer


def freeze_bert_layers(model: BertForSequenceClassification, num_layers: int) -> None:
    if num_layers <= 0:
        print("Frozen: no transformer layers")
        return

    for parameter in model.bert.embeddings.parameters():
        parameter.requires_grad = False
    for layer in model.bert.encoder.layer[:num_layers]:
        for parameter in layer.parameters():
            parameter.requires_grad = False

    print(
        f"Frozen: embeddings + layers 0-{num_layers - 1} "
        f"({1 + num_layers}/{1 + len(model.bert.encoder.layer)} components)"
    )


def compute_class_weights_for_ids(label_ids: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.ones(len(LABEL2ID), dtype=np.float32)

    raw_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(LABEL2ID)),
        y=label_ids,
    )
    if mode == "sqrt_balanced":
        raw_weights = np.sqrt(raw_weights)
    elif mode != "balanced":
        raise ValueError(f"Unsupported class_weight_mode: {mode}")
    return raw_weights / raw_weights.min()


def build_training_args(output_dir: Path, logging_dir: Path, seed: int | None = None) -> TrainingArguments:
    effective_seed = seed if seed is not None else WINNING_CONFIG.training_seed
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=WINNING_CONFIG.epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=WINNING_CONFIG.learning_rate,
        weight_decay=WINNING_CONFIG.weight_decay,
        warmup_ratio=WINNING_CONFIG.warmup_ratio,
        max_grad_norm=WINNING_CONFIG.max_grad_norm,
        logging_dir=str(logging_dir),
        logging_steps=10,
        seed=effective_seed,
        data_seed=effective_seed,
        fp16=torch.cuda.is_available(),
        group_by_length=True,
        dataloader_pin_memory=torch.cuda.is_available(),
        report_to="none",
        eval_strategy="no",
        save_strategy="no",
    )


def create_model() -> BertForSequenceClassification:
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
        problem_type="single_label_classification",
    )
    freeze_bert_layers(model, WINNING_CONFIG.freeze_layers)
    model.to(device)
    return model


# =========================================================================
# Ensemble Training Helpers
# =========================================================================


class TrainingPredictions(NamedTuple):
    """Logits from a trained model on calibration and holdout splits."""

    calibration: np.ndarray
    holdout: np.ndarray


ENSEMBLE_BLEND_WEIGHT = 0.5  # Equal contribution from primary + secondary seeds


def _get_trainer_predictions(
    trainer: ChampionTrainer,
    dataframe: pd.DataFrame,
    tokenizer,
) -> tuple[np.ndarray, np.ndarray]:
    """Run trainer.predict and return (logits, label_ids)."""
    dataset = NewsDataset(dataframe, tokenizer, MAX_LEN)
    result = trainer.predict(dataset)
    logits = result.predictions
    labels = np.array([LABEL2ID[lbl] for lbl in dataframe["human_label"].tolist()])
    return logits, labels


def _cleanup_model_resources(trainer: ChampionTrainer, model: BertForSequenceClassification) -> None:
    """Delete trainer/model objects and empty the GPU cache."""
    del trainer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _setup_training_prerequisites(
    model_train_df: pd.DataFrame,
    tokenizer,
) -> tuple[NewsDataset, DataCollatorWithPadding, torch.Tensor]:
    """Build training dataset, data collator, and class-weight tensor."""
    train_dataset = NewsDataset(model_train_df, tokenizer, MAX_LEN)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    train_label_ids = np.array([LABEL2ID[label] for label in model_train_df["human_label"]])
    raw_weights = compute_class_weights_for_ids(train_label_ids, WINNING_CONFIG.class_weight_mode)
    class_weights = torch.tensor(raw_weights, dtype=torch.float)

    print(
        "Class weights: "
        + ", ".join(
            f"{ID2LABEL[i]}={raw_weights[i]:.3f}" for i in range(len(LABEL2ID))
        )
    )
    return train_dataset, data_collator, class_weights


def train_single_seed(
    model_dir: Path,
    seed: int,
    train_dataset: NewsDataset,
    data_collator: DataCollatorWithPadding,
    class_weights: torch.Tensor,
    calibration_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    tokenizer,
    seed_label: str,
) -> TrainingPredictions:
    """Train one model seed and return logits for calibration and holdout splits.

    Args:
        model_dir: Output directory for saved model.
        seed: Random seed for reproducibility.
        train_dataset: NewsDataset used for training.
        data_collator: Collator for dynamic padding.
        class_weights: Per-class loss weights tensor.
        calibration_df: DataFrame used for calibration predictions.
        holdout_df: DataFrame used for holdout predictions.
        tokenizer: HuggingFace tokenizer (saved alongside the model).
        seed_label: Human-readable label, e.g. "primary (seed=789)".

    Returns:
        TrainingPredictions with .calibration and .holdout logit arrays.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    model = create_model()
    training_args = build_training_args(model_dir, model_dir / "training_logs", seed=seed)

    trainer = ChampionTrainer(
        class_weights=class_weights,
        focal_gamma=WINNING_CONFIG.focal_gamma,
        loss_type=WINNING_CONFIG.loss_type,
        label_smoothing=WINNING_CONFIG.label_smoothing,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print(f"\n--- Training {seed_label} model ({WINNING_CONFIG.name}) ---")
    trainer.train()

    cal_logits, _ = _get_trainer_predictions(trainer, calibration_df, tokenizer)
    holdout_logits, _ = _get_trainer_predictions(trainer, holdout_df, tokenizer)

    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    _cleanup_model_resources(trainer, model)

    return TrainingPredictions(calibration=cal_logits, holdout=holdout_logits)


def blend_ensemble_predictions(
    primary_preds: TrainingPredictions,
    secondary_preds: TrainingPredictions,
    weight: float = ENSEMBLE_BLEND_WEIGHT,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend logits from primary and secondary seeds.

    Args:
        primary_preds: Predictions from primary seed.
        secondary_preds: Predictions from secondary seed.
        weight: Weight assigned to primary (default 0.5 → equal blend).

    Returns:
        (blended_cal_logits, blended_holdout_logits) tuple.
    """
    blended_cal = weight * primary_preds.calibration + (1 - weight) * secondary_preds.calibration
    blended_holdout = weight * primary_preds.holdout + (1 - weight) * secondary_preds.holdout
    return blended_cal, blended_holdout
