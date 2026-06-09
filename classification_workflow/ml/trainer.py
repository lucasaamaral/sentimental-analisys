"""Training infrastructure for the selected fine-tuned ensemble."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import (
    BertForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from classification_workflow.config.labels import ID2LABEL, LABEL2ID
from classification_workflow.config.paths import BASE_MODEL_NAME as MODEL_NAME


MAX_LEN = 256
BATCH_SIZE = 8
ENSEMBLE_SEEDS = (789, 123, 456)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class TrainingConfig:
    name: str
    description: str
    freeze_layers: int
    epochs: int
    learning_rate: float
    weight_decay: float
    training_seed: int
    label_smoothing: float
    warmup_ratio: float = 0.1
    max_grad_norm: float = 0.5


WINNING_CONFIG = TrainingConfig(
    name="smooth_ce_s010_lr1e5_3seed",
    description=(
        "Selected strategy after the 624-row review: label-smoothed cross-entropy "
        "(label_smoothing=0.10), lr=1e-5, weight_decay=0.03, 4 epochs, "
        "sqrt-balanced class weights, and a 3-seed logits ensemble "
        "(789, 123, 456). Fixed-holdout macro_f1=0.732."
    ),
    freeze_layers=0,
    epochs=4,
    learning_rate=1e-5,
    weight_decay=0.03,
    training_seed=789,
    label_smoothing=0.10,
)


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
    """Trainer using the selected label-smoothed cross-entropy loss."""

    def __init__(
        self,
        class_weights: torch.Tensor,
        label_smoothing: float,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self.class_weights.to(logits.device)
        loss = F.cross_entropy(
            logits,
            labels,
            weight=weights,
            label_smoothing=self.label_smoothing,
        )
        return (loss, outputs) if return_outputs else loss


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


def compute_class_weights_for_ids(label_ids: np.ndarray) -> np.ndarray:
    raw_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(LABEL2ID)),
        y=label_ids,
    )
    raw_weights = np.sqrt(raw_weights)
    return raw_weights / raw_weights.min()


def build_training_args(output_dir: Path, logging_dir: Path, seed: int) -> TrainingArguments:
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
        seed=seed,
        data_seed=seed,
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


class TrainingPredictions(NamedTuple):
    """Logits from a trained model on reference and holdout splits."""

    reference: np.ndarray
    holdout: np.ndarray


def _get_trainer_predictions(
    trainer: ChampionTrainer,
    dataframe: pd.DataFrame,
    tokenizer,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = NewsDataset(dataframe, tokenizer, MAX_LEN)
    result = trainer.predict(dataset)
    logits = result.predictions
    labels = np.array([LABEL2ID[label] for label in dataframe["human_label"].tolist()])
    return logits, labels


def _cleanup_model_resources(trainer: ChampionTrainer, model: BertForSequenceClassification) -> None:
    del trainer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _setup_training_prerequisites(
    model_train_df: pd.DataFrame,
    tokenizer,
) -> tuple[NewsDataset, DataCollatorWithPadding, torch.Tensor]:
    train_dataset = NewsDataset(model_train_df, tokenizer, MAX_LEN)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    train_label_ids = np.array([LABEL2ID[label] for label in model_train_df["human_label"]])
    raw_weights = compute_class_weights_for_ids(train_label_ids)
    class_weights = torch.tensor(raw_weights, dtype=torch.float)

    print(
        "Class weights: "
        + ", ".join(
            f"{ID2LABEL[index]}={raw_weights[index]:.3f}" for index in range(len(LABEL2ID))
        )
    )
    return train_dataset, data_collator, class_weights


def train_single_seed(
    model_dir: Path,
    seed: int,
    train_dataset: NewsDataset,
    data_collator: DataCollatorWithPadding,
    class_weights: torch.Tensor,
    reference_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    tokenizer,
    seed_label: str,
) -> TrainingPredictions:
    model_dir.mkdir(parents=True, exist_ok=True)
    model = create_model()
    training_args = build_training_args(model_dir, model_dir / "training_logs", seed=seed)

    trainer = ChampionTrainer(
        class_weights=class_weights,
        label_smoothing=WINNING_CONFIG.label_smoothing,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print(f"\n--- Training {seed_label} model ({WINNING_CONFIG.name}) ---")
    trainer.train()

    reference_logits, _ = _get_trainer_predictions(trainer, reference_df, tokenizer)
    holdout_logits, _ = _get_trainer_predictions(trainer, holdout_df, tokenizer)

    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    _cleanup_model_resources(trainer, model)

    return TrainingPredictions(reference=reference_logits, holdout=holdout_logits)


def average_seed_predictions(predictions: list[TrainingPredictions]) -> tuple[np.ndarray, np.ndarray]:
    if not predictions:
        raise ValueError("At least one seed prediction is required.")
    reference_logits = np.mean([item.reference for item in predictions], axis=0)
    holdout_logits = np.mean([item.holdout for item in predictions], axis=0)
    return reference_logits, holdout_logits
