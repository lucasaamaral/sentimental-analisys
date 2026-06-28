"""Training infrastructure for fine-tuning and validation-based model selection."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import (
    BertForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from classification_workflow.config.labels import ID2LABEL, LABEL2ID
from classification_workflow.config.paths import BASE_MODEL_NAME as MODEL_NAME


MAX_LEN = 256
BATCH_SIZE = 8
ENSEMBLE_SEEDS = (789, 123, 456)
SEARCH_SEED = 789

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


CANDIDATE_CONFIGS = (
    TrainingConfig(
        name="ce_lr5e6_wd0_s000",
        description="Cross-entropy, lr=5e-6, weight_decay=0.00, no label smoothing.",
        freeze_layers=0,
        epochs=4,
        learning_rate=5e-6,
        weight_decay=0.00,
        training_seed=SEARCH_SEED,
        label_smoothing=0.00,
    ),
    TrainingConfig(
        name="ce_lr1e5_wd0_s000",
        description="Cross-entropy, lr=1e-5, weight_decay=0.00, no label smoothing.",
        freeze_layers=0,
        epochs=4,
        learning_rate=1e-5,
        weight_decay=0.00,
        training_seed=SEARCH_SEED,
        label_smoothing=0.00,
    ),
    TrainingConfig(
        name="smooth_lr1e5_wd003_s010",
        description="Label smoothing=0.10, lr=1e-5, weight_decay=0.03.",
        freeze_layers=0,
        epochs=4,
        learning_rate=1e-5,
        weight_decay=0.03,
        training_seed=SEARCH_SEED,
        label_smoothing=0.10,
    ),
    TrainingConfig(
        name="smooth_lr2e5_wd003_s010",
        description="Label smoothing=0.10, lr=2e-5, weight_decay=0.03.",
        freeze_layers=0,
        epochs=4,
        learning_rate=2e-5,
        weight_decay=0.03,
        training_seed=SEARCH_SEED,
        label_smoothing=0.10,
    ),
    TrainingConfig(
        name="smooth_lr1e5_wd001_s005",
        description="Label smoothing=0.05, lr=1e-5, weight_decay=0.01.",
        freeze_layers=0,
        epochs=4,
        learning_rate=1e-5,
        weight_decay=0.01,
        training_seed=SEARCH_SEED,
        label_smoothing=0.05,
    ),
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
    """Trainer using weighted, optionally label-smoothed cross-entropy."""

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


def compute_metrics(eval_prediction) -> dict[str, float]:
    logits, labels = eval_prediction
    pred_ids = np.argmax(logits, axis=1)
    y_true = [ID2LABEL[int(label)] for label in labels]
    y_pred = [ID2LABEL[int(pred)] for pred in pred_ids]
    return classification_metrics(y_true, y_pred)


def classification_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(LABEL2ID.keys()),
                average="macro",
                zero_division=0,
            )
        ),
    }


def build_training_args(
    output_dir: Path,
    logging_dir: Path,
    seed: int,
    training_config: TrainingConfig,
) -> TrainingArguments:
    strategy_key = (
        "eval_strategy"
        if "eval_strategy" in signature(TrainingArguments.__init__).parameters
        else "evaluation_strategy"
    )
    args = {
        "output_dir": str(output_dir),
        "num_train_epochs": training_config.epochs,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "learning_rate": training_config.learning_rate,
        "weight_decay": training_config.weight_decay,
        "warmup_ratio": training_config.warmup_ratio,
        "max_grad_norm": training_config.max_grad_norm,
        "logging_dir": str(logging_dir),
        "logging_steps": 10,
        "seed": seed,
        "data_seed": seed,
        "fp16": torch.cuda.is_available(),
        "group_by_length": True,
        "dataloader_pin_memory": torch.cuda.is_available(),
        "report_to": "none",
        strategy_key: "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
    }
    return TrainingArguments(**args)


def create_model(training_config: TrainingConfig) -> BertForSequenceClassification:
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
        problem_type="single_label_classification",
    )
    freeze_bert_layers(model, training_config.freeze_layers)
    model.to(device)
    return model


class TrainingPredictions(NamedTuple):
    """Logits from a trained model on reference and evaluation splits."""

    reference: np.ndarray
    evaluation: np.ndarray


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


def _cleanup_model_resources(
    trainer: ChampionTrainer,
    model: BertForSequenceClassification,
) -> None:
    del trainer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _setup_training_prerequisites(
    model_train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    tokenizer,
) -> tuple[NewsDataset, NewsDataset, DataCollatorWithPadding, torch.Tensor]:
    train_dataset = NewsDataset(model_train_df, tokenizer, MAX_LEN)
    validation_dataset = NewsDataset(validation_df, tokenizer, MAX_LEN)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    train_label_ids = np.array([LABEL2ID[label] for label in model_train_df["human_label"]])
    raw_weights = compute_class_weights_for_ids(train_label_ids)
    class_weights = torch.tensor(raw_weights, dtype=torch.float)

    print(
        "Class weights: "
        + ", ".join(
            f"{ID2LABEL[index]}={raw_weights[index]:.3f}"
            for index in range(len(LABEL2ID))
        )
    )
    return train_dataset, validation_dataset, data_collator, class_weights


def train_single_seed(
    model_dir: Path,
    seed: int,
    training_config: TrainingConfig,
    train_dataset: NewsDataset,
    validation_dataset: NewsDataset,
    data_collator: DataCollatorWithPadding,
    class_weights: torch.Tensor,
    reference_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    tokenizer,
    seed_label: str,
) -> TrainingPredictions:
    model_dir.mkdir(parents=True, exist_ok=True)
    model = create_model(training_config)
    training_args = build_training_args(
        model_dir,
        model_dir / "training_logs",
        seed=seed,
        training_config=training_config,
    )

    trainer = ChampionTrainer(
        class_weights=class_weights,
        label_smoothing=training_config.label_smoothing,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"\n--- Training {seed_label} model ({training_config.name}) ---")
    trainer.train()

    reference_logits, _ = _get_trainer_predictions(trainer, reference_df, tokenizer)
    evaluation_logits, _ = _get_trainer_predictions(trainer, evaluation_df, tokenizer)

    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    _cleanup_model_resources(trainer, model)

    return TrainingPredictions(reference=reference_logits, evaluation=evaluation_logits)


def predictions_to_labels(logits: np.ndarray) -> list[str]:
    pred_ids = np.argmax(logits, axis=1)
    return [ID2LABEL[int(pred)] for pred in pred_ids]


def labels_from_dataframe(dataframe: pd.DataFrame) -> list[str]:
    return dataframe["human_label"].tolist()


def average_seed_predictions(
    predictions: list[TrainingPredictions],
) -> tuple[np.ndarray, np.ndarray]:
    if not predictions:
        raise ValueError("At least one seed prediction is required.")
    reference_logits = np.mean([item.reference for item in predictions], axis=0)
    evaluation_logits = np.mean([item.evaluation for item in predictions], axis=0)
    return reference_logits, evaluation_logits