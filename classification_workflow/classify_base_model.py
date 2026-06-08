"""Stage 1: classify the full dataset with the base FinBERT model."""

from __future__ import annotations

from tqdm import tqdm

from classification_workflow.config.labels import BASE_MODEL_LABEL_MAP
from classification_workflow.config.paths import BASE_CLASSIFIED_NEWS_PATH, BASE_MODEL_NAME, DATASET_REPO
from classification_workflow.data.dataset_source import load_financial_news_records
from classification_workflow.data.records import (
    finalize_records_file,
    load_seen_values,
    open_checkpoint_handle,
    write_jsonl_record,
)
from classification_workflow.ml.inference import describe_device, load_sequence_classifier, predict_scores


BATCH_SIZE = 32


def main() -> None:
    print(f"Loading model: {BASE_MODEL_NAME}")
    tokenizer, model, device = load_sequence_classifier(BASE_MODEL_NAME)
    print(f"Device: {describe_device(device)}")

    print(f"\nReading dataset from Hugging Face: {DATASET_REPO}")
    records = load_financial_news_records()
    print(f"Total records: {len(records):,}")

    already_done = load_seen_values(BASE_CLASSIFIED_NEWS_PATH, "url")
    if already_done:
        print(f"Checkpoint: {len(already_done):,} records already classified, resuming...")

    remaining = [record for record in records if record.get("url") not in already_done]
    print(f"Records to classify: {len(remaining):,}")

    total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
    with open_checkpoint_handle(BASE_CLASSIFIED_NEWS_PATH) as handle:
        for start in tqdm(
            range(0, len(remaining), BATCH_SIZE),
            total=total_batches,
            desc="Classifying",
        ):
            batch_records = remaining[start : start + BATCH_SIZE]
            batch_texts = [str(record.get("sentiment_text") or "") for record in batch_records]
            predictions = predict_scores(
                batch_texts,
                tokenizer,
                model,
                device,
                label_aliases=BASE_MODEL_LABEL_MAP,
                batch_size=BATCH_SIZE,
            )

            for record, predicted_label, scores in zip(
                batch_records,
                predictions.predicted_labels,
                predictions.score_dicts,
            ):
                write_jsonl_record(
                    handle,
                    {
                        "source": record.get("source"),
                        "url": record.get("url"),
                        "title": record.get("title"),
                        "description": record.get("description"),
                        "published_at": record.get("published_at"),
                        "week_key": record.get("week_key"),
                        "sentiment_text": record.get("sentiment_text"),
                        "predicted_label": predicted_label,
                        "score_positive": scores.get("POSITIVE", 0.0),
                        "score_negative": scores.get("NEGATIVE", 0.0),
                        "score_neutral": scores.get("NEUTRAL", 0.0),
                    },
                )

    finalize_records_file(BASE_CLASSIFIED_NEWS_PATH)
    print(f"\nClassification complete. Results saved to: {BASE_CLASSIFIED_NEWS_PATH}")


if __name__ == "__main__":
    main()
