from datasets import Features, Sequence, Value, load_dataset

from sentiment_workflow.config.paths import DATASET_REPO


DATASET_FEATURES = Features(
    {
        "source": Value("string"),
        "url": Value("string"),
        "section": Value("string"),
        "authors": Sequence(Value("string")),
        "tags": Sequence(Value("string")),
        "title": Value("string"),
        "description": Value("string"),
        "lead": Value("string"),
        "sentiment_text": Value("string"),
        "published_at": Value("string"),
        "week_key": Value("string"),
        "week_start": Value("string"),
        "week_end": Value("string"),
        "weekday": Value("int64"),
    }
)


def load_financial_news_records() -> list[dict]:
    dataset = load_dataset(DATASET_REPO, split="train", features=DATASET_FEATURES)
    return dataset.to_list()
