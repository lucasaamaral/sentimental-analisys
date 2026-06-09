from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"
LABELED_SAMPLES_DIR = BASE_DIR / "labeled_samples"

DATASET_REPO = "lucasalmda/pt-br-financial-news-sentiment"
BASE_MODEL_NAME = "lucas-leme/FinBERT-PT-BR"
FINETUNED_MODEL_HF_REPO = "lucasalmda/pt-br-financial-sentimental-analysis"
FINETUNED_MODEL_DIR = MODELS_DIR / "finbert-pt-br-finetuned"
FINETUNED_MODEL_DIR_SEED_123 = MODELS_DIR / "finbert-pt-br-finetuned-seed-123"
FINETUNED_MODEL_DIR_SEED_456 = MODELS_DIR / "finbert-pt-br-finetuned-seed-456"
ENSEMBLE_MODEL_DIRS = (
    FINETUNED_MODEL_DIR,
    FINETUNED_MODEL_DIR_SEED_123,
    FINETUNED_MODEL_DIR_SEED_456,
)
TRAINING_METADATA_PATH = FINETUNED_MODEL_DIR / "training_strategy.json"

BASE_CLASSIFIED_NEWS_PATH = OUTPUT_DIR / "base_model_classified.jsonl"
FINETUNED_CLASSIFIED_NEWS_PATH = OUTPUT_DIR / "finetuned_model_classified.jsonl"
LABELED_SAMPLES_PATH = LABELED_SAMPLES_DIR / "labeled_samples.csv"

TRAIN_SPLIT_PATH = OUTPUT_DIR / "samples_train.csv"
TRAIN_MODEL_SPLIT_PATH = (
    OUTPUT_DIR / "samples_train_model.csv"
)  # Temporary, created and deleted during training
TEST_SPLIT_PATH = OUTPUT_DIR / "samples_test.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LABELED_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
