# PT-BR Financial Sentiment Analysis — Fine-Tuning Workflow

A comprehensive end-to-end pipeline for sentiment classification of Brazilian Portuguese financial news using FinBERT, featuring multi-seed ensemble fine-tuning and post-hoc probability calibration.

## 📊 Pre-Trained Models

The fine-tuned models are published on Hugging Face:

- **[lucasalmda/pt-br-financial-sentimental-analysis](https://huggingface.co/lucasalmda/pt-br-financial-sentimental-analysis)** — Production model (calibrated, 64.3% holdout accuracy)
- **lucasalmda/pt-br-financial-sentimental-analysis-secondary** — Secondary ensemble model (private)

Both models classify financial news as **POSITIVE**, **NEGATIVE**, or **NEUTRAL**, trained on 629 hand-annotated Brazilian financial news headlines.

## Architecture

The workflow executes **3 sequential stages** to build and deploy a calibrated sentiment classifier:

### Stage 1: Base Classification
Generates baseline predictions on the full dataset using the pre-trained [lucas-leme/FinBERT-PT-BR](https://huggingface.co/lucas-leme/FinBERT-PT-BR) model.

### Stage 2: Fine-Tuning & Calibration
- Trains a 2-seed ensemble (smooth cross-entropy loss) on 402 annotated examples
- Reserves 101 examples for exclusive calibration split
- Learns additive per-class logit biases (calibration) to optimize Macro-F1
- Evaluates on fixed 126-example holdout (never seen during training/calibration)

### Stage 3: Final Classification
Re-classifies the complete dataset with the calibrated production model, optionally blending with the secondary ensemble model.

## Project Structure

```
sentimental-analisys/
├── main.py                              # Entry point — dispatch to workflow tasks
├── README.md                            # Documentation
├── .gitignore                           # Git exclusions (excludes /models, /output)
├── requirements.txt                     # Python dependencies
├── labeled_samples/                     # Hand-annotated training data
│   ├── labeled_samples.csv              # Master training set (629 rows)
│   ├── labeled_samples_base_eval.csv    # Base model evaluation
│   └── labeled_samples_finetuned_eval.csv
├── sentiment_workflow/                  # Main package
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── labels.py                   # Label mapping (POSITIVE, NEGATIVE, NEUTRAL)
│   │   └── paths.py                    # File and model paths
│   ├── data/
│   │   ├── __init__.py
│   │   ├── csv_utils.py                # CSV I/O utilities
│   │   ├── data_splits.py              # Holdout & calibration split creation
│   │   ├── dataset_source.py           # HuggingFace dataset loader
│   │   └── records.py                  # JSONL serialization helpers
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── trainer.py                  # Training config and ensemble blending
│   │   ├── inference.py                # Model loading and batch inference
│   │   ├── calibration.py              # Per-class bias optimization
│   │   └── evaluation.py               # Metrics and classification reports
│   ├── classify_base_model.py          # Stage 1: Base model classification
│   ├── train_finetuned_model.py        # Stage 2: Fine-tuning & calibration
│   └── classify_finetuned_model.py     # Stage 3: Final calibrated classification
└── output/                              # Generated artifacts (gitignored)
    ├── base_model_classified.jsonl      # Stage 1 results
    ├── finetuned_model_classified.jsonl # Stage 3 results
    ├── samples_train.csv                # Model selection split (80%)
    └── samples_test.csv                 # Holdout split (20%, fixed seed=2026)
```

**Note:** `.gitignore` excludes `/models`, `/output`, `.venv/`, and `__pycache__/` (generated during pipeline execution).

## Installation

```pwsh
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

### Run Complete Pipeline

```pwsh
python main.py
```

Executes sequentially: **classify-base-model** → **train-finetuned-model** → **classify-finetuned-model**

### Run Individual Stages

```pwsh
# Stage 1: Classify dataset with base model
python main.py classify-base-model

# Stage 2: Fine-tune and calibrate
python main.py train-finetuned-model

# Stage 3: Re-classify with calibrated model
python main.py classify-finetuned-model
```

## Output Format (JSONL)

Each line is a JSON object with predictions:

```json
{
  "source": "InfoMoney",
  "url": "https://www.infomoney.com.br/...",
  "title": "Ibovespa fecha em alta...",
  "description": "Índice sobe com expectativa...",
  "published_at": "2025-04-21T10:30:00+00:00",
  "week_key": "2025-W17",
  "sentiment_text": "Titulo: Ibovespa fecha em alta...\nResumo: Índice sobe...",
  "predicted_label": "POSITIVE",
  "score_positive": 0.87,
  "score_negative": 0.08,
  "score_neutral": 0.05
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `source` | string | News outlet (e.g., "InfoMoney", "Valor Econômico") |
| `url` | string | Article URL |
| `title` | string | Article headline |
| `description` | string | Subtitle or summary |
| `published_at` | ISO 8601 | Publication timestamp |
| `week_key` | string | ISO week format (e.g., "2025-W17") |
| `sentiment_text` | string | Combined text for classification: title + description |
| `predicted_label` | string | Calibrated prediction: "POSITIVE", "NEGATIVE", or "NEUTRAL" |
| `score_positive` | float | Post-calibration probability for POSITIVE |
| `score_negative` | float | Post-calibration probability for NEGATIVE |
| `score_neutral` | float | Post-calibration probability for NEUTRAL |
- `models/finbert-pt-br-finetuned/` — Primary trained model
- `models/finbert-pt-br-finetuned-secondary/` — Secondary model (ensemble)
- `models/finbert-pt-br-finetuned/training_strategy.json` — Winning strategy metadata (seeds, calibration, metrics)
- `output/samples_train.csv` — Model selection split
- `output/samples_test.csv` — Holdout split (evaluation)

### After Stage 3 (Finetuned Classification)
- `output/finetuned_model_classified.jsonl` — Dataset re-classified with calibrated model

## Training Details

### Dataset

- **Total labeled examples**: 629 hand-annotated Brazilian financial news headlines
- **Train split** (model selection): 503 examples (fixed seed=2026)
- **Holdout split** (evaluation): 126 examples (20%, fixed seed=2026, never seen during training)
- **Within train split**:
  - **Training examples**: 402 (for fine-tuning)
  - **Calibration examples**: 101 (exclusive, for bias optimization)

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Loss function | Label-smoothed cross-entropy (label_smoothing=0.1) |
| Epochs | 4 |
| Learning rate | 7e-6 |
| Weight decay | 0.03 |
| Class weighting | Square-root balanced |
| Warmup ratio | 0.1 |
| Max gradient norm | 0.5 |

### Calibration Method

Post-hoc additive bias calibration learned on the exclusive 101-example calibration split:

```json
{
  "method": "bias_override",
  "scales": {"POSITIVE": 1.0, "NEGATIVE": 1.0, "NEUTRAL": 1.0},
  "biases": {
    "POSITIVE": -0.65,
    "NEGATIVE": -0.20,
    "NEUTRAL": 0.00
  }
}
```

These biases are added to the model logits before softmax to adjust class probabilities for better Macro-F1 on the holdout set.

## Evaluation Results

**Holdout set (126 examples, stratified 20%):**

| Model | Accuracy | Macro F1 | POSITIVE F1 | NEGATIVE F1 | NEUTRAL F1 |
|-------|----------|----------|-------------|-------------|------------|
| Base (lucas-leme/FinBERT-PT-BR) | 34.1% | 0.331 | — | — | — |
| Fine-tuned (calibrated) | **64.3%** | **0.643** | 0.72 | 0.66 | 0.58 |

The fine-tuned model achieves **+30.2 pp accuracy** and **+0.312 macro F1** over the base model.

## Main Dependencies

- **transformers** — HuggingFace Transformers (BERT, AutoTokenizer, model loading)
- **torch** — PyTorch (GPU/CPU computation)
- **datasets** — HuggingFace Datasets (loading pre-built datasets)
- **huggingface_hub** — HuggingFace Hub API (model downloading)
- **scikit-learn** — Metrics, preprocessing, and model evaluation
- **pandas** — DataFrame manipulation and CSV I/O
- **numpy** — Numerical operations
- **tqdm** — Progress bars

See `requirements.txt` for exact versions.

## Model Loading from HuggingFace

Models load automatically from HuggingFace Hub during stage classification:

```python
from transformers import AutoTokenizer, BertForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("lucasalmda/pt-br-financial-sentimental-analysis")
model = BertForSequenceClassification.from_pretrained("lucasalmda/pt-br-financial-sentimental-analysis")
```

The calibration parameters (`training_strategy.json`) are downloaded automatically when needed.

## Troubleshooting

### GPU not detected
```python
import torch
print(torch.cuda.is_available())       # True/False
print(torch.cuda.get_device_name(0))   # GPU name
```

### CUDA out of memory
- Reduce `BATCH_SIZE` in `sentiment_workflow/ml/trainer.py`
- Enable mixed precision or gradient accumulation in `TrainingArguments`

### Dataset not found
- Verify `labeled_samples/labeled_samples.csv` exists and has at least 1 labeled row
- Check file encoding is UTF-8

### Model download fails
- Ensure internet connectivity
- Check HuggingFace token if model is private: `huggingface-cli login`

## Advanced Configuration

### Use a different base model

Edit `sentiment_workflow/config/paths.py`:

```python
BASE_MODEL_NAME = "distilbert-base-multilingual-cased"
```

### Adjust split sizes

Edit `sentiment_workflow/data/data_splits.py`:

```python
HOLDOUT_TEST_SIZE = 0.2      # 20% of total
CALIBRATION_SIZE = 0.2       # 20% of model-selection split
```

### Change fine-tuning hyperparameters

Edit `sentiment_workflow/ml/trainer.py` or `sentiment_workflow/config/paths.py`.

## Citation

If you use this work, please cite:

- Base model: [lucas-leme/FinBERT-PT-BR](https://huggingface.co/lucas-leme/FinBERT-PT-BR)
- Fine-tuned models: [lucasalmda/pt-br-financial-sentimental-analysis](https://huggingface.co/lucasalmda/pt-br-financial-sentimental-analysis)

## License

This project is part of an undergraduate thesis (TCC). Academic use permitted under institutional conditions.

The annotated dataset and fine-tuned models are made available for research and educational purposes.