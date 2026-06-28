# PT-BR Financial Sentiment Analysis - Fine-Tuning Workflow

A comprehensive end-to-end pipeline for sentiment classification of Brazilian Portuguese financial news using FinBERT. The current workflow compares the original base model against a locally fine-tuned ensemble and exports article-level predictions plus a weekly sentiment score.

## Pre-Trained Models

The project starts from the public base model:

- **[lucas-leme/FinBERT-PT-BR](https://huggingface.co/lucas-leme/FinBERT-PT-BR)** - Base Portuguese financial sentiment model

The fine-tuned model is trained locally under `models/` and uses a three-seed ensemble for final inference:

- `models/finbert-pt-br-finetuned/`
- `models/finbert-pt-br-finetuned-seed-123/`
- `models/finbert-pt-br-finetuned-seed-456/`

All models classify financial news as **POSITIVE**, **NEGATIVE**, or **NEUTRAL**.

## Architecture

The workflow executes **4 sequential stages**:

### Stage 1: Base Classification

Generates baseline predictions on the full dataset using the pre-trained `lucas-leme/FinBERT-PT-BR` model.

### Stage 2: Fine-Tuning

Trains the fine-tuned classifier on a stratified training split using square-root balanced class weights. A separate validation split controls the hyperparameter search and best-checkpoint selection. The independent holdout test split is evaluated only after the final strategy is selected.

The validation-selected strategy uses:

- 4 epochs
- learning rate `2e-5`
- weight decay `0.03`
- label smoothing `0.10`
- selected by validation macro-F1 from candidate learning-rate, weight-decay and label-smoothing settings
- seeds `789`, `123` and `456`
- validation-based best-checkpoint selection
- simple average of ensemble logits
- final decision by direct argmax

### Stage 3: Final Classification

Re-classifies the complete dataset with the fine-tuned ensemble and writes the final predictions to `output/finetuned_model_classified.jsonl`.

### Stage 4: Weekly Sentiment Scoring

Aggregates fine-tuned article predictions by ISO week and writes centered weekly sentiment scores plus moving averages to `output/weekly_sentiment_scores.csv`.

## Project Structure

```text
sentimental-analisys/
|-- main.py                              # Entry point - dispatch to workflow tasks
|-- README.md                            # Documentation
|-- requirements.txt                     # Python dependencies
|-- labeled_samples/                     # Hand-annotated data and eval files
|   |-- labeled_samples.csv
|   |-- labeled_base_model_eval.csv
|   `-- labeled_finetuned_model_eval.csv
|-- classification_workflow/             # Main package
|   |-- config/
|   |   |-- labels.py
|   |   `-- paths.py
|   |-- data/
|   |   |-- csv_utils.py
|   |   |-- data_splits.py
|   |   |-- dataset_source.py
|   |   `-- records.py
|   |-- ml/
|   |   |-- trainer.py
|   |   |-- inference.py
|   |   `-- evaluation.py
|   |-- classify_base_model.py 
|   |-- run_model_finetuning.py
|   |-- classify_finetuned_model.py
|   `-- generate_sentiment_scores.py
|-- notebooks/
|   |-- base_model_analisys.ipynb
|   `-- finetuned_model_comparison.ipynb
```

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

Executes sequentially:

1. `classify-base-model`
2. `train-finetuned-model`
3. `classify-finetuned-model`
4. `generate-sentiment-scores`

### Run Individual Stages

```pwsh
python main.py classify-base-model
python main.py train-finetuned-model
python main.py classify-finetuned-model
python main.py generate-sentiment-scores
```

## Output Format (JSONL)

Each line is a JSON object with predictions:

```json
{
  "source": "InfoMoney",
  "url": "https://www.infomoney.com.br/...",
  "title": "Ibovespa fecha em alta...",
  "description": "Indice sobe com expectativa...",
  "published_at": "2025-04-21T10:30:00+00:00",
  "week_key": "2025-W17",
  "sentiment_text": "Titulo: Ibovespa fecha em alta...\nResumo: Indice sobe...",
  "predicted_label": "POSITIVE",
  "score_positive": 0.87,
  "score_negative": 0.08,
  "score_neutral": 0.05
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `source` | string | News outlet, such as InfoMoney, Valor Economico or Exame |
| `url` | string | Article URL |
| `title` | string | Article headline |
| `description` | string | Subtitle or summary |
| `published_at` | ISO 8601 | Publication timestamp |
| `week_key` | string | ISO week format, such as `2025-W17` |
| `sentiment_text` | string | Combined text used for classification |
| `predicted_label` | string | Prediction: `POSITIVE`, `NEGATIVE` or `NEUTRAL` |
| `score_positive` | float | Probability for POSITIVE |
| `score_negative` | float | Probability for NEGATIVE |
| `score_neutral` | float | Probability for NEUTRAL |

## Evaluation Results

Fixed stratified train/validation/test split: 399 train, 100 validation and 125 independent test examples. The test split uses seed `2026`; the validation split uses seed `2027`.
Validation search ranking by macro-F1:

| Rank | Configuration | Validation Accuracy | Validation Macro F1 |
|-----:|---------------|--------------------:|--------------------:|
| 1 | `lr=2e-5`, `weight_decay=0.03`, `label_smoothing=0.10` | 0.670 | 0.671 |
| 2 | `lr=1e-5`, `weight_decay=0.03`, `label_smoothing=0.10` | 0.650 | 0.647 |
| 3 | `lr=1e-5`, `weight_decay=0.01`, `label_smoothing=0.05` | 0.640 | 0.641 |
| 4 | `lr=1e-5`, `weight_decay=0.00`, `label_smoothing=0.00` | 0.640 | 0.640 |
| 5 | `lr=5e-6`, `weight_decay=0.00`, `label_smoothing=0.00` | 0.610 | 0.605 |

| Model | Accuracy | Macro F1 |
|-------|---------:|---------:|
| Base (`lucas-leme/FinBERT-PT-BR`) | 0.336 | 0.325 |
| Fine-tuned ensemble | 0.784 | 0.783 |

The current labeled sample contains 624 reviewed examples in `labeled_samples/labeled_samples.csv`.

## Notebooks

The `notebooks/` folder contains the analytical layer used for diagnostics. Weekly sentiment score generation is part of the main pipeline:

- `base_model_analisys.ipynb` - Baseline behavior, negative bias and base-model diagnostics
- `finetuned_model_comparison.ipynb` - Fine-tuned model quality, F1 comparison and bias correction
- Weekly sentiment score generation now runs through `python main.py generate-sentiment-scores`.

Before running the notebooks, make sure the pipeline has generated:

- `output/base_model_classified.jsonl`
- `output/finetuned_model_classified.jsonl`
- `output/weekly_sentiment_scores.csv`

## Main Dependencies

- `transformers` - HuggingFace Transformers
- `torch` - PyTorch
- `datasets` - HuggingFace Datasets
- `huggingface_hub` - HuggingFace Hub API
- `scikit-learn` - Metrics and evaluation
- `pandas` - DataFrame manipulation and CSV I/O
- `numpy` - Numerical operations
- `tqdm` - Progress bars

See `requirements.txt` for exact versions.

## Troubleshooting

### GPU not detected

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

### CUDA out of memory
- Reduce `BATCH_SIZE` in `sentiment_workflow/ml/trainer.py`
- Enable mixed precision or gradient accumulation in `TrainingArguments`

### Dataset not found

- Verify that `labeled_samples/labeled_samples.csv` exists.
- Verify that the generated prediction files exist under `output/`.

### Notebook files not found

Run the main workflow before opening the notebooks:

```pwsh
python main.py
```

## License

This project is part of an undergraduate thesis (TCC). Academic use permitted under institutional conditions.

The annotated dataset and fine-tuned models are made available for research and educational purposes.