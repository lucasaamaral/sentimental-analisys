LABELS = ("POSITIVE", "NEGATIVE", "NEUTRAL")
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}

BASE_MODEL_LABEL_MAP = {
    "LABEL_0": "POSITIVE",
    "LABEL_1": "NEGATIVE",
    "LABEL_2": "NEUTRAL",
}


def normalise_label(value: object) -> str:
    return str(value).strip().upper()