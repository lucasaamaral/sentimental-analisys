from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


LINE_BREAK_PATTERN = re.compile(r"\s*[\r\n]+\s*")


def normalize_csv_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    if "\n" not in value and "\r" not in value:
        return value

    return LINE_BREAK_PATTERN.sub(" | ", value).strip()


def normalize_csv_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    normalized_df = dataframe.copy()
    text_columns = normalized_df.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        normalized_df[column] = normalized_df[column].map(normalize_csv_value)
    return normalized_df


def write_normalized_csv(dataframe: pd.DataFrame, path: Path, **to_csv_kwargs: Any) -> None:
    normalized_df = normalize_csv_dataframe(dataframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized_df.to_csv(path, **to_csv_kwargs)