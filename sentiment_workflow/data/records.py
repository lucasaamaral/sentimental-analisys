from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    if path.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON array in {path}")
        return payload

    return _load_jsonl_records(path)


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    return load_records(path)


def get_checkpoint_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.checkpoint.jsonl")


def load_records_with_checkpoint(path: Path) -> list[dict[str, Any]]:
    records = load_records(path)
    checkpoint_path = get_checkpoint_path(path)
    if checkpoint_path.exists():
        records.extend(_load_jsonl_records(checkpoint_path))
    return records


def load_seen_values(path: Path, key: str) -> set[str]:
    seen_values: set[str] = set()
    for record in load_records_with_checkpoint(path):
        value = record.get(key)
        if value:
            seen_values.add(str(value))
    return seen_values


def open_checkpoint_handle(path: Path):
    checkpoint_path = get_checkpoint_path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    return open(checkpoint_path, "a", encoding="utf-8")


def write_jsonl_record(handle, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def finalize_records_file(path: Path) -> None:
    checkpoint_path = get_checkpoint_path(path)
    if not checkpoint_path.exists():
        return

    records = load_records(path)
    records.extend(_load_jsonl_records(checkpoint_path))
    with open(path, "w", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            json.dump(records, handle, ensure_ascii=False)
        else:
            for record in records:
                write_jsonl_record(handle, record)
    checkpoint_path.unlink()
