"""Shared CSV/threshold helpers for gold-set inference."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def load_metadata(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:limit] if limit is not None else rows


def ensure_csv(path: Path, fieldnames: list[str], resume: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if resume and path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore").writeheader()


def append_csv(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writerow(row)
        handle.flush()


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if any(float(row["threshold"]) != 0.5 for row in rows if row.get("threshold")):
        raise ValueError("Legacy threshold in predictions. Choose a new output path for this run.")
    return rows


def completed_sample_ids(path: Path) -> set[int]:
    return {int(row["sample_id"]) for row in load_prediction_rows(path) if row.get("sample_id")}


def completed_success_ids(path: Path) -> set[int]:
    return {int(row["sample_id"]) for row in load_prediction_rows(path) if row.get("sample_id") and row.get("error", "") == ""}
