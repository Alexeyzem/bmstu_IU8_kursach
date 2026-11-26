"""Prediction records and summary metrics shared by the gold inference jobs."""
import json
import time
from pathlib import Path
from typing import Any

from pe_static_analyzer.core.metrics import scored_rows_metrics
from pe_static_analyzer.features.ember import FEATURE_DIM


PREDICTION_FIELDS = [
    "sample_id",
    "label",
    "source",
    "sha256",
    "stored_path",
    "input_mode",
    "feature_dim_original",
    "feature_dim_model",
    "malware_probability",
    "threshold",
    "prediction",
    "prediction_label",
    "error",
]
ERROR_FIELDS = ["sample_id", "label", "sha256", "stored_path", "error"]


def error_row(row: dict[str, str], feature_dim_model: int, threshold: float, error: str) -> dict[str, Any]:
    return {
        "sample_id": int(row.get("sample_id", 0) or 0),
        "label": row.get("label", ""),
        "source": row.get("source", ""),
        "sha256": row.get("sha256", ""),
        "stored_path": row.get("stored_path", ""),
        "input_mode": "",
        "feature_dim_original": FEATURE_DIM,
        "feature_dim_model": feature_dim_model,
        "malware_probability": "",
        "threshold": threshold,
        "prediction": "",
        "prediction_label": "",
        "error": error,
    }


def scored_row(row: dict[str, Any], dimension: int, probability: float, threshold: float) -> dict[str, Any]:
    prediction = int(probability >= threshold)
    return {
        "sample_id": int(row["sample_id"]),
        "label": int(row["label"]),
        "source": row.get("source", ""),
        "sha256": row["sha256"],
        "stored_path": row.get("stored_path", ""),
        "input_mode": row.get("input_mode", ""),
        "feature_dim_original": FEATURE_DIM,
        "feature_dim_model": dimension,
        "malware_probability": probability,
        "threshold": threshold,
        "prediction": prediction,
        "prediction_label": "malicious_or_dangerous" if prediction else "benign_or_safe",
        "error": "",
    }


def write_metrics(path: Path, rows: list[dict[str, Any]], threshold: float, settings, threshold_source: str, started: float, **run_details: Any) -> None:
    payload = scored_rows_metrics(rows, threshold)
    payload.update(
        {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metadata": str(settings.metadata),
            "model": str(settings.model),
            "threshold_source": threshold_source,
            "predictions": str(settings.predictions_output),
            "errors": str(settings.errors_output),
            "feature_dim_original": FEATURE_DIM,
            "feature_dim_model": int(settings.feature_dim_model),
            "postprocess": settings.postprocess,
            "sample_timeout": settings.sample_timeout,
            **run_details,
            "runtime_seconds": round(time.time() - started, 3),
            "safety": "PE bytes are parsed statically only; MalwareBazaar ZIP members are read in memory and raw malware PE is not written to disk.",
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
