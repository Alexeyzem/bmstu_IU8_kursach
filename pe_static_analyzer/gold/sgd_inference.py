#!/usr/bin/env python3
"""Batch inference over the raw external gold PE dataset.

Safety model:
- Reads benign defanged PE files (`.exe.txt`, `.dll.txt`, `.pe.txt`) as bytes only.
- Reads MalwareBazaar `.zip.txt` samples as one encrypted ZIP member in memory
  using password candidate `infected`; raw malware PE is never written to disk.
- Performs static parsing/feature extraction only; samples are never executed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np


from pe_static_analyzer.core.metrics import scored_rows_metrics
from pe_static_analyzer.features.extraction import FEATURE_DIM, gold_vector
from pe_static_analyzer.gold.io import append_csv, completed_sample_ids, ensure_csv, load_metadata, load_prediction_rows

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


@dataclass
class GoldSgdSettings:
    metadata: Path = Path('outputs/gold/gold_final_metadata.csv')
    model: Path = Path('outputs/models/baseline_sgd.pkl')
    predictions_output: Path = Path('outputs/metrics/predictions_gold.csv')
    metrics_output: Path = Path('outputs/metrics/gold_metrics.json')
    errors_output: Path = Path('outputs/metrics/gold_inference_errors.csv')
    limit: int | None = None
    dry_run: bool = False
    resume: bool = False
    sample_timeout: int = 60
    postprocess: str = 'model'


def base_error_row(row: dict[str, str], payload: dict[str, Any], threshold: float, error: str) -> dict[str, Any]:
    return {
        "sample_id": int(row.get("sample_id", 0) or 0),
        "label": row.get("label", ""),
        "source": row.get("source", ""),
        "sha256": row.get("sha256", ""),
        "stored_path": row.get("stored_path", ""),
        "input_mode": "",
        "feature_dim_original": FEATURE_DIM,
        "feature_dim_model": len(payload.get("selected_indices", [])),
        "malware_probability": "",
        "threshold": threshold,
        "prediction": "",
        "prediction_label": "",
        "error": error,
    }


def write_metrics(path: Path, rows: list[dict[str, Any]], threshold: float, settings: GoldSgdSettings, threshold_source: str, started: float) -> None:
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
            "runtime_seconds": round(time.time() - started, 3),
            "safety": "PE bytes are parsed statically only; MalwareBazaar ZIP members are read in memory and raw malware PE is not written to disk.",
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def score_one(row: dict[str, str], payload: dict[str, Any], threshold: float, postprocess: str) -> dict[str, Any]:
    label = int(row["label"])
    if postprocess == "model":
        postprocess = "sorel_signed_log" if payload.get("postprocess") == "sorel_signed_log" else "raw"
    vector, feature_sha, input_mode = gold_vector(row, postprocess, require_hash=False)
    selected_indices = np.asarray(payload["selected_indices"], dtype=np.int32)
    x = vector[selected_indices].reshape(1, -1)
    x_scaled = payload["scaler"].transform(x)
    probability = float(payload["model"].predict_proba(x_scaled)[0, 1])
    prediction = int(probability >= threshold)
    return {
        "sample_id": int(row["sample_id"]),
        "label": label,
        "source": row.get("source", ""),
        "sha256": feature_sha,
        "stored_path": row.get("stored_path", ""),
        "input_mode": input_mode,
        "feature_dim_original": FEATURE_DIM,
        "feature_dim_model": int(len(selected_indices)),
        "malware_probability": probability,
        "threshold": threshold,
        "prediction": prediction,
        "prediction_label": "malicious_or_dangerous" if prediction else "benign_or_safe",
        "error": "",
    }


def score_one_catching(row: dict[str, str], payload: dict[str, Any], threshold: float, postprocess: str) -> dict[str, Any]:
    try:
        return score_one(row, payload, threshold, postprocess)
    except Exception as exc:  # noqa: BLE001 - batch inference records per-sample failures and continues.
        return base_error_row(row, payload, threshold, str(exc))


def score_one_worker(row: dict[str, str], payload: dict[str, Any], threshold: float, postprocess: str, queue: mp.Queue) -> None:
    queue.put(score_one_catching(row, payload, threshold, postprocess))


def score_one_with_timeout(row: dict[str, str], payload: dict[str, Any], threshold: float, postprocess: str, timeout: int) -> dict[str, Any]:
    if timeout <= 0:
        return score_one_catching(row, payload, threshold, postprocess)
    ctx = mp.get_context("fork")
    queue: mp.Queue = ctx.Queue(maxsize=1)
    process = ctx.Process(target=score_one_worker, args=(row, payload, threshold, postprocess, queue))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(5)
        return base_error_row(row, payload, threshold, f"sample_timeout_seconds={timeout}")
    if process.exitcode != 0:
        return base_error_row(row, payload, threshold, f"worker_exitcode={process.exitcode}")
    if queue.empty():
        return base_error_row(row, payload, threshold, "worker_returned_no_result")
    return queue.get()


class GoldSgdInferenceJob:
    """Runnable job object for this CLI pipeline."""

    def __init__(self, settings: GoldSgdSettings) -> None:
        self.settings = settings

    def run(self) -> int:
        settings = self.settings
        threshold, threshold_source = 0.5, "fixed"
        rows = load_metadata(settings.metadata, settings.limit)
        payload = joblib.load(settings.model)
        selected_indices = payload.get("selected_indices")
        if selected_indices is None:
            raise RuntimeError("model payload missing selected_indices")
        settings.feature_dim_model = len(selected_indices)
        if settings.feature_dim_model != 2192:
            raise RuntimeError(f"model selected feature count {settings.feature_dim_model}; expected 2192")

        if settings.dry_run:
            print(json.dumps({
                "metadata": str(settings.metadata),
                "rows_selected": len(rows),
                "model": str(settings.model),
                "feature_dim_original": FEATURE_DIM,
                "feature_dim_model": settings.feature_dim_model,
                "threshold": threshold,
                "threshold_source": threshold_source,
                "outputs": {
                    "predictions": str(settings.predictions_output),
                    "metrics": str(settings.metrics_output),
                    "errors": str(settings.errors_output),
                },
                "dry_run": True,
            }, indent=2, ensure_ascii=False))
            return 0

        started = time.time()
        done = completed_sample_ids(settings.predictions_output) if settings.resume else set()
        ensure_csv(settings.predictions_output, PREDICTION_FIELDS, settings.resume)
        ensure_csv(settings.errors_output, ERROR_FIELDS, settings.resume)
        all_rows: list[dict[str, Any]] = load_prediction_rows(settings.predictions_output) if settings.resume else []

        for row in rows:
            if int(row["sample_id"]) in done:
                continue
            result = score_one_with_timeout(row, payload, threshold, settings.postprocess, settings.sample_timeout)
            append_csv(settings.predictions_output, PREDICTION_FIELDS, result)
            all_rows.append(result)
            if result.get("error", "") != "":
                append_csv(settings.errors_output, ERROR_FIELDS, result)
            else:
                append_csv(settings.errors_output, ERROR_FIELDS, {"sample_id": result["sample_id"], "label": result["label"], "sha256": result["sha256"], "stored_path": result["stored_path"], "error": ""})
            write_metrics(settings.metrics_output, all_rows, threshold, settings, threshold_source, started)

        write_metrics(settings.metrics_output, all_rows, threshold, settings, threshold_source, started)
        print(json.dumps({"processed_rows": len(all_rows), "predictions": str(settings.predictions_output), "metrics": str(settings.metrics_output), "errors": str(settings.errors_output)}, indent=2, ensure_ascii=False))
        return 0


def main() -> int:
    return GoldSgdInferenceJob(GoldSgdSettings()).run()


if __name__ == "__main__":
    raise SystemExit(main())
