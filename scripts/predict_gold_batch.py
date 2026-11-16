#!/usr/bin/env python3
"""Batch inference over the raw external gold PE dataset.

Safety model:
- Reads benign defanged PE files (`.exe.txt`, `.dll.txt`, `.pe.txt`) as bytes only.
- Reads MalwareBazaar `.zip.txt` samples as one encrypted ZIP member in memory
  using password candidate `infected`; raw malware PE is never written to disk.
- Performs static parsing/feature extraction only; samples are never executed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_gold_features import FEATURE_DIM, PEFeatureExtractor, is_probably_pe, read_gold_bytes, sorel_signed_log_postprocess

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


def load_threshold(path: Path | None, default: float) -> tuple[float, str]:
    if path is None:
        return float(default), "cli_default"
    if not path.exists():
        raise FileNotFoundError(f"threshold file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    candidates: list[Any] = [data.get("threshold"), data.get("selected_threshold"), data.get("best_threshold")]
    if isinstance(data.get("chosen"), dict):
        candidates.append(data["chosen"].get("threshold"))
    if isinstance(data.get("selected"), dict):
        candidates.append(data["selected"].get("threshold"))
    for value in candidates:
        if value is not None:
            return float(value), str(path)
    raise ValueError(f"no threshold key found in {path}")


def load_metadata(path: Path, limit: int | None) -> list[dict[str, str]]:
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


def completed_sample_ids(path: Path) -> set[int]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["sample_id"]) for row in csv.DictReader(handle) if row.get("sample_id")}


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def prediction_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    ok = [row for row in rows if row.get("error", "") == ""]
    labels = np.asarray([int(row["label"]) for row in ok], dtype=np.int8)
    probabilities = np.asarray([float(row["malware_probability"]) for row in ok], dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int8) if len(ok) else np.asarray([], dtype=np.int8)
    output: dict[str, Any] = {
        "threshold": float(threshold),
        "processed_rows": len(rows),
        "scored_rows": len(ok),
        "error_rows": len(rows) - len(ok),
        "label_counts": {
            "benign_or_safe_0": int(np.sum(labels == 0)) if len(ok) else 0,
            "malicious_or_dangerous_1": int(np.sum(labels == 1)) if len(ok) else 0,
        },
    }
    if len(ok) == 0:
        return output
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    output.update(
        {
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall_tpr": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
            "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }
    )
    if len(set(labels.tolist())) == 2:
        output["roc_auc"] = float(roc_auc_score(labels, probabilities))
        output["pr_auc"] = float(average_precision_score(labels, probabilities))
    return output


def write_metrics(path: Path, rows: list[dict[str, Any]], threshold: float, args: argparse.Namespace, threshold_source: str, started: float) -> None:
    payload = prediction_metrics(rows, threshold)
    payload.update(
        {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metadata": str(args.metadata),
            "model": str(args.model),
            "threshold_source": threshold_source,
            "predictions": str(args.predictions_output),
            "errors": str(args.errors_output),
            "feature_dim_original": FEATURE_DIM,
            "feature_dim_model": int(args.feature_dim_model),
            "postprocess": args.postprocess,
            "sample_timeout": args.sample_timeout,
            "runtime_seconds": round(time.time() - started, 3),
            "safety": "PE bytes are parsed statically only; MalwareBazaar ZIP members are read in memory and raw malware PE is not written to disk.",
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def score_one(row: dict[str, str], payload: dict[str, Any], threshold: float, postprocess: str) -> dict[str, Any]:
    label = int(row["label"])
    data, input_mode = read_gold_bytes(row)
    if not is_probably_pe(data):
        raise RuntimeError("extracted bytes are not PE")
    feature_sha = hashlib.sha256(data).hexdigest()
    expected_sha = row.get("sha256", "").lower()
    if expected_sha and feature_sha.lower() != expected_sha:
        raise RuntimeError(f"feature bytes sha256 mismatch: {feature_sha}")
    vector = PEFeatureExtractor().feature_vector(data)
    if vector.shape != (FEATURE_DIM,):
        raise RuntimeError(f"feature shape {vector.shape}; expected ({FEATURE_DIM},)")
    if postprocess == "model":
        if payload.get("postprocess") == "sorel_signed_log":
            vector = sorel_signed_log_postprocess(vector)
    elif postprocess == "sorel_signed_log":
        vector = sorel_signed_log_postprocess(vector)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("outputs/gold/gold_final_metadata.csv"))
    parser.add_argument("--model", type=Path, default=Path("outputs/models/baseline_sgd.pkl"))
    parser.add_argument("--threshold-file", type=Path, default=Path("outputs/metrics/threshold_selection.json"))
    parser.add_argument("--threshold", type=float, default=0.5, help="Fallback only when --threshold-file is omitted.")
    parser.add_argument("--predictions-output", type=Path, default=Path("outputs/metrics/predictions_gold.csv"))
    parser.add_argument("--metrics-output", type=Path, default=Path("outputs/metrics/gold_metrics.json"))
    parser.add_argument("--errors-output", type=Path, default=Path("outputs/metrics/gold_inference_errors.csv"))
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for smoke runs after training/validation is complete.")
    parser.add_argument("--dry-run", action="store_true", help="Load metadata/model/threshold and print planned work; do not process samples or write outputs.")
    parser.add_argument("--resume", action="store_true", help="Skip sample_ids already present in predictions output.")
    parser.add_argument("--sample-timeout", type=int, default=60, help="Seconds allowed per sample; use 0 to disable process isolation/timeouts.")
    parser.add_argument("--postprocess", choices=["model", "sorel_signed_log", "raw"], default="model")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    threshold, threshold_source = load_threshold(args.threshold_file, args.threshold) if args.threshold_file else load_threshold(None, args.threshold)
    rows = load_metadata(args.metadata, args.limit)
    payload = joblib.load(args.model)
    selected_indices = payload.get("selected_indices")
    if selected_indices is None:
        raise RuntimeError("model payload missing selected_indices")
    args.feature_dim_model = len(selected_indices)
    if args.feature_dim_model != 2192:
        raise RuntimeError(f"model selected feature count {args.feature_dim_model}; expected 2192")

    if args.dry_run:
        print(json.dumps({
            "metadata": str(args.metadata),
            "rows_selected": len(rows),
            "model": str(args.model),
            "feature_dim_original": FEATURE_DIM,
            "feature_dim_model": args.feature_dim_model,
            "threshold": threshold,
            "threshold_source": threshold_source,
            "outputs": {
                "predictions": str(args.predictions_output),
                "metrics": str(args.metrics_output),
                "errors": str(args.errors_output),
            },
            "dry_run": True,
        }, indent=2, ensure_ascii=False))
        return 0

    started = time.time()
    done = completed_sample_ids(args.predictions_output) if args.resume else set()
    ensure_csv(args.predictions_output, PREDICTION_FIELDS, args.resume)
    ensure_csv(args.errors_output, ERROR_FIELDS, args.resume)
    all_rows: list[dict[str, Any]] = load_prediction_rows(args.predictions_output) if args.resume else []

    for row in rows:
        if int(row["sample_id"]) in done:
            continue
        result = score_one_with_timeout(row, payload, threshold, args.postprocess, args.sample_timeout)
        append_csv(args.predictions_output, PREDICTION_FIELDS, result)
        all_rows.append(result)
        if result.get("error", "") != "":
            append_csv(args.errors_output, ERROR_FIELDS, result)
        else:
            append_csv(args.errors_output, ERROR_FIELDS, {"sample_id": result["sample_id"], "label": result["label"], "sha256": result["sha256"], "stored_path": result["stored_path"], "error": ""})
        write_metrics(args.metrics_output, all_rows, threshold, args, threshold_source, started)

    write_metrics(args.metrics_output, all_rows, threshold, args, threshold_source, started)
    print(json.dumps({"processed_rows": len(all_rows), "predictions": str(args.predictions_output), "metrics": str(args.metrics_output), "errors": str(args.errors_output)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
