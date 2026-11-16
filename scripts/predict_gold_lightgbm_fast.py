#!/usr/bin/env python3
"""Fast parallel raw-file gold inference for the separate LightGBM PE model.

This avoids passing a LightGBM Booster into per-sample timeout workers. Workers
only read/decrypt PE bytes and extract EMBER-compatible features. The parent
process owns the LightGBM Booster and performs prediction after a feature vector
returns. That keeps LIEF parsing isolated while avoiding LightGBM/OpenMP + fork
hangs on macOS.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import queue as queue_module
import sys
import time
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

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


def load_payload(path: Path) -> dict[str, Any]:
    import lightgbm as lgb

    payload = joblib.load(path)
    if payload.get("model_type") != "lightgbm":
        raise RuntimeError(f"model payload is not LightGBM: {path}")
    if "model_string" not in payload:
        raise RuntimeError("LightGBM payload missing model_string")
    if "selected_indices" not in payload:
        raise RuntimeError("LightGBM payload missing selected_indices")
    payload["booster"] = lgb.Booster(model_str=payload["model_string"])
    return payload


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


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def completed_success_ids(path: Path) -> set[int]:
    return {int(row["sample_id"]) for row in load_prediction_rows(path) if row.get("sample_id") and row.get("error", "") == ""}


def base_row(row: dict[str, str], feature_dim_model: int, threshold: float, error: str) -> dict[str, Any]:
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


def extract_feature(row: dict[str, str], extraction_postprocess: str) -> dict[str, Any]:
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
    if extraction_postprocess == "sorel_signed_log":
        vector = sorel_signed_log_postprocess(vector)
    elif extraction_postprocess != "raw":
        raise ValueError(f"unknown extraction postprocess: {extraction_postprocess}")
    return {
        "sample_id": int(row["sample_id"]),
        "label": label,
        "source": row.get("source", ""),
        "sha256": feature_sha,
        "stored_path": row.get("stored_path", ""),
        "input_mode": input_mode,
        "vector": vector.astype(np.float32, copy=False),
    }


def extract_worker(row: dict[str, str], extraction_postprocess: str, queue: mp.Queue) -> None:
    try:
        queue.put({"ok": True, "result": extract_feature(row, extraction_postprocess)})
    except Exception as exc:  # noqa: BLE001 - per-sample errors are reported and processing continues.
        queue.put({"ok": False, "error": str(exc)})


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
            "extraction_postprocess": args.extraction_postprocess,
            "sample_timeout": args.sample_timeout,
            "workers": args.workers,
            "runtime_seconds": round(time.time() - started, 3),
            "safety": "PE bytes are parsed statically only; MalwareBazaar ZIP members are read in memory and raw malware PE is not written to disk.",
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def predict_from_feature(feature: dict[str, Any], payload: dict[str, Any], threshold: float) -> dict[str, Any]:
    selected_indices = np.asarray(payload["selected_indices"], dtype=np.int32)
    vector = feature.pop("vector")
    x = vector[selected_indices].reshape(1, -1)
    probability = float(payload["booster"].predict(x)[0])
    prediction = int(probability >= threshold)
    return {
        "sample_id": int(feature["sample_id"]),
        "label": int(feature["label"]),
        "source": feature.get("source", ""),
        "sha256": feature["sha256"],
        "stored_path": feature.get("stored_path", ""),
        "input_mode": feature.get("input_mode", ""),
        "feature_dim_original": FEATURE_DIM,
        "feature_dim_model": int(len(selected_indices)),
        "malware_probability": probability,
        "threshold": threshold,
        "prediction": prediction,
        "prediction_label": "malicious_or_dangerous" if prediction else "benign_or_safe",
        "error": "",
    }


def finish_process(process: mp.Process, queue: mp.Queue, row: dict[str, str], payload: dict[str, Any], threshold: float, error: str | None = None) -> dict[str, Any]:
    feature_dim_model = int(len(payload["selected_indices"]))
    if error is not None:
        return base_row(row, feature_dim_model, threshold, error)
    if process.exitcode != 0:
        return base_row(row, feature_dim_model, threshold, f"worker_exitcode={process.exitcode}")
    try:
        message = queue.get(timeout=1)
    except queue_module.Empty:
        return base_row(row, feature_dim_model, threshold, "worker_returned_no_result")
    if not message.get("ok"):
        return base_row(row, feature_dim_model, threshold, str(message.get("error", "unknown_error")))
    return predict_from_feature(message["result"], payload, threshold)


def append_outputs(result: dict[str, Any], predictions_output: Path, errors_output: Path) -> None:
    append_csv(predictions_output, PREDICTION_FIELDS, result)
    append_csv(
        errors_output,
        ERROR_FIELDS,
        {
            "sample_id": result["sample_id"],
            "label": result["label"],
            "sha256": result["sha256"],
            "stored_path": result["stored_path"],
            "error": result.get("error", ""),
        },
    )


def run_parallel(rows: list[dict[str, str]], payload: dict[str, Any], threshold: float, args: argparse.Namespace, threshold_source: str) -> list[dict[str, Any]]:
    ctx = mp.get_context(args.start_method)
    started = time.time()
    done_success = completed_success_ids(args.predictions_output) if args.resume else set()
    all_rows: list[dict[str, Any]] = [row for row in load_prediction_rows(args.predictions_output) if row.get("error", "") == ""] if args.resume else []
    pending = [row for row in rows if int(row["sample_id"]) not in done_success]
    active: dict[int, tuple[mp.Process, mp.Queue, dict[str, str], float]] = {}
    cursor = 0
    processed_this_run = 0

    def start_next() -> bool:
        nonlocal cursor
        if cursor >= len(pending):
            return False
        row = pending[cursor]
        cursor += 1
        queue: mp.Queue = ctx.Queue(maxsize=1)
        process = ctx.Process(target=extract_worker, args=(row, args.extraction_postprocess, queue))
        process.start()
        active[process.pid or -cursor] = (process, queue, row, time.monotonic())
        return True

    while len(active) < args.workers and start_next():
        pass

    while active:
        finished: list[int] = []
        now = time.monotonic()
        for key, (process, queue, row, process_started) in list(active.items()):
            error = None
            if process.is_alive() and now - process_started >= args.sample_timeout:
                process.terminate()
                process.join(2)
                error = f"sample_timeout_seconds={args.sample_timeout}"
            if error is None and process.is_alive():
                continue
            if error is None:
                process.join(0)
            result = finish_process(process, queue, row, payload, threshold, error)
            try:
                queue.close()
            except Exception:
                pass
            append_outputs(result, args.predictions_output, args.errors_output)
            all_rows.append(result)
            processed_this_run += 1
            finished.append(key)

            if args.metrics_every > 0 and processed_this_run % args.metrics_every == 0:
                write_metrics(args.metrics_output, all_rows, threshold, args, threshold_source, started)
            if args.progress_every > 0 and processed_this_run % args.progress_every == 0:
                print(json.dumps({"stage": "gold_progress", "processed_this_run": processed_this_run, "total_rows_in_outputs": len(all_rows), "pending_total": len(pending), "scored_rows": sum(1 for item in all_rows if item.get("error", "") == ""), "runtime_seconds": round(time.time() - started, 3)}, ensure_ascii=False), flush=True)

        for key in finished:
            active.pop(key, None)
            start_next()
        if not finished:
            time.sleep(0.05)

    write_metrics(args.metrics_output, all_rows, threshold, args, threshold_source, started)
    return all_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("outputs/gold/gold_final_metadata.csv"))
    parser.add_argument("--model", type=Path, default=Path("outputs/models/lightgbm_pe_full.pkl"))
    parser.add_argument("--threshold-file", type=Path, default=Path("outputs/metrics/lightgbm_threshold_selection.json"))
    parser.add_argument("--threshold", type=float, default=0.5, help="Fallback only when --threshold-file is omitted.")
    parser.add_argument("--predictions-output", type=Path, default=Path("outputs/metrics/predictions_gold_lightgbm_fast.csv"))
    parser.add_argument("--metrics-output", type=Path, default=Path("outputs/metrics/gold_metrics_lightgbm_fast.json"))
    parser.add_argument("--errors-output", type=Path, default=Path("outputs/metrics/gold_inference_errors_lightgbm_fast.csv"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Load metadata/model/threshold and print planned work; do not process samples or write outputs.")
    parser.add_argument("--resume", action="store_true", help="Skip only previously successful rows; previous error rows are retried.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sample-timeout", type=int, default=10)
    parser.add_argument("--start-method", choices=["spawn", "fork", "forkserver"], default="spawn")
    parser.add_argument("--postprocess", choices=["model", "sorel_signed_log", "raw"], default="model")
    parser.add_argument("--metrics-every", type=int, default=25)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.sample_timeout <= 0:
        raise SystemExit("--sample-timeout must be > 0 for isolated parallel extraction")
    threshold, threshold_source = load_threshold(args.threshold_file, args.threshold) if args.threshold_file else load_threshold(None, args.threshold)
    rows = load_metadata(args.metadata, args.limit)
    payload = load_payload(args.model)
    args.feature_dim_model = len(payload["selected_indices"])
    if args.postprocess == "model":
        args.extraction_postprocess = "sorel_signed_log" if payload.get("postprocess") == "sorel_signed_log" else "raw"
    else:
        args.extraction_postprocess = args.postprocess

    if args.dry_run:
        print(json.dumps({
            "metadata": str(args.metadata),
            "rows_selected": len(rows),
            "model": str(args.model),
            "feature_dim_original": FEATURE_DIM,
            "feature_dim_model": args.feature_dim_model,
            "threshold": threshold,
            "threshold_source": threshold_source,
            "workers": args.workers,
            "sample_timeout": args.sample_timeout,
            "start_method": args.start_method,
            "postprocess": args.postprocess,
            "extraction_postprocess": args.extraction_postprocess,
            "outputs": {
                "predictions": str(args.predictions_output),
                "metrics": str(args.metrics_output),
                "errors": str(args.errors_output),
            },
            "dry_run": True,
        }, indent=2, ensure_ascii=False))
        return 0

    ensure_csv(args.predictions_output, PREDICTION_FIELDS, args.resume)
    ensure_csv(args.errors_output, ERROR_FIELDS, args.resume)
    result_rows = run_parallel(rows, payload, threshold, args, threshold_source)
    print(json.dumps({"processed_rows": len(result_rows), "scored_rows": sum(1 for row in result_rows if row.get("error", "") == ""), "predictions": str(args.predictions_output), "metrics": str(args.metrics_output), "errors": str(args.errors_output)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
