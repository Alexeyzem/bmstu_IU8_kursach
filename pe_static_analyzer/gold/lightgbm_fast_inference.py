#!/usr/bin/env python3
"""Fast parallel raw-file gold inference for the separate LightGBM PE model.

This avoids passing a LightGBM Booster into per-sample timeout workers. Workers
only read/decrypt PE bytes and extract EMBER-compatible features. The parent
process owns the LightGBM Booster and performs prediction after a feature vector
returns. That keeps LIEF parsing isolated while avoiding LightGBM/OpenMP + fork
hangs on macOS.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import multiprocessing as mp
from pathlib import Path
import queue as queue_module
import time
from typing import Any

import numpy as np

from pe_static_analyzer.prediction.model import LightGbmDetector
from pe_static_analyzer.settings import DEFAULT_MODEL
from pe_static_analyzer.features.ember import FEATURE_DIM
from pe_static_analyzer.gold.extraction import gold_vector
from pe_static_analyzer.gold.io import append_csv, completed_success_ids, ensure_csv, load_metadata, load_prediction_rows
from pe_static_analyzer.gold.results import ERROR_FIELDS, PREDICTION_FIELDS, error_row, scored_row, write_metrics


@dataclass
class ParallelGoldLightGbmSettings:
    metadata: Path = Path('outputs/gold/gold_final_metadata.csv')
    model: Path = DEFAULT_MODEL
    predictions_output: Path = Path('outputs/metrics/predictions_gold_without_header_600.csv')
    metrics_output: Path = Path('outputs/metrics/gold_metrics_without_header_600.json')
    errors_output: Path = Path('outputs/metrics/gold_inference_errors_without_header_600.csv')
    limit: int | None = None
    dry_run: bool = False
    resume: bool = False
    workers: int = 4
    sample_timeout: int = 90
    start_method: str = 'spawn'
    postprocess: str = 'model'
    metrics_every: int = 25
    progress_every: int = 25


def load_payload(path: Path) -> dict[str, Any]:
    import lightgbm as lgb

    if path.suffix == ".txt":
        detector = LightGbmDetector(path)
        return {"booster": detector.booster, "selected_indices": detector.selected_indices,
                "postprocess": "sorel_signed_log"}

    # Historical full-corpus experiments stored the booster and indices together.
    import joblib

    payload = joblib.load(path)
    if payload.get("model_type") != "lightgbm":
        raise RuntimeError(f"model payload is not LightGBM: {path}")
    if "model_string" not in payload:
        raise RuntimeError("LightGBM payload missing model_string")
    if "selected_indices" not in payload:
        raise RuntimeError("LightGBM payload missing selected_indices")
    payload["booster"] = lgb.Booster(model_str=payload["model_string"])
    return payload


def extract_feature(row: dict[str, str], extraction_postprocess: str) -> dict[str, Any]:
    label = int(row["label"])
    vector, feature_sha, input_mode = gold_vector(row, extraction_postprocess, require_hash=False)
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


def predict_from_feature(feature: dict[str, Any], payload: dict[str, Any], threshold: float) -> dict[str, Any]:
    selected_indices = np.asarray(payload["selected_indices"], dtype=np.int32)
    vector = feature["vector"]
    x = vector[selected_indices].reshape(1, -1)
    probability = float(payload["booster"].predict(x)[0])
    return scored_row(feature, len(selected_indices), probability, threshold)


def finish_process(process: mp.Process, queue: mp.Queue, row: dict[str, str], payload: dict[str, Any], threshold: float, error: str | None = None) -> dict[str, Any]:
    feature_dim_model = int(len(payload["selected_indices"]))
    if error is not None:
        return error_row(row, feature_dim_model, threshold, error)
    if process.exitcode != 0:
        return error_row(row, feature_dim_model, threshold, f"worker_exitcode={process.exitcode}")
    try:
        message = queue.get(timeout=1)
    except queue_module.Empty:
        return error_row(row, feature_dim_model, threshold, "worker_returned_no_result")
    if not message.get("ok"):
        return error_row(row, feature_dim_model, threshold, str(message.get("error", "unknown_error")))
    return predict_from_feature(message["result"], payload, threshold)


def append_outputs(result: dict[str, Any], predictions_output: Path, errors_output: Path) -> None:
    append_csv(predictions_output, PREDICTION_FIELDS, result)
    if not result.get("error"):
        return
    append_csv(errors_output, ERROR_FIELDS, result)


def run_parallel(rows: list[dict[str, str]], payload: dict[str, Any], threshold: float, settings: ParallelGoldLightGbmSettings, threshold_source: str) -> list[dict[str, Any]]:
    ctx = mp.get_context(settings.start_method)
    started = time.time()
    def save_metrics():
        write_metrics(settings.metrics_output, all_rows, threshold, settings, threshold_source, started,
                      extraction_postprocess=settings.extraction_postprocess, workers=settings.workers)

    done_success = completed_success_ids(settings.predictions_output) if settings.resume else set()
    all_rows: list[dict[str, Any]] = [row for row in load_prediction_rows(settings.predictions_output) if row.get("error", "") == ""] if settings.resume else []
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
        process = ctx.Process(target=extract_worker, args=(row, settings.extraction_postprocess, queue))
        process.start()
        active[process.pid or -cursor] = (process, queue, row, time.monotonic())
        return True

    while len(active) < settings.workers and start_next():
        pass

    while active:
        finished: list[int] = []
        now = time.monotonic()
        for key, (process, queue, row, process_started) in list(active.items()):
            error = None
            if process.is_alive() and now - process_started >= settings.sample_timeout:
                process.terminate()
                process.join(2)
                error = f"sample_timeout_seconds={settings.sample_timeout}"
            if error is None and process.is_alive():
                continue
            if error is None:
                process.join(0)
            result = finish_process(process, queue, row, payload, threshold, error)
            try:
                queue.close()
            except Exception:
                pass
            append_outputs(result, settings.predictions_output, settings.errors_output)
            all_rows.append(result)
            processed_this_run += 1
            finished.append(key)

            if settings.metrics_every > 0 and processed_this_run % settings.metrics_every == 0:
                save_metrics()
            if settings.progress_every > 0 and processed_this_run % settings.progress_every == 0:
                print(json.dumps({"stage": "gold_progress", "processed_this_run": processed_this_run, "total_rows_in_outputs": len(all_rows), "pending_total": len(pending), "scored_rows": sum(1 for item in all_rows if item.get("error", "") == ""), "runtime_seconds": round(time.time() - started, 3)}, ensure_ascii=False), flush=True)

        for key in finished:
            active.pop(key, None)
            start_next()
        if not finished:
            time.sleep(0.05)

    save_metrics()
    return all_rows


class GoldLightGbmFastInferenceJob:
    """Runnable job object for this CLI pipeline."""

    def __init__(self, settings: ParallelGoldLightGbmSettings) -> None:
        self.settings = settings

    def run(self) -> int:
        settings = self.settings
        if settings.workers < 1:
            raise SystemExit("workers must be >= 1")
        if settings.sample_timeout <= 0:
            raise SystemExit("sample_timeout must be > 0 for isolated parallel extraction")
        threshold, threshold_source = 0.5, "fixed"
        rows = load_metadata(settings.metadata, settings.limit)
        payload = load_payload(settings.model)
        settings.feature_dim_model = len(payload["selected_indices"])
        if settings.postprocess == "model":
            settings.extraction_postprocess = "sorel_signed_log" if payload.get("postprocess") == "sorel_signed_log" else "raw"
        else:
            settings.extraction_postprocess = settings.postprocess

        if settings.dry_run:
            print(json.dumps({
                "metadata": str(settings.metadata),
                "rows_selected": len(rows),
                "model": str(settings.model),
                "feature_dim_original": FEATURE_DIM,
                "feature_dim_model": settings.feature_dim_model,
                "threshold": threshold,
                "threshold_source": threshold_source,
                "workers": settings.workers,
                "sample_timeout": settings.sample_timeout,
                "start_method": settings.start_method,
                "postprocess": settings.postprocess,
                "extraction_postprocess": settings.extraction_postprocess,
                "outputs": {
                    "predictions": str(settings.predictions_output),
                    "metrics": str(settings.metrics_output),
                    "errors": str(settings.errors_output),
                },
                "dry_run": True,
            }, indent=2, ensure_ascii=False))
            return 0

        ensure_csv(settings.predictions_output, PREDICTION_FIELDS, settings.resume)
        ensure_csv(settings.errors_output, ERROR_FIELDS, settings.resume)
        result_rows = run_parallel(rows, payload, threshold, settings, threshold_source)
        print(json.dumps({"processed_rows": len(result_rows), "scored_rows": sum(1 for row in result_rows if row.get("error", "") == ""), "predictions": str(settings.predictions_output), "metrics": str(settings.metrics_output), "errors": str(settings.errors_output)}, indent=2, ensure_ascii=False))
        return 0


def main() -> int:
    return GoldLightGbmFastInferenceJob(ParallelGoldLightGbmSettings()).run()


if __name__ == "__main__":
    raise SystemExit(main())
