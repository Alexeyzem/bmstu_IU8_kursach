#!/usr/bin/env python3
"""Train a separate LightGBM PE classifier from SOREL LMDB features.

This is intentionally separate from the SGD baseline scripts/artifacts. It reads
SOREL train/validation/test manifests, streams train rows from LMDB in chunks,
continues LightGBM training chunk-by-chunk for full-dataset passes, saves the
best validation epoch, and writes separate metrics/threshold artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
import zlib
from typing import Any, Iterator

import joblib
import lmdb
import msgpack
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - runtime dependency message.
    raise SystemExit("LightGBM is missing. Install with: .venv/bin/pip install lightgbm") from exc

FEATURE_DIM = 2381


def sorel_signed_log_postprocess(x: np.ndarray) -> np.ndarray:
    out = x.astype(np.float32, copy=True)
    negative = out < 0
    positive = out > 0
    out[negative] = -np.log1p(-out[negative])
    out[positive] = np.log1p(out[positive])
    return out


def unpack_feature_vector(blob: bytes) -> np.ndarray:
    unpacked = msgpack.unpackb(zlib.decompress(blob), raw=False, strict_map_key=False)
    vector = unpacked.get(0) if isinstance(unpacked, dict) else unpacked
    if vector is None and isinstance(unpacked, dict):
        vector = unpacked.get("0")
    arr = np.asarray(vector, dtype=np.float32)
    if arr.shape != (FEATURE_DIM,):
        raise ValueError(f"feature vector shape {arr.shape}; expected ({FEATURE_DIM},)")
    return sorel_signed_log_postprocess(arr)


def load_selected_indices(path: Path | None) -> list[int]:
    if path is None:
        return list(range(FEATURE_DIM))
    data = json.loads(path.read_text(encoding="utf-8"))
    return [int(item["index"]) for item in data["selected_features"]]


def iter_manifest(path: Path, limit: int | None = None) -> Iterator[tuple[str, int]]:
    count = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            yield row["sha256"], int(row["label"])
            count += 1
            if limit is not None and count >= limit:
                return


def batches(manifest: Path, env: lmdb.Environment, indices: np.ndarray, batch_size: int, limit: int | None = None) -> Iterator[tuple[np.ndarray, np.ndarray, dict[str, int]]]:
    x_batch: list[np.ndarray] = []
    y_batch: list[int] = []
    skipped_missing = 0
    skipped_bad = 0
    total_seen = 0
    with env.begin(buffers=True) as txn:
        for sha256, label in iter_manifest(manifest, limit):
            total_seen += 1
            blob = txn.get(sha256.encode("ascii"))
            if blob is None:
                skipped_missing += 1
                continue
            try:
                vector = unpack_feature_vector(bytes(blob))[indices]
            except Exception:
                skipped_bad += 1
                continue
            x_batch.append(vector)
            y_batch.append(label)
            if len(x_batch) >= batch_size:
                yield np.vstack(x_batch).astype(np.float32), np.asarray(y_batch, dtype=np.int8), {"seen": total_seen, "skipped_missing": skipped_missing, "skipped_bad": skipped_bad}
                x_batch.clear()
                y_batch.clear()
        if x_batch:
            yield np.vstack(x_batch).astype(np.float32), np.asarray(y_batch, dtype=np.int8), {"seen": total_seen, "skipped_missing": skipped_missing, "skipped_bad": skipped_bad}


def collect_eval(manifest: Path, env: lmdb.Environment, indices: np.ndarray, batch_size: int, limit: int | None) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    stats = {"seen": 0, "skipped_missing": 0, "skipped_bad": 0}
    for x, y, batch_stats in batches(manifest, env, indices, batch_size, limit):
        xs.append(x)
        ys.append(y)
        stats = batch_stats
    if not xs:
        return np.empty((0, len(indices)), dtype=np.float32), np.empty((0,), dtype=np.int8), stats
    return np.vstack(xs).astype(np.float32), np.concatenate(ys), stats


def count_classes(manifest: Path, env: lmdb.Environment, indices: np.ndarray, batch_size: int, limit: int | None) -> tuple[dict[int, int], dict[str, int]]:
    del env, indices, batch_size
    counts = {0: 0, 1: 0}
    stats = {"seen": 0, "skipped_missing": 0, "skipped_bad": 0}
    for _sha256, label in iter_manifest(manifest, limit):
        stats["seen"] += 1
        counts[int(label)] += 1
    return counts, stats


def predict_manifest(manifest: Path, env: lmdb.Environment, indices: np.ndarray, batch_size: int, limit: int | None, booster: lgb.Booster) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    stats = {"seen": 0, "skipped_missing": 0, "skipped_bad": 0}
    for x, y, batch_stats in batches(manifest, env, indices, batch_size, limit):
        probabilities.append(booster_predict(booster, x))
        labels.append(y.copy())
        stats = batch_stats
    if not labels:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.int8), stats
    return np.concatenate(probabilities), np.concatenate(labels), stats


def metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, object]:
    y_pred = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_tpr": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if len(set(y_true.tolist())) == 2 else None,
        "pr_auc": float(average_precision_score(y_true, probabilities)) if len(set(y_true.tolist())) == 2 else None,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else None,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray, max_curve_rows: int = 5000) -> tuple[float, dict[str, object], list[dict[str, object]]]:
    y_true = np.asarray(y_true, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if y_true.size == 0:
        raise ValueError("threshold selection sample is empty")

    order = np.argsort(-probabilities, kind="mergesort")
    sorted_probabilities = probabilities[order]
    sorted_labels = y_true[order]
    group_ends = np.r_[np.flatnonzero(sorted_probabilities[1:] != sorted_probabilities[:-1]), len(sorted_probabilities) - 1]
    thresholds = sorted_probabilities[group_ends]
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    tp = np.cumsum(sorted_labels == 1)[group_ends].astype(np.float64)
    fp = np.cumsum(sorted_labels == 0)[group_ends].astype(np.float64)
    fn = positives - tp
    tn = negatives - fp
    accuracy = (tp + tn) / len(y_true)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, positives, out=np.zeros_like(tp), where=positives != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
    fpr = np.divide(fp, fp + tn, out=np.zeros_like(tp), where=(fp + tn) != 0)
    fnr = np.divide(fn, fn + tp, out=np.zeros_like(tp), where=(fn + tp) != 0)
    roc_auc = float(roc_auc_score(y_true, probabilities)) if len(set(y_true.tolist())) == 2 else None
    pr_auc = float(average_precision_score(y_true, probabilities)) if len(set(y_true.tolist())) == 2 else None

    def row_at(index: int) -> dict[str, object]:
        return {
            "threshold": float(thresholds[index]),
            "accuracy": float(accuracy[index]),
            "precision": float(precision[index]),
            "recall_tpr": float(recall[index]),
            "f1": float(f1[index]),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "fpr": float(fpr[index]),
            "fnr": float(fnr[index]),
            "confusion_matrix": {"tn": int(tn[index]), "fp": int(fp[index]), "fn": int(fn[index]), "tp": int(tp[index])},
        }

    best_index = int(np.lexsort((-fpr, recall, f1))[-1])
    best_f1 = row_at(best_index)
    selected_indices = [best_index]
    operating_points: dict[str, dict[str, object] | None] = {}
    for name, max_fpr in (("fpr_10_percent", 0.10), ("fpr_5_percent", 0.05), ("fpr_1_percent", 0.01), ("fpr_0_1_percent", 0.001)):
        feasible = np.flatnonzero(fpr <= max_fpr)
        if feasible.size:
            local_index = int(feasible[np.lexsort((f1[feasible], precision[feasible], recall[feasible]))[-1]])
            operating_points[name] = row_at(local_index)
            selected_indices.append(local_index)
        else:
            operating_points[name] = None
    selected_name = "fpr_5_percent" if operating_points["fpr_5_percent"] is not None else "best_f1"
    selected_metrics = operating_points["fpr_5_percent"] if selected_name == "fpr_5_percent" else best_f1
    assert selected_metrics is not None

    if max_curve_rows <= 0:
        curve_indices = np.asarray(sorted(set(selected_indices)), dtype=np.int64)
    elif len(thresholds) <= max_curve_rows:
        curve_indices = np.arange(len(thresholds), dtype=np.int64)
    else:
        sampled = np.linspace(0, len(thresholds) - 1, max_curve_rows, dtype=np.int64)
        curve_indices = np.unique(np.r_[sampled, np.asarray(selected_indices, dtype=np.int64)])
    rows = [row_at(int(index)) for index in curve_indices]
    return float(selected_metrics["threshold"]), {
        "selected_policy": selected_name,
        "selected_metrics": selected_metrics,
        "best_f1_metrics": best_f1,
        "operating_points": operating_points,
        "candidate_count": int(len(thresholds)),
        "threshold_curve_rows": int(len(rows)),
        "tradeoff": "Lower FPR thresholds reduce false alarms on benign/safe PE files but increase missed malicious/dangerous PE files.",
    }, rows


def write_threshold_curve(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["threshold", "accuracy", "precision", "recall_tpr", "f1", "roc_auc", "pr_auc", "fpr", "fnr", "tn", "fp", "fn", "tp"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in rows:
            confusion = item["confusion_matrix"]
            assert isinstance(confusion, dict)
            writer.writerow({**{key: item.get(key) for key in fieldnames[:9]}, **confusion})


def write_epoch_history(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "best", "threshold", "accuracy", "precision", "recall_tpr", "f1", "roc_auc", "pr_auc", "fpr", "fnr", "tn", "fp", "fn", "tp", "trained_rows", "train_chunks", "total_trees", "runtime_seconds"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            confusion = row.get("confusion_matrix", {})
            assert isinstance(confusion, dict)
            writer.writerow({**{key: row.get(key) for key in fieldnames[:11]}, **confusion, "trained_rows": row.get("trained_rows"), "train_chunks": row.get("train_chunks"), "total_trees": row.get("total_trees"), "runtime_seconds": row.get("runtime_seconds")})


def booster_predict(booster: lgb.Booster, x: np.ndarray) -> np.ndarray:
    return np.asarray(booster.predict(x), dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=Path("outputs/matrices/sorel_train_manifest.csv"))
    parser.add_argument("--validation-manifest", type=Path, default=Path("outputs/matrices/sorel_validation_manifest.csv"))
    parser.add_argument("--test-manifest", type=Path, default=Path("outputs/matrices/sorel_test_manifest.csv"))
    parser.add_argument("--lmdb-dir", type=Path, default=Path("datasets/sorel20m/ember_features"))
    parser.add_argument("--selected-features", type=Path, default=Path("outputs/features/selected_features.json"))
    parser.add_argument("--use-all-features", action="store_true", help="Use all 2381 features instead of selected 2192 features.")
    parser.add_argument("--output-model", type=Path, default=Path("outputs/models/lightgbm_pe_full.pkl"))
    parser.add_argument("--output-booster", type=Path, default=Path("outputs/models/lightgbm_pe_full.txt"))
    parser.add_argument("--metrics-output", type=Path, default=Path("outputs/metrics/lightgbm_validation_metrics.json"))
    parser.add_argument("--test-metrics-output", type=Path, default=Path("outputs/metrics/lightgbm_test_metrics.json"))
    parser.add_argument("--threshold-output", type=Path, default=Path("outputs/metrics/lightgbm_threshold_selection.json"))
    parser.add_argument("--threshold-curve-output", type=Path, default=Path("outputs/metrics/lightgbm_threshold_curve.csv"))
    parser.add_argument("--epoch-metrics-output", type=Path, default=Path("outputs/metrics/lightgbm_training_history.csv"))
    parser.add_argument("--threshold-curve-max-rows", type=int, default=5000, help="Maximum rows written to the threshold curve CSV; selection still evaluates all unique probabilities.")
    parser.add_argument("--progress-every", type=int, default=10, help="Print a JSON progress line every N train chunks; 0 disables chunk progress.")
    parser.add_argument("--feature-list-output", type=Path, default=Path("outputs/models/lightgbm_feature_list.json"))
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=20000)
    parser.add_argument("--test-limit", type=int, default=20000)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--rounds-per-chunk", type=int, default=1, help="Boosting rounds appended for each train chunk.")
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-data-in-leaf", type=int, default=100)
    parser.add_argument("--feature-fraction", type=float, default=0.9)
    parser.add_argument("--bagging-fraction", type=float, default=0.9)
    parser.add_argument("--bagging-freq", type=int, default=1)
    parser.add_argument("--lambda-l2", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=20261115)
    return parser.parse_args()


def normalize_limit(value: int | None) -> int | None:
    if value is not None and value < 0:
        return None
    return value


def main() -> int:
    args = parse_args()
    args.train_limit = normalize_limit(args.train_limit)
    args.validation_limit = normalize_limit(args.validation_limit)
    args.test_limit = normalize_limit(args.test_limit)
    started = time.time()
    selected_indices = list(range(FEATURE_DIM)) if args.use_all_features else load_selected_indices(args.selected_features)
    indices = np.asarray(selected_indices, dtype=np.int32)
    env = lmdb.open(str(args.lmdb_dir), readonly=True, lock=False, readahead=False, max_readers=1)

    print(json.dumps({"stage": "validation", "mode": "streamed_prediction", "limit": args.validation_limit}, ensure_ascii=False), flush=True)

    print(json.dumps({"stage": "count_train_classes", "limit": args.train_limit}, ensure_ascii=False), flush=True)
    class_counts, count_stats = count_classes(args.train_manifest, env, indices, args.chunk_size, args.train_limit)
    total_count = class_counts[0] + class_counts[1]
    if class_counts[0] == 0 or class_counts[1] == 0:
        raise SystemExit(f"training sample must contain both classes, got {class_counts}")
    class_weights = {label: total_count / (2 * count) for label, count in class_counts.items()}

    params: dict[str, Any] = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "boosting_type": "gbdt",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "lambda_l2": args.lambda_l2,
        "seed": args.random_state,
        "feature_pre_filter": False,
        "force_col_wise": True,
        "verbosity": -1,
        "num_threads": 0,
    }

    booster: lgb.Booster | None = None
    best_score: tuple[float, float, float] | None = None
    best_epoch = 0
    best_model_string = ""
    epoch_rows: list[dict[str, object]] = []
    train_rows = 0
    train_chunks = 0
    last_train_stats: dict[str, int] | None = None

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        for x_train, y_train, last_train_stats in batches(args.train_manifest, env, indices, args.chunk_size, args.train_limit):
            sample_weight = np.asarray([class_weights[int(label)] for label in y_train], dtype=np.float32)
            dataset = lgb.Dataset(x_train, label=y_train, weight=sample_weight, free_raw_data=True)
            booster = lgb.train(params, dataset, num_boost_round=args.rounds_per_chunk, init_model=booster, keep_training_booster=True)
            train_rows += len(y_train)
            train_chunks += 1
            if args.progress_every > 0 and train_chunks % args.progress_every == 0:
                print(json.dumps({"stage": "train_progress", "epoch": epoch, "train_chunks": train_chunks, "trained_rows": train_rows, "seen": last_train_stats.get("seen") if last_train_stats else None, "runtime_seconds": round(time.time() - started, 3)}, ensure_ascii=False), flush=True)
        assert booster is not None
        val_probabilities, y_val, val_stats = predict_manifest(args.validation_manifest, env, indices, args.eval_batch_size, args.validation_limit, booster)
        if len(y_val) == 0:
            raise SystemExit("validation sample is empty")
        selected_threshold, threshold_selection, _threshold_rows = select_threshold(y_val, val_probabilities, args.threshold_curve_max_rows)
        selected_metrics = threshold_selection["selected_metrics"]
        assert isinstance(selected_metrics, dict)
        epoch_payload = metrics(y_val, val_probabilities, selected_threshold)
        epoch_payload.update(
            {
                "epoch": epoch,
                "threshold": selected_threshold,
                "trained_rows": train_rows,
                "train_chunks": train_chunks,
                "total_trees": booster.num_trees(),
                "runtime_seconds": round(time.time() - started, 3),
                "epoch_runtime_seconds": round(time.time() - epoch_started, 3),
            }
        )
        key = (float(epoch_payload["f1"]), float(epoch_payload["recall_tpr"]), -float(epoch_payload["fpr"] or 0.0))
        is_best = best_score is None or key > best_score
        epoch_payload["best"] = is_best
        epoch_rows.append(epoch_payload)
        print(json.dumps({"stage": "epoch_done", "metrics": epoch_payload}, ensure_ascii=False), flush=True)
        if is_best:
            best_score = key
            best_epoch = epoch
            best_model_string = booster.model_to_string()
            args.output_booster.parent.mkdir(parents=True, exist_ok=True)
            booster.save_model(args.output_booster)

    if not best_model_string:
        raise SystemExit("no best LightGBM model captured")
    best_booster = lgb.Booster(model_str=best_model_string)
    val_probabilities, y_val, val_stats = predict_manifest(args.validation_manifest, env, indices, args.eval_batch_size, args.validation_limit, best_booster)
    selected_threshold, threshold_selection, threshold_rows = select_threshold(y_val, val_probabilities, args.threshold_curve_max_rows)
    metric_payload = metrics(y_val, val_probabilities, selected_threshold)
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metric_payload.update(
        {
            "created_at": created_at,
            "model": "LightGBM streamed chunk-wise GBDT",
            "best_epoch": best_epoch,
            "epochs_requested": args.epochs,
            "rounds_per_chunk": args.rounds_per_chunk,
            "total_trees": best_booster.num_trees(),
            "train_manifest": str(args.train_manifest),
            "validation_manifest": str(args.validation_manifest),
            "test_manifest": str(args.test_manifest),
            "train_limit": args.train_limit,
            "validation_limit": args.validation_limit,
            "test_limit": args.test_limit,
            "trained_rows": train_rows,
            "train_chunks": train_chunks,
            "train_class_counts": class_counts,
            "train_count_stats": count_stats,
            "train_stats_last_batch": last_train_stats,
            "validation_rows": int(len(y_val)),
            "validation_stats": val_stats,
            "feature_dim_original": FEATURE_DIM,
            "feature_dim_model": len(selected_indices),
            "selected_features": "all" if args.use_all_features else str(args.selected_features),
            "params": params,
            "runtime_seconds": round(time.time() - started, 3),
        }
    )
    threshold_payload = {
        "created_at": created_at,
        "selection_split": "validation",
        "selection_objective": "choose max-recall operating point at FPR <= 5% when available; also record FPR <= 10%, 1%, 0.1%, and best F1",
        "selected_threshold": selected_threshold,
        **threshold_selection,
        "default_0_5_metrics": metrics(y_val, val_probabilities, 0.5),
        "candidate_count": threshold_selection.get("candidate_count", len(threshold_rows)),
        "threshold_curve_rows": len(threshold_rows),
        "best_epoch": best_epoch,
        "validation_manifest": str(args.validation_manifest),
        "validation_limit": args.validation_limit,
        "validation_rows": int(len(y_val)),
        "validation_stats": val_stats,
    }

    test_payload = None
    if args.test_manifest.exists() and args.test_limit != 0:
        test_probabilities, y_test, test_stats = predict_manifest(args.test_manifest, env, indices, args.eval_batch_size, args.test_limit, best_booster)
        if len(y_test) > 0:
            test_payload = metrics(y_test, test_probabilities, selected_threshold)
            test_payload.update(
                {
                    "created_at": created_at,
                    "model": metric_payload["model"],
                    "best_epoch": best_epoch,
                    "total_trees": best_booster.num_trees(),
                    "train_manifest": str(args.train_manifest),
                    "validation_manifest": str(args.validation_manifest),
                    "test_manifest": str(args.test_manifest),
                    "train_limit": args.train_limit,
                    "validation_limit": args.validation_limit,
                    "test_limit": args.test_limit,
                    "trained_rows": train_rows,
                    "test_rows": int(len(y_test)),
                    "test_stats": test_stats,
                    "feature_dim_original": FEATURE_DIM,
                    "feature_dim_model": len(selected_indices),
                    "selected_features": metric_payload["selected_features"],
                }
            )

    payload = {
        "model_type": "lightgbm",
        "model_string": best_model_string,
        "selected_indices": selected_indices,
        "postprocess": "sorel_signed_log",
        "feature_dim_original": FEATURE_DIM,
        "feature_dim_model": len(selected_indices),
        "threshold": selected_threshold,
        "best_epoch": best_epoch,
        "params": params,
    }
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.threshold_output.parent.mkdir(parents=True, exist_ok=True)
    args.threshold_curve_output.parent.mkdir(parents=True, exist_ok=True)
    args.epoch_metrics_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.output_model)
    args.output_booster.write_text(best_model_string, encoding="utf-8")
    args.feature_list_output.write_text(json.dumps({"selected_indices": selected_indices, "feature_dim_model": len(selected_indices)}, indent=2), encoding="utf-8")
    args.metrics_output.write_text(json.dumps(metric_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.threshold_output.write_text(json.dumps(threshold_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_threshold_curve(args.threshold_curve_output, threshold_rows)
    write_epoch_history(args.epoch_metrics_output, epoch_rows)
    if test_payload is not None:
        args.test_metrics_output.write_text(json.dumps(test_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"validation": metric_payload, "threshold_selection": threshold_payload, "test": test_payload}, indent=2, ensure_ascii=False))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
