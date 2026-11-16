#!/usr/bin/env python3
"""Train an incremental SOREL baseline classifier from LMDB EMBER features.

The persisted model is inference-compatible with raw PE files: training uses the
same SOREL signed-log postprocess and the same selected feature indices that
`predict_pe_file.py` applies after extracting EMBER-v2 features from bytes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
import zlib

import joblib
import lmdb
import msgpack
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

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


def iter_manifest(path: Path, limit: int | None = None):
    count = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            yield row["sha256"], int(row["label"])
            count += 1
            if limit is not None and count >= limit:
                return


def batches(manifest: Path, env: lmdb.Environment, indices: np.ndarray, batch_size: int, limit: int | None = None):
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


def collect_eval(manifest: Path, env: lmdb.Environment, indices: np.ndarray, batch_size: int, limit: int | None):
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    stats = {"seen": 0, "skipped_missing": 0, "skipped_bad": 0}
    for x, y, batch_stats in batches(manifest, env, indices, batch_size, limit):
        xs.append(x)
        ys.append(y)
        stats = batch_stats
    if not xs:
        return np.empty((0, len(indices)), dtype=np.float32), np.empty((0,), dtype=np.int8), stats
    return np.vstack(xs), np.concatenate(ys), stats


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


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, object], list[dict[str, object]]]:
    candidates = np.unique(np.concatenate([np.linspace(0.001, 0.999, 999), probabilities]))
    rows: list[dict[str, object]] = []
    for threshold in candidates:
        rows.append(metrics(y_true, probabilities, float(threshold)))

    best_f1 = max(rows, key=lambda item: (float(item["f1"]), float(item["recall_tpr"]), -float(item["fpr"] or 0.0)))
    operating_points: dict[str, dict[str, object] | None] = {}
    for name, max_fpr in (("fpr_10_percent", 0.10), ("fpr_5_percent", 0.05), ("fpr_1_percent", 0.01), ("fpr_0_1_percent", 0.001)):
        feasible = [item for item in rows if item["fpr"] is not None and float(item["fpr"]) <= max_fpr]
        operating_points[name] = max(feasible, key=lambda item: (float(item["recall_tpr"]), float(item["precision"]), float(item["f1"]))) if feasible else None

    selected_name = "fpr_5_percent" if operating_points["fpr_5_percent"] is not None else "best_f1"
    selected_metrics = operating_points["fpr_5_percent"] if selected_name == "fpr_5_percent" else best_f1
    assert selected_metrics is not None
    selection = {
        "selected_policy": selected_name,
        "selected_metrics": selected_metrics,
        "best_f1_metrics": best_f1,
        "operating_points": operating_points,
        "tradeoff": "Lower FPR thresholds reduce false alarms on benign/safe PE files but increase missed malicious/dangerous PE files.",
    }
    return float(selected_metrics["threshold"]), selection, rows


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
    fieldnames = ["epoch", "threshold", "accuracy", "precision", "recall_tpr", "f1", "roc_auc", "pr_auc", "fpr", "fnr", "tn", "fp", "fn", "tp", "trained_rows", "train_batches", "runtime_seconds"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            confusion = row.get("confusion_matrix", {})
            assert isinstance(confusion, dict)
            writer.writerow({**{key: row.get(key) for key in fieldnames[:10]}, **confusion, "trained_rows": row.get("trained_rows"), "train_batches": row.get("train_batches"), "runtime_seconds": row.get("runtime_seconds")})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=Path("outputs/matrices/sorel_train_manifest.csv"))
    parser.add_argument("--validation-manifest", type=Path, default=Path("outputs/matrices/sorel_validation_manifest.csv"))
    parser.add_argument("--test-manifest", type=Path, default=Path("outputs/matrices/sorel_test_manifest.csv"))
    parser.add_argument("--lmdb-dir", type=Path, default=Path("datasets/sorel20m/ember_features"))
    parser.add_argument("--selected-features", type=Path, default=Path("outputs/features/selected_features.json"))
    parser.add_argument("--output-model", type=Path, default=Path("outputs/models/baseline_sgd.pkl"))
    parser.add_argument("--metrics-output", type=Path, default=Path("outputs/metrics/sorel_validation_metrics.json"))
    parser.add_argument("--test-metrics-output", type=Path, default=Path("outputs/metrics/sorel_test_metrics.json"))
    parser.add_argument("--threshold-output", type=Path, default=Path("outputs/metrics/threshold_selection.json"))
    parser.add_argument("--threshold-curve-output", type=Path, default=Path("outputs/metrics/threshold_curve.csv"))
    parser.add_argument("--epoch-metrics-output", type=Path, default=Path("outputs/metrics/training_history.csv"))
    parser.add_argument("--feature-list-output", type=Path, default=Path("outputs/models/feature_list.json"))
    parser.add_argument("--input-model", type=Path, default=None, help="Existing model payload for --eval-only; defaults to --output-model.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and evaluate an existing model on validation/test manifests.")
    parser.add_argument("--train-limit", type=int, default=None, help="Limit training rows for smoke/debug; omit for full SOREL train split.")
    parser.add_argument("--validation-limit", type=int, default=20000, help="Validation rows sampled from SOREL validation manifest; keep bounded to avoid materializing millions of rows.")
    parser.add_argument("--test-limit", type=int, default=20000, help="Test rows sampled from SOREL test manifest; keep bounded to avoid materializing millions of rows.")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=20261108)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()

    env = lmdb.open(str(args.lmdb_dir), readonly=True, lock=False, readahead=False, max_readers=1)
    scale_stats = None
    train_stats = None
    train_rows = 0
    train_batches = 0
    epoch_history: list[dict[str, object]] = []

    if args.eval_only:
        model_path = args.input_model or args.output_model
        payload = joblib.load(model_path)
        selected_indices = [int(index) for index in payload["selected_indices"]]
        scaler = payload["scaler"]
        clf = payload["model"]
        model_description = f"existing model from {model_path}"
    else:
        selected_indices = load_selected_indices(args.selected_features)
        indices_for_training = np.asarray(selected_indices, dtype=np.int32)
        scaler = StandardScaler()
        for x, _y, scale_stats in batches(args.train_manifest, env, indices_for_training, args.batch_size, args.train_limit):
            scaler.partial_fit(x)

        class_counts = {0: 0, 1: 0}
        for _x, y, _stats in batches(args.train_manifest, env, indices_for_training, args.batch_size, args.train_limit):
            class_counts[0] += int(np.sum(y == 0))
            class_counts[1] += int(np.sum(y == 1))
        total_count = class_counts[0] + class_counts[1]
        if class_counts[0] == 0 or class_counts[1] == 0:
            raise SystemExit(f"training sample must contain both classes, got {class_counts}")
        class_weights = {label: total_count / (2 * count) for label, count in class_counts.items()}

        clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-5, random_state=args.random_state)
        x_epoch_val, y_epoch_val, epoch_val_stats = collect_eval(args.validation_manifest, env, indices_for_training, args.batch_size, args.validation_limit)
        if len(y_epoch_val) == 0:
            raise SystemExit("validation sample is empty")
        for epoch in range(1, args.epochs + 1):
            for x, y, train_stats in batches(args.train_manifest, env, indices_for_training, args.batch_size, args.train_limit):
                sample_weight = np.asarray([class_weights[int(label)] for label in y], dtype=np.float32)
                clf.partial_fit(scaler.transform(x), y, classes=np.asarray([0, 1], dtype=np.int8), sample_weight=sample_weight)
                train_batches += 1
                train_rows += len(y)
            epoch_probabilities = clf.predict_proba(scaler.transform(x_epoch_val))[:, 1]
            epoch_payload = metrics(y_epoch_val, epoch_probabilities, 0.5)
            epoch_payload.update({"epoch": epoch, "trained_rows": train_rows, "train_batches": train_batches, "runtime_seconds": round(time.time() - started, 3)})
            epoch_history.append(epoch_payload)
            print(json.dumps({"epoch": epoch, "validation": epoch_payload}, ensure_ascii=False), flush=True)
        model_description = "SGDClassifier(log_loss, explicit balanced sample weights)"

    indices = np.asarray(selected_indices, dtype=np.int32)
    x_val, y_val, val_stats = collect_eval(args.validation_manifest, env, indices, args.batch_size, args.validation_limit)
    if len(y_val) == 0:
        raise SystemExit("validation sample is empty")
    val_probabilities = clf.predict_proba(scaler.transform(x_val))[:, 1]
    selected_threshold, threshold_selection, threshold_rows = select_threshold(y_val, val_probabilities)
    selected_metrics = threshold_selection["selected_metrics"]
    assert isinstance(selected_metrics, dict)
    metric_payload = metrics(y_val, val_probabilities, selected_threshold)
    metric_payload.update(
        {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": model_description,
            "train_manifest": str(args.train_manifest),
            "validation_manifest": str(args.validation_manifest),
            "test_manifest": str(args.test_manifest),
            "train_limit": args.train_limit,
            "validation_limit": args.validation_limit,
            "test_limit": args.test_limit,
            "epochs": None if args.eval_only else args.epochs,
            "eval_only": args.eval_only,
            "trained_rows": train_rows,
            "train_batches": train_batches,
            "train_stats_last_batch": train_stats,
            "scale_stats_last_batch": scale_stats,
            "validation_rows": int(len(y_val)),
            "validation_stats": val_stats,
            "feature_dim_original": FEATURE_DIM,
            "feature_dim_model": len(selected_indices),
            "selected_features": str(args.selected_features),
            "runtime_seconds": round(time.time() - started, 3),
        }
    )

    threshold_payload = {
        "created_at": metric_payload["created_at"],
        "selection_split": "validation",
        "selection_objective": "choose max-recall operating point at FPR <= 5% when available; also record FPR <= 10%, 1%, 0.1%, and best F1",
        "selected_threshold": selected_threshold,
        **threshold_selection,
        "default_0_5_metrics": metrics(y_val, val_probabilities, 0.5),
        "candidate_count": len(threshold_rows),
        "validation_manifest": str(args.validation_manifest),
        "validation_limit": args.validation_limit,
        "validation_rows": int(len(y_val)),
        "validation_stats": val_stats,
    }

    test_payload = None
    if args.test_manifest.exists() and args.test_limit != 0:
        x_test, y_test, test_stats = collect_eval(args.test_manifest, env, indices, args.batch_size, args.test_limit)
        if len(y_test) > 0:
            test_probabilities = clf.predict_proba(scaler.transform(x_test))[:, 1]
            test_payload = metrics(y_test, test_probabilities, selected_threshold)
            test_payload.update(
                {
                    "created_at": metric_payload["created_at"],
                    "model": metric_payload["model"],
                    "train_manifest": str(args.train_manifest),
                    "validation_manifest": str(args.validation_manifest),
                    "test_manifest": str(args.test_manifest),
                    "train_limit": args.train_limit,
                    "validation_limit": args.validation_limit,
                    "test_limit": args.test_limit,
                    "epochs": None if args.eval_only else args.epochs,
                    "eval_only": args.eval_only,
                    "trained_rows": train_rows,
                    "test_rows": int(len(y_test)),
                    "test_stats": test_stats,
                    "feature_dim_original": FEATURE_DIM,
                    "feature_dim_model": len(selected_indices),
                    "selected_features": str(args.selected_features),
                }
            )

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.threshold_output.parent.mkdir(parents=True, exist_ok=True)
    args.threshold_curve_output.parent.mkdir(parents=True, exist_ok=True)
    args.epoch_metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.test_metrics_output.parent.mkdir(parents=True, exist_ok=True)
    if not args.eval_only:
        model_payload = {
            "model": clf,
            "scaler": scaler,
            "selected_indices": selected_indices,
            "postprocess": "sorel_signed_log",
            "feature_dim_original": FEATURE_DIM,
            "feature_dim_model": len(selected_indices),
            "threshold": selected_threshold,
        }
        joblib.dump(model_payload, args.output_model)
        args.feature_list_output.write_text(json.dumps({"selected_indices": selected_indices, "feature_dim_model": len(selected_indices)}, indent=2), encoding="utf-8")
        write_epoch_history(args.epoch_metrics_output, epoch_history)
    args.metrics_output.write_text(json.dumps(metric_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.threshold_output.write_text(json.dumps(threshold_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_threshold_curve(args.threshold_curve_output, threshold_rows)
    if test_payload is not None:
        args.test_metrics_output.write_text(json.dumps(test_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"validation": metric_payload, "threshold_selection": threshold_payload, "test": test_payload}, indent=2, ensure_ascii=False))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
