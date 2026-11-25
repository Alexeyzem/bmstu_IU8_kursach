#!/usr/bin/env python3
"""Train an incremental SOREL baseline classifier from LMDB EMBER features.

The persisted model is inference-compatible with raw PE files: training uses the
same SOREL signed-log postprocess and selected feature indices that
`pe_static_analyzer.prediction.single_file` applies after extracting EMBER-v2 features from bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
import time

import joblib
import lmdb
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from pe_static_analyzer.core.metrics import binary_classification_metrics as metrics, fixed_threshold_report
from pe_static_analyzer.sorel.data import FEATURE_DIM, collect_eval, feature_batches as batches, load_selected_indices


@dataclass
class SgdTrainingSettings:
    train_manifest: Path = Path('outputs/matrices/sorel_train_manifest.csv')
    validation_manifest: Path = Path('outputs/matrices/sorel_validation_manifest.csv')
    test_manifest: Path = Path('outputs/matrices/sorel_test_manifest.csv')
    lmdb_dir: Path = Path('datasets/sorel20m/ember_features')
    selected_features: Path = Path('outputs/features/selected_features.json')
    output_model: Path = Path('outputs/models/baseline_sgd.pkl')
    metrics_output: Path = Path('outputs/metrics/sorel_validation_metrics.json')
    test_metrics_output: Path = Path('outputs/metrics/sorel_test_metrics.json')
    threshold_output: Path = Path('outputs/metrics/threshold_selection.json')
    threshold_curve_output: Path = Path('outputs/metrics/threshold_curve.csv')
    epoch_metrics_output: Path = Path('outputs/metrics/training_history.csv')
    feature_list_output: Path = Path('outputs/models/feature_list.json')
    input_model: Path | None = None
    eval_only: bool = False
    train_limit: int | None = None
    validation_limit: int = 20000
    test_limit: int = 20000
    batch_size: int = 2048
    epochs: int = 5
    random_state: int = 20261108


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


class SgdTrainingJob:
    """Runnable job object for this CLI pipeline."""

    def __init__(self, settings: SgdTrainingSettings) -> None:
        self.settings = settings

    def run(self) -> int:
        settings = self.settings
        started = time.time()

        env = lmdb.open(str(settings.lmdb_dir), readonly=True, lock=False, readahead=False, max_readers=1)
        scale_stats = None
        train_stats = None
        train_rows = 0
        train_batches = 0
        epoch_history: list[dict[str, object]] = []

        if settings.eval_only:
            model_path = settings.input_model or settings.output_model
            payload = joblib.load(model_path)
            selected_indices = [int(index) for index in payload["selected_indices"]]
            scaler = payload["scaler"]
            clf = payload["model"]
            model_description = f"existing model from {model_path}"
        else:
            selected_indices = load_selected_indices(settings.selected_features)
            indices_for_training = np.asarray(selected_indices, dtype=np.int32)
            scaler = StandardScaler()
            for x, _y, scale_stats in batches(settings.train_manifest, env, indices_for_training, settings.batch_size, settings.train_limit):
                scaler.partial_fit(x)

            class_counts = {0: 0, 1: 0}
            for _x, y, _stats in batches(settings.train_manifest, env, indices_for_training, settings.batch_size, settings.train_limit):
                class_counts[0] += int(np.sum(y == 0))
                class_counts[1] += int(np.sum(y == 1))
            total_count = class_counts[0] + class_counts[1]
            if class_counts[0] == 0 or class_counts[1] == 0:
                raise SystemExit(f"training sample must contain both classes, got {class_counts}")
            class_weights = {label: total_count / (2 * count) for label, count in class_counts.items()}

            clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-5, random_state=settings.random_state)
            x_epoch_val, y_epoch_val, epoch_val_stats = collect_eval(settings.validation_manifest, env, indices_for_training, settings.batch_size, settings.validation_limit)
            if len(y_epoch_val) == 0:
                raise SystemExit("validation sample is empty")
            for epoch in range(1, settings.epochs + 1):
                for x, y, train_stats in batches(settings.train_manifest, env, indices_for_training, settings.batch_size, settings.train_limit):
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
        x_val, y_val, val_stats = collect_eval(settings.validation_manifest, env, indices, settings.batch_size, settings.validation_limit)
        if len(y_val) == 0:
            raise SystemExit("validation sample is empty")
        val_probabilities = clf.predict_proba(scaler.transform(x_val))[:, 1]
        selected_threshold, threshold_selection, threshold_rows = fixed_threshold_report(y_val, val_probabilities)
        selected_metrics = threshold_selection["selected_metrics"]
        assert isinstance(selected_metrics, dict)
        metric_payload = metrics(y_val, val_probabilities, selected_threshold)
        metric_payload.update(
            {
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model": model_description,
                "train_manifest": str(settings.train_manifest),
                "validation_manifest": str(settings.validation_manifest),
                "test_manifest": str(settings.test_manifest),
                "train_limit": settings.train_limit,
                "validation_limit": settings.validation_limit,
                "test_limit": settings.test_limit,
                "epochs": None if settings.eval_only else settings.epochs,
                "eval_only": settings.eval_only,
                "trained_rows": train_rows,
                "train_batches": train_batches,
                "train_stats_last_batch": train_stats,
                "scale_stats_last_batch": scale_stats,
                "validation_rows": int(len(y_val)),
                "validation_stats": val_stats,
                "feature_dim_original": FEATURE_DIM,
                "feature_dim_model": len(selected_indices),
                "selected_features": str(settings.selected_features),
                "runtime_seconds": round(time.time() - started, 3),
            }
        )

        threshold_payload = {
            "created_at": metric_payload["created_at"],
            "selection_split": None,
            "selection_objective": "fixed probability >= 0.5",
            "selected_threshold": selected_threshold,
            **threshold_selection,
            "default_0_5_metrics": metrics(y_val, val_probabilities, 0.5),
            "candidate_count": len(threshold_rows),
            "validation_manifest": str(settings.validation_manifest),
            "validation_limit": settings.validation_limit,
            "validation_rows": int(len(y_val)),
            "validation_stats": val_stats,
        }

        test_payload = None
        if settings.test_manifest.exists() and settings.test_limit != 0:
            x_test, y_test, test_stats = collect_eval(settings.test_manifest, env, indices, settings.batch_size, settings.test_limit)
            if len(y_test) > 0:
                test_probabilities = clf.predict_proba(scaler.transform(x_test))[:, 1]
                test_payload = metrics(y_test, test_probabilities, selected_threshold)
                test_payload.update(
                    {
                        "created_at": metric_payload["created_at"],
                        "model": metric_payload["model"],
                        "train_manifest": str(settings.train_manifest),
                        "validation_manifest": str(settings.validation_manifest),
                        "test_manifest": str(settings.test_manifest),
                        "train_limit": settings.train_limit,
                        "validation_limit": settings.validation_limit,
                        "test_limit": settings.test_limit,
                        "epochs": None if settings.eval_only else settings.epochs,
                        "eval_only": settings.eval_only,
                        "trained_rows": train_rows,
                        "test_rows": int(len(y_test)),
                        "test_stats": test_stats,
                        "feature_dim_original": FEATURE_DIM,
                        "feature_dim_model": len(selected_indices),
                        "selected_features": str(settings.selected_features),
                    }
                )

        settings.output_model.parent.mkdir(parents=True, exist_ok=True)
        settings.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        settings.threshold_output.parent.mkdir(parents=True, exist_ok=True)
        settings.threshold_curve_output.parent.mkdir(parents=True, exist_ok=True)
        settings.epoch_metrics_output.parent.mkdir(parents=True, exist_ok=True)
        settings.test_metrics_output.parent.mkdir(parents=True, exist_ok=True)
        if not settings.eval_only:
            model_payload = {
                "model": clf,
                "scaler": scaler,
                "selected_indices": selected_indices,
                "postprocess": "sorel_signed_log",
                "feature_dim_original": FEATURE_DIM,
                "feature_dim_model": len(selected_indices),
                "threshold": selected_threshold,
            }
            joblib.dump(model_payload, settings.output_model)
            settings.feature_list_output.write_text(json.dumps({"selected_indices": selected_indices, "feature_dim_model": len(selected_indices)}, indent=2), encoding="utf-8")
            write_epoch_history(settings.epoch_metrics_output, epoch_history)
        settings.metrics_output.write_text(json.dumps(metric_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        settings.threshold_output.write_text(json.dumps(threshold_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        write_threshold_curve(settings.threshold_curve_output, threshold_rows)
        if test_payload is not None:
            settings.test_metrics_output.write_text(json.dumps(test_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"validation": metric_payload, "threshold_selection": threshold_payload, "test": test_payload}, indent=2, ensure_ascii=False))
        env.close()
        return 0


def main() -> int:
    return SgdTrainingJob(SgdTrainingSettings()).run()


if __name__ == "__main__":
    raise SystemExit(main())
