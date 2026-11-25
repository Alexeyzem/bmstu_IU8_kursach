"""Shared binary-classification metrics and threshold selection."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def binary_classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, object]:
    """Return standard binary metrics for PE dangerous/safe predictions."""
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


def scored_rows_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Return metrics for gold inference CSV rows, preserving processed/error counts."""
    ok = [row for row in rows if row.get("error", "") == ""]
    labels = np.asarray([int(row["label"]) for row in ok], dtype=np.int8)
    probabilities = np.asarray([float(row["malware_probability"]) for row in ok], dtype=np.float64)
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
    metric_payload = binary_classification_metrics(labels, probabilities, threshold)
    confusion = metric_payload.get("confusion_matrix", {})
    if isinstance(confusion, dict):
        metric_payload["fpr"] = float(confusion["fp"] / (confusion["fp"] + confusion["tn"])) if (confusion["fp"] + confusion["tn"]) else 0.0
        metric_payload["fnr"] = float(confusion["fn"] / (confusion["fn"] + confusion["tp"])) if (confusion["fn"] + confusion["tp"]) else 0.0
    output.update(metric_payload)
    return output


def fixed_threshold_report(y_true: np.ndarray, probabilities: np.ndarray):
    """Keep the legacy report layout for consumers using the fixed protocol."""
    result = binary_classification_metrics(y_true, probabilities, 0.5)
    return 0.5, {"selected_policy": "fixed", "selected_metrics": result}, [result]
