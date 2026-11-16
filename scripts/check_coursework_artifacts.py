#!/usr/bin/env python3
"""Check that expected coursework artifacts exist and are non-empty."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    "datasets/sorel20m/meta.db",
    "datasets/sorel20m/ember_features/data.mdb",
    "outputs/gold/gold_final_metadata.csv",
    "outputs/dedup/dedup_summary.json",
    "outputs/features/selected_features.json",
    "outputs/matrices/sorel_train_manifest.csv",
    "outputs/matrices/sorel_validation_manifest.csv",
    "outputs/matrices/sorel_test_manifest.csv",
    "outputs/models/baseline_sgd.pkl",
    "outputs/models/feature_list.json",
    "outputs/metrics/sorel_validation_metrics.json",
    "outputs/metrics/sorel_test_metrics.json",
    "outputs/metrics/threshold_selection.json",
    "outputs/metrics/threshold_curve.csv",
    "main.py",
    "scripts/predict_pe_file.py",
    "scripts/predict_gold_raw.py",
    "scripts/predict_gold_batch.py",
    "scripts/generate_coursework_figures.py",
    "scripts/generate_coursework_report.py",
    "requirements.txt",
]

GOLD_PATHS = [
    "outputs/metrics/predictions_gold.csv",
    "outputs/metrics/gold_metrics.json",
    "outputs/metrics/gold_inference_errors.csv",
]

FIGURE_PATHS = [
    "outputs/figures/01_datasets/sorel_total_composition.svg",
    "outputs/figures/01_datasets/sorel_split_composition.svg",
    "outputs/figures/01_datasets/gold_class_composition.svg",
    "outputs/figures/01_datasets/gold_top_sources.svg",
    "outputs/figures/02_deduplication/gold_dedup_funnel.svg",
    "outputs/figures/03_features/feature_selection.svg",
    "outputs/figures/03_features/feature_removal_reasons.svg",
    "outputs/figures/04_model_quality/sorel_validation_confusion_matrix.svg",
    "outputs/figures/04_model_quality/sorel_test_confusion_matrix.svg",
    "outputs/figures/04_model_quality/threshold_tradeoff.svg",
    "outputs/figures/04_model_quality/gold_confusion_matrix.svg",
    "outputs/figures/04_model_quality/epoch_accuracy_curve.svg",
    "outputs/figures/04_model_quality/model_comparison_validation.svg",
    "outputs/figures/04_model_quality/model_comparison_test.svg",
    "outputs/figures/04_model_quality/model_comparison_gold.svg",
    "outputs/model_comparison_1epoch_vs_5epoch.md",
    "outputs/coursework_report.md",
]


def status(path_text: str) -> dict[str, object]:
    path = ROOT / path_text
    return {
        "path": path_text,
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "ok": path.exists() and (path.is_dir() or path.stat().st_size > 0),
    }


def main() -> int:
    required = [status(path) for path in REQUIRED_PATHS]
    gold = [status(path) for path in GOLD_PATHS]
    figures = [status(path) for path in FIGURE_PATHS]
    payload = {
        "required": required,
        "gold": gold,
        "figures": figures,
        "ok_required": all(item["ok"] for item in required),
        "ok_gold": all(item["ok"] for item in gold),
        "ok_figures": all(item["ok"] for item in figures),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["ok_required"] and payload["ok_gold"] and payload["ok_figures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
