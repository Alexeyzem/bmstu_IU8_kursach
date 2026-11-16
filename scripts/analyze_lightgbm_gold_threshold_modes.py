#!/usr/bin/env python3
"""Analyze LightGBM gold operating points for strict/balanced/sensitive modes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

MODE_SPECS = [
    {
        "mode": "strict",
        "label_ru": "Строгий",
        "description_ru": "минимум ложных тревог; SOREL-selected threshold",
        "policy": "fixed_threshold",
        "threshold": 0.1223439930122863,
    },
    {
        "mode": "balanced",
        "label_ru": "Сбалансированный",
        "description_ru": "компромисс для внешних PE; максимум recall при FPR <= 20% на gold analysis",
        "policy": "max_recall_at_fpr",
        "max_fpr": 0.20,
    },
    {
        "mode": "sensitive",
        "label_ru": "Чувствительный",
        "description_ru": "ловить подозрительные PE; минимальный FPR при recall >= 95% на gold analysis",
        "policy": "min_fpr_at_recall",
        "min_recall": 0.95,
    },
]

STYLE = """
<style>
text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #1f2937; font-size: 13px; }
.title { font-size: 18px; font-weight: 700; }
.axis { stroke: #6b7280; stroke-width: 1; }
.grid { stroke: #e5e7eb; stroke-width: 1; }
.recall { fill: none; stroke: #059669; stroke-width: 3; }
.fpr { fill: none; stroke: #dc2626; stroke-width: 3; }
.fnr { fill: none; stroke: #d97706; stroke-width: 3; }
.bar-strict { fill: #2563eb; }
.bar-balanced { fill: #7c3aed; }
.bar-sensitive { fill: #059669; }
.dot-strict { fill: #2563eb; }
.dot-balanced { fill: #7c3aed; }
.dot-sensitive { fill: #059669; }
</style>
""".strip()


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def svg(width: int, height: int, body: str) -> str:
    return f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' width='{width}' height='{height}'>\n{STYLE}\n{body}\n</svg>\n"


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    labels: list[int] = []
    probabilities: list[float] = []
    errors: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            error = row.get("error", "")
            if error:
                errors[error] = errors.get(error, 0) + 1
                continue
            labels.append(int(row["label"]))
            probabilities.append(float(row["malware_probability"]))
    if not labels:
        raise RuntimeError(f"no successful prediction rows in {path}")
    return np.asarray(labels, dtype=np.int8), np.asarray(probabilities, dtype=np.float64), errors


def metrics_at(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_tpr": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def all_threshold_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> list[dict[str, Any]]:
    thresholds = np.unique(probabilities)
    rows = [metrics_at(y_true, probabilities, float(threshold)) for threshold in thresholds]
    return sorted(rows, key=lambda row: row["threshold"], reverse=True)


def select_mode_row(spec: dict[str, Any], rows: list[dict[str, Any]], y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    if spec["policy"] == "fixed_threshold":
        selected = metrics_at(y_true, probabilities, float(spec["threshold"]))
    elif spec["policy"] == "max_recall_at_fpr":
        feasible = [row for row in rows if row["fpr"] <= float(spec["max_fpr"])]
        if not feasible:
            raise RuntimeError(f"no feasible threshold for {spec}")
        selected = max(feasible, key=lambda row: (row["recall_tpr"], row["precision"], row["f1"], row["threshold"]))
    elif spec["policy"] == "min_fpr_at_recall":
        feasible = [row for row in rows if row["recall_tpr"] >= float(spec["min_recall"])]
        if not feasible:
            raise RuntimeError(f"no feasible threshold for {spec}")
        selected = min(feasible, key=lambda row: (row["fpr"], -row["precision"], -row["threshold"]))
    else:
        raise ValueError(f"unknown policy: {spec['policy']}")
    return {**spec, **selected}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "label_ru",
        "description_ru",
        "policy",
        "threshold",
        "accuracy",
        "precision",
        "recall_tpr",
        "f1",
        "fpr",
        "fnr",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_curve_csv(path: Path, rows: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) > max_rows:
        indices = np.unique(np.linspace(0, len(rows) - 1, max_rows, dtype=np.int64))
        sampled = [rows[int(index)] for index in indices]
    else:
        sampled = rows
    fieldnames = ["threshold", "accuracy", "precision", "recall_tpr", "f1", "fpr", "fnr", "tn", "fp", "fn", "tp"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sampled)
    return sampled


def mode_bars_svg(path: Path, modes: list[dict[str, Any]]) -> None:
    width, height = 980, 460
    left, top, bottom = 80, 70, 75
    plot_h = height - top - bottom
    group_w = 150
    metrics = [("recall_tpr", "Recall"), ("fpr", "FPR"), ("fnr", "FNR"), ("precision", "Precision"), ("f1", "F1")]
    classes = ["bar-strict", "bar-balanced", "bar-sensitive"]
    parts = ["<text class='title' x='20' y='34'>LightGBM gold: режимы threshold</text>"]
    for idx, mode in enumerate(modes):
        x = 650 + idx * 105
        parts.append(f"<rect class='{classes[idx]}' x='{x}' y='22' width='14' height='14'/><text x='{x + 20}' y='34'>{esc(mode['label_ru'])}</text>")
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = top + (1 - frac) * plot_h
        parts.append(f"<line class='grid' x1='{left}' y1='{y:.1f}' x2='{width - 35}' y2='{y:.1f}'/>")
        parts.append(f"<text x='30' y='{y + 4:.1f}'>{frac:.2f}</text>")
    for metric_index, (metric, label) in enumerate(metrics):
        base_x = left + metric_index * group_w + 30
        for mode_index, mode in enumerate(modes):
            value = float(mode[metric])
            h = value * plot_h
            x = base_x + mode_index * 34
            parts.append(f"<rect class='{classes[mode_index]}' x='{x}' y='{top + plot_h - h:.1f}' width='28' height='{h:.1f}' rx='3'/>")
        parts.append(f"<text x='{base_x - 5}' y='{height - 30}'>{esc(label)}</text>")
    parts.append(f"<line class='axis' x1='{left}' y1='{top + plot_h}' x2='{width - 35}' y2='{top + plot_h}'/>")
    parts.append(f"<line class='axis' x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}'/>")
    path.write_text(svg(width, height, "\n".join(parts)), encoding="utf-8")


def tradeoff_svg(path: Path, curve_rows: list[dict[str, Any]], modes: list[dict[str, Any]]) -> None:
    width, height = 980, 460
    left, top, right, bottom = 80, 70, 190, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    min_t = min(float(row["threshold"]) for row in curve_rows if float(row["threshold"]) > 0)
    max_t = max(float(row["threshold"]) for row in curve_rows)
    log_min, log_max = np.log10(min_t), np.log10(max_t)

    def xy(row: dict[str, Any], metric: str) -> tuple[float, float]:
        threshold = max(float(row["threshold"]), min_t)
        x = left + (np.log10(threshold) - log_min) / max(1e-12, (log_max - log_min)) * plot_w
        y = top + (1 - float(row[metric])) * plot_h
        return float(x), float(y)

    def poly(metric: str, klass: str) -> str:
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(row, metric) for row in curve_rows))
        return f"<polyline class='{klass}' points='{points}'/>"

    parts = ["<text class='title' x='20' y='34'>LightGBM gold: tradeoff по threshold</text>"]
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = top + (1 - frac) * plot_h
        parts.append(f"<line class='grid' x1='{left}' y1='{y:.1f}' x2='{left + plot_w}' y2='{y:.1f}'/>")
        parts.append(f"<text x='30' y='{y + 4:.1f}'>{frac:.2f}</text>")
    for threshold in (0.001, 0.005, 0.01, 0.02, 0.05, 0.1):
        if min_t <= threshold <= max_t:
            x = left + (np.log10(threshold) - log_min) / max(1e-12, (log_max - log_min)) * plot_w
            parts.append(f"<line class='grid' x1='{x:.1f}' y1='{top}' x2='{x:.1f}' y2='{top + plot_h}'/>")
            parts.append(f"<text x='{x - 20:.1f}' y='{height - 30}'>{threshold:g}</text>")
    parts.extend([poly("recall_tpr", "recall"), poly("fpr", "fpr"), poly("fnr", "fnr")])
    mode_classes = {"strict": "dot-strict", "balanced": "dot-balanced", "sensitive": "dot-sensitive"}
    for mode in modes:
        for metric in ("recall_tpr", "fpr", "fnr"):
            x, y = xy(mode, metric)
            parts.append(f"<circle class='{mode_classes[mode['mode']]}' cx='{x:.1f}' cy='{y:.1f}' r='4'/>")
    parts.append(f"<line class='axis' x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}'/>")
    parts.append(f"<line class='axis' x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}'/>")
    parts.append("<text x='790' y='95'>Recall</text><line class='recall' x1='735' y1='90' x2='780' y2='90'/>")
    parts.append("<text x='790' y='125'>FPR</text><line class='fpr' x1='735' y1='120' x2='780' y2='120'/>")
    parts.append("<text x='790' y='155'>FNR</text><line class='fnr' x1='735' y1='150' x2='780' y2='150'/>")
    parts.append("<text x='405' y='440'>threshold, log scale</text>")
    path.write_text(svg(width, height, "\n".join(parts)), encoding="utf-8")


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Mode | Threshold | Recall/TPR | FNR | FPR | Precision | F1 | TN | FP | FN | TP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label_ru']} | {row['threshold']:.6f} | {pct(row['recall_tpr'])} | {pct(row['fnr'])} | {pct(row['fpr'])} | {pct(row['precision'])} | {pct(row['f1'])} | {row['tn']} | {row['fp']} | {row['fn']} | {row['tp']} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("outputs/metrics/predictions_gold_lightgbm_fast.csv"))
    parser.add_argument("--modes-output", type=Path, default=Path("outputs/metrics/lightgbm_gold_threshold_modes.csv"))
    parser.add_argument("--json-output", type=Path, default=Path("outputs/metrics/lightgbm_gold_threshold_modes.json"))
    parser.add_argument("--curve-output", type=Path, default=Path("outputs/metrics/lightgbm_gold_threshold_curve.csv"))
    parser.add_argument("--report-output", type=Path, default=Path("outputs/lightgbm_threshold_modes_report.md"))
    parser.add_argument("--figures-dir", type=Path, default=Path("outputs/figures/05_lightgbm_comparison"))
    parser.add_argument("--curve-max-rows", type=int, default=1500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    y_true, probabilities, errors = load_predictions(args.predictions)
    rows = all_threshold_metrics(y_true, probabilities)
    modes = [select_mode_row(spec, rows, y_true, probabilities) for spec in MODE_SPECS]
    curve_rows = write_curve_csv(args.curve_output, rows, args.curve_max_rows)
    write_csv(args.modes_output, modes)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "predictions": str(args.predictions),
        "scored_rows": int(len(y_true)),
        "label_counts": {"benign_or_safe_0": int(np.sum(y_true == 0)), "malicious_or_dangerous_1": int(np.sum(y_true == 1))},
        "errors": errors,
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "modes": modes,
        "note": "Gold-threshold modes are exploratory calibration/operating points, not SOREL-only threshold selection.",
    }
    args.json_output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    mode_bars_svg(args.figures_dir / "lightgbm_gold_threshold_modes.svg", modes)
    tradeoff_svg(args.figures_dir / "lightgbm_gold_threshold_tradeoff.svg", curve_rows, modes)
    args.report_output.write_text(
        "# LightGBM gold threshold modes\n\n"
        + markdown_table(modes)
        + "\n\nExploratory gold operating points; final honest evaluation still uses SOREL-selected threshold unless a separate calibration set is introduced.\n",
        encoding="utf-8",
    )
    print(json.dumps({"modes": str(args.modes_output), "json": str(args.json_output), "report": str(args.report_output)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
