#!/usr/bin/env python3
"""Generate separate SGD-vs-LightGBM comparison artifacts.

This script is additive: it does not modify existing SGD scripts/artifacts.
It reads the final SGD 5-epoch artifacts and LightGBM-specific artifacts,
then writes LightGBM comparison CSV/JSON/Markdown and SVG figures.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs" / "metrics"
FIGURES = ROOT / "outputs" / "figures" / "05_lightgbm_comparison"
REPORT = ROOT / "outputs" / "lightgbm_comparison_report.md"

STYLE = """
<style>
text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #1f2937; font-size: 13px; }
.title { font-size: 18px; font-weight: 700; }
.axis { stroke: #6b7280; stroke-width: 1; }
.grid { stroke: #e5e7eb; stroke-width: 1; }
.bar0 { fill: #2563eb; }
.bar1 { fill: #7c3aed; }
.line1 { fill: none; stroke: #7c3aed; stroke-width: 3; }
.line2 { fill: none; stroke: #059669; stroke-width: 3; }
.dot1 { fill: #7c3aed; }
.dot2 { fill: #059669; }
</style>
""".strip()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: Any) -> str:
    if value is None or value == "":
        return "—"
    return f"{100.0 * float(value):.2f}%"


def num(value: Any, digits: int = 6) -> str:
    if value is None or value == "":
        return "—"
    return f"{float(value):.{digits}f}"


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg(width: int, height: int, body: str) -> str:
    return f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' width='{width}' height='{height}'>\n{STYLE}\n{body}\n</svg>\n"


def metric_row(model: str, split: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {"model": model, "split": split, "available": False}
    return {
        "model": model,
        "split": split,
        "available": True,
        "threshold": payload.get("threshold"),
        "accuracy": payload.get("accuracy"),
        "precision": payload.get("precision"),
        "recall_tpr": payload.get("recall_tpr"),
        "f1": payload.get("f1"),
        "roc_auc": payload.get("roc_auc"),
        "pr_auc": payload.get("pr_auc"),
        "fpr": payload.get("fpr"),
        "fnr": payload.get("fnr"),
        "runtime_seconds": payload.get("runtime_seconds"),
        "trained_rows": payload.get("trained_rows"),
        "best_epoch": payload.get("best_epoch"),
        "processed_rows": payload.get("processed_rows"),
        "scored_rows": payload.get("scored_rows"),
        "error_rows": payload.get("error_rows"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "split", "available", "threshold", "accuracy", "precision", "recall_tpr", "f1", "roc_auc", "pr_auc", "fpr", "fnr", "runtime_seconds", "trained_rows", "best_epoch", "processed_rows", "scored_rows", "error_rows"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    output = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(output)


def grouped_metric_bars(path: Path, title: str, rows: list[dict[str, Any]], split: str) -> None:
    metrics = ["accuracy", "precision", "recall_tpr", "f1", "fpr", "fnr"]
    sgd = next((row for row in rows if row["model"] == "sgd_5epoch" and row["split"] == split and row.get("available")), None)
    lgbm = next((row for row in rows if row["model"] == "lightgbm" and row["split"] == split and row.get("available")), None)
    if not sgd or not lgbm:
        return
    width, height = 980, 430
    left, top, bottom, group_w = 80, 70, 60, 135
    plot_h = height - top - bottom
    parts = [f"<text class='title' x='20' y='32'>{esc(title)}</text>"]
    parts.append("<rect class='bar0' x='710' y='25' width='16' height='16'/><text x='732' y='38'>SGD 5 epochs</text>")
    parts.append("<rect class='bar1' x='840' y='25' width='16' height='16'/><text x='862' y='38'>LightGBM</text>")
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = top + (1 - frac) * plot_h
        parts.append(f"<line class='grid' x1='{left}' y1='{y:.1f}' x2='{width - 35}' y2='{y:.1f}'/>")
        parts.append(f"<text x='30' y='{y + 4:.1f}'>{frac:.2f}</text>")
    for idx, metric in enumerate(metrics):
        x = left + idx * group_w + 25
        v1 = float(sgd.get(metric) or 0)
        v2 = float(lgbm.get(metric) or 0)
        h1 = v1 * plot_h
        h2 = v2 * plot_h
        parts.append(f"<rect class='bar0' x='{x}' y='{top + plot_h - h1:.1f}' width='40' height='{h1:.1f}' rx='3'/>")
        parts.append(f"<rect class='bar1' x='{x + 46}' y='{top + plot_h - h2:.1f}' width='40' height='{h2:.1f}' rx='3'/>")
        parts.append(f"<text x='{x - 12}' y='{height - 25}'>{esc(metric)}</text>")
    parts.append(f"<line class='axis' x1='{left}' y1='{top + plot_h}' x2='{width - 35}' y2='{top + plot_h}'/>")
    parts.append(f"<line class='axis' x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}'/>")
    path.write_text(svg(width, height, "\n".join(parts)), encoding="utf-8")


def lightgbm_epoch_plot(path: Path, history_path: Path) -> None:
    if not history_path.exists():
        return
    rows: list[dict[str, float]] = []
    with history_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row.get("epoch"):
                continue
            rows.append({"epoch": float(row["epoch"]), "accuracy": float(row["accuracy"]), "f1": float(row["f1"])})
    if not rows:
        return
    width, height = 900, 420
    left, top, right, bottom = 70, 60, 150, 60
    plot_w, plot_h = width - left - right, height - top - bottom
    max_epoch = max(row["epoch"] for row in rows) or 1

    def point(row: dict[str, float], metric: str) -> tuple[float, float]:
        x = left + (row["epoch"] - 1) / max(1, max_epoch - 1) * plot_w
        y = top + (1 - row[metric]) * plot_h
        return x, y

    def poly(metric: str, klass: str) -> str:
        return f"<polyline class='{klass}' points='" + " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(row, metric) for row in rows)) + "'/>"

    parts = ["<text class='title' x='20' y='32'>LightGBM validation metrics by epoch</text>"]
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = top + (1 - frac) * plot_h
        parts.append(f"<line class='grid' x1='{left}' y1='{y:.1f}' x2='{left + plot_w}' y2='{y:.1f}'/>")
        parts.append(f"<text x='25' y='{y + 4:.1f}'>{frac:.2f}</text>")
    for row in rows:
        x = left + (row["epoch"] - 1) / max(1, max_epoch - 1) * plot_w
        parts.append(f"<line class='grid' x1='{x:.1f}' y1='{top}' x2='{x:.1f}' y2='{top + plot_h}'/>")
        parts.append(f"<text x='{x - 4:.1f}' y='{height - 25}'>{int(row['epoch'])}</text>")
    parts.append(poly("accuracy", "line1"))
    parts.append(poly("f1", "line2"))
    for row in rows:
        for metric, klass in (("accuracy", "dot1"), ("f1", "dot2")):
            x, y = point(row, metric)
            parts.append(f"<circle class='{klass}' cx='{x:.1f}' cy='{y:.1f}' r='4'/>")
    parts.append(f"<line class='axis' x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}'/>")
    parts.append(f"<line class='axis' x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}'/>")
    parts.append("<rect class='bar1' x='770' y='90' width='16' height='4'/><text x='792' y='96'>accuracy</text>")
    parts.append("<rect class='bar0' x='770' y='120' width='16' height='4'/><text x='792' y='126'>F1</text>")
    path.write_text(svg(width, height, "\n".join(parts)), encoding="utf-8")


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = [
        metric_row("sgd_5epoch", "validation", load_json(METRICS / "sorel_validation_metrics.json")),
        metric_row("sgd_5epoch", "test", load_json(METRICS / "sorel_test_metrics.json")),
        metric_row("sgd_5epoch", "gold", load_json(METRICS / "gold_metrics.json")),
        metric_row("lightgbm", "validation", load_json(METRICS / "lightgbm_validation_metrics.json")),
        metric_row("lightgbm", "test", load_json(METRICS / "lightgbm_test_metrics.json")),
        metric_row("lightgbm", "gold", load_json(METRICS / "gold_metrics_lightgbm_fast.json")),
    ]
    write_csv(METRICS / "model_comparison_sgd_lightgbm.csv", rows)
    (METRICS / "model_comparison_sgd_lightgbm.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    grouped_metric_bars(FIGURES / "sgd_vs_lightgbm_validation.svg", "SGD vs LightGBM: SOREL validation", rows, "validation")
    grouped_metric_bars(FIGURES / "sgd_vs_lightgbm_test.svg", "SGD vs LightGBM: SOREL test", rows, "test")
    grouped_metric_bars(FIGURES / "sgd_vs_lightgbm_gold.svg", "SGD vs LightGBM: external gold", rows, "gold")
    lightgbm_epoch_plot(FIGURES / "lightgbm_epoch_curve.svg", METRICS / "lightgbm_training_history.csv")

    table_rows = []
    for row in rows:
        if row.get("available"):
            table_rows.append([row["model"], row["split"], num(row.get("threshold")), pct(row.get("accuracy")), pct(row.get("precision")), pct(row.get("recall_tpr")), pct(row.get("f1")), pct(row.get("fpr")), pct(row.get("fnr"))])
    REPORT.write_text("# SGD vs LightGBM\n\n" + markdown_table(["Model", "Split", "Threshold", "Accuracy", "Precision", "Recall", "F1", "FPR", "FNR"], table_rows) + "\n", encoding="utf-8")
    print(json.dumps({"comparison_csv": str(METRICS / "model_comparison_sgd_lightgbm.csv"), "report": str(REPORT)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
