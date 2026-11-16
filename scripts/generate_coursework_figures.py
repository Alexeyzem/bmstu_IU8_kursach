#!/usr/bin/env python3
"""Generate coursework figures/tables from produced JSON/CSV artifacts.

The script uses only the Python standard library so it works in the project
virtualenv without plotting dependencies. It writes SVG figures and a compact
Markdown table bundle for the report.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "report_tables.md"


SVG_STYLE = """
<style>
text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #1f2937; font-size: 13px; }
.title { font-size: 18px; font-weight: 700; }
.subtitle { fill: #4b5563; font-size: 12px; }
.axis { stroke: #6b7280; stroke-width: 1; }
.grid { stroke: #e5e7eb; stroke-width: 1; }
.bar0 { fill: #2563eb; }
.bar1 { fill: #dc2626; }
.bar2 { fill: #059669; }
.bar3 { fill: #d97706; }
.line1 { fill: none; stroke: #2563eb; stroke-width: 2; }
.line2 { fill: none; stroke: #dc2626; stroke-width: 2; }
.line3 { fill: none; stroke: #059669; stroke-width: 2; }
.cell-good { fill: #d1fae5; }
.cell-bad { fill: #fee2e2; }
.cell-neutral { fill: #e0f2fe; }
</style>
""".strip()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_int(value: int | float | str | None) -> str:
    if value is None or value == "":
        return "—"
    return f"{int(value):,}".replace(",", " ")


def fmt_float(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "—"
    return f"{float(value):.{digits}f}"


def pct(value: Any) -> str:
    if value is None or value == "":
        return "—"
    return f"{100.0 * float(value):.2f}%"


def ensure_dirs() -> None:
    for path in [
        FIGURES / "01_datasets",
        FIGURES / "02_deduplication",
        FIGURES / "03_features",
        FIGURES / "04_model_quality",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def svg_doc(width: int, height: int, body: str) -> str:
    return f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' width='{width}' height='{height}'>\n{SVG_STYLE}\n{body}\n</svg>\n"


def bar_chart(path: Path, title: str, rows: list[tuple[str, float, str]], width: int = 900, height: int | None = None) -> None:
    height = height or max(260, 90 + len(rows) * 42)
    left, right, top, bottom = 260, 40, 56, 42
    plot_w = width - left - right
    max_value = max((value for _, value, _ in rows), default=1.0)
    max_value = max(max_value, 1.0)
    parts = [f"<text class='title' x='20' y='30'>{title}</text>"]
    for idx, (label, value, color_class) in enumerate(rows):
        y = top + idx * 42
        bar_w = 0 if max_value == 0 else value / max_value * plot_w
        parts.append(f"<text x='20' y='{y + 22}'>{escape(label)}</text>")
        parts.append(f"<rect class='{color_class}' x='{left}' y='{y}' width='{bar_w:.1f}' height='26' rx='3'/>")
        parts.append(f"<text x='{left + bar_w + 8:.1f}' y='{y + 19}'>{fmt_int(value)}</text>")
    parts.append(f"<line class='axis' x1='{left}' y1='{height - bottom}' x2='{width - right}' y2='{height - bottom}'/>")
    path.write_text(svg_doc(width, height, "\n".join(parts)), encoding="utf-8")


def grouped_bar_chart(path: Path, title: str, groups: list[tuple[str, float, float]], series_labels: tuple[str, str], width: int = 900) -> None:
    height = max(320, 100 + len(groups) * 72)
    left, right, top = 230, 50, 70
    plot_w = width - left - right
    max_value = max([v for _, a, b in groups for v in (a, b)] or [1.0])
    parts = [f"<text class='title' x='20' y='30'>{title}</text>"]
    parts.append(f"<rect class='bar0' x='20' y='45' width='14' height='14'/><text x='42' y='57'>{escape(series_labels[0])}</text>")
    parts.append(f"<rect class='bar1' x='190' y='45' width='14' height='14'/><text x='212' y='57'>{escape(series_labels[1])}</text>")
    for idx, (label, a, b) in enumerate(groups):
        y = top + idx * 72
        aw = a / max_value * plot_w if max_value else 0
        bw = b / max_value * plot_w if max_value else 0
        parts.append(f"<text x='20' y='{y + 30}'>{escape(label)}</text>")
        parts.append(f"<rect class='bar0' x='{left}' y='{y}' width='{aw:.1f}' height='24' rx='3'/><text x='{left + aw + 8:.1f}' y='{y + 17}'>{fmt_int(a)}</text>")
        parts.append(f"<rect class='bar1' x='{left}' y='{y + 30}' width='{bw:.1f}' height='24' rx='3'/><text x='{left + bw + 8:.1f}' y='{y + 47}'>{fmt_int(b)}</text>")
    path.write_text(svg_doc(width, height, "\n".join(parts)), encoding="utf-8")


def confusion_matrix_svg(path: Path, title: str, cm: dict[str, Any]) -> None:
    width, height = 620, 430
    left, top, cell = 210, 100, 150
    values = {key: int(cm.get(key, 0)) for key in ("tn", "fp", "fn", "tp")}
    max_value = max(values.values()) or 1
    cells = [
        ("tn", 0, 0, "True safe", "Pred safe", "cell-good"),
        ("fp", 1, 0, "False alarm", "Pred dangerous", "cell-bad"),
        ("fn", 0, 1, "Missed dangerous", "Pred safe", "cell-bad"),
        ("tp", 1, 1, "True dangerous", "Pred dangerous", "cell-good"),
    ]
    parts = [f"<text class='title' x='20' y='34'>{escape(title)}</text>"]
    parts.append(f"<text class='subtitle' x='{left}' y='72'>Predicted class</text>")
    parts.append(f"<text transform='translate(40 250) rotate(-90)' class='subtitle'>True class</text>")
    parts.append(f"<text x='{left + 35}' y='94'>safe</text><text x='{left + cell + 25}' y='94'>dangerous</text>")
    parts.append(f"<text x='90' y='{top + 80}'>safe</text><text x='72' y='{top + cell + 80}'>dangerous</text>")
    for key, xidx, yidx, label, _pred, klass in cells:
        x = left + xidx * cell
        y = top + yidx * cell
        opacity = 0.25 + 0.75 * values[key] / max_value
        parts.append(f"<rect class='{klass}' x='{x}' y='{y}' width='{cell - 6}' height='{cell - 6}' rx='8' opacity='{opacity:.3f}'/>")
        parts.append(f"<text x='{x + 18}' y='{y + 48}'>{escape(label)}</text>")
        parts.append(f"<text class='title' x='{x + 18}' y='{y + 88}'>{fmt_int(values[key])}</text>")
    path.write_text(svg_doc(width, height, "\n".join(parts)), encoding="utf-8")


def threshold_plot(path: Path, curve_path: Path) -> None:
    if not curve_path.exists():
        return
    rows: list[dict[str, float]] = []
    with curve_path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            try:
                t = float(raw["threshold"])
            except (ValueError, TypeError):
                continue
            if not math.isfinite(t) or t < 0 or t > 1:
                continue
            rows.append({
                "threshold": t,
                "precision": float(raw["precision"]),
                "recall_tpr": float(raw["recall_tpr"]),
                "fpr": float(raw["fpr"]),
                "fnr": float(raw["fnr"]),
            })
    if not rows:
        return
    rows = sorted(rows, key=lambda item: item["threshold"])
    # Downsample deterministically for compact SVG.
    step = max(1, len(rows) // 900)
    rows = rows[::step]
    width, height = 950, 420
    left, right, top, bottom = 70, 170, 56, 56
    plot_w, plot_h = width - left - right, height - top - bottom

    def point(metric: str, row: dict[str, float]) -> tuple[float, float]:
        x = left + row["threshold"] * plot_w
        y = top + (1.0 - row[metric]) * plot_h
        return x, y

    def polyline(metric: str, klass: str) -> str:
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(metric, row) for row in rows))
        return f"<polyline class='{klass}' points='{pts}'/>"

    parts = [f"<text class='title' x='20' y='30'>Threshold tradeoff on SOREL validation</text>"]
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = top + (1.0 - frac) * plot_h
        x = left + frac * plot_w
        parts.append(f"<line class='grid' x1='{left}' y1='{y:.1f}' x2='{left + plot_w}' y2='{y:.1f}'/>")
        parts.append(f"<text x='24' y='{y + 4:.1f}'>{frac:.2f}</text>")
        parts.append(f"<line class='grid' x1='{x:.1f}' y1='{top}' x2='{x:.1f}' y2='{top + plot_h}'/>")
        parts.append(f"<text x='{x - 10:.1f}' y='{top + plot_h + 25}'>{frac:.2f}</text>")
    parts.append(polyline("precision", "line1"))
    parts.append(polyline("recall_tpr", "line2"))
    parts.append(polyline("fpr", "line3"))
    parts.append(f"<line class='axis' x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}'/>")
    parts.append(f"<line class='axis' x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}'/>")
    parts.append(f"<text x='{left + plot_w / 2 - 40}' y='{height - 10}'>threshold</text>")
    parts.append("<rect class='bar0' x='800' y='80' width='16' height='4'/><text x='824' y='86'>precision</text>")
    parts.append("<rect class='bar1' x='800' y='110' width='16' height='4'/><text x='824' y='116'>recall/TPR</text>")
    parts.append("<rect class='bar2' x='800' y='140' width='16' height='4'/><text x='824' y='146'>FPR</text>")
    path.write_text(svg_doc(width, height, "\n".join(parts)), encoding="utf-8")


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def read_gold_source_counts(metadata_path: Path) -> tuple[dict[str, int], dict[str, int]]:
    label_counts = {"benign_or_safe_0": 0, "malicious_or_dangerous_1": 0}
    source_counts: dict[str, int] = {}
    if not metadata_path.exists():
        return label_counts, source_counts
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label_counts["malicious_or_dangerous_1" if row.get("label") == "1" else "benign_or_safe_0"] += 1
            source = row.get("source") or "unknown"
            source_counts[source] = source_counts.get(source, 0) + 1
    return label_counts, source_counts


def markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(out)


def main() -> int:
    ensure_dirs()
    matrix = load_json(ROOT / "outputs" / "matrices" / "matrix_build_summary.json") or {}
    dedup = load_json(ROOT / "outputs" / "dedup" / "dedup_summary.json") or {}
    features = load_json(ROOT / "outputs" / "features" / "feature_analysis_summary.json") or {}
    validation = load_json(ROOT / "outputs" / "metrics" / "sorel_validation_metrics.json")
    test = load_json(ROOT / "outputs" / "metrics" / "sorel_test_metrics.json")
    gold = load_json(ROOT / "outputs" / "metrics" / "gold_metrics.json")

    splits = matrix.get("splits", {})
    if splits:
        grouped_bar_chart(
            FIGURES / "01_datasets" / "sorel_split_composition.svg",
            "SOREL split composition",
            [(name, payload.get("benign_0", 0), payload.get("malware_1", 0)) for name, payload in splits.items()],
            ("benign/safe", "malicious/dangerous"),
        )
        total_benign = sum(payload.get("benign_0", 0) for payload in splits.values())
        total_malware = sum(payload.get("malware_1", 0) for payload in splits.values())
        bar_chart(
            FIGURES / "01_datasets" / "sorel_total_composition.svg",
            "SOREL total class composition",
            [("benign/safe", total_benign, "bar0"), ("malicious/dangerous", total_malware, "bar1")],
        )

    gold_labels, gold_sources = read_gold_source_counts(ROOT / "outputs" / "gold" / "gold_final_metadata.csv")
    if sum(gold_labels.values()):
        bar_chart(
            FIGURES / "01_datasets" / "gold_class_composition.svg",
            "External gold class composition",
            [("benign/safe", gold_labels["benign_or_safe_0"], "bar0"), ("malicious/dangerous", gold_labels["malicious_or_dangerous_1"], "bar1")],
        )
        top_sources = sorted(gold_sources.items(), key=lambda item: item[1], reverse=True)[:10]
        bar_chart(FIGURES / "01_datasets" / "gold_top_sources.svg", "Top gold sources", [(source, count, "bar2") for source, count in top_sources], height=560)

    if dedup:
        rows = [
            ("initial", dedup.get("gold_initial_total", 0), "bar0"),
            ("after SOREL exact-overlap removal", dedup.get("after_exact_total", 0), "bar2"),
            ("final", dedup.get("gold_final_total", 0), "bar3"),
        ]
        bar_chart(FIGURES / "02_deduplication" / "gold_dedup_funnel.svg", "Gold deduplication funnel", rows)

    if features:
        bar_chart(
            FIGURES / "03_features" / "feature_selection.svg",
            "Feature selection: EMBER/SOREL vector",
            [("original features", features.get("feature_dim_original", 0), "bar0"), ("selected features", features.get("feature_dim_selected", 0), "bar2"), ("removed features", features.get("removed_count", 0), "bar1")],
        )
        reasons = features.get("reason_counts", {})
        bar_chart(FIGURES / "03_features" / "feature_removal_reasons.svg", "Removed feature reasons", [(key, value, "bar3") for key, value in reasons.items()], height=360)

    for name, payload in (("sorel_validation", validation), ("sorel_test", test), ("gold", gold)):
        if payload and payload.get("confusion_matrix"):
            confusion_matrix_svg(FIGURES / "04_model_quality" / f"{name}_confusion_matrix.svg", f"{name.replace('_', ' ').title()} confusion matrix", payload["confusion_matrix"])

    threshold_plot(FIGURES / "04_model_quality" / "threshold_tradeoff.svg", ROOT / "outputs" / "metrics" / "threshold_curve.csv")

    metric_rows = []
    for label, payload in (("SOREL validation", validation), ("SOREL test", test), ("Gold", gold)):
        if payload:
            metric_rows.append([
                label,
                fmt_float(payload.get("threshold"), 6),
                pct(payload.get("accuracy")),
                pct(payload.get("precision")),
                pct(payload.get("recall_tpr")),
                pct(payload.get("f1")),
                pct(payload.get("roc_auc")),
                pct(payload.get("pr_auc")),
                pct(payload.get("fpr")),
                pct(payload.get("fnr")),
            ])
    table_parts = [
        "# Report tables\n",
        "## Model quality\n",
        markdown_table(["Dataset", "Threshold", "Accuracy", "Precision", "Recall/TPR", "F1", "ROC-AUC", "PR-AUC", "FPR", "FNR"], metric_rows),
        "\n## SOREL split composition\n",
    ]
    if splits:
        table_parts.append(markdown_table(["Split", "Total", "Benign/safe", "Malicious/dangerous"], [[name, fmt_int(payload.get("total")), fmt_int(payload.get("benign_0")), fmt_int(payload.get("malware_1"))] for name, payload in splits.items()]))
    table_parts.extend([
        "\n## Feature selection\n",
        markdown_table(["Metric", "Value"], [["Original features", fmt_int(features.get("feature_dim_original"))], ["Selected features", fmt_int(features.get("feature_dim_selected"))], ["Removed features", fmt_int(features.get("removed_count"))]]),
        "\n## Gold deduplication\n",
        markdown_table(["Metric", "Value"], [["Initial gold", fmt_int(dedup.get("gold_initial_total"))], ["Exact overlap with SOREL", fmt_int(dedup.get("exact_overlap_sorel"))], ["Final gold", fmt_int(dedup.get("gold_final_total"))], ["Near-dedup status", dedup.get("near_dedup_status", "—")]]),
    ])
    TABLES.write_text("\n".join(table_parts) + "\n", encoding="utf-8")
    print(json.dumps({"figures_dir": str(FIGURES), "tables": str(TABLES)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
