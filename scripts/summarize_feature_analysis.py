#!/usr/bin/env python3
"""Create a concise Markdown report from SOREL feature-analysis outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("outputs/features"))
    parser.add_argument("--output", type=Path, default=Path("outputs/features/feature_reduction_notes.md"))
    parser.add_argument("--top", type=int, default=15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = args.features_dir / "feature_analysis_summary.json"
    removed_path = args.features_dir / "removed_features.csv"
    pairs_path = args.features_dir / "high_correlation_pairs.csv"
    assoc_path = args.features_dir / "target_association.csv"
    selected_path = args.features_dir / "selected_features.json"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    removed = read_csv(removed_path)
    pairs = read_csv(pairs_path)
    assoc = read_csv(assoc_path)
    selected = json.loads(selected_path.read_text(encoding="utf-8"))

    reason_counts = Counter(row["reason"] for row in removed)
    group_removed = Counter(row["group"] for row in removed)
    group_rows = []
    for group, values in summary["group_summary"].items():
        group_rows.append([group, values["original"], values["removed"], values["selected"]])

    reason_rows = [[reason, count] for reason, count in reason_counts.most_common()]
    group_removed_rows = [[group, count] for group, count in group_removed.most_common()]

    top_assoc_rows = []
    for row in assoc[: args.top]:
        top_assoc_rows.append([
            row["index"],
            row["name"],
            row["group"],
            f"{float(row['mutual_information_bits']):.6f}",
            f"{float(row['target_point_biserial']):.6f}",
        ])

    top_pair_rows = []
    for row in pairs[: args.top]:
        top_pair_rows.append([
            row["feature_i"],
            row["name_i"],
            row["feature_j"],
            row["name_j"],
            row["group_i"] + " / " + row["group_j"],
            f"{float(row['pearson']):.6f}",
        ])

    lines: list[str] = []
    lines.append("# Анализ корреляции и сокращения признаков SOREL/EMBER")
    lines.append("")
    lines.append("## Проверенный объём")
    lines.append("")
    lines.append(markdown_table(
        ["Параметр", "Значение"],
        [
            ["Источник", summary["lmdb_dir"]],
            ["Split", summary.get("split", "")],
            ["Postprocess", summary.get("postprocess", "")],
            ["Sample size", summary["sample_size"]],
            ["Benign", summary["label_counts"]["benign_0"]],
            ["Malware", summary["label_counts"]["malware_1"]],
            ["Missing LMDB keys skipped", summary["sample_stats"].get("lmdb_missing", 0)],
            ["Исходная размерность", summary["feature_dim_original"]],
            ["Удалено", summary["removed_count"]],
            ["Осталось", summary["feature_dim_selected"]],
        ],
    ))
    lines.append("")
    lines.append("## Сокращение по группам")
    lines.append("")
    lines.append(markdown_table(["Группа", "Было", "Удалено", "Осталось"], group_rows))
    lines.append("")
    lines.append("## Причины удаления")
    lines.append("")
    lines.append(markdown_table(["Причина", "Признаков"], reason_rows))
    lines.append("")
    lines.append("## Где удалялось больше всего")
    lines.append("")
    lines.append(markdown_table(["Группа", "Удалено"], group_removed_rows))
    lines.append("")
    lines.append("## Наиболее связанные с целевой меткой признаки")
    lines.append("")
    lines.append(markdown_table(["Index", "Name", "Group", "MI bits", "Point-biserial"], top_assoc_rows))
    lines.append("")
    lines.append("## Самые сильные корреляции Pearson")
    lines.append("")
    if top_pair_rows:
        lines.append(markdown_table(["i", "feature i", "j", "feature j", "groups", "Pearson"], top_pair_rows))
    else:
        lines.append("Нет пар выше заданного порога.")
    lines.append("")
    lines.append("## Правило сокращения")
    lines.append("")
    rules = selected.get("rules", {})
    for key in ["constant", "near_constant", "high_zero_low_mi", "correlated", "leakage"]:
        if key in rules:
            lines.append(f"- `{key}`: {rules[key]}")
    lines.append("")
    lines.append("## Ограничения")
    lines.append("")
    lines.append("- Результат относится к выбранной подвыборке SOREL train и должен фиксироваться вместе с seed/sample SHA list.")
    lines.append("- Hashed-группы imports/exports/sections нельзя однозначно развернуть в исходные имена из-за hashing trick и коллизий.")
    lines.append("- Gold features должны использовать тот же EMBER-v2 порядок признаков и тот же postprocess, иначе selected_features.json несовместим.")
    lines.append("- Feature importance после baseline-модели ещё нужно добавить после обучения модели; текущий этап закрывает статистику признаков, MI и корреляции.")
    lines.append("")
    lines.append("## Выходные файлы")
    lines.append("")
    for name, path in summary["outputs"].items():
        lines.append(f"- `{name}`: `{path}`")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
