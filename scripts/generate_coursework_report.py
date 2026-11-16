#!/usr/bin/env python3
"""Generate the coursework report from reproducible artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs" / "coursework_report.md"


def load_json(path: Path, required: bool = True) -> dict[str, Any] | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_int(value: Any) -> str:
    if value is None or value == "":
        return "—"
    return f"{int(value):,}".replace(",", " ")


def pct(value: Any) -> str:
    if value is None or value == "":
        return "—"
    return f"{100 * float(value):.2f}%"


def flt(value: Any, digits: int = 6) -> str:
    if value is None or value == "":
        return "—"
    return f"{float(value):.{digits}f}"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    output = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        output.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(output)


def metric_row(name: str, payload: dict[str, Any]) -> list[str]:
    return [
        name,
        flt(payload.get("threshold"), 6),
        pct(payload.get("accuracy")),
        pct(payload.get("precision")),
        pct(payload.get("recall_tpr")),
        pct(payload.get("f1")),
        pct(payload.get("roc_auc")),
        pct(payload.get("pr_auc")),
        pct(payload.get("fpr")),
        pct(payload.get("fnr")),
    ]


def confusion_row(name: str, payload: dict[str, Any]) -> list[str]:
    cm = payload.get("confusion_matrix", {})
    return [name, fmt_int(cm.get("tn")), fmt_int(cm.get("fp")), fmt_int(cm.get("fn")), fmt_int(cm.get("tp"))]


def figure(path: str, caption: str) -> str:
    return f"![{caption}]({path})\n\n*{caption}*"


def main() -> int:
    matrix = load_json(ROOT / "outputs" / "matrices" / "matrix_build_summary.json") or {}
    dedup = load_json(ROOT / "outputs" / "dedup" / "dedup_summary.json") or {}
    features = load_json(ROOT / "outputs" / "features" / "feature_analysis_summary.json") or {}
    threshold = load_json(ROOT / "outputs" / "metrics" / "threshold_selection.json") or {}
    validation = load_json(ROOT / "outputs" / "metrics" / "sorel_validation_metrics.json") or {}
    test = load_json(ROOT / "outputs" / "metrics" / "sorel_test_metrics.json") or {}
    gold = load_json(ROOT / "outputs" / "metrics" / "gold_metrics.json") or {}
    comparison = load_json(ROOT / "outputs" / "metrics" / "model_comparison_1epoch_vs_5epoch.json", required=False) or []

    splits = matrix.get("splits", {})
    split_rows = [
        [name, fmt_int(data.get("total")), fmt_int(data.get("benign_0")), fmt_int(data.get("malware_1"))]
        for name, data in splits.items()
    ]

    metric_rows = [metric_row("SOREL validation", validation), metric_row("SOREL test", test), metric_row("External gold", gold)]
    confusion_rows = [confusion_row("SOREL validation", validation), confusion_row("SOREL test", test), confusion_row("External gold", gold)]

    selected = threshold.get("selected_metrics", {})
    default = threshold.get("default_0_5_metrics", {})
    operating = threshold.get("operating_points", {})
    comparison_rows = [
        [item.get("model"), item.get("split"), flt(item.get("threshold"), 6), pct(item.get("accuracy")), pct(item.get("precision")), pct(item.get("recall_tpr")), pct(item.get("f1")), pct(item.get("fpr")), pct(item.get("fnr"))]
        for item in comparison
        if item.get("available")
    ]

    report = f"""# Курсовая работа: статическая классификация PE-файлов

## Постановка задачи

Цель работы — бинарная классификация Windows PE-файлов на два класса: безопасный PE и опасный/вредоносный PE. Положительный класс означает опасный/вредоносный PE по меткам SOREL/MalwareBazaar, а не CVE-уязвимость продукта. Мультиклассовая постановка не используется: для практического детектора и учебного baseline важнее отделить опасные файлы от безопасных, чем предсказывать семейство.

Анализ статический: модель получает признаки из байтов PE и не запускает файл. MalwareBazaar samples хранятся как password-protected ZIP с расширением `.zip.txt`; raw PE читается только в память. Benign PE хранятся с defanged-расширениями `.exe.txt`, `.dll.txt`, `.pe.txt`.

## Данные

### SOREL-20M

SOREL используется только для обучения, validation, test и выбора threshold. Gold-набор не влияет на обучение, feature selection или threshold selection.

{table(["Split", "Total", "Benign/safe", "Malicious/dangerous"], split_rows)}

{figure("figures/01_datasets/sorel_split_composition.svg", "Состав SOREL train/validation/test по классам")}

### External gold holdout

Gold-набор используется как внешний holdout после выбора модели и threshold. После exact dedup осталось {fmt_int(dedup.get("gold_final_total"))} samples: {fmt_int(dedup.get("gold_final_negative"))} benign/safe и {fmt_int(dedup.get("gold_final_positive"))} malicious/dangerous. Exact overlap с SOREL: {fmt_int(dedup.get("exact_overlap_sorel"))}. Near-dedup byte-level `>90%` не заявляется: {dedup.get("near_dedup_limitation", "ограничение не указано")}.

{figure("figures/01_datasets/gold_class_composition.svg", "Состав external gold по классам")}

{figure("figures/02_deduplication/gold_dedup_funnel.svg", "Воронка exact dedup для gold")}

## Признаки

Используется EMBER/SOREL-compatible vector из {fmt_int(features.get("feature_dim_original"))} признаков. Перед моделью применяется `sorel_signed_log`. Feature selection удалил {fmt_int(features.get("removed_count"))} признаков; осталось {fmt_int(features.get("feature_dim_selected"))}. Filename/path/source/url не входят в feature vector, чтобы не давать leakage.

{table(["Removal reason", "Features"], [[key, fmt_int(value)] for key, value in (features.get("reason_counts") or {}).items()])}

{figure("figures/03_features/feature_selection.svg", "Количество признаков до/после feature selection")}

{figure("figures/03_features/feature_removal_reasons.svg", "Причины удаления признаков")}

## Модель

Основная модель — `SGDClassifier(loss=\"log_loss\")` с `StandardScaler` и balanced sample weights. Это линейный baseline: быстрый inference, простая воспроизводимость и достаточное качество для демонстрации полного безопасного pipeline. Сложная модель вроде LightGBM оставлена как возможное улучшение после честного сравнения; текущий baseline уже закрывает end-to-end задачу и raw-file inference.

Инференс: raw file/ZIP → проверка `MZ` и `PE\\0\\0` → EMBER-v2-compatible признаки → `sorel_signed_log` → 2192 selected features → scaler → модель → probability → threshold.

## Выбор threshold

Порог выбран на SOREL validation. Основная политика: максимальный recall при `FPR <= 5%`, если такой operating point есть. Это снижает ложные срабатывания на безопасных PE по сравнению с threshold `0.5`, но увеличивает долю пропущенных опасных PE.

Выбранный threshold: `{flt(threshold.get("selected_threshold"), 12)}`. Политика: `{threshold.get("selected_policy", "—")}`.

| Operating point | Threshold | Precision | Recall/TPR | FPR | FNR | F1 |
|---|---:|---:|---:|---:|---:|---:|
| default 0.5 | {flt(default.get("threshold"), 6)} | {pct(default.get("precision"))} | {pct(default.get("recall_tpr"))} | {pct(default.get("fpr"))} | {pct(default.get("fnr"))} | {pct(default.get("f1"))} |
| selected | {flt(selected.get("threshold"), 6)} | {pct(selected.get("precision"))} | {pct(selected.get("recall_tpr"))} | {pct(selected.get("fpr"))} | {pct(selected.get("fnr"))} | {pct(selected.get("f1"))} |
| FPR <= 1% | {flt((operating.get("fpr_1_percent") or {}).get("threshold"), 6)} | {pct((operating.get("fpr_1_percent") or {}).get("precision"))} | {pct((operating.get("fpr_1_percent") or {}).get("recall_tpr"))} | {pct((operating.get("fpr_1_percent") or {}).get("fpr"))} | {pct((operating.get("fpr_1_percent") or {}).get("fnr"))} | {pct((operating.get("fpr_1_percent") or {}).get("f1"))} |

{figure("figures/04_model_quality/threshold_tradeoff.svg", "Tradeoff precision/recall/FPR по threshold")}

## Результаты

{table(["Dataset", "Threshold", "Accuracy", "Precision", "Recall/TPR", "F1", "ROC-AUC", "PR-AUC", "FPR", "FNR"], metric_rows)}

{table(["Dataset", "TN", "FP", "FN", "TP"], confusion_rows)}

{figure("figures/04_model_quality/sorel_validation_confusion_matrix.svg", "Confusion matrix SOREL validation")}

{figure("figures/04_model_quality/sorel_test_confusion_matrix.svg", "Confusion matrix SOREL test")}

{figure("figures/04_model_quality/gold_confusion_matrix.svg", "Confusion matrix external gold")}

## Сравнение 1 эпохи и 5 эпох

После ночного полного запуска сравниваются исходный 1-epoch baseline и модель `baseline_sgd_5epoch_full.pkl`. График обучения строится по validation accuracy/F1 после каждой эпохи.

{table(["Model", "Split", "Threshold", "Accuracy", "Precision", "Recall/TPR", "F1", "FPR", "FNR"], comparison_rows) if comparison_rows else "Сравнение будет заполнено после завершения полного 5-epoch run."}

{figure("figures/04_model_quality/epoch_accuracy_curve.svg", "Validation accuracy/F1 по эпохам")}

{figure("figures/04_model_quality/model_comparison_test.svg", "Сравнение 1 epoch vs 5 epochs на SOREL test")}

{figure("figures/04_model_quality/model_comparison_gold.svg", "Сравнение 1 epoch vs 5 epochs на external gold")}

## GUI и batch inference

`main.py` запускает небольшой Tkinter-интерфейс на русском языке. Пользователь выбирает или перетаскивает PE/defanged PE/MalwareBazaar ZIP и нажимает `Проверить`; приложение статически извлекает признаки, применяет модель и показывает `уязвимо/опасно` или `не уязвимо/безопасно`, probability, threshold, hash и режим чтения.

`scripts/predict_gold_raw.py` является совместимой точкой входа для batch raw-file inference gold-набора. Скрипт пишет `outputs/metrics/predictions_gold.csv`, `outputs/metrics/gold_metrics.json`, `outputs/metrics/gold_inference_errors.csv` инкрементально, поддерживает `--resume`, `--sample-timeout` и не читает `gold_features.parquet`.

## Ограничения

- SOREL старее современных угроз, поэтому внешний gold нужен для проверки distribution drift.
- MalwareBazaar часть содержит только malicious samples; benign часть собрана отдельно из official/WinGet источников.
- Labels не идеальны: официальный benign источник не гарантирует отсутствие PUA/adware, а malware tags не являются CVE labels.
- Near dedup `>90%` не доказан без отдельного безопасного raw-byte сопоставления.
- Высокое качество на SOREL не гарантирует такое же качество на новых PE.
- Линейный baseline хуже моделирует нелинейные взаимодействия признаков, чем boosting, но проще и быстрее для воспроизводимого курсача.

## Воспроизводимость

Ключевые команды:

```bash
.venv/bin/python scripts/train_binary_classifier.py --eval-only --input-model outputs/models/baseline_sgd.pkl --validation-limit 20000 --test-limit 20000 --threshold-output outputs/metrics/threshold_selection.json --threshold-curve-output outputs/metrics/threshold_curve.csv --test-metrics-output outputs/metrics/sorel_test_metrics.json
.venv/bin/python scripts/train_binary_classifier.py --epochs 5 --train-limit 5000 --validation-limit 1000 --test-limit 1000 --output-model outputs/models/baseline_sgd_5epoch_smoke.pkl --metrics-output outputs/metrics/sorel_validation_metrics_5epoch_smoke.json --test-metrics-output outputs/metrics/sorel_test_metrics_5epoch_smoke.json --threshold-output outputs/metrics/threshold_selection_5epoch_smoke.json --threshold-curve-output outputs/metrics/threshold_curve_5epoch_smoke.csv
.venv/bin/python main.py --check-imports
.venv/bin/python scripts/predict_gold_raw.py --sample-timeout 20 --resume
.venv/bin/python scripts/generate_coursework_figures.py
.venv/bin/python scripts/generate_coursework_report.py
```

Полный 5-epoch run без `--train-limit` по всему SOREL train ожидаемо долгий: предыдущий 1-epoch full run занял около 9976 секунд. Для проверки кода добавлен 5-epoch smoke run с отдельными артефактами, не перезаписывающими full baseline.
"""
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(REPORT)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
