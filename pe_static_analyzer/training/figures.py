"""Render report figures from the saved comparison tables."""
import json
from pe_static_analyzer.settings import ACTIVE_EXPERIMENT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def render(report):
    keys = ["random_lightgbm", "stratified_lightgbm", "stratified_catboost"]
    labels = ["LightGBM случайная", "LightGBM страты", "CatBoost страты"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, split, title in zip(axes, ["", "gold"], ["SOREL test, 19 639 PE", "Внешняя панель, 256 PE"]):
        matrices = [json.loads((report / key / split / "metrics.json").read_text())["confusion_matrix"] for key in keys]
        pd.DataFrame(matrices, index=labels)[["fp", "fn"]].plot.barh(ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Число файлов")
    fig.tight_layout()
    fig.savefig(report / "errors.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    for ax, key in zip(axes, keys[1:]):
        importance = pd.read_csv(report / key / "gold/importance.csv", index_col=0)
        importance.head(5).iloc[::-1, 0].plot.barh(ax=ax)
        ax.set_title(key.split("_")[1])
        ax.set_xlabel("Средний абсолютный вклад SHAP в исходную оценку")
    fig.tight_layout()
    fig.savefig(report / "importance.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for col, key in enumerate(keys[1:]):
        examples = pd.read_csv(report / key / "gold/examples.csv")
        for row, (label, title) in enumerate([(0, "ложная тревога"), (1, "пропуск")]):
            group = examples[(examples.label == label) & (examples.prediction != label)]
            case = group[group.sha256 == group.sha256.iloc[0]].sort_values("contribution")
            ax = axes[row, col]
            ax.barh(case.feature, case.contribution, color=["#a74232" if v > 0 else "#306291" for v in case.contribution])
            ax.axvline(0, color="black", linewidth=0.6)
            ax.set_title(f"{key.split('_')[1]} · {title}")
            ax.set_xlabel("Вклад в исходную оценку (log-odds)")
    fig.tight_layout()
    fig.savefig(report / "local_shap.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    render(ACTIVE_EXPERIMENT.report)
