"""Compare signed contributions for missed and detected malware on the gold panel."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from pe_static_analyzer.evaluation.shap_values import tree_contributions
from pe_static_analyzer.features.schema import build_feature_schema
from pe_static_analyzer.gold.panel import ExternalPanel
from pe_static_analyzer.settings import ACTIVE_EXPERIMENT
from pe_static_analyzer.sorel.data import load_selected_indices
from pe_static_analyzer.training.boosting import FINAL_MODEL_NAME, BoostingModel


def compare_gold_errors(model_name=FINAL_MODEL_NAME):
    work = ACTIVE_EXPERIMENT.work
    out = ACTIVE_EXPERIMENT.report / model_name / "gold"
    if model_name == FINAL_MODEL_NAME:
        indices = json.loads((work / f"{model_name}.indices.json").read_text())
    else:
        indices = load_selected_indices(Path("outputs/features/selected_features.json"))
    panel = ExternalPanel.load(work / "gold_panel", indices)
    features = panel.features[panel.rows.label == 1]
    model = BoostingModel.load(work / f"{model_name}.txt").estimator
    contributions, margins = tree_contributions(model, features)
    detected = margins >= 0.0
    if detected.all() or not detected.any():
        raise ValueError("The panel must contain both missed and detected malware.")

    schema = build_feature_schema()
    groups = [schema[i].group for i in indices]
    grouped = pd.DataFrame(contributions[:, :-1].T).groupby(groups).sum().T
    table = pd.DataFrame({
        "missed": grouped[~detected].mean(),
        "detected": grouped[detected].mean(),
    })
    table["difference"] = table.detected - table.missed
    table = table.sort_values("difference", ascending=False)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "shap_groups.csv", index_label="group")
    ax = table[["missed", "detected"]].plot.barh(figsize=(8, 5), color=["#a44832", "#287181"])
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.7)
    ax.legend([ax.containers[0], ax.containers[1]],
              [f"Пропущены ({sum(~detected)})", f"Обнаружены ({sum(detected)})"])
    ax.set(xlabel="Средний суммарный SHAP-вклад, log-odds", ylabel="",
           title="Вредоносные файлы gold · LightGBM")
    ax.figure.tight_layout()
    ax.figure.savefig(out / "shap_groups.png", dpi=150)
    plt.close(ax.figure)
    print(table.round(3).to_string())
    print(f"Mean margins: missed {margins[~detected].mean():.3f}, detected {margins[detected].mean():.3f}")


if __name__ == "__main__":
    compare_gold_errors()
