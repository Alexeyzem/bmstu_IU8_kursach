"""Reliability diagrams on SOREL test for the two stratified boosters."""
from pe_static_analyzer.settings import ACTIVE_EXPERIMENT
from pathlib import Path

import lmdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pe_static_analyzer.training.boosting import BoostingModel
from pe_static_analyzer.evaluation.calibration import calibrated_probabilities

from pe_static_analyzer.sorel.data import load_selected_indices, load_split


def plot_reliability():
    indices = load_selected_indices(Path("outputs/features/selected_features.json"))
    with lmdb.open("datasets/sorel20m/ember_features", readonly=True, lock=False) as env:
        xv, val = load_split(Path("outputs/matrices/sorel_validation_manifest.csv"), env, indices, 10000)
        xt, test = load_split(Path("outputs/matrices/sorel_test_manifest.csv"), env, indices, 20000)
    folder = ACTIVE_EXPERIMENT.work
    models = {
        "LightGBM": BoostingModel.load(folder / "stratified_lightgbm.txt"),
        "CatBoost": BoostingModel.load(folder / "stratified_catboost.cbm"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    tables = []
    for ax, (name, model) in zip(axes, models.items()):
        scores = model.scores(xt)
        calibrated = calibrated_probabilities(model.scores(xv), val.label, scores)
        for label, p in [("Исходная", scores), ("После калибровки", calibrated)]:
            rows = pd.DataFrame({"score": p, "label": test.label})
            groups = rows.groupby(pd.cut(p, np.linspace(0, 1, 11), include_lowest=True), observed=True)
            bins = groups.agg(score=("score", "mean"), frequency=("label", "mean"), count=("label", "size"))
            ax.plot(bins.score, bins.frequency, "o-", label=label)
            tables.append(bins.assign(model=name, series=label))
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set(title=name, xlabel="Средняя оценка", ylabel="Доля вредоносных")
        ax.legend()
    out = ACTIVE_EXPERIMENT.report
    pd.concat(tables).to_csv(out / "reliability.csv")
    fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_reliability()
