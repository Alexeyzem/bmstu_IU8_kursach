"""Repeat external validation using already trained models."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pe_static_analyzer.settings import ACTIVE_EXPERIMENT
from pe_static_analyzer.sorel.data import load_selected_indices
from pe_static_analyzer.training.boosting import FINAL_MODEL_NAME, BoostingModel


@dataclass
class ExternalPanel:
    features: np.ndarray
    rows: pd.DataFrame
    requested: int

    @classmethod
    def load(cls, folder, indices):
        summary = json.loads((folder / "gold_feature_summary.json").read_text())
        if summary["postprocess"] != "sorel_signed_log":
            raise ValueError("Expected signed-log features")
        rows = pd.read_parquet(folder / "gold_features.parquet")
        features = rows[[f"f_{i:04d}" for i in indices]].to_numpy()
        return cls(features, rows[["sha256", "label"]], summary["requested_rows"])


def evaluate(work, report):
    indices = load_selected_indices(Path("outputs/features/selected_features.json"))
    panel = ExternalPanel.load(work / "gold_panel", indices)
    filenames = ["stratified_lightgbm.txt", "random_lightgbm.txt", "stratified_catboost.cbm"]
    for stem in ("final_lightgbm", FINAL_MODEL_NAME):
        if (work / f"{stem}.txt").exists():
            filenames.append(f"{stem}.txt")
    for filename in filenames:
        path = work / filename
        model = BoostingModel.load(path)
        model.validate_on_external(panel, indices, report / path.stem / "gold")


if __name__ == "__main__":
    evaluate(ACTIVE_EXPERIMENT.work, ACTIVE_EXPERIMENT.report)
