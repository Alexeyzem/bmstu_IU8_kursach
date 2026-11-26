"""Repeat external validation using already trained models."""
import json
from pathlib import Path

from pe_static_analyzer.gold.panel import ExternalPanel
from pe_static_analyzer.settings import ACTIVE_EXPERIMENT
from pe_static_analyzer.sorel.data import load_selected_indices
from pe_static_analyzer.training.boosting import FINAL_MODEL_NAME, FULL_FEATURE_MODEL_NAME, BoostingModel


def evaluate(work, report):
    indices = load_selected_indices(Path("outputs/features/selected_features.json"))
    panel = ExternalPanel.load(work / "gold_panel", indices)
    filenames = ["stratified_lightgbm.txt", "random_lightgbm.txt", "stratified_catboost.cbm"]
    for stem in ("final_lightgbm", FULL_FEATURE_MODEL_NAME):
        if (work / f"{stem}.txt").exists():
            filenames.append(f"{stem}.txt")
    for filename in filenames:
        path = work / filename
        model = BoostingModel.load(path)
        model.validate_on_external(panel, indices, report / path.stem / "gold")

    stem = work / FINAL_MODEL_NAME
    if stem.with_suffix(".txt").exists():
        indices = json.loads(stem.with_suffix(".indices.json").read_text())
        panel = ExternalPanel.load(work / "gold_panel", indices)
        model = BoostingModel.load(stem.with_suffix(".txt"))
        model.validate_on_external(panel, indices, report / stem.name / "gold")


if __name__ == "__main__":
    evaluate(ACTIVE_EXPERIMENT.work, ACTIVE_EXPERIMENT.report)
