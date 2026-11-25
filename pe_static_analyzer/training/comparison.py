"""Train, validate and save the boosters for one experiment profile."""
from pathlib import Path
from time import monotonic

import lmdb

from pe_static_analyzer.gold.comparison import ExternalPanel
from pe_static_analyzer.settings import ACTIVE_EXPERIMENT
from pe_static_analyzer.sorel.data import load_selected_indices, load_split
from pe_static_analyzer.training.boosting import COMPARISON_TREES, FINAL_MODEL_NAME, BoostingModel


def compare_boosters(work, report):
    indices = load_selected_indices(Path("outputs/features/selected_features.json"))
    folder = work / "gold_panel"
    external = ExternalPanel.load(folder, indices) if folder.exists() else None
    if external is None:
        print(f"External evaluation deferred until features are available in {folder}.")
    matrices = Path("outputs/matrices")
    with lmdb.open("datasets/sorel20m/ember_features", readonly=True, lock=False) as env:
        test = load_split(matrices / "sorel_test_manifest.csv", env, indices, 20000)
        validation = load_split(matrices / "sorel_validation_manifest.csv", env, indices, 10000)
        for split in ("stratified", "random"):
            features, rows = load_split(work / f"{split}.csv", env, indices)
            models = {f"{split}_lightgbm": BoostingModel.lightgbm(trees=COMPARISON_TREES)}
            if split == "stratified":
                models["stratified_catboost"] = BoostingModel.catboost()
                models[FINAL_MODEL_NAME] = BoostingModel.lightgbm()

            for key, model in models.items():
                start = monotonic()
                model.train(features, rows.label)
                elapsed = monotonic() - start
                model.save(work / key)
                metrics = model.validate_on_sorel(test, validation, indices, report / key, len(rows), elapsed)
                if external is not None:
                    model.validate_on_external(external, indices, report / key / "gold")
                print(key, metrics, flush=True)
            del features


if __name__ == "__main__":
    compare_boosters(ACTIVE_EXPERIMENT.work, ACTIVE_EXPERIMENT.report)
