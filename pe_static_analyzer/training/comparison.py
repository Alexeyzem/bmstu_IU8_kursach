"""Train boosters on the experiment manifests and evaluate them on fixed splits."""
import json
from pathlib import Path
from time import monotonic

import lmdb

from pe_static_analyzer.features.schema import build_feature_schema
from pe_static_analyzer.gold.panel import ExternalPanel
from pe_static_analyzer.settings import ACTIVE_EXPERIMENT
from pe_static_analyzer.sorel.data import load_selected_indices, load_split
from pe_static_analyzer.training.boosting import (
    COMPARISON_TREES,
    FINAL_MODEL_NAME,
    FULL_FEATURE_MODEL_NAME,
    BoostingModel,
)


def train_and_evaluate(model, stem, train, test, validation, indices, report, panel, *, save_indices=False):
    features, rows = train
    start = monotonic()
    model.train(features, rows.label)
    elapsed = monotonic() - start
    model.save(stem)
    if save_indices:
        stem.with_suffix(".indices.json").write_text(json.dumps(indices))
    metrics = model.validate_on_sorel(test, validation, indices, report, len(rows), elapsed)
    if panel is not None:
        model.validate_on_external(panel, indices, report / "gold")
    print(stem.name, metrics, flush=True)


def train_without_header(work, report, train, test, validation, indices, panel):
    schema = build_feature_schema()
    columns = [j for j, index in enumerate(indices) if schema[index].group != "header"]
    selected = [indices[j] for j in columns]

    def select_columns(split):
        features, rows = split
        return features[:, columns], rows

    if panel is not None:
        panel = ExternalPanel(panel.features[:, columns], panel.rows, panel.requested)
    stem = work / FINAL_MODEL_NAME
    train_and_evaluate(
        BoostingModel.lightgbm(), stem, select_columns(train), select_columns(test),
        select_columns(validation), selected, report / stem.name, panel, save_indices=True,
    )


def compare_boosters(work, report):
    indices = load_selected_indices(Path("outputs/features/selected_features.json"))
    folder = work / "gold_panel"
    panel = ExternalPanel.load(folder, indices) if folder.exists() else None
    if panel is None:
        print(f"External evaluation deferred until features are available in {folder}.")
    matrices = Path("outputs/matrices")
    with lmdb.open("datasets/sorel20m/ember_features", readonly=True, lock=False) as env:
        test = load_split(matrices / "sorel_test_manifest.csv", env, indices, 20000)
        validation = load_split(matrices / "sorel_validation_manifest.csv", env, indices, 10000)
        for split in ("stratified", "random"):
            train = load_split(work / f"{split}.csv", env, indices)
            models = {f"{split}_lightgbm": BoostingModel.lightgbm(trees=COMPARISON_TREES)}
            if split == "stratified":
                models["stratified_catboost"] = BoostingModel.catboost()
                models[FULL_FEATURE_MODEL_NAME] = BoostingModel.lightgbm()

            for name, model in models.items():
                train_and_evaluate(model, work / name, train, test, validation, indices, report / name, panel)
            if split == "stratified":
                train_without_header(work, report, train, test, validation, indices, panel)
            del train


if __name__ == "__main__":
    compare_boosters(ACTIVE_EXPERIMENT.work, ACTIVE_EXPERIMENT.report)
