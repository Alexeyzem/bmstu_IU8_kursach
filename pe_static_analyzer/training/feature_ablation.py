"""Retrain LightGBM on fixed SOREL rows while varying its input columns."""
import json
from pathlib import Path
from time import monotonic

import lmdb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from pe_static_analyzer.core.metrics import binary_classification_metrics
from pe_static_analyzer.features.schema import build_feature_schema
from pe_static_analyzer.settings import ACTIVE_EXPERIMENT
from pe_static_analyzer.sorel.data import load_selected_indices, load_split
from pe_static_analyzer.training.boosting import COMPARISON_TREES, BoostingModel


ABLATION_TREES = 400


def compare_feature_sets():
    work, report = ACTIVE_EXPERIMENT.work, ACTIVE_EXPERIMENT.report
    schema = build_feature_schema()
    all_indices = list(range(len(schema)))
    selected = load_selected_indices(Path("outputs/features/selected_features.json"))
    without_header = [i for i in selected if schema[i].group != "header"]
    without_histogram = [i for i in selected if schema[i].group != "histogram"]
    byte_groups = {"histogram", "byteentropy", "strings"}
    byte_features = [i for i in selected if schema[i].group in byte_groups]
    matrices = Path("outputs/matrices")
    with lmdb.open("datasets/sorel20m/ember_features", readonly=True, lock=False) as env:
        train_x, train = load_split(work / "stratified.csv", env, all_indices)
        test_x, test = load_split(matrices / "sorel_test_manifest.csv", env, all_indices, 20000)
        valid_x, valid = load_split(matrices / "sorel_validation_manifest.csv", env, all_indices, 10000, offset=10000)
    panel = pd.read_parquet(work / "gold_panel/gold_features.parquet")

    experiments = [
        (COMPARISON_TREES, {"all": all_indices, "without_header": without_header}, "feature_ablation.json"),
        (ABLATION_TREES, {"selected": selected, "without_header": without_header,
                          "without_histogram": without_histogram, "byte_features": byte_features}, f"feature_ablation_{ABLATION_TREES}.json"),
    ]
    report.mkdir(parents=True, exist_ok=True)
    for trees, variants, filename in experiments:
        results = {}
        for name, indices in variants.items():
            model = BoostingModel.lightgbm(trees=trees)
            features = np.ascontiguousarray(train_x[:, indices])
            started = monotonic()
            model.train(features, train.label)
            seconds = monotonic() - started
            gold_x = panel[[f"f_{i:04d}" for i in indices]].to_numpy()
            splits = {"test": (test_x[:, indices], test.label), "gold": (gold_x, panel.label)}
            if trees == ABLATION_TREES:
                splits["validation"] = (valid_x[:, indices], valid.label)
                splits["train"] = (features, train.label)
                if name in {"without_header", "byte_features"}:
                    stem = work / f"{name}_{trees}"
                    model.save(stem)
                    stem.with_suffix(".indices.json").write_text(json.dumps(indices))
            results[name] = {}
            for split, (matrix, labels) in splits.items():
                scores = model.scores(matrix)
                metrics = binary_classification_metrics(labels.to_numpy(), scores)
                if trees == ABLATION_TREES:
                    metrics.update(brier=brier_score_loss(labels, scores), log_loss=log_loss(labels, scores))
                results[name][split] = metrics
            if trees == ABLATION_TREES:
                results[name]["fit"] = dict(seconds=seconds, features=len(indices), trees=trees)
        (report / filename).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    compare_feature_sets()
