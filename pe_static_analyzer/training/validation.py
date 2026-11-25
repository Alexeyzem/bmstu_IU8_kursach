"""One-factor parameter checks on validation; keep test out of model selection."""
from pe_static_analyzer.settings import ACTIVE_EXPERIMENT
from pathlib import Path
from functools import partial
import json
from time import monotonic

from pe_static_analyzer.training.boosting import COMPARISON_TREES, LIGHTGBM_TREES, BoostingModel
from pe_static_analyzer.core.metrics import binary_classification_metrics
from pe_static_analyzer.gold.comparison import ExternalPanel
import lmdb
import pandas as pd
from sklearn.metrics import average_precision_score

from pe_static_analyzer.sorel.data import load_selected_indices, load_split


def compare_hyperparameters():
    indices = load_selected_indices(Path("outputs/features/selected_features.json"))
    with lmdb.open("datasets/sorel20m/ember_features", readonly=True, lock=False) as env:
        x, rows = load_split(ACTIVE_EXPERIMENT.work / "stratified.csv", env, indices)
        v, validation = load_split(Path("outputs/matrices/sorel_validation_manifest.csv"), env, indices, 10000, offset=10000)
    results = []
    for name, initialize, variants in [
        ("lightgbm", partial(BoostingModel.lightgbm, trees=COMPARISON_TREES), [{}, {"num_leaves": 31}, {"min_child_samples": 300}, {"learning_rate": 0.03}]),
        ("catboost", BoostingModel.catboost, [{}, {"depth": 8}, {"l2_leaf_reg": 8}]),
    ]:
        for change in variants:
            model = initialize(**change)
            started = monotonic()
            model.train(x, rows.label)
            p = model.scores(v)
            results.append(dict(model=name, change=str(change), ap=average_precision_score(validation.label, p),
                                seconds=monotonic()-started))
    pd.DataFrame(results).to_csv(ACTIVE_EXPERIMENT.report / "validation.csv", index=False)


def compare_tree_counts():
    """Compare fixed training budgets; choose by validation AP before scoring test and gold."""
    work, report = ACTIVE_EXPERIMENT.work, ACTIVE_EXPERIMENT.report
    indices = load_selected_indices(Path("outputs/features/selected_features.json"))
    matrices = Path("outputs/matrices")
    budgets = (COMPARISON_TREES, 200, 400, LIGHTGBM_TREES, 1000)
    records = []
    model = BoostingModel.lightgbm(trees=max(budgets))

    def measure(split, count, features, labels):
        probabilities = model.estimator.booster_.predict(features, num_iteration=count, num_threads=4)
        metrics = binary_classification_metrics(labels.to_numpy(), probabilities)
        records.append(dict(split=split, trees=count, rows=len(labels), **metrics))
        return metrics["pr_auc"]

    with lmdb.open("datasets/sorel20m/ember_features", readonly=True, lock=False) as env:
        features, train = load_split(work / "stratified.csv", env, indices)
        valid_x, valid = load_split(matrices / "sorel_validation_manifest.csv", env, indices, 10000, offset=10000)
        model.train(features, train.label)
        validation_ap = {}
        for count in budgets:
            measure("train", count, features, train.label)
            validation_ap[count] = measure("validation", count, valid_x, valid.label)
        best = max(budgets, key=validation_ap.get)
        test_x, test = load_split(matrices / "sorel_test_manifest.csv", env, indices, 20000)
    panel = ExternalPanel.load(work / "gold_panel", indices)
    for count in sorted({budgets[0], 400, LIGHTGBM_TREES, best}):
        measure("test", count, test_x, test.label)
        measure("gold", count, panel.features, panel.rows.label)
    model.estimator.booster_.save_model(str(work / "tree_count_candidate.txt"), num_iteration=best)
    report.mkdir(parents=True, exist_ok=True)
    (report / "tree_counts.json").write_text(json.dumps({
        "selection": "highest validation average precision; prefer fewer trees on ties",
        "selected_trees": best, "default_trees": LIGHTGBM_TREES, "metrics": records,
    }, indent=2))


if __name__ == "__main__":
    compare_hyperparameters()
