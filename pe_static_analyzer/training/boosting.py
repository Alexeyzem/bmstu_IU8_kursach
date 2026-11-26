"""Model lifecycle for the published PE experiments."""
import json
from pathlib import Path

from catboost import CatBoostClassifier
from lightgbm import Booster, LGBMClassifier

from pe_static_analyzer.core.metrics import binary_classification_metrics
from pe_static_analyzer.evaluation.calibration import calibration_metrics
from pe_static_analyzer.evaluation.shap_values import explain

LIGHTGBM_TREES = 600
FULL_FEATURE_MODEL_NAME = f"final_lightgbm_{LIGHTGBM_TREES}"
FINAL_MODEL_NAME = f"without_header_{LIGHTGBM_TREES}"
COMPARISON_TREES = 100
LEARNING_RATE = 0.08
MODEL_SEED = 42
THREADS = 4


class BoostingModel:
    def __init__(self, estimator):
        self.estimator = estimator

    @classmethod
    def lightgbm(cls, num_leaves=63, min_child_samples=100, learning_rate=LEARNING_RATE, trees=LIGHTGBM_TREES):
        return cls(LGBMClassifier(
            n_estimators=trees, learning_rate=learning_rate, num_leaves=num_leaves,
            min_child_samples=min_child_samples, random_state=MODEL_SEED,
            n_jobs=THREADS, verbosity=-1,
        ))

    @classmethod
    def catboost(cls, depth=6, l2_leaf_reg=3):
        return cls(CatBoostClassifier(
            iterations=COMPARISON_TREES, learning_rate=LEARNING_RATE, depth=depth,
            l2_leaf_reg=l2_leaf_reg, random_seed=MODEL_SEED, thread_count=THREADS,
            verbose=False, allow_writing_files=False,
        ))

    def train(self, features, labels):
        if isinstance(self.estimator, Booster):
            raise ValueError("Initialize a new LightGBM model with BoostingModel.lightgbm() to train.")
        weights = len(labels) / (2 * labels.map(labels.value_counts()))
        self.estimator.fit(features, labels, sample_weight=weights.to_numpy())

    def scores(self, features):
        if isinstance(self.estimator, Booster):
            return self.estimator.predict(features)
        return self.estimator.predict_proba(features)[:, 1]

    def save(self, stem: Path):
        if isinstance(self.estimator, CatBoostClassifier):
            self.estimator.save_model(str(stem.with_suffix(".cbm")))
        else:
            booster = self.estimator if isinstance(self.estimator, Booster) else self.estimator.booster_
            booster.save_model(str(stem.with_suffix(".txt")))

    @classmethod
    def load(cls, path: Path):
        if path.suffix == ".cbm":
            return cls(CatBoostClassifier().load_model(str(path)))
        if path.suffix == ".txt":
            return cls(Booster(model_file=str(path)))
        raise ValueError(f"Unknown model format: {path}")

    def validate_on_sorel(self, test, validation, indices, out, train_rows, train_seconds):
        features, rows = test
        validation_x, validation_rows = validation
        scores = self.scores(features)
        metrics = binary_classification_metrics(rows.label.to_numpy(), scores)
        metrics.update(calibration_metrics(self.scores(validation_x), validation_rows.label, scores, rows.label))
        metrics.update(train_rows=train_rows, train_seconds=train_seconds, test_rows=len(rows))
        self._write_report(features, rows, indices, out, metrics)
        return metrics

    def validate_on_external(self, panel, indices, out):
        scores = self.scores(panel.features)
        rows = panel.rows
        metrics = binary_classification_metrics(rows.label.to_numpy(), scores)
        metrics.update(requested=panel.requested, scored=len(rows))
        self._write_report(panel.features, rows, indices, out, metrics)
        rows.assign(score=scores, prediction=(scores >= 0.5).astype(int)).to_csv(out / "predictions.csv", index=False)
        return metrics

    def _write_report(self, features, rows, indices, out, metrics):
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        explain(self.estimator, features, rows, indices, out)
