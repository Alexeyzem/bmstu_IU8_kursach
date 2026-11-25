"""Calibration and SHAP reports shared by trained and saved boosters."""
from catboost import CatBoostClassifier, Pool
from lightgbm import Booster
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from pe_static_analyzer.features.schema import build_feature_schema


def logits(scores):
    p = np.clip(scores, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p)).reshape(-1, 1)


def calibrated_probabilities(validation_scores, validation_labels, scores):
    calibrator = LogisticRegression().fit(logits(validation_scores), validation_labels)
    return calibrator.predict_proba(logits(scores))[:, 1]


def calibration_metrics(validation_scores, validation_labels, scores, labels):
    calibrated = calibrated_probabilities(validation_scores, validation_labels, scores)
    return dict(brier=brier_score_loss(labels, scores),
                calibrated_brier=brier_score_loss(labels, calibrated),
                log_loss=log_loss(labels, scores),
                calibrated_log_loss=log_loss(labels, calibrated))


def explain(model, x, rows, indices, out):
    sample = rows.sample(n=min(512, len(rows)), random_state=42)
    values = x[sample.index]
    if isinstance(model, CatBoostClassifier):
        contributions = model.get_feature_importance(Pool(values), type="ShapValues", thread_count=4)
        margins = model.predict(values, prediction_type="RawFormulaVal")
    else:
        booster = model if isinstance(model, Booster) else model.booster_
        contributions = booster.predict(values, pred_contrib=True)
        margins = booster.predict(values, raw_score=True)
    np.testing.assert_allclose(contributions.sum(axis=1), margins, atol=1e-5)
    schema = build_feature_schema()
    names = [schema[i].name for i in indices]
    shap = pd.DataFrame(contributions[:, :-1], columns=names, index=sample.sha256)
    shap.abs().mean().sort_values(ascending=False).to_csv(out / "importance.csv", header=["mean_abs_shap"])
    sample = sample.assign(prediction=(margins >= 0).astype(int), margin=margins, base=contributions[:, -1])
    details = []
    for _, group in sample.groupby(["label", "prediction"]):
        for row in group.head(3).itertuples():
            impacts = shap.loc[row.sha256]
            for name in impacts.abs().nlargest(5).index:
                details.append(dict(sha256=row.sha256, label=row.label, prediction=row.prediction,
                                    margin=row.margin, base=row.base, feature=name, contribution=impacts[name]))
    pd.DataFrame(details).to_csv(out / "examples.csv", index=False)
