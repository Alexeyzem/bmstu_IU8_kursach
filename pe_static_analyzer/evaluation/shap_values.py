"""TreeSHAP contributions for LightGBM and CatBoost."""
from catboost import CatBoostClassifier, Pool
from lightgbm import Booster
import numpy as np
import pandas as pd

from pe_static_analyzer.features.schema import build_feature_schema


def tree_contributions(model, features):
    if isinstance(model, CatBoostClassifier):
        contributions = model.get_feature_importance(Pool(features), type="ShapValues", thread_count=4)
        margins = model.predict(features, prediction_type="RawFormulaVal")
    else:
        booster = model if isinstance(model, Booster) else model.booster_
        contributions = booster.predict(features, pred_contrib=True)
        margins = booster.predict(features, raw_score=True)
    np.testing.assert_allclose(contributions.sum(axis=1), margins, atol=1e-5)
    return contributions, margins


def explain(model, x, rows, indices, out):
    sample = rows.reset_index(drop=True).sample(n=min(512, len(rows)), random_state=42)
    contributions, margins = tree_contributions(model, x[sample.index])
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
