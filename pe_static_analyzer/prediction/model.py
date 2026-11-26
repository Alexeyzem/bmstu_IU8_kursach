"""Load saved detectors and score raw EMBER feature vectors."""
import json
from pathlib import Path

import joblib
import numpy as np

from pe_static_analyzer.features.ember import sorel_signed_log_postprocess


def load_model_payload(path: Path) -> dict:
    if path.suffix != ".txt":
        return joblib.load(path)
    return {
        "model_type": "lightgbm",
        "model_string": path.read_text(),
        "selected_indices": json.loads(path.with_suffix(".indices.json").read_text()),
        "postprocess": "sorel_signed_log",
    }


class ModelPayloadScorer:
    """Score selected EMBER features with an SGD or LightGBM payload."""

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.selected_indices = np.asarray(payload["selected_indices"], dtype=np.int32)
        self.model_type = str(payload.get("model_type", "sgd"))

    def score_vector(self, vector: np.ndarray) -> float:
        if self.payload.get("postprocess") == "sorel_signed_log":
            vector = sorel_signed_log_postprocess(vector)
        x = vector[self.selected_indices].reshape(1, -1)
        if self.model_type == "lightgbm":
            import lightgbm as lgb

            booster = lgb.Booster(model_str=self.payload["model_string"])
            return float(booster.predict(x)[0])
        x_scaled = self.payload["scaler"].transform(x)
        return float(self.payload["model"].predict_proba(x_scaled)[0, 1])
