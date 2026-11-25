#!/usr/bin/env python3
"""Run trained malware classifier inference on raw PE bytes.

Input is a PE file, not a precomputed feature matrix. The script extracts the
same EMBER-v2-style static features used for SOREL training, applies the stored
selected feature indices, then scores either the SGD baseline payload or the
separate LightGBM payload.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from pe_static_analyzer.core.pe_input import read_pe_bytes
from pe_static_analyzer.features.ember import FEATURE_DIM, PEFeatureExtractor, is_probably_pe, sorel_signed_log_postprocess


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


def predict_file(path: Path, model_path: Path, malwarebazaar_zip: bool, threshold: float) -> dict[str, object]:
    payload = joblib.load(model_path)
    data = read_pe_bytes(path, malwarebazaar_zip)
    input_mode = "malwarebazaar_zip_in_memory" if malwarebazaar_zip else "raw_pe"
    if not is_probably_pe(data):
        raise SystemExit(f"input is not a PE file after reading: {path}")

    vector = PEFeatureExtractor().feature_vector(data)
    if vector.shape != (FEATURE_DIM,):
        raise RuntimeError(f"feature shape {vector.shape}; expected ({FEATURE_DIM},)")

    scorer = ModelPayloadScorer(payload)
    probability = scorer.score_vector(vector)
    return {
        "path": str(path),
        "input_mode": input_mode,
        "sha256": hashlib.sha256(data).hexdigest(),
        "feature_dim_original": FEATURE_DIM,
        "feature_dim_model": int(len(scorer.selected_indices)),
        "malware_probability": probability,
        "threshold": threshold,
        "prediction": int(probability >= threshold),
        "prediction_label": "malware_or_dangerous" if probability >= threshold else "benign_or_safe",
        "model_type": scorer.model_type,
        "safety": "PE bytes are parsed statically only; input is not executed.",
    }


INPUT_FILE = Path("sample.exe")
MODEL_FILE = Path("outputs/models/baseline_sgd.pkl")
READ_ENCRYPTED_ARCHIVE = False


def main() -> int:
    if not INPUT_FILE.is_file():
        raise SystemExit(f"Укажите существующий PE-файл в INPUT_FILE (сейчас {INPUT_FILE}).")
    result = predict_file(INPUT_FILE, MODEL_FILE, READ_ENCRYPTED_ARCHIVE, 0.5)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
