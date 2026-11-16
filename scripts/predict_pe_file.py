#!/usr/bin/env python3
"""Run trained malware classifier inference on raw PE bytes.

Input is a PE file, not a precomputed feature matrix. The script extracts the
same EMBER-v2-style static features used for SOREL training, applies the stored
selected feature indices, then scores either the SGD baseline payload or the
separate LightGBM payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pyzipper

from extract_gold_features import MALWAREBAZAAR_ZIP_PASSWORD, PEFeatureExtractor, FEATURE_DIM, is_probably_pe, sorel_signed_log_postprocess


def read_input_bytes(path: Path, malwarebazaar_zip: bool) -> tuple[bytes, str]:
    if not malwarebazaar_zip:
        return path.read_bytes(), "raw_pe"
    with pyzipper.AESZipFile(path) as zf:
        zf.setpassword(MALWAREBAZAAR_ZIP_PASSWORD)
        names = [name for name in zf.namelist() if not name.endswith("/")]
        if len(names) != 1:
            raise RuntimeError(f"expected one archive member, got {len(names)}")
        return zf.read(names[0]), "malwarebazaar_zip_in_memory"


def predict_file(path: Path, model_path: Path, malwarebazaar_zip: bool, threshold: float) -> dict[str, object]:
    payload = joblib.load(model_path)
    data, input_mode = read_input_bytes(path, malwarebazaar_zip)
    if not is_probably_pe(data):
        raise SystemExit(f"input is not a PE file after reading: {path}")

    extractor = PEFeatureExtractor()
    vector = extractor.feature_vector(data)
    if vector.shape != (FEATURE_DIM,):
        raise RuntimeError(f"feature shape {vector.shape}; expected ({FEATURE_DIM},)")
    if payload.get("postprocess") == "sorel_signed_log":
        vector = sorel_signed_log_postprocess(vector)

    selected_indices = np.asarray(payload["selected_indices"], dtype=np.int32)
    x = vector[selected_indices].reshape(1, -1)
    model_type = payload.get("model_type", "sgd")
    if model_type == "lightgbm":
        import lightgbm as lgb

        booster = lgb.Booster(model_str=payload["model_string"])
        probability = float(booster.predict(x)[0])
    else:
        x_scaled = payload["scaler"].transform(x)
        probability = float(payload["model"].predict_proba(x_scaled)[0, 1])
    return {
        "path": str(path),
        "input_mode": input_mode,
        "sha256": hashlib.sha256(data).hexdigest(),
        "feature_dim_original": FEATURE_DIM,
        "feature_dim_model": int(len(selected_indices)),
        "malware_probability": probability,
        "threshold": threshold,
        "prediction": int(probability >= threshold),
        "prediction_label": "malware_or_dangerous" if probability >= threshold else "benign_or_safe",
        "model_type": str(model_type),
        "safety": "PE bytes are parsed statically only; input is not executed.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--model", type=Path, default=Path("outputs/models/baseline_sgd.pkl"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--malwarebazaar-zip", action="store_true", help="Read a MalwareBazaar password-protected ZIP member in memory; raw PE is not written to disk.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = predict_file(args.path, args.model, args.malwarebazaar_zip, args.threshold)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
