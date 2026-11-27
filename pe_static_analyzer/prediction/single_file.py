"""Classify one PE file with the published LightGBM detector."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pe_static_analyzer.prediction.model import LightGbmDetector
from pe_static_analyzer.settings import DEFAULT_MODEL
from pe_static_analyzer.core.pe_input import read_pe_bytes
from pe_static_analyzer.features.ember import FEATURE_DIM, PEFeatureExtractor, is_probably_pe


def predict_file(path: Path, model_path: Path = DEFAULT_MODEL, malwarebazaar_zip: bool = False, threshold: float = 0.5) -> dict[str, object]:
    data = read_pe_bytes(path, malwarebazaar_zip)
    input_mode = "malwarebazaar_zip_in_memory" if malwarebazaar_zip else "raw_pe"
    if not is_probably_pe(data):
        raise SystemExit(f"input is not a PE file after reading: {path}")

    vector = PEFeatureExtractor().feature_vector(data)
    if vector.shape != (FEATURE_DIM,):
        raise RuntimeError(f"feature shape {vector.shape}; expected ({FEATURE_DIM},)")

    scorer = LightGbmDetector(model_path)
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
        "model_type": "lightgbm",
        "safety": "PE bytes are parsed statically only; input is not executed.",
    }


INPUT_FILE = Path("sample.exe")
READ_ENCRYPTED_ARCHIVE = False


def main() -> int:
    if not INPUT_FILE.is_file():
        raise SystemExit(f"Укажите существующий PE-файл в INPUT_FILE (сейчас {INPUT_FILE}).")
    result = predict_file(INPUT_FILE, malwarebazaar_zip=READ_ENCRYPTED_ARCHIVE)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
