"""Score raw EMBER vectors with the published LightGBM detector."""
import json
from pathlib import Path

import numpy as np
from lightgbm import Booster

from pe_static_analyzer.features.ember import sorel_signed_log_postprocess
from pe_static_analyzer.settings import DEFAULT_MODEL


class LightGbmDetector:
    def __init__(self, path: Path = DEFAULT_MODEL):
        self.selected_indices = np.asarray(json.loads(path.with_suffix(".indices.json").read_text()), dtype=np.int32)
        self.booster = Booster(model_file=str(path))

    def score_vector(self, vector: np.ndarray) -> float:
        selected = sorel_signed_log_postprocess(vector[self.selected_indices])
        return float(self.booster.predict(selected.reshape(1, -1))[0])
