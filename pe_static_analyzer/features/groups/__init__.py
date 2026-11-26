"""Base interface for raw and vectorized EMBER features."""
from typing import Any

import numpy as np


class FeatureType:
    name = ""
    dim = 0

    def raw_features(self, bytez: bytes, lief_binary: Any) -> Any:
        raise NotImplementedError

    def process_raw_features(self, raw_obj: Any) -> np.ndarray:
        raise NotImplementedError
