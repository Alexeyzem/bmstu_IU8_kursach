import json
import unittest

import numpy as np
from lightgbm import Booster

from pe_static_analyzer.features.schema import build_feature_schema
from pe_static_analyzer.prediction.model import LightGbmDetector
from pe_static_analyzer.settings import DEFAULT_MODEL


class DefaultDetectorTest(unittest.TestCase):
    def test_published_model_and_index_order_match_the_demo(self):
        model = Booster(model_file=str(DEFAULT_MODEL))
        indices = json.loads(DEFAULT_MODEL.with_suffix(".indices.json").read_text())
        schema = build_feature_schema()
        self.assertEqual(model.num_trees(), 600)
        self.assertEqual(model.num_feature(), 2155)
        self.assertEqual(len(indices), model.num_feature())
        self.assertEqual(indices, sorted(set(indices)))
        self.assertFalse(any(schema[i].group == "header" for i in indices))
        with np.load(DEFAULT_MODEL.parent / "benign_features.npz") as sample:
            selected = sample["features"]
        raw = np.zeros(len(schema), dtype=np.float32)
        raw[indices] = np.sign(selected) * np.expm1(np.abs(selected))
        scorer = LightGbmDetector()
        self.assertAlmostEqual(scorer.score_vector(raw), 0.0002613412521455382, places=10)
