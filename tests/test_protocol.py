import unittest

import numpy as np
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory

from pe_static_analyzer.core.metrics import fixed_threshold_report
from pe_static_analyzer.sorel.sampling import write_samples


class ProtocolTest(unittest.TestCase):
    def test_cutoff_includes_equality(self):
        threshold, _, rows = fixed_threshold_report(np.array([0, 1, 1]), np.array([0.49, 0.5, 0.51]))
        self.assertEqual(threshold, 0.5)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["confusion_matrix"], dict(tn=1, fp=0, fn=0, tp=2))

    def test_stratified_manifest(self):
        data = pd.DataFrame(dict(sha256=list("abcdefghijkl"), label=[0]*8+[1]*4, month=["2018-01"]*12))
        with TemporaryDirectory() as directory:
            output = Path(directory)
            write_samples(data, 6, 42, output)
            result = pd.read_csv(output / "stratified.csv")
            self.assertEqual(result.label.value_counts().to_dict(), {0: 4, 1: 2})
            self.assertTrue(result.sha256.is_unique)
            write_samples(data, 6, 42, output)
            pd.testing.assert_frame_equal(result, pd.read_csv(output / "stratified.csv"))
