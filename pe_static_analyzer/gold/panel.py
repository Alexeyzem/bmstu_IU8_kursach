"""Prepare the labeled external sample used by the comparison."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pe_static_analyzer.settings import ACTIVE_EXPERIMENT

GOLD_METADATA = Path("outputs/gold/gold_final_metadata.csv")
FILES_PER_CLASS = 128
PANEL_SEED = 42  # Keep the external files identical across training runs.


@dataclass
class ExternalPanel:
    features: np.ndarray
    rows: pd.DataFrame
    requested: int

    @classmethod
    def load(cls, folder, indices):
        summary = json.loads((folder / "gold_feature_summary.json").read_text())
        if summary["postprocess"] != "sorel_signed_log":
            raise ValueError("Expected signed-log features")
        rows = pd.read_parquet(folder / "gold_features.parquet")
        features = rows[[f"f_{i:04d}" for i in indices]].to_numpy()
        return cls(features, rows[["sha256", "label"]], summary["requested_rows"])


def main() -> None:
    metadata = pd.read_csv(GOLD_METADATA)
    shuffled = metadata.sample(frac=1, random_state=PANEL_SEED)
    panel = shuffled.groupby("label").head(FILES_PER_CLASS)
    ACTIVE_EXPERIMENT.work.mkdir(parents=True, exist_ok=True)
    panel.to_csv(ACTIVE_EXPERIMENT.work / "gold_panel.csv", index=False)


if __name__ == "__main__":
    main()
