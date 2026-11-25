"""Prepare the labeled external sample used by the comparison."""
from pathlib import Path

import pandas as pd

from pe_static_analyzer.settings import ACTIVE_EXPERIMENT

GOLD_METADATA = Path("outputs/gold/gold_final_metadata.csv")
FILES_PER_CLASS = 128
PANEL_SEED = 42  # Keep the external files identical across training runs.


def main() -> None:
    metadata = pd.read_csv(GOLD_METADATA)
    shuffled = metadata.sample(frac=1, random_state=PANEL_SEED)
    panel = shuffled.groupby("label").head(FILES_PER_CLASS)
    ACTIVE_EXPERIMENT.work.mkdir(parents=True, exist_ok=True)
    panel.to_csv(ACTIVE_EXPERIMENT.work / "gold_panel.csv", index=False)


if __name__ == "__main__":
    main()
