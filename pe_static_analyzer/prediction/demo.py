"""Offline smoke prediction with one published benign feature vector."""
from pathlib import Path

from lightgbm import Booster
import numpy as np


def main():
    examples = Path(__file__).resolve().parents[2] / "examples"
    model = Booster(model_file=str(examples / "lightgbm.txt"))
    with np.load(examples / "benign_features.npz") as sample:
        probability = model.predict(sample["features"].reshape(1, -1))[0]
        print(f"sha256={sample['sha256']} score={probability:.6f} class={int(probability >= 0.5)}")


if __name__ == "__main__":
    main()
