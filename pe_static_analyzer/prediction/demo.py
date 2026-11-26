"""Offline smoke prediction with one published benign feature vector."""
from pe_static_analyzer.settings import DEFAULT_MODEL

from lightgbm import Booster
import numpy as np


def main():
    model = Booster(model_file=str(DEFAULT_MODEL))
    with np.load(DEFAULT_MODEL.parent / "benign_features.npz") as sample:
        probability = model.predict(sample["features"].reshape(1, -1))[0]
        print(f"sha256={sample['sha256']} score={probability:.6f} class={int(probability >= 0.5)}")


if __name__ == "__main__":
    main()
