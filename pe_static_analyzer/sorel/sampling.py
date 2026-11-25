"""Create reproducible training manifests from an indexed random SHA prefix range."""
from pe_static_analyzer.settings import ACTIVE_EXPERIMENT
import sqlite3
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd

from pe_static_analyzer.sorel.data import TRAIN_VALIDATION_SPLIT


def build_manifests(size: int, seed: int, output: Path) -> None:
    if size < 1:
        raise ValueError("size must be positive")
    rng = np.random.default_rng(seed)
    prefixes = rng.permutation(256)
    rows = []
    with sqlite3.connect("file:datasets/sorel20m/meta.db?mode=ro", uri=True) as db, lmdb.open(
        "datasets/sorel20m/ember_features", readonly=True, lock=False
    ) as env, env.begin() as txn:
        for prefix in prefixes:
            low = f"{prefix:02x}"
            high = f"{prefix + 1:02x}" if prefix < 255 else "g"
            query = """SELECT sha256, is_malware, strftime('%Y-%m', rl_fs_t, 'unixepoch')
                       FROM meta WHERE sha256 >= ? AND sha256 < ? AND rl_fs_t < ?
                       ORDER BY sha256"""
            cursor = txn.cursor()
            rows.extend(row for row in db.execute(query, (low, high, TRAIN_VALIDATION_SPLIT))
                        if cursor.set_key(row[0].encode()))
            if len(rows) >= size * 2:
                break
    frame = pd.DataFrame(rows, columns=["sha256", "label", "month"])
    write_samples(frame, size, seed, output)


def write_samples(frame, size, seed, output):
    frame = frame.sample(frac=1, random_state=seed).reset_index(drop=True)
    if len(frame) < size:
        raise ValueError("insufficient eligible vectors")
    output.mkdir(parents=True, exist_ok=True)
    frame.head(size).to_csv(output / "random.csv", index=False)
    groups = frame.groupby(["label", "month"])
    quotas = groups.size() * size / len(frame)
    counts = quotas.astype(int)
    remainder = (quotas - counts).nlargest(size - counts.sum()).index
    counts.loc[remainder] += 1
    selected = pd.concat([group.head(counts[key]) for key, group in groups]).sort_index()
    selected.to_csv(output / "stratified.csv", index=False)
    pd.concat([groups.size().rename("eligible_pool"), counts.rename("selected")], axis=1).to_csv(output / "quotas.csv")


if __name__ == "__main__":
    if ACTIVE_EXPERIMENT.parent:
        write_samples(pd.read_csv(ACTIVE_EXPERIMENT.parent), ACTIVE_EXPERIMENT.size, ACTIVE_EXPERIMENT.seed, ACTIVE_EXPERIMENT.work)
    else:
        build_manifests(ACTIVE_EXPERIMENT.size, ACTIVE_EXPERIMENT.seed, ACTIVE_EXPERIMENT.work)
