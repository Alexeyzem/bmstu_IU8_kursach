"""Shared SOREL LMDB feature loading utilities."""

from __future__ import annotations

import csv
import json
import zlib
from pathlib import Path
from typing import Iterator

import lmdb
import msgpack
import numpy as np
import pandas as pd

from pe_static_analyzer.features.ember import FEATURE_DIM, sorel_signed_log_postprocess

TRAIN_VALIDATION_SPLIT = 1543542570.0
VALIDATION_TEST_SPLIT = 1547279640.0


def load_selected_indices(path: Path | None) -> list[int]:
    """Load selected EMBER feature indices, or all indices when no file is supplied."""
    if path is None:
        return list(range(FEATURE_DIM))
    data = json.loads(path.read_text(encoding="utf-8"))
    return [int(item["index"]) for item in data["selected_features"]]


def unpack_feature_vector(blob: bytes, postprocess: str = "sorel_signed_log") -> np.ndarray:
    """Decode one SOREL msgpack+zlib EMBER vector from LMDB."""
    unpacked = msgpack.unpackb(zlib.decompress(blob), raw=False, strict_map_key=False)
    vector = unpacked.get(0) if isinstance(unpacked, dict) else unpacked
    if vector is None and isinstance(unpacked, dict):
        vector = unpacked.get("0")
    arr = np.asarray(vector, dtype=np.float32)
    if arr.shape != (FEATURE_DIM,):
        raise ValueError(f"feature vector shape {arr.shape}; expected ({FEATURE_DIM},)")
    if postprocess == "sorel_signed_log":
        return sorel_signed_log_postprocess(arr)
    if postprocess == "raw":
        return arr
    raise ValueError(f"unknown postprocess policy: {postprocess}")


def iter_manifest(path: Path, limit: int | None = None) -> Iterator[tuple[str, int]]:
    count = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            yield row["sha256"], int(row["label"])
            count += 1
            if limit is not None and count >= limit:
                return


def feature_batches(manifest: Path, env: lmdb.Environment, indices: np.ndarray, batch_size: int, limit: int | None = None) -> Iterator[tuple[np.ndarray, np.ndarray, dict[str, int]]]:
    """Stream selected SOREL features in bounded dense batches."""
    x_batch: list[np.ndarray] = []
    y_batch: list[int] = []
    skipped_missing = 0
    skipped_bad = 0
    total_seen = 0
    with env.begin(buffers=True) as txn:
        for sha256, label in iter_manifest(manifest, limit):
            total_seen += 1
            blob = txn.get(sha256.encode("ascii"))
            if blob is None:
                skipped_missing += 1
                continue
            try:
                vector = unpack_feature_vector(bytes(blob))[indices]
            except Exception:
                skipped_bad += 1
                continue
            x_batch.append(vector)
            y_batch.append(label)
            if len(x_batch) >= batch_size:
                yield np.vstack(x_batch).astype(np.float32), np.asarray(y_batch, dtype=np.int8), {"seen": total_seen, "skipped_missing": skipped_missing, "skipped_bad": skipped_bad}
                x_batch.clear()
                y_batch.clear()
        if x_batch:
            yield np.vstack(x_batch).astype(np.float32), np.asarray(y_batch, dtype=np.int8), {"seen": total_seen, "skipped_missing": skipped_missing, "skipped_bad": skipped_bad}


def collect_eval(manifest: Path, env: lmdb.Environment, indices: np.ndarray, batch_size: int, limit: int | None) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    stats = {"seen": 0, "skipped_missing": 0, "skipped_bad": 0}
    for x, y, batch_stats in feature_batches(manifest, env, indices, batch_size, limit):
        xs.append(x)
        ys.append(y)
        stats = batch_stats
    if not xs:
        return np.empty((0, len(indices)), dtype=np.float32), np.empty((0,), dtype=np.int8), stats
    return np.vstack(xs).astype(np.float32), np.concatenate(ys), stats


def split_where_clause(split: str) -> str:
    if split == "all":
        return "1 = 1"
    if split == "train":
        return f"rl_fs_t < {TRAIN_VALIDATION_SPLIT}"
    if split == "validation":
        return f"rl_fs_t >= {TRAIN_VALIDATION_SPLIT} AND rl_fs_t < {VALIDATION_TEST_SPLIT}"
    if split == "test":
        return f"rl_fs_t >= {VALIDATION_TEST_SPLIT}"
    raise ValueError(f"unknown split: {split}")


def load_split(path, env, indices, limit=None, offset=0):
    rows = pd.read_csv(path, nrows=limit, skiprows=range(1, offset + 1))
    matrix = np.empty((len(rows), len(indices)), dtype=np.float32)
    kept = []
    with env.begin() as txn:
        for i, sha in enumerate(rows.sha256):
            blob = txn.get(sha.encode())
            if blob is not None:
                matrix[len(kept)] = unpack_feature_vector(blob)[indices]
                kept.append(i)
    return matrix[:len(kept)], rows.iloc[kept].reset_index(drop=True)
