"""Target association, correlations and redundant-feature selection."""
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pe_static_analyzer.features.ember import FEATURE_DIM
from pe_static_analyzer.features.schema import Feature


def binary_mutual_information(values: np.ndarray, labels: np.ndarray, bins: int) -> float:
    finite = np.isfinite(values)
    if finite.sum() == 0:
        return 0.0
    x = values[finite]
    y = labels[finite]
    if np.all(x == x[0]):
        return 0.0
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(x, quantiles))
    if len(edges) <= 2:
        threshold = edges[0]
        binned = (x > threshold).astype(np.int16)
        bin_count = 2
    else:
        binned = np.searchsorted(edges[1:-1], x, side="right").astype(np.int16)
        bin_count = len(edges) - 1
    contingency = np.zeros((bin_count, 2), dtype=np.float64)
    np.add.at(contingency, (binned, y.astype(np.int8)), 1.0)
    total = contingency.sum()
    if total == 0:
        return 0.0
    pxy = contingency / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    expected = px @ py
    mask = pxy > 0
    return float(np.sum(pxy[mask] * np.log2(pxy[mask] / expected[mask])))


def target_point_biserial(values: np.ndarray, labels: np.ndarray, std: float) -> float:
    if std == 0 or not math.isfinite(std):
        return 0.0
    y = labels.astype(np.float64)
    y_std = float(y.std(ddof=1))
    if y_std == 0.0:
        return 0.0
    corr = np.corrcoef(values.astype(np.float64), y)[0, 1]
    if not math.isfinite(corr):
        return 0.0
    return float(corr)


def compute_high_correlations(
    x: np.ndarray,
    features: list[Feature],
    std: np.ndarray,
    threshold: float,
    block_size: int,
    output_limit: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    valid = np.where(np.isfinite(std) & (std > 0))[0]
    centered = x[:, valid].astype(np.float64, copy=True)
    centered -= centered.mean(axis=0)
    denom = centered.std(axis=0, ddof=1)
    keep = denom > 0
    valid = valid[keep]
    z = centered[:, keep] / denom[keep]
    n = z.shape[0]
    all_pairs: list[dict[str, object]] = []
    for start in range(0, z.shape[1], block_size):
        stop = min(start + block_size, z.shape[1])
        corr = (z[:, start:stop].T @ z) / max(n - 1, 1)
        for local_i in range(stop - start):
            global_i_pos = start + local_i
            row = corr[local_i]
            hits = np.where(np.abs(row) >= threshold)[0]
            for global_j_pos in hits:
                if global_j_pos <= global_i_pos:
                    continue
                left = features[int(valid[global_i_pos])]
                right = features[int(valid[global_j_pos])]
                pair = {
                    "feature_i": left.index,
                    "name_i": left.name,
                    "group_i": left.group,
                    "feature_j": right.index,
                    "name_j": right.name,
                    "group_j": right.group,
                    "pearson": float(row[global_j_pos]),
                }
                all_pairs.append(pair)
    all_pairs.sort(key=lambda row: abs(float(row["pearson"])), reverse=True)
    return all_pairs, all_pairs[:output_limit], len(all_pairs)


def write_correlation_matrix_parquet(path: Path, x: np.ndarray, features: list[Feature], std: np.ndarray) -> None:
    valid = np.where(np.isfinite(std) & (std > 0))[0]
    corr = np.full((FEATURE_DIM, FEATURE_DIM), np.nan, dtype=np.float32)
    if valid.size:
        centered = x[:, valid].astype(np.float64, copy=True)
        centered -= centered.mean(axis=0)
        denom = centered.std(axis=0, ddof=1)
        keep = denom > 0
        valid = valid[keep]
        z = centered[:, keep] / denom[keep]
        subcorr = (z.T @ z) / max(z.shape[0] - 1, 1)
        corr[np.ix_(valid, valid)] = subcorr.astype(np.float32)
    arrays: dict[str, pa.Array] = {
        "feature_index": pa.array([f.index for f in features], type=pa.int32()),
        "feature_name": pa.array([f.name for f in features], type=pa.string()),
        "feature_group": pa.array([f.group for f in features], type=pa.string()),
    }
    for idx in range(FEATURE_DIM):
        arrays[f"corr_{idx:04d}"] = pa.array(corr[:, idx], type=pa.float32())
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(arrays), path, compression="zstd")


def choose_correlated_removals(
    pairs: list[dict[str, object]],
    mi: np.ndarray,
    zero_fraction: np.ndarray,
) -> set[int]:
    removed: set[int] = set()
    for pair in pairs:
        i = int(pair["feature_i"])
        j = int(pair["feature_j"])
        if i in removed or j in removed:
            continue
        if mi[i] > mi[j]:
            removed.add(j)
        elif mi[j] > mi[i]:
            removed.add(i)
        elif zero_fraction[i] < zero_fraction[j]:
            removed.add(j)
        elif zero_fraction[j] < zero_fraction[i]:
            removed.add(i)
        else:
            removed.add(max(i, j))
    return removed
