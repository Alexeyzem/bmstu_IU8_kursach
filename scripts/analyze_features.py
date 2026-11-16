#!/usr/bin/env python3
"""Analyze SOREL/EMBER feature sparsity, target association, and correlations.

The script samples feature vectors from SOREL's processed LMDB without touching PE
binaries. It is intentionally sample-based: the full SOREL LMDB is ~72 GiB and a
full dense 2381x2381 correlation pass over all rows is unnecessary for coursework
feature-reduction decisions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import lmdb
import msgpack
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq



def sorel_signed_log_postprocess(x: np.ndarray) -> np.ndarray:
    """Match SOREL's official Dataset.features_postproc_func."""
    out = x.astype(np.float32, copy=True)
    negative = out < 0
    positive = out > 0
    out[negative] = -np.log1p(-out[negative])
    out[positive] = np.log1p(out[positive])
    return out

FEATURE_DIM = 2381
DATA_DIRECTORIES = [
    "EXPORT_TABLE",
    "IMPORT_TABLE",
    "RESOURCE_TABLE",
    "EXCEPTION_TABLE",
    "CERTIFICATE_TABLE",
    "BASE_RELOCATION_TABLE",
    "DEBUG",
    "ARCHITECTURE",
    "GLOBAL_PTR",
    "TLS_TABLE",
    "LOAD_CONFIG_TABLE",
    "BOUND_IMPORT",
    "IAT",
    "DELAY_IMPORT_DESCRIPTOR",
    "CLR_RUNTIME_HEADER",
]

TRAIN_VALIDATION_SPLIT = 1543542570.0
VALIDATION_TEST_SPLIT = 1547279640.0


def split_where_clause(split: str) -> str:
    if split == "all":
        return "1 = 1"
    if split == "train":
        return f"rl_fs_t <= {TRAIN_VALIDATION_SPLIT}"
    if split == "validation":
        return f"rl_fs_t >= {TRAIN_VALIDATION_SPLIT} AND rl_fs_t < {VALIDATION_TEST_SPLIT}"
    if split == "test":
        return f"rl_fs_t >= {VALIDATION_TEST_SPLIT}"
    raise ValueError(f"unknown split: {split}")


@dataclass(frozen=True)
class Feature:
    index: int
    name: str
    group: str


def numbered(prefix: str, count: int) -> list[str]:
    width = max(2, len(str(count - 1)))
    return [f"{prefix}_{i:0{width}d}" for i in range(count)]


def build_feature_schema() -> list[Feature]:
    groups: list[tuple[str, list[str]]] = []
    groups.append(("histogram", numbered("histogram_byte", 256)))
    groups.append(("byteentropy", numbered("byteentropy_bin", 256)))
    groups.append((
        "strings",
        ["strings_numstrings", "strings_avlength", "strings_printables"]
        + numbered("strings_printabledist", 96)
        + ["strings_entropy", "strings_paths", "strings_urls", "strings_registry", "strings_mz"],
    ))
    groups.append((
        "general",
        [
            "general_size",
            "general_vsize",
            "general_has_debug",
            "general_exports",
            "general_imports",
            "general_has_relocations",
            "general_has_resources",
            "general_has_signature",
            "general_has_tls",
            "general_symbols",
        ],
    ))
    groups.append((
        "header",
        ["header_coff_timestamp"]
        + numbered("header_coff_machine_hash", 10)
        + numbered("header_coff_characteristics_hash", 10)
        + numbered("header_optional_subsystem_hash", 10)
        + numbered("header_optional_dll_characteristics_hash", 10)
        + numbered("header_optional_magic_hash", 10)
        + [
            "header_major_image_version",
            "header_minor_image_version",
            "header_major_linker_version",
            "header_minor_linker_version",
            "header_major_operating_system_version",
            "header_minor_operating_system_version",
            "header_major_subsystem_version",
            "header_minor_subsystem_version",
            "header_sizeof_code",
            "header_sizeof_headers",
            "header_sizeof_heap_commit",
        ],
    ))
    groups.append((
        "section",
        [
            "section_count",
            "section_zero_size_count",
            "section_empty_name_count",
            "section_rx_count",
            "section_w_count",
        ]
        + numbered("section_size_hash", 50)
        + numbered("section_entropy_hash", 50)
        + numbered("section_vsize_hash", 50)
        + numbered("section_entry_name_hash", 50)
        + numbered("section_entry_characteristics_hash", 50),
    ))
    groups.append(("imports", numbered("imports_libraries_hash", 256) + numbered("imports_functions_hash", 1024)))
    groups.append(("exports", numbered("exports_hash", 128)))
    data_dir_names: list[str] = []
    for name in DATA_DIRECTORIES:
        lower = name.lower()
        data_dir_names.extend([f"datadirectories_{lower}_size", f"datadirectories_{lower}_rva"])
    groups.append(("datadirectories", data_dir_names))

    features: list[Feature] = []
    for group, names in groups:
        for name in names:
            features.append(Feature(len(features), name, group))
    if len(features) != FEATURE_DIM:
        raise RuntimeError(f"feature schema has {len(features)} entries; expected {FEATURE_DIM}")
    return features


def stable_pivot(seed: int, label: int, attempt: int) -> str:
    payload = f"{seed}:{label}:{attempt}".encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def query_one_sha(conn: sqlite3.Connection, label: int, pivot: str, split_clause: str) -> tuple[str, int] | None:
    row = conn.execute(
        f"""
        SELECT sha256, is_malware
        FROM meta
        WHERE is_malware = ? AND sha256 >= ? AND {split_clause}
        ORDER BY sha256
        LIMIT 1
        """,
        (label, pivot),
    ).fetchone()
    if row is not None:
        return str(row[0]), int(row[1])
    row = conn.execute(
        f"""
        SELECT sha256, is_malware
        FROM meta
        WHERE is_malware = ? AND {split_clause}
        ORDER BY sha256
        LIMIT 1
        """,
        (label,),
    ).fetchone()
    if row is None:
        return None
    return str(row[0]), int(row[1])


def unpack_feature_vector(blob: bytes, postprocess: str) -> np.ndarray:
    unpacked = msgpack.unpackb(zlib.decompress(blob), raw=False, strict_map_key=False)
    if isinstance(unpacked, dict):
        vector = unpacked.get(0)
        if vector is None:
            vector = unpacked.get("0")
    else:
        vector = unpacked
    arr = np.asarray(vector, dtype=np.float32)
    if arr.shape != (FEATURE_DIM,):
        raise ValueError(f"feature vector shape {arr.shape}; expected ({FEATURE_DIM},)")
    if postprocess == "sorel_signed_log":
        return sorel_signed_log_postprocess(arr)
    if postprocess == "raw":
        return arr
    raise ValueError(f"unknown postprocess policy: {postprocess}")


def sample_vectors(
    *,
    meta_db: Path,
    lmdb_dir: Path,
    sample_size: int,
    seed: int,
    max_attempts: int,
    postprocess: str,
    split: str,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
    target_per_label = {0: sample_size // 2, 1: sample_size - sample_size // 2}
    by_label = {0: 0, 1: 0}
    rows: list[np.ndarray] = []
    labels: list[int] = []
    hashes: list[str] = []
    seen: set[str] = set()
    attempts = 0
    lmdb_missing = 0
    duplicate = 0
    split_clause = split_where_clause(split)

    conn = sqlite3.connect(f"file:{meta_db}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    env = lmdb.open(str(lmdb_dir), readonly=True, lock=False, readahead=False, max_readers=1)
    try:
        with env.begin(buffers=True) as txn:
            while sum(by_label.values()) < sample_size and attempts < max_attempts:
                label = 0 if by_label[0] < target_per_label[0] else 1
                if by_label[label] >= target_per_label[label]:
                    label = 1 - label
                pivot = stable_pivot(seed, label, attempts)
                attempts += 1
                selected = query_one_sha(conn, label, pivot, split_clause)
                if selected is None:
                    continue
                sha, actual_label = selected
                if sha in seen:
                    duplicate += 1
                    continue
                blob = txn.get(sha.encode("ascii"))
                if blob is None:
                    lmdb_missing += 1
                    seen.add(sha)
                    continue
                try:
                    vector = unpack_feature_vector(bytes(blob), postprocess)
                except Exception as exc:  # noqa: BLE001 - keep sampling while reporting count.
                    print(f"skip undecodable feature vector sha256={sha}: {exc}")
                    seen.add(sha)
                    continue
                rows.append(vector)
                labels.append(actual_label)
                hashes.append(sha)
                seen.add(sha)
                by_label[actual_label] += 1
    finally:
        env.close()
        conn.close()

    if len(rows) < sample_size:
        raise RuntimeError(
            f"sampled only {len(rows)} rows out of requested {sample_size}; "
            f"attempts={attempts}, lmdb_missing={lmdb_missing}, duplicate={duplicate}"
        )
    stats = {"attempts": attempts, "lmdb_missing": lmdb_missing, "duplicate": duplicate, "label_0": by_label[0], "label_1": by_label[1]}
    return np.vstack(rows).astype(np.float32), np.asarray(labels, dtype=np.int8), hashes, stats


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


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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
    output_pairs: list[dict[str, object]] = []
    total_pairs = 0
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
                output_pairs.append(pair)
                total_pairs += 1
        if len(output_pairs) > output_limit * 4:
            output_pairs.sort(key=lambda row: abs(float(row["pearson"])), reverse=True)
            output_pairs = output_pairs[:output_limit]
    all_pairs.sort(key=lambda row: abs(float(row["pearson"])), reverse=True)
    output_pairs.sort(key=lambda row: abs(float(row["pearson"])), reverse=True)
    return all_pairs, output_pairs[:output_limit], total_pairs


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-db", type=Path, default=Path("datasets/sorel20m/meta.db"))
    parser.add_argument("--lmdb-dir", type=Path, default=Path("datasets/sorel20m/ember_features"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/features"))
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="train", help="Metadata split used for sampling; feature selection should use train.")
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--max-attempts", type=int, default=200000)
    parser.add_argument("--correlation-threshold", type=float, default=0.95)
    parser.add_argument("--near-constant-zero-fraction", type=float, default=0.999)
    parser.add_argument("--high-zero-fraction", type=float, default=0.995)
    parser.add_argument("--low-mi-threshold", type=float, default=1e-4)
    parser.add_argument("--mi-bins", type=int, default=16)
    parser.add_argument("--correlation-block-size", type=int, default=256)
    parser.add_argument("--high-correlation-output-limit", type=int, default=200000)
    parser.add_argument("--postprocess", choices=["sorel_signed_log", "raw"], default="sorel_signed_log", help="Feature scale policy; default matches official SOREL Dataset loader.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    features = build_feature_schema()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    schema_rows = [{"index": f.index, "name": f.name, "group": f.group} for f in features]
    (args.output_dir / "feature_schema.json").write_text(json.dumps(schema_rows, indent=2), encoding="utf-8")

    x, y, hashes, sample_stats = sample_vectors(
        meta_db=args.meta_db,
        lmdb_dir=args.lmdb_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        max_attempts=args.max_attempts,
        postprocess=args.postprocess,
        split=args.split,
    )
    write_csv(
        args.output_dir / "sorel_feature_sample.csv",
        ["sample_id", "sha256", "is_malware"],
        ({"sample_id": i, "sha256": sha, "is_malware": int(y[i])} for i, sha in enumerate(hashes)),
    )

    finite = np.isfinite(x)
    missing_fraction = 1.0 - finite.mean(axis=0)
    x_for_stats = np.where(finite, x, 0.0)
    mean = x_for_stats.mean(axis=0)
    std = x_for_stats.std(axis=0, ddof=1)
    min_value = x_for_stats.min(axis=0)
    max_value = x_for_stats.max(axis=0)
    zero_fraction = (x_for_stats == 0.0).mean(axis=0)
    nonzero_fraction = 1.0 - zero_fraction

    mi = np.zeros(FEATURE_DIM, dtype=np.float64)
    target_corr = np.zeros(FEATURE_DIM, dtype=np.float64)
    for idx in range(FEATURE_DIM):
        values = x_for_stats[:, idx]
        mi[idx] = binary_mutual_information(values, y, args.mi_bins)
        target_corr[idx] = target_point_biserial(values, y, float(std[idx]))

    stats_rows = []
    for f in features:
        stats_rows.append(
            {
                "index": f.index,
                "name": f.name,
                "group": f.group,
                "mean": float(mean[f.index]),
                "std": float(std[f.index]),
                "min": float(min_value[f.index]),
                "max": float(max_value[f.index]),
                "missing_fraction": float(missing_fraction[f.index]),
                "zero_fraction": float(zero_fraction[f.index]),
                "nonzero_fraction": float(nonzero_fraction[f.index]),
            }
        )
    write_csv(
        args.output_dir / "feature_statistics.csv",
        ["index", "name", "group", "mean", "std", "min", "max", "missing_fraction", "zero_fraction", "nonzero_fraction"],
        stats_rows,
    )

    assoc_rows = [
        {
            "index": f.index,
            "name": f.name,
            "group": f.group,
            "mutual_information_bits": float(mi[f.index]),
            "target_point_biserial": float(target_corr[f.index]),
            "abs_target_point_biserial": abs(float(target_corr[f.index])),
        }
        for f in features
    ]
    assoc_rows.sort(key=lambda row: (float(row["mutual_information_bits"]), float(row["abs_target_point_biserial"])), reverse=True)
    write_csv(
        args.output_dir / "target_association.csv",
        ["index", "name", "group", "mutual_information_bits", "target_point_biserial", "abs_target_point_biserial"],
        assoc_rows,
    )

    all_high_pairs, high_pairs, total_high_correlation_pairs = compute_high_correlations(
        x_for_stats,
        features,
        std,
        args.correlation_threshold,
        args.correlation_block_size,
        args.high_correlation_output_limit,
    )
    write_csv(
        args.output_dir / "high_correlation_pairs.csv",
        ["feature_i", "name_i", "group_i", "feature_j", "name_j", "group_j", "pearson"],
        high_pairs,
    )
    correlation_matrix_path = args.output_dir / "correlation_matrix.parquet"
    write_correlation_matrix_parquet(correlation_matrix_path, x_for_stats, features, std)

    constant = {int(i) for i in np.where((std == 0.0) | ~np.isfinite(std))[0]}
    near_constant = {
        int(i)
        for i in np.where((zero_fraction >= args.near_constant_zero_fraction) & (mi <= args.low_mi_threshold))[0]
        if int(i) not in constant
    }
    high_zero_low_mi = {
        int(i)
        for i in np.where((zero_fraction >= args.high_zero_fraction) & (mi <= args.low_mi_threshold))[0]
        if int(i) not in constant and int(i) not in near_constant
    }
    correlated = choose_correlated_removals(all_high_pairs, mi, zero_fraction) - constant - near_constant - high_zero_low_mi

    removal_reason: dict[int, str] = {}
    for idx in sorted(constant):
        removal_reason[idx] = "constant"
    for idx in sorted(near_constant):
        removal_reason[idx] = f"near_constant_zero_fraction>={args.near_constant_zero_fraction}_and_mi<={args.low_mi_threshold}"
    for idx in sorted(high_zero_low_mi):
        removal_reason[idx] = f"high_zero_fraction>={args.high_zero_fraction}_and_mi<={args.low_mi_threshold}"
    for idx in sorted(correlated):
        removal_reason[idx] = f"correlated_abs_pearson>={args.correlation_threshold}_lower_mi_or_more_sparse"

    removed_rows = []
    for idx, reason in sorted(removal_reason.items()):
        f = features[idx]
        removed_rows.append(
            {
                "index": idx,
                "name": f.name,
                "group": f.group,
                "reason": reason,
                "std": float(std[idx]),
                "zero_fraction": float(zero_fraction[idx]),
                "mutual_information_bits": float(mi[idx]),
                "target_point_biserial": float(target_corr[idx]),
            }
        )
    write_csv(
        args.output_dir / "removed_features.csv",
        ["index", "name", "group", "reason", "std", "zero_fraction", "mutual_information_bits", "target_point_biserial"],
        removed_rows,
    )

    kept = [f for f in features if f.index not in removal_reason]
    selected_payload = {
        "source": "sampled SOREL ember_features LMDB",
        "sample_size": int(x.shape[0]),
        "seed": args.seed,
        "feature_dim_original": FEATURE_DIM,
        "feature_dim_selected": len(kept),
        "removed_count": len(removal_reason),
        "postprocess": args.postprocess,
        "split": args.split,
        "rules": {
            "constant": "std == 0 on sampled vectors",
            "near_constant": f"zero_fraction >= {args.near_constant_zero_fraction} and mutual_information_bits <= {args.low_mi_threshold}",
            "high_zero_low_mi": f"zero_fraction >= {args.high_zero_fraction} and mutual_information_bits <= {args.low_mi_threshold}",
            "correlated": f"abs(Pearson) >= {args.correlation_threshold}; keep higher MI, then less sparse, then lower index",
            "leakage": "EMBER vector excludes filename/source/path/download_url; those fields remain metadata only",
        },
        "selected_features": [{"index": f.index, "name": f.name, "group": f.group} for f in kept],
    }
    (args.output_dir / "selected_features.json").write_text(json.dumps(selected_payload, indent=2), encoding="utf-8")

    group_summary: dict[str, dict[str, int]] = {}
    for f in features:
        item = group_summary.setdefault(f.group, {"original": 0, "removed": 0, "selected": 0})
        item["original"] += 1
        if f.index in removal_reason:
            item["removed"] += 1
        else:
            item["selected"] += 1

    reason_counts: dict[str, int] = {}
    for reason in removal_reason.values():
        reason_key = reason.split("_and_")[0]
        reason_counts[reason_key] = reason_counts.get(reason_key, 0) + 1

    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.time() - started, 3),
        "meta_db": str(args.meta_db),
        "lmdb_dir": str(args.lmdb_dir),
        "postprocess": args.postprocess,
        "split": args.split,
        "sample_stats": sample_stats,
        "sample_size": int(x.shape[0]),
        "label_counts": {"benign_0": int((y == 0).sum()), "malware_1": int((y == 1).sum())},
        "feature_dim_original": FEATURE_DIM,
        "feature_dim_selected": len(kept),
        "removed_count": len(removal_reason),
        "reason_counts": reason_counts,
        "group_summary": group_summary,
        "high_correlation_pairs": len(high_pairs),
        "total_high_correlation_pairs": total_high_correlation_pairs,
        "high_correlation_pairs_output_limit": args.high_correlation_output_limit,
        "correlation_threshold": args.correlation_threshold,
        "top_target_associations": assoc_rows[:20],
        "outputs": {
            "feature_schema": str(args.output_dir / "feature_schema.json"),
            "feature_statistics": str(args.output_dir / "feature_statistics.csv"),
            "target_association": str(args.output_dir / "target_association.csv"),
            "high_correlation_pairs": str(args.output_dir / "high_correlation_pairs.csv"),
            "removed_features": str(args.output_dir / "removed_features.csv"),
            "correlation_matrix": str(correlation_matrix_path),
            "selected_features": str(args.output_dir / "selected_features.json"),
            "sorel_feature_sample": str(args.output_dir / "sorel_feature_sample.csv"),
        },
    }
    (args.output_dir / "feature_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
