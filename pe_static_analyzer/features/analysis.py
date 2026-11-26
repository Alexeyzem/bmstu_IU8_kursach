"""Select EMBER columns from a sample of the SOREL training period."""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from pe_static_analyzer.features.ember import FEATURE_DIM
from pe_static_analyzer.features.schema import build_feature_schema
from pe_static_analyzer.features.statistics import (
    binary_mutual_information,
    choose_correlated_removals,
    compute_high_correlations,
    target_point_biserial,
    write_correlation_matrix_parquet,
)
from pe_static_analyzer.sorel.vector_sampling import sample_vectors


@dataclass
class FeatureAnalysisSettings:
    meta_db: Path = Path('datasets/sorel20m/meta.db')
    lmdb_dir: Path = Path('datasets/sorel20m/ember_features')
    output_dir: Path = Path('outputs/features')
    sample_size: int = 100_000
    split: str = 'train'
    seed: int = 20261107
    max_attempts: int = 200000
    correlation_threshold: float = 0.95
    near_constant_zero_fraction: float = 0.999
    high_zero_fraction: float = 0.995
    low_mi_threshold: float = 0.0001
    mi_bins: int = 16
    correlation_block_size: int = 256
    high_correlation_output_limit: int = 200000
    postprocess: str = 'sorel_signed_log'


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def feature_statistics(settings, features, x, y):
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
        mi[idx] = binary_mutual_information(values, y, settings.mi_bins)
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
        settings.output_dir / "feature_statistics.csv",
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
        settings.output_dir / "target_association.csv",
        ["index", "name", "group", "mutual_information_bits", "target_point_biserial", "abs_target_point_biserial"],
        assoc_rows,
    )
    return x_for_stats, std, zero_fraction, mi, target_corr, assoc_rows


def removal_reasons(settings, std, zero_fraction, mi, all_high_pairs):
    constant = {int(i) for i in np.where((std == 0.0) | ~np.isfinite(std))[0]}
    near_constant = {
        int(i)
        for i in np.where((zero_fraction >= settings.near_constant_zero_fraction) & (mi <= settings.low_mi_threshold))[0]
        if int(i) not in constant
    }
    high_zero_low_mi = {
        int(i)
        for i in np.where((zero_fraction >= settings.high_zero_fraction) & (mi <= settings.low_mi_threshold))[0]
        if int(i) not in constant and int(i) not in near_constant
    }
    correlated = choose_correlated_removals(all_high_pairs, mi, zero_fraction) - constant - near_constant - high_zero_low_mi

    removal_reason: dict[int, str] = {}
    for idx in sorted(constant):
        removal_reason[idx] = "constant"
    for idx in sorted(near_constant):
        removal_reason[idx] = f"near_constant_zero_fraction>={settings.near_constant_zero_fraction}_and_mi<={settings.low_mi_threshold}"
    for idx in sorted(high_zero_low_mi):
        removal_reason[idx] = f"high_zero_fraction>={settings.high_zero_fraction}_and_mi<={settings.low_mi_threshold}"
    for idx in sorted(correlated):
        removal_reason[idx] = f"correlated_abs_pearson>={settings.correlation_threshold}_lower_mi_or_more_sparse"
    return removal_reason


def main() -> int:
    settings = FeatureAnalysisSettings()
    started = time.time()
    features = build_feature_schema()
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    schema_rows = [{"index": f.index, "name": f.name, "group": f.group} for f in features]
    (settings.output_dir / "feature_schema.json").write_text(json.dumps(schema_rows, indent=2), encoding="utf-8")

    x, y, hashes, sample_stats = sample_vectors(
        meta_db=settings.meta_db,
        lmdb_dir=settings.lmdb_dir,
        sample_size=settings.sample_size,
        seed=settings.seed,
        max_attempts=settings.max_attempts,
        postprocess=settings.postprocess,
        split=settings.split,
    )
    write_csv(
        settings.output_dir / "sorel_feature_sample.csv",
        ["sample_id", "sha256", "is_malware"],
        ({"sample_id": i, "sha256": sha, "is_malware": int(y[i])} for i, sha in enumerate(hashes)),
    )

    x_for_stats, std, zero_fraction, mi, target_corr, assoc_rows = feature_statistics(
        settings, features, x, y
    )

    all_high_pairs, high_pairs, total_high_correlation_pairs = compute_high_correlations(
        x_for_stats,
        features,
        std,
        settings.correlation_threshold,
        settings.correlation_block_size,
        settings.high_correlation_output_limit,
    )
    write_csv(
        settings.output_dir / "high_correlation_pairs.csv",
        ["feature_i", "name_i", "group_i", "feature_j", "name_j", "group_j", "pearson"],
        high_pairs,
    )
    correlation_matrix_path = settings.output_dir / "correlation_matrix.parquet"
    write_correlation_matrix_parquet(correlation_matrix_path, x_for_stats, features, std)

    removal_reason = removal_reasons(settings, std, zero_fraction, mi, all_high_pairs)

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
        settings.output_dir / "removed_features.csv",
        ["index", "name", "group", "reason", "std", "zero_fraction", "mutual_information_bits", "target_point_biserial"],
        removed_rows,
    )

    kept = [f for f in features if f.index not in removal_reason]
    selected_payload = {
        "source": "sampled SOREL ember_features LMDB",
        "sample_size": int(x.shape[0]),
        "seed": settings.seed,
        "feature_dim_original": FEATURE_DIM,
        "feature_dim_selected": len(kept),
        "removed_count": len(removal_reason),
        "postprocess": settings.postprocess,
        "split": settings.split,
        "rules": {
            "constant": "std == 0 on sampled vectors",
            "near_constant": f"zero_fraction >= {settings.near_constant_zero_fraction} and mutual_information_bits <= {settings.low_mi_threshold}",
            "high_zero_low_mi": f"zero_fraction >= {settings.high_zero_fraction} and mutual_information_bits <= {settings.low_mi_threshold}",
            "correlated": f"abs(Pearson) >= {settings.correlation_threshold}; keep higher MI, then less sparse, then lower index",
            "leakage": "EMBER vector excludes filename/source/path/download_url; those fields remain metadata only",
        },
        "selected_features": [{"index": f.index, "name": f.name, "group": f.group} for f in kept],
    }
    (settings.output_dir / "selected_features.json").write_text(json.dumps(selected_payload, indent=2), encoding="utf-8")

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
        "meta_db": str(settings.meta_db),
        "lmdb_dir": str(settings.lmdb_dir),
        "postprocess": settings.postprocess,
        "split": settings.split,
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
        "high_correlation_pairs_output_limit": settings.high_correlation_output_limit,
        "correlation_threshold": settings.correlation_threshold,
        "top_target_associations": assoc_rows[:20],
        "outputs": {
            "feature_schema": str(settings.output_dir / "feature_schema.json"),
            "feature_statistics": str(settings.output_dir / "feature_statistics.csv"),
            "target_association": str(settings.output_dir / "target_association.csv"),
            "high_correlation_pairs": str(settings.output_dir / "high_correlation_pairs.csv"),
            "removed_features": str(settings.output_dir / "removed_features.csv"),
            "correlation_matrix": str(correlation_matrix_path),
            "selected_features": str(settings.output_dir / "selected_features.json"),
            "sorel_feature_sample": str(settings.output_dir / "sorel_feature_sample.csv"),
        },
    }
    (settings.output_dir / "feature_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
