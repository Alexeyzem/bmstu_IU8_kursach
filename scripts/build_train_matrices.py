#!/usr/bin/env python3
"""Build SOREL split manifests and verify external gold feature matrix.

This does not materialize the full SOREL feature matrix by default. SOREL features
remain in LMDB and are read by SHA keys during training. Gold is an external
holdout matrix and is never included in SOREL train/validation/test.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import time
from pathlib import Path

import pyarrow.parquet as pq

TRAIN_VALIDATION_SPLIT = 1543542570.0
VALIDATION_TEST_SPLIT = 1547279640.0


def split_clause(split: str) -> str:
    if split == "train":
        return f"rl_fs_t <= {TRAIN_VALIDATION_SPLIT}"
    if split == "validation":
        return f"rl_fs_t >= {TRAIN_VALIDATION_SPLIT} AND rl_fs_t < {VALIDATION_TEST_SPLIT}"
    if split == "test":
        return f"rl_fs_t >= {VALIDATION_TEST_SPLIT}"
    raise ValueError(split)


def export_split(conn: sqlite3.Connection, split: str, output: Path, limit: int | None) -> dict[str, int | str]:
    query = f"SELECT sha256, is_malware FROM meta WHERE {split_clause(split)} ORDER BY sha256"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    positives = 0
    negatives = 0
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sha256", "label", "split"])
        writer.writeheader()
        for sha256, label in conn.execute(query):
            label = int(label)
            writer.writerow({"sha256": sha256, "label": label, "split": split})
            total += 1
            positives += int(label == 1)
            negatives += int(label == 0)
    return {"path": str(output), "total": total, "malware_1": positives, "benign_0": negatives}


def count_csv(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-db", type=Path, default=Path("datasets/sorel20m/meta.db"))
    parser.add_argument("--sorel-lmdb", type=Path, default=Path("datasets/sorel20m/ember_features"))
    parser.add_argument("--gold-features", type=Path, default=Path("outputs/features/gold_features.parquet"))
    parser.add_argument("--gold-labels", type=Path, default=Path("outputs/features/gold_labels.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/matrices"))
    parser.add_argument("--limit-per-split", type=int, default=None, help="Optional smoke-test cap; omit for full SOREL split manifests.")
    parser.add_argument("--skip-gold", action="store_true", help="Build only SOREL train/validation/test manifests; gold holdout verification is deferred until gold features exist.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    conn = sqlite3.connect(f"file:{args.meta_db}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    splits = {
        split: export_split(conn, split, args.output_dir / f"sorel_{split}_manifest.csv", args.limit_per_split)
        for split in ["train", "validation", "test"]
    }
    conn.close()

    if not args.sorel_lmdb.exists():
        raise SystemExit(f"SOREL LMDB directory missing: {args.sorel_lmdb}")
    gold_summary: dict[str, object]
    if args.skip_gold:
        gold_summary = {
            "status": "deferred",
            "reason": "gold external holdout features are intentionally extracted after SOREL model training; gold is not part of train/validation/test",
        }
    else:
        if not args.gold_features.exists():
            raise SystemExit(f"gold features missing: {args.gold_features}")
        if not args.gold_labels.exists():
            raise SystemExit(f"gold labels missing: {args.gold_labels}")
        gold_pf = pq.ParquetFile(args.gold_features)
        gold_label_rows = count_csv(args.gold_labels)
        if gold_pf.metadata.num_columns != 2384:
            raise SystemExit(f"gold feature parquet columns {gold_pf.metadata.num_columns}; expected 2384")
        if gold_pf.metadata.num_rows != gold_label_rows:
            raise SystemExit(f"gold feature rows {gold_pf.metadata.num_rows}; labels rows {gold_label_rows}")
        gold_summary = {
            "status": "verified_external_holdout",
            "features": str(args.gold_features),
            "labels": str(args.gold_labels),
            "rows": gold_pf.metadata.num_rows,
            "columns": gold_pf.metadata.num_columns,
        }

    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "policy": "SOREL train/validation/test only; gold is external holdout and is not used for training, tuning, threshold selection, or feature selection.",
        "meta_db": str(args.meta_db),
        "sorel_lmdb": str(args.sorel_lmdb),
        "splits": splits,
        "gold": gold_summary,
        "runtime_seconds": round(time.time() - started, 3),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "matrix_build_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
