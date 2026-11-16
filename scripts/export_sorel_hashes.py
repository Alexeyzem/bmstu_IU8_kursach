#!/usr/bin/env python3
"""Export SOREL SHA-256 hashes from meta.db for gold exact-dedup."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-db", type=Path, default=Path("datasets/sorel20m/meta.db"))
    parser.add_argument("--output", type=Path, default=Path("outputs/dedup/sorel_hashes.sha256.txt"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/dedup/sorel_hashes_summary.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(f"file:{args.meta_db}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    started = time.time()
    count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for (sha256,) in conn.execute("SELECT sha256 FROM meta ORDER BY sha256"):
            handle.write(str(sha256).lower())
            handle.write("\n")
            count += 1
    conn.close()
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "meta_db": str(args.meta_db),
        "output": str(args.output),
        "hash_count": count,
        "runtime_seconds": round(time.time() - started, 3),
    }
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
