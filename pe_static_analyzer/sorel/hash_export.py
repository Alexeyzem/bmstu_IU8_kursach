#!/usr/bin/env python3
"""Export SOREL SHA-256 hashes from meta.db for gold exact-dedup."""

from __future__ import annotations

from dataclasses import dataclass
import json
import sqlite3
import time
from pathlib import Path


@dataclass
class HashExportSettings:
    meta_db: Path = Path('datasets/sorel20m/meta.db')
    output: Path = Path('outputs/dedup/sorel_hashes.sha256.txt')
    summary: Path = Path('outputs/dedup/sorel_hashes_summary.json')


def main() -> int:
    settings = HashExportSettings()
    settings.output.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(f"file:{settings.meta_db}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    started = time.time()
    count = 0
    with settings.output.open("w", encoding="utf-8") as handle:
        for (sha256,) in conn.execute("SELECT sha256 FROM meta ORDER BY sha256"):
            handle.write(str(sha256).lower())
            handle.write("\n")
            count += 1
    conn.close()
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "meta_db": str(settings.meta_db),
        "output": str(settings.output),
        "hash_count": count,
        "runtime_seconds": round(time.time() - started, 3),
    }
    settings.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
