#!/usr/bin/env python3
"""Exact-deduplicate gold metadata against SOREL hashes and inside gold.

Near-dedup is intentionally conservative here: malicious samples are stored as
password-protected MalwareBazaar ZIPs and must not be extracted automatically.
Without normalized raw bytes for both SOREL and gold, byte-level >90% comparison
against SOREL cannot be completed safely. The script records that limitation and
still produces the planned exact-dedup/final metadata artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import time
from collections import Counter
from pathlib import Path

from pe_static_analyzer.gold.io import load_metadata
from typing import Iterable

FIELDNAMES = [
    "sample_id",
    "label",
    "source",
    "sha256",
    "stored_sha256",
    "stored_path",
    "size_bytes",
    "first_seen",
    "file_type",
    "imphash",
    "tlsh",
    "tags",
    "source_url",
    "collected_at",
    "dedup_status",
    "dedup_reason",
    "dedup_similarity_max",
    "matched_sha256",
]


@dataclass
class GoldDeduplicationSettings:
    metadata: Path = Path('datasets/gold_pe/metadata.csv')
    sorel_hashes: Path = Path('outputs/dedup/sorel_hashes.sha256.txt')
    output_dir: Path = Path('outputs/dedup')
    gold_output_dir: Path = Path('outputs/gold')


def read_hashes(path: Path) -> set[str]:
    hashes: set[str] = set()
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            value = line.strip().split(",", 1)[0].lower()
            if len(value) == 64:
                hashes.add(value)
    return hashes


def write_rows(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else FIELDNAMES
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalized_row(row: dict[str, str], sample_id: int, status: str, reason: str = "", matched: str = "", similarity: str = "") -> dict[str, object]:
    stored_path = Path(row.get("stored_path", ""))
    size = stored_path.stat().st_size if stored_path.exists() else ""
    label_value = row.get("label", "")
    if label_value == "malicious":
        label = 1
    elif label_value == "benign":
        label = 0
    else:
        label = label_value
    return {
        "sample_id": sample_id,
        "label": label,
        "source": row.get("source", ""),
        "sha256": (row.get("sha256") or "").lower(),
        "stored_sha256": (row.get("stored_sha256") or "").lower(),
        "stored_path": row.get("stored_path", ""),
        "size_bytes": size,
        "first_seen": row.get("first_seen", ""),
        "file_type": row.get("file_type", ""),
        "imphash": row.get("imphash", ""),
        "tlsh": row.get("tlsh", ""),
        "tags": row.get("tags", ""),
        "source_url": row.get("source_url", ""),
        "collected_at": row.get("collected_at", ""),
        "dedup_status": status,
        "dedup_reason": reason,
        "dedup_similarity_max": similarity,
        "matched_sha256": matched,
    }


def main() -> int:
    settings = GoldDeduplicationSettings()
    started = time.time()
    rows = load_metadata(settings.metadata)
    sorel_hashes = read_hashes(settings.sorel_hashes)

    before_rows = [normalized_row(row, idx, "before_dedup") for idx, row in enumerate(rows, start=1)]
    write_rows(settings.output_dir / "gold_before_dedup.csv", before_rows, FIELDNAMES)

    seen_gold: dict[str, dict[str, str]] = {}
    exact_removed: list[dict[str, object]] = []
    after_exact: list[dict[str, object]] = []
    kept_original_rows: list[dict[str, str]] = []

    for row in rows:
        sha = (row.get("sha256") or "").lower()
        if sha in sorel_hashes:
            exact_removed.append(normalized_row(row, len(exact_removed) + 1, "removed", "exact_overlap_sorel", sha, "1.0"))
            continue
        if sha in seen_gold:
            exact_removed.append(normalized_row(row, len(exact_removed) + 1, "removed", "exact_duplicate_inside_gold", sha, "1.0"))
            continue
        seen_gold[sha] = row
        kept_original_rows.append(row)
        after_exact.append(normalized_row(row, len(after_exact) + 1, "kept"))

    write_rows(settings.output_dir / "gold_exact_removed.csv", exact_removed, FIELDNAMES)
    write_rows(settings.output_dir / "gold_after_exact.csv", after_exact, FIELDNAMES)

    # No safe byte-level SOREL-vs-gold near-dedup can be completed with current artifacts:
    # SOREL raw bytes are not downloaded, and malicious gold bytes are password-protected ZIPs.
    near_candidates: list[dict[str, object]] = []
    near_removed: list[dict[str, object]] = []
    write_rows(settings.output_dir / "gold_near_duplicate_candidates.csv", near_candidates, ["gold_sha256", "candidate_sha256", "candidate_source", "reason"])
    write_rows(settings.output_dir / "gold_near_removed.csv", near_removed, FIELDNAMES)

    final_rows = [normalized_row(row, idx, "kept", "", "", "0") for idx, row in enumerate(kept_original_rows, start=1)]
    write_rows(settings.output_dir / "gold_final.csv", final_rows, FIELDNAMES)
    settings.gold_output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(settings.gold_output_dir / "gold_final_metadata.csv", final_rows, FIELDNAMES)

    initial_counts = Counter(row.get("label", "") for row in rows)
    final_counts = Counter("malicious" if row["label"] == 1 else "benign" if row["label"] == 0 else str(row["label"]) for row in final_rows)
    removal_reasons = Counter(str(row["dedup_reason"]) for row in exact_removed)
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": str(settings.metadata),
        "sorel_hashes": str(settings.sorel_hashes),
        "sorel_hash_count": len(sorel_hashes),
        "gold_initial_total": len(rows),
        "gold_initial_positive": initial_counts.get("malicious", 0),
        "gold_initial_negative": initial_counts.get("benign", 0),
        "exact_overlap_sorel": removal_reasons.get("exact_overlap_sorel", 0),
        "exact_duplicate_inside_gold": removal_reasons.get("exact_duplicate_inside_gold", 0),
        "after_exact_total": len(after_exact),
        "after_exact_positive": final_counts.get("malicious", 0),
        "after_exact_negative": final_counts.get("benign", 0),
        "near_duplicate_candidates": 0,
        "near_duplicate_removed": 0,
        "gold_final_total": len(final_rows),
        "gold_final_positive": final_counts.get("malicious", 0),
        "gold_final_negative": final_counts.get("benign", 0),
        "near_dedup_status": "not_performed_raw_byte_comparison_unavailable",
        "near_dedup_limitation": "SOREL raw PE bytes are not downloaded; malicious gold samples are stored as password-protected ZIPs and are not extracted automatically. Exact hash dedup is complete; byte-level >90% near-dedup requires a separate safe raw-byte extraction/candidate-generation step.",
        "runtime_seconds": round(time.time() - started, 3),
        "outputs": {
            "gold_before_dedup": str(settings.output_dir / "gold_before_dedup.csv"),
            "gold_exact_removed": str(settings.output_dir / "gold_exact_removed.csv"),
            "gold_after_exact": str(settings.output_dir / "gold_after_exact.csv"),
            "gold_near_duplicate_candidates": str(settings.output_dir / "gold_near_duplicate_candidates.csv"),
            "gold_near_removed": str(settings.output_dir / "gold_near_removed.csv"),
            "gold_final": str(settings.output_dir / "gold_final.csv"),
            "gold_final_metadata": str(settings.gold_output_dir / "gold_final_metadata.csv"),
        },
    }
    (settings.output_dir / "dedup_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
