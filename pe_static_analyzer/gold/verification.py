#!/usr/bin/env python3
"""Verify the defanged gold PE collection metadata and storage safety."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pe_static_analyzer.core.pe_input import sha256_file
import json
import re
import time
from collections import Counter
from pathlib import Path

from pe_static_analyzer.gold.io import load_metadata
from typing import Iterable

SAFE_SUFFIXES = (".txt",)
MALICIOUS_SAFE_PATTERN = re.compile(r"^[a-f0-9]{64}\.malwarebazaar\.zip\.txt$")
HASH_RE = re.compile(r"^[a-f0-9]{64}$")


@dataclass
class GoldVerificationSettings:
    dest: Path = Path('datasets/gold_pe')
    metadata: Path | None = None
    output: Path = Path('outputs/gold_pe/verify_gold_summary.json')
    errors_output: Path = Path('outputs/gold_pe/verify_gold_errors.csv')
    target_min_total: int = 3000
    target_min_per_class: int = 1500
    verify_hashes: bool = False
    benign_pe_magic: bool = False


def is_probably_pe(path: Path) -> bool:
    with path.open("rb") as handle:
        header = handle.read(0x1000)
    if len(header) < 0x40 or header[:2] != b"MZ":
        return False
    pe_offset = int.from_bytes(header[0x3C:0x40], "little", signed=False)
    if pe_offset <= 0:
        return False
    with path.open("rb") as handle:
        handle.seek(pe_offset)
        return handle.read(4) == b"PE\0\0"


def check_rows(rows: list[dict[str, str]], dest: Path, verify_hashes: bool, benign_pe_magic: bool) -> tuple[dict[str, object], list[dict[str, object]]]:
    errors: list[dict[str, object]] = []
    counts = Counter(row.get("label", "") for row in rows)
    sources = Counter(row.get("source", "") for row in rows)
    stored_paths_seen: set[str] = set()
    sha_seen: set[str] = set()
    stored_sha_seen: set[str] = set()
    total_bytes = 0

    for index, row in enumerate(rows, start=1):
        label = row.get("label", "")
        sha = (row.get("sha256") or "").lower()
        stored_sha = (row.get("stored_sha256") or "").lower()
        stored_path_value = row.get("stored_path", "")
        if not HASH_RE.fullmatch(sha):
            errors.append({"row": index, "field": "sha256", "error": "invalid sha256", "value": sha})
        if stored_sha and not HASH_RE.fullmatch(stored_sha):
            errors.append({"row": index, "field": "stored_sha256", "error": "invalid stored_sha256", "value": stored_sha})
        if sha in sha_seen:
            errors.append({"row": index, "field": "sha256", "error": "duplicate sha256", "value": sha})
        sha_seen.add(sha)
        if stored_sha:
            if stored_sha in stored_sha_seen:
                errors.append({"row": index, "field": "stored_sha256", "error": "duplicate stored_sha256", "value": stored_sha})
            stored_sha_seen.add(stored_sha)

        if label not in {"benign", "malicious"}:
            errors.append({"row": index, "field": "label", "error": "unexpected label", "value": label})
        if not stored_path_value:
            errors.append({"row": index, "field": "stored_path", "error": "empty stored_path"})
            continue
        if stored_path_value in stored_paths_seen:
            errors.append({"row": index, "field": "stored_path", "error": "duplicate stored_path", "value": stored_path_value})
        stored_paths_seen.add(stored_path_value)

        stored_path = Path(stored_path_value)
        if not stored_path.is_absolute():
            stored_path = Path(stored_path_value)
        try:
            stored_path.relative_to(dest)
        except ValueError:
            errors.append({"row": index, "field": "stored_path", "error": "stored path is outside dataset directory", "value": stored_path_value})
        if not stored_path.exists():
            errors.append({"row": index, "field": "stored_path", "error": "stored file missing", "value": stored_path_value})
            continue
        if not stored_path.name.endswith(SAFE_SUFFIXES):
            errors.append({"row": index, "field": "stored_path", "error": "stored file is not defanged with .txt", "value": stored_path_value})
        total_bytes += stored_path.stat().st_size

        if label == "malicious":
            if not MALICIOUS_SAFE_PATTERN.fullmatch(stored_path.name):
                errors.append({"row": index, "field": "stored_path", "error": "malicious sample filename is not sha256.malwarebazaar.zip.txt", "value": stored_path.name})
            if ".zip.txt" not in stored_path.name:
                errors.append({"row": index, "field": "stored_path", "error": "malicious sample is not stored as defanged zip", "value": stored_path.name})
        elif label == "benign":
            if not stored_path.name.endswith((".exe.txt", ".dll.txt", ".sys.txt", ".scr.txt", ".ocx.txt", ".cpl.txt", ".drv.txt", ".pe.txt")):
                errors.append({"row": index, "field": "stored_path", "error": "benign sample extension is not PE-defanged", "value": stored_path.name})
            if benign_pe_magic and not is_probably_pe(stored_path):
                errors.append({"row": index, "field": "stored_path", "error": "benign file does not have PE magic", "value": stored_path.name})

        if verify_hashes and stored_sha:
            actual = sha256_file(stored_path)
            if actual != stored_sha:
                errors.append({"row": index, "field": "stored_sha256", "error": "stored file hash mismatch", "expected": stored_sha, "actual": actual, "value": stored_path_value})

    summary: dict[str, object] = {
        "metadata_rows": len(rows),
        "label_counts": dict(counts),
        "source_counts": dict(sources),
        "unique_sha256": len(sha_seen),
        "unique_stored_sha256": len(stored_sha_seen),
        "unique_stored_paths": len(stored_paths_seen),
        "total_stored_bytes": total_bytes,
        "errors": len(errors),
    }
    return summary, errors


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fieldnames = sorted({key for row in rows for key in row}) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    settings = GoldVerificationSettings()
    metadata = settings.metadata or settings.dest / "metadata.csv"
    if not metadata.exists():
        raise SystemExit(f"metadata not found: {metadata}")
    safety = settings.dest / "README_SAFETY.txt"
    rows = load_metadata(metadata)
    summary, errors = check_rows(rows, settings.dest, settings.verify_hashes, settings.benign_pe_magic)
    counts = summary["label_counts"]
    assert isinstance(counts, dict)
    summary.update(
        {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dest": str(settings.dest),
            "metadata": str(metadata),
            "readme_safety_exists": safety.exists(),
            "target_min_total": settings.target_min_total,
            "target_min_per_class": settings.target_min_per_class,
            "meets_target_total": int(summary["metadata_rows"]) >= settings.target_min_total,
            "meets_target_per_class": counts.get("benign", 0) >= settings.target_min_per_class and counts.get("malicious", 0) >= settings.target_min_per_class,
            "verify_hashes": settings.verify_hashes,
            "benign_pe_magic": settings.benign_pe_magic,
        }
    )
    if not safety.exists():
        errors.append({"row": 0, "field": "README_SAFETY.txt", "error": "missing safety readme"})
        summary["errors"] = int(summary["errors"]) + 1
    settings.output.parent.mkdir(parents=True, exist_ok=True)
    settings.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(settings.errors_output, errors)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if errors:
        print(f"errors written to {settings.errors_output}")
        return 1
    if not summary["meets_target_total"] or not summary["meets_target_per_class"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
