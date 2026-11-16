#!/usr/bin/env python3
"""Verify the defanged gold PE collection metadata and storage safety."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

SAFE_SUFFIXES = (".txt",)
MALICIOUS_SAFE_PATTERN = re.compile(r"^[a-f0-9]{64}\.malwarebazaar\.zip\.txt$")
HASH_RE = re.compile(r"^[a-f0-9]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("datasets/gold_pe"))
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/gold_pe/verify_gold_summary.json"))
    parser.add_argument("--errors-output", type=Path, default=Path("outputs/gold_pe/verify_gold_errors.csv"))
    parser.add_argument("--target-min-total", type=int, default=3000)
    parser.add_argument("--target-min-per-class", type=int, default=1500)
    parser.add_argument("--verify-hashes", action="store_true", help="Hash every stored file; slow for large benign installers.")
    parser.add_argument("--benign-pe-magic", action="store_true", help="Check MZ/PE magic for every benign file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = args.metadata or args.dest / "metadata.csv"
    if not metadata.exists():
        raise SystemExit(f"metadata not found: {metadata}")
    safety = args.dest / "README_SAFETY.txt"
    rows = read_metadata(metadata)
    summary, errors = check_rows(rows, args.dest, args.verify_hashes, args.benign_pe_magic)
    counts = summary["label_counts"]
    assert isinstance(counts, dict)
    summary.update(
        {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dest": str(args.dest),
            "metadata": str(metadata),
            "readme_safety_exists": safety.exists(),
            "target_min_total": args.target_min_total,
            "target_min_per_class": args.target_min_per_class,
            "meets_target_total": int(summary["metadata_rows"]) >= args.target_min_total,
            "meets_target_per_class": counts.get("benign", 0) >= args.target_min_per_class and counts.get("malicious", 0) >= args.target_min_per_class,
            "verify_hashes": args.verify_hashes,
            "benign_pe_magic": args.benign_pe_magic,
        }
    )
    if not safety.exists():
        errors.append({"row": 0, "field": "README_SAFETY.txt", "error": "missing safety readme"})
        summary["errors"] = int(summary["errors"]) + 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(args.errors_output, errors)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if errors:
        print(f"errors written to {args.errors_output}")
        return 1
    if not summary["meets_target_total"] or not summary["meets_target_per_class"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
