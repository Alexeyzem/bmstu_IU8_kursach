#!/usr/bin/env python3
"""Download selected SOREL-20M preprocessed data from the public S3 bucket.

Default variant is `baseline-lmdb`: meta.db + EMBER feature LMDB (~80 GB),
which is the smallest official form sufficient for the neural-network baseline.
Use --dry-run first: SOREL-20M is large and governed by the Sophos terms.
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import certifi
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = "https://sorel-20m.s3.amazonaws.com/09-DEC-2020"
TERMS_URL = "https://github.com/sophos-ai/SOREL-20M/blob/master/Terms%20and%20Conditions%20of%20Use.pdf"

FILES = {
    "meta.db": {
        "path": "meta.db",
        "url": f"{BASE_URL}/processed-data/meta.db",
        "size": 3_788_979_200,
        "description": "SQLite index: sha256, labels, tags, counts, first/last seen timestamps",
    },
    "ember_features/data.mdb": {
        "path": "ember_features/data.mdb",
        "url": f"{BASE_URL}/processed-data/ember_features/data.mdb",
        "size": 76_865_335_296,
        "description": "LMDB with EMBER-v2 feature vectors serialized by msgpack and zlib",
    },
    "ember_features/lock.mdb": {
        "path": "ember_features/lock.mdb",
        "url": f"{BASE_URL}/processed-data/ember_features/lock.mdb",
        "size": 65_664,
        "description": "LMDB lock file for ember_features",
    },
    "pe_metadata/data.mdb": {
        "path": "pe_metadata/data.mdb",
        "url": f"{BASE_URL}/processed-data/pe_metadata/data.mdb",
        "size": 515_343_990_784,
        "description": "LMDB with pefile.dump_dict() metadata for deeper PE analysis",
    },
    "pe_metadata/lock.mdb": {
        "path": "pe_metadata/lock.mdb",
        "url": f"{BASE_URL}/processed-data/pe_metadata/lock.mdb",
        "size": 65_664,
        "description": "LMDB lock file for pe_metadata",
    },
    "lightGBM-features/train-features.npz": {
        "path": "lightGBM-features/train-features.npz",
        "url": f"{BASE_URL}/lightGBM-features/train-features.npz",
        "size": 121_046_992_510,
        "description": "Prebuilt LightGBM train matrix; binary labels only",
    },
    "lightGBM-features/validation-features.npz": {
        "path": "lightGBM-features/validation-features.npz",
        "url": f"{BASE_URL}/lightGBM-features/validation-features.npz",
        "size": 23_790_175_898,
        "description": "Prebuilt LightGBM validation matrix; binary labels only",
    },
    "lightGBM-features/test-features.npz": {
        "path": "lightGBM-features/test-features.npz",
        "url": f"{BASE_URL}/lightGBM-features/test-features.npz",
        "size": 39_987_140_938,
        "description": "Prebuilt LightGBM test matrix; binary labels only",
    },
}

VARIANTS = {
    "metadata-only": ["meta.db"],
    "baseline-lmdb": ["meta.db", "ember_features/data.mdb", "ember_features/lock.mdb"],
    "lightgbm-npz": [
        "lightGBM-features/train-features.npz",
        "lightGBM-features/validation-features.npz",
        "lightGBM-features/test-features.npz",
    ],
    "full-processed": [
        "meta.db",
        "ember_features/data.mdb",
        "ember_features/lock.mdb",
        "pe_metadata/data.mdb",
        "pe_metadata/lock.mdb",
    ],
}


def human_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{value} B"


def selected_files(variant: str, only: Iterable[str] | None, max_files: int | None) -> list[dict[str, object]]:
    keys = list(VARIANTS[variant])
    if only:
        wanted = set(only)
        unknown = wanted - set(FILES)
        if unknown:
            raise SystemExit(f"Unknown --only key(s): {', '.join(sorted(unknown))}")
        keys = [key for key in keys if key in wanted]
    if max_files is not None:
        keys = keys[:max_files]
    return [FILES[key] for key in keys]


def existing_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def free_bytes(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free

def https_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=certifi.where())


def download_file(url: str, target: Path, expected_size: int, chunk_size: int, retries: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")

    if target.exists() and target.stat().st_size == expected_size:
        print(f"skip complete: {target} ({human_bytes(expected_size)})")
        return
    if target.exists() and target.stat().st_size != expected_size:
        raise RuntimeError(
            f"Refusing to overwrite size-mismatched final file {target}: "
            f"have {target.stat().st_size}, expected {expected_size}"
        )

    resume_at = existing_size(part)
    headers = {"User-Agent": "kursach-konnova-sorel-downloader/1.0"}
    if resume_at:
        headers["Range"] = f"bytes={resume_at}-"

    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=60, context=https_context()) as response:
                status = getattr(response, "status", 200)
                mode = "ab" if resume_at and status == 206 else "wb"
                if resume_at and status != 206:
                    print(f"server ignored resume for {target}; restarting partial download")
                    resume_at = 0
                downloaded = resume_at if mode == "ab" else 0
                started = time.time()
                last_report = started
                with part.open(mode + ("" if "b" in mode else "b")) as handle:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if now - last_report >= 5:
                            pct = downloaded / expected_size * 100 if expected_size else 0
                            speed = (downloaded - resume_at) / max(now - started, 0.001)
                            print(
                                f"{target}: {pct:6.2f}% "
                                f"({human_bytes(downloaded)}/{human_bytes(expected_size)}), "
                                f"{human_bytes(int(speed))}/s"
                            )
                            last_report = now
            final_size = part.stat().st_size
            if final_size != expected_size:
                raise RuntimeError(
                    f"Incomplete download for {target}: got {final_size}, expected {expected_size}"
                )
            part.replace(target)
            print(f"done: {target} ({human_bytes(expected_size)})")
            return
        except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
            if attempt == retries:
                raise
            wait = min(60, 2**attempt)
            print(f"attempt {attempt}/{retries} failed for {target}: {exc}; retrying in {wait}s")
            time.sleep(wait)
            resume_at = existing_size(part)
            headers = {"User-Agent": "kursach-konnova-sorel-downloader/1.0"}
            if resume_at:
                headers["Range"] = f"bytes={resume_at}-"


def write_manifest(dest: Path, files: list[dict[str, object]], variant: str) -> None:
    manifest = {
        "dataset": "SOREL-20M",
        "variant": variant,
        "terms_url": TERMS_URL,
        "source_prefix": BASE_URL,
        "files": files,
        "total_size_bytes": sum(int(item["size"]) for item in files),
    }
    (dest / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("datasets/sorel20m"))
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="baseline-lmdb")
    parser.add_argument("--only", action="append", choices=sorted(FILES), help="Download only a selected manifest key. Repeatable.")
    parser.add_argument("--dry-run", action="store_true", help="Print manifest and disk requirement; do not download.")
    parser.add_argument("--max-files", type=int, help="For smoke tests: limit selected files after variant/only filtering.")
    parser.add_argument("--chunk-size", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--yes", action="store_true", help="Acknowledge SOREL terms and skip interactive confirmation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = selected_files(args.variant, args.only, args.max_files)
    total = sum(int(item["size"]) for item in files)
    available = free_bytes(args.dest)

    print(f"SOREL-20M terms: {TERMS_URL}")
    print(f"destination: {args.dest}")
    print(f"variant: {args.variant}")
    print(f"selected files: {len(files)}")
    for item in files:
        print(f"- {item['path']}: {human_bytes(int(item['size']))} — {item['description']}")
    print(f"total expected size: {human_bytes(total)}")
    print(f"free space at destination: {human_bytes(available)}")

    if available < total:
        raise SystemExit(
            f"Not enough free disk space for selected files: need {human_bytes(total)}, have {human_bytes(available)}"
        )

    write_manifest(args.dest, files, args.variant)
    if args.dry_run:
        print("dry-run: no files downloaded")
        return 0

    if not args.yes:
        print("Use --yes after reading and accepting the SOREL-20M terms of use.", file=sys.stderr)
        return 2

    for item in files:
        download_file(
            str(item["url"]),
            args.dest / str(item["path"]),
            int(item["size"]),
            args.chunk_size,
            args.retries,
        )
    print("all selected SOREL-20M files downloaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
