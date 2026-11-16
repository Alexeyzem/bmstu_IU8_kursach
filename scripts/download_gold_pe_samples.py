#!/usr/bin/env python3
"""Collect a defanged modern PE gold-validation sample set.

Malicious source: MalwareBazaar API. The script stores MalwareBazaar responses as
password-protected ZIP files and renames them to `.zip.txt`; it never extracts or
executes malware.

Benign source: explicit HTTPS URLs from a JSON manifest. Downloaded files are
validated by PE magic (`MZ` + `PE\0\0`) and stored as `.pe.txt`; they are never
executed. Add more trusted vendor URLs to reach the desired benign count.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import csv
import hashlib
import json
import os
import ssl
import certifi
import re
import time
from pathlib import Path
from typing import Any
from http.client import IncompleteRead
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

MALWAREBAZAAR_API = "https://mb-api.abuse.ch/api/v1/"
MALWAREBAZAAR_TERMS = "https://abuse.ch/terms-of-use/"
DEFAULT_MALWARE_QUERIES = [
    {"query": "get_file_type", "file_type": "exe", "limit": "1000"},
    {"query": "get_file_type", "file_type": "dll", "limit": "1000"},
    {"query": "get_siginfo", "signature": "LummaStealer", "limit": "1000"},
    {"query": "get_siginfo", "signature": "RedLineStealer", "limit": "1000"},
    {"query": "get_siginfo", "signature": "Formbook", "limit": "1000"},
    {"query": "get_siginfo", "signature": "RemcosRAT", "limit": "1000"},
    {"query": "get_siginfo", "signature": "AsyncRAT", "limit": "1000"},
    {"query": "get_siginfo", "signature": "AgentTesla", "limit": "1000"},
    {"query": "get_siginfo", "signature": "CobaltStrike", "limit": "1000"},
    {"query": "get_recent", "selector": "100"},
]

PE_EXTENSIONS = {".exe", ".dll", ".sys", ".scr", ".ocx", ".cpl", ".drv"}
DOWNLOAD_ERRORS = (HTTPError, URLError, TimeoutError, ConnectionError, IncompleteRead, ssl.SSLError)
DOWNLOAD_ERRORS_WITH_RUNTIME = (*DOWNLOAD_ERRORS, RuntimeError)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_name(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._+-]+", "_", value).strip("._")
    return cleaned[:120] or fallback


def local_malwarebazaar_auth_key() -> str | None:
    for path in (Path(".env.local"), Path("env.local")):
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().removeprefix("export ").strip()
            value = value.strip().strip("'\"")
            if key == "MALWAREBAZAAR_AUTH_KEY" and value:
                return value
    return os.environ.get("MALWAREBAZAAR_AUTH_KEY")

def https_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=certifi.where())


def form_body(data: dict[str, str]) -> bytes:
    return urlencode(data).encode("utf-8")


def request_json_post(data: dict[str, str], auth_key: str | None, timeout: int) -> dict[str, Any]:
    body = form_body(data)
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "kursach-konnova-gold-pe-collector/1.0",
    }
    if auth_key:
        headers["Auth-Key"] = auth_key
    request = Request(MALWAREBAZAAR_API, data=body, headers=headers, method="POST")
    with urlopen(request, timeout=timeout, context=https_context()) as response:
        payload = response.read()
    try:
        return json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Expected JSON from MalwareBazaar; got {payload[:120]!r}") from exc


def request_binary_post(data: dict[str, str], auth_key: str, timeout: int) -> tuple[bytes, str]:
    body = form_body(data)
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "kursach-konnova-gold-pe-collector/1.0",
        "Auth-Key": auth_key,
    }
    request = Request(MALWAREBAZAAR_API, data=body, headers=headers, method="POST")
    with urlopen(request, timeout=timeout, context=https_context()) as response:
        return response.read(), response.headers.get("Content-Type", "")


def looks_like_error_json(blob: bytes) -> bool:
    prefix = blob[:32].lstrip()
    return prefix.startswith(b"{") or prefix.startswith(b"[")


def is_probably_pe(data: bytes) -> bool:
    if len(data) < 0x40 or data[:2] != b"MZ":
        return False
    pe_offset = int.from_bytes(data[0x3C:0x40], "little", signed=False)
    if pe_offset <= 0 or pe_offset + 4 > len(data):
        return False
    return data[pe_offset:pe_offset + 4] == b"PE\x00\x00"


def read_existing_hashes(path: Path) -> set[str]:
    hashes: set[str] = set()
    if not path.exists():
        return hashes
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = row.get("sha256") or row.get("original_sha256") or row.get("stored_sha256")
            if value:
                hashes.add(value.lower())
    return hashes

def read_existing_source_urls(path: Path) -> set[str]:
    urls: set[str] = set()
    if not path.exists():
        return urls
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = row.get("source_url")
            if value:
                urls.add(value)
    return urls



def load_sorel_exclusion(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {str(item).lower() for item in data}
        if isinstance(data, dict):
            return {str(item).lower() for item in data.keys()}
    hashes: set[str] = set()
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = re.search(r"\b[a-fA-F0-9]{64}\b", line)
            if match:
                hashes.add(match.group(0).lower())
    return hashes


def query_malware_candidates(auth_key: str, timeout: int, exclude: set[str]) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for query in DEFAULT_MALWARE_QUERIES:
        print(f"query MalwareBazaar: {query}")
        try:
            result = request_json_post(query, auth_key, timeout)
        except (HTTPError, URLError, TimeoutError, ConnectionError, IncompleteRead, ssl.SSLError) as exc:
            print(f"  skipped: request failed: {exc}")
            continue
        status = result.get("query_status")
        if status != "ok":
            print(f"  skipped: query_status={status}")
            continue
        for item in result.get("data") or []:
            sha = str(item.get("sha256_hash", "")).lower()
            if not re.fullmatch(r"[a-f0-9]{64}", sha):
                continue
            if sha in exclude:
                continue
            file_type = str(item.get("file_type") or "").lower()
            mime = str(item.get("file_type_mime") or "").lower()
            magika = str(item.get("magika") or "").lower()
            # Keep Windows PE-like samples. MalwareBazaar usually marks these as exe/dll and x-dosexec.
            if file_type not in {"exe", "dll", "sys", "scr"} and "dosexec" not in mime and magika != "pebin":
                continue
            candidates.setdefault(sha, item)
        time.sleep(1.0)
    ordered = sorted(candidates.values(), key=lambda row: str(row.get("first_seen") or ""), reverse=True)
    return ordered


def malware_metadata_row(item: dict[str, Any], sha: str, stored_sha: str, target: Path) -> dict[str, Any]:
    return {
        "label": "malicious",
        "source": "MalwareBazaar",
        "sha256": sha,
        "stored_sha256": stored_sha,
        "stored_path": str(target),
        "stored_format": "password-protected zip renamed to .zip.txt; password from MalwareBazaar docs is infected",
        "first_seen": item.get("first_seen", ""),
        "file_name": item.get("file_name", ""),
        "file_type": item.get("file_type", ""),
        "file_type_mime": item.get("file_type_mime", ""),
        "signature": item.get("signature", ""),
        "imphash": item.get("imphash", ""),
        "tlsh": item.get("tlsh", ""),
        "tags": json.dumps(item.get("tags") or [], ensure_ascii=False),
        "source_url": "https://bazaar.abuse.ch/api/",
        "collected_at": now_iso(),
    }


def write_temp_bytes(directory: Path, data: bytes, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=directory, prefix=".download-", suffix=suffix) as handle:
        handle.write(data)
        return Path(handle.name)


def discard_temp(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def download_malware_item(item: dict[str, Any], malware_dir: Path, auth_key: str, timeout: int) -> dict[str, Any]:
    sha = str(item["sha256_hash"]).lower()
    target = malware_dir / f"{sha}.malwarebazaar.zip.txt"
    try:
        blob, _content_type = request_binary_post({"query": "get_file", "sha256_hash": sha}, auth_key, timeout)
    except DOWNLOAD_ERRORS as exc:
        return {"status": "failed", "sha256": sha, "message": str(exc)}
    if looks_like_error_json(blob):
        return {"status": "skipped", "sha256": sha, "message": f"MalwareBazaar returned JSON instead of ZIP: {blob[:200]!r}"}
    # MalwareBazaar says samples are password-protected ZIPs; keep the archive defanged.
    temp_path = write_temp_bytes(malware_dir, blob, ".malwarebazaar.zip.txt")
    stored_sha = sha256_bytes(blob)
    return {
        "status": "downloaded",
        "sha256": sha,
        "target": target,
        "temp_path": temp_path,
        "row": malware_metadata_row(item, sha, stored_sha, target),
    }


def select_malware_downloads(candidates: list[dict[str, Any]], limit: int, malware_dir: Path, existing: set[str]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen = set(existing)
    for item in candidates:
        if len(selected) >= limit:
            break
        sha = str(item.get("sha256_hash", "")).lower()
        if sha in seen:
            continue
        seen.add(sha)
        target = malware_dir / f"{sha}.malwarebazaar.zip.txt"
        if target.exists():
            existing.add(sha)
            continue
        selected.append(item)
    return selected


def download_malware(dest: Path, auth_key: str | None, limit: int, timeout: int, dry_run: bool, understand: bool, exclude: set[str], workers: int) -> int:
    if limit <= 0:
        return 0
    if not auth_key:
        raise SystemExit("MalwareBazaar downloads require --malwarebazaar-auth-key or MALWAREBAZAAR_AUTH_KEY.")
    candidates = query_malware_candidates(auth_key, timeout, exclude)

    metadata_path = dest / "metadata.csv"
    existing = read_existing_hashes(metadata_path) | exclude
    malware_dir = dest / "malicious"
    malware_dir.mkdir(parents=True, exist_ok=True)
    selected = select_malware_downloads(candidates, limit, malware_dir, existing)
    print(f"malware candidates: {len(candidates)}; selected for download: {len(selected)}")

    if dry_run:
        for item in selected[:20]:
            print(f"dry-run malware: {item.get('sha256_hash')} {item.get('first_seen')} {item.get('file_type')} {item.get('signature')}")
        if len(selected) > 20:
            print(f"dry-run malware: ... {len(selected) - 20} more")
        return 0
    if not understand:
        raise SystemExit("Refusing to download malware without --i-understand-malware-handling.")

    downloaded = 0
    max_workers = max(1, workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_malware_item, item, malware_dir, auth_key, timeout): str(item["sha256_hash"]).lower()
            for item in selected
        }
        total = len(futures)
        for completed, future in enumerate(as_completed(futures), start=1):
            sha = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"download malware {completed}/{total}: {sha} failed: {exc}")
                continue
            status = result.get("status")
            if status != "downloaded":
                print(f"download malware {completed}/{total}: {sha} {status}: {result.get('message', '')}")
                continue
            temp_path = result["temp_path"]
            if sha in existing:
                discard_temp(temp_path)
                print(f"download malware {completed}/{total}: {sha} skipped duplicate")
                continue
            target = result["target"]
            os.replace(temp_path, target)
            append_metadata(metadata_path, [result["row"]])
            existing.add(sha)
            downloaded += 1
            print(f"download malware {completed}/{total}: {sha} saved ({downloaded} new)")
    return downloaded


def load_benign_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Benign manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("Benign manifest must be a JSON array")
    return data


def download_url(url: str, timeout: int, max_bytes: int) -> bytes:
    request = Request(url, headers={"User-Agent": "kursach-konnova-gold-pe-collector/1.0"})
    with urlopen(request, timeout=timeout, context=https_context()) as response:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise RuntimeError(f"File exceeds --max-benign-bytes ({max_bytes})")
            chunks.append(chunk)
    return b"".join(chunks)


def benign_metadata_row(item: dict[str, Any], url: str, file_sha: str, target: Path, original_name: str, original_suffix: str) -> dict[str, Any]:
    return {
        "label": "benign",
        "source": item.get("source", "trusted-url-manifest"),
        "sha256": file_sha,
        "stored_sha256": file_sha,
        "stored_path": str(target),
        "stored_format": f"raw PE renamed to {original_suffix}.txt",
        "first_seen": item.get("first_seen", ""),
        "file_name": original_name,
        "file_type": original_suffix.lstrip("."),
        "file_type_mime": "application/x-dosexec",
        "signature": "",
        "imphash": "",
        "tlsh": "",
        "tags": json.dumps(item.get("tags") or [], ensure_ascii=False),
        "source_url": url,
        "collected_at": now_iso(),
    }


def select_benign_downloads(items: list[dict[str, Any]], limit: int, existing_urls: set[str] | None = None) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_urls: set[str] = set(existing_urls or set())
    for item in items:
        if len(selected) >= limit:
            break
        url = str(item.get("url") or "")
        if not url.startswith("https://"):
            print(f"skip non-HTTPS benign URL: {url}")
            continue
        if url in seen_urls:
            print(f"skip duplicate benign URL: {url}")
            continue
        seen_urls.add(url)
        selected.append(item)
    return selected


def download_benign_item(item: dict[str, Any], benign_dir: Path, timeout: int, max_bytes: int) -> dict[str, Any]:
    url = str(item.get("url") or "")
    try:
        blob = download_url(url, timeout, max_bytes)
    except DOWNLOAD_ERRORS_WITH_RUNTIME as exc:
        return {"status": "failed", "url": url, "message": str(exc)}
    file_sha = sha256_bytes(blob).lower()
    if not is_probably_pe(blob):
        return {"status": "skipped", "url": url, "sha256": file_sha, "message": "file does not look like a raw PE; use direct .exe/.dll URLs or pre-extract manually"}
    parsed = urlparse(url)
    original_name = safe_name(Path(parsed.path).name, f"benign_{file_sha[:12]}.exe")
    original_suffix = Path(original_name).suffix.lower()
    if original_suffix not in PE_EXTENSIONS:
        original_suffix = ".pe"
    target = benign_dir / f"{file_sha}.{original_name}.txt"
    temp_path = write_temp_bytes(benign_dir, blob, f"{original_suffix}.txt")
    return {
        "status": "downloaded",
        "url": url,
        "sha256": file_sha,
        "target": target,
        "temp_path": temp_path,
        "row": benign_metadata_row(item, url, file_sha, target, original_name, original_suffix),
    }


def download_benign(dest: Path, manifest_path: Path, limit: int, timeout: int, max_bytes: int, dry_run: bool, exclude: set[str], workers: int) -> int:
    if limit <= 0:
        return 0
    items = select_benign_downloads(load_benign_manifest(manifest_path), limit, read_existing_source_urls(dest / "metadata.csv"))
    benign_dir = dest / "benign"
    benign_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = dest / "metadata.csv"
    existing = read_existing_hashes(metadata_path) | exclude
    print(f"benign URLs selected for download: {len(items)}")

    if dry_run:
        for item in items:
            print(f"dry-run benign: {item.get('url')} label_source={item.get('label_source', 'trusted publisher URL')}")
        return 0

    downloaded = 0
    max_workers = max(1, workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_benign_item, item, benign_dir, timeout, max_bytes): str(item.get("url") or "")
            for item in items
        }
        total = len(futures)
        for completed, future in enumerate(as_completed(futures), start=1):
            url = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"download benign {completed}/{total}: {url} failed: {exc}")
                continue
            status = result.get("status")
            if status != "downloaded":
                print(f"download benign {completed}/{total}: {url} {status}: {result.get('message', '')}")
                continue
            file_sha = result["sha256"]
            temp_path = result["temp_path"]
            if file_sha in existing:
                discard_temp(temp_path)
                print(f"download benign {completed}/{total}: {url} skipped duplicate sha256={file_sha}")
                continue
            target = result["target"]
            if target.exists():
                discard_temp(temp_path)
                existing.add(file_sha)
                print(f"download benign {completed}/{total}: {url} skipped existing file sha256={file_sha}")
                continue
            os.replace(temp_path, target)
            append_metadata(metadata_path, [result["row"]])
            existing.add(file_sha)
            downloaded += 1
            print(f"download benign {completed}/{total}: {url} saved sha256={file_sha} ({downloaded} new)")
    return downloaded


def append_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label", "source", "sha256", "stored_sha256", "stored_path", "stored_format", "first_seen",
        "file_name", "file_type", "file_type_mime", "signature", "imphash", "tlsh", "tags", "source_url", "collected_at",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("datasets/gold_pe"))
    parser.add_argument("--malware-limit", type=int, default=2500)
    parser.add_argument("--benign-limit", type=int, default=2500)
    parser.add_argument("--benign-manifest", type=Path, default=Path("benign_pe_sources.json"))
    parser.add_argument("--malwarebazaar-auth-key", default=local_malwarebazaar_auth_key())
    parser.add_argument("--exclude-hashes", type=Path, help="Optional text/CSV/JSON file with SHA-256 hashes to exclude, e.g. SOREL/EMBER/BODMAS hashes.")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-benign-bytes", type=int, default=200 * 1024 * 1024)
    parser.add_argument("--benign-workers", type=int, default=8, help="Parallel benign download workers.")
    parser.add_argument("--malware-workers", type=int, default=4, help="Parallel MalwareBazaar download workers; keep conservative for rate limiting and safety.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--i-understand-malware-handling", action="store_true", help="Required for actual MalwareBazaar downloads.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    (args.dest / "README_SAFETY.txt").write_text(
        "Gold PE validation set storage rules:\n"
        "- malicious samples from MalwareBazaar are stored as password-protected ZIP files renamed to .zip.txt; never extract outside an isolated lab VM;\n"
        "- benign PE files are stored as raw bytes renamed to .pe.txt/.exe.txt/.dll.txt; never execute collected files during dataset work;\n"
        "- metadata.csv is the primary index for labels, source URLs, hashes, timestamps, and family/signature hints;\n"
        f"- MalwareBazaar terms: {MALWAREBAZAAR_TERMS}\n",
        encoding="utf-8",
    )
    exclude = load_sorel_exclusion(args.exclude_hashes)
    print(f"destination: {args.dest}")
    print(f"exclude hashes: {len(exclude)}")
    print(f"MalwareBazaar terms: {MALWAREBAZAAR_TERMS}")

    malware_count = download_malware(
        args.dest,
        args.malwarebazaar_auth_key,
        args.malware_limit,
        args.timeout,
        args.dry_run,
        args.i_understand_malware_handling,
        exclude,
        args.malware_workers,
    )
    benign_count = download_benign(
        args.dest,
        args.benign_manifest,
        args.benign_limit,
        args.timeout,
        args.max_benign_bytes,
        args.dry_run,
        exclude,
        args.benign_workers,
    )
    print(f"downloaded: malicious={malware_count}, benign={benign_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
