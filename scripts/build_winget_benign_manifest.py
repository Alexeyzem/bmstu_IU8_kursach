#!/usr/bin/env python3
"""Build a benign PE URL manifest from Microsoft WinGet package manifests.

The script does not download installers. It reads the public winget-pkgs tree,
selects latest package installer manifests, extracts HTTPS `.exe` installer URLs
and official InstallerSha256 values, then writes JSON consumable by
`scripts/download_gold_pe_samples.py`.
"""

from __future__ import annotations

import argparse
import json
import re
import ssl
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import certifi

TREE_URL = "https://api.github.com/repos/microsoft/winget-pkgs/git/trees/master?recursive=1"
RAW_BASE = "https://raw.githubusercontent.com/microsoft/winget-pkgs/master"
PE_URL_EXTENSIONS = {".exe"}


def https_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=certifi.where())


def fetch_text(url: str, timeout: int) -> str:
    request = Request(url, headers={"User-Agent": "kursach-konnova-winget-manifest-builder/1.0"})
    with urlopen(request, timeout=timeout, context=https_context()) as response:
        return response.read().decode("utf-8", errors="replace")


def fetch_json(url: str, timeout: int) -> dict[str, Any]:
    return json.loads(fetch_text(url, timeout))


def version_key(version: str) -> tuple[Any, ...]:
    parts: list[Any] = []
    for token in re.split(r"([0-9]+)", version.lower()):
        if not token:
            continue
        if token.isdigit():
            parts.append((1, int(token)))
        else:
            parts.append((0, token))
    return tuple(parts)


def latest_installer_paths(tree: list[dict[str, Any]]) -> list[str]:
    latest_by_package: dict[str, tuple[tuple[Any, ...], str]] = {}
    for item in tree:
        path = str(item.get("path", ""))
        if item.get("type") != "blob" or not path.endswith(".installer.yaml"):
            continue
        parts = path.split("/")
        if len(parts) < 5 or parts[0] != "manifests":
            continue
        package_dir = "/".join(parts[:-2])
        version = parts[-2]
        key = version_key(version)
        previous = latest_by_package.get(package_dir)
        if previous is None or key > previous[0]:
            latest_by_package[package_dir] = (key, path)
    return [value[1] for value in latest_by_package.values()]


def strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] in {'"', "'"} and value[-1] == value[0]:
        return value[1:-1]
    return value


def yaml_scalar(line: str) -> tuple[str, str] | None:
    if ":" not in line:
        return None
    key, value = line.split(":", 1)
    key = key.strip().lstrip("- ").strip()
    value = strip_quotes(value.strip())
    if not key:
        return None
    return key, value


def parse_installer_yaml(text: str) -> tuple[dict[str, str], list[dict[str, str]]]:
    top: dict[str, str] = {}
    installers: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    in_installers = False

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()

        if indent == 0 and stripped == "Installers:":
            in_installers = True
            current = None
            continue
        if indent == 0 and not stripped.startswith("-"):
            in_installers = False
            parsed = yaml_scalar(stripped)
            if parsed:
                top[parsed[0]] = parsed[1]
            continue

        if in_installers:
            if stripped.startswith("-"):
                current = {}
                installers.append(current)
                stripped = stripped[1:].strip()
                if not stripped:
                    continue
            if current is None:
                continue
            parsed = yaml_scalar(stripped)
            if parsed:
                current[parsed[0]] = parsed[1]

    return top, installers


def url_looks_like_pe(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        return False
    suffix = Path(parsed.path).suffix.lower()
    return suffix in PE_URL_EXTENSIONS


def build_entries(paths: list[str], limit: int, timeout: int, sleep_seconds: float) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    seen_hashes: set[str] = set()

    for index, path in enumerate(paths, start=1):
        if len(entries) >= limit:
            break
        raw_url = f"{RAW_BASE}/{path}"
        try:
            text = fetch_text(raw_url, timeout)
        except Exception as exc:  # network/HTTP failures should not stop manifest generation
            print(f"skip manifest fetch failure {path}: {exc}")
            continue
        top, installers = parse_installer_yaml(text)
        package_id = top.get("PackageIdentifier", "")
        package_version = top.get("PackageVersion", path.split("/")[-2])

        for installer in installers:
            url = installer.get("InstallerUrl", "")
            sha = (installer.get("InstallerSha256") or installer.get("Sha256") or "").lower()
            if not url_looks_like_pe(url):
                continue
            if not re.fullmatch(r"[a-f0-9]{64}", sha):
                continue
            if url in seen_urls or sha in seen_hashes:
                continue
            seen_urls.add(url)
            seen_hashes.add(sha)
            arch = installer.get("Architecture", "unknown")
            installer_type = installer.get("InstallerType", top.get("InstallerType", "exe"))
            entries.append(
                {
                    "url": url,
                    "source": "Microsoft WinGet manifests",
                    "label_source": "WinGet official installer URL with InstallerSha256; verify signature/reputation before publication",
                    "package_identifier": package_id,
                    "package_version": package_version,
                    "installer_sha256": sha,
                    "architecture": arch,
                    "installer_type": installer_type,
                    "winget_manifest_path": path,
                    "tags": ["benign", "winget", "official-manifest", arch, installer_type],
                }
            )
            if len(entries) >= limit:
                break
        if index % 100 == 0:
            print(f"processed manifests={index}, entries={len(entries)}")
        if sleep_seconds:
            time.sleep(sleep_seconds)
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benign_pe_sources.json"))
    parser.add_argument("--limit", type=int, default=2500)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--sleep", type=float, default=0.05, help="Delay between raw manifest fetches to avoid hammering GitHub.")
    parser.add_argument("--tree-cache", type=Path, default=Path("winget_tree_cache.json"))
    parser.add_argument("--reuse-tree-cache", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.reuse_tree_cache and args.tree_cache.exists():
        tree_data = json.loads(args.tree_cache.read_text(encoding="utf-8"))
    else:
        print(f"fetch tree: {TREE_URL}")
        tree_data = fetch_json(TREE_URL, args.timeout)
        args.tree_cache.write_text(json.dumps(tree_data), encoding="utf-8")

    paths = latest_installer_paths(tree_data.get("tree", []))
    print(f"latest installer manifests: {len(paths)}")
    entries = build_entries(paths, args.limit, args.timeout, args.sleep)
    args.output.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(entries)} benign PE URLs to {args.output}")
    if len(entries) < args.limit:
        print("warning: fewer .exe entries than requested; rerun with a larger source/cache or add Scoop/official vendor manifests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
