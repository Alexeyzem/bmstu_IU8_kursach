#!/usr/bin/env python3
"""Extract EMBER-v2-compatible static features for the external gold PE set.

Safety model:
- PE files are read as bytes only; they are never executed.
- Benign samples are defanged raw PE bytes (`.exe.txt`, `.dll.txt`, `.pe.txt`).
- MalwareBazaar samples are password-protected ZIP files renamed to `.zip.txt`;
  this script reads the encrypted archive member in memory with password
  `infected` and never writes the raw PE to disk.
"""

from __future__ import annotations

from pe_static_analyzer.settings import ACTIVE_EXPERIMENT
from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
import multiprocessing as mp
import time
from typing import Any

import lief
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pe_static_analyzer.core.pe_input import read_pe_bytes
from pe_static_analyzer.gold.io import load_metadata
from pe_static_analyzer.features.ember import FEATURE_DIM, PEFeatureExtractor, is_probably_pe, sorel_signed_log_postprocess

@dataclass
class GoldExtractionSettings:
    metadata: Path = Path('outputs/gold/gold_final_metadata.csv')
    output_dir: Path = Path('outputs/features')
    limit: int | None = None
    postprocess: str = 'sorel_signed_log'
    allow_malware_zip: bool = False
    sample_timeout: int = 60


GOLD_PANEL = GoldExtractionSettings(
    metadata=ACTIVE_EXPERIMENT.work / "gold_panel.csv",
    output_dir=ACTIVE_EXPERIMENT.work / "gold_panel",
    allow_malware_zip=True,
    sample_timeout=20,
)


def gold_vector(row: dict[str, str], postprocess: str, *, require_hash: bool = True) -> tuple[np.ndarray, str, str]:
    label = row["label"]
    if label not in {"0", "1"}:
        raise ValueError(f"unknown label: {label}")
    archived = label == "1"
    data = read_pe_bytes(Path(row["stored_path"]), archived)
    if not is_probably_pe(data):
        raise RuntimeError("extracted bytes are not PE")
    digest = hashlib.sha256(data).hexdigest()
    expected = row.get("sha256", "").lower()
    if (require_hash or expected) and digest != expected:
        raise RuntimeError(f"feature bytes sha256 mismatch: {digest}")
    vector = PEFeatureExtractor().feature_vector(data)
    if vector.shape != (FEATURE_DIM,):
        raise RuntimeError(f"feature shape {vector.shape}; expected ({FEATURE_DIM},)")
    if postprocess == "sorel_signed_log":
        vector = sorel_signed_log_postprocess(vector)
    elif postprocess != "raw":
        raise ValueError(f"unknown extraction postprocess: {postprocess}")
    mode = "malwarebazaar_zip_in_memory" if archived else "raw_defanged_pe"
    return vector, digest, mode


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def extract_one(row: dict[str, str], postprocess: str, allow_malware_zip: bool) -> tuple[np.ndarray | None, int | None, dict[str, Any] | None, dict[str, Any] | None]:
    sample_id = int(row["sample_id"])
    label = int(row["label"])
    if label == 1 and not allow_malware_zip:
        return None, None, None, {"sample_id": sample_id, "sha256": row["sha256"], "stored_path": row["stored_path"], "error": "malware_zip_requires_allow_malware_zip"}
    try:
        vector, _, extraction_mode = gold_vector(row, postprocess)
        label_row = {
            "sample_id": sample_id,
            "sha256": row["sha256"],
            "label": label,
            "source": row.get("source", ""),
            "stored_path": row.get("stored_path", ""),
            "extraction_mode": extraction_mode,
        }
        return vector.astype(np.float32), label, label_row, None
    except Exception as exc:  # noqa: BLE001 - report and keep going.
        return None, None, None, {"sample_id": sample_id, "sha256": row.get("sha256", ""), "stored_path": row.get("stored_path", ""), "error": str(exc)}


def extract_one_worker(row: dict[str, str], postprocess: str, allow_malware_zip: bool, queue: mp.Queue) -> None:
    queue.put(extract_one(row, postprocess, allow_malware_zip))


def extract_one_with_timeout(row: dict[str, str], postprocess: str, allow_malware_zip: bool, timeout: int) -> tuple[np.ndarray | None, int | None, dict[str, Any] | None, dict[str, Any] | None]:
    if timeout <= 0:
        return extract_one(row, postprocess, allow_malware_zip)
    ctx = mp.get_context("fork")
    queue: mp.Queue = ctx.Queue(maxsize=1)
    process = ctx.Process(target=extract_one_worker, args=(row, postprocess, allow_malware_zip, queue))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(5)
        return None, None, None, {"sample_id": int(row["sample_id"]), "sha256": row.get("sha256", ""), "stored_path": row.get("stored_path", ""), "error": f"sample_timeout_seconds={timeout}"}
    if process.exitcode != 0:
        return None, None, None, {"sample_id": int(row["sample_id"]), "sha256": row.get("sha256", ""), "stored_path": row.get("stored_path", ""), "error": f"worker_exitcode={process.exitcode}"}
    if queue.empty():
        return None, None, None, {"sample_id": int(row["sample_id"]), "sha256": row.get("sha256", ""), "stored_path": row.get("stored_path", ""), "error": "worker_returned_no_result"}
    return queue.get()


class GoldFeatureExtractionJob:
    """Extract a gold metadata CSV into feature matrix, label CSV, and error CSV."""

    def __init__(self, settings: GoldExtractionSettings) -> None:
        self.settings = settings
        self.started = time.time()
        self.vectors: list[np.ndarray] = []
        self.labels: list[int] = []
        self.label_rows: list[dict[str, Any]] = []
        self.error_rows: list[dict[str, Any]] = []

    def run(self) -> int:
        rows = load_metadata(self.settings.metadata, self.settings.limit)
        for row in rows:
            self._extract_row(row)
        self._write_outputs(rows)
        return 0 if not self.error_rows else 1

    def _extract_row(self, row: dict[str, str]) -> None:
        vector, label, label_row, error_row = extract_one_with_timeout(row, self.settings.postprocess, self.settings.allow_malware_zip, self.settings.sample_timeout)
        if error_row is not None:
            self.error_rows.append(error_row)
            return
        if vector is None or label is None or label_row is None:
            self.error_rows.append({"sample_id": int(row["sample_id"]), "sha256": row.get("sha256", ""), "stored_path": row.get("stored_path", ""), "error": "empty_extraction_result"})
            return
        self.vectors.append(vector)
        self.labels.append(label)
        self.label_rows.append(label_row)

    def _feature_table(self) -> pa.Table:
        if self.vectors:
            matrix = np.vstack(self.vectors).astype(np.float32)
        else:
            matrix = np.empty((0, FEATURE_DIM), dtype=np.float32)
        arrays: dict[str, pa.Array] = {
            "sample_id": pa.array([row["sample_id"] for row in self.label_rows], type=pa.int32()),
            "sha256": pa.array([row["sha256"] for row in self.label_rows], type=pa.string()),
            "label": pa.array(self.labels, type=pa.int8()),
        }
        for idx in range(FEATURE_DIM):
            arrays[f"f_{idx:04d}"] = pa.array(matrix[:, idx], type=pa.float32())
        return pa.table(arrays)

    def _write_outputs(self, rows: list[dict[str, str]]) -> None:
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(self._feature_table(), self.settings.output_dir / "gold_features.parquet", compression="zstd")
        write_csv(self.settings.output_dir / "gold_labels.csv", self.label_rows, ["sample_id", "sha256", "label", "source", "stored_path", "extraction_mode"])
        write_csv(self.settings.output_dir / "gold_feature_errors.csv", self.error_rows, ["sample_id", "sha256", "stored_path", "error"])
        summary = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metadata": str(self.settings.metadata),
            "output_features": str(self.settings.output_dir / "gold_features.parquet"),
            "output_labels": str(self.settings.output_dir / "gold_labels.csv"),
            "output_errors": str(self.settings.output_dir / "gold_feature_errors.csv"),
            "feature_dim": FEATURE_DIM,
            "postprocess": self.settings.postprocess,
            "allow_malware_zip": self.settings.allow_malware_zip,
            "requested_rows": len(rows),
            "extracted_rows": len(self.label_rows),
            "error_rows": len(self.error_rows),
            "label_counts": {"benign_0": int(sum(1 for item in self.labels if item == 0)), "malware_1": int(sum(1 for item in self.labels if item == 1))},
            "runtime_seconds": round(time.time() - self.started, 3),
            "safety": "PE bytes are parsed statically only; MalwareBazaar ZIP members are read in memory and raw malware PE is not written to disk.",
            "lief_version": getattr(lief, "__version__", "unknown"),
        }
        (self.settings.output_dir / "gold_feature_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> int:
    return GoldFeatureExtractionJob(GOLD_PANEL).run()


if __name__ == "__main__":
    raise SystemExit(main())
