"""Draw a class-balanced feature sample from SOREL metadata and LMDB."""
import hashlib
import sqlite3
from pathlib import Path

import lmdb
import numpy as np

from pe_static_analyzer.sorel.data import split_where_clause, unpack_feature_vector


def stable_pivot(seed: int, label: int, attempt: int) -> str:
    payload = f"{seed}:{label}:{attempt}".encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def query_one_sha(conn: sqlite3.Connection, label: int, pivot: str, split_clause: str) -> tuple[str, int] | None:
    row = conn.execute(
        f"""
        SELECT sha256, is_malware
        FROM meta
        WHERE is_malware = ? AND sha256 >= ? AND {split_clause}
        ORDER BY sha256
        LIMIT 1
        """,
        (label, pivot),
    ).fetchone()
    if row is not None:
        return str(row[0]), int(row[1])
    row = conn.execute(
        f"""
        SELECT sha256, is_malware
        FROM meta
        WHERE is_malware = ? AND {split_clause}
        ORDER BY sha256
        LIMIT 1
        """,
        (label,),
    ).fetchone()
    if row is None:
        return None
    return str(row[0]), int(row[1])


def sample_vectors(
    *,
    meta_db: Path,
    lmdb_dir: Path,
    sample_size: int,
    seed: int,
    max_attempts: int,
    postprocess: str,
    split: str,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
    target_per_label = {0: sample_size // 2, 1: sample_size - sample_size // 2}
    by_label = {0: 0, 1: 0}
    rows: list[np.ndarray] = []
    labels: list[int] = []
    hashes: list[str] = []
    seen: set[str] = set()
    attempts = 0
    lmdb_missing = 0
    duplicate = 0
    split_clause = split_where_clause(split)

    conn = sqlite3.connect(f"file:{meta_db}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    env = lmdb.open(str(lmdb_dir), readonly=True, lock=False, readahead=False, max_readers=1)
    try:
        with env.begin(buffers=True) as txn:
            while sum(by_label.values()) < sample_size and attempts < max_attempts:
                label = 0 if by_label[0] < target_per_label[0] else 1
                if by_label[label] >= target_per_label[label]:
                    label = 1 - label
                pivot = stable_pivot(seed, label, attempts)
                attempts += 1
                selected = query_one_sha(conn, label, pivot, split_clause)
                if selected is None:
                    continue
                sha, actual_label = selected
                if sha in seen:
                    duplicate += 1
                    continue
                blob = txn.get(sha.encode("ascii"))
                if blob is None:
                    lmdb_missing += 1
                    seen.add(sha)
                    continue
                try:
                    vector = unpack_feature_vector(bytes(blob), postprocess)
                except Exception as exc:  # noqa: BLE001 - keep sampling while reporting count.
                    print(f"skip undecodable feature vector sha256={sha}: {exc}")
                    seen.add(sha)
                    continue
                rows.append(vector)
                labels.append(actual_label)
                hashes.append(sha)
                seen.add(sha)
                by_label[actual_label] += 1
    finally:
        env.close()
        conn.close()

    if len(rows) < sample_size:
        raise RuntimeError(
            f"sampled only {len(rows)} rows out of requested {sample_size}; "
            f"attempts={attempts}, lmdb_missing={lmdb_missing}, duplicate={duplicate}"
        )
    stats = {"attempts": attempts, "lmdb_missing": lmdb_missing, "duplicate": duplicate, "label_0": by_label[0], "label_1": by_label[1]}
    return np.vstack(rows).astype(np.float32), np.asarray(labels, dtype=np.int8), hashes, stats
