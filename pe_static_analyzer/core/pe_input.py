"""Read PE input without executing it or extracting archives to disk."""
import hashlib
from pathlib import Path

import pyzipper


def read_pe_bytes(path: Path, archived: bool) -> bytes:
    if not archived:
        return path.read_bytes()
    with pyzipper.AESZipFile(path) as archive:
        archive.setpassword(b"infected")
        files = [entry for entry in archive.infolist() if not entry.is_dir()]
        if len(files) != 1:
            raise RuntimeError(f"expected one archive member, got {len(files)}")
        return archive.read(files[0])


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
