import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from pe_static_analyzer.core.pe_input import read_pe_bytes
from pe_static_analyzer.sorel.data import TRAIN_VALIDATION_SPLIT, VALIDATION_TEST_SPLIT, split_where_clause


class SharedInputTest(unittest.TestCase):
    def test_time_boundaries_assign_each_file_once(self):
        times = [TRAIN_VALIDATION_SPLIT - 1, TRAIN_VALIDATION_SPLIT, VALIDATION_TEST_SPLIT - 1, VALIDATION_TEST_SPLIT]
        with sqlite3.connect(":memory:") as db:
            db.execute("CREATE TABLE samples (rl_fs_t REAL)")
            db.executemany("INSERT INTO samples VALUES (?)", [(t,) for t in times])
            for split, expected in [("train", times[:1]), ("validation", times[1:3]), ("test", times[3:])]:
                found = db.execute(f"SELECT rl_fs_t FROM samples WHERE {split_where_clause(split)} ORDER BY rl_fs_t")
                self.assertEqual([t for t, in found], expected)

    def test_archive_reads_one_file_and_skips_directories(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "example.zip"
            with ZipFile(path, "w") as archive:
                archive.writestr("samples/", b"")
                archive.writestr("samples/example", b"example")
            self.assertEqual(read_pe_bytes(path, archived=True), b"example")
            with ZipFile(path, "a") as archive:
                archive.writestr("second", b"another example")
            with self.assertRaisesRegex(RuntimeError, "expected one archive member"):
                read_pe_bytes(path, archived=True)
