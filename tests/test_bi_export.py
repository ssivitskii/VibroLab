import csv
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

EXPORT = Path(__file__).resolve().parents[1] / "bi" / "export.py"


class ExportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source.db"
        self.output = self.root / "out.csv"
        with sqlite3.connect(self.source) as db:
            db.executescript("""
                CREATE TABLE asset (id TEXT PRIMARY KEY, asset_type TEXT, name TEXT, owner_id TEXT);
                CREATE TABLE inspection (id TEXT PRIMARY KEY, asset_id TEXT, created_at TEXT,
                    predicted_class TEXT, state_key TEXT, work_status TEXT, confidence REAL,
                    input_type TEXT, note TEXT, user_id TEXT);
                INSERT INTO asset VALUES ('asset-id', 'gearbox', 'Private asset', 'user-id');
                INSERT INTO inspection VALUES
                    ('i1', 'asset-id', '2026-01-01 12:00:00', 'normal', 'monitor', 'observe', 0.9,
                     'file', 'secret note', 'user-id'),
                    ('i2', 'asset-id', '2026-01-02 12:00:00', 'tooth_miss', 'critical', 'repair', 0.8,
                     'demo', 'secret note', 'user-id');
            """)

    def tearDown(self):
        self.temp.cleanup()

    def run_export(self, source=None):
        return subprocess.run(
            [
                sys.executable,
                str(EXPORT),
                "--source",
                str(source or self.source),
                "--output",
                str(self.output),
            ],
            capture_output=True,
            text=True,
        )

    def test_exports_saved_inspections_without_private_fields(self):
        result = self.run_export()
        self.assertEqual(result.returncode, 0, result.stderr)
        with self.output.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            rows[0],
            {
                "created_at": "2026-01-01 12:00:00",
                "asset_type": "gearbox",
                "predicted_class": "normal",
                "state_key": "monitor",
                "work_status": "observe",
                "confidence": "0.9",
                "input_type": "file",
            },
        )
        self.assertEqual(rows[1]["predicted_class"], "tooth_miss")
        self.assertEqual(rows[1]["input_type"], "demo")
        self.assertNotIn("secret", self.output.read_text())
        self.assertNotIn("user-id", self.output.read_text())
        self.assertNotIn("Private asset", self.output.read_text())

    def test_unrecognized_free_text_is_not_exported(self):
        with sqlite3.connect(self.source) as db:
            db.execute("UPDATE asset SET asset_type = ?", ("private@example.invalid",))
            db.execute(
                "UPDATE inspection SET predicted_class = ?, state_key = ?, work_status = ?, input_type = ?",
                ("private@example.invalid",) * 4,
            )
        self.assertEqual(self.run_export().returncode, 0)
        with self.output.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            for field in (
                "asset_type",
                "predicted_class",
                "state_key",
                "work_status",
                "input_type",
            ):
                self.assertEqual(row[field], "unknown")
        self.assertNotIn("private@example.invalid", self.output.read_text())

    def test_missing_source_and_invalid_schema_fail_without_creating_output(self):
        result = self.run_export(self.root / "missing.db")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())
        invalid = self.root / "invalid.db"
        sqlite3.connect(invalid).close()
        result = self.run_export(invalid)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())

    def test_empty_database_exports_header(self):
        with sqlite3.connect(self.source) as db:
            db.execute("DELETE FROM inspection")
        self.assertEqual(self.run_export().returncode, 0)
        with self.output.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "created_at")


if __name__ == "__main__":
    unittest.main()
