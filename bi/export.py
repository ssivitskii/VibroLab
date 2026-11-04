#!/usr/bin/env python3
"""Export saved VibroLab inspections for the BI dashboard."""

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

COLUMNS = (
    "created_at",
    "asset_type",
    "predicted_class",
    "state_key",
    "work_status",
    "confidence",
    "input_type",
)
# Backend status dictionaries, model meta.json classes, and frontend input types.
ALLOWED = {
    1: {"gearbox"},
    2: {
        "ball_fault",
        "combination",
        "inner_race",
        "normal",
        "outer_race",
        "root_crack",
        "surface_wear",
        "tooth_chip",
        "tooth_miss",
    },
    3: {
        "healthy",
        "baseline",
        "warning",
        "monitor",
        "inspect",
        "service",
        "critical",
        "after_maintenance",
    },
    4: {"observe", "inspect", "repair", "replaced"},
    6: {"demo", "file", "sensor", "saved"},
}
QUERY = """SELECT i.created_at, COALESCE(a.asset_type, 'unknown'), i.predicted_class,
                  i.state_key, i.work_status, i.confidence, i.input_type
           FROM inspection AS i LEFT JOIN asset AS a ON a.id = i.asset_id
           ORDER BY i.created_at, i.id"""


def export(source: Path, output: Path) -> int:
    if not source.is_file():
        raise ValueError(f"Source SQLite database does not exist: {source}")
    if source.resolve() == output.resolve():
        raise ValueError("Source and output must be different files")
    # mode=ro prevents SQLite from silently creating an empty database.
    with sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute(QUERY).fetchall()
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        for row in rows:
            safe = list(row)
            for index, allowed in ALLOWED.items():
                if safe[index] not in allowed:
                    safe[index] = "unknown"
            writer.writerow(safe)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="VibroLab SQLite database"
    )
    parser.add_argument("--output", type=Path, required=True, help="UTF-8 CSV output")
    args = parser.parse_args()
    try:
        count = export(args.source, args.output)
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    print(f"Exported {count} inspections to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
