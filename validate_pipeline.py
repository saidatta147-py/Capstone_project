"""Lightweight validation for a database produced by run_pipeline.py."""
import sqlite3
from pathlib import Path

db = Path(__file__).parents[1] / "output" / "zepto_catalog.db"
if not db.exists():
    raise SystemExit("Run data_pipeline/run_pipeline.py before this validation.")
with sqlite3.connect(db) as conn:
    count = conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
    foreign_keys = conn.execute("PRAGMA foreign_key_check").fetchall()
    invalid = conn.execute("SELECT COUNT(*) FROM books WHERE rating NOT BETWEEN 1 AND 5 OR price_inr != ROUND(price_gbp * 105.50, 2)").fetchone()[0]
assert count >= 60, f"Expected 60+ books, found {count}"
assert not foreign_keys, foreign_keys
assert invalid == 0, f"Found {invalid} invalid converted/rated rows"
print("Pipeline validation passed.")
