#!/usr/bin/env python3
"""Phase 1 — Track 1 dataset acquisition (Logical Schema & Resolver Validation).

Pulls 100 validation queries + schema metadata from Hugging Face into
data/sql_validation_100.json. Tries, in order:
  1. birdsql/bird_mini_dev
  2. white-bird/bird-bench
  3. xlangai/spider  (fallback)
"""
from __future__ import annotations

import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_PATH = os.path.join(DATA_DIR, "sql_validation_100.json")


def _load_hf(name: str, split: str = "validation"):
    from datasets import load_dataset

    return load_dataset(name, split=split)


def main() -> int:
    os.makedirs(DATA_DIR, exist_ok=True)

    ds = None
    source = None
    for name, split in (
        ("birdsql/bird_mini_dev", "validation"),
        ("white-bird/bird-bench", "validation"),
        ("xlangai/spider", "validation"),
    ):
        try:
            print(f"[fetch_bird] trying {name} ...", flush=True)
            ds = _load_hf(name, split)
            source = f"{name}:{split}"
            break
        except Exception as e:  # noqa: BLE001 - try next candidate
            print(f"[fetch_bird]   failed: {type(e).__name__}: {str(e)[:160]}")

    if ds is None:
        print("[fetch_bird] ERROR: all dataset candidates failed", file=sys.stderr)
        return 1

    records = []
    for i, row in enumerate(ds):
        if i >= 100:
            break
        records.append(
            {
                "idx": i,
                "question": str(row.get("question", "")),
                "sql": str(row.get("SQL", row.get("query", ""))),
                "db_id": str(row.get("db_id", "")),
                "difficulty": str(row.get("difficulty", "")),
            }
        )

    payload = {"source": source, "n": len(records), "queries": records}
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    size = os.path.getsize(OUT_PATH)
    print(f"[fetch_bird] wrote {OUT_PATH}: {len(records)} queries from {source} ({size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
