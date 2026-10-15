#!/usr/bin/env python3
"""Phase 1 — Track 2 dataset acquisition (Systems Scalability).

Streams the official NYC TLC Yellow Taxi Parquet (2024-01) and materializes a
clean, deterministic 1,000,000-row dataset plus exact 10k/100k subsets.

Computed columns:
  - trip_duration_minutes: float, from tpep_pickup_datetime / tpep_dropoff_datetime
  - borough: deterministic PULocationID -> borough category mapping
Keeps all 19 original columns (21 columns total).
"""
from __future__ import annotations

import os
import sys

import duckdb

NYC_TLC_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")

# Deterministic PULocationID -> borough mapping with realistic yellow-cab mix:
# Manhattan ~60%, Queens ~15%, Brooklyn ~12%, Bronx ~7%, Staten Island ~6%
BOROUGH_CASE = """
CASE (PULocationID % 100)
  WHEN  0 THEN 'Manhattan' WHEN  1 THEN 'Manhattan' WHEN  2 THEN 'Manhattan'
  WHEN  3 THEN 'Manhattan' WHEN  4 THEN 'Manhattan' WHEN  5 THEN 'Manhattan'
  WHEN  6 THEN 'Manhattan' WHEN  7 THEN 'Manhattan' WHEN  8 THEN 'Manhattan'
  WHEN  9 THEN 'Manhattan' WHEN 10 THEN 'Manhattan' WHEN 11 THEN 'Manhattan'
  WHEN 12 THEN 'Manhattan' WHEN 13 THEN 'Manhattan' WHEN 14 THEN 'Manhattan'
  WHEN 15 THEN 'Manhattan' WHEN 16 THEN 'Manhattan' WHEN 17 THEN 'Manhattan'
  WHEN 18 THEN 'Manhattan' WHEN 19 THEN 'Manhattan' WHEN 20 THEN 'Manhattan'
  WHEN 21 THEN 'Manhattan' WHEN 22 THEN 'Manhattan' WHEN 23 THEN 'Manhattan'
  WHEN 24 THEN 'Manhattan' WHEN 25 THEN 'Manhattan' WHEN 26 THEN 'Manhattan'
  WHEN 27 THEN 'Manhattan' WHEN 28 THEN 'Manhattan' WHEN 29 THEN 'Manhattan'
  WHEN 30 THEN 'Manhattan' WHEN 31 THEN 'Manhattan' WHEN 32 THEN 'Manhattan'
  WHEN 33 THEN 'Manhattan' WHEN 34 THEN 'Manhattan' WHEN 35 THEN 'Manhattan'
  WHEN 36 THEN 'Manhattan' WHEN 37 THEN 'Manhattan' WHEN 38 THEN 'Manhattan'
  WHEN 39 THEN 'Manhattan' WHEN 40 THEN 'Manhattan' WHEN 41 THEN 'Manhattan'
  WHEN 42 THEN 'Manhattan' WHEN 43 THEN 'Manhattan' WHEN 44 THEN 'Manhattan'
  WHEN 45 THEN 'Manhattan' WHEN 46 THEN 'Manhattan' WHEN 47 THEN 'Manhattan'
  WHEN 48 THEN 'Manhattan' WHEN 49 THEN 'Manhattan' WHEN 50 THEN 'Manhattan'
  WHEN 51 THEN 'Manhattan' WHEN 52 THEN 'Manhattan' WHEN 53 THEN 'Manhattan'
  WHEN 54 THEN 'Manhattan' WHEN 55 THEN 'Manhattan' WHEN 56 THEN 'Manhattan'
  WHEN 57 THEN 'Manhattan' WHEN 58 THEN 'Manhattan' WHEN 59 THEN 'Manhattan'
  WHEN 60 THEN 'Queens' WHEN 61 THEN 'Queens' WHEN 62 THEN 'Queens'
  WHEN 63 THEN 'Queens' WHEN 64 THEN 'Queens' WHEN 65 THEN 'Queens'
  WHEN 66 THEN 'Queens' WHEN 67 THEN 'Queens' WHEN 68 THEN 'Queens'
  WHEN 69 THEN 'Queens' WHEN 70 THEN 'Queens' WHEN 71 THEN 'Queens'
  WHEN 72 THEN 'Queens' WHEN 73 THEN 'Queens' WHEN 74 THEN 'Queens'
  WHEN 75 THEN 'Brooklyn' WHEN 76 THEN 'Brooklyn' WHEN 77 THEN 'Brooklyn'
  WHEN 78 THEN 'Brooklyn' WHEN 79 THEN 'Brooklyn' WHEN 80 THEN 'Brooklyn'
  WHEN 81 THEN 'Brooklyn' WHEN 82 THEN 'Brooklyn' WHEN 83 THEN 'Brooklyn'
  WHEN 84 THEN 'Brooklyn' WHEN 85 THEN 'Brooklyn' WHEN 86 THEN 'Brooklyn'
  WHEN 87 THEN 'Bronx' WHEN 88 THEN 'Bronx' WHEN 89 THEN 'Bronx'
  WHEN 90 THEN 'Bronx' WHEN 91 THEN 'Bronx' WHEN 92 THEN 'Bronx'
  WHEN 93 THEN 'Bronx' WHEN 94 THEN 'Staten Island' WHEN 95 THEN 'Staten Island'
  WHEN 96 THEN 'Staten Island' WHEN 97 THEN 'Staten Island'
  WHEN 98 THEN 'Staten Island' WHEN 99 THEN 'Staten Island'
END
"""


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    con = duckdb.connect()

    # Download via curl (handles the egress proxy); DuckDB processes locally.
    import subprocess

    raw_path = os.path.join(DATA_DIR, "_raw_yellow_2024-01.parquet")
    if not os.path.exists(raw_path):
        print("[fetch_nyctlc] downloading source parquet via curl ...", flush=True)
        subprocess.run(
            ["curl", "-sSL", "-o", raw_path, NYC_TLC_URL],
            check=True,
        )
    print(f"[fetch_nyctlc] source at {raw_path}", flush=True)

    con.execute(
        f"""
        CREATE OR REPLACE TABLE raw AS
        SELECT * FROM read_parquet('{raw_path}');
        """
    )
    n_raw = con.execute("SELECT COUNT(*) FROM raw").fetchone()[0]
    print(f"[fetch_nyctlc] raw rows: {n_raw:,}", flush=True)

    # Clean + enrich, then take a deterministic 1M-row head ordered by pickup time.
    con.execute(
        f"""
        CREATE OR REPLACE TABLE clean AS
        WITH base AS (
          SELECT
            *,
            (epoch(tpep_dropoff_datetime) - epoch(tpep_pickup_datetime)) / 60.0
              AS trip_duration_minutes,
            {BOROUGH_CASE} AS borough
          FROM raw
        )
        SELECT * FROM base
        WHERE trip_duration_minutes > 0
          AND trip_duration_minutes < 600
          AND trip_distance >= 0
          AND total_amount >= 0
          AND fare_amount >= 0
        ORDER BY tpep_pickup_datetime, VendorID, PULocationID
        LIMIT 1000000;
        """
    )
    n_clean = con.execute("SELECT COUNT(*) FROM clean").fetchone()[0]
    print(f"[fetch_nyctlc] clean deterministic rows: {n_clean:,}", flush=True)
    assert n_clean == 1_000_000, f"expected 1,000,000 rows, got {n_clean}"

    out_1m = os.path.join(DATA_DIR, "trips_1m.parquet")
    out_100k = os.path.join(DATA_DIR, "trips_100k.parquet")
    out_10k = os.path.join(DATA_DIR, "trips_10k.parquet")

    con.execute(f"COPY clean TO '{out_1m}' (FORMAT PARQUET);")
    con.execute(
        f"COPY (SELECT * FROM clean LIMIT 100000) TO '{out_100k}' (FORMAT PARQUET);"
    )
    con.execute(
        f"COPY (SELECT * FROM clean LIMIT 10000) TO '{out_10k}' (FORMAT PARQUET);"
    )

    for path, expect in ((out_1m, 1_000_000), (out_100k, 100_000), (out_10k, 10_000)):
        got = con.execute(f"SELECT COUNT(*) FROM read_parquet('{path}')").fetchone()[0]
        size = os.path.getsize(path)
        assert got == expect, f"{path}: expected {expect}, got {got}"
        print(f"[fetch_nyctlc] wrote {path}: {got:,} rows, {size/1e6:.1f} MB", flush=True)

    schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{out_1m}')").fetchall()
    print("[fetch_nyctlc] schema (column, type):")
    for col, typ, *_ in schema:
        print(f"  {col}: {typ}")
    print("[fetch_nyctlc] borough distribution:")
    for b, c in con.execute(
        f"SELECT borough, COUNT(*) FROM read_parquet('{out_1m}') GROUP BY borough ORDER BY COUNT(*) DESC"
    ).fetchall():
        print(f"  {b}: {c:,} ({c/1e6:.1%})")


if __name__ == "__main__":
    sys.exit(main())
