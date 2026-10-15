#!/usr/bin/env python3
"""Phase 1 — Track 2 dataset acquisition (Systems Scalability) — Phase 7 revision.

Streams the official NYC TLC Yellow Taxi Parquet (2024-01) and materializes
deterministic tiers: 10k / 100k / 1M / ~3M rows.

Phase 7 changes vs the original:
  * 6 wide enterprise string/metadata columns are added
    (trip_uuid, driver_license_hash, dispatch_hub_code, shift_audit_signature,
    routing_metadata_json, dispatcher_notes — ~250 bytes of string payload per
    row), so the 1M-row file is a realistic ~250-300 MB enterprise payload.
    This punishes monolithic pd.read_parquet() (SELECT *) while column
    projection pushdown skips the wide columns entirely.
  * A ~3M-row tier (trips_3m.parquet) is materialized from the full cleaned
    source (2,964,624 raw rows).
All string columns are deterministic SQL functions of existing columns
(md5/sha256/lpad) — no randomness, fully reproducible.
"""
from __future__ import annotations

import os
import sys

import duckdb

NYC_TLC_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")

# Deterministic PULocationID -> borough mapping (unchanged from Phase 1).
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

# Phase 7: wide enterprise string/metadata payload (~250 bytes/row), all
# deterministic SQL functions of existing columns.
WIDE_COLUMNS = """
md5(CAST(VendorID AS VARCHAR) || '|' || CAST(tpep_pickup_datetime AS VARCHAR)
    || '|' || CAST(PULocationID AS VARCHAR)) AS trip_uuid,
sha256('drv' || CAST(((PULocationID * 7919 + DOLocationID * 104729) % 1000000) AS VARCHAR))
    AS driver_license_hash,
'HUB-' || lpad(CAST(((PULocationID * 13) % 100000) AS VARCHAR), 5, '0')
    AS dispatch_hub_code,
substr(sha256(CAST(tpep_dropoff_datetime AS VARCHAR) || 'audit'), 1, 44)
    AS shift_audit_signature,
'{"route":"R' || CAST(((PULocationID % 900) + 100) AS VARCHAR)
    || '","hub":"HUB-' || lpad(CAST(((DOLocationID * 13) % 100000) AS VARCHAR), 5, '0')
    || '","prio":' || CAST(((passenger_count % 5) + 1) AS VARCHAR) || '}'
    AS routing_metadata_json,
'Dispatch verified; trip logged under hub manifest #'
    || CAST(((PULocationID * 31 + DOLocationID) % 1000000) AS VARCHAR)
    || ' ref ' || sha256(CAST(tpep_pickup_datetime AS VARCHAR) || 'note')
    AS dispatcher_notes
"""


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    con = duckdb.connect()

    import subprocess

    raw_path = os.path.join(DATA_DIR, "_raw_yellow_2024-01.parquet")
    if not os.path.exists(raw_path):
        print("[fetch_nyctlc] downloading source parquet via curl ...", flush=True)
        subprocess.run(["curl", "-sSL", "-o", raw_path, NYC_TLC_URL], check=True)
    print(f"[fetch_nyctlc] source at {raw_path}", flush=True)

    con.execute(f"CREATE OR REPLACE TABLE raw AS SELECT * FROM read_parquet('{raw_path}');")
    n_raw = con.execute("SELECT COUNT(*) FROM raw").fetchone()[0]
    print(f"[fetch_nyctlc] raw rows: {n_raw:,}", flush=True)

    # Clean + enrich with wide enterprise columns; deterministic order.
    con.execute(
        f"""
        CREATE OR REPLACE TABLE clean AS
        WITH base AS (
          SELECT
            *,
            (epoch(tpep_dropoff_datetime) - epoch(tpep_pickup_datetime)) / 60.0
              AS trip_duration_minutes,
            {BOROUGH_CASE} AS borough,
            {WIDE_COLUMNS}
          FROM raw
        )
        SELECT * FROM base
        WHERE trip_duration_minutes > 0
          AND trip_duration_minutes < 600
          AND trip_distance >= 0
          AND total_amount >= 0
          AND fare_amount >= 0
        ORDER BY tpep_pickup_datetime, VendorID, PULocationID;
        """
    )
    n_clean = con.execute("SELECT COUNT(*) FROM clean").fetchone()[0]
    print(f"[fetch_nyctlc] clean deterministic rows: {n_clean:,}", flush=True)

    tiers = [
        ("trips_3m.parquet", None),          # full cleaned set (~2.9M)
        ("trips_1m.parquet", 1_000_000),
        ("trips_100k.parquet", 100_000),
        ("trips_10k.parquet", 10_000),
    ]
    for fname, limit in tiers:
        path = os.path.join(DATA_DIR, fname)
        q = "SELECT * FROM clean" + (f" LIMIT {limit}" if limit else "")
        con.execute(f"COPY ({q}) TO '{path}' (FORMAT PARQUET);")
        got = con.execute(f"SELECT COUNT(*) FROM read_parquet('{path}')").fetchone()[0]
        size = os.path.getsize(path)
        expect = limit or n_clean
        assert got == expect, f"{path}: expected {expect}, got {got}"
        print(f"[fetch_nyctlc] wrote {path}: {got:,} rows, {size/1e6:.1f} MB", flush=True)

    out_1m = os.path.join(DATA_DIR, "trips_1m.parquet")
    schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{out_1m}')").fetchall()
    print(f"[fetch_nyctlc] schema ({len(schema)} columns):")
    for col, typ, *_ in schema:
        print(f"  {col}: {typ}")
    print("[fetch_nyctlc] borough distribution (1M):")
    for b, c in con.execute(
        f"SELECT borough, COUNT(*) FROM read_parquet('{out_1m}') GROUP BY borough ORDER BY COUNT(*) DESC"
    ).fetchall():
        print(f"  {b}: {c:,} ({c/1e6:.1%})")
    avg_wide = con.execute(
        f"""SELECT AVG(length(trip_uuid) + length(driver_license_hash)
              + length(dispatch_hub_code) + length(shift_audit_signature)
              + length(routing_metadata_json) + length(dispatcher_notes))
            FROM read_parquet('{out_1m}')"""
    ).fetchone()[0]
    print(f"[fetch_nyctlc] avg wide-string payload: {avg_wide:.0f} bytes/row")


if __name__ == "__main__":
    sys.exit(main())
