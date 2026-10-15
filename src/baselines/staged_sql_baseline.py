"""B4 — Disk-staged SQL pushdown baseline (IPC/disk serialization penalty).

Runs DuckDB SQL pushdown, exports the filtered slice to a temporary Parquet
file on disk, and reloads it via pandas — isolating the serialization cost
that the proposed zero-copy Arrow handoff eliminates.
"""
from __future__ import annotations

import os
import tempfile
import time
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa

from engine.schemas import AnalyticalIntent
from engine.sieve_executor import build_feature_matrix, run_kernel
from ._common import KERNEL_FEATURES, where_sql


def run(intent: AnalyticalIntent, clauses: list[dict], parquet_path: str,
        sort_rows: bool = True) -> dict:
    tel: dict[str, Any] = {"query_id": intent.query_id, "pipeline": "B4_staged_sql"}
    con = duckdb.connect()
    try:
        cols = ", ".join(intent.target_columns)
        sql = f"SELECT {cols} FROM read_parquet('{parquet_path}') WHERE {where_sql(clauses)}"

        t0 = time.perf_counter_ns()
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            stage_path = tmp.name
        con.execute(f"COPY ({sql}) TO '{stage_path}' (FORMAT PARQUET)")
        tel["t_pushdown_ms"] = (time.perf_counter_ns() - t0) / 1e6

        t0 = time.perf_counter_ns()
        df = pd.read_parquet(stage_path)
        tel["t_stage_read_ms"] = (time.perf_counter_ns() - t0) / 1e6
        tel["bytes_materialized"] = int(df.memory_usage(deep=True).sum())

        t0 = time.perf_counter_ns()
        table = pa.Table.from_pandas(df, preserve_index=False)
        features = KERNEL_FEATURES[intent.downstream_task]
        X = build_feature_matrix(table, features, sort_rows=sort_rows)
        tel["t_handoff_ms"] = (time.perf_counter_ns() - t0) / 1e6
        tel["n_rows_kernel"] = int(X.shape[0])
        tel["bytes_X"] = int(X.nbytes)

        t0 = time.perf_counter_ns()
        result = run_kernel(intent.downstream_task, X)
        tel["t_kernel_ms"] = (time.perf_counter_ns() - t0) / 1e6
    finally:
        con.close()
        if os.path.exists(stage_path):
            os.unlink(stage_path)

    tel["t_tool_ms"] = (
        tel["t_pushdown_ms"] + tel["t_stage_read_ms"] + tel["t_handoff_ms"] + tel["t_kernel_ms"]
    )
    return {"result": result, "telemetry": tel}
