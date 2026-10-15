"""B3 — Modern streaming Polars baseline (Rust-backed code generation).

Uses pl.scan_parquet(...).filter(...).select(...).collect() on every turn
(lazy pushdown + projection in the Polars engine), then hands the collected
frame to the shared Arrow-backed feature-matrix builder.
"""
from __future__ import annotations

import time
from typing import Any

import polars as pl

from engine.schemas import AnalyticalIntent
from engine.sieve_executor import build_feature_matrix, run_kernel
from ._common import KERNEL_FEATURES, polars_filter


def run(intent: AnalyticalIntent, clauses: list[dict], parquet_path: str) -> dict:
    tel: dict[str, Any] = {"query_id": intent.query_id, "pipeline": "B3_polars"}

    t0 = time.perf_counter_ns()
    lf = pl.scan_parquet(parquet_path)
    lf = polars_filter(lf, clauses).select(intent.target_columns)
    frame = lf.collect()
    tel["t_pushdown_ms"] = (time.perf_counter_ns() - t0) / 1e6
    tel["bytes_materialized"] = int(frame.estimated_size())

    t0 = time.perf_counter_ns()
    table = frame.to_arrow()
    features = KERNEL_FEATURES[intent.downstream_task]
    X = build_feature_matrix(table, features)
    tel["t_handoff_ms"] = (time.perf_counter_ns() - t0) / 1e6
    tel["n_rows_kernel"] = int(X.shape[0])
    tel["bytes_X"] = int(X.nbytes)

    t0 = time.perf_counter_ns()
    result = run_kernel(intent.downstream_task, X)
    tel["t_kernel_ms"] = (time.perf_counter_ns() - t0) / 1e6

    tel["t_tool_ms"] = tel["t_pushdown_ms"] + tel["t_handoff_ms"] + tel["t_kernel_ms"]
    return {"result": result, "telemetry": tel}
