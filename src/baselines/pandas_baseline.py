"""B1 — Monolithic Pandas baseline (CodeAct / TaskWeaver / LangChain Pandas agent).

Reads the ENTIRE parquet (all columns, all rows) into pandas on every turn,
filters in pandas, converts to Arrow for the shared feature-matrix builder.
Repeats from scratch on every turn (stateless).
"""
from __future__ import annotations

import time
from typing import Any

import pandas as pd
import pyarrow as pa

from engine.schemas import AnalyticalIntent
from engine.sieve_executor import build_feature_matrix, run_kernel
from ._common import KERNEL_FEATURES, pandas_mask


def run(intent: AnalyticalIntent, clauses: list[dict], parquet_path: str,
        sort_rows: bool = True) -> dict:
    tel: dict[str, Any] = {"query_id": intent.query_id, "pipeline": "B1_pandas"}

    t0 = time.perf_counter_ns()
    df = pd.read_parquet(parquet_path)
    tel["t_io_ms"] = (time.perf_counter_ns() - t0) / 1e6
    tel["bytes_materialized"] = int(df.memory_usage(deep=True).sum())

    t0 = time.perf_counter_ns()
    sub = df[pandas_mask(df, clauses)]
    tel["t_filter_ms"] = (time.perf_counter_ns() - t0) / 1e6

    t0 = time.perf_counter_ns()
    table = pa.Table.from_pandas(sub[intent.target_columns], preserve_index=False)
    features = KERNEL_FEATURES[intent.downstream_task]
    X = build_feature_matrix(table, features, sort_rows=sort_rows)
    tel["t_handoff_ms"] = (time.perf_counter_ns() - t0) / 1e6
    tel["n_rows_kernel"] = int(X.shape[0])
    tel["bytes_X"] = int(X.nbytes)

    t0 = time.perf_counter_ns()
    result = run_kernel(intent.downstream_task, X)
    tel["t_kernel_ms"] = (time.perf_counter_ns() - t0) / 1e6

    tel["t_tool_ms"] = tel["t_io_ms"] + tel["t_filter_ms"] + tel["t_handoff_ms"] + tel["t_kernel_ms"]
    return {"result": result, "telemetry": tel}
