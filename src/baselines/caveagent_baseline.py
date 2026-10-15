"""B2 — Stateful opaque DataFrame cache (CaveAgent-style, arXiv:2601.01569).

Loads the full pandas DataFrame ONCE per session and persists the Python
object in a dictionary across turns. Each turn performs pandas boolean
masking on the cached object — no columnar projection pushdown, no lineage
metadata, no view planning. This is the "opaque object state" design the
IMG is differentiated against.
"""
from __future__ import annotations

import time
from typing import Any, Optional

import pandas as pd
import pyarrow as pa

from engine.schemas import AnalyticalIntent
from engine.sieve_executor import build_feature_matrix, run_kernel
from ._common import KERNEL_FEATURES, pandas_mask


class CaveAgentBaseline:
    def __init__(self):
        self._sessions: dict[str, pd.DataFrame] = {}
        self._loads = 0

    def run(
        self,
        intent: AnalyticalIntent,
        clauses: list[dict],
        parquet_path: str,
        sort_rows: bool = True,
    ) -> dict:
        tel: dict[str, Any] = {"query_id": intent.query_id, "pipeline": "B2_caveagent"}
        session = intent.session_id or intent.query_id

        t0 = time.perf_counter_ns()
        if session not in self._sessions:
            df = pd.read_parquet(parquet_path)
            self._sessions[session] = df
            self._loads += 1
            cache_hit = False
        else:
            df = self._sessions[session]
            cache_hit = True
        tel["t_io_ms"] = (time.perf_counter_ns() - t0) / 1e6
        tel["cache_hit"] = cache_hit
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

    def reset(self) -> None:
        self._sessions.clear()
        self._loads = 0
