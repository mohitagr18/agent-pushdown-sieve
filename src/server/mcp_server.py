"""Phase 6 — FastMCP server packaging the analytical sieve.

Tools:
  - execute_analytical_sieve(intent): resolver -> two-phase pushdown
    (or IMG parent view when parent_turn_id is supplied) -> ML kernel,
    returning the statistical result + telemetry breakdown.
  - inspect_parquet_catalog(dataset_name): column names, Arrow types, and
    low-cardinality enum samples in <150 tokens.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastmcp import FastMCP  # noqa: E402

from engine.img_graph import IntermediateMaterializationGraph  # noqa: E402
from engine.resolver import SchemaResolver  # noqa: E402
from engine.schemas import AnalyticalIntent  # noqa: E402
from engine.sieve_executor import SieveExecutor  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")

DATASETS = {
    "trips_1m": os.path.join(DATA_DIR, "trips_1m.parquet"),
    "trips_100k": os.path.join(DATA_DIR, "trips_100k.parquet"),
    "trips_10k": os.path.join(DATA_DIR, "trips_10k.parquet"),
}

mcp = FastMCP("columnar-analytics-sieve")

_resolvers: dict[str, SchemaResolver] = {}
_executors: dict[str, SieveExecutor] = {}
_img = IntermediateMaterializationGraph()


def _executor_for(dataset: str) -> SieveExecutor:
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset '{dataset}'; known: {sorted(DATASETS)}")
    if dataset not in _executors:
        _resolvers[dataset] = SchemaResolver(DATASETS[dataset])
        _executors[dataset] = SieveExecutor(_resolvers[dataset], img=_img)
    return _executors[dataset]


@mcp.tool()
def execute_analytical_sieve(intent: dict, dataset: str = "trips_1m") -> dict:
    """Execute a structured analytical intent through the two-phase sieve.

    intent: AnalyticalIntent JSON (query_id, target_columns, filter_predicates,
    downstream_task, session_id, parent_turn_id). When parent_turn_id names a
    previously executed turn, the IMG parent view is used instead of
    rescanning the base Parquet file.
    """
    parsed = AnalyticalIntent(**intent)
    ex = _executor_for(dataset)
    out = ex.execute(parsed)
    # Register every executed turn in the IMG so later turns in the same
    # session can plan refinements against this materialized view.
    _img.register(
        turn_id=parsed.query_id,
        arrow_table=out["arrow_table"],
        source_table=DATASETS[dataset],
        projected_columns=out["resolved_columns"],
        predicate_clauses=[out["resolved_where"]],
        session_id=parsed.session_id,
        parent_turn_id=parsed.parent_turn_id,
    )
    tel = out["telemetry"]
    return {
        "query_id": parsed.query_id,
        "kernel": out["result"].kernel,
        "summary": out["result"].summary,
        "telemetry_ms": {
            "t_resolve": round(tel["t_resolve_ms"], 3),
            "t_pushdown": round(tel["t_pushdown_ms"], 3),
            "t_handoff": round(tel["t_handoff_ms"], 3),
            "t_kernel": round(tel["t_kernel_ms"], 3),
            "t_tool": round(tel["t_tool_ms"], 3),
        },
        "zero_copy_1d_verified": tel["zero_copy_1d_verified"],
        "img_view_used": tel["img_view_used"],
        "n_rows_kernel": tel["n_rows_kernel"],
    }


@mcp.tool()
def inspect_parquet_catalog(dataset_name: str = "trips_1m") -> dict:
    """Return column names, Arrow types, and enum samples for a dataset."""
    if dataset_name not in DATASETS:
        raise ValueError(f"unknown dataset '{dataset_name}'")
    path = DATASETS[dataset_name]
    resolver = _resolvers.get(dataset_name) or SchemaResolver(path)
    _resolvers[dataset_name] = resolver
    cols = [
        {"name": n, "arrow_type": str(resolver._schema.field(n).type)}
        for n in resolver.columns
    ]
    enums: dict[str, list[str]] = {}
    for col in ("borough", "store_and_fwd_flag"):
        if col in resolver.columns:
            enums[col] = sorted(resolver._distinct_strings(col))[:6]
    return {"dataset": dataset_name, "n_columns": len(cols),
            "columns": cols, "enum_samples": enums}


if __name__ == "__main__":
    mcp.run()
