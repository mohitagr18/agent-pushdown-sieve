"""Phase 4 — shared baseline utilities.

All baselines run the EXACT same sklearn estimators (random_state=42) via the
engine's run_kernel() and build their feature matrix with the engine's
build_feature_matrix() so numerical parity is strict and mechanical.

Baselines receive REPAIRED clauses (canonical columns, repaired categorical
values) from the resolver — the same contract the proposed engine consumes.
"""
from __future__ import annotations

import operator
import re
from typing import Any

import pandas as pd
import polars as pl

from engine.resolver import SchemaResolver
from engine.schemas import AnalyticalIntent, FilterClause

KERNEL_FEATURES = {
    "isolation_forest": ["trip_distance", "fare_amount", "total_amount"],
    "linear_regression": ["trip_duration_minutes", "total_amount"],
    "kmeans_clustering": ["trip_distance", "tip_amount", "total_amount"],
}

_RAW_RE = re.compile(r"^\s*([A-Za-z_][\w]*)\s*(=|!=|>=|<=|>|<)\s*'?([^']*)'?\s*$")

_OPS = {
    "=": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}


def parse_raw_predicate(raw: str) -> dict:
    """Parse a controlled raw predicate like \"borough = 'Manhatten'\"."""
    m = _RAW_RE.match(raw)
    if not m:
        raise ValueError(f"cannot parse raw predicate: {raw!r}")
    col, op, val = m.group(1), m.group(2), m.group(3).strip()
    try:
        parsed: Any = float(val)
    except ValueError:
        parsed = val
    return {"column": col, "op": op, "value": parsed}


def resolve_trace(trace: dict, resolver: SchemaResolver) -> tuple[AnalyticalIntent, list[dict]]:
    """Resolve a frozen trace (clean or dirty).

    Returns (repaired AnalyticalIntent with canonical columns, repaired
    predicate clauses as plain dicts).
    """
    if trace.get("filter_predicates"):
        raw_clauses = [dict(p) for p in trace["filter_predicates"]]
    else:
        raw_clauses = [parse_raw_predicate(trace["filter_predicates_raw"])]

    resolved = resolver.resolve(trace["target_columns"], raw_clauses)

    # Rebuild clauses with canonical columns + repaired values.
    repaired_clauses: list[dict] = []
    for p in raw_clauses:
        canon_col, _ = resolver.bind_column(p["column"])
        val, _ = resolver.repair_value(canon_col, p["value"])
        repaired_clauses.append({"column": canon_col, "op": p["op"], "value": val})

    intent = AnalyticalIntent(
        query_id=trace["query_id"],
        target_columns=resolved.columns,
        filter_predicates=[FilterClause(**c) for c in repaired_clauses],
        downstream_task=trace["downstream_task"],
        session_id=trace.get("session_id"),
        parent_turn_id=trace.get("parent_turn_id"),
    )
    return intent, repaired_clauses


def where_sql(clauses: list[dict]) -> str:
    parts = []
    for c in clauses:
        v = c["value"]
        rhs = f"'{v}'".replace("''", "'") if isinstance(v, str) else str(v)
        if isinstance(v, str):
            rhs = "'" + v.replace("'", "''") + "'"
        parts.append(f"{c['column']} {c['op']} {rhs}")
    return " AND ".join(parts) if parts else "TRUE"


def pandas_mask(df: pd.DataFrame, clauses: list[dict]) -> "pd.Series":
    mask = pd.Series(True, index=df.index)
    for c in clauses:
        mask &= _OPS[c["op"]](df[c["column"]], c["value"])
    return mask


def polars_filter(lf: pl.LazyFrame, clauses: list[dict]) -> pl.LazyFrame:
    expr = None
    for c in clauses:
        col = pl.col(c["column"])
        rhs: Any = c["value"]
        op = c["op"]
        e = {
            "=": col == rhs,
            "!=": col != rhs,
            ">": col > rhs,
            ">=": col >= rhs,
            "<": col < rhs,
            "<=": col <= rhs,
        }[op]
        expr = e if expr is None else (expr & e)
    return lf.filter(expr) if expr is not None else lf
