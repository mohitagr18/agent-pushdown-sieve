"""IMG lineage tests: view registration, parent-view planning, speedup."""
import os
import time

import pytest

from engine.img_graph import IntermediateMaterializationGraph
from engine.resolver import SchemaResolver
from engine.schemas import AnalyticalIntent, FilterClause
from engine.sieve_executor import SieveExecutor

BASE_COLS = ["trip_distance", "trip_duration_minutes", "fare_amount",
             "tip_amount", "total_amount", "borough"]


def _turn(qid, preds, task, parent=None):
    return AnalyticalIntent(
        query_id=qid,
        target_columns=list(BASE_COLS),
        filter_predicates=[FilterClause(column=c, op=o, value=v) for c, o, v in preds],
        downstream_task=task,
        session_id="test_session",
        parent_turn_id=parent,
    )


def _register(img, ex, out, turn):
    img.register(
        turn_id=turn.query_id,
        arrow_table=out["arrow_table"],
        source_table=ex.parquet_path,
        projected_columns=out["resolved_columns"],
        predicate_clauses=[out["resolved_where"]],
        session_id=turn.session_id,
        parent_turn_id=turn.parent_turn_id,
    )


def test_multiturn_lineage_and_view_use(resolver: SchemaResolver):
    img = IntermediateMaterializationGraph()
    ex = SieveExecutor(resolver, img=img)

    t1 = _turn("t1", [("borough", "=", "Manhattan"), ("trip_distance", ">", 1.0)],
               "linear_regression")
    out1 = ex.execute(t1)
    assert out1["telemetry"]["img_view_used"] is None
    _register(img, ex, out1, t1)

    t2 = _turn("t2", [("fare_amount", ">", 25.0)], "isolation_forest", parent="t1")
    out2 = ex.execute(t2)
    assert out2["telemetry"]["img_view_used"] is not None, "Turn 2 must plan against parent view"
    _register(img, ex, out2, t2)

    t3 = _turn("t3", [("tip_amount", ">", 0.0)], "kmeans_clustering", parent="t2")
    out3 = ex.execute(t3)
    assert out3["telemetry"]["img_view_used"] is not None, "Turn 3 must plan against parent view"
    _register(img, ex, out3, t3)

    assert img.lineage("t3") == ["t1", "t2", "t3"]
    assert len(img) == 3
    # Row counts must be monotonically non-increasing down the lineage.
    assert out1["telemetry"]["n_rows_kernel"] >= out2["telemetry"]["n_rows_kernel"]


def _evict_page_cache(path: str) -> None:
    """Evict a file's pages from the OS page cache (POSIX_FADV_DONTNEED).

    Makes the 'rescan the base Parquet file' measurement deterministic:
    without this, a warm page cache makes the base scan artificially fast
    and the measured speedup depends on test ordering.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def test_img_child_faster_than_base_rescan(resolver: SchemaResolver, parquet_1m: str):
    """IMG data-plane speedup: child turns query the parent Arrow view instead
    of rescanning the 1M-row base Parquet file.

    Uses a selective base (Staten Island) so the parent view is small. The
    assertion targets the pushdown phase (the data plane the IMG optimizes —
    the sklearn kernel is identical in both paths), plus a directional check
    that total T_tool is also faster.
    """
    img = IntermediateMaterializationGraph()
    ex_img = SieveExecutor(resolver, img=img)
    ex_base = SieveExecutor(resolver, img=None)

    t1 = _turn("t1", [("borough", "=", "Staten Island"), ("trip_distance", ">", 1.0)],
               "linear_regression")
    out1 = ex_img.execute(t1)
    _register(img, ex_img, out1, t1)
    t2 = _turn("t2", [("fare_amount", ">", 20.0)], "linear_regression", parent="t1")
    # Equivalent cumulative predicate against the base table (no IMG).
    t2_base = _turn(
        "t2b",
        [("borough", "=", "Staten Island"), ("trip_distance", ">", 1.0),
         ("fare_amount", ">", 20.0)],
        "linear_regression",
    )
    ex_img.execute(t2)  # warm
    ex_base.execute(t2_base)  # warm

    import statistics

    img_push, base_push, img_tool, base_tool = [], [], [], []
    for _ in range(2):  # warmups
        ex_img.execute(t2)
        ex_base.execute(t2_base)
    for _ in range(7):
        oi = ex_img.execute(t2)
        _evict_page_cache(parquet_1m)  # force a true disk rescan for the base
        ob = ex_base.execute(t2_base)
        img_push.append(oi["telemetry"]["t_pushdown_ms"])
        base_push.append(ob["telemetry"]["t_pushdown_ms"])
        img_tool.append(oi["telemetry"]["t_tool_ms"])
        base_tool.append(ob["telemetry"]["t_tool_ms"])

    # Minima are robust to contention noise on a shared VM (contention can
    # only slow runs down); the ratio of minima estimates the true speedup.
    pushdown_speedup = min(base_push) / max(min(img_push), 1e-9)
    assert pushdown_speedup >= 10.0, (
        f"IMG pushdown speedup {pushdown_speedup:.1f}x < 10x "
        f"(img min {min(img_push):.1f} ms vs base min {min(base_push):.1f} ms)"
    )
    assert min(img_tool) < min(base_tool), "IMG total T_tool must beat base rescan"
    # Numerical parity between IMG path and base rescan path.
    d = max(abs(a - b) for a, b in zip(
        oi["result"].parity_vector, ob["result"].parity_vector))
    assert d <= 1e-5
