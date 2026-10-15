"""End-to-end test of the FastMCP server tool handlers."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "server")))

import mcp_server  # noqa: E402


def test_inspect_catalog():
    out = mcp_server.inspect_parquet_catalog("trips_10k")
    assert out["n_columns"] == 21
    names = [c["name"] for c in out["columns"]]
    assert "trip_duration_minutes" in names and "borough" in names
    assert "Manhattan" in out["enum_samples"]["borough"]


def test_execute_sieve_tool():
    intent = {
        "query_id": "mcp_smoke",
        "target_columns": ["trip_distance", "trip_duration_minutes", "total_amount"],
        "filter_predicates": [
            {"column": "borough", "op": "=", "value": "Queens"},
            {"column": "trip_distance", "op": ">", "value": 1.0},
        ],
        "downstream_task": "linear_regression",
    }
    out = mcp_server.execute_analytical_sieve(intent, "trips_10k")
    assert out["query_id"] == "mcp_smoke"
    assert "r2" in out["summary"]
    tel = out["telemetry_ms"]
    assert tel["t_tool"] > 0
    assert abs(tel["t_tool"] - (tel["t_resolve"] + tel["t_pushdown"] + tel["t_handoff"] + tel["t_kernel"])) < 1.0
    assert out["zero_copy_1d_verified"] is True


def test_execute_sieve_with_img_lineage():
    t1 = {
        "query_id": "mcp_t1",
        "target_columns": ["trip_distance", "trip_duration_minutes", "fare_amount",
                           "tip_amount", "total_amount", "borough"],
        "filter_predicates": [
            {"column": "borough", "op": "=", "value": "Manhattan"},
            {"column": "trip_distance", "op": ">", "value": 1.0},
        ],
        "downstream_task": "linear_regression",
        "session_id": "mcp_sess",
    }
    r1 = mcp_server.execute_analytical_sieve(t1, "trips_10k")
    assert r1["img_view_used"] is None
    t2 = {
        "query_id": "mcp_t2",
        "target_columns": ["trip_distance", "trip_duration_minutes", "fare_amount",
                           "tip_amount", "total_amount", "borough"],
        "filter_predicates": [{"column": "fare_amount", "op": ">", "value": 25.0}],
        "downstream_task": "isolation_forest",
        "session_id": "mcp_sess",
        "parent_turn_id": "mcp_t1",
    }
    r2 = mcp_server.execute_analytical_sieve(t2, "trips_10k")
    assert r2["img_view_used"] is not None
