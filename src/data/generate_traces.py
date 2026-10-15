#!/usr/bin/env python3
"""Phase 1 — Frozen trace fixtures (trace-driven replay, zero LLM API cost).

Generates data/benchmark_intents.json:
  - 15 single-turn ML traces  (3 kernels x 5 selectivity configs; 1%/5%/20% targets
    derived from REAL quantiles of data/trips_1m.parquet)
  - 10 dirty/adversarial schema traces (aliases + misspelled categoricals)
  - 15 multi-turn conversational traces (5 sessions x 3 turns, lineage via parent_turn_id)

All selectivities are computed from the actual 1M-row dataset — no hardcoded
thresholds.
"""
from __future__ import annotations

import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
PARQUET_1M = os.path.join(DATA_DIR, "trips_1m.parquet")
OUT_PATH = os.path.join(DATA_DIR, "benchmark_intents.json")

KERNEL_FEATURES = {
    "isolation_forest": ["trip_distance", "fare_amount", "total_amount"],
    "linear_regression": ["trip_duration_minutes", "total_amount"],
    "kmeans_clustering": ["trip_distance", "tip_amount", "total_amount"],
}

ALIAS_TRACES = [
    # (alias columns, misspelled predicate) -> exercises the resolver
    (["dist", "fare"], "borough = 'Manhatten'"),
    (["duration", "fare"], "borough = 'Quens'"),
    (["tip", "dist"], "borough = 'Brooklin'"),
    (["dist", "duration", "tip"], "borough = 'Bronks'"),
    (["fare", "duration"], "borough = 'Staten Island'"),
    (["dist"], "borough = 'manhattan'"),
    (["fare", "tip"], "borough = 'MANHATTAN'"),
    (["duration"], "borough = 'Queens '"),
    (["dist", "fare", "tip"], "borough = 'brooklyn'"),
    (["tip", "duration"], "borough = 'bronx'"),
]


def q(con, table: str, col: str, frac: float) -> float:
    return float(
        con.execute(
            f"SELECT approx_quantile({col}, {frac}) FROM {table}"
        ).fetchone()[0]
    )


def selectivity(con, table: str, where: str) -> float:
    total = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    hit = con.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}").fetchone()[0]
    return hit / total


def main() -> int:
    import duckdb

    assert os.path.exists(PARQUET_1M), f"missing {PARQUET_1M}; run fetch_nyctlc.py first"
    con = duckdb.connect()
    con.execute(f"CREATE VIEW trips AS SELECT * FROM read_parquet('{PARQUET_1M}')")

    # Real quantiles for selectivity targeting.
    quants = {}
    for col in ("trip_distance", "fare_amount", "total_amount", "tip_amount"):
        for frac in (0.75, 0.80, 0.90, 0.95, 0.99):
            quants[(col, frac)] = q(con, "trips", col, frac)
    print("[generate_traces] quantiles:")
    for k, v in sorted(quants.items()):
        print(f"  {k}: {v:.3f}")

    traces = []

    # ---- 15 single-turn ML traces: 3 kernels x 5 configs -------------------
    kernel_pred_templates = {
        "isolation_forest": [
            ("trip_distance", ">", quants[("trip_distance", 0.99)]),   # ~1%
            ("fare_amount", ">", quants[("fare_amount", 0.95)]),       # ~5%
            ("total_amount", ">", quants[("total_amount", 0.80)]),     # ~20%
            None,  # borough-combo below
            None,
        ],
        "linear_regression": [
            ("trip_distance", ">", quants[("trip_distance", 0.99)]),
            ("fare_amount", ">", quants[("fare_amount", 0.95)]),
            ("total_amount", ">", quants[("total_amount", 0.80)]),
            None,
            None,
        ],
        "kmeans_clustering": [
            ("trip_distance", ">", quants[("trip_distance", 0.99)]),
            ("fare_amount", ">", quants[("fare_amount", 0.95)]),
            ("total_amount", ">", quants[("total_amount", 0.80)]),
            None,
            None,
        ],
    }
    combo_filters = [
        [("borough", "=", "Manhattan"), ("trip_distance", ">", quants[("trip_distance", 0.90)])],
        [("borough", "=", "Queens"), ("fare_amount", ">", quants[("fare_amount", 0.75)])],
    ]

    tid = 0
    for kernel, features in KERNEL_FEATURES.items():
        for ci, tmpl in enumerate(kernel_pred_templates[kernel]):
            if tmpl is None:
                preds = [
                    {"column": c, "op": o, "value": v}
                    for c, o, v in combo_filters[ci - 3]
                ]
            else:
                c, o, v = tmpl
                preds = [{"column": c, "op": o, "value": v}]
            where = " AND ".join(
                f"{p['column']} {p['op']} '{p['value']}'"
                if isinstance(p["value"], str)
                else f"{p['column']} {p['op']} {p['value']}"
                for p in preds
            )
            sel = selectivity(con, "trips", where)
            cols = sorted(set(features) | {p["column"] for p in preds})
            tid += 1
            traces.append(
                {
                    "query_id": f"single_{tid:02d}",
                    "trace_type": "single_turn",
                    "target_columns": cols,
                    "filter_predicates": preds,
                    "downstream_task": kernel,
                    "session_id": None,
                    "parent_turn_id": None,
                    "expected_selectivity": round(sel, 4),
                }
            )

    # ---- 10 dirty/adversarial schema traces --------------------------------
    for i, (alias_cols, pred_raw) in enumerate(ALIAS_TRACES, start=1):
        kernel = list(KERNEL_FEATURES)[i % 3]
        tid += 1
        traces.append(
            {
                "query_id": f"dirty_{i:02d}",
                "trace_type": "dirty",
                "target_columns": sorted(set(alias_cols) | set(KERNEL_FEATURES[kernel])),
                "filter_predicates_raw": pred_raw,
                "filter_predicates": [],
                "downstream_task": kernel,
                "session_id": None,
                "parent_turn_id": None,
                "note": "unresolved aliases + misspelled categorical; harness must run resolver first",
            }
        )

    # ---- 15 multi-turn traces (5 sessions x 3 turns) -------------------------
    base6 = ["trip_distance", "trip_duration_minutes", "fare_amount",
             "tip_amount", "total_amount", "borough"]
    session_params = [
        ("Manhattan", 1.0, 25.0),
        ("Queens", 1.5, 30.0),
        ("Brooklyn", 1.0, 20.0),
        ("Manhattan", 2.0, 35.0),
        ("Queens", 1.0, 25.0),
    ]
    for s, (boro, d1, f2) in enumerate(session_params, start=1):
        sid = f"session_{s:02d}"
        t1 = f"{sid}_t1"
        traces.append(
            {
                "query_id": t1,
                "trace_type": "multiturn",
                "target_columns": base6,
                "filter_predicates": [
                    {"column": "borough", "op": "=", "value": boro},
                    {"column": "trip_distance", "op": ">", "value": d1},
                ],
                "downstream_task": "linear_regression",
                "session_id": sid,
                "parent_turn_id": None,
            }
        )
        t2 = f"{sid}_t2"
        traces.append(
            {
                "query_id": t2,
                "trace_type": "multiturn",
                "target_columns": base6,
                "filter_predicates": [
                    {"column": "fare_amount", "op": ">", "value": f2},
                ],
                "downstream_task": "isolation_forest",
                "session_id": sid,
                "parent_turn_id": t1,
            }
        )
        traces.append(
            {
                "query_id": t2.replace("_t2", "_t3"),
                "trace_type": "multiturn",
                "target_columns": base6,
                "filter_predicates": [
                    {"column": "tip_amount", "op": ">", "value": 0.0},
                ],
                "downstream_task": "kmeans_clustering",
                "session_id": sid,
                "parent_turn_id": t2,
            }
        )

    payload = {
        "dataset": "data/trips_1m.parquet",
        "n_traces": len(traces),
        "traces": traces,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    by_type = {}
    for t in traces:
        by_type[t["trace_type"]] = by_type.get(t["trace_type"], 0) + 1
    print(f"[generate_traces] wrote {OUT_PATH}: {len(traces)} traces {by_type}")
    singles = [t for t in traces if t["trace_type"] == "single_turn"]
    print("[generate_traces] single-turn selectivities:")
    for t in singles:
        print(f"  {t['query_id']} {t['downstream_task']}: {t['expected_selectivity']:.3%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
