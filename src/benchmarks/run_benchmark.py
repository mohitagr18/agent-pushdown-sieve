#!/usr/bin/env python3
"""Phase 5 — dual-track benchmark sweep & fine-grained telemetry.

Experiment 1: scale sweep — 15 single-turn traces x {10k, 100k, 1M} x
              {B1, B2, B3, B4, Proposed}. 5 warm trials; mean + p95.
Experiment 2: multi-turn IMG ablation on 1M — 5 sessions x 3 turns x
              {B1, B2, B3, Proposed-no-IMG, Proposed-with-IMG}.
Experiment 3: resolver resilience (10 dirty traces, healing verification)
              + vocabulary cardinality sweep (1e1..1e5).

Parity: max |parity_vector - reference| <= 1e-5 vs B1 (and cross-checked
B3/B4/Proposed).

Outputs: results/benchmark_results.parquet, results/benchmark_results.json,
         results/summary_tables.md
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from baselines import caveagent_baseline, pandas_baseline, polars_baseline, staged_sql_baseline  # noqa: E402
from baselines._common import KERNEL_FEATURES, resolve_trace  # noqa: E402
from benchmarks.profiler import atti, profile_call, sigma_ship  # noqa: E402
from engine.img_graph import IntermediateMaterializationGraph  # noqa: E402
from engine.resolver import SchemaResolver  # noqa: E402
from engine.schemas import AnalyticalIntent  # noqa: E402
from engine.sieve_executor import SieveExecutor  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
N_TRIALS = 5
PARITY_EPS = 1e-5

PIPELINES_EXP1 = ["B1", "B2", "B3", "B4", "Proposed"]
PIPELINES_EXP2 = ["B1", "B2", "B3", "Proposed_noIMG", "Proposed_IMG"]


def load_traces() -> list[dict]:
    with open(os.path.join(DATA_DIR, "benchmark_intents.json")) as f:
        return json.load(f)["traces"]


def cumulative_clauses(trace: dict, by_id: dict) -> list[dict]:
    """Walk parent_turn_id chain -> cumulative predicate dicts (root first)."""
    chain: list[dict] = []
    t: dict | None = trace
    while t is not None:
        chain.append(t)
        t = by_id.get(t.get("parent_turn_id")) if t.get("parent_turn_id") else None
    chain.reverse()
    out: list[dict] = []
    for t in chain:
        out.extend(t.get("filter_predicates", []))
    return out


def run_baseline(name: str, intent: AnalyticalIntent, clauses: list[dict],
                 parquet_path: str, b2: "caveagent_baseline.CaveAgentBaseline") -> dict:
    if name == "B1":
        return pandas_baseline.run(intent, clauses, parquet_path)
    if name == "B2":
        return b2.run(intent, clauses, parquet_path)
    if name == "B3":
        return polars_baseline.run(intent, clauses, parquet_path)
    if name == "B4":
        return staged_sql_baseline.run(intent, clauses, parquet_path)
    raise ValueError(name)


def parity_delta(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n == 0 or len(a) != len(b):
        return float("inf")
    return float(max(abs(x - y) for x, y in zip(a[:n], b[:n])))


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    traces = load_traces()
    by_id = {t["query_id"]: t for t in traces}
    singles = [t for t in traces if t["trace_type"] == "single_turn"]
    sessions: dict[str, list[dict]] = {}
    for t in traces:
        if t["trace_type"] == "multiturn":
            sessions.setdefault(t["session_id"], []).append(t)
    for s in sessions.values():
        s.sort(key=lambda t: t["query_id"])

    scales = {
        "10k": os.path.join(DATA_DIR, "trips_10k.parquet"),
        "100k": os.path.join(DATA_DIR, "trips_100k.parquet"),
        "1M": os.path.join(DATA_DIR, "trips_1m.parquet"),
    }
    resolvers = {k: SchemaResolver(p) for k, p in scales.items()}
    rows: list[dict] = []

    # ================= Experiment 1: scale sweep =================
    print("=== Experiment 1: scale sweep ===", flush=True)
    for scale, parquet_path in scales.items():
        resolver = resolvers[scale]
        for trace in singles:
            intent_r, clauses = resolve_trace(trace, resolver)
            for pipe in PIPELINES_EXP1:
                trial_tools: list[float] = []
                trial_attis: list[float] = []
                trial_sigmas: list[float] = []
                trial_rss: list[float] = []
                tel0: dict = {}
                for trial in range(N_TRIALS):
                    gc.collect()
                    # B2 is instantiated per trial: each single-turn trace is
                    # an independent agent turn, so the session cache is cold.
                    b2 = caveagent_baseline.CaveAgentBaseline()
                    if pipe == "Proposed":
                        ex = SieveExecutor(resolver)
                        out, peak = profile_call(
                            lambda: ex.execute(
                                AnalyticalIntent(
                                    query_id=intent_r.query_id,
                                    target_columns=intent_r.target_columns,
                                    filter_predicates=intent_r.filter_predicates,
                                    downstream_task=intent_r.downstream_task,
                                )
                            )
                        )
                    else:
                        out, peak = profile_call(
                            lambda: run_baseline(pipe, intent_r, clauses, parquet_path, b2)
                        )
                    t = out["telemetry"]
                    trial_tools.append(t["t_tool_ms"])
                    trial_attis.append(atti(t["t_tool_ms"]))
                    trial_sigmas.append(sigma_ship(t.get("bytes_materialized", t.get("bytes_arrow", 0)),
                                                   t.get("bytes_X", 1)))
                    trial_rss.append(peak)
                    tel0 = t
                rows.append({
                    "experiment": "exp1_scale",
                    "scale": scale,
                    "query_id": trace["query_id"],
                    "kernel": trace["downstream_task"],
                    "pipeline": pipe,
                    "t_tool_mean_ms": float(np.mean(trial_tools)),
                    "t_tool_p95_ms": float(np.percentile(trial_tools, 95)),
                    "atti_mean": float(np.mean(trial_attis)),
                    "sigma_ship_mean": float(np.mean(trial_sigmas)),
                    "peak_rss_mb_mean": float(np.mean(trial_rss)),
                    "parity_ref": "self",
                    "n_rows_kernel": tel0.get("n_rows_kernel"),
                })
            print(f"  scale={scale} done", flush=True)

    # ================= Experiment 2: multi-turn IMG ablation =================
    print("=== Experiment 2: multi-turn IMG ablation (1M) ===", flush=True)
    parquet_1m = scales["1M"]
    resolver = resolvers["1M"]
    for sid, turns in sorted(sessions.items()):
        for pipe in PIPELINES_EXP2:
            b2 = caveagent_baseline.CaveAgentBaseline()
            img = IntermediateMaterializationGraph() if pipe == "Proposed_IMG" else None
            for trace in turns:
                cum = cumulative_clauses(trace, by_id)
                intent_r, clauses = resolve_trace(
                    {**trace, "filter_predicates": cum}, resolver
                )
                trial_tools = []
                trial_rss = []
                for trial in range(N_TRIALS):
                    gc.collect()
                    if pipe.startswith("Proposed"):
                        from engine.schemas import FilterClause  # noqa: E402

                        if pipe == "Proposed_IMG":
                            # IMG turn intent carries ONLY its own predicates;
                            # the parent view encodes ancestors. The
                            # executor's internal resolver binds/repairs them.
                            turn_intent = AnalyticalIntent(
                                query_id=trace["query_id"],
                                target_columns=list(trace["target_columns"]),
                                filter_predicates=[
                                    FilterClause(**p)
                                    for p in trace.get("filter_predicates", [])
                                ],
                                downstream_task=trace["downstream_task"],
                                session_id=trace.get("session_id"),
                                parent_turn_id=trace.get("parent_turn_id"),
                            )
                        else:
                            # No-IMG ablation: cumulative predicate against the
                            # base table (same semantics as B1/B2/B3).
                            turn_intent = AnalyticalIntent(
                                query_id=intent_r.query_id,
                                target_columns=intent_r.target_columns,
                                filter_predicates=intent_r.filter_predicates,
                                downstream_task=intent_r.downstream_task,
                                session_id=intent_r.session_id,
                            )
                        ex = SieveExecutor(resolver, img=img)
                        out, peak = profile_call(lambda: ex.execute(turn_intent))
                        if img is not None:
                            img.register(
                                turn_id=trace["query_id"],
                                arrow_table=out["arrow_table"],
                                source_table=parquet_1m,
                                projected_columns=out["resolved_columns"],
                                predicate_clauses=[out["resolved_where"]],
                                session_id=trace.get("session_id"),
                                parent_turn_id=trace.get("parent_turn_id"),
                            )
                    else:
                        out, peak = profile_call(
                            lambda: run_baseline(pipe, intent_r, clauses, parquet_1m, b2)
                        )
                    trial_tools.append(out["telemetry"]["t_tool_ms"])
                    trial_rss.append(peak)
                t = out["telemetry"]
                rows.append({
                    "experiment": "exp2_multiturn",
                    "scale": "1M",
                    "query_id": trace["query_id"],
                    "kernel": trace["downstream_task"],
                    "pipeline": pipe,
                    "t_tool_mean_ms": float(np.mean(trial_tools)),
                    "t_tool_p95_ms": float(np.percentile(trial_tools, 95)),
                    "atti_mean": float(np.mean([atti(x) for x in trial_tools])),
                    "sigma_ship_mean": sigma_ship(t.get("bytes_materialized", t.get("bytes_arrow", 0)),
                                                  t.get("bytes_X", 1)),
                    "peak_rss_mb_mean": float(np.mean(trial_rss)),
                    "img_view_used": t.get("img_view_used"),
                    "n_rows_kernel": t.get("n_rows_kernel"),
                })
        print(f"  {sid} done", flush=True)

    # ================= Experiment 3: resolver resilience =================
    print("=== Experiment 3: resolver resilience ===", flush=True)
    resolver = resolvers["1M"]
    dirty = [t for t in traces if t["trace_type"] == "dirty"]
    healed = 0
    for trace in dirty:
        try:
            intent_r, clauses = resolve_trace(trace, resolver)
            healed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED to heal {trace['query_id']}: {e}")
    rows.append({
        "experiment": "exp3_resolver",
        "scale": "1M",
        "query_id": "dirty_all",
        "kernel": "-",
        "pipeline": "resolver",
        "t_tool_mean_ms": 0.0,
        "t_tool_p95_ms": 0.0,
        "atti_mean": 0.0,
        "sigma_ship_mean": 0.0,
        "peak_rss_mb_mean": 0.0,
        "healed": healed,
        "n_dirty": len(dirty),
    })
    vocab_curve = resolver.benchmark_vocabulary_scaling()
    for k, v in vocab_curve.items():
        rows.append({
            "experiment": "exp3_vocab",
            "scale": "-",
            "query_id": f"vocab_{k}",
            "kernel": "-",
            "pipeline": "resolver",
            "t_tool_mean_ms": v,
            "t_tool_p95_ms": v,
            "atti_mean": 0.0,
            "sigma_ship_mean": 0.0,
            "peak_rss_mb_mean": 0.0,
        })
    print(f"  healed {healed}/{len(dirty)} dirty traces; vocab curve done", flush=True)

    # ================= Parity verification =================
    print("=== Parity verification (B1 reference) ===", flush=True)
    max_delta = 0.0
    parity_rows: list[dict] = []
    for scale, parquet_path in scales.items():
        resolver = resolvers[scale]
        for trace in singles:
            intent_r, clauses = resolve_trace(trace, resolver)
            ref = pandas_baseline.run(intent_r, clauses, parquet_path)["result"].parity_vector
            for name, fn in (
                ("B3", lambda: polars_baseline.run(intent_r, clauses, parquet_path)),
                ("B4", lambda: staged_sql_baseline.run(intent_r, clauses, parquet_path)),
            ):
                vec = fn()["result"].parity_vector
                d = parity_delta(ref, vec)
                max_delta = max(max_delta, d)
                parity_rows.append({"scale": scale, "query_id": trace["query_id"],
                                    "pipeline": name, "parity_delta": d})
            ex = SieveExecutor(resolver)
            vec = ex.execute(AnalyticalIntent(
                query_id=intent_r.query_id, target_columns=intent_r.target_columns,
                filter_predicates=intent_r.filter_predicates,
                downstream_task=intent_r.downstream_task))["result"].parity_vector
            d = parity_delta(ref, vec)
            max_delta = max(max_delta, d)
            parity_rows.append({"scale": scale, "query_id": trace["query_id"],
                                "pipeline": "Proposed", "parity_delta": d})
    print(f"  max parity delta: {max_delta:.2e} (eps={PARITY_EPS})", flush=True)
    assert max_delta <= PARITY_EPS, f"PARITY FAILURE: {max_delta} > {PARITY_EPS}"

    # ================= Save outputs =================
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(RESULTS_DIR, "benchmark_results.parquet"), index=False)
    with open(os.path.join(RESULTS_DIR, "benchmark_results.json"), "w") as f:
        json.dump({"rows": rows, "parity": parity_rows,
                   "max_parity_delta": max_delta}, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "summary_tables.md"), "w") as f:
        f.write("# Benchmark Summary Tables\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n\n")
        f.write("## Experiment 1 — Scale sweep (mean T_tool ms)\n\n")
        piv = df[df.experiment == "exp1_scale"].pivot_table(
            index=["scale", "kernel"], columns="pipeline",
            values="t_tool_mean_ms", aggfunc="mean").round(2)
        f.write(piv.to_markdown() + "\n\n")
        f.write("## Experiment 1 — ATTI (mean)\n\n")
        piv = df[df.experiment == "exp1_scale"].pivot_table(
            index=["scale", "kernel"], columns="pipeline",
            values="atti_mean", aggfunc="mean").round(4)
        f.write(piv.to_markdown() + "\n\n")
        f.write("## Experiment 2 — Multi-turn IMG ablation, 1M rows (mean T_tool ms per turn)\n\n")
        piv = df[df.experiment == "exp2_multiturn"].pivot_table(
            index=["query_id"], columns="pipeline",
            values="t_tool_mean_ms", aggfunc="mean").round(2)
        f.write(piv.to_markdown() + "\n\n")
        f.write("## Experiment 3 — Resolver\n\n")
        f.write(f"Healed {healed}/{len(dirty)} dirty traces.\n\n")
        f.write("Vocabulary scaling (ms per resolve):\n\n")
        for k in sorted(vocab_curve):
            f.write(f"- 1e{len(str(k))-1} ({k}): {vocab_curve[k]:.3f} ms\n")
        f.write(f"\n## Parity\n\nMax |Δ| vs B1 reference: {max_delta:.2e} (eps {PARITY_EPS})\n")

    print(f"[run_benchmark] wrote {len(rows)} result rows; max parity delta {max_delta:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
