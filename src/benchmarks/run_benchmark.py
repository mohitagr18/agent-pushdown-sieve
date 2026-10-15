#!/usr/bin/env python3
"""Phase 5 — dual-track benchmark sweep & fine-grained telemetry (Phase 7 revision).

Experiment 1: scale sweep — 15 single-turn traces x {10k, 100k, 1M} x
              {B1, B2, B3, B4, Proposed} + 3M spot-check (3 traces x {B1, Proposed}).
              5 warm trials; mean + p95. DECOMPOSED stage timings recorded.
Experiment 2: multi-turn IMG ablation on 1M — 5 sessions x 3 turns x
              {B1, B2, B3, Proposed-no-IMG, Proposed-with-IMG}, with
              session cumulative memory accounting.
Experiment 3: resolver resilience (10 dirty traces, healing verification)
              + vocabulary cardinality sweep (1e1..1e5).

Latency mode: production (canonical_sort=False / sort_rows=False) so the
lexsort test-harness overhead does not penalize the data plane. Numerical
parity is verified in a SEPARATE parity-locked pass
(canonical_sort=True / sort_rows=True, eps <= 1e-5).

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
SPOTCHECK_TRACES = ["single_01", "single_06", "single_11"]  # one per kernel


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
                 parquet_path: str, b2: "caveagent_baseline.CaveAgentBaseline",
                 sort_rows: bool = False) -> dict:
    if name == "B1":
        return pandas_baseline.run(intent, clauses, parquet_path, sort_rows=sort_rows)
    if name == "B2":
        return b2.run(intent, clauses, parquet_path, sort_rows=sort_rows)
    if name == "B3":
        return polars_baseline.run(intent, clauses, parquet_path, sort_rows=sort_rows)
    if name == "B4":
        return staged_sql_baseline.run(intent, clauses, parquet_path, sort_rows=sort_rows)
    raise ValueError(name)


def stage_times(pipeline: str, t: dict) -> dict:
    """Normalize per-pipeline telemetry into decomposed stage timings (ms)."""
    t_resolve = float(t.get("t_resolve_ms", 0.0))
    if pipeline == "Proposed":
        t_data = float(t.get("t_pushdown_ms", 0.0))
        t_h1 = float(t.get("t_handoff_1d_ms", 0.0))
        t_h2 = float(t.get("t_handoff_2d_ms", 0.0))
    elif pipeline == "B4":
        t_data = float(t.get("t_pushdown_ms", 0.0)) + float(t.get("t_stage_read_ms", 0.0))
        t_h1 = t_h2 = 0.0
    elif pipeline == "B3":
        t_data = float(t.get("t_pushdown_ms", 0.0))
        t_h1 = t_h2 = 0.0
    else:  # B1, B2: full read + pandas filter
        t_data = float(t.get("t_io_ms", 0.0)) + float(t.get("t_filter_ms", 0.0))
        t_h1 = t_h2 = 0.0
    t_handoff = float(t.get("t_handoff_ms", 0.0))
    t_kernel = float(t.get("t_kernel_ms", 0.0))
    t_tool = float(t.get("t_tool_ms", 0.0))
    return {
        "t_resolve_ms": t_resolve,
        "t_data_ms": t_data,
        "t_handoff_1d_ms": t_h1,
        "t_handoff_2d_ms": t_h2,
        "t_handoff_ms": t_handoff,
        "t_kernel_ms": t_kernel,
        "t_tool_ms": t_tool,
        "bytes_materialized": float(t.get("bytes_materialized", t.get("bytes_arrow", 0))),
        "bytes_X": float(t.get("bytes_X", 1)),
        "n_rows_kernel": t.get("n_rows_kernel"),
    }


def mean_stage(trials: list[dict], key: str) -> float:
    return float(np.mean([s[key] for s in trials]))


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
    spot_path = os.path.join(DATA_DIR, "trips_3m.parquet")
    resolvers = {k: SchemaResolver(p) for k, p in scales.items()}
    rows: list[dict] = []

    def sweep_scale(scale: str, parquet_path: str, resolver: SchemaResolver,
                    trace_subset: list[dict], pipes: list[str], n_trials: int,
                    exp_name: str) -> None:
        for trace in trace_subset:
            intent_r, clauses = resolve_trace(trace, resolver)
            per_pipe_stage: dict[str, list[dict]] = {p: [] for p in pipes}
            per_pipe_tool: dict[str, list[float]] = {p: [] for p in pipes}
            per_pipe_rss: dict[str, list[float]] = {p: [] for p in pipes}
            last_stage: dict[str, dict] = {}
            for pipe in pipes:
                for trial in range(n_trials):
                    gc.collect()
                    # B2 is instantiated per trial: each single-turn trace is
                    # an independent agent turn, so the session cache is cold.
                    b2 = caveagent_baseline.CaveAgentBaseline()
                    if pipe == "Proposed":
                        # Production latency: no canonical sort.
                        ex = SieveExecutor(resolver, canonical_sort=False)
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
                            lambda: run_baseline(pipe, intent_r, clauses,
                                                 parquet_path, b2, sort_rows=False)
                        )
                    st = stage_times(pipe, out["telemetry"])
                    per_pipe_stage[pipe].append(st)
                    per_pipe_tool[pipe].append(st["t_tool_ms"])
                    per_pipe_rss[pipe].append(peak)
                    last_stage[pipe] = st
            # Speedups vs B1, computed after all pipelines ran this trace.
            ref = last_stage["B1"]
            prop = last_stage["Proposed"]
            dp_speedup = ((ref["t_data_ms"] + ref["t_handoff_ms"])
                          / max(prop["t_data_ms"] + prop["t_handoff_ms"], 1e-9))
            ee_speedup = (float(np.mean(per_pipe_tool["B1"]))
                          / max(float(np.mean(per_pipe_tool["Proposed"])), 1e-9))
            for pipe in pipes:
                rows.append({
                    "experiment": exp_name,
                    "scale": scale,
                    "query_id": trace["query_id"],
                    "kernel": trace["downstream_task"],
                    "pipeline": pipe,
                    "t_resolve_mean_ms": mean_stage(per_pipe_stage[pipe], "t_resolve_ms"),
                    "t_data_mean_ms": mean_stage(per_pipe_stage[pipe], "t_data_ms"),
                    "t_handoff_1d_mean_ms": mean_stage(per_pipe_stage[pipe], "t_handoff_1d_ms"),
                    "t_handoff_2d_mean_ms": mean_stage(per_pipe_stage[pipe], "t_handoff_2d_ms"),
                    "t_handoff_mean_ms": mean_stage(per_pipe_stage[pipe], "t_handoff_ms"),
                    "t_kernel_mean_ms": mean_stage(per_pipe_stage[pipe], "t_kernel_ms"),
                    "t_tool_mean_ms": float(np.mean(per_pipe_tool[pipe])),
                    "t_tool_p95_ms": float(np.percentile(per_pipe_tool[pipe], 95)),
                    "atti_mean": float(np.mean([atti(x) for x in per_pipe_tool[pipe]])),
                    "sigma_ship_mean": float(np.mean([
                        sigma_ship(s["bytes_materialized"], s["bytes_X"])
                        for s in per_pipe_stage[pipe]])),
                    "peak_rss_mb_mean": float(np.mean(per_pipe_rss[pipe])),
                    "data_plane_speedup_vs_b1": float(dp_speedup) if pipe == "Proposed" else None,
                    "end_to_end_speedup_vs_b1": float(ee_speedup) if pipe == "Proposed" else None,
                    "n_rows_kernel": last_stage[pipe]["n_rows_kernel"],
                })
            print(f"  scale={scale} trace={trace['query_id']} done", flush=True)

    # ================= Experiment 1: scale sweep =================
    print("=== Experiment 1: scale sweep ===", flush=True)
    for scale, parquet_path in scales.items():
        sweep_scale(scale, parquet_path, resolvers[scale], singles,
                    PIPELINES_EXP1, N_TRIALS, "exp1_scale")
        print(f"  scale={scale} done", flush=True)

    # 3M spot-check: 3 traces (one per kernel) x {B1, Proposed} x 3 trials.
    print("=== Experiment 1b: 3M spot-check ===", flush=True)
    spot_resolver = SchemaResolver(spot_path)
    spot_traces = [by_id[tid] for tid in SPOTCHECK_TRACES]
    sweep_scale("3M", spot_path, spot_resolver, spot_traces,
                ["B1", "Proposed"], 3, "exp1b_spotcheck")

    # ================= Experiment 2: multi-turn IMG ablation =================
    print("=== Experiment 2: multi-turn IMG ablation (1M) ===", flush=True)
    parquet_1m = scales["1M"]
    resolver = resolvers["1M"]
    # Per-pipeline accumulators that persist ACROSS the 5 sessions, modelling
    # 5 concurrent sessions: B2's opaque frame cache and the IMG's view DAG.
    b2_accum = caveagent_baseline.CaveAgentBaseline()
    img_accum = IntermediateMaterializationGraph()
    cum_mem: dict[str, float] = {}
    for sid, turns in sorted(sessions.items()):
        for pipe in PIPELINES_EXP2:
            img = img_accum if pipe == "Proposed_IMG" else None
            b2 = b2_accum if pipe == "B2" else caveagent_baseline.CaveAgentBaseline()
            for trace in turns:
                cum = cumulative_clauses(trace, by_id)
                intent_r, clauses = resolve_trace(
                    {**trace, "filter_predicates": cum}, resolver
                )
                trial_stages: list[dict] = []
                trial_tools: list[float] = []
                trial_rss: list[float] = []
                for trial in range(N_TRIALS):
                    gc.collect()
                    if pipe.startswith("Proposed"):
                        from engine.schemas import FilterClause  # noqa: E402

                        if pipe == "Proposed_IMG":
                            # IMG turn intent carries ONLY its own predicates;
                            # the parent view encodes ancestors.
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
                        ex = SieveExecutor(resolver, img=img, canonical_sort=False)
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
                            lambda: run_baseline(pipe, intent_r, clauses, parquet_1m,
                                                 b2, sort_rows=False)
                        )
                    st = stage_times(pipe if not pipe.startswith("Proposed") else "Proposed",
                                     out["telemetry"])
                    trial_stages.append(st)
                    trial_tools.append(st["t_tool_ms"])
                    trial_rss.append(peak)
                last = trial_stages[-1]
                rows.append({
                    "experiment": "exp2_multiturn",
                    "scale": "1M",
                    "query_id": trace["query_id"],
                    "kernel": trace["downstream_task"],
                    "pipeline": pipe,
                    "t_resolve_mean_ms": mean_stage(trial_stages, "t_resolve_ms"),
                    "t_data_mean_ms": mean_stage(trial_stages, "t_data_ms"),
                    "t_handoff_1d_mean_ms": mean_stage(trial_stages, "t_handoff_1d_ms"),
                    "t_handoff_2d_mean_ms": mean_stage(trial_stages, "t_handoff_2d_ms"),
                    "t_handoff_mean_ms": mean_stage(trial_stages, "t_handoff_ms"),
                    "t_kernel_mean_ms": mean_stage(trial_stages, "t_kernel_ms"),
                    "t_tool_mean_ms": float(np.mean(trial_tools)),
                    "t_tool_p95_ms": float(np.percentile(trial_tools, 95)),
                    "atti_mean": float(np.mean([atti(x) for x in trial_tools])),
                    "sigma_ship_mean": float(np.mean([
                        sigma_ship(s["bytes_materialized"], s["bytes_X"])
                        for s in trial_stages])),
                    "peak_rss_mb_mean": float(np.mean(trial_rss)),
                    "img_view_used": out["telemetry"].get("img_view_used"),
                    "n_rows_kernel": last["n_rows_kernel"],
                })
        print(f"  {sid} done", flush=True)
    # Session cumulative memory across the 5 sessions (MB).
    b2_cum_mb = sum(
        float(df.memory_usage(deep=True).sum())
        for df in b2_accum._sessions.values()
    ) / 1e6
    img_cum_mb = sum(
        img_accum.session_memory_bytes(sid) for sid in sessions
    ) / 1e6
    cum_mem = {"B2_caveagent_frames": b2_cum_mb, "Proposed_IMG_views": img_cum_mb}
    print(f"  cumulative session memory: B2={b2_cum_mb:.0f} MB, IMG={img_cum_mb:.0f} MB",
          flush=True)
    for pipe, mb in cum_mem.items():
        rows.append({
            "experiment": "exp2_session_memory",
            "scale": "1M",
            "query_id": "cumulative_5_sessions",
            "kernel": "-",
            "pipeline": pipe,
            "t_resolve_mean_ms": 0.0, "t_data_mean_ms": 0.0,
            "t_handoff_1d_mean_ms": 0.0, "t_handoff_2d_mean_ms": 0.0,
            "t_handoff_mean_ms": 0.0, "t_kernel_mean_ms": 0.0,
            "t_tool_mean_ms": 0.0, "t_tool_p95_ms": 0.0,
            "atti_mean": 0.0, "sigma_ship_mean": 0.0,
            "peak_rss_mb_mean": 0.0,
            "cumulative_session_memory_mb": float(mb),
            "n_rows_kernel": None,
        })

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
        "t_resolve_mean_ms": 0.0, "t_data_mean_ms": 0.0,
        "t_handoff_1d_mean_ms": 0.0, "t_handoff_2d_mean_ms": 0.0,
        "t_handoff_mean_ms": 0.0, "t_kernel_mean_ms": 0.0,
        "t_tool_mean_ms": 0.0, "t_tool_p95_ms": 0.0,
        "atti_mean": 0.0, "sigma_ship_mean": 0.0,
        "peak_rss_mb_mean": 0.0,
        "healed": healed,
        "n_dirty": len(dirty),
        "n_rows_kernel": None,
    })
    vocab_curve = resolver.benchmark_vocabulary_scaling()
    for k, v in vocab_curve.items():
        rows.append({
            "experiment": "exp3_vocab",
            "scale": "-",
            "query_id": f"vocab_{k}",
            "kernel": "-",
            "pipeline": "resolver",
            "t_resolve_mean_ms": v, "t_data_mean_ms": 0.0,
            "t_handoff_1d_mean_ms": 0.0, "t_handoff_2d_mean_ms": 0.0,
            "t_handoff_mean_ms": 0.0, "t_kernel_mean_ms": 0.0,
            "t_tool_mean_ms": v, "t_tool_p95_ms": v,
            "atti_mean": 0.0, "sigma_ship_mean": 0.0,
            "peak_rss_mb_mean": 0.0,
            "n_rows_kernel": None,
        })
    print(f"  healed {healed}/{len(dirty)} dirty traces; vocab curve done", flush=True)

    # ================= Parity verification (LOCKED, not timed) =================
    print("=== Parity verification, canonical_sort=True (B1 reference) ===", flush=True)
    max_delta = 0.0
    parity_rows: list[dict] = []
    for scale, parquet_path in scales.items():
        resolver = resolvers[scale]
        for trace in singles:
            intent_r, clauses = resolve_trace(trace, resolver)
            ref = pandas_baseline.run(intent_r, clauses, parquet_path,
                                      sort_rows=True)["result"].parity_vector
            for name, fn in (
                ("B3", lambda: polars_baseline.run(intent_r, clauses, parquet_path,
                                                   sort_rows=True)),
                ("B4", lambda: staged_sql_baseline.run(intent_r, clauses, parquet_path,
                                                       sort_rows=True)),
            ):
                vec = fn()["result"].parity_vector
                d = parity_delta(ref, vec)
                max_delta = max(max_delta, d)
                parity_rows.append({"scale": scale, "query_id": trace["query_id"],
                                    "pipeline": name, "parity_delta": d})
            ex = SieveExecutor(resolver, canonical_sort=True)
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
                   "max_parity_delta": max_delta,
                   "session_cumulative_memory_mb": cum_mem}, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "summary_tables.md"), "w") as f:
        f.write("# Benchmark Summary Tables (Phase 7 — decomposed stage timings)\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n\n")
        f.write("Latency mode: production (`canonical_sort=False` / `sort_rows=False`).\n")
        f.write("Parity verified separately in locked mode (eps 1e-5).\n\n")
        stage_cols = ["t_resolve_mean_ms", "t_data_mean_ms", "t_handoff_1d_mean_ms",
                      "t_handoff_2d_mean_ms", "t_handoff_mean_ms", "t_kernel_mean_ms",
                      "t_tool_mean_ms"]
        f.write("## Experiment 1 — Scale sweep, decomposed stage timings (mean ms)\n\n")
        piv = df[df.experiment == "exp1_scale"].pivot_table(
            index=["scale", "kernel", "pipeline"], values=stage_cols,
            aggfunc="mean").round(3)
        f.write(piv.to_markdown() + "\n\n")
        f.write("## Experiment 1 — Speedups vs B1 (mean)\n\n")
        sp = df[(df.experiment == "exp1_scale") & (df.pipeline == "Proposed")][
            ["scale", "kernel", "data_plane_speedup_vs_b1",
             "end_to_end_speedup_vs_b1"]].round(2)
        f.write(sp.to_markdown(index=False) + "\n\n")
        f.write("## Experiment 1 — ATTI (mean)\n\n")
        piv = df[df.experiment == "exp1_scale"].pivot_table(
            index=["scale", "pipeline"], values="atti_mean", aggfunc="mean").round(4)
        f.write(piv.to_markdown() + "\n\n")
        f.write("## Experiment 1 — sigma_ship (mean)\n\n")
        piv = df[df.experiment == "exp1_scale"].pivot_table(
            index=["scale", "pipeline"], values="sigma_ship_mean", aggfunc="mean").round(2)
        f.write(piv.to_markdown() + "\n\n")
        f.write("## Experiment 1 — Peak RSS MB (mean)\n\n")
        piv = df[df.experiment == "exp1_scale"].pivot_table(
            index=["scale", "pipeline"], values="peak_rss_mb_mean", aggfunc="mean").round(1)
        f.write(piv.to_markdown() + "\n\n")
        f.write("## Experiment 1b — 3M spot-check, decomposed (mean ms)\n\n")
        piv = df[df.experiment == "exp1b_spotcheck"].pivot_table(
            index=["query_id", "pipeline"], values=stage_cols,
            aggfunc="mean").round(3)
        f.write(piv.to_markdown() + "\n\n")
        f.write("## Experiment 2 — Multi-turn IMG ablation, 1M, decomposed (mean ms)\n\n")
        piv = df[df.experiment == "exp2_multiturn"].pivot_table(
            index=["query_id", "pipeline"], values=stage_cols,
            aggfunc="mean").round(3)
        f.write(piv.to_markdown() + "\n\n")
        f.write("## Experiment 2 — Session cumulative memory, 5 sessions (MB)\n\n")
        for pipe, mb in cum_mem.items():
            f.write(f"- {pipe}: {mb:.1f} MB\n")
        f.write("\n## Experiment 3 — Resolver\n\n")
        f.write(f"Healed {healed}/{len(dirty)} dirty traces.\n\n")
        f.write("Vocabulary scaling (ms per resolve):\n\n")
        for k in sorted(vocab_curve):
            f.write(f"- 1e{len(str(k))-1} ({k}): {vocab_curve[k]:.3f} ms\n")
        f.write(f"\n## Parity (locked mode)\n\nMax |Δ| vs B1 reference: {max_delta:.2e} (eps {PARITY_EPS})\n")

    print(f"[run_benchmark] wrote {len(rows)} result rows; max parity delta {max_delta:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
