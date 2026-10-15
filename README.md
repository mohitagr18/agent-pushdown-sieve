# agent-pushdown-sieve

Empirically proving and mitigating **Amdahl's Tool Bottleneck** in LLM
data-analytics / code-interpreter agents.

**Thesis.** On non-SQL statistical/ML workloads (Scikit-Learn, SciPy) over
100k–1M+ row tabular datasets, token-generation latency $T_{token}$ is
negligible (<5%) versus data ingestion + serialization latency $T_{tool}$
(>95%) in agents like CodeAct / LangChain Pandas Agent.

**Mitigation — the 3-part architecture.**
1. **Constrained Schema Resolver** — forces the agent out of monolithic
   Python generation into a structured pushdown contract
   (target columns + filter predicates + ML kernel); deterministic
   fuzzy-matching heals column aliases and categorical typos pre-execution.
2. **Two-Phase Pushdown-to-Kernel via Zero-Copy Arrow** — embedded
   in-process DuckDB executes predicate pushdown + column projection on
   Parquet; zero-copy Apache Arrow buffers feed in-process Python ML
   kernels (no disk staging, no IPC serialization).
3. **Intermediate Materialization Graph (IMG)** — lineage-aware DAG of
   `(source_table, projected_columns, predicate_clauses)` caching
   intermediate columnar views across multi-turn conversations, so
   follow-up queries plan against parent views instead of rescanning
   base tables.

All benchmarks are **trace-driven replays** (frozen JSON intents, zero live
LLM API calls, fixed $T_{token}=350$ ms) — see `PROJECT_EXECUTION_LOG.md`
for the living engineering record.

## Quick start

```bash
uv sync
uv run python src/data/fetch_nyctlc.py     # NYC TLC 1M-row dataset
uv run python src/data/fetch_bird.py       # BIRD-SQL validation (Track 1)
uv run python src/data/generate_traces.py  # frozen intent traces
uv run python src/benchmarks/run_benchmark.py
uv run pytest -v
uv run python src/server/mcp_server.py     # FastMCP server
```

## Layout

- `src/engine/` — `schemas.py` (Pydantic contract), `resolver.py`,
  `sieve_executor.py` (two-phase DuckDB→Arrow→sklearn + pointer audit),
  `img_graph.py`
- `src/baselines/` — B1 monolithic Pandas, B2 CaveAgent-style stateful
  cache, B3 streaming Polars, B4 disk-staged SQL pushdown
- `src/benchmarks/` — `profiler.py` (2 ms RSS sampling), `run_benchmark.py`
- `src/server/` — `mcp_server.py` (FastMCP: `execute_analytical_sieve`,
  `inspect_parquet_catalog`)
- `src/data/` — acquisition + trace generation scripts
- `data/` — `trips_1m.parquet` (1M rows), `trips_100k/10k.parquet`,
  `benchmark_intents.json` (40 frozen traces), `sql_validation_100.json`
- `results/` — `benchmark_results.{parquet,json}`, `summary_tables.md`
- `tests/` — resolver, zero-copy + parity, IMG lineage, MCP server
