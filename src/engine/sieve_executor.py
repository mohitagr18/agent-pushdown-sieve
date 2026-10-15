"""Phase 2 — Two-Phase pushdown-to-kernel sieve executor.

Phase 1   : in-process DuckDB compiles STRICT `SELECT <resolved_cols> ...`
            (SELECT * prohibited) with predicate pushdown + column projection
            on the Parquet file, fetched as an Apache Arrow table.
Phase 1.5 : zero-copy handoff — 1D Arrow buffers -> NumPy with
            zero_copy_only=True; raw C-buffer addresses are audited
            (col.buffers()[1].address vs numpy __array_interface__ data ptr).
            2D feature-matrix consolidation is instrumented SEPARATELY because
            sklearn needs a C-contiguous 2D X (np.column_stack copies).
Phase 2   : in-process sklearn kernels (random_state=42), deterministic.
"""
from __future__ import annotations

import time
from typing import Any, Optional

import duckdb
import numpy as np
import pyarrow as pa
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

from .resolver import SchemaResolver
from .schemas import AnalyticalIntent, KernelResult


def build_feature_matrix(
    table: pa.Table, feature_columns: list[str], sort_rows: bool = True
) -> np.ndarray:
    """SHARED null-handling + matrix construction used by engine AND baselines.

    Guarantees bit-identical X across pipelines (required for parity).
    Drops rows containing ANY null across the feature columns, deterministically.
    sort_rows=True applies a lexicographic canonical row order, because
    DuckDB/Polars parallel scans do not guarantee output row order and
    KMeans/IsolationForest are order-sensitive (breaks parity otherwise).
    """
    cols_1d: list[np.ndarray] = []
    for c in feature_columns:
        arr = table.column(c).combine_chunks()
        np1d = arr.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        cols_1d.append(np1d)
    mask = np.ones(len(cols_1d[0]), dtype=bool)
    for v in cols_1d:
        mask &= ~np.isnan(v)
    stacked = np.column_stack([v[mask] for v in cols_1d])
    if sort_rows and stacked.shape[0] > 0:
        order = np.lexsort(tuple(stacked[:, i] for i in range(stacked.shape[1] - 1, -1, -1)))
        stacked = stacked[order]
    return np.ascontiguousarray(stacked, dtype=np.float64)


def run_kernel(task: str, X: np.ndarray) -> KernelResult:
    if X.shape[0] == 0:
        raise ValueError("empty feature matrix after null-drop; nothing to fit")
    if task == "isolation_forest":
        est = IsolationForest(
            n_estimators=50, contamination=0.02, random_state=42, n_jobs=1
        )
        pred = est.fit_predict(X)
        scores = est.decision_function(X)
        n_anom = int((pred == -1).sum())
        summary = {
            "n_rows": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "anomaly_count": n_anom,
            "anomaly_rate": n_anom / X.shape[0],
        }
        parity = [
            float(n_anom),
            float(scores.mean()),
            float(scores.std()),
            float(scores.min()),
            float(scores.max()),
        ]
    elif task == "linear_regression":
        est = LinearRegression()
        est.fit(X[:, :1], X[:, 1])
        r2 = float(est.score(X[:, :1], X[:, 1]))
        summary = {
            "n_rows": int(X.shape[0]),
            "coef": [float(c) for c in est.coef_],
            "intercept": float(est.intercept_),
            "r2": r2,
        }
        parity = [float(est.coef_[0]), float(est.intercept_), r2]
    elif task == "kmeans_clustering":
        est = KMeans(n_clusters=4, random_state=42, n_init=5)
        est.fit(X)
        centers = est.cluster_centers_
        summary = {
            "n_rows": int(X.shape[0]),
            "inertia": float(est.inertia_),
            "cluster_centers": centers.tolist(),
        }
        parity = [float(est.inertia_)] + [float(v) for v in centers.ravel()]
    else:  # pragma: no cover - contract enforces the literal
        raise ValueError(f"unknown kernel {task}")
    return KernelResult(kernel=task, summary=summary, parity_vector=parity)


class SieveExecutor:
    """Two-phase executor. Optionally consults the IMG for child turns.

    canonical_sort: when True, apply the np.lexsort canonical row order in
        the 2D consolidation step (identical to the shared
        build_feature_matrix(sort_rows=True) used by baselines) so
        KMeans/IsolationForest parity assertions hold bit-identically.
        Default False = production latency path: no test-harness sort
        penalizes T_handoff.
    """

    def __init__(
        self,
        resolver: SchemaResolver,
        img: Optional[Any] = None,
        parquet_path: Optional[str] = None,
        canonical_sort: bool = False,
    ):
        self.resolver = resolver
        self.img = img
        self.parquet_path = parquet_path or resolver.parquet_path
        self.canonical_sort = canonical_sort
        self.con = duckdb.connect()

    # ------------------------------------------------------------------ #
    def _source_clause(
        self, intent: AnalyticalIntent, resolved_columns: list[str]
    ) -> tuple[str, Optional[str]]:
        """Return (FROM clause, view_name_or_None). Prefers IMG parent view.

        Coverage is checked against RESOLVED columns (canonical names).
        """
        if intent.parent_turn_id and self.img is not None:
            node = self.img.get(intent.parent_turn_id)
            if node is not None:
                need = set(resolved_columns) | {
                    self.resolver.bind_column(p.column)[0]
                    for p in intent.filter_predicates
                }
                if node.covers(sorted(need)):
                    view = f"img_view_{intent.parent_turn_id.replace('-', '_')}"
                    self.con.register(view, node.arrow_table)
                    return view, view
        return f"read_parquet('{self.parquet_path}')", None

    # ------------------------------------------------------------------ #
    def execute(self, intent: AnalyticalIntent) -> dict:
        tel: dict[str, Any] = {"query_id": intent.query_id}

        # ---- Phase 0: resolve ------------------------------------------------
        t0 = time.perf_counter_ns()
        resolved = self.resolver.resolve(
            intent.target_columns,
            [p.model_dump() for p in intent.filter_predicates],
        )
        tel["t_resolve_ms"] = (time.perf_counter_ns() - t0) / 1e6
        tel["repairs"] = resolved.repairs

        # ---- Phase 1: DuckDB pushdown -> Arrow -------------------------------
        source, view_name = self._source_clause(intent, resolved.columns)
        sql = f"SELECT {', '.join(resolved.columns)} FROM {source}"
        if resolved.where_sql and resolved.where_sql != "TRUE":
            sql += f" WHERE {resolved.where_sql}"
        tel["compiled_sql"] = sql
        tel["img_view_used"] = view_name

        t0 = time.perf_counter_ns()
        arrow_table: pa.Table = self.con.execute(sql).to_arrow_table()
        tel["t_pushdown_ms"] = (time.perf_counter_ns() - t0) / 1e6
        tel["n_rows_pushed"] = arrow_table.num_rows
        tel["bytes_arrow"] = int(arrow_table.nbytes)

        # ---- Phase 1.5: zero-copy 1D extraction (pointer-audited) ------------
        feature_cols = self._kernel_features(intent.downstream_task, resolved.columns)
        audit: list[dict[str, Any]] = []
        t0 = time.perf_counter_ns()
        cols_1d: list[np.ndarray] = []
        zero_copy_all = True
        for c in feature_cols:
            chunked = arrow_table.column(c)
            arr1d = chunked.combine_chunks()
            buf_addr = arr1d.buffers()[1].address if len(arr1d.buffers()) > 1 else None
            np1d = arr1d.to_numpy(zero_copy_only=True).astype(np.float64, copy=False)
            # astype(copy=False) is a no-op view when already float64
            np_addr = np1d.__array_interface__["data"][0]
            identical = buf_addr is not None and buf_addr == np_addr
            zero_copy_all &= identical
            audit.append(
                {
                    "column": c,
                    "arrow_buffer_addr": hex(buf_addr) if buf_addr else None,
                    "numpy_data_addr": hex(np_addr),
                    "address_identity": identical,
                    "n_bytes": int(np1d.nbytes),
                }
            )
            cols_1d.append(np1d)
        tel["t_handoff_1d_ms"] = (time.perf_counter_ns() - t0) / 1e6
        tel["zero_copy_audit"] = audit
        tel["zero_copy_1d_verified"] = bool(zero_copy_all)

        # ---- Phase 1.5b: 2D consolidation (explicitly NOT zero-copy) ---------
        t0 = time.perf_counter_ns()
        mask = np.ones(len(cols_1d[0]), dtype=bool)
        for v in cols_1d:
            mask &= ~np.isnan(v)
        X = np.column_stack([v[mask] for v in cols_1d])
        if self.canonical_sort and X.shape[0] > 0:
            # Canonical lexicographic row order — identical to the shared
            # build_feature_matrix(sort_rows=True) used by baselines (parity
            # requirement). Disabled by default: the sort is a test-harness
            # artifact, not part of the production data plane.
            order = np.lexsort(tuple(X[:, i] for i in range(X.shape[1] - 1, -1, -1)))
            X = X[order]
        X = np.ascontiguousarray(X, dtype=np.float64)
        tel["t_handoff_2d_ms"] = (time.perf_counter_ns() - t0) / 1e6
        tel["t_handoff_ms"] = tel["t_handoff_1d_ms"] + tel["t_handoff_2d_ms"]
        tel["n_rows_kernel"] = int(X.shape[0])
        tel["bytes_X"] = int(X.nbytes)
        tel["null_rows_dropped"] = int((~mask).sum())

        # ---- Phase 2: in-process ML kernel -----------------------------------
        t0 = time.perf_counter_ns()
        result = run_kernel(intent.downstream_task, X)
        tel["t_kernel_ms"] = (time.perf_counter_ns() - t0) / 1e6

        tel["t_tool_ms"] = (
            tel["t_resolve_ms"]
            + tel["t_pushdown_ms"]
            + tel["t_handoff_ms"]
            + tel["t_kernel_ms"]
        )
        return {
            "result": result,
            "telemetry": tel,
            "arrow_table": arrow_table,  # for IMG registration by the caller
            "resolved_columns": resolved.columns,
            "resolved_where": resolved.where_sql,
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _kernel_features(task: str, resolved_columns: list[str]) -> list[str]:
        want = {
            "isolation_forest": ["trip_distance", "fare_amount", "total_amount"],
            "linear_regression": ["trip_duration_minutes", "total_amount"],
            "kmeans_clustering": ["trip_distance", "tip_amount", "total_amount"],
        }[task]
        missing = [c for c in want if c not in resolved_columns]
        if missing:
            raise ValueError(f"kernel {task} needs columns {missing} not in projection")
        return want
