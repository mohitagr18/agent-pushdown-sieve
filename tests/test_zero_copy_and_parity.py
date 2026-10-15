"""Zero-copy pointer audit + cross-pipeline numerical parity (eps <= 1e-5)."""
import numpy as np
import pyarrow as pa
import pytest

from baselines import pandas_baseline, polars_baseline
from baselines._common import resolve_trace
from engine.resolver import SchemaResolver
from engine.schemas import AnalyticalIntent
from engine.sieve_executor import SieveExecutor

EPS = 1e-5
TRACE_IDS = ["single_01", "single_06", "single_11"]  # one per kernel


def _traces():
    import json
    import os

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    with open(os.path.join(root, "data", "benchmark_intents.json")) as f:
        return {t["query_id"]: t for t in json.load(f)["traces"]}


def test_zero_copy_address_identity(resolver: SchemaResolver, parquet_1m: str):
    ex = SieveExecutor(resolver)
    intent = AnalyticalIntent(
        query_id="zc_test",
        target_columns=["trip_distance", "fare_amount", "total_amount"],
        filter_predicates=[],
        downstream_task="isolation_forest",
    )
    out = ex.execute(intent)
    tel = out["telemetry"]
    assert tel["zero_copy_1d_verified"] is True
    for a in tel["zero_copy_audit"]:
        assert a["address_identity"] is True, a


def test_zero_copy_only_for_contiguous_nonnull():
    # Documents the operational precision: chunked/non-contiguous arrays
    # cannot be extracted zero-copy; combine_chunks() is the remedy.
    arr = pa.chunked_array([pa.array([1.0, 2.0]), pa.array([3.0])])
    with pytest.raises(Exception):
        arr.to_numpy(zero_copy_only=True)
    flat = arr.combine_chunks().to_numpy(zero_copy_only=True)
    assert np.array_equal(flat, [1.0, 2.0, 3.0])


def test_parity_pandas_polars_proposed(resolver: SchemaResolver, parquet_1m: str):
    traces = _traces()
    for tid in TRACE_IDS:
        trace = traces[tid]
        intent_r, clauses = resolve_trace(trace, resolver)
        ref = pandas_baseline.run(intent_r, clauses, parquet_1m)["result"].parity_vector
        b3 = polars_baseline.run(intent_r, clauses, parquet_1m)["result"].parity_vector
        ex = SieveExecutor(resolver)
        pr = ex.execute(
            AnalyticalIntent(
                query_id=intent_r.query_id,
                target_columns=intent_r.target_columns,
                filter_predicates=intent_r.filter_predicates,
                downstream_task=intent_r.downstream_task,
            )
        )["result"].parity_vector
        assert len(ref) == len(b3) == len(pr)
        for name, vec in (("polars", b3), ("proposed", pr)):
            delta = max(abs(a - b) for a, b in zip(ref, vec))
            assert delta <= EPS, f"{tid}/{name}: parity delta {delta:.2e} > {EPS}"
