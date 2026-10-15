"""Phase 2 — Resilient pre-execution schema resolver.

Binding order per column: exact match -> synonym dictionary -> rapidfuzz
vectorized C++ matching (cutoff >= 75.0). Fail-closed: raises
SchemaResolutionError on unresolvable identifiers.

Categorical value repair: distinct string enums are cached per (table, column)
and string literals inside predicates are repaired with rapidfuzz.
"""
from __future__ import annotations

import time
from typing import Optional

import pyarrow.parquet as pq
from rapidfuzz import process

# Colloquial aliases -> canonical column names (NYC TLC).
SYNONYMS: dict[str, str] = {
    "dist": "trip_distance",
    "distance": "trip_distance",
    "fare": "fare_amount",
    "fare_amt": "fare_amount",
    "duration": "trip_duration_minutes",
    "trip_duration": "trip_duration_minutes",
    "trip_time": "trip_duration_minutes",
    "tip": "tip_amount",
    "tips": "tip_amount",
    "total": "total_amount",
    "total_amt": "total_amount",
    "borough": "borough",
    "pickup": "tpep_pickup_datetime",
    "dropoff": "tpep_dropoff_datetime",
    "passengers": "passenger_count",
    "passenger": "passenger_count",
    "payment": "payment_type",
    "vendor": "VendorID",
    "pu": "PULocationID",
    "do": "DOLocationID",
}

FUZZ_CUTOFF = 75.0


class SchemaResolutionError(Exception):
    """Raised when an identifier cannot be bound to the dataset schema."""


class ResolvedIntent:
    """Resolver output: canonical columns + SQL predicate string."""

    def __init__(self, columns: list[str], where_sql: str, repairs: list[str]):
        self.columns = columns
        self.where_sql = where_sql
        self.repairs = repairs  # human-readable audit trail


class SchemaResolver:
    def __init__(self, parquet_path: str):
        self.parquet_path = parquet_path
        t0 = time.perf_counter_ns()
        # Zero row reads: footer metadata only.
        self._schema = pq.read_schema(parquet_path)
        self.footer_ms = (time.perf_counter_ns() - t0) / 1e6
        self._columns = list(self._schema.names)
        self._lower = {c.lower(): c for c in self._columns}
        self._enum_cache: dict[str, list[str]] = {}

    @property
    def columns(self) -> list[str]:
        return self._columns

    def bind_column(self, name: str) -> tuple[str, str]:
        """Return (canonical_column, method) — method in {exact, synonym, fuzzy}."""
        if name in self._columns:
            return name, "exact"
        low = name.strip().lower()
        if low in self._lower:
            return self._lower[low], "exact"
        if low in SYNONYMS and SYNONYMS[low] in self._columns:
            return SYNONYMS[low], "synonym"
        match = self._fuzzy_best(low, self._columns)
        if match is not None:
            return match[0], "fuzzy"
        raise SchemaResolutionError(
            f"column '{name}' unresolvable against {len(self._columns)} schema columns"
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_block_index(vocab: list[str]) -> dict[tuple[str, int], list[str]]:
        """Blocking index: (first_char, length//4) -> candidates.

        Single-typo errors preserve the first character and (almost always)
        the length bucket, so probing the probe's bucket plus adjacent
        length buckets is a safe prefilter for typo repair.
        """
        index: dict[tuple[str, int], list[str]] = {}
        for v in vocab:
            key = (v[0].lower() if v else "", len(v) // 4)
            index.setdefault(key, []).append(v)
        return index

    @classmethod
    def _fuzzy_best(
        cls, probe: str, vocab: list[str], index: Optional[dict] = None
    ) -> Optional[tuple]:
        if len(vocab) > 1000:
            # High-cardinality path: blocking prefilter -> rapidfuzz on bucket.
            index = index or cls._build_block_index(vocab)
            key = (probe[0].lower() if probe else "", len(probe) // 4)
            cands: list[str] = []
            for db in (-1, 0, 1):
                cands.extend(index.get((key[0], key[1] + db), []))
            pool = cands or vocab
        else:
            pool = vocab
        return process.extractOne(probe, pool, score_cutoff=FUZZ_CUTOFF)

    def _distinct_strings(self, column: str) -> list[str]:
        if column not in self._enum_cache:
            import duckdb

            con = duckdb.connect()
            vals = con.execute(
                f"SELECT DISTINCT {column} FROM read_parquet('{self.parquet_path}') "
                f"WHERE {column} IS NOT NULL"
            ).fetchall()
            con.close()
            self._enum_cache[column] = [str(v[0]) for v in vals]
        return self._enum_cache[column]

    def repair_value(self, column: str, literal: object) -> tuple[object, Optional[str]]:
        """Repair a string literal against the column's distinct enum values."""
        if not isinstance(literal, str):
            return literal, None
        enums = self._distinct_strings(column)
        if literal in enums:
            return literal, None
        match = self._fuzzy_best(literal.strip(), enums)
        if match is not None:
            return match[0], f"'{literal}' -> '{match[0]}'"
        return literal, None  # leave as-is; DuckDB will simply match 0 rows

    def resolve(
        self,
        target_columns: list[str],
        predicates: list[dict],
        extra_union_columns: Optional[list[str]] = None,
    ) -> ResolvedIntent:
        """Bind columns + repair categorical literals -> ResolvedIntent."""
        t0 = time.perf_counter_ns()
        repairs: list[str] = []
        bound: list[str] = []
        seen: set[str] = set()
        for raw in target_columns:
            canon, method = self.bind_column(raw)
            if method != "exact":
                repairs.append(f"column {raw} -> {canon} ({method})")
            if canon not in seen:
                seen.add(canon)
                bound.append(canon)
        for extra in extra_union_columns or []:
            canon, method = self.bind_column(extra)
            if canon not in seen:
                seen.add(canon)
                bound.append(canon)

        clauses: list[str] = []
        for p in predicates:
            canon_col, method = self.bind_column(p["column"])
            if method != "exact":
                repairs.append(f"filter column {p['column']} -> {canon_col} ({method})")
            val, rep = self.repair_value(canon_col, p["value"])
            if rep:
                repairs.append(f"value {rep} in {canon_col}")
            op = p["op"]
            if isinstance(val, str):
                rhs = "'" + val.replace("'", "''") + "'"
            else:
                rhs = str(val)
            clauses.append(f"{canon_col} {op} {rhs}" if op != "IN" else f"{canon_col} IN ({rhs})")

        ms = (time.perf_counter_ns() - t0) / 1e6
        repairs.append(f"resolve_wall_ms={ms:.3f}")
        return ResolvedIntent(columns=bound, where_sql=" AND ".join(clauses) or "TRUE", repairs=repairs)

    def benchmark_vocabulary_scaling(self) -> dict[int, float]:
        """Micro-benchmark: resolution latency across synthetic vocabulary sizes.

        Uses realistic mixed vocabularies (varied first chars / lengths) and
        the blocking prefilter + rapidfuzz vectorized C++ matching.
        """
        import random
        import string

        rng = random.Random(42)
        words = ["".join(rng.choices(string.ascii_lowercase, k=rng.randint(4, 10)))
                 for _ in range(2000)]
        results: dict[int, float] = {}
        for vocab_size in (10, 100, 1_000, 10_000, 100_000):
            vocab = [f"{rng.choice(words)}_{i}" for i in range(vocab_size)]
            probe_src = rng.choice(vocab)
            # single-typo probe: delete one interior char
            j = rng.randint(1, len(probe_src) - 2)
            probe = probe_src[:j] + probe_src[j + 1:]
            index = self._build_block_index(vocab)
            self._fuzzy_best(probe, vocab, index)  # warm
            t0 = time.perf_counter_ns()
            for _ in range(20):
                self._fuzzy_best(probe, vocab, index)
            results[vocab_size] = (time.perf_counter_ns() - t0) / 1e6 / 20
        return results
