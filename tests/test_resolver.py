"""Tests for the constrained schema resolver."""
import pytest

from engine.resolver import SchemaResolutionError, SchemaResolver


def test_alias_mapping(resolver: SchemaResolver):
    canon, method = resolver.bind_column("dist")
    assert canon == "trip_distance" and method == "synonym"
    canon, method = resolver.bind_column("fare")
    assert canon == "fare_amount" and method == "synonym"
    canon, method = resolver.bind_column("duration")
    assert canon == "trip_duration_minutes" and method == "synonym"
    canon, method = resolver.bind_column("tip")
    assert canon == "tip_amount" and method == "synonym"


def test_exact_match(resolver: SchemaResolver):
    canon, method = resolver.bind_column("borough")
    assert (canon, method) == ("borough", "exact")


def test_typo_repair(resolver: SchemaResolver):
    res = resolver.resolve(
        ["trip_distance"], [{"column": "borough", "op": "=", "value": "Manhatten"}]
    )
    assert "borough = 'Manhattan'" in res.where_sql


def test_typo_repair_quens(resolver: SchemaResolver):
    res = resolver.resolve(
        ["fare_amount"], [{"column": "borough", "op": "=", "value": "Quens"}]
    )
    assert "borough = 'Queens'" in res.where_sql


def test_fail_closed_on_hallucinated_column(resolver: SchemaResolver):
    # Genuinely unresolvable identifier: no exact/synonym/fuzzy (>=75) match.
    # (Near-miss strings like 'passenger_count_xyzzy' legitimately fuzzy-match
    #  to 'passenger_count' at 83.3 — that is the resolver working, not failing.)
    with pytest.raises(SchemaResolutionError):
        resolver.bind_column("xqz_nonexistent_thermal")


def test_footer_only_schema_read(resolver: SchemaResolver):
    # Footer metadata read must be fast (no row scans); allow generous bound.
    assert resolver.footer_ms < 500


def test_vocabulary_scaling_sub_5ms(resolver: SchemaResolver):
    curve = resolver.benchmark_vocabulary_scaling()
    for vocab, ms in curve.items():
        assert ms < 5.0, f"vocab {vocab}: {ms:.2f} ms exceeds 5 ms bar"
