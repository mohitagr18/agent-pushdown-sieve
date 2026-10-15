"""Shared pytest fixtures."""
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from engine.resolver import SchemaResolver  # noqa: E402


@pytest.fixture(scope="session")
def parquet_1m() -> str:
    p = os.path.join(REPO_ROOT, "data", "trips_1m.parquet")
    assert os.path.exists(p), f"missing dataset {p}; run Phase 1 first"
    return p


@pytest.fixture(scope="session")
def resolver(parquet_1m: str) -> SchemaResolver:
    return SchemaResolver(parquet_1m)
