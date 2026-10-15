"""Phase 2 — Structured execution contract for the analytical sieve.

The LLM is forced OUT of monolithic Python generation into this pushdown
contract: (target columns + filter predicates + ML kernel). The resolver
repairs the contract pre-execution; the sieve executor compiles it to
DuckDB SQL.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class FilterClause(BaseModel):
    """One atomic predicate clause, e.g. borough = 'Manhattan'."""

    column: str = Field(..., description="Column name (may be an unresolved alias pre-resolver)")
    op: Literal["=", "!=", ">", ">=", "<", "<=", "IN"] = Field(...)
    value: object = Field(..., description="Literal value (str/float/int/list)")

    def to_sql(self, resolved_column: str, resolved_value: object) -> str:
        if isinstance(resolved_value, str):
            val = "'" + resolved_value.replace("'", "''") + "'"
        elif isinstance(resolved_value, (list, tuple)):
            inner = ", ".join(
                f"'{v}'".replace("'", "''") if isinstance(v, str) else str(v)
                for v in resolved_value
            )
            val = f"({inner})"
        else:
            val = str(resolved_value)
        if self.op == "IN":
            return f'{resolved_column} IN {val}'
        return f"{resolved_column} {self.op} {val}"


class AnalyticalIntent(BaseModel):
    """Frozen tool contract — the only thing the (simulated) agent emits."""

    query_id: str
    target_columns: list[str]
    filter_predicates: list[FilterClause] = Field(default_factory=list)
    downstream_task: Literal["isolation_forest", "linear_regression", "kmeans_clustering"]
    session_id: Optional[str] = None
    parent_turn_id: Optional[str] = None

    @field_validator("target_columns")
    @classmethod
    def _nonempty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("target_columns must be non-empty; SELECT * is prohibited")
        return v

    def required_columns(self) -> list[str]:
        """All columns the turn needs: projection ∪ filter columns."""
        cols = list(self.target_columns)
        for p in self.filter_predicates:
            if p.column not in cols:
                cols.append(p.column)
        return cols


class KernelResult(BaseModel):
    """Deterministic ML kernel output for parity checking."""

    kernel: str
    summary: dict
    parity_vector: list[float] = Field(
        ..., description="Flat numeric vector used for cross-pipeline parity assertions"
    )
