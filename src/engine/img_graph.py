"""Phase 3 — Lineage-aware Intermediate Materialization Graph (IMG).

Unlike CaveAgent's opaque Python-DataFrame runtime namespace, every IMG node
carries explicit lineage metadata:

    (source_table, projected_columns, predicate_clauses)

Child turns with `parent_turn_id` are planned against the parent view:
the parent's zero-copy Arrow table is registered back into DuckDB
(`con.register(view_name, arrow_table)`) and the child predicate executes
against the parent view instead of rescanning the 1M-row base Parquet file.

Coverage rule: the child turn may proceed on the parent view iff the parent's
projected_columns superset the child's required columns (projection ∪ filter
columns). Parents auto-union required columns across registered session turns
at registration time when `union_session_columns=True`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pyarrow as pa


@dataclass
class IMGNode:
    turn_id: str
    parent_turn_id: Optional[str]
    session_id: Optional[str]
    source_table: str
    projected_columns: set[str] = field(default_factory=set)
    predicate_clauses: list[str] = field(default_factory=list)
    arrow_table: Optional[pa.Table] = None
    row_count: int = 0
    memory_bytes: int = 0

    @property
    def compiled_predicate(self) -> str:
        return " AND ".join(self.predicate_clauses) if self.predicate_clauses else "TRUE"

    def covers(self, required_columns: list[str]) -> bool:
        return set(required_columns) <= self.projected_columns


class IntermediateMaterializationGraph:
    """Per-session DAG of materialized columnar views."""

    def __init__(self):
        self._nodes: dict[str, IMGNode] = {}

    # ------------------------------------------------------------------ #
    def register(
        self,
        turn_id: str,
        arrow_table: pa.Table,
        source_table: str,
        projected_columns: list[str],
        predicate_clauses: list[str],
        session_id: Optional[str] = None,
        parent_turn_id: Optional[str] = None,
    ) -> IMGNode:
        """Register a materialized turn. Coverage is honest: a child turn may
        plan against this node only if the node's projected columns already
        contain every column the child needs (projection ∪ filter columns).
        No silent projection widening — if a parent lacks a child column, the
        child falls back to the base table (logged by the executor)."""
        node = IMGNode(
            turn_id=turn_id,
            parent_turn_id=parent_turn_id,
            session_id=session_id,
            source_table=source_table,
            projected_columns=set(projected_columns),
            predicate_clauses=list(predicate_clauses),
            arrow_table=arrow_table,
            row_count=arrow_table.num_rows,
            memory_bytes=int(arrow_table.nbytes),
        )
        self._nodes[turn_id] = node
        return node

    def get(self, turn_id: str) -> Optional[IMGNode]:
        return self._nodes.get(turn_id)

    def lineage(self, turn_id: str) -> list[str]:
        """Root-to-leaf chain of turn_ids for auditability."""
        chain: list[str] = []
        node = self._nodes.get(turn_id)
        while node is not None:
            chain.append(node.turn_id)
            node = self._nodes.get(node.parent_turn_id) if node.parent_turn_id else None
        return list(reversed(chain))

    def session_memory_bytes(self, session_id: str) -> int:
        return sum(
            n.memory_bytes for n in self._nodes.values() if n.session_id == session_id
        )

    def __len__(self) -> int:
        return len(self._nodes)
