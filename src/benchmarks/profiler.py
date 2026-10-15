"""Phase 5 — fine-grained telemetry profiler.

Wall-clock via time.perf_counter_ns(); memory via a background thread polling
psutil RSS every 2 ms so transient allocations are captured.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable

import psutil

T_TOKEN_MS = 350.0  # fixed empirical token-generation baseline (trace-driven replay)


class RSSSampler:
    """High-frequency RSS sampler running in a background thread."""

    def __init__(self, interval_s: float = 0.002):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._peak = 0
        self._proc = psutil.Process()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss
                if rss > self._peak:
                    self._peak = rss
            except psutil.NoSuchProcess:
                break
            time.sleep(self.interval_s)

    def __enter__(self) -> "RSSSampler":
        self._peak = self._proc.memory_info().rss
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    @property
    def peak_mb(self) -> float:
        return self._peak / (1024 * 1024)


def profile_call(fn: Callable[[], Any]) -> tuple[Any, float]:
    """Run fn() under RSS sampling. Returns (fn_result, peak_rss_mb)."""
    with RSSSampler() as sampler:
        out = fn()
    return out, sampler.peak_mb


def atti(t_tool_ms: float) -> float:
    """Agent Tool-Time Index = T_tool / (T_token + T_tool)."""
    return t_tool_ms / (T_TOKEN_MS + t_tool_ms)


def sigma_ship(bytes_materialized: float, bytes_x: float) -> float:
    """Data-shipping overhead factor."""
    return bytes_materialized / max(bytes_x, 1.0)
