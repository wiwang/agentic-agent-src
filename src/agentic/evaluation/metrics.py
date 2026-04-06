"""Evaluation metrics — latency, token usage, accuracy tracking."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Iterator

from pydantic import BaseModel


class RunMetrics(BaseModel):
    """Metrics for a single agent run."""

    run_id: str
    agent_id: str
    start_time: float
    end_time: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    n_llm_calls: int = 0
    n_tool_calls: int = 0
    n_iterations: int = 0
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = {}

    @property
    def latency_seconds(self) -> float:
        end = self.end_time or time.monotonic()
        return end - self.start_time

    @property
    def tokens_per_second(self) -> float:
        lat = self.latency_seconds
        return self.total_tokens / lat if lat > 0 else 0.0


class MetricsCollector:
    """Collects and aggregates metrics across multiple agent runs.

    Subscribes to EventBus events to auto-track LLM and tool calls.
    """

    def __init__(self) -> None:
        self._runs: dict[str, RunMetrics] = {}
        self._completed: list[RunMetrics] = []

    def start_run(self, run_id: str, agent_id: str) -> RunMetrics:
        m = RunMetrics(
            run_id=run_id,
            agent_id=agent_id,
            start_time=time.monotonic(),
        )
        self._runs[run_id] = m
        return m

    def end_run(self, run_id: str, success: bool = True, error: str | None = None) -> None:
        m = self._runs.pop(run_id, None)
        if m:
            m.end_time = time.monotonic()
            m.success = success
            m.error = error
            self._completed.append(m)

    def record_tokens(self, run_id: str, prompt: int, completion: int) -> None:
        m = self._runs.get(run_id)
        if m:
            m.prompt_tokens += prompt
            m.completion_tokens += completion
            m.total_tokens += prompt + completion
            m.n_llm_calls += 1

    def record_tool_call(self, run_id: str) -> None:
        m = self._runs.get(run_id)
        if m:
            m.n_tool_calls += 1

    def summary(self) -> dict[str, Any]:
        if not self._completed:
            return {"runs": 0}
        lats = [m.latency_seconds for m in self._completed]
        tokens = [m.total_tokens for m in self._completed]
        success_rate = sum(1 for m in self._completed if m.success) / len(self._completed)
        return {
            "runs": len(self._completed),
            "success_rate": success_rate,
            "avg_latency_s": sum(lats) / len(lats),
            "p50_latency_s": sorted(lats)[len(lats) // 2],
            "p95_latency_s": sorted(lats)[int(len(lats) * 0.95)],
            "avg_tokens": sum(tokens) / len(tokens),
            "total_tokens": sum(tokens),
        }

    def get_run(self, run_id: str) -> RunMetrics | None:
        return self._completed_by_id().get(run_id)

    def _completed_by_id(self) -> dict[str, RunMetrics]:
        return {m.run_id: m for m in self._completed}

    @contextmanager
    def measure(self, run_id: str, agent_id: str = "") -> Iterator[RunMetrics]:
        m = self.start_run(run_id, agent_id)
        try:
            yield m
            self.end_run(run_id, success=True)
        except Exception as exc:
            self.end_run(run_id, success=False, error=str(exc))
            raise


# Global collector
_default_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    return _default_collector
