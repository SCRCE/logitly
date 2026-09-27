from __future__ import annotations

import gc
import statistics
import time
from collections.abc import Sequence
from typing import Any

from ..engine import DecisionEngine
from ..types import ChoiceRequest
from .dataset import BenchmarkItem


def sweep_latency(
    engine: DecisionEngine,
    items: Sequence[BenchmarkItem],
    *,
    batch_sizes: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128),
    warmups: int = 5,
    repetitions: int = 20,
) -> list[dict[str, Any]]:
    if not items:
        raise ValueError("latency sweep requires benchmark items")
    pool = _median_length_pool(items, max(batch_sizes))
    results: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        requests = [_request(pool[index % len(pool)]) for index in range(batch_size)]
        try:
            for _ in range(warmups):
                engine.decide_many(requests, batch_size=batch_size)
            wall_samples: list[float] = []
            model_samples: list[float] = []
            engine.backend.reset_peak_memory()
            for _ in range(repetitions):
                started = time.perf_counter()
                engine.decide_many(requests, batch_size=batch_size)
                wall_samples.append(time.perf_counter() - started)
                model_samples.append(engine.backend.last_forward_seconds)
            allocated, reserved = engine.backend.peak_memory_bytes()
            total_decisions = batch_size * repetitions
            total_wall = sum(wall_samples)
            results.append(
                {
                    "batch_size": batch_size,
                    "status": "ok",
                    "wall_p50_seconds": statistics.median(wall_samples),
                    "wall_p95_seconds": _percentile(wall_samples, 95),
                    "model_p50_seconds": statistics.median(model_samples),
                    "model_p95_seconds": _percentile(model_samples, 95),
                    "decisions_per_second": total_decisions / total_wall,
                    "model_decisions_per_second": total_decisions / sum(model_samples),
                    "peak_allocated_bytes": allocated,
                    "peak_reserved_bytes": reserved,
                }
            )
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            results.append({"batch_size": batch_size, "status": "oom", "error": str(error)})
            _clear_cuda()
            break
    return results


def largest_safe_batch(rows: Sequence[dict[str, Any]]) -> int:
    safe = [int(row["batch_size"]) for row in rows if row["status"] == "ok"]
    if not safe:
        raise RuntimeError("no latency batch size completed successfully")
    return max(safe)


def _request(item: BenchmarkItem) -> ChoiceRequest:
    return ChoiceRequest(
        state={"domain": item.category},
        question=item.question,
        choices={str(index): option for index, option in enumerate(item.options)},
        id=f"latency-{item.question_id}",
    )


def _median_length_pool(items: Sequence[BenchmarkItem], count: int) -> list[BenchmarkItem]:
    ordered = sorted(items, key=lambda item: len(item.question) + sum(map(len, item.options)))
    width = min(len(ordered), count)
    start = max(0, len(ordered) // 2 - width // 2)
    return ordered[start : start + width]


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _clear_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        return
