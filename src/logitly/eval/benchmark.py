from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import platform
import time
from collections import OrderedDict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from ..backends.factory import create_backend
from ..constants import (
    DEFAULT_DATASET_SIZE,
    DEFAULT_PERMUTATIONS,
    DEFAULT_SEED,
    MMLU_PRO_REPO,
    MMLU_PRO_REVISION,
    MODEL_SPECS,
)
from ..engine import DecisionEngine
from ..storage import project_root
from ..types import ChoiceRequest, ChoiceResult
from .dataset import BenchmarkItem, category_counts, load_benchmark_items
from .latency import largest_safe_batch, sweep_latency
from .metrics import compute_model_metrics
from .permutations import option_orders
from .reporting import (
    create_plots,
    write_json,
    write_jsonl,
    write_markdown_report,
    write_latency_csv,
    write_summary_csv,
)


def run_benchmark(
    *,
    models: Sequence[str] = ("glm", "qwen"),
    dataset_size: int = DEFAULT_DATASET_SIZE,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
    output_dir: Path | None = None,
    batch_size: int | None = None,
    warmups: int = 5,
    repetitions: int = 20,
) -> Path:
    items = load_benchmark_items(size=dataset_size, seed=seed)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir or project_root() / "results" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "dataset-manifest.json", _dataset_manifest(items, dataset_size, permutations, seed))

    summaries: list[dict[str, Any]] = []
    latency_by_model: dict[str, Sequence[dict[str, Any]]] = {}
    manifest = _base_manifest(items, dataset_size, permutations, seed)
    for model_key in models:
        if model_key not in MODEL_SPECS:
            raise ValueError(f"unknown model key {model_key!r}")
        backend = create_backend(model_key)
        try:
            engine = DecisionEngine(backend)
            latency = sweep_latency(engine, items, warmups=warmups, repetitions=repetitions)
            safe_batch = batch_size or largest_safe_batch(latency)
            rows = _predict(engine, items, permutations=permutations, seed=seed, batch_size=safe_batch)
            metrics = compute_model_metrics(rows)
            metrics["model"] = backend.model_name
            metrics["model_key"] = model_key
            metrics["evaluation_batch_size"] = safe_batch
            write_jsonl(run_dir / f"predictions-{model_key}.jsonl", rows)
            write_json(run_dir / f"metrics-{model_key}.json", metrics)
            write_json(run_dir / f"latency-{model_key}.json", latency)
            create_plots(run_dir, backend.model_name, metrics, latency)
            summaries.append(metrics)
            latency_by_model[backend.model_name] = latency
            manifest["models"].append(
                {
                    **asdict(MODEL_SPECS[model_key]),
                    "evaluation_batch_size": safe_batch,
                    "peak_memory": backend.peak_memory_bytes(),
                }
            )
        finally:
            backend.close()
            del backend
            _release_cuda()
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "summary.json", summaries)
    write_summary_csv(run_dir / "summary.csv", summaries)
    write_latency_csv(run_dir / "latency.csv", latency_by_model)
    write_markdown_report(run_dir / "report.md", summaries, latency_by_model)
    return run_dir


def _predict(
    engine: DecisionEngine,
    items: Sequence[BenchmarkItem],
    *,
    permutations: int,
    seed: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    work: list[tuple[BenchmarkItem, int, tuple[int, ...], ChoiceRequest]] = []
    for item in items:
        for ordering, order in enumerate(
            option_orders(
                len(item.options),
                question_id=item.question_id,
                seed=seed,
                permutation_count=permutations,
            )
        ):
            choices = OrderedDict((str(index), item.options[index]) for index in order)
            work.append(
                (
                    item,
                    ordering,
                    order,
                    ChoiceRequest(
                        state={"domain": item.category},
                        question=item.question,
                        choices=choices,
                        id=f"{item.question_id}:{ordering}",
                    ),
                )
            )

    rows: list[dict[str, Any]] = []
    for start in range(0, len(work), batch_size):
        chunk = work[start : start + batch_size]
        started = time.perf_counter()
        results = engine.decide_many([entry[3] for entry in chunk], batch_size=len(chunk))
        elapsed = time.perf_counter() - started
        per_decision = elapsed / len(chunk)
        for (item, ordering, order, _), result in zip(chunk, results, strict=True):
            if not isinstance(result, ChoiceResult):
                raise RuntimeError("choice benchmark received a non-choice result")
            rows.append(
                {
                    "model": engine.backend.model_name,
                    "question_id": item.question_id,
                    "category": item.category,
                    "ordering": ordering,
                    "presented_order": list(order),
                    "correct_id": str(item.answer_index),
                    "predicted_id": result.choice,
                    "confidence": result.confidence,
                    "probabilities": {
                        str(index): result.probabilities[str(index)] for index in range(len(item.options))
                    },
                    "batch_size": len(chunk),
                    "wall_seconds_per_decision": per_decision,
                    "model_seconds_per_batch": engine.backend.last_forward_seconds,
                }
            )
    return rows


def _dataset_manifest(
    items: Sequence[BenchmarkItem], size: int, permutations: int, seed: int
) -> dict[str, Any]:
    return {
        "repo_id": MMLU_PRO_REPO,
        "revision": MMLU_PRO_REVISION,
        "split": "test",
        "requested_size": size,
        "selected_size": len(items),
        "seed": seed,
        "permutations_per_item": permutations,
        "category_counts": category_counts(items),
        "question_ids": [item.question_id for item in items],
    }


def _base_manifest(
    items: Sequence[BenchmarkItem], size: int, permutations: int, seed: int
) -> dict[str, Any]:
    prompt_contract = "STATE/QUESTION/CHOICES + one-label answer; model-native assistant prefix"
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": "direct-restricted-next-token-logits",
        "generated_tokens": 0,
        "training_updates": 0,
        "temperature": None,
        "prompt_contract": prompt_contract,
        "prompt_contract_sha256": hashlib.sha256(prompt_contract.encode()).hexdigest(),
        "dataset": _dataset_manifest(items, size, permutations, seed),
        "environment": _environment_manifest(),
        "models": [],
    }


def _environment_manifest() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "accelerate", "compressed-tensors", "datasets", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu: dict[str, Any] = {}
    try:
        import torch

        if torch.cuda.is_available():
            gpu = {
                "name": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "total_memory": torch.cuda.get_device_properties(0).total_memory,
                "cuda": torch.version.cuda,
            }
    except ImportError:
        pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "gpu": gpu,
    }


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        return
