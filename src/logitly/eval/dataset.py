from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from ..constants import DEFAULT_DATASET_SIZE, DEFAULT_SEED, MMLU_PRO_REPO, MMLU_PRO_REVISION
from ..storage import project_root


@dataclass(frozen=True)
class BenchmarkItem:
    question_id: int
    category: str
    question: str
    options: tuple[str, ...]
    answer_index: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["options"] = list(self.options)
        return data


def load_benchmark_items(
    *,
    size: int = DEFAULT_DATASET_SIZE,
    seed: int = DEFAULT_SEED,
    cache_dir: Path | None = None,
) -> list[BenchmarkItem]:
    if size < 1:
        raise ValueError("dataset size must be positive")
    from datasets import load_dataset

    dataset = load_dataset(
        MMLU_PRO_REPO,
        split="test",
        revision=MMLU_PRO_REVISION,
        cache_dir=str(cache_dir or project_root() / ".cache" / "huggingface" / "datasets"),
    )
    items = [normalize_row(row) for row in dataset]
    eligible = [item for item in items if item is not None]
    return stratified_sample(eligible, size=size, seed=seed)


def normalize_row(row: dict[str, Any]) -> BenchmarkItem | None:
    options = tuple(str(option).strip() for option in row["options"])
    answer_index = int(row["answer_index"])
    if (
        not 3 <= len(options) <= 20
        or any(not option for option in options)
        or not 0 <= answer_index < len(options)
    ):
        return None
    question = str(row["question"]).strip()
    category = str(row["category"]).strip()
    if not question or not category:
        return None
    return BenchmarkItem(
        question_id=int(row["question_id"]),
        category=category,
        question=question,
        options=options,
        answer_index=answer_index,
    )


def stratified_sample(
    items: Iterable[BenchmarkItem],
    *,
    size: int,
    seed: int,
) -> list[BenchmarkItem]:
    values = list(items)
    if size > len(values):
        raise ValueError(f"requested {size} rows from only {len(values)} eligible rows")
    grouped: dict[str, list[BenchmarkItem]] = defaultdict(list)
    for item in values:
        grouped[item.category].append(item)

    counts = Counter(item.category for item in values)
    exact = {category: size * count / len(values) for category, count in counts.items()}
    allocation = {category: math.floor(value) for category, value in exact.items()}
    remainder = size - sum(allocation.values())
    priority = sorted(
        grouped,
        key=lambda category: (-(exact[category] - allocation[category]), category),
    )
    for category in priority[:remainder]:
        allocation[category] += 1

    selected: list[BenchmarkItem] = []
    for category in sorted(grouped):
        category_seed = _stable_seed(seed, category)
        rng = random.Random(category_seed)
        population = sorted(grouped[category], key=lambda item: item.question_id)
        selected.extend(rng.sample(population, allocation[category]))
    return sorted(selected, key=lambda item: item.question_id)


def category_counts(items: Iterable[BenchmarkItem]) -> dict[str, int]:
    return dict(sorted(Counter(item.category for item in items).items()))


def _stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")
