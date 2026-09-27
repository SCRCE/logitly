from __future__ import annotations

from collections import Counter

from logitly.eval.dataset import BenchmarkItem, stratified_sample
from logitly.eval.permutations import option_orders


def _items() -> list[BenchmarkItem]:
    return [
        BenchmarkItem(index, "a" if index < 60 else "b", f"q{index}", ("a", "b", "c"), 0)
        for index in range(100)
    ]


def test_stratified_sample_is_reproducible_and_proportional() -> None:
    first = stratified_sample(_items(), size=20, seed=123)
    second = stratified_sample(_items(), size=20, seed=123)
    assert first == second
    assert Counter(item.category for item in first) == {"a": 12, "b": 8}


def test_permutations_are_unique_nonidentity_and_reproducible() -> None:
    first = option_orders(5, question_id=7, seed=9, permutation_count=5)
    second = option_orders(5, question_id=7, seed=9, permutation_count=5)
    assert first == second
    assert first[0] == tuple(range(5))
    assert len(first) == len(set(first)) == 6
    assert all(order != first[0] for order in first[1:])
