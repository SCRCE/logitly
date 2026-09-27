from __future__ import annotations

import hashlib
import random


def option_orders(
    option_count: int,
    *,
    question_id: int,
    seed: int,
    permutation_count: int,
) -> list[tuple[int, ...]]:
    if option_count < 3:
        raise ValueError("at least three options are required for five meaningful permutations")
    if permutation_count < 0:
        raise ValueError("permutation_count cannot be negative")
    identity = tuple(range(option_count))
    maximum = _factorial(option_count) - 1
    if permutation_count > maximum:
        raise ValueError(f"requested {permutation_count} non-identity orders, maximum is {maximum}")
    digest = hashlib.sha256(f"{seed}:{question_id}:permutations".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    orders = [identity]
    seen = {identity}
    while len(orders) <= permutation_count:
        candidate = list(identity)
        rng.shuffle(candidate)
        order = tuple(candidate)
        if order not in seen:
            seen.add(order)
            orders.append(order)
    return orders


def remap_probabilities(
    presented_order: tuple[int, ...],
    presented_probabilities: dict[str, float],
) -> dict[str, float]:
    expected_keys = {str(index) for index in presented_order}
    if set(presented_probabilities) != expected_keys:
        raise ValueError("presented probabilities do not match semantic option ids")
    return {str(index): float(presented_probabilities[str(index)]) for index in range(len(presented_order))}


def _factorial(value: int) -> int:
    result = 1
    for factor in range(2, value + 1):
        result *= factor
    return result
