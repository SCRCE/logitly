from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np


def accuracy(probabilities: Sequence[Sequence[float]], targets: Sequence[int]) -> float:
    _validate_inputs(probabilities, targets)
    correct = [int(np.argmax(row)) == target for row, target in zip(probabilities, targets, strict=True)]
    return float(np.mean(correct))


def negative_log_likelihood(
    probabilities: Sequence[Sequence[float]],
    targets: Sequence[int],
    epsilon: float = 1e-12,
) -> float:
    _validate_inputs(probabilities, targets)
    target_values = np.asarray(
        [row[target] for row, target in zip(probabilities, targets, strict=True)],
        dtype=np.float64,
    )
    return float(-np.mean(np.log(np.clip(target_values, epsilon, 1.0))))


def multiclass_brier(probabilities: Sequence[Sequence[float]], targets: Sequence[int]) -> float:
    _validate_inputs(probabilities, targets)
    scores = []
    for row, target in zip(probabilities, targets, strict=True):
        scores.append(sum((value - float(index == target)) ** 2 for index, value in enumerate(row)))
    return float(np.mean(scores))


def expected_calibration_error(
    probabilities: Sequence[Sequence[float]],
    targets: Sequence[int],
    bins: int = 10,
) -> dict[str, Any]:
    _validate_inputs(probabilities, targets)
    if bins < 1:
        raise ValueError("bins must be positive")
    predictions = np.asarray([int(np.argmax(row)) for row in probabilities])
    confidences = np.asarray([max(row) for row in probabilities], dtype=np.float64)
    correctness = predictions == np.asarray(targets)
    edges = np.linspace(0.0, 1.0, bins + 1)
    details: list[dict[str, float | int]] = []
    ece = 0.0
    for index in range(bins):
        lower, upper = float(edges[index]), float(edges[index + 1])
        mask = (confidences >= lower) & (confidences < upper if index < bins - 1 else confidences <= upper)
        count = int(mask.sum())
        if count:
            mean_confidence = float(confidences[mask].mean())
            bin_accuracy = float(correctness[mask].mean())
            ece += count / len(targets) * abs(mean_confidence - bin_accuracy)
        else:
            mean_confidence = 0.0
            bin_accuracy = 0.0
        details.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "confidence": mean_confidence,
                "accuracy": bin_accuracy,
            }
        )
    return {"ece": float(ece), "bins": details}


def risk_coverage(
    probabilities: Sequence[Sequence[float]],
    targets: Sequence[int],
    thresholds: Iterable[float] = (0.50, 0.70, 0.80, 0.90, 0.95, 0.99),
) -> list[dict[str, float | int]]:
    _validate_inputs(probabilities, targets)
    confidences = np.asarray([max(row) for row in probabilities], dtype=np.float64)
    predictions = np.asarray([int(np.argmax(row)) for row in probabilities])
    correct = predictions == np.asarray(targets)
    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        mask = confidences >= threshold
        selected = int(mask.sum())
        rows.append(
            {
                "threshold": float(threshold),
                "selected": selected,
                "coverage": float(selected / len(targets)),
                "accuracy": float(correct[mask].mean()) if selected else 0.0,
            }
        )
    return rows


def permutation_stability(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["question_id"])].append(row)
    deltas: list[float] = []
    agreements: list[float] = []
    permuted_correct: list[float] = []
    for question_rows in grouped.values():
        canonical = next((row for row in question_rows if int(row["ordering"]) == 0), None)
        if canonical is None:
            raise ValueError("each question requires a canonical row")
        canonical_probabilities = canonical["probabilities"]
        canonical_choice = str(canonical["predicted_id"])
        for row in question_rows:
            if int(row["ordering"]) == 0:
                continue
            for option_id, probability in canonical_probabilities.items():
                deltas.append(abs(float(probability) - float(row["probabilities"][option_id])))
            agreements.append(float(str(row["predicted_id"]) == canonical_choice))
            permuted_correct.append(float(str(row["predicted_id"]) == str(row["correct_id"])))
    return {
        "argmax_agreement": float(np.mean(agreements)) if agreements else 1.0,
        "mean_probability_delta": float(np.mean(deltas)) if deltas else 0.0,
        "p95_probability_delta": float(np.percentile(deltas, 95)) if deltas else 0.0,
        "max_probability_delta": float(np.max(deltas)) if deltas else 0.0,
        "permutation_accuracy": float(np.mean(permuted_correct)) if permuted_correct else 0.0,
    }


def compute_model_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    canonical = [row for row in rows if int(row["ordering"]) == 0]
    if not canonical:
        raise ValueError("no canonical predictions supplied")
    probability_rows = [_ordered_probabilities(row) for row in canonical]
    targets = [int(row["correct_id"]) for row in canonical]
    calibration = expected_calibration_error(probability_rows, targets)
    return {
        "examples": len(canonical),
        "accuracy": accuracy(probability_rows, targets),
        "nll": negative_log_likelihood(probability_rows, targets),
        "brier": multiclass_brier(probability_rows, targets),
        "ece": calibration["ece"],
        "calibration_bins": calibration["bins"],
        "risk_coverage": risk_coverage(probability_rows, targets),
        "permutation_stability": permutation_stability(rows),
    }


def _ordered_probabilities(row: dict[str, Any]) -> list[float]:
    probabilities = row["probabilities"]
    return [float(probabilities[str(index)]) for index in range(len(probabilities))]


def _validate_inputs(probabilities: Sequence[Sequence[float]], targets: Sequence[int]) -> None:
    if not probabilities or len(probabilities) != len(targets):
        raise ValueError("probabilities and targets must be non-empty and have equal length")
    for row, target in zip(probabilities, targets, strict=True):
        width = len(row)
        if width < 2:
            raise ValueError("probability rows must contain at least two choices")
        if not 0 <= target < width:
            raise ValueError("target index is outside the probability row")
        if not math.isclose(sum(row), 1.0, abs_tol=1e-5):
            raise ValueError("probability row does not sum to one")
        if any(value < 0.0 or value > 1.0 for value in row):
            raise ValueError("probabilities must be in [0, 1]")
