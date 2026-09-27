from __future__ import annotations

import math

import pytest

from logitly.eval.metrics import (
    accuracy,
    expected_calibration_error,
    multiclass_brier,
    negative_log_likelihood,
    permutation_stability,
    risk_coverage,
)


def test_metrics_match_hand_calculation() -> None:
    probabilities = [[0.8, 0.2], [0.4, 0.6]]
    targets = [0, 0]
    assert accuracy(probabilities, targets) == 0.5
    assert negative_log_likelihood(probabilities, targets) == pytest.approx(-(math.log(0.8) + math.log(0.4)) / 2)
    assert multiclass_brier(probabilities, targets) == pytest.approx((0.08 + 0.72) / 2)
    calibration = expected_calibration_error(probabilities, targets, bins=10)
    assert 0 <= calibration["ece"] <= 1
    coverage = risk_coverage(probabilities, targets, thresholds=[0.7])
    assert coverage == [{"threshold": 0.7, "selected": 1, "coverage": 0.5, "accuracy": 1.0}]


def test_metrics_support_variable_choice_counts() -> None:
    probabilities = [[0.7, 0.3], [0.1, 0.2, 0.7]]
    assert accuracy(probabilities, [0, 2]) == 1.0
    assert multiclass_brier(probabilities, [0, 2]) > 0


def test_permutation_stability_uses_semantic_ids() -> None:
    rows = [
        {
            "question_id": 1,
            "ordering": 0,
            "correct_id": "0",
            "predicted_id": "0",
            "probabilities": {"0": 0.8, "1": 0.2},
        },
        {
            "question_id": 1,
            "ordering": 1,
            "correct_id": "0",
            "predicted_id": "0",
            "probabilities": {"0": 0.7, "1": 0.3},
        },
    ]
    result = permutation_stability(rows)
    assert result["argmax_agreement"] == 1.0
    assert result["permutation_accuracy"] == 1.0
    assert result["mean_probability_delta"] == pytest.approx(0.1)
