from __future__ import annotations

import math

import pytest

from logitly.engine import DecisionEngine
from logitly.types import ChoiceRequest, ChoiceResult, NoulResult, ScoreResult


class FakeBackend:
    model_name = "fake"
    last_forward_seconds = 0.0

    def score(self, decisions):
        rows = []
        for decision in decisions:
            if decision.kind == "noul":
                rows.append([0.8, 0.2])
            elif decision.kind == "choice":
                raw = list(range(1, len(decision.labels) + 1))
                total = sum(raw)
                rows.append([value / total for value in raw])
            else:
                rows.append([0.1, 0.2, 0.7][: len(decision.labels)])
        return rows


def test_public_primitives() -> None:
    engine = DecisionEngine(FakeBackend())  # type: ignore[arg-type]
    noul = engine.noul({"case": 1}, "Is it true?")
    assert noul == NoulResult(yes=0.8, no=0.2)

    choice = engine.choice("state", "Pick", {"a": "First", "b": "Second", "c": "Third"})
    assert isinstance(choice, ChoiceResult)
    assert choice.choice == "c"
    assert choice.confidence == pytest.approx(0.5)
    assert sum(choice.probabilities.values()) == pytest.approx(1.0)

    score = engine.score("state", "Rate", ["low", "medium", "high"])
    assert isinstance(score, ScoreResult)
    assert score.score == pytest.approx(1.6)
    assert score.confidence == pytest.approx(0.7)


def test_decide_many_preserves_order_and_batches() -> None:
    engine = DecisionEngine(FakeBackend(), default_batch_size=2)  # type: ignore[arg-type]
    requests = [
        ChoiceRequest("s", f"q{index}", {"a": "A", "b": "B"}, id=str(index))
        for index in range(5)
    ]
    results = engine.decide_many(requests)
    assert len(results) == 5
    assert all(isinstance(result, ChoiceResult) and result.choice == "b" for result in results)


@pytest.mark.parametrize(
    "choices",
    [[], ["only"], [str(index) for index in range(21)], ["same", "same"]],
)
def test_invalid_choices_are_rejected(choices) -> None:
    engine = DecisionEngine(FakeBackend())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        engine.choice("state", "question", choices)


def test_non_finite_state_is_rejected() -> None:
    engine = DecisionEngine(FakeBackend())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        engine.noul({"bad": math.nan}, "question")
