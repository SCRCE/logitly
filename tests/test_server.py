from __future__ import annotations

import pytest

from logitly.engine import DecisionEngine
from logitly.server import execute_decision


class FakeBackend:
    def score(self, decisions):
        rows = []
        for decision in decisions:
            if decision.kind == "noul":
                rows.append([0.8, 0.2])
            elif decision.kind == "choice":
                rows.append([0.25, 0.75])
            else:
                rows.append([0.1, 0.2, 0.7])
        return rows


def test_execute_all_playground_decisions() -> None:
    engine = DecisionEngine(FakeBackend())  # type: ignore[arg-type]
    noul = execute_decision(engine, {"type": "noul", "state": "s", "question": "q"})
    choice = execute_decision(
        engine,
        {
            "type": "choice",
            "state": {"case": 1},
            "question": "q",
            "choices": [
                {"id": "close", "description": "Close it"},
                {"id": "review", "description": "Review it"},
            ],
        },
    )
    score = execute_decision(
        engine,
        {"type": "score", "state": "s", "question": "q", "levels": ["low", "medium", "high"]},
    )
    assert noul["result"]["yes"] == 0.8
    assert choice["result"]["choice"] == "review"
    assert score["result"]["score"] == pytest.approx(1.6)


def test_execute_rejects_unknown_type() -> None:
    engine = DecisionEngine(FakeBackend())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="type must be"):
        execute_decision(engine, {"type": "generate", "state": "s", "question": "q"})
