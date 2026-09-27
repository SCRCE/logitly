from __future__ import annotations

from collections.abc import Mapping, Sequence

from .backends.base import DirectLogitBackend
from .constants import CHOICE_LABELS, NOUL_LABELS, SCORE_LABELS
from .prompts import render_choice_prompt, render_noul_prompt, render_score_prompt
from .serialization import normalize_choices, normalize_levels, serialize_state, validate_text
from .types import (
    ChoiceRequest,
    ChoiceResult,
    DecisionRequest,
    DecisionResult,
    NoulRequest,
    NoulResult,
    PreparedDecision,
    ScoreRequest,
    ScoreResult,
    State,
)


class DecisionEngine:
    """Decisions extracted from restricted next-token logits."""

    def __init__(self, backend: DirectLogitBackend, *, default_batch_size: int = 1) -> None:
        if default_batch_size < 1:
            raise ValueError("default_batch_size must be positive")
        self.backend = backend
        self.default_batch_size = default_batch_size

    def noul(self, state: State, question: str) -> NoulResult:
        return self.decide_many([NoulRequest(state=state, question=question)])[0]  # type: ignore[return-value]

    def choice(
        self,
        state: State,
        question: str,
        choices: Mapping[str, str | None] | Sequence[str],
    ) -> ChoiceResult:
        return self.decide_many([ChoiceRequest(state=state, question=question, choices=choices)])[0]  # type: ignore[return-value]

    def score(self, state: State, question: str, levels: Sequence[str]) -> ScoreResult:
        return self.decide_many([ScoreRequest(state=state, question=question, levels=levels)])[0]  # type: ignore[return-value]

    def decide_many(
        self,
        requests: Sequence[DecisionRequest],
        batch_size: int | None = None,
    ) -> list[DecisionResult]:
        if not requests:
            return []
        effective_batch = batch_size or self.default_batch_size
        if effective_batch < 1:
            raise ValueError("batch_size must be positive")
        prepared = [self._prepare(request) for request in requests]
        all_probabilities: list[list[float]] = []
        for start in range(0, len(prepared), effective_batch):
            all_probabilities.extend(self.backend.score(prepared[start : start + effective_batch]))
        return [
            self._result(decision, probabilities)
            for decision, probabilities in zip(prepared, all_probabilities, strict=True)
        ]

    def _prepare(self, request: DecisionRequest) -> PreparedDecision:
        state = serialize_state(request.state)
        question = validate_text(request.question, "question")
        request_id = validate_text(request.id, "request id", 256)
        if isinstance(request, NoulRequest):
            return PreparedDecision(
                request_id=request_id,
                kind="noul",
                prompt=render_noul_prompt(state, question),
                labels=NOUL_LABELS,
                semantic_ids=("yes", "no"),
            )
        if isinstance(request, ChoiceRequest):
            ids, descriptions = normalize_choices(request.choices)
            labels = CHOICE_LABELS[: len(ids)]
            return PreparedDecision(
                request_id=request_id,
                kind="choice",
                prompt=render_choice_prompt(state, question, labels, descriptions),
                labels=labels,
                semantic_ids=ids,
            )
        if isinstance(request, ScoreRequest):
            levels = normalize_levels(request.levels)
            labels = SCORE_LABELS[: len(levels)]
            return PreparedDecision(
                request_id=request_id,
                kind="score",
                prompt=render_score_prompt(state, question, labels, levels),
                labels=labels,
                semantic_ids=tuple(str(index) for index in range(len(levels))),
                levels=levels,
            )
        raise TypeError(f"unsupported request type: {type(request).__name__}")

    @staticmethod
    def _result(decision: PreparedDecision, probabilities: Sequence[float]) -> DecisionResult:
        if len(probabilities) != len(decision.semantic_ids):
            raise RuntimeError("backend returned the wrong number of probabilities")
        total = float(sum(probabilities))
        if not 0.999 <= total <= 1.001:
            raise RuntimeError(f"probabilities sum to {total}, expected 1")
        values = [float(value) for value in probabilities]
        winner = max(range(len(values)), key=values.__getitem__)
        if decision.kind == "noul":
            return NoulResult(yes=values[0], no=values[1])
        if decision.kind == "choice":
            mapped = dict(zip(decision.semantic_ids, values, strict=True))
            return ChoiceResult(
                probabilities=mapped,
                choice=decision.semantic_ids[winner],
                confidence=values[winner],
            )
        if decision.kind == "score":
            expected = sum(index * probability for index, probability in enumerate(values))
            return ScoreResult(
                distribution=values,
                score=expected,
                confidence=values[winner],
            )
        raise RuntimeError(f"unknown prepared decision kind {decision.kind!r}")
