from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, TypeAlias

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
State: TypeAlias = str | JsonValue


@dataclass(frozen=True)
class NoulRequest:
    state: State
    question: str
    id: str = "noul"


@dataclass(frozen=True)
class ChoiceRequest:
    state: State
    question: str
    choices: Mapping[str, str | None] | Sequence[str]
    id: str = "choice"


@dataclass(frozen=True)
class ScoreRequest:
    state: State
    question: str
    levels: Sequence[str]
    id: str = "score"


DecisionRequest: TypeAlias = NoulRequest | ChoiceRequest | ScoreRequest


@dataclass(frozen=True)
class NoulResult:
    yes: float
    no: float


@dataclass(frozen=True)
class ChoiceResult:
    probabilities: dict[str, float]
    choice: str
    confidence: float


@dataclass(frozen=True)
class ScoreResult:
    distribution: list[float]
    score: float
    confidence: float


DecisionResult: TypeAlias = NoulResult | ChoiceResult | ScoreResult


@dataclass(frozen=True)
class PreparedDecision:
    request_id: str
    kind: str
    prompt: str
    labels: tuple[str, ...]
    semantic_ids: tuple[str, ...]
    levels: tuple[str, ...] = ()
