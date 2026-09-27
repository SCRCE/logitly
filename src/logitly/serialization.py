from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence

from .constants import MAX_CRITERION_CHARS, MAX_QUESTION_CHARS, MAX_STATE_CHARS
from .types import State


def serialize_state(state: State) -> str:
    if isinstance(state, str):
        rendered = state
    else:
        _validate_json_value(state)
        rendered = json.dumps(
            state,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    if not rendered.strip():
        raise ValueError("state must not be empty")
    if len(rendered) > MAX_STATE_CHARS:
        raise ValueError(f"state exceeds {MAX_STATE_CHARS} characters")
    return rendered


def validate_text(value: str, name: str, maximum: int = MAX_QUESTION_CHARS) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds {maximum} characters")
    return value.strip()


def normalize_choices(
    choices: Mapping[str, str | None] | Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if isinstance(choices, Mapping):
        pairs = list(choices.items())
        ids = tuple(validate_text(str(key), "choice id", 256) for key, _ in pairs)
        descriptions = tuple(
            validate_text(str(description if description is not None else key), "choice", MAX_CRITERION_CHARS)
            for key, description in pairs
        )
    elif isinstance(choices, Sequence) and not isinstance(choices, (str, bytes)):
        descriptions = tuple(validate_text(str(item), "choice", MAX_CRITERION_CHARS) for item in choices)
        ids = descriptions
    else:
        raise ValueError("choices must be a mapping or a non-string sequence")
    if not 2 <= len(ids) <= 20:
        raise ValueError("choice requires between 2 and 20 options")
    if len(set(ids)) != len(ids):
        raise ValueError("choice ids must be unique")
    return ids, descriptions


def normalize_levels(levels: Sequence[str]) -> tuple[str, ...]:
    if isinstance(levels, (str, bytes)) or not isinstance(levels, Sequence):
        raise ValueError("levels must be a non-string sequence")
    normalized = tuple(validate_text(str(level), "level", MAX_CRITERION_CHARS) for level in levels)
    if not 2 <= len(normalized) <= 10:
        raise ValueError("score requires between 2 and 10 levels")
    if len(set(normalized)) != len(normalized):
        raise ValueError("score levels must be unique")
    return normalized


def _validate_json_value(value: object, path: str = "state") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_json_value(child, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string object key")
            _validate_json_value(child, f"{path}.{key}")
        return
    raise ValueError(f"{path} contains unsupported value {type(value).__name__}")
