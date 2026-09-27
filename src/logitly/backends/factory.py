from __future__ import annotations

from .base import DirectLogitBackend
from .glm import GlmDirectBackend
from .lfm import LfmDirectBackend
from .qwen import QwenDirectBackend


def create_backend(name: str, **kwargs: object) -> DirectLogitBackend:
    normalized = name.strip().lower()
    if normalized == "lfm":
        return LfmDirectBackend(**kwargs)
    if normalized == "glm":
        return GlmDirectBackend(**kwargs)
    if normalized == "qwen":
        return QwenDirectBackend(**kwargs)
    raise ValueError(f"unknown backend {name!r}; expected 'lfm', 'glm', or 'qwen'")
