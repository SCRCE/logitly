from __future__ import annotations

from collections.abc import Sequence


def render_choice_prompt(
    state: str,
    question: str,
    labels: Sequence[str],
    descriptions: Sequence[str],
) -> str:
    options = "\n".join(f"{label}. {description}" for label, description in zip(labels, descriptions, strict=True))
    allowed = "\n".join(labels)
    return (
        f"STATE:\n{state}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CHOICES:\n{options}\n\n"
        f"Return exactly one label from:\n{allowed}\n\n"
        "Answer:"
    )


def render_noul_prompt(state: str, question: str) -> str:
    return (
        f"STATE:\n{state}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Choose exactly one:\n\n"
        "Y = Yes\n"
        "N = No\n\n"
        "Return exactly one label from:\nY\nN\n\n"
        "Answer:"
    )


def render_score_prompt(
    state: str,
    question: str,
    labels: Sequence[str],
    levels: Sequence[str],
) -> str:
    options = "\n".join(f"{label}. {level}" for label, level in zip(labels, levels, strict=True))
    allowed = "\n".join(labels)
    return (
        f"STATE:\n{state}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"ORDERED LEVELS:\n{options}\n\n"
        f"Return exactly one level label from:\n{allowed}\n\n"
        "Answer:"
    )
