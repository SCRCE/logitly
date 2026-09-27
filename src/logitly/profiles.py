from dataclasses import dataclass, field
from typing import Callable, Any

from .errors import CompatibilityError


GPT_OSS_FINAL_PREFIX = "<|start|>assistant<|channel|>final<|message|>"


def render_gpt_oss_final(tokenizer, prompt):
    """Render Harmony through the start of an explicit final-channel message."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=False,
        reasoning_effort="low",
    )
    return rendered + GPT_OSS_FINAL_PREFIX


@dataclass(frozen=True)
class ModelProfile:
    name: str = "generic"
    chat_template_kwargs: dict = field(default_factory=dict)
    assistant_suffix: str | None = None
    renderer: Callable[[Any, str], str] | None = None
    choice_labels: tuple[str, ...] = tuple("ABCDEFGHIJKLMNOPQRST")
    noul_labels: tuple[str, ...] = ("Y", "N")
    score_labels: tuple[str, ...] = tuple(map(str, range(10)))

    def __post_init__(self):
        for labels, count in ((self.choice_labels, 20), (self.noul_labels, 2), (self.score_labels, 10)):
            if len(labels) != count or len(set(labels)) != count or any(not x.strip() for x in labels):
                raise ValueError(f"profile requires {count} distinct non-empty labels")

    def render(self, tokenizer, prompt):
        if self.renderer:
            rendered = self.renderer(tokenizer, prompt)
        else:
            if not getattr(tokenizer, "chat_template", None):
                raise CompatibilityError("Tokenizer has no chat template; supply a ModelProfile renderer")
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, **self.chat_template_kwargs,
            )
        if self.assistant_suffix and not rendered.endswith(self.assistant_suffix):
            raise CompatibilityError(f"{self.name}: unexpected assistant answer boundary")
        suffix = rendered[rendered.rfind(prompt) + len(prompt):] if prompt in rendered else rendered
        if "<think>" in suffix:
            reasoning = suffix.rsplit("<think>", 1)[1]
            if "</think>" not in reasoning or reasoning.split("</think>", 1)[0].strip():
                raise CompatibilityError("Template leaves reasoning enabled; supply a verified non-thinking profile")
        if suffix.rstrip().endswith("<|start|>assistant"):
            raise CompatibilityError("Harmony requires an explicit final-channel renderer")
        return rendered


PROFILES = {
    "generic": ModelProfile(),
    "lfm": ModelProfile("lfm", assistant_suffix="<|im_start|>assistant\n"),
    "glm": ModelProfile("glm", assistant_suffix="<|assistant|>"),
    "qwen": ModelProfile("qwen", {"enable_thinking": False, "preserve_thinking": False}, "</think>\n\n"),
    "gpt_oss": ModelProfile(
        "gpt_oss",
        {"reasoning_effort": "low"},
        GPT_OSS_FINAL_PREFIX,
        render_gpt_oss_final,
    ),
}


def register_profile(name, profile):
    if name in PROFILES:
        raise ValueError(f"profile already registered: {name}")
    PROFILES[name] = profile


def resolve_profile(profile, model_id=""):
    if isinstance(profile, ModelProfile):
        return profile
    if profile:
        return PROFILES[profile]
    lower = str(model_id).lower()
    key = (
        "lfm" if "lfm2" in lower
        else "qwen" if "qwen3" in lower
        else "gpt_oss" if "gpt-oss" in lower or "gpt_oss" in lower
        else "glm" if "glm-4-32b-0414" in lower
        else "generic"
    )
    return PROFILES[key]
