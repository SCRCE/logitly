import math
import threading
from dataclasses import replace
from types import SimpleNamespace

from .engine import DecisionEngine
from .prompts import render_choice_prompt, render_noul_prompt, render_score_prompt
from .serialization import serialize_state, validate_text, normalize_choices, normalize_levels
from .types import ChoiceRequest, NoulRequest, ScoreRequest
from .errors import CompatibilityError
from .profiles import resolve_profile


def restricted_softmax(logits):
    if len(logits) < 2 or not all(math.isfinite(x) for x in logits):
        raise CompatibilityError("Expected at least two finite label logits")
    maximum = max(logits)
    values = [math.exp(float(x) - maximum) for x in logits]
    total = sum(values)
    return [value / total for value in values]


class RuntimeBackend:
    """Only selected label logits cross the runtime boundary."""
    def __init__(self, runtime, profile):
        self.runtime = runtime
        self.profile = profile
        self.model_name = runtime.model_name
        self.spec = SimpleNamespace(revision=runtime.revision)
        self.last_forward_seconds = 0.0

    def score(self, decisions):
        rendered = [self.profile.render(self.runtime.tokenizer, d.prompt) for d in decisions]
        token_ids, label_ids = [], []
        for text, decision in zip(rendered, decisions, strict=True):
            ids = self.runtime.encode(text)
            if len(ids) > self.runtime.max_input_tokens:
                raise ValueError(f"prompt exceeds {self.runtime.max_input_tokens} tokens; truncation is disabled")
            labels = []
            for label in decision.labels:
                continued = self.runtime.encode(text + label)
                if continued[:-1] != ids or len(continued) != len(ids) + 1:
                    raise CompatibilityError(f"{label!r} is not a single-token continuation at this answer boundary")
                labels.append(continued[-1])
            if len(set(labels)) != len(labels):
                raise CompatibilityError("Decision labels map to duplicate tokens")
            token_ids.append(ids)
            label_ids.append(labels)
        rows = self.runtime.restricted_logits(token_ids, label_ids)
        self.last_forward_seconds = self.runtime.last_forward_seconds
        if len(rows) != len(decisions) or any(len(row) != len(d.labels) for row, d in zip(rows, decisions)):
            raise CompatibilityError("Runtime returned the wrong number of decisions or logits")
        return [restricted_softmax(row) for row in rows]

    def close(self):
        self.runtime.close()

    def peak_memory_bytes(self):
        return self.runtime.peak_memory_bytes()

    def reset_peak_memory(self):
        self.runtime.reset_peak_memory()


RUNTIMES = {}


def register_runtime(name, factory):
    if name in RUNTIMES or name in ("transformers", "llama_cpp", "vllm"):
        raise ValueError(f"runtime already registered: {name}")
    RUNTIMES[name] = factory


class DecisionModel(DecisionEngine):
    def __init__(self, runtime, *, profile=None, batch_size=1):
        self.profile = resolve_profile(profile, runtime.model_name)
        self._lock = threading.RLock()
        self._closed = False
        super().__init__(RuntimeBackend(runtime, self.profile), default_batch_size=batch_size)

    @classmethod
    def from_pretrained(cls, model, *, runtime="transformers", profile=None, batch_size=1, **kwargs):
        from .constants import MODEL_SPECS
        if str(model) in MODEL_SPECS and runtime != "llama_cpp":
            spec = MODEL_SPECS[str(model)]
            kwargs.setdefault("revision", spec.revision)
            model = spec.repo_id
        selected = resolve_profile(profile, model)
        if runtime == "transformers":
            from .runtimes.transformers import TransformersRuntime as factory
        elif runtime == "llama_cpp":
            from .runtimes.llama_cpp import LlamaCppRuntime as factory
        elif runtime == "vllm":
            from .runtimes.vllm import VllmRuntime as factory
        else:
            try:
                factory = RUNTIMES[runtime]
            except KeyError:
                raise ValueError(f"Unknown runtime: {runtime}") from None
        instance = factory(model, **kwargs)
        try:
            result = cls(instance, profile=selected, batch_size=batch_size)
            result.validate()
            return result
        except BaseException:
            instance.close()
            raise

    @classmethod
    def from_model(cls, model, tokenizer, *, profile=None, batch_size=1, **kwargs):
        from .runtimes.transformers import TransformersRuntime
        runtime = TransformersRuntime.from_model(model, tokenizer, **kwargs)
        try:
            result = cls(runtime, profile=profile, batch_size=batch_size)
            result.validate()
            return result
        except BaseException:
            runtime.close()
            raise

    def _prepare(self, request):
        prepared = super()._prepare(request)
        state, question = serialize_state(request.state), validate_text(request.question, "question")
        if isinstance(request, ChoiceRequest):
            _, descriptions = normalize_choices(request.choices)
            labels = self.profile.choice_labels[:len(descriptions)]
            prompt = render_choice_prompt(state, question, labels, descriptions)
        elif isinstance(request, ScoreRequest):
            levels = normalize_levels(request.levels)
            labels = self.profile.score_labels[:len(levels)]
            prompt = render_score_prompt(state, question, labels, levels)
        else:
            labels = self.profile.noul_labels
            prompt = render_noul_prompt(state, question) if labels == ("Y", "N") else render_choice_prompt(state, question, labels, ("Yes", "No"))
        return replace(prepared, labels=labels, prompt=prompt)

    def decide_many(self, requests, batch_size=None):
        with self._lock:
            if self._closed:
                raise RuntimeError("DecisionModel is closed")
            effective = self.default_batch_size if batch_size is None else batch_size
            if effective < 1:
                raise ValueError("batch_size must be positive")
            results = []
            for start in range(0, len(requests), effective):
                results.extend(super().decide_many(requests[start:start + effective], batch_size=effective))
            return results

    def validate(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("DecisionModel is closed")
            runtime = self.backend.runtime
            runtime.validate_device()
            prefix = self.profile.render(runtime.tokenizer, "STATE:\nprobe\nQUESTION:\nSelect a label.\nAnswer:")
            ids = runtime.encode(prefix)
            labels = dict.fromkeys((*self.profile.choice_labels, *self.profile.noul_labels, *self.profile.score_labels))
            palette = {}
            for label in labels:
                full = runtime.encode(prefix + label)
                if len(full) != len(ids) + 1 or full[:-1] != ids:
                    raise CompatibilityError(f"Invalid boundary token for label {label!r}; supply explicit profile labels")
                palette[label] = full[-1]
            if hasattr(runtime, "prepare_labels"):
                runtime.prepare_labels(list(palette.values()))
            return {"model": runtime.model_name, "revision": runtime.revision,
                    "runtime": runtime.name, "device": runtime.device, "profile": self.profile.name,
                    "labels": palette, "generated_tokens": 0}

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self.backend.close()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("DecisionModel is closed")
        return self

    def __exit__(self, *exc):
        self.close()
