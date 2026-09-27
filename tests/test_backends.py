from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from logitly.backends.base import DirectLogitBackend
from logitly.backends.lfm import LfmDirectBackend
from logitly.backends.qwen import QwenDirectBackend
from logitly.types import PreparedDecision


class TensorTokenizer:
    def __call__(self, rendered, **kwargs):
        del rendered, kwargs
        return {
            "input_ids": torch.tensor([[1, 2], [1, 2]]),
            "attention_mask": torch.tensor([[1, 1], [1, 1]]),
        }


class NoGenerateModel:
    def generate(self, *args, **kwargs):
        raise AssertionError("generate must never be called")

    def __call__(self, **kwargs):
        assert kwargs["use_cache"] is False
        assert kwargs["logits_to_keep"] == 1
        logits = torch.zeros((2, 1, 20))
        logits[:, :, 10] = 1.0
        logits[:, :, 11] = 3.0
        return SimpleNamespace(logits=logits)


class FakeDirectBackend(DirectLogitBackend):
    def _load_model(self):
        raise NotImplementedError

    def render_answer_prefix(self, user_prompt):
        return user_prompt


def test_direct_backend_never_calls_generate_and_batches_equally() -> None:
    backend = object.__new__(FakeDirectBackend)
    backend.model = NoGenerateModel()
    backend.tokenizer = TensorTokenizer()
    backend.primary_device = torch.device("cpu")
    backend.max_input_tokens = 16
    backend.last_forward_seconds = 0.0
    backend._label_token_ids = {"A": 10, "B": 11}
    decisions = [
        PreparedDecision(str(index), "choice", "prompt", ("A", "B"), ("x", "y"))
        for index in range(2)
    ]
    rows = backend.score(decisions)
    assert len(rows) == 2
    assert rows[0] == pytest.approx(rows[1])
    assert sum(rows[0]) == pytest.approx(1.0)
    assert rows[0][1] > rows[0][0]


def test_qwen_renderer_forces_both_non_thinking_flags() -> None:
    calls = {}

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.update(kwargs)
            return "prefix<think>\n\n</think>\n\n"

    backend = object.__new__(QwenDirectBackend)
    backend.tokenizer = Tokenizer()
    rendered = backend.render_answer_prefix("prompt")
    assert rendered.endswith("</think>\n\n")
    assert calls["enable_thinking"] is False
    assert calls["preserve_thinking"] is False


def test_qwen_renderer_rejects_reasoning_content() -> None:
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "prefix<think>reasoning</think>"

    backend = object.__new__(QwenDirectBackend)
    backend.tokenizer = Tokenizer()
    with pytest.raises(RuntimeError, match="reasoning content"):
        backend.render_answer_prefix("prompt")


def test_lfm_renderer_stops_at_assistant_boundary() -> None:
    calls = {}

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.update(kwargs)
            return "<|startoftext|><|im_start|>user\nprompt<|im_end|>\n<|im_start|>assistant\n"

    backend = object.__new__(LfmDirectBackend)
    backend.tokenizer = Tokenizer()
    rendered = backend.render_answer_prefix("prompt")
    assert rendered.endswith("<|im_start|>assistant\n")
    assert calls["tokenize"] is False
    assert calls["add_generation_prompt"] is True


def test_close_drops_model_without_moving_it_to_cpu() -> None:
    class ModelThatMustNotMove:
        def to(self, *args, **kwargs):
            raise AssertionError("backend teardown must never copy weights to CPU")

    backend = object.__new__(FakeDirectBackend)
    backend.model = ModelThatMustNotMove()
    backend.tokenizer = object()
    backend.close()
    assert backend.model is None
    assert backend.tokenizer is None
