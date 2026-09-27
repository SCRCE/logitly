import json
import math
from types import SimpleNamespace

import pytest

from logitly import DecisionModel, ModelProfile, CompatibilityError, ChoiceRequest, NoulRequest, ScoreRequest
from logitly.core import restricted_softmax
from logitly.benchmark import throughput
from logitly.profiles import GPT_OSS_FINAL_PREFIX, PROFILES, resolve_profile


class Tokenizer:
    chat_template = "test"

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"] + "\nassistant\n"


class Runtime:
    name = "test"
    model_name = "fixture"
    revision = "fixture-v1"
    device = "cuda:0"
    max_input_tokens = 10000
    last_forward_seconds = .001

    def __init__(self):
        self.tokenizer = Tokenizer()
        self.closed = 0
        self.calls = []

    def validate_device(self):
        pass

    def encode(self, text):
        return list(map(ord, text))

    def restricted_logits(self, prompts, labels):
        self.calls.append((prompts, labels))
        return [[float(i) for i in range(len(row))] for row in labels]

    def close(self):
        self.closed += 1

    def reset_peak_memory(self):
        pass

    def peak_memory_bytes(self):
        return 10, 20


def test_heterogeneous_decisions_custom_labels_and_cleanup():
    runtime = Runtime()
    profile = ModelProfile(choice_labels=tuple("abcdefghijklmnopqrst"), noul_labels=("+", "-"))
    with DecisionModel(runtime, profile=profile) as engine:
        engine.validate()
        rows = engine.decide_many([
            ChoiceRequest({"b": 2, "a": 1}, "choose", {"x": "First", "y": "Second", "z": "Third"}),
            NoulRequest("state", "yes?"), ScoreRequest("state", "rate", ["low", "high"]),
        ], batch_size=3)
        assert rows[0].choice == "z"
        assert rows[0].probabilities["z"] == pytest.approx(math.exp(2) / (1 + math.e + math.exp(2)))
        assert rows[1].yes + rows[1].no == pytest.approx(1)
        assert rows[2].score == pytest.approx(rows[2].distribution[1])
        assert runtime.calls[0][1] == [[ord(x) for x in "abc"], [ord("+"), ord("-")], [ord("0"), ord("1")]]
        assert '"a":1,"b":2' in ''.join(map(chr, runtime.calls[0][0][0]))
    engine.close()
    assert runtime.closed == 1
    with pytest.raises(RuntimeError, match="closed"):
        engine.choice("s", "q", ["one", "two"])


def test_invalid_batch_and_nonfinite_logits():
    engine = DecisionModel(Runtime())
    with pytest.raises(ValueError, match="batch_size"):
        engine.decide_many([NoulRequest("s", "q")], batch_size=0)
    for row in ([float("nan"), 0], [float("inf"), 0], [0]):
        with pytest.raises(CompatibilityError):
            restricted_softmax(row)
    assert restricted_softmax([10000, 10000]) == [.5, .5]


def test_boundary_checked_on_every_actual_request():
    runtime = Runtime()
    original = runtime.encode
    runtime.encode = lambda text: original(text)[:-2] if "bad-question" in text and text.endswith("A") else original(text)
    engine = DecisionModel(runtime)
    engine.validate()
    with pytest.raises(CompatibilityError, match="single-token"):
        engine.choice("s", "bad-question", ["one", "two"])
    assert not runtime.calls


def test_same_workload_at_every_batch_size_and_incremental_output(tmp_path):
    runtime = Runtime()
    engine = DecisionModel(runtime)
    requests = [ChoiceRequest("s", f"q{i}", ["one", "two"]) for i in range(4)]
    output = tmp_path / "results.json"
    report = throughput(engine, requests, batch_sizes=(1, 2, 4), warmups=0, repetitions=1, output=output)
    assert [r["measured_decisions"] for r in report["rows"]] == [4, 4, 4]
    assert [r["measured_batches"] for r in report["rows"]] == [4, 2, 1]
    assert [p for call in runtime.calls[:4] for p in call[0]] == [p for call in runtime.calls[4:6] for p in call[0]] == runtime.calls[6][0]
    assert json.loads(output.read_text())["rows"] == report["rows"]


def test_softmax_is_restricted_only():
    assert restricted_softmax([10.2, 13.8, 11.1])[1] > .9


def test_wrapper_does_not_move_or_destroy_borrowed_model(monkeypatch):
    import torch
    from logitly.runtimes.transformers import TransformersRuntime
    class Borrowed:
        training = False
        config = SimpleNamespace(_name_or_path="borrowed")
        def forward(self, logits_to_keep=1):
            pass
        def eval(self):
            raise AssertionError("Do not mutate a caller-owned model")
        def to(self, *args):
            raise AssertionError("Do not move model weights")
    monkeypatch.setattr("logitly.runtimes.transformers.require_cuda", lambda d: "cuda:0")
    monkeypatch.setattr(TransformersRuntime, "validate_device", lambda self: None)
    model = Borrowed()
    tokenizer = Tokenizer()
    tokenizer.pad_token_id = 0
    runtime = TransformersRuntime.from_model(model, tokenizer)
    runtime.close()
    assert model.training is False
    assert runtime.model is None
    assert tokenizer.pad_token_id == 0


def test_missing_last_position_support_is_rejected(monkeypatch):
    from logitly.runtimes.transformers import TransformersRuntime
    monkeypatch.setattr("logitly.runtimes.transformers.require_cuda", lambda d: "cuda:0")
    monkeypatch.setattr(TransformersRuntime, "validate_device", lambda self: None)
    model = SimpleNamespace(training=False, forward=lambda input_ids: None)
    with pytest.raises(CompatibilityError, match="final-position"):
        TransformersRuntime.from_model(model, SimpleNamespace(pad_token_id=0))


def test_gpt_oss_profile_bypasses_analysis_with_explicit_final_channel():
    class HarmonyTokenizer:
        chat_template = "harmony"

        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "decision"}]
            assert kwargs == {
                "tokenize": False,
                "add_generation_prompt": False,
                "reasoning_effort": "low",
            }
            return "<|start|>user<|message|>decision<|end|>"

    rendered = PROFILES["gpt_oss"].render(HarmonyTokenizer(), "decision")
    assert rendered.endswith(GPT_OSS_FINAL_PREFIX)
    assert "<|channel|>analysis" not in rendered
    assert resolve_profile(None, "openai/gpt-oss-20b").name == "gpt_oss"
