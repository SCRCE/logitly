"""Explicit opt-in CUDA acceptance tests; never download a model in pytest."""
import os
import weakref

import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture
def model_path():
    value = os.environ.get("LOGITLY_TEST_MODEL")
    if not value:
        pytest.skip("Set LOGITLY_TEST_MODEL to an existing local checkpoint")
    return value


def test_transformers_real_logits_batching_no_generation_and_release(model_path, monkeypatch):
    import torch
    from transformers.generation.utils import GenerationMixin
    from logitly import DecisionModel, ChoiceRequest, NoulRequest, ScoreRequest
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    def forbidden(*a, **kw):
        raise AssertionError("generation is forbidden")
    monkeypatch.setattr(GenerationMixin, "generate", forbidden)
    before = torch.cuda.memory_allocated()
    engine = DecisionModel.from_pretrained(model_path, profile="lfm")
    runtime = engine.backend.runtime
    model_ref = weakref.ref(runtime.model)
    requests = [ChoiceRequest("The sky is blue.", "What color is the sky?", ["blue", "red"]),
                NoulRequest("Two plus two equals four.", "Is the statement correct?"),
                ScoreRequest("The item is highly relevant.", "Rate relevance.", ["low", "medium", "high"])]
    try:
        single = [engine.decide_many([request])[0] for request in requests]
        batched = engine.decide_many(requests, batch_size=3)
        assert single[0].probabilities == pytest.approx(batched[0].probabilities, abs=.025)
        assert single[1].yes == pytest.approx(batched[1].yes, abs=.025)
        assert single[2].distribution == pytest.approx(batched[2].distribution, abs=.025)
        prepared = engine._prepare(requests[0])
        prefix = engine.profile.render(runtime.tokenizer, prepared.prompt)
        tokens = runtime.encode(prefix)
        labels = [runtime.encode(prefix + label)[-1] for label in prepared.labels]
        with torch.inference_mode():
            original = runtime.model(input_ids=torch.tensor([tokens], device=runtime.device),
                                     use_cache=False, logits_to_keep=1).logits[0, -1, labels].float()
            expected = original.softmax(-1).tolist()
        assert list(single[0].probabilities.values()) == pytest.approx(expected, abs=1e-6)
        del original
        borrowed = DecisionModel.from_model(runtime.model, runtime.tokenizer, profile="lfm")
        borrowed.close()
        assert engine.choice("s", "Pick one", ["one", "two"])
    finally:
        engine.close()
    assert model_ref() is None
    # CUDA's cuBLAS workspace is process-global, not model-owned (8.5 MB).
    # Verify a second load/close does not accumulate another model or workspace.
    after_first = torch.cuda.memory_allocated()
    assert after_first <= before + 16_000_000
    with DecisionModel.from_pretrained(model_path, profile="lfm") as second:
        second.choice("s", "Pick one", ["one", "two"])
    assert torch.cuda.memory_allocated() <= after_first + 1_000_000
