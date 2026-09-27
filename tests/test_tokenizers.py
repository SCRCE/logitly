from __future__ import annotations

import pytest

from logitly.backends import create_backend


@pytest.mark.network
@pytest.mark.parametrize("model", ["lfm", "glm", "qwen"])
def test_pinned_tokenizer_boundaries(model: str) -> None:
    backend = create_backend(model, load_model=False)
    rendered = backend.render_answer_prefix("STATE:\nx\n\nQUESTION:\ny\n\nAnswer:")
    if model == "glm":
        assert rendered.endswith("<|assistant|>")
        assert "<think>" not in rendered
    elif model == "lfm":
        assert rendered.endswith("<|im_start|>assistant\n")
        assert "<think>" not in rendered
    else:
        assert rendered.endswith("</think>\n\n")
