from __future__ import annotations

from ..constants import MODEL_SPECS
from .base import DirectLogitBackend


class QwenDirectBackend(DirectLogitBackend):
    CHAT_TEMPLATE_KWARGS = {
        "enable_thinking": False,
        "preserve_thinking": False,
    }

    def __init__(self, **kwargs: object) -> None:
        super().__init__(MODEL_SPECS["qwen"], **kwargs)

    def render_answer_prefix(self, user_prompt: str) -> str:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **self.CHAT_TEMPLATE_KWARGS,
        )
        if rendered.count("<think>") != 1 or rendered.count("</think>") != 1:
            raise RuntimeError("Qwen non-thinking template did not produce exactly one empty thinking block")
        content = rendered.split("<think>", 1)[1].split("</think>", 1)[0]
        if content.strip():
            raise RuntimeError("Qwen non-thinking template contains reasoning content")
        return rendered

    def _load_model(self) -> None:
        import torch
        from transformers import AutoModelForMultimodalLM

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen backend requires a CUDA GPU")
        self.primary_device = torch.device("cuda:0")
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.spec.repo_id,
            revision=self.spec.revision,
            cache_dir=str(self.cache_dir),
            dtype=torch.bfloat16,
            device_map={
                "model.visual": "cpu",
                "model.language_model": 0,
                "lm_head": 0,
            },
            low_cpu_mem_usage=True,
        )
        self.model.eval()
