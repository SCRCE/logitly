from __future__ import annotations

from ..constants import MODEL_SPECS
from .base import DirectLogitBackend


class GlmDirectBackend(DirectLogitBackend):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(MODEL_SPECS["glm"], **kwargs)

    def render_answer_prefix(self, user_prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _load_model(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        if not torch.cuda.is_available():
            raise RuntimeError("GLM backend requires a CUDA GPU")
        self.primary_device = torch.device("cuda:0")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.spec.repo_id,
            revision=self.spec.revision,
            cache_dir=str(self.cache_dir),
            dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        self.model.eval()
