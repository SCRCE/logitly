from __future__ import annotations

import gc
import os
from pathlib import Path

from ..constants import MODEL_SPECS
from .base import DirectLogitBackend


class LfmDirectBackend(DirectLogitBackend):
    """Fully-GPU BF16 backend for LiquidAI LFM2.5-1.2B-Instruct."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(MODEL_SPECS["lfm"], **kwargs)

    def render_answer_prefix(self, user_prompt: str) -> str:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if not rendered.endswith("<|im_start|>assistant\n"):
            raise RuntimeError("LFM chat template did not end at the assistant answer boundary")
        return rendered

    def _load_model(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        if not torch.cuda.is_available():
            raise RuntimeError("LFM backend requires a CUDA GPU")
        self.primary_device = torch.device("cuda:0")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.spec.repo_id,
            revision=self.spec.revision,
            cache_dir=str(self.cache_dir),
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            low_cpu_mem_usage=True,
            offload_state_dict=False,
            offload_buffers=False,
            weights_only=True,
        )
        self.model.eval()
        non_cuda_parameters = [name for name, value in self.model.named_parameters() if value.device.type != "cuda"]
        non_cuda_buffers = [name for name, value in self.model.named_buffers() if value.device.type != "cuda"]
        if non_cuda_parameters or non_cuda_buffers:
            samples = (non_cuda_parameters + non_cuda_buffers)[:5]
            raise RuntimeError(f"LFM must be fully CUDA-resident; non-CUDA tensors: {samples}")
        device_map = getattr(self.model, "hf_device_map", {})
        offloaded = {name: device for name, device in device_map.items() if not str(device).startswith("cuda")}
        if offloaded:
            raise RuntimeError(f"LFM offload is forbidden; non-CUDA device map entries: {offloaded}")
        torch.cuda.synchronize(self.primary_device)
        gc.collect()
        self._evict_checkpoint_file_cache()

    def _evict_checkpoint_file_cache(self) -> None:
        """Tell Linux that clean checkpoint pages are no longer needed after CUDA load."""
        if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
            return
        repo_dir = f"models--{self.spec.repo_id.replace('/', '--')}"
        snapshot = Path(self.cache_dir) / repo_dir / "snapshots" / self.spec.revision
        for path in snapshot.glob("*.safetensors"):
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(descriptor)
