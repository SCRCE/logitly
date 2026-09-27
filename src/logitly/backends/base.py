from __future__ import annotations

import gc
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..constants import (
    CHOICE_LABELS,
    DEFAULT_MAX_INPUT_TOKENS,
    NOUL_LABELS,
    SCORE_LABELS,
    ModelSpec,
)
from ..storage import hf_cache_dir, project_root
from ..types import PreparedDecision


class DirectLogitBackend(ABC):
    """Shared one-forward-pass restricted-logit implementation."""

    def __init__(
        self,
        spec: ModelSpec,
        *,
        cache_dir: Path | None = None,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        load_model: bool = True,
    ) -> None:
        self.spec = spec
        self.cache_dir = cache_dir or hf_cache_dir(project_root())
        self.max_input_tokens = max_input_tokens
        self.tokenizer: Any = None
        self.model: Any = None
        self.primary_device: Any = None
        self.last_forward_seconds = 0.0
        self._label_token_ids: dict[str, int] = {}
        self._load_tokenizer()
        self._verify_label_palette()
        if load_model:
            self._load_model()

    @property
    def model_name(self) -> str:
        return self.spec.display_name

    def _load_tokenizer(self) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.spec.repo_id,
            revision=self.spec.revision,
            cache_dir=str(self.cache_dir),
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise RuntimeError(f"{self.spec.repo_id} tokenizer has neither pad nor EOS token")
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @abstractmethod
    def _load_model(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def render_answer_prefix(self, user_prompt: str) -> str:
        raise NotImplementedError

    def _verify_label_palette(self) -> None:
        probe = (
            "STATE:\nprobe\n\nQUESTION:\nSelect a label.\n\n"
            "CHOICES:\nA. first\nB. second\n\n"
            "Return exactly one label from:\nA\nB\n\nAnswer:"
        )
        prefix = self.render_answer_prefix(probe)
        base_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        for label in (*CHOICE_LABELS, *NOUL_LABELS, *SCORE_LABELS):
            combined = self.tokenizer.encode(prefix + label, add_special_tokens=False)
            if combined[: len(base_ids)] != base_ids or len(combined) != len(base_ids) + 1:
                raise RuntimeError(
                    f"label {label!r} is not one token at the rendered answer boundary for {self.spec.repo_id}"
                )
            token_id = int(combined[-1])
            isolated = self.tokenizer.encode(label, add_special_tokens=False)
            if isolated != [token_id]:
                raise RuntimeError(
                    f"label {label!r} changes token identity at the answer boundary for {self.spec.repo_id}"
                )
            self._label_token_ids[label] = token_id

    def score(self, decisions: Sequence[PreparedDecision]) -> list[list[float]]:
        if not decisions:
            return []
        if self.model is None:
            raise RuntimeError("backend was initialized without model weights")

        import torch

        rendered = [self.render_answer_prefix(decision.prompt) for decision in decisions]
        cpu_encoded = self.tokenizer(
            rendered,
            add_special_tokens=False,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        sequence_length = int(cpu_encoded["input_ids"].shape[1])
        if sequence_length > self.max_input_tokens:
            raise ValueError(
                f"rendered batch contains {sequence_length} tokens; maximum is {self.max_input_tokens}"
            )
        encoded = {
            key: value.to(device=self.primary_device, non_blocking=False)
            for key, value in cpu_encoded.items()
        }
        # The forward pass owns only CUDA tensors. Do not retain a host copy of
        # the padded batch or the rendered prompt strings while the model runs.
        cpu_encoded.clear()
        del cpu_encoded, rendered

        uses_cuda = torch.cuda.is_available() and torch.device(self.primary_device).type == "cuda"
        if uses_cuda:
            torch.cuda.synchronize(self.primary_device)
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(
                **encoded,
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            )
        if uses_cuda:
            torch.cuda.synchronize(self.primary_device)
        self.last_forward_seconds = time.perf_counter() - start

        final_logits = outputs.logits[:, -1, :]
        result: list[list[float]] = []
        for row, decision in enumerate(decisions):
            token_ids = torch.tensor(
                [self._label_token_ids[label] for label in decision.labels],
                dtype=torch.long,
                device=final_logits.device,
            )
            restricted = final_logits[row].index_select(0, token_ids)
            probabilities = torch.softmax(restricted.float(), dim=-1)
            result.append([float(value) for value in probabilities.cpu().tolist()])
        return result

    def close(self) -> None:
        """Release the backend without ever moving its weights back to CPU."""
        model = self.model
        self.model = None
        self.tokenizer = None
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except (ImportError, RuntimeError):
            pass

    def peak_memory_bytes(self) -> tuple[int, int]:
        try:
            import torch

            if torch.cuda.is_available() and torch.device(self.primary_device).type == "cuda":
                return (
                    int(torch.cuda.max_memory_allocated(self.primary_device)),
                    int(torch.cuda.max_memory_reserved(self.primary_device)),
                )
        except (ImportError, RuntimeError):
            pass
        return 0, 0

    def reset_peak_memory(self) -> None:
        try:
            import torch

            if torch.cuda.is_available() and torch.device(self.primary_device).type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.primary_device)
        except (ImportError, RuntimeError):
            return
