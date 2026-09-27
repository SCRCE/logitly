"""vLLM 0.26.0 pooling models retaining the original causal LM head.

This module is lazy-imported by vLLM workers, never by core library imports.
"""
import weakref

import torch
from vllm.model_executor.layers.pooler.abstract import Pooler
from vllm.model_executor.layers.pooler.seqwise.methods import LastPool
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.model_executor.models.lfm2 import Lfm2ForCausalLM
from vllm.model_executor.models.glm4 import Glm4ForCausalLM
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm.model_executor.models.gpt_oss import GptOssForCausalLM


class VocabularyPooler(Pooler):
    def __init__(self, model, token_ids):
        super().__init__()
        # A weak reference avoids a model -> pooler -> model module cycle.
        self._owner = weakref.ref(model)
        self.last = LastPool()
        self.register_buffer("label_ids", torch.tensor(token_ids, dtype=torch.long), persistent=False)

    def get_supported_tasks(self):
        return {"embed"}

    def forward(self, hidden_states, pooling_metadata):
        owner = self._owner()
        if owner is None:
            raise RuntimeError("LM head owner has been released")
        if not hidden_states.is_cuda:
            raise RuntimeError("Decision pooling requires CUDA hidden states")
        if not getattr(self, "_placement_checked", False):
            bad = [name for name, value in owner.named_parameters() if not value.is_cuda]
            if bad:
                raise RuntimeError(f"Non-CUDA model parameters: {bad[:5]}")
            self._placement_checked = True
        last = self.last(hidden_states, pooling_metadata)
        logits = owner.compute_logits(last)
        if logits is None:
            raise RuntimeError("Decision pooling requires single-GPU logits")
        return logits.index_select(-1, self.label_ids.to(logits.device)).float()


class DecisionPoolingMixin:
    is_pooling_model = True
    default_seq_pooling_type = "LAST"
    default_tok_pooling_type = "ALL"
    attn_type = "decoder"
    score_type = "bi-encoder"

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        ids = vllm_config.model_config.hf_config.logitly_label_ids
        self.pooler = VocabularyPooler(self, ids)


class DecisionLlama(DecisionPoolingMixin, LlamaForCausalLM):
    pass


class DecisionQwen2(DecisionPoolingMixin, Qwen2ForCausalLM):
    pass


class DecisionLfm2(DecisionPoolingMixin, Lfm2ForCausalLM):
    pass


class DecisionGlm4(DecisionPoolingMixin, Glm4ForCausalLM):
    pass


class DecisionQwen3_5(DecisionPoolingMixin, Qwen3_5ForConditionalGeneration):
    pass


class DecisionGptOss(DecisionPoolingMixin, GptOssForCausalLM):
    pass
