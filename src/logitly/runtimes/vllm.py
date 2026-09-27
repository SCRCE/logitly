import importlib.metadata
import time

from ..errors import CompatibilityError
from .common import require_cuda, resolve_checkpoint


def register_vllm_models():
    """Entry point runs in both the parent and spawned vLLM workers."""
    from vllm import ModelRegistry
    for architecture in VllmRuntime.ARCHITECTURES.values():
        ModelRegistry.register_model(architecture, f"logitly.runtimes.vllm_plugin:{architecture}")
    import os
    if os.environ.get("LOGITLY_ASSERT_NO_SAMPLING") == "1":
        from vllm.v1.sample.sampler import Sampler
        def forbidden(*args, **kwargs):
            raise AssertionError("vLLM generation sampler was invoked during decision inference")
        Sampler.forward = forbidden


class VllmRuntime:
    name = "vllm"
    VERSION = "0.26.0"
    ARCHITECTURES = {
        "LlamaForCausalLM": "DecisionLlama",
        "Qwen2ForCausalLM": "DecisionQwen2",
        "Lfm2ForCausalLM": "DecisionLfm2",
        "Glm4ForCausalLM": "DecisionGlm4",
        "Qwen3_5ForConditionalGeneration": "DecisionQwen3_5",
        "GptOssForCausalLM": "DecisionGptOss",
    }

    def __init__(self, model, *, revision=None, cache_dir=None, device="cuda:0", dtype="bfloat16",
                 max_input_tokens=4096):
        if importlib.metadata.version("vllm") != self.VERSION:
            raise CompatibilityError(f"Decision pooling requires vllm=={self.VERSION}")
        self.device = require_cuda(device)
        if self.device != "cuda:0":
            raise CompatibilityError("vLLM adapter requires cuda:0 in its isolated worker")
        self.model_name = str(model)
        self.engine = None
        self.max_input_tokens = max_input_tokens
        self.last_forward_seconds = 0.0
        self.path, self.revision = resolve_checkpoint(model, revision, cache_dir)
        from transformers import AutoConfig, AutoTokenizer
        config = AutoConfig.from_pretrained(self.path, local_files_only=True)
        self.architecture = next((self.ARCHITECTURES[a] for a in config.architectures if a in self.ARCHITECTURES), None)
        if self.architecture is None:
            raise CompatibilityError(f"No vLLM decision pooler for {config.architectures}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        self.dtype = dtype
        self._palette = None

    def validate_device(self):
        require_cuda(self.device)
        # cpu_offload_gb=0 and single CUDA worker are fixed below.

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def prepare_labels(self, palette):
        if self.engine is not None:
            if not set(palette).issubset(self._palette):
                raise CompatibilityError("Label palette changed after vLLM initialization")
            return
        from vllm import LLM
        register_vllm_models()
        self._palette = tuple(sorted(set(palette)))
        model_options = {}
        if self.architecture == "DecisionQwen3_5":
            # This experiment is text-only; do not profile or accept images/video.
            model_options["limit_mm_per_prompt"] = {"image": 0, "video": 0}
            # FlashInfer JIT requires nvcc, which the target machine does not have.
            model_options["attention_config"] = {"backend": "TRITON_ATTN"}
            # Qwen's checkpoint requests FP8 KV, but Ampere Triton attention
            # needs BF16 KV on the RTX 3090.
            model_options["kv_cache_dtype"] = "bfloat16"
        self.engine = LLM(
            model=self.path, runner="pooling", convert="none", dtype=self.dtype,
            hf_overrides={"architectures": [self.architecture], "logitly_label_ids": list(self._palette)},
            pooler_config={"pooling_type": "LAST", "use_activation": False},
            tensor_parallel_size=1, cpu_offload_gb=0, enforce_eager=True,
            gpu_memory_utilization=0.90, load_format="safetensors",
            safetensors_load_strategy="lazy", use_tqdm_on_load=False,
            enable_prefix_caching=False, enable_chunked_prefill=False,
            max_model_len=self.max_input_tokens,
            **model_options,
        )

    def restricted_logits(self, token_ids, label_ids):
        from vllm import PoolingParams
        self.prepare_labels([x for row in label_ids for x in row])
        positions = {token: index for index, token in enumerate(self._palette)}
        started = time.perf_counter()
        outputs = self.engine.encode(
            [{"prompt_token_ids": ids} for ids in token_ids],
            pooling_params=PoolingParams(task="embed", use_activation=False), pooling_task="embed", use_tqdm=False,
        )
        self.last_forward_seconds = time.perf_counter() - started
        return [[float(out.outputs.data[positions[token]]) for token in labels]
                for out, labels in zip(outputs, label_ids, strict=True)]

    def peak_memory_bytes(self):
        # PyTorch counters in the parent do not measure a vLLM worker.
        return None, None

    def reset_peak_memory(self):
        pass

    def close(self):
        engine, self.engine = self.engine, None
        if engine is not None:
            engine.llm_engine.engine_core.shutdown()
        self.tokenizer = None
