import gc
import inspect
import time

from ..errors import CompatibilityError
from .common import TorchMemory, require_cuda, resolve_checkpoint


class TransformersRuntime(TorchMemory):
    name = "transformers"

    def __init__(self, model, *, revision=None, cache_dir=None, device="cuda:0",
                 dtype="bfloat16", max_input_tokens=4096):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
        self.device = require_cuda(device)
        self.model_name = str(model)
        self.model = None
        self.tokenizer = None
        self.owned = True
        self.max_input_tokens = max_input_tokens
        self.last_forward_seconds = 0.0
        if dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError("dtype must be bfloat16, float16, or float32")
        path, self.revision = resolve_checkpoint(model, revision, cache_dir)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            config = AutoConfig.from_pretrained(path, local_files_only=True)
            architectures = set(getattr(config, "architectures", ()) or ())
            model_factory = (
                AutoModelForImageTextToText
                if "Qwen3_5ForConditionalGeneration" in architectures
                else AutoModelForCausalLM
            )
            self.model = model_factory.from_pretrained(
                path, local_files_only=True, dtype=getattr(torch, dtype),
                device_map={"": self.device}, offload_buffers=False,
                trust_remote_code=False,
            )
            self._configure()
        except BaseException:
            self.close()
            raise

    @classmethod
    def from_model(cls, model, tokenizer, *, device="cuda:0", max_input_tokens=4096):
        obj = cls.__new__(cls)
        obj.device = require_cuda(device)
        obj.model, obj.tokenizer = model, tokenizer
        obj.model_name = getattr(getattr(model, "config", None), "_name_or_path", "caller-model")
        obj.revision = getattr(getattr(model, "config", None), "_commit_hash", None) or "caller-owned"
        obj.owned = False
        obj.max_input_tokens = max_input_tokens
        obj.last_forward_seconds = 0.0
        obj._configure()
        return obj

    def _configure(self):
        self.validate_device()
        if not self.owned and self.model.training:
            raise CompatibilityError("Call model.eval() before wrapping a caller-owned model")
        if self.owned:
            self.model.eval()
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tokenizer.eos_token_id
        if self.pad_id is None:
            raise CompatibilityError("Tokenizer must provide a pad or EOS token")
        signature = inspect.signature(self.model.forward).parameters
        self.last_logits_arg = next((key for key in ("logits_to_keep", "num_logits_to_keep") if key in signature), None)
        if self.last_logits_arg is None:
            raise CompatibilityError("Architecture needs a custom final-position LM-head adapter; full-sequence logits are disabled")

    def validate_device(self):
        import torch
        require_cuda(self.device)
        target = torch.device(self.device)
        bad = [name for name, tensor in (*self.model.named_parameters(), *self.model.named_buffers()) if tensor.device != target]
        if bad:
            raise CompatibilityError(f"Model tensors are not on {target}: {bad[:5]}")
        for placement in getattr(self.model, "hf_device_map", {}).values():
            if str(placement) not in (str(target), str(target.index)):
                raise CompatibilityError(f"Offloaded model placement: {placement}")

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def restricted_logits(self, token_ids, label_ids):
        import torch
        if not token_ids:
            return []
        if self.model is None:
            raise RuntimeError("Runtime is closed")
        width = max(map(len, token_ids))
        # Allocate the padded batch directly on CUDA. Only short token lists
        # and the final selected scores exist as Python host objects.
        inputs = torch.full((len(token_ids), width), self.pad_id, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(inputs)
        for row, ids in enumerate(token_ids):
            inputs[row, -len(ids):] = torch.tensor(ids, dtype=torch.long, device=self.device)
            mask[row, -len(ids):] = 1
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(input_ids=inputs, attention_mask=mask, use_cache=False,
                                 return_dict=True, output_hidden_states=False, output_attentions=False,
                                 **{self.last_logits_arg: 1})
            if outputs.logits.device != torch.device(self.device):
                raise CompatibilityError("Model returned logits outside the requested CUDA device")
            if outputs.logits.shape[:2] != (len(token_ids), 1):
                raise CompatibilityError("Model did not honor final-position-only logits")
            rows = [outputs.logits[row, -1, ids].float() for row, ids in enumerate(label_ids)]
            torch.cuda.synchronize(self.device)
            self.last_forward_seconds = time.perf_counter() - started
            result = [row.tolist() for row in rows]
        del rows, outputs, inputs, mask
        return result

    def close(self):
        model, self.model = self.model, None
        self.tokenizer = None
        del model
        if self.owned:
            gc.collect()
            import torch
            if torch.cuda.is_available():
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
