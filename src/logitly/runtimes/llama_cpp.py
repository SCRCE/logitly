"""Native llama.cpp prefill only: no Llama completion object or sampler."""
import ctypes as C
import hashlib
import importlib.metadata
import math
import re
import threading
import time
from pathlib import Path

from ..errors import CompatibilityError

_LOAD_LOCK = threading.Lock()
_INITIALIZED = False


class TensorBufferOverride(C.Structure):
    _fields_ = [("pattern", C.c_char_p), ("buft", C.c_void_p)]


class NativeTokenizer:
    def __init__(self, runtime, template):
        self.runtime = runtime
        self.chat_template = template

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True, **kwargs):
        if kwargs:
            raise CompatibilityError("Native GGUF templates cannot apply thinking flags; supply a custom verified profile renderer")
        lib = self.runtime.lib
        values = (lib.llama_chat_message * len(messages))()
        for i, message in enumerate(messages):
            values[i].role = message["role"].encode()
            values[i].content = message["content"].encode()
        size = sum(len(m["content"].encode()) for m in messages) + 4096
        buffer = C.create_string_buffer(size)
        needed = lib.llama_chat_apply_template(self.chat_template, values, len(messages), add_generation_prompt, buffer, size)
        if needed < 0:
            raise CompatibilityError("llama.cpp does not support this GGUF chat template")
        if needed >= size:
            buffer = C.create_string_buffer(needed + 1)
            needed = lib.llama_chat_apply_template(self.chat_template, values, len(messages), add_generation_prompt, buffer, needed + 1)
        text = buffer.raw[:needed].decode()
        return self.runtime.encode(text) if tokenize else text


class LlamaCppRuntime:
    name = "llama_cpp"
    VERSION = "0.3.16"

    def __init__(self, model, *, device="cuda:0", max_input_tokens=4096, **kwargs):
        if kwargs:
            raise TypeError(f"Unsupported GGUF options: {', '.join(kwargs)}")
        if device != "cuda:0":
            raise CompatibilityError("Native GGUF adapter currently supports cuda:0 only")
        if importlib.metadata.version("llama-cpp-python") != self.VERSION:
            raise CompatibilityError(f"GGUF adapter requires llama-cpp-python=={self.VERSION} built with GGML_CUDA=ON")
        from llama_cpp import llama_cpp as lib
        source_path = Path(model)
        path = source_path.resolve(strict=True)
        if not path.is_file() or source_path.suffix.lower() != ".gguf":
            raise ValueError("llama_cpp requires an explicit local .gguf file")
        self.lib = lib
        self.model = self.ctx = None
        self.device = device
        self.model_name = str(source_path.absolute())
        self.max_input_tokens = max_input_tokens
        self.last_forward_seconds = 0.0
        self.tokenizer = None
        self._gpu_verified = False
        with path.open("rb") as source:
            self.revision = "sha256:" + hashlib.file_digest(source, "sha256").hexdigest()
        global _INITIALIZED
        logs = []

        @lib.llama_log_callback
        def capture(level, text, user_data):
            logs.append(text.decode(errors="replace"))

        try:
            with _LOAD_LOCK:
                if not _INITIALIZED:
                    lib.llama_backend_init()
                    _INITIALIZED = True
                if not lib.llama_supports_gpu_offload():
                    raise CompatibilityError("llama.cpp was built without GPU support")
                lib.llama_log_set(capture, None)
                try:
                    params = lib.llama_model_default_params()
                    params.n_gpu_layers = 2**31 - 1
                    params.main_gpu = 0
                    params.split_mode = lib.LLAMA_SPLIT_MODE_NONE
                    params.use_mmap = True
                    params.use_mlock = False
                    # n_gpu_layers alone can leave the input embedding on CPU.
                    # Bind the native buffer override to put every weight on CUDA.
                    device_by_name = lib._lib.ggml_backend_dev_by_name
                    device_by_name.argtypes = [C.c_char_p]
                    device_by_name.restype = C.c_void_p
                    buffer_type = lib._lib.ggml_backend_dev_buffer_type
                    buffer_type.argtypes = [C.c_void_p]
                    buffer_type.restype = C.c_void_p
                    cuda = device_by_name(b"CUDA0")
                    if not cuda:
                        raise CompatibilityError("Native CUDA0 backend not found")
                    self._buffer_overrides = (TensorBufferOverride * 2)(
                        TensorBufferOverride(b".*", buffer_type(cuda)), TensorBufferOverride(None, None))
                    params.tensor_buft_overrides = C.cast(self._buffer_overrides, C.c_void_p)
                    self.model = lib.llama_model_load_from_file(str(path).encode(), params)
                finally:
                    lib.llama_log_set(lib.llama_log_callback(), None)
            if not self.model:
                raise CompatibilityError("GGUF model loading failed")
            log = "".join(logs)
            counts = re.findall(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU", log)
            if not counts or counts[-1][0] != counts[-1][1] or "CUDA" not in log:
                raise CompatibilityError("Cannot verify complete CUDA layer placement from native loader")
            # Some architectures keep embedding weights on the host even with
            # n_gpu_layers=-1. Reject them under this library's GPU contract.
            if re.search(r"CPU(?:_Mapped)?\s+model buffer size\s*=\s*(?!0\.00\b)[\d.]+", log):
                raise CompatibilityError("GGUF retains a CPU model buffer; this architecture requires a GPU placement adapter")
            self._gpu_verified = True
            template = lib.llama_model_chat_template(self.model, None)
            if not template:
                raise CompatibilityError("GGUF has no chat template")
            self.vocab = lib.llama_model_get_vocab(self.model)
            self.tokenizer = NativeTokenizer(self, template)
            self._context_shape = None
        except BaseException:
            self.close()
            raise

    def validate_device(self):
        if not self._gpu_verified:
            raise CompatibilityError("GGUF CUDA placement is not verified")

    def encode(self, text):
        data = text.encode()
        size = len(data) + 8
        tokens = (self.lib.llama_token * size)()
        count = self.lib.llama_tokenize(self.vocab, data, len(data), tokens, size, False, True)
        if count < 0:
            raise CompatibilityError("GGUF tokenization failed")
        return list(tokens[:count])

    def restricted_logits(self, token_ids, label_ids):
        lib = self.lib
        total = sum(map(len, token_ids))
        shape = (len(token_ids), math.ceil(max(map(len, token_ids)) / 256) * 256)
        if self._context_shape is None or self._context_shape[0] != shape[0] or self._context_shape[1] < shape[1]:
            if self.ctx:
                lib.llama_free(self.ctx)
                self.ctx = None
            params = lib.llama_context_default_params()
            params.n_ctx = shape[0] * shape[1]
            params.n_batch = params.n_ubatch = params.n_ctx
            params.n_seq_max = shape[0]
            params.offload_kqv = params.op_offload = True
            self.ctx = lib.llama_init_from_model(self.model, params)
            if not self.ctx:
                raise RuntimeError("GGUF CUDA context creation failed")
            self._context_shape = shape
        batch = lib.llama_batch_init(total, 0, 1)
        memory = lib.llama_get_memory(self.ctx)
        try:
            lib.llama_memory_clear(memory, True)
            batch.n_tokens = total
            offset = 0
            last_positions = []
            for sequence, tokens in enumerate(token_ids):
                for i, token in enumerate(tokens):
                    index = offset + i
                    batch.token[index] = token
                    batch.pos[index] = i
                    batch.n_seq_id[index] = 1
                    batch.seq_id[index][0] = sequence
                    batch.logits[index] = i == len(tokens) - 1
                offset += len(tokens)
                last_positions.append(offset - 1)
            started = time.perf_counter()
            if lib.llama_decode(self.ctx, batch) != 0:
                raise RuntimeError("GGUF prefill failed")
            lib.llama_synchronize(self.ctx)
            self.last_forward_seconds = time.perf_counter() - started
            rows = []
            for position, labels in zip(last_positions, label_ids, strict=True):
                logits = lib.llama_get_logits_ith(self.ctx, position)
                if not logits:
                    raise RuntimeError("GGUF returned no final-position logits")
                rows.append([float(logits[token]) for token in labels])
        finally:
            lib.llama_batch_free(batch)
            lib.llama_memory_clear(memory, True)
        return rows

    def peak_memory_bytes(self):
        return None, None

    def reset_peak_memory(self):
        pass

    def close(self):
        if self.ctx:
            self.lib.llama_free(self.ctx)
            self.ctx = None
        if self.model:
            self.lib.llama_model_free(self.model)
            self.model = None
        self.tokenizer = None
