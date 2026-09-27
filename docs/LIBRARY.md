# Logitly

`logitly` wraps frozen causal LLMs as Noul, Choice, and Score decision engines. It reads the original LM-head vocabulary logits at the assistant answer boundary, selects the requested label tokens, and normalizes only those scores. No decoding, sampler, training, or replacement head is used.

The distribution, import package, and CLI are all named `logitly`.

## Install

Install the runtime you need from PyPI:

```bash
python -m pip install 'logitly[transformers]'
python -m pip install 'logitly[transformers,quantized]'  # quantized checkpoints
```

For development, install from a checkout with `python -m pip install '.[transformers]'`. Use separate environments for Transformers and vLLM; their pinned PyTorch versions differ. For a native CUDA llama.cpp build, set `CMAKE_ARGS='-DGGML_CUDA=ON'` and install the `llama-cpp` extra. The base package has no runtime dependencies and imports neither Torch nor vLLM; runtime imports are lazy.

The optional browser demo uses the same `browser-harness==0.1.13` package as
`browser-use/jev-ultrafast`:

```bash
python -m pip install 'logitly[vllm,browser]'
python -m logitly.demos.browser --model qwen --runtime vllm
```

Browser Harness must be connected to a running Chrome DevTools endpoint through
`BU_CDP_URL`. Model sampling remains disabled; the browser policy batches the
operation and applicable target questions through `decide_many`.

## Python

```python
from logitly import DecisionModel, ChoiceRequest, NoulRequest

with DecisionModel.from_pretrained('lfm', runtime='transformers') as engine:
    print(engine.validate())
    print(engine.choice(
        {'device': 'unknown', 'amount': 9700},
        'What action is appropriate?',
        {'close': 'Close as benign', 'review': 'Request analyst review', 'block': 'Block immediately'},
    ))
    print(engine.decide_many([
        NoulRequest('The order arrived damaged.', 'Does this require follow-up?'),
        ChoiceRequest('Customer requests a refund.', 'Which team?', ['billing', 'engineering']),
    ], batch_size=2))
```

`lfm`, `glm`, and `qwen` aliases resolve the existing pinned repositories. An arbitrary HF repository ID resolves to an immutable revision before download. Local directories are accepted. A generic runtime cannot load every architecture: an unsupported model raises `CompatibilityError`. The old multimodal Qwen CPU vision-tower placement is not permitted in this CUDA-only API.

To wrap an existing model, call `model.eval()` first, then `DecisionModel.from_model(model, tokenizer, profile=...)`. The wrapper never changes its placement or training mode. Closing the wrapper leaves caller-owned weights alive. Concurrent access through a single wrapper is serialized.

## Profiles and extensions

`ModelProfile` provides chat-template keyword arguments, an expected assistant suffix, optional `renderer(tokenizer, prompt)`, and explicit label palettes. Defaults are A–T, Y/N, and 0–9. Every actual rendered request verifies single-token continuation and distinct label IDs. The library does not guess reasoning/channel boundaries.

```python
from logitly import ModelProfile, register_profile

register_profile('my-model', ModelProfile(
    name='my-model',
    chat_template_kwargs={'enable_thinking': False},
    assistant_suffix='<|im_start|>assistant\n',
))
```

Register a runtime with `register_runtime(name, factory)`. Factories accept a model ID/path and loader options. A runtime exposes `tokenizer`, `encode(text)`, `restricted_logits(token_ids, label_ids)`, `validate_device()`, `close()`, `peak_memory_bytes()`, `reset_peak_memory()`, and metadata attributes `name`, `model_name`, `revision`, `device`, `max_input_tokens`, `last_forward_seconds`. Returned rows contain raw selected scores in request/label order; core performs softmax. Unavailable GPU measurements use `None`.

## Runtimes

- **Transformers:** `AutoModelForCausalLM` with explicit single-device CUDA placement and final-position logits. A model without a supported last-logit argument needs a custom adapter; the library will not allocate full-sequence vocabulary logits as a fallback.
- **vLLM 0.26.0:** custom causal pooling models for Llama, Qwen2, and LFM2 architecture classes. They retain and call the original LM head; `encode()` uses the pooling runner and never calls the generation sampler. Prefix reuse and chunked prefill are disabled. Other architectures need a pooling adapter. Integration requires GPU validation for each supported model/version.
- **llama-cpp-python 0.3.16:** native CUDA GGUF loading, tokenization, chat rendering, and batched prompt evaluation with separate sequence IDs. No high-level completion object is constructed. All weight buffers, including input embeddings, are assigned to CUDA. Requires a local GGUF file, a native-supported chat template, and verified complete GPU model placement. Native templates cannot silently ignore thinking flags.

All runtimes require CUDA for model computation. Host tokenization, loader bookkeeping, and small returned scores necessarily consume CPU RAM. `close()` drops owned resources without moving weights to CPU. A CLI supervisor owns its worker process group, including subprocesses, and tears it down on success, interruption, or failure.

## CLI and benchmark

```bash
logitly validate lfm
logitly playground lfm --port 8001
logitly benchmark lfm --size 128 --batch-sizes 1,2,4,8,16,32,64,128 --output results/library-lfm
logitly validate /absolute/path/model.gguf --runtime llama_cpp --profile generic
logitly validate /absolute/path/model --runtime vllm --profile generic
```

The playground reuses existing presets and API. Choose a free port while another playground is active. The benchmark uses identical MMLU-Pro requests at every batch size, records token lengths and padding, performs five warmups and twenty repetitions by default, and checkpoints throughput JSON after each size. It also writes canonical/permuted predictions, accuracy/calibration/stability metrics, CSV tables, plots, and a Markdown report. Runtime timing is synchronous adapter time; vLLM includes scheduling/IPC, so it is not presented as isolated GPU kernel time. `None` GPU memory means unavailable, not zero usage. Host high-water measurements include startup and are labeled separately from current RSS.

Runtime environments and caches can be kept inside the project by setting the documented cache environment variables.

## Docker

The Compose service provides the CUDA runtime and mounts the checkout at `/workspace`:

```bash
cp .env.example .env
docker compose build
docker compose run --rm logitly validate lfm
docker compose run --rm --service-ports logitly playground lfm --host 0.0.0.0
```

Open <http://localhost:8000> for the playground. Model downloads and runtime caches remain in the mounted `.cache/` directory.

## Validation

```bash
PYTHONPATH=src pytest -m 'not network and not gpu'
LOGITLY_TEST_MODEL=/absolute/path/local-lfm-checkpoint PYTHONPATH=src pytest -m gpu
python -m build
```

Confidence is `max(probabilities)`, not a calibrated correctness guarantee. Unsupported runtimes/models fail rather than generating an answer or changing the requested runtime.

### Verified on the RTX 3090

Real GPU checks cover LFM2.5-1.2B-Instruct through Transformers and SmolLM2-135M-Instruct through all three runtimes (Q8_0 GGUF for llama.cpp). They exercise all three primitives, mixed batching, repeated calls, and shutdown with generation/sampling patched to raise. The LFM test also compares against the original model's selected logits and checks repeated load/close releases model allocations. Other architecture adapters remain unverified on real weights.

The FP32 Transformers and vLLM validation runs used identical prompt tokens. Maximum single-versus-batch probability differences were 0.00000201 and 0.000605 respectively. The GGUF validation difference was 0.0111; its native renderer omits SmolLM's default system message, so that run is not a cross-runtime probability equivalence test. Supply an explicit common profile renderer when comparing runtime quality.

BF16 is the default for memory/performance, but it is not numerically invariant: the small SmolLM Transformers test showed up to 0.0567 probability drift when batched. Use `dtype="float32"` in `from_pretrained` for a precision control, where memory permits. Casting selected logits to FP32 does not undo rounding earlier in the model.

The included evaluation tests exercise the end-to-end reporting pipeline; they are not a quality or throughput claim for LFM. Full-size benchmarks remain a separate run.

For reproducing native/vLLM conformance, run `scripts/validate_runtime.py` with a local model path, `--runtime`, and `--output`. Build environments need Python development headers; native CUDA wheels additionally need their CUDA runtime libraries on the loader path.
