# Logitly

Logitly turns compatible causal LLMs into fast, bounded decision engines.

Give it a state, a question, and named choices. Logitly performs one prefill forward pass, reads the model's original next-token logits for fixed labels, and returns a probability distribution over only the choices you supplied.

```text
state + question + choices
            ↓
      frozen causal LLM
            ↓
   original LM-head logits
            ↓
 select A/B/C/… label logits
            ↓
          softmax
            ↓
 named probability distribution
```

No generated answer tokens. No fine-tuning. No replacement model head.

## Install

Logitly supports Python 3.11–3.13 and currently requires an NVIDIA CUDA GPU for Python inference.

```bash
python -m pip install "logitly[transformers]"
```

Optional runtimes and features are installed separately:

```bash
python -m pip install "logitly[transformers,quantized]"
python -m pip install "logitly[vllm]"
python -m pip install "logitly[llama-cpp]"
python -m pip install "logitly[browser]"
```

Use separate environments for Transformers and vLLM because their pinned runtime dependencies differ.

## Quick start

```python
from logitly import DecisionModel

with DecisionModel.from_pretrained("lfm") as model:
    result = model.choice(
        state={"amount": 9700, "device": "unknown"},
        question="What action should be taken?",
        choices={
            "close": "Close as benign",
            "review": "Request analyst review",
            "block": "Block immediately",
        },
    )

print(result.choice)
print(result.confidence)
print(result.probabilities)
```

Logitly provides three decision primitives:

```python
model.noul(state, question)
model.choice(state, question, choices)
model.score(state, question, levels)
```

It can also evaluate mixed requests in a batch:

```python
from logitly import ChoiceRequest, NoulRequest

results = model.decide_many([
    NoulRequest(
        state="The order arrived damaged.",
        question="Does this require follow-up?",
    ),
    ChoiceRequest(
        state="A customer requested a refund.",
        question="Which team should receive the case?",
        choices={
            "billing": "Billing",
            "support": "Customer support",
        },
    ),
])
```

## How it works

For choice labels \(\ell_1, \ldots, \ell_k\), Logitly reads the final-position vocabulary logit for each label token:

```text
zᵢ = model(prompt)[last_position, token(labelᵢ)]
pᵢ = softmax([z₁, …, zₖ])ᵢ
```

The softmax is calculated over the selected labels, not the entire vocabulary. Semantic IDs such as `review` or `block` are mapped back onto the resulting probabilities.

Each model profile defines its native chat boundary and verbalizers. Logitly verifies that every label is a distinct, single-token continuation at the exact assistant-answer boundary before inference begins.

## Model profiles

| Alias | Checkpoint | Runtime notes |
|---|---|---|
| `lfm` | `LiquidAI/LFM2.5-1.2B-Instruct` | Pinned BF16 checkpoint; recommended starting point |
| `qwen` | `RedHatAI/Qwen3.8-27B-INT4` | Pinned INT4 checkpoint; thinking disabled |
| `glm` | `mratsim/GLM-4-32B-0414.w4a16-gptq` | Pinned W4A16 GPTQ checkpoint |

Compatible Hugging Face model IDs, local checkpoints, and GGUF files can also be supplied directly. Unsupported architectures fail with a compatibility error instead of changing inference modes.

## Runtimes

- **Transformers** reads only the final-position logits through supported causal-LM interfaces.
- **vLLM** uses custom pooling adapters that retain the model's original LM head without invoking its generation sampler.
- **llama.cpp** evaluates GGUF prompts and reads selected logits without constructing a completion.

All three runtimes implement the same `DecisionModel` contract.

## CLI

```bash
logitly validate lfm
logitly playground lfm --port 8000
logitly benchmark lfm --size 128 --output results/lfm
```

The playground provides an interactive local interface for Choice, Noul, and Score. The benchmark records predictions, accuracy, calibration, permutation stability, latency, throughput, and GPU-memory measurements.

## Docker

```bash
cp .env.example .env
docker compose build
docker compose run --rm logitly validate lfm
docker compose run --rm --service-ports logitly playground lfm --host 0.0.0.0
```

Model and runtime caches remain under the ignored project-local `.cache/` directory.

## Demos

The package includes two bounded decision demonstrations:

```bash
python -m logitly.demos.snake --model lfm --steps 20 --show-board
python -m logitly.demos.browser --help
```

The browser policy turns visible DOM controls into a finite set of operations and targets. The model cannot invent selectors, coordinates, JavaScript, or unsupported actions. Browser support uses the same `browser-harness` transport as the public `browser-use/jev-ultrafast` project.

## Apple implementation

The [`apple/`](apple/) directory contains a Swift package and iPhone demo built around llama.cpp and Metal. It exposes the same Choice, Noul, Score, and batch interface with a bundled LiquidAI GGUF model and no sampler or completion loop.

```bash
./apple/scripts/bootstrap.sh
open apple/LogitlyDemo.xcodeproj
```

The Apple target requires macOS and Xcode for compilation and physical-device validation.

## Development

```bash
python -m pip install -e ".[test,benchmark]"
python -m pytest -m "not network and not gpu"
python -m build
```

GPU conformance tests accept an existing local checkpoint:

```bash
LOGITLY_TEST_MODEL=/absolute/path/to/model \
python -m pytest -m gpu
```

## Documentation

- [Library and runtime guide](docs/LIBRARY.md)
- [Logitly and Jev: an open-source path to machine-native decisions](docs/LOGITLY_AND_JEV.md)
- [Apple runtime guide](apple/README.md)

## Scope

`confidence` is the maximum probability in the restricted distribution. It is not automatically a calibrated correctness guarantee. Applications should validate outcomes, measure calibration on representative data, and place deterministic checks around consequential actions.

## License

Logitly is released under the [MIT License](LICENSE). Model checkpoints remain subject to their respective licenses.
