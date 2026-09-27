<p align="center">
  <img src="https://raw.githubusercontent.com/SCRCE/logitly/main/assets/logitly-logo.svg" alt="Logitly" width="560">
</p>

<p align="center">
  Turn compatible causal LLMs into fast, bounded decision engines.
</p>

<p align="center">
  <a href="https://github.com/SCRCE/logitly/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/SCRCE/logitly/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/logitly/"><img alt="PyPI" src="https://img.shields.io/pypi/v/logitly.svg"></a>
  <a href="https://pypi.org/project/logitly/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/logitly.svg"></a>
  <img alt="Typed" src="https://img.shields.io/badge/typing-typed-006F66">
  <a href="https://github.com/SCRCE/logitly/blob/main/LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/license-MIT-B72B5B.svg"></a>
</p>

Logitly turns state, a question, and named choices into a probability
distribution. It performs one prefill forward pass, reads the model's original
next-token logits for fixed labels, and normalizes only the supplied choices.

```bash
pip install "logitly[transformers]"
```

## Quick Start

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

No answer tokens are generated. Logitly does not fine-tune the model, replace
its LM head, or run a completion loop.

## Decision Primitives

Logitly exposes three bounded decision operations:

```python
model.noul(state, question)                  # yes/no probabilities
model.choice(state, question, choices)       # named choice distribution
model.score(state, question, levels)         # ordinal distribution + mean
```

Mixed decisions can be evaluated together:

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

## How It Works

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

For labels `A` through `T`, Logitly gathers the final-position vocabulary
logit for each valid label and computes a softmax over that restricted set.
Semantic IDs such as `review` and `block` are mapped back onto the resulting
probabilities.

Every model profile defines its native chat boundary and verbalizers. Logitly
verifies that each label is a distinct, single-token continuation at the exact
assistant-answer boundary before inference begins.

## Models and Runtimes

| Alias | Pinned checkpoint | Notes |
|---|---|---|
| `lfm` | `LiquidAI/LFM2.5-1.2B-Instruct` | Recommended starting point |
| `qwen` | `RedHatAI/Qwen3.8-27B-INT4` | INT4; thinking disabled |
| `glm` | `mratsim/GLM-4-32B-0414.w4a16-gptq` | W4A16 GPTQ |

Compatible Hugging Face model IDs, local checkpoints, and GGUF files can also
be supplied directly. Unsupported architectures fail explicitly instead of
silently switching inference modes.

Available runtimes:

- **Transformers** — direct final-position logits from causal-LM interfaces.
- **vLLM** — custom pooling adapters using the model's original LM head.
- **llama.cpp** — selected GGUF logits without a completion sampler.

Install only the runtime you need:

```bash
pip install "logitly[transformers,quantized]"
pip install "logitly[vllm]"
pip install "logitly[llama-cpp]"
pip install "logitly[browser]"
```

Transformers and vLLM should use separate environments because their pinned
runtime dependencies differ. Python inference currently requires an NVIDIA
CUDA GPU.

## CLI

```bash
logitly validate lfm
logitly playground lfm --port 8000
logitly benchmark lfm --size 128 --output results/lfm
```

The playground exposes Choice, Noul, and Score interactively. The benchmark
records accuracy, calibration, permutation stability, latency, throughput,
and GPU-memory measurements.

## Demos

```bash
python -m logitly.demos.snake --model lfm --steps 20 --show-board
python -m logitly.demos.browser --help
```

The browser demo converts visible DOM controls into finite operation and
target sets. The model cannot invent selectors, coordinates, JavaScript, or
unsupported actions.

## Apple Runtime

The [`apple/`](apple/) directory contains a Swift package and iPhone demo using
llama.cpp and Metal. It implements the same Choice, Noul, Score, and batch
contract without a sampler or completion loop.

```bash
./apple/scripts/bootstrap.sh
open apple/LogitlyDemo.xcodeproj
```

The Apple target requires macOS and Xcode for compilation and physical-device
validation.

## Documentation

- [Library and runtime guide](docs/LIBRARY.md)
- [Logitly and Jev](docs/LOGITLY_AND_JEV.md)
- [Apple runtime guide](apple/README.md)

## Development

```bash
git clone https://github.com/SCRCE/logitly.git
cd logitly
python -m pip install -e ".[test,benchmark]"
python -m pytest -m "not network and not gpu"
python -m build
```

GPU conformance tests accept an existing local checkpoint through
`LOGITLY_TEST_MODEL`.

## Scope

`confidence` is the maximum probability in the restricted distribution, not
an automatic correctness guarantee. Validate outcomes and calibration on data
representative of your application, and place deterministic checks around
consequential actions.

## License

Logitly is available under the [MIT License](LICENSE). Model checkpoints remain
subject to their respective licenses.
