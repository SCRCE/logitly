<p align="center">
  <img src="https://raw.githubusercontent.com/SCRCE/logitly/main/assets/logitly-logo-pypi.svg" alt="Logitly" width="560">
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

```python
model.noul(state, question)                  # yes/no probabilities
model.choice(state, question, choices)       # named choice distribution
model.score(state, question, levels)         # ordinal distribution + mean
```

`model.decide_many(requests, batch_size=...)` evaluates mixed decisions in a
batch while preserving the same result schemas.

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

Logitly gathers the final-position vocabulary logit for each valid label and
computes a softmax over only that restricted set. Every model profile verifies
that its labels are distinct, single-token continuations at the exact native
assistant-answer boundary before inference begins.

## Models and Runtimes

| Alias | Pinned checkpoint | Notes |
|---|---|---|
| `lfm` | `LiquidAI/LFM2.5-1.2B-Instruct` | Recommended starting point |
| `qwen` | `RedHatAI/Qwen3.8-27B-INT4` | INT4; thinking disabled |
| `glm` | `mratsim/GLM-4-32B-0414.w4a16-gptq` | W4A16 GPTQ |

Compatible Hugging Face model IDs and local checkpoints are also accepted.
The Python package provides Transformers, vLLM, and llama.cpp runtimes behind
the same API.

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

## Documentation

- [Source and full guide](https://github.com/SCRCE/logitly)
- [Library and runtime guide](https://github.com/SCRCE/logitly/blob/main/docs/LIBRARY.md)
- [Apple runtime guide](https://github.com/SCRCE/logitly/blob/main/apple/README.md)

## Scope

`confidence` is the maximum probability in the restricted distribution, not
an automatic correctness guarantee. Validate outcomes and calibration on data
representative of your application.

## License

Logitly is available under the [MIT License](https://github.com/SCRCE/logitly/blob/main/LICENSE).
