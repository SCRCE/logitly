# Logitly

Logitly turns a compatible LLM into a decision model. Give it a state, a question, and named choices; it returns a probability distribution using one forward pass and the model's existing LM-head logits. The model weights stay unchanged, and no answer tokens are generated.

## Install

Python 3.11–3.13 and an NVIDIA CUDA GPU are required for inference.

```bash
python -m pip install "logitly[transformers]"
```

For supported quantized checkpoints, also install `logitly[quantized]`. The `vllm` and `llama-cpp` extras are available for their respective GPU runtimes; use separate environments when their dependency versions differ.

## Example

```python
from logitly import DecisionModel

with DecisionModel.from_pretrained("lfm") as model:
    result = model.choice(
        state={"amount": 9700, "device": "unknown"},
        question="What action should we take?",
        choices={
            "close": "Close as benign",
            "review": "Request analyst review",
            "block": "Block immediately",
        },
    )

print(result.choice)
print(result.probabilities)
```

The `lfm`, `glm`, and `qwen` aliases select pinned checkpoints. Compatible Hugging Face model IDs and local checkpoints are also accepted. `model.noul(state, question)` returns yes/no probabilities; `model.score(state, question, levels)` returns an ordered distribution and expected score; `model.decide_many(requests, batch_size=...)` processes batches.

Logitly selects single-token label logits after the assistant answer boundary and normalizes only those logits. It does not use `generate()`, change model weights, or add a new prediction head. Returned confidence is the maximum choice probability and is not a calibrated correctness guarantee.

The CLI provides `logitly validate`, `logitly playground`, and `logitly benchmark`.
