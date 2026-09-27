# Logitly and Jev

## An open-source path to machine-native decisions

Large language models are usually asked to produce text. They generate one token after another, even when the application ultimately needs something much smaller: a yes-or-no judgment, one action from a known set, or a score on a fixed scale.

Jev and Logitly explore a different interface for intelligence:

```text
state + typed question + permitted answers
                    ↓
             decision model
                    ↓
       probabilities software can use
```

**Jev** is TypeSafe AI's purpose-built System One Model for fast, typed, probabilistic decisions inside software. **Logitly** is an MIT-licensed, open-source decision layer that extracts the same broad class of bounded decision from compatible open-weight causal LLMs—locally, with their weights unchanged and without generating answer tokens.

They reach a similar software interface by different routes. Jev introduces a specialized model architecture, sampler, and training method. Logitly turns existing LLMs into decision engines by reading selected values from their original next-token vocabulary logits.

---

## What Jev is

TypeSafe describes Jev as its first public **System One Model**: a model designed for decisions used directly by software rather than prose written for people. Instead of returning an unconstrained string that must be parsed and validated, Jev returns typed values, probability distributions, and confidence information over outputs defined by the caller.

The public Jev interface is centered on three primitives:

- **Noul:** a yes-or-no decision with probabilities.
- **Choice:** one option from a caller-defined set, with a distribution over the choices.
- **Score:** a rating over an ordered scale, with a score, level distribution, and confidence.

TypeSafe's workflow evaluations show how these primitives can be composed with ordinary code: narrow judgments are delegated to the model, deterministic rules stay in the program, and the resulting probabilities drive routing, review thresholds, and automation. This makes the model behave more like an intelligence primitive inside a compute graph than a conversational endpoint.

Jev is purpose-built for this workload. TypeSafe says its stack includes a new architecture, a parallel sampler, and **Reinforcement Learning for Calibrated Decisions (RLCD)**. Its stated goal is not merely to choose the most likely answer, but to return useful uncertainty estimates so applications can decide when to automate and when to escalate. Jev can answer multiple independent typed questions against the same state in parallel, and its Choice interface supports high-cardinality decisions.

Official references:

- [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- [TypeSafe AI](https://typesafe.ai/)
- [TypeSafe workflow evaluations and primitive definitions](https://evals.typesafe.ai/)

---

## What Logitly is

Logitly is an open-source Python library that gives compatible causal LLMs a Jev-style decision interface without retraining them.

The caller supplies:

1. A state: text or deterministic JSON-compatible data.
2. A question.
3. A bounded set of valid answers.

Logitly renders that request at the model's native assistant-answer boundary, performs one prefill forward pass, selects only the logits belonging to fixed one-token labels, and normalizes those selected values with softmax.

```text
state + question + choices
            │
            ▼
      frozen causal LLM
            │
            ▼
   original LM-head logits
            │
            ▼
 select logits for A/B/C/…
            │
            ▼
          softmax
            │
            ▼
 named probability distribution
```

There is no call to `generate()`, no autoregressive answer loop, no chain-of-thought output, no fine-tuning, and no replacement classification head. The model's original weights and LM head remain intact.

For choice labels \(\ell_1,\ldots,\ell_k\), Logitly reads the final-position logit for each label token:

$$
z_i = \operatorname{LM}(x)_{-1,\operatorname{token}(\ell_i)}
$$

It then normalizes only that restricted set:

$$
P(c_i \mid x) = \frac{\exp(z_i)}{\sum_{j=1}^{k}\exp(z_j)}
$$

The full vocabulary is never treated as the answer space. The caller defines the legal semantic outputs; the LLM only scores them.

---

## The Logitly interface

Logitly exposes the same three useful decision shapes:

```python
result = model.noul(
    state={"account": "active", "payment": "late"},
    question="Should this account be reviewed?",
)
# NoulResult(yes=0.87, no=0.13)
```

```python
result = model.choice(
    state={"amount": 9700, "device": "unknown"},
    question="What action should be taken?",
    choices={
        "close": "Close as benign",
        "review": "Request analyst review",
        "block": "Block immediately",
    },
)
# ChoiceResult(
#   probabilities={"close": 0.03, "review": 0.82, "block": 0.15},
#   choice="review",
#   confidence=0.82,
# )
```

```python
result = model.score(
    state="The submitted evidence strongly supports the claim.",
    question="How strong is the evidence?",
    levels=["none", "weak", "moderate", "strong", "conclusive"],
)
# Returns the full level distribution and its probability-weighted expected index.
```

`decide_many()` evaluates mixed decision requests in batches. This is important because decision workloads frequently contain many narrow judgments over the same kind of structured state.

Logitly uses:

- `A`–`T` for Choice labels.
- `Y` and `N` for Noul.
- `0`–`9` for Score levels.

Every profile validates that its labels are distinct, single-token continuations at the model's exact assistant-answer boundary. This protects the experiment from subtle tokenizer and chat-template errors.

---

## Same destination, different route

| | Jev | Logitly |
|---|---|---|
| Core purpose | Machine-native, typed probabilistic decisions | Machine-native, bounded probabilistic decisions |
| Model strategy | Purpose-built System One Model | Decision layer over compatible causal LLMs |
| Training | Specialized RLCD training described by TypeSafe | No training; model weights remain frozen |
| Output | Typed decisions, probabilities, confidence | Typed Python results and restricted-label probabilities |
| Inference shape | Purpose-built parallel decision sampling | One LLM prefill forward pass per decision batch |
| Text generation | Gives up free-form string generation for its decision interface | Generates zero answer tokens in direct mode |
| Deployment | Managed TypeSafe service | Local and self-hosted on user-controlled hardware |
| Model choice | Jev | Pluggable compatible open-weight models |
| Runtime choice | TypeSafe's stack | Transformers, vLLM, and llama.cpp |
| Transparency | Public API and published methodology | Open implementation from prompt serialization through selected logits and softmax |
| Composition | Noul, Choice, Score, parallel questions | Noul, Choice, Score, `decide_many()` batching |

Jev represents the specialized-model approach: build and train a system directly for calibrated decisions. Logitly represents the open-model extraction approach: reuse the representations already present in an instruction-tuned LLM and expose them through a strict decision contract.

That makes Logitly useful as:

- A local, inspectable decision engine.
- A research platform for studying how much decision capability exists before autoregressive decoding.
- A common interface for comparing different LLMs as policies or classifiers.
- A way to prototype Jev-shaped workflows with open weights and familiar inference stacks.
- A foundation for bounded agents whose legal actions are supplied by deterministic software.

---

## What has been built

### A reusable Python library

Logitly is packaged as a Python project with a stable `DecisionModel` API, typed request/result objects, deterministic state serialization, validation, lifecycle management, and CLI commands.

```bash
logitly validate lfm
logitly playground lfm --port 8000
logitly benchmark lfm --size 128 --output results/lfm-run
```

The library supports caller-owned models as well as model loading from pinned or local checkpoints. Unsupported architectures and invalid verbalizers fail explicitly rather than silently falling back to generation.

### Multiple inference runtimes

The same decision contract has been implemented across:

- **Transformers:** direct access to the last-position model logits.
- **vLLM:** custom pooling adapters that retain the model's original LM head while bypassing generation sampling.
- **llama.cpp:** native GGUF prompt evaluation and selected-logit extraction without constructing a completion.

Runtime validation artifacts cover Choice, Noul, Score, mixed batching, repeated calls, token boundaries, and shutdown behavior. Sampling and generation paths are patched to raise during conformance checks.

The conformance suite covers Choice, Noul, Score, mixed batching, repeated calls, token boundaries, and shutdown behavior across the supported runtime interfaces.

### Pluggable model profiles

Profiles isolate model-specific details such as:

- Chat-template rendering.
- The precise assistant-answer boundary.
- Non-thinking template flags.
- Label palettes and token IDs.
- Context limits and compatibility checks.

The project includes profiles for:

- LiquidAI LFM2.5-1.2B-Instruct.
- Qwen3.8-27B INT4 with thinking and preserved thinking disabled.
- GLM-4-32B-0414 GPTQ.
- Generic compatible causal models and GGUFs.

This keeps the public decision API independent from tokenizer and chat-protocol differences.

### A browser policy driven entirely by decisions

Logitly includes a bounded browser agent built on the same `browser-harness` transport used by the public [`browser-use/jev-ultrafast`](https://github.com/browser-use/jev-ultrafast) project.

The browser runtime converts the visible page into a finite action space:

```text
current DOM and goal
        ↓
visible legal operations and targets
        ↓
Logitly operation + target decision heads
        ↓
validated deterministic browser action
        ↓
new page state
```

The model never invents CSS selectors, coordinates, JavaScript, or browser commands. It can only choose an operation and target that the browser harness observed and offered.

The contextual policy adds a deterministic requirement ledger and compact action transcript. It tracks exact entered values, control states, submission requirements, completed navigation, and pending goal conditions. Qwen successfully completed the full hotel workflow:

```text
TYPE_TEXT  Destination = Lisbon
CLICK      Free cancellation
SELECT     Design
CLICK      Find stays
CLICK      View Casa Flora
DONE
```

The independent verifier confirmed the final page contained **Casa Flora**, **Design**, **Free cancellation enabled**, and **Destination Lisbon**. The same policy successfully opened the requested live Wikipedia article. Both runs generated zero answer tokens.

The browser-policy and deterministic-executor behavior are covered by the repository test suite. Local run artifacts remain intentionally excluded from the source repository.

The public Jev browser demonstration uses the same larger system pattern: structured browser observation, bounded operation and target questions, deterministic execution, and a conventional helper only when free-form text must be produced. Its published implementation calls this “a browser agent that chooses instead of generating.”

References:

- [`browser-use/jev-ultrafast`](https://github.com/browser-use/jev-ultrafast)
- [Jev Ultrafast performance notes](https://github.com/browser-use/jev-ultrafast/blob/main/docs/performance.md)

### Decision-driven Snake

The Snake experiments use Logitly as a real-time policy over four legal directions. Every move is one forward pass followed by softmax over the four direction-label logits.

Across three 50-move scored-route trials, Qwen:

- Ate 16 pieces of food.
- Chose the shortest supplied safe-cycle route on 149 of 150 moves.
- Covered an average of 31.7 unique cells.
- Completed the trials without a safety-shield intervention.

The Snake work demonstrates how deterministic planning information and a probabilistic decision model can be composed: code computes hard constraints and useful route features; the model chooses among the bounded actions.

The reproducible Snake implementation is included in `src/logitly/demos/snake.py`.

### Measured local throughput

On an NVIDIA RTX 3090, the pinned LiquidAI LFM2.5-1.2B-Instruct BF16 model reached a measured peak of **139.9 end-to-end decisions per second** at batch size 8. Model-only throughput at that point was **150.9 decisions per second**.

The benchmark used five warmup batches and twenty measured repetitions at each batch size, with zero generated tokens. Results include latency percentiles and peak GPU memory through batch size 128.

The benchmark command can reproduce the throughput sweep and write a fresh result bundle for the installed model and hardware.

### Evaluation and reporting

Logitly includes an evaluation harness for:

- Accuracy.
- Negative log likelihood.
- Multiclass Brier score.
- Expected calibration error.
- Risk/coverage curves.
- Label-permutation stability.
- End-to-end and model-only latency.
- Decisions per second.
- Peak GPU memory.

It emits raw JSONL predictions, JSON and CSV summaries, plots, and a Markdown report. This turns decision extraction into a measurable experiment rather than a visual demo alone.

### Web playground and packaging

The project includes a local playground for interactively testing state, questions, and choices, along with Docker configurations, pinned dependency sets, project-local caches, and explicit storage controls.

Logitly is packaged under the `logitly` distribution and import name with an MIT license. The base package keeps runtime dependencies optional and lazily loaded, allowing users to install only the backend they need.

### Native Apple implementation

An Apple implementation under `apple/` brings the same Choice, Noul, Score, and batching contract to Swift. It uses llama.cpp and Metal with a pinned LiquidAI GGUF model and contains no sampler or completion loop.

The iPhone demo includes:

- Choice, Noul, and Score playgrounds.
- Local benchmarking.
- A Snake demonstration.
- Deterministic prompt/state fixtures shared with Python.
- Offline bundled-model operation.

This extends the Logitly idea beyond a Python/CUDA experiment toward an embeddable, on-device decision runtime.

---

## Why bounded decisions matter

Many software problems are naturally closed-world at the moment a decision is made:

- Which queue should receive this case?
- Is this transaction suspicious?
- Which visible browser control should be used next?
- Which safe move should a game agent choose?
- How strongly does this evidence support a claim?
- Should the system automate, review, or escalate?

Traditional LLM integration often asks a generative model to describe or serialize the answer and then adds parsing, retries, and validation around the response. A decision interface moves the contract into the input: the application defines what is legal before inference.

```text
unbounded generation
state → reasoning text → answer text → parse → validate → action

bounded decision
state → permitted choices → probabilities → validated action
```

This changes the role of the surrounding software. The harness performs deterministic work—constructing the legal action space, maintaining state, checking preconditions, and verifying outcomes—while the model handles the fuzzy judgment that remains.

Jev was designed specifically around this philosophy. Logitly makes the philosophy available as an open, local, inspectable layer over existing models.

---

## The central idea

Logitly began with a simple hypothesis:

> A meaningful amount of an instruction-tuned LLM's decision-making ability may already be available in the representation at the end of the prompt, before it generates a single answer token.

The project turns that hypothesis into a working library and a collection of measurable systems.

It demonstrates that open-weight LLMs can be used as:

- Typed decision functions.
- Local probabilistic policies.
- Batched classifiers with caller-defined semantics.
- Bounded browser controllers.
- Real-time game decision engines.
- Portable on-device decision components.

Jev shows what becomes possible when decision-making is treated as a first-class model capability. Logitly opens that design space to existing LLMs, local hardware, multiple inference runtimes, and direct experimentation.

**Jev is the purpose-built System One model. Logitly is the open-source decision layer for the models developers can run themselves.**
