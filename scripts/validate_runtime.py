"""Small real-GPU conformance run. Writes evidence, not throughput claims."""
import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from logitly import DecisionModel, ChoiceRequest, NoulRequest, ScoreRequest
from logitly.benchmark import memory_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--runtime", default="transformers")
    parser.add_argument("--profile", default="generic")
    parser.add_argument("--dtype", default="float32", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # vLLM imports this diagnostic in its spawned worker via the plugin hook.
    os.environ["LOGITLY_ASSERT_NO_SAMPLING"] = "1"
    if args.runtime == "llama_cpp":
        from llama_cpp import llama_cpp as native
        def forbidden(*a, **kw):
            raise AssertionError("Native sampler was invoked")
        native.llama_sampler_sample = forbidden
    else:
        from transformers.generation.utils import GenerationMixin
        def forbidden(*a, **kw):
            raise AssertionError("generate() was invoked")
        GenerationMixin.generate = forbidden
    requests = [ChoiceRequest("Sky is blue.", "What color?", ["blue", "red"]),
                NoulRequest("2+2=4.", "Is this correct?"),
                ScoreRequest("Useful.", "Rate usefulness.", ["low", "high"])]
    before = memory_snapshot()
    options = {} if args.runtime == "llama_cpp" else {"dtype": args.dtype}
    with DecisionModel.from_pretrained(args.model, runtime=args.runtime, profile=args.profile, max_input_tokens=512, **options) as engine:
        report = engine.validate()
        report["validation_dtype"] = "GGUF Q8_0" if args.runtime == "llama_cpp" else args.dtype
        single = [asdict(engine.decide_many([request])[0]) for request in requests]
        batched = [asdict(result) for result in engine.decide_many(requests, batch_size=3)]
        def values(row):
            if "probabilities" in row:
                return list(row["probabilities"].values())
            if "distribution" in row:
                return row["distribution"]
            return [row["yes"], row["no"]]
        drift = max(abs(x - y) for a, b in zip(single, batched) for x, y in zip(values(a), values(b)))
        assert drift < .03, f"single/batch probability difference: {drift}"
        warm = memory_snapshot()
        for _ in range(10):
            engine.decide_many(requests, batch_size=3)
        report.update(single=single, batched=batched, maximum_batch_drift=drift,
                      host_before=before, host_after_warmup=warm, host_after_repetitions=memory_snapshot())
        runtime = engine.backend.runtime
        prepared = [engine._prepare(r) for r in requests]
        prefixes = [engine.profile.render(runtime.tokenizer, d.prompt) for d in prepared]
        tokens = [runtime.encode(p) for p in prefixes]
        labels = [[runtime.encode(p + label)[-1] for label in d.labels] for p, d in zip(prefixes, prepared)]
        report["prompt_token_ids"] = tokens
        report["selected_logits"] = runtime.restricted_logits(tokens, labels)
    report["host_after_close"] = memory_snapshot()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("prompt_token_ids", "labels")}, indent=2))


if __name__ == "__main__":
    main()
