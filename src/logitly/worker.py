import json
import signal
import sys
from pathlib import Path
from types import SimpleNamespace


def run(args):
    from . import DecisionModel, ChoiceRequest
    items = None
    if args.command == "benchmark":
        from .eval.dataset import load_benchmark_items
        sizes = tuple(int(value) for value in args.batch_sizes.split(","))
        if args.size < max(sizes) or args.warmups < 0 or args.repetitions < 1:
            raise ValueError("Invalid workload size or repetitions")
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        items = load_benchmark_items(size=args.size, seed=args.seed)
    options = {"runtime": args.runtime, "profile": args.profile,
               "device": args.device, "max_input_tokens": args.max_input_tokens}
    if args.revision:
        options["revision"] = args.revision
    with DecisionModel.from_pretrained(args.model, **options) as engine:
        if args.command == "validate":
            print(json.dumps(engine.validate(), indent=2))
        elif args.command == "playground":
            from .server import serve_playground
            serve_playground(args.model, host=args.host, port=args.port, engine=engine)
        else:
            from .benchmark import throughput, _save
            from .eval.benchmark import _predict, _dataset_manifest
            from .eval.metrics import compute_model_metrics
            from .eval.reporting import write_jsonl, write_summary_csv, write_latency_csv, create_plots, write_markdown_report
            requests = [ChoiceRequest({"domain": item.category}, item.question,
                                      {str(i): text for i, text in enumerate(item.options)}, id=str(item.question_id)) for item in items]
            _save(output / "dataset.json", _dataset_manifest(items, args.size, args.permutations, args.seed))
            measured = throughput(engine, requests, batch_sizes=sizes, warmups=args.warmups,
                                  repetitions=args.repetitions, output=output / "throughput.json")
            completed = [row for row in measured["rows"] if row["status"] == "ok"]
            if not completed:
                raise RuntimeError("No benchmark batch completed")
            best = max(completed, key=lambda row: row["decisions_per_second"])["batch_size"]
            predictions = _predict(engine, items, permutations=args.permutations, seed=args.seed, batch_size=best)
            write_jsonl(output / "predictions.jsonl", predictions)
            metrics = compute_model_metrics(predictions)
            metrics["model"] = engine.backend.model_name
            _save(output / "metrics.json", metrics)
            write_summary_csv(output / "summary.csv", [metrics])
            write_latency_csv(output / "latency.csv", {metrics["model"]: measured["rows"]})
            create_plots(output, metrics["model"], metrics, measured["rows"])
            write_markdown_report(output / "report.md", [metrics], {metrics["model"]: measured["rows"]})


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        run(SimpleNamespace(**json.loads(sys.argv[1])))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
