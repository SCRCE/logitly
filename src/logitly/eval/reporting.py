from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def write_summary_csv(path: Path, summaries: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "examples",
        "accuracy",
        "nll",
        "brier",
        "ece",
        "permutation_accuracy",
        "argmax_agreement",
        "mean_probability_delta",
        "p95_probability_delta",
        "max_probability_delta",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            stability = summary["permutation_stability"]
            writer.writerow(
                {
                    "model": summary["model"],
                    "examples": summary["examples"],
                    "accuracy": summary["accuracy"],
                    "nll": summary["nll"],
                    "brier": summary["brier"],
                    "ece": summary["ece"],
                    **stability,
                }
            )


def write_latency_csv(path: Path, rows_by_model: dict[str, Sequence[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "batch_size",
        "status",
        "wall_p50_seconds",
        "wall_p95_seconds",
        "model_p50_seconds",
        "model_p95_seconds",
        "decisions_per_second",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model, rows in rows_by_model.items():
            for row in rows:
                writer.writerow({field: model if field == "model" else row.get(field, "") for field in fields})


def create_plots(output_dir: Path, model: str, metrics: dict[str, Any], latency: Sequence[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import hashlib
    import re
    stem = re.sub(r"[^a-z0-9_-]+", "-", model.lower()).strip("-_")[:80] or "model"
    slug = stem + "-" + hashlib.sha256(model.encode()).hexdigest()[:8]
    bins = [row for row in metrics["calibration_bins"] if row["count"]]
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    axis.plot([row["confidence"] for row in bins], [row["accuracy"] for row in bins], "o-", label=model)
    axis.set(xlabel="Mean confidence", ylabel="Accuracy", title=f"Calibration — {model}", xlim=(0, 1), ylim=(0, 1))
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"{slug}-calibration.png", dpi=160)
    plt.close(figure)

    risk = metrics["risk_coverage"]
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot([row["coverage"] for row in risk], [row["accuracy"] for row in risk], "o-")
    axis.set(xlabel="Coverage", ylabel="Accuracy", title=f"Risk / coverage — {model}", xlim=(0, 1), ylim=(0, 1))
    figure.tight_layout()
    figure.savefig(output_dir / f"{slug}-risk-coverage.png", dpi=160)
    plt.close(figure)

    stability = metrics["permutation_stability"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    drift_names = ["Mean", "p95", "Maximum"]
    drift_values = [
        stability["mean_probability_delta"],
        stability["p95_probability_delta"],
        stability["max_probability_delta"],
    ]
    axes[0].bar(drift_names, drift_values)
    axes[0].set(ylabel="Absolute probability drift", title="Semantic probability drift", ylim=(0, 1))
    agreement_names = ["Argmax agreement", "Permutation accuracy"]
    agreement_values = [stability["argmax_agreement"], stability["permutation_accuracy"]]
    axes[1].bar(agreement_names, agreement_values)
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].set(ylabel="Rate", title="Permutation outcomes", ylim=(0, 1))
    figure.suptitle(f"Permutation stability — {model}")
    figure.tight_layout()
    figure.savefig(output_dir / f"{slug}-permutation-stability.png", dpi=160)
    plt.close(figure)

    completed = [row for row in latency if row["status"] == "ok"]
    if completed:
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.plot([row["batch_size"] for row in completed], [row["decisions_per_second"] for row in completed], "o-")
        axis.set_xscale("log", base=2)
        axis.set(xlabel="Batch size", ylabel="Decisions / second", title=f"Throughput — {model}")
        figure.tight_layout()
        figure.savefig(output_dir / f"{slug}-throughput.png", dpi=160)
        plt.close(figure)


def write_markdown_report(
    path: Path,
    summaries: Sequence[dict[str, Any]],
    latency_by_model: dict[str, Sequence[dict[str, Any]]],
) -> None:
    lines = [
        "# Logitly direct-logit benchmark",
        "",
        "All results use one prefill forward pass and restricted next-token logits. No tokens were generated.",
        "",
        "| Model | Accuracy | NLL | Brier | ECE | Permutation accuracy | Argmax agreement |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        stability = summary["permutation_stability"]
        lines.append(
            f"| {summary['model']} | {summary['accuracy']:.4f} | {summary['nll']:.4f} | "
            f"{summary['brier']:.4f} | {summary['ece']:.4f} | "
            f"{stability['permutation_accuracy']:.4f} | {stability['argmax_agreement']:.4f} |"
        )
    lines.extend(["", "## Throughput", ""])
    for model, rows in latency_by_model.items():
        lines.extend(
            [
                f"### {model}",
                "",
                "| Batch | Status | p50 wall (s) | p95 wall (s) | p50 model (s) | p95 model (s) | Decisions/s | Peak VRAM (GB) |",
                "|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            if row["status"] == "ok":
                memory = row.get("peak_reserved_bytes")
                memory_text = f"{memory / 1e9:.2f}" if memory is not None else "unavailable"
                lines.append(
                    f"| {row['batch_size']} | ok | {row['wall_p50_seconds']:.4f} | "
                    f"{row['wall_p95_seconds']:.4f} | {row['model_p50_seconds']:.4f} | "
                    f"{row['model_p95_seconds']:.4f} | {row['decisions_per_second']:.2f} | "
                    f"{memory_text} |"
                )
            else:
                lines.append(f"| {row['batch_size']} | OOM | — | — | — | — | — | — |")
        lines.append("")

    lines.extend(["## Permutation stability", ""])
    for summary in summaries:
        stability = summary["permutation_stability"]
        lines.extend(
            [
                f"### {summary['model']}",
                "",
                f"- Mean semantic probability drift: {stability['mean_probability_delta']:.4f}",
                f"- p95 semantic probability drift: {stability['p95_probability_delta']:.4f}",
                f"- Maximum semantic probability drift: {stability['max_probability_delta']:.4f}",
                "",
                "| Confidence threshold | Coverage | Accuracy |",
                "|---:|---:|---:|",
            ]
        )
        for row in summary["risk_coverage"]:
            lines.append(f"| {row['threshold']:.2f} | {row['coverage']:.4f} | {row['accuracy']:.4f} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
