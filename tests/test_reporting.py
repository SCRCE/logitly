from __future__ import annotations

import csv
from pathlib import Path

from logitly.eval.reporting import (
    create_plots,
    write_latency_csv,
    write_markdown_report,
)


def test_complete_report_artifacts_are_written(tmp_path: Path) -> None:
    model = "fake model"
    metrics = {
        "model": model,
        "accuracy": 0.75,
        "nll": 0.5,
        "brier": 0.3,
        "ece": 0.1,
        "calibration_bins": [
            {"count": 2, "confidence": 0.75, "accuracy": 0.5},
        ],
        "risk_coverage": [
            {"threshold": 0.5, "selected": 2, "coverage": 1.0, "accuracy": 0.75},
        ],
        "permutation_stability": {
            "permutation_accuracy": 0.7,
            "argmax_agreement": 0.8,
            "mean_probability_delta": 0.05,
            "p95_probability_delta": 0.1,
            "max_probability_delta": 0.2,
        },
    }
    latency = [
        {
            "batch_size": 1,
            "status": "ok",
            "wall_p50_seconds": 0.2,
            "wall_p95_seconds": 0.3,
            "model_p50_seconds": 0.1,
            "model_p95_seconds": 0.2,
            "decisions_per_second": 5.0,
            "peak_allocated_bytes": 100,
            "peak_reserved_bytes": 200,
        }
    ]
    create_plots(tmp_path, model, metrics, latency)
    write_latency_csv(tmp_path / "latency.csv", {model: latency})
    write_markdown_report(tmp_path / "report.md", [metrics], {model: latency})

    assert len(list(tmp_path.glob("*.png"))) == 4
    assert "p50 model" in (tmp_path / "report.md").read_text()
    with (tmp_path / "latency.csv").open(newline="") as handle:
        assert next(csv.DictReader(handle))["model"] == model
