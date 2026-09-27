"""Throughput using the same complete workload at every batch size."""
import importlib.metadata
import json
import math
import statistics
import time
from pathlib import Path


def memory_snapshot():
    result = {}
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                name, amount, _ = line.split()
                result[name.rstrip(":") + "_bytes"] = int(amount) * 1024
    for name in ("memory.current", "memory.peak"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            result["cgroup_" + name.replace(".", "_") + "_bytes"] = int(path.read_text())
    try:
        import psutil
        parent = psutil.Process()
        tree = [parent, *parent.children(recursive=True)]
        try:
            result["process_tree_rss_sum_bytes"] = sum(p.memory_info().rss for p in tree if p.is_running())
        except psutil.Error:
            pass  # A runtime child may exit between enumeration and measurement.
    except ImportError:
        pass
    except (ProcessLookupError, PermissionError):
        pass
    return result


def _save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def throughput(engine, requests, *, batch_sizes=(1, 2, 4, 8, 16, 32, 64, 128),
               warmups=5, repetitions=20, output=None):
    if not requests or warmups < 0 or repetitions < 1:
        raise ValueError("Provide requests, non-negative warmups, and positive repetitions")
    if not batch_sizes or any(b < 1 for b in batch_sizes) or len(set(batch_sizes)) != len(batch_sizes):
        raise ValueError("batch sizes must be distinct positive integers")
    if max(batch_sizes) > len(requests):
        raise ValueError("workload must contain at least the largest batch size")
    runtime = engine.backend.runtime
    lengths = [len(runtime.encode(engine.profile.render(runtime.tokenizer, engine._prepare(r).prompt))) for r in requests]
    payload = {
        **engine.validate(), "warmups": warmups, "repetitions": repetitions,
        "workload_size": len(requests), "prompt_token_lengths": lengths,
        "workload_policy": "same ordered requests, once per repetition, at every batch size",
        "timing": "wall includes rendering, boundary validation, tokenization and output; runtime timing excludes those",
        "runtime_timing_scope": "synchronous runtime call; vLLM includes scheduling/IPC",
        "host_measurement_notes": "RSS sum includes runtime children and can double-count shared pages; cgroup values include everything in that cgroup, not necessarily only this run",
        "prefix_reuse": False, "rows": [], "packages": {},
    }
    for name in ("torch", "transformers", "vllm", "llama-cpp-python"):
        try:
            payload["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    if output:
        _save(output, payload)
    for size in batch_sizes:
        chunks = [requests[i:i + size] for i in range(0, len(requests), size)]
        print(f"batch {size}: {len(chunks)} batches per repetition", flush=True)
        try:
            for i in range(warmups):
                engine.decide_many(chunks[i % len(chunks)], batch_size=size)
            runtime.reset_peak_memory()
            walls, forwards = [], []
            host_start = memory_snapshot()
            host_peak = dict(host_start)
            for _ in range(repetitions):
                for chunk in chunks:
                    started = time.perf_counter()
                    engine.decide_many(chunk, batch_size=size)
                    walls.append(time.perf_counter() - started)
                    forwards.append(engine.backend.last_forward_seconds)
                    for key, value in memory_snapshot().items():
                        host_peak[key] = max(host_peak.get(key, 0), value)
            count = len(requests) * repetitions
            allocated, reserved = runtime.peak_memory_bytes()
            row = {"batch_size": size, "status": "ok", "measured_decisions": count,
                   "measured_batches": len(walls), "decisions_per_second": count / sum(walls),
                   "runtime_decisions_per_second": count / sum(forwards),
                   "wall_p50_seconds": statistics.median(walls), "wall_p95_seconds": _percentile(walls, .95),
                   "model_p50_seconds": statistics.median(forwards), "model_p95_seconds": _percentile(forwards, .95),
                   "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
                   "host_start": host_start, "host_peak": host_peak, "host_end": memory_snapshot(),
                   "padded_input_tokens_per_repetition": sum(max(lengths[i:i + size]) * len(lengths[i:i + size]) for i in range(0, len(lengths), size))}
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            row = {"batch_size": size, "status": "oom", "error": str(error)}
        payload["rows"].append(row)
        if output:
            _save(output, payload)
        print(json.dumps(row), flush=True)
        if row["status"] == "oom":
            break
    return payload


def _percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)
