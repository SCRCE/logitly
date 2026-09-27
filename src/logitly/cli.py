"""CLI parent owns the complete worker process group."""
import argparse
import json
import os
import signal
import subprocess
import sys


def parser():
    root = argparse.ArgumentParser(prog="logitly")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("validate", "playground", "benchmark"):
        sub = commands.add_parser(name)
        sub.add_argument("model")
        sub.add_argument("--runtime", choices=("transformers", "llama_cpp", "vllm"), default="transformers")
        sub.add_argument("--profile")
        sub.add_argument("--revision")
        sub.add_argument("--device", default="cuda:0")
        sub.add_argument("--max-input-tokens", type=int, default=4096)
        if name == "playground":
            sub.add_argument("--host", default="127.0.0.1")
            sub.add_argument("--port", type=int, default=8000)
        if name == "benchmark":
            sub.add_argument("--size", type=int, default=128)
            sub.add_argument("--seed", type=int, default=20260920)
            sub.add_argument("--batch-sizes", default="1,2,4,8,16,32,64,128")
            sub.add_argument("--warmups", type=int, default=5)
            sub.add_argument("--repetitions", type=int, default=20)
            sub.add_argument("--permutations", type=int, default=5)
            sub.add_argument("--output", default="results/logitly")
    return root


def run_worker(args):
    if os.name != "posix":
        raise RuntimeError("CLI process ownership currently requires Linux/WSL")
    from pathlib import Path
    environment = os.environ.copy()
    cache = Path(os.environ.get("LOGITLY_PROJECT_ROOT", Path.cwd())) / ".cache"
    for key, folder in (("HF_HOME", "huggingface"), ("VLLM_CACHE_ROOT", "vllm"),
                        ("TORCH_HOME", "torch"), ("TRITON_CACHE_DIR", "triton"),
                        ("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("MPLCONFIGDIR", "logitly-matplotlib")):
        environment.setdefault(key, str(cache / folder))
    process = subprocess.Popen([sys.executable, "-m", "logitly.worker", json.dumps(vars(args))],
                               start_new_session=True, env=environment)
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        return process.wait()
    except KeyboardInterrupt:
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous)
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # Includes orphaned descendants if the direct worker has already exited.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def main(argv=None):
    return run_worker(parser().parse_args(argv))
