from __future__ import annotations

import subprocess
from pathlib import Path

from logitly.prompts import render_choice_prompt, render_noul_prompt
from logitly.serialization import serialize_state


ROOT = Path(__file__).resolve().parents[1]
APPLE = ROOT / "apple"


def test_swift_golden_fixture_matches_python_contract() -> None:
    assert serialize_state(
        {"z": [2, True, None], "a": {"message": "hello"}}
    ) == '{"a":{"message":"hello"},"z":[2,true,null]}'
    assert render_choice_prompt(
        "case", "Act?", ("A", "B"), ("Approve", "Review")
    ) == "STATE:\ncase\n\nQUESTION:\nAct?\n\nCHOICES:\nA. Approve\nB. Review\n\nReturn exactly one label from:\nA\nB\n\nAnswer:"
    assert render_noul_prompt("case", "Valid?") == (
        "STATE:\ncase\n\nQUESTION:\nValid?\n\nChoose exactly one:\n\nY = Yes\nN = No\n\n"
        "Return exactly one label from:\nY\nN\n\nAnswer:"
    )


def test_apple_runtime_contains_no_generation_or_sampler_calls() -> None:
    sources = "\n".join(path.read_text() for path in (APPLE / "Sources").rglob("*.swift"))
    assert "llama_sampler_" not in sources
    assert "completion_loop" not in sources
    assert ".generate(" not in sources
    assert sources.count("llama_decode(") == 1


def test_model_artifacts_are_ignored_and_not_tracked() -> None:
    ignored = subprocess.run(
        ["git", "check-ignore", "apple/LogitlyDemo/Resources/Models/test.gguf"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert ignored.returncode == 0
    tracked = subprocess.run(
        ["git", "ls-files", "*.gguf", "*.safetensors"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert not tracked.stdout.strip()


def test_apple_pins_are_present() -> None:
    package = (APPLE / "Package.swift").read_text()
    bootstrap = (APPLE / "scripts/bootstrap.sh").read_text()
    assert "b11147" in package
    assert "d04512c7973241323faff78154340064d16be1330bba7dd97dbdffcd12890e06" in package
    assert "8ed288026e23958ad9dfa92d53ed773a8eee7125" in bootstrap
    assert "bb741ebb106d543e9de114b843a3d3d73d51c74b5801e69da2abde821a0cb3e1" in bootstrap
    assert "100000000000" in bootstrap
    assert "90000000000" in bootstrap


def test_native_context_caps_full_vocabulary_output_rows() -> None:
    source = (APPLE / "Sources/Logitly/NativeLlamaRuntime.swift").read_text()
    assert "contextParameters.n_outputs_max = UInt32(configuration.maxBatchSize)" in source
    assert "contextParameters.n_outputs_max_per_seq = 1" in source
