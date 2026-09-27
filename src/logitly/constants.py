from __future__ import annotations

from dataclasses import dataclass

CHOICE_LABELS = tuple("ABCDEFGHIJKLMNOPQRST")
NOUL_LABELS = ("Y", "N")
SCORE_LABELS = tuple(str(value) for value in range(10))

MMLU_PRO_REPO = "TIGER-Lab/MMLU-Pro"
MMLU_PRO_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    revision: str
    expected_bytes: int
    display_name: str


MODEL_SPECS = {
    "lfm": ModelSpec(
        key="lfm",
        repo_id="LiquidAI/LFM2.5-1.2B-Instruct",
        revision="0f604ada3f766f9f257460c4c9f0b5d6f69d431b",
        expected_bytes=2_340_697_936,
        display_name="LiquidAI LFM2.5-1.2B Instruct BF16",
    ),
    "glm": ModelSpec(
        key="glm",
        repo_id="mratsim/GLM-4-32B-0414.w4a16-gptq",
        revision="7649a2140aa2dd535d23c8794be7662f23a7aee8",
        expected_bytes=19_698_695_535,
        display_name="GLM-4-32B-0414 W4A16 GPTQ",
    ),
    "qwen": ModelSpec(
        key="qwen",
        repo_id="RedHatAI/Qwen3.8-27B-INT4",
        revision="779ba0009b8c9ceaa6b110f4573a5a50cac12026",
        expected_bytes=19_472_989_617,
        display_name="Qwen3.8-27B INT4 non-thinking",
    ),
}

DEFAULT_SEED = 20_260_920
DEFAULT_DATASET_SIZE = 1_000
DEFAULT_PERMUTATIONS = 5
DEFAULT_MAX_INPUT_TOKENS = 4_096
MAX_STATE_CHARS = 131_072
MAX_QUESTION_CHARS = 16_384
MAX_CRITERION_CHARS = 16_384
