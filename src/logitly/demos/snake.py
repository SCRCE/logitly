"""A GPU-only, direct-logit Snake trial using a Logitly-compatible model.

Run with ``python -m logitly.demos.snake --model glm --steps 20``. The model chooses a
direction on every turn; a deterministic shield prevents illegal moves.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

from logitly import DecisionModel


DIRECTIONS = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0)}
PROMPT_VARIANTS = (
    "baseline", "grid", "goal", "grid_goal", "grid_history", "scored_moves",
    "compact_scores", "ranked_moves", "best_flag", "planner_direction",
)
BASE_QUESTION = "Which direction should the snake move next to reach food while staying alive? Choose one direction."
GOAL_QUESTION = (
    "Choose one safe move toward the food. Prefer a safe move that reduces the head's Manhattan distance "
    "to the food; if none does, choose a safe move that avoids circling in a short loop."
)
HISTORY_QUESTION = (
    "Choose a safe move toward the food. Prefer a destination not in the recent head positions; "
    "among safe unvisited destinations, move closer to food. If all are recently visited, still choose a safe move."
)
SCORED_QUESTION = (
    "Choose only a SAFE option. For each safe option, the listed cycle steps to food are the remaining "
    "steps on a collision-safe route after that move. Choose the safe option with the SMALLEST listed number; "
    "break ties by avoiding a recently visited destination."
)
COMPACT_SCORES_QUESTION = (
    "Use only the option fields. Select a choice with SAFE=1 and the smallest COST integer. "
    "SAFE=0 is forbidden. If safe COST values tie, prefer NEW=1."
)
RANKED_MOVES_QUESTION = (
    "Use only the option fields. Select the single choice with PRIORITY=0. "
    "Never select PRIORITY=99."
)
BEST_FLAG_QUESTION = (
    "Use only the option fields. Select the single choice marked BEST=YES. "
    "Every BEST=NO choice must be rejected."
)
MODEL_DEFAULTS = {
    "lfm": {"runtime": "transformers", "profile": "lfm", "display": "LiquidAI LFM2.5-1.2B"},
    "glm": {"runtime": "vllm", "profile": "glm", "display": "GLM-4-32B-0414 GPTQ"},
    "qwen": {"runtime": "vllm", "profile": "qwen", "display": "Qwen3.8-27B INT4 non-thinking"},
}


def hamiltonian_cycle(width: int, height: int) -> list[tuple[int, int]]:
    """Return a simple cycle on an even-height rectangular grid."""
    if width < 4 or height < 4 or height % 2:
        raise ValueError("Snake requires width >= 4 and even height >= 4")
    cells = [(x, 0) for x in range(width)]
    for y in range(1, height):
        xs = range(width - 1, 0, -1) if y % 2 else range(1, width)
        cells.extend((x, y) for x in xs)
    cells.extend((0, y) for y in range(height - 1, 0, -1))
    return cells


class SnakeGame:
    def __init__(self, width: int = 8, height: int = 6, seed: int = 20260922):
        self.width, self.height = width, height
        self.cycle = hamiltonian_cycle(width, height)
        self.position = {cell: index for index, cell in enumerate(self.cycle)}
        self.body = [self.cycle[i] for i in (3, 2, 1)]
        self.rng = random.Random(seed)
        self.food = self._new_food()
        self.steps = 0
        self.food_eaten = 0
        self.head_history = [self.body[0]]

    def _new_food(self):
        free = [cell for cell in self.cycle if cell not in self.body]
        return self.rng.choice(free) if free else None

    def candidate(self, direction: str) -> tuple[int, int]:
        dx, dy = DIRECTIONS[direction]
        x, y = self.body[0]
        return x + dx, y + dy

    def safe_moves(self) -> dict[str, bool]:
        head_index = self.position[self.body[0]]
        tail_gap = (self.position[self.body[-1]] - head_index) % len(self.cycle)
        result = {}
        for direction in DIRECTIONS:
            cell = self.candidate(direction)
            if cell not in self.position:
                result[direction] = False
                continue
            occupied = cell in self.body[:-1] or (cell == self.body[-1] and cell == self.food)
            advance = (self.position[cell] - head_index) % len(self.cycle)
            result[direction] = not occupied and 0 < advance < tail_gap
        return result

    def board(self) -> str:
        body = set(self.body)
        lines = []
        for y in range(self.height):
            chars = []
            for x in range(self.width):
                cell = (x, y)
                chars.append("H" if cell == self.body[0] else "o" if cell in body else "*" if cell == self.food else ".")
            lines.append("".join(chars))
        return "\n".join(lines)

    def state(self) -> dict:
        return {
            "board": self.board(),
            "legend": "H=head, o=body, *=food, .=empty",
            "head": list(self.body[0]),
            "food": list(self.food) if self.food else None,
            "body": [list(cell) for cell in self.body],
            "coordinates": "(0,0) is top-left; x increases right; y increases down",
        }

    def choices(self) -> dict[str, str]:
        safety = self.safe_moves()
        return {
            direction: f"Move {direction.lower()}; {'safe' if safety[direction] else 'unsafe'}"
            for direction in DIRECTIONS
        }

    def step(self, direction: str) -> bool:
        if not self.safe_moves().get(direction, False):
            raise ValueError(f"unsafe move: {direction}")
        cell = self.candidate(direction)
        ate = cell == self.food
        self.body.insert(0, cell)
        if ate:
            self.food_eaten += 1
            self.food = self._new_food()
        else:
            self.body.pop()
        self.steps += 1
        self.head_history.append(cell)
        return ate


def visual_state(game: SnakeGame, *, include_history: bool = False) -> str:
    """The same state facts as the JSON state, but with visible grid rows."""
    columns = "".join(str(x % 10) for x in range(game.width))
    rows = "\n".join(f"{y:>2}  {row}" for y, row in enumerate(game.board().splitlines()))
    rendered = (
        f"BOARD: {game.width} columns x {game.height} rows. (0,0) is top-left; x right, y down.\n"
        f"    {columns}\n{rows}\n"
        "Legend: H=head, o=body, *=food, .=empty.\n"
        f"Head: {game.body[0]}; food: {game.food}; body head-to-tail: {game.body}."
    )
    if include_history:
        rendered += f"\nRecent head positions (oldest to newest): {game.head_history[-8:]}"
    return rendered


def decision_inputs(game: SnakeGame, variant: str):
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown prompt variant: {variant}")
    state = game.state() if variant in ("baseline", "goal") else visual_state(
        game, include_history=variant in (
            "grid_history", "scored_moves", "compact_scores", "ranked_moves",
            "best_flag", "planner_direction",
        )
    )
    question = {
        "baseline": BASE_QUESTION,
        "grid": BASE_QUESTION,
        "goal": GOAL_QUESTION,
        "grid_goal": GOAL_QUESTION,
        "grid_history": HISTORY_QUESTION,
        "scored_moves": SCORED_QUESTION,
        "compact_scores": COMPACT_SCORES_QUESTION,
        "ranked_moves": RANKED_MOVES_QUESTION,
        "best_flag": BEST_FLAG_QUESTION,
        "planner_direction": "Select the direction recommended by the navigation planner.",
    }[variant]
    choices = game.choices()
    if variant in (
        "scored_moves", "compact_scores", "ranked_moves", "best_flag", "planner_direction"
    ):
        safety = game.safe_moves()
        recent = set(game.head_history[-8:])
        costs = {
            direction: (game.position[game.food] - game.position[game.candidate(direction)]) % len(game.cycle)
            for direction in DIRECTIONS if safety[direction]
        }
        ranked = sorted(
            costs,
            key=lambda direction: (
                costs[direction],
                game.candidate(direction) in recent,
                tuple(DIRECTIONS).index(direction),
            ),
        )
        priorities = {direction: rank for rank, direction in enumerate(ranked)}
        best = ranked[0]
        if variant == "planner_direction":
            question = (
                f"The navigation planner recommends DIRECTION={best}. "
                "Select the choice with that exact DIRECTION value."
            )
        choices = {}
        for direction in DIRECTIONS:
            cell = game.candidate(direction)
            if variant == "scored_moves":
                if not safety[direction]:
                    choices[direction] = f"Move {direction.lower()} to {cell}; UNSAFE"
                    continue
                remaining = costs[direction]
                revisit = "yes" if cell in recent else "no"
                choices[direction] = (
                    f"Move {direction.lower()} to {cell}; SAFE; cycle steps to food: {remaining}; "
                    f"recently visited: {revisit}"
                )
            elif variant == "compact_scores":
                choices[direction] = (
                    f"DIRECTION={direction} SAFE={int(safety[direction])} "
                    f"COST={costs.get(direction, 999)} NEW={int(cell not in recent)}"
                )
            elif variant == "ranked_moves":
                choices[direction] = (
                    f"DIRECTION={direction} PRIORITY={priorities.get(direction, 99)} "
                    f"SAFE={int(safety[direction])}"
                )
            elif variant == "best_flag":
                choices[direction] = (
                    f"DIRECTION={direction} BEST={'YES' if direction == best else 'NO'} "
                    f"SAFE={int(safety[direction])}"
                )
            else:
                choices[direction] = f"DIRECTION={direction}"
    return state, question, choices


def choose_move(model, game: SnakeGame, variant: str = "baseline") -> dict:
    safety = game.safe_moves()
    if not any(safety.values()):
        raise RuntimeError("no safe move")
    state, question, choices = decision_inputs(game, variant)
    head_before, food_before = game.body[0], game.food
    started = time.perf_counter()
    result = model.choice(state, question, choices)
    elapsed = time.perf_counter() - started
    probabilities = dict(result.probabilities)
    if set(probabilities) != set(DIRECTIONS):
        raise RuntimeError("model returned the wrong direction labels")
    raw = result.choice
    executed = raw if safety[raw] else max(
        (direction for direction in DIRECTIONS if safety[direction]),
        key=lambda direction: probabilities[direction],
    )
    ate = game.step(executed)
    return {
        "step": game.steps,
        "variant": variant,
        "head_before": head_before,
        "head_after": game.body[0],
        "food_before": food_before,
        "raw": raw,
        "executed": executed,
        "shielded": raw != executed,
        "probabilities": probabilities,
        "safe": safety,
        "ate": ate,
        "food_eaten": game.food_eaten,
        "elapsed_seconds": elapsed,
        "model_seconds": getattr(getattr(model, "backend", None), "last_forward_seconds", None),
    }


def compare_prompts(model, *, seeds: list[int], width: int, height: int, steps: int,
                    variants=PROMPT_VARIANTS, output=None) -> dict:
    trials = []
    for variant in variants:
        for seed in seeds:
            game = SnakeGame(width, height, seed)
            heads = [game.body[0]]
            records = []
            for _ in range(steps):
                if game.food is None:
                    break
                record = choose_move(model, game, variant)
                record["seed"] = seed
                records.append(record)
                heads.append(game.body[0])
                if output:
                    output.write(json.dumps(record, sort_keys=True) + "\n")
                    output.flush()
            elapsed = sum(record["elapsed_seconds"] for record in records)
            trial = {
                "variant": variant,
                "seed": seed,
                "decisions": len(records),
                "food_eaten": game.food_eaten,
                "unique_head_cells": len(set(heads)),
                "four_step_returns": sum(heads[i] == heads[i - 4] for i in range(4, len(heads))),
                "shield_interventions": sum(record["shielded"] for record in records),
                "elapsed_seconds": elapsed,
            }
            trials.append(trial)
            print("TRIAL " + json.dumps(trial, sort_keys=True), flush=True)
    aggregates = []
    for variant in variants:
        group = [trial for trial in trials if trial["variant"] == variant]
        aggregates.append({
            "variant": variant,
            "seeds": len(group),
            "decisions": sum(trial["decisions"] for trial in group),
            "food_eaten": sum(trial["food_eaten"] for trial in group),
            "mean_unique_head_cells": sum(trial["unique_head_cells"] for trial in group) / len(group),
            "four_step_returns": sum(trial["four_step_returns"] for trial in group),
            "shield_interventions": sum(trial["shield_interventions"] for trial in group),
            "decisions_per_second": sum(trial["decisions"] for trial in group) / sum(trial["elapsed_seconds"] for trial in group),
        })
    return {"steps_per_trial": steps, "seeds": seeds, "variants": list(variants),
            "trials": trials, "aggregates": aggregates,
            "generated_tokens": 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="lfm", help="model alias, Hugging Face ID, or local checkpoint path")
    parser.add_argument("--runtime", choices=("transformers", "vllm"), help="inference runtime")
    parser.add_argument("--profile", help="explicit Logitly model profile")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--height", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--variant", choices=PROMPT_VARIANTS, default="baseline",
                        help="prompt variant for a single Snake run")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--report", type=Path, help="write a JSON summary for --compare-prompts")
    parser.add_argument("--show-board", action="store_true")
    parser.add_argument("--compare-prompts", action="store_true")
    parser.add_argument("--seeds", default="20260922,20260923,20260924",
                        help="comma-separated seeds for --compare-prompts")
    parser.add_argument("--variants", default=",".join(PROMPT_VARIANTS),
                        help="comma-separated prompt variants for --compare-prompts")
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 1000:
        parser.error("--steps must be between 1 and 1000")
    if args.compare_prompts and args.show_board:
        parser.error("--show-board is unavailable with --compare-prompts")
    try:
        seeds = [int(item) for item in args.seeds.split(",")]
    except ValueError:
        parser.error("--seeds must be comma-separated integers")
    if not 1 <= len(seeds) <= 20 or len(set(seeds)) != len(seeds):
        parser.error("--seeds must contain 1–20 unique integers")
    variants = args.variants.split(",")
    if not variants or len(set(variants)) != len(variants) or any(item not in PROMPT_VARIANTS for item in variants):
        parser.error(f"--variants must be unique names from: {', '.join(PROMPT_VARIANTS)}")
    game = SnakeGame(args.width, args.height, args.seed)
    defaults = MODEL_DEFAULTS.get(args.model, {})
    runtime = args.runtime or defaults.get("runtime", "transformers")
    profile = args.profile or defaults.get("profile")
    display_name = defaults.get("display", args.model)

    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
    output_context = args.record.open("w", encoding="utf-8") if args.record else nullcontext(None)
    records = []
    with output_context as output:
        print(f"Loading {display_name} with {runtime} on CUDA.", flush=True)
        with DecisionModel.from_pretrained(args.model, runtime=runtime, profile=profile,
                                           device="cuda:0", max_input_tokens=1024) as model:
            if args.compare_prompts:
                report = compare_prompts(model, seeds=seeds, width=args.width, height=args.height,
                                         steps=args.steps, variants=variants, output=output)
                if args.report:
                    args.report.parent.mkdir(parents=True, exist_ok=True)
                    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                print("PROMPT_COMPARISON " + json.dumps(report["aggregates"], sort_keys=True), flush=True)
                return 0
            for _ in range(args.steps):
                if game.food is None:
                    break
                if args.show_board:
                    print(game.board(), flush=True)
                record = choose_move(model, game, args.variant)
                records.append(record)
                if output:
                    output.write(json.dumps(record, sort_keys=True) + "\n")
                    output.flush()
                probs = " ".join(f"{key}={value:.3f}" for key, value in record["probabilities"].items())
                intervention = " [shielded]" if record["shielded"] else ""
                print(f"{record['step']:3d}  {probs}  raw={record['raw']} move={record['executed']}{intervention}  {record['elapsed_seconds']:.2f}s", flush=True)
            peak_allocated, peak_reserved = model.backend.peak_memory_bytes()
    elapsed = sum(record["elapsed_seconds"] for record in records)
    summary = {
        "model": args.model,
        "variant": args.variant,
        "decisions": len(records), "food_eaten": game.food_eaten,
        "shield_interventions": sum(record["shielded"] for record in records),
        "decisions_per_second": len(records) / elapsed if elapsed else 0,
        "peak_gpu_allocated_gb": round(peak_allocated / 1e9, 2) if peak_allocated is not None else None,
        "peak_gpu_reserved_gb": round(peak_reserved / 1e9, 2) if peak_reserved is not None else None,
        "generated_tokens": 0,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
