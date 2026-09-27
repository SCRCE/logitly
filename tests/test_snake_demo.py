from types import SimpleNamespace
from io import StringIO

import pytest

from logitly.demos.snake import (
    BASE_QUESTION, DIRECTIONS, GOAL_QUESTION, PROMPT_VARIANTS, SnakeGame,
    choose_move, compare_prompts, decision_inputs, hamiltonian_cycle,
)


def test_cycle_covers_board_with_adjacent_edges():
    cycle = hamiltonian_cycle(8, 6)
    assert len(cycle) == len(set(cycle)) == 48
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(cycle, cycle[1:] + cycle[:1]))
    with pytest.raises(ValueError):
        hamiltonian_cycle(8, 5)


def test_every_move_uses_one_choice_call_and_shields_unsafe_winner():
    game = SnakeGame(seed=7)

    class FakeModel:
        calls = 0

        def choice(self, state, question, choices):
            self.calls += 1
            assert set(choices) == set(DIRECTIONS)
            assert state["head"] == list(game.body[0])
            unsafe = next(direction for direction, safe in game.safe_moves().items() if not safe)
            values = {direction: (0.7 if direction == unsafe else 0.1) for direction in DIRECTIONS}
            return SimpleNamespace(choice=unsafe, probabilities=values)

    model = FakeModel()
    for _ in range(12):
        record = choose_move(model, game)
        assert record["shielded"]
        assert record["safe"][record["executed"]]
    assert model.calls == game.steps == 12


def test_prompt_variants_isolate_grid_format_and_question():
    game = SnakeGame(seed=7)
    base_state, base_question, base_choices = decision_inputs(game, "baseline")
    grid_state, grid_question, grid_choices = decision_inputs(game, "grid")
    goal_state, goal_question, goal_choices = decision_inputs(game, "goal")
    both_state, both_question, both_choices = decision_inputs(game, "grid_goal")
    assert isinstance(base_state, dict) and goal_state == base_state
    assert isinstance(grid_state, str) and both_state == grid_state
    assert "\n 0  " in grid_state and game.board().splitlines()[0] in grid_state
    assert base_question == grid_question == BASE_QUESTION
    assert goal_question == both_question == GOAL_QUESTION
    assert base_choices == grid_choices == goal_choices == both_choices
    with pytest.raises(ValueError, match="Unknown prompt variant"):
        decision_inputs(game, "unknown")


def test_comparison_reuses_one_model_and_records_each_variant():
    class FakeModel:
        calls = 0

        def choice(self, state, question, choices):
            self.calls += 1
            if "recommends DIRECTION=" in question:
                recommended = question.split("recommends DIRECTION=", 1)[1].split(".", 1)[0]
                winner = recommended
            else:
                winner = next(key for key, description in choices.items()
                              if description.endswith("; safe") or "; SAFE;" in description
                              or "SAFE=1" in description)
            return SimpleNamespace(
                choice=winner,
                probabilities={key: (0.7 if key == winner else 0.1) for key in choices},
            )

    model, output = FakeModel(), StringIO()
    report = compare_prompts(model, seeds=[7], width=8, height=6, steps=3, output=output)
    assert model.calls == 3 * len(PROMPT_VARIANTS)
    assert len(report["trials"]) == len(report["aggregates"]) == len(PROMPT_VARIANTS)
    assert report["generated_tokens"] == 0
    assert len(output.getvalue().splitlines()) == 3 * len(PROMPT_VARIANTS)


def test_history_and_scored_prompts_expose_new_information():
    game = SnakeGame(seed=7)
    game.step(next(direction for direction, safe in game.safe_moves().items() if safe))
    history_state, history_question, history_choices = decision_inputs(game, "grid_history")
    scored_state, scored_question, scored_choices = decision_inputs(game, "scored_moves")
    assert "Recent head positions" in history_state
    assert scored_state == history_state
    assert "recent head positions" in history_question
    assert "SMALLEST" in scored_question
    assert history_choices == game.choices()
    assert any("cycle steps to food:" in choice for choice in scored_choices.values())


def test_compact_and_ranked_prompts_keep_fixed_direction_order():
    game = SnakeGame(seed=7)
    _, compact_question, compact_choices = decision_inputs(game, "compact_scores")
    _, ranked_question, ranked_choices = decision_inputs(game, "ranked_moves")
    assert tuple(compact_choices) == tuple(ranked_choices) == tuple(DIRECTIONS)
    assert "smallest COST" in compact_question
    assert all("SAFE=" in value and "COST=" in value for value in compact_choices.values())
    assert "PRIORITY=0" in ranked_question
    assert sum("PRIORITY=0" in value for value in ranked_choices.values()) == 1
    assert all("PRIORITY=99" in ranked_choices[d] for d, safe in game.safe_moves().items() if not safe)


def test_explicit_planner_prompts_do_not_reorder_choice_labels():
    game = SnakeGame(seed=7)
    _, flag_question, flag_choices = decision_inputs(game, "best_flag")
    _, direction_question, direction_choices = decision_inputs(game, "planner_direction")
    assert tuple(flag_choices) == tuple(direction_choices) == tuple(DIRECTIONS)
    assert "BEST=YES" in flag_question
    assert sum("BEST=YES" in value for value in flag_choices.values()) == 1
    recommended = next(direction for direction, value in flag_choices.items() if "BEST=YES" in value)
    assert f"DIRECTION={recommended}" in direction_question
    assert direction_choices == {direction: f"DIRECTION={direction}" for direction in DIRECTIONS}
