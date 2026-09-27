from types import SimpleNamespace

from logitly.demos.browser import (
    CONTEXTUAL_POLICY, HIGH_DELIBERATION, BrowserAgent, BrowserPolicy, StalePage,
    _parse_text_values, action_space, discover_requirements, requirement_status,
)


def _page():
    return {
        "url": "http://example.test/",
        "title": "Index",
        "text": "Research Index Mathematics archive Travel notes",
        "marker": "one",
        "actions": [
            {"id": "e1", "node": 10, "kind": "click", "role": "link", "label": "Travel notes", "value": ""},
            {"id": "e2", "node": 11, "kind": "click", "role": "link", "label": "Mathematics archive", "value": ""},
            {"id": "e3", "node": 12, "kind": "fill", "role": "textbox", "label": "Search", "value": ""},
            {"id": "wait", "kind": "wait", "label": "Wait for the page to update"},
        ],
    }


def test_action_space_excludes_unsupplied_text_and_never_invents_it():
    elements, targets, controls = action_space(_page()["actions"])
    assert set(targets) == {"CLICK"}
    assert "WAIT" in controls
    assert [element["index"] for element in elements] == ["1", "2"]

    elements, targets, _ = action_space(_page()["actions"], {"Search": "Gödel"})
    assert set(targets) == {"CLICK", "TYPE_TEXT"}
    assert targets["TYPE_TEXT"]["3"]["text"] == "Gödel"
    assert len(elements) == 3


def test_policy_fans_operation_and_targets_out_in_one_batch():
    class FakeModel:
        def __init__(self):
            self.calls = []

        def decide_many(self, requests, batch_size=None):
            self.calls.append((requests, batch_size))
            answers = []
            for request in requests:
                if request.id == "operation":
                    values = {key: (0.9 if key == "CLICK" else 0.1 / (len(request.choices) - 1))
                              for key in request.choices}
                    answers.append(SimpleNamespace(choice="CLICK", confidence=0.9, probabilities=values))
                else:
                    values = {key: (0.8 if key == "2" else 0.2) for key in request.choices}
                    answers.append(SimpleNamespace(choice="2", confidence=0.8, probabilities=values))
            return answers

    model = FakeModel()
    decision = BrowserPolicy(model).choose(_page(), "Open the mathematics archive", [])
    assert len(model.calls) == 1
    requests, batch_size = model.calls[0]
    assert [request.id for request in requests] == ["operation", "click_target"]
    assert batch_size == 2
    assert decision.operation == "CLICK"
    assert decision.choice == "e2"
    assert decision.target == "2"
    assert decision.heads == 2


def test_single_target_is_deterministic_without_a_second_head():
    page = _page()
    page["actions"] = [page["actions"][1], page["actions"][-1]]

    class FakeModel:
        def decide_many(self, requests, batch_size=None):
            assert len(requests) == batch_size == 1
            values = {key: (0.8 if key == "CLICK" else 0.1) for key in requests[0].choices}
            return [SimpleNamespace(choice="CLICK", confidence=0.8, probabilities=values)]

    decision = BrowserPolicy(FakeModel()).choose(page, "Open mathematics", [])
    assert decision.choice == "e2"
    assert decision.target == "1"
    assert decision.target_probabilities == {"1": 1.0}


def test_high_deliberation_is_prompt_only_and_keeps_one_batch():
    class FakeModel:
        calls = []

        def decide_many(self, requests, batch_size=None):
            self.calls.append((requests, batch_size))
            assert all(HIGH_DELIBERATION in request.question for request in requests)
            answers = []
            for request in requests:
                winner = "CLICK" if request.id == "operation" else "2"
                answers.append(SimpleNamespace(
                    choice=winner,
                    confidence=1.0,
                    probabilities={key: float(key == winner) for key in request.choices},
                ))
            return answers

    model = FakeModel()
    result = BrowserPolicy(model, deliberation="high").choose(_page(), "Open mathematics", [])
    assert result.choice == "e2"
    assert len(model.calls) == 1


def test_contextual_policy_exposes_control_state_to_target_head():
    page = _page()
    page["actions"].insert(2, {
        "id": "e4", "node": 13, "kind": "click", "role": "checkbox",
        "label": "Free cancellation", "value": "on", "checked": "false",
    })

    class FakeModel:
        requests = []

        def decide_many(self, requests, batch_size=None):
            self.requests = list(requests)
            answers = []
            for request in requests:
                winner = "CLICK" if request.id == "operation" else next(iter(request.choices))
                answers.append(SimpleNamespace(
                    choice=winner,
                    confidence=1.0,
                    probabilities={key: float(key == winner) for key in request.choices},
                ))
            return answers

    model = FakeModel()
    BrowserPolicy(model, deliberation="contextual").choose(
        page, "Enable Free cancellation, then open Mathematics", [], {"Search": "Gödel"},
    )
    click_request = next(request for request in model.requests if request.id == "click_target")
    assert CONTEXTUAL_POLICY in click_request.question
    assert any("Free cancellation" in value and "checked=false" in value
               for value in click_request.choices.values())


def test_requirement_ledger_tracks_exact_values_and_control_history():
    actions = [
        {"id": "text", "node": 1, "kind": "fill", "role": "searchbox",
         "label": "Destination", "value": ""},
        {"id": "toggle", "node": 2, "kind": "click", "role": "checkbox",
         "label": "Free cancellation", "value": "on", "checked": "false"},
        {"id": "design", "node": 3, "kind": "select", "role": "combobox",
         "label": "Stay category → Design", "value": "Design", "current_value": "All stays"},
        {"id": "casa", "node": 4, "kind": "click", "role": "button",
         "label": "View Casa Flora", "value": ""},
        {"id": "submit", "node": 5, "kind": "click", "role": "button",
         "label": "Find stays", "value": ""},
    ]
    _, targets, _ = action_space(actions, {"Destination": "Lisbon"})
    requirements = discover_requirements(
        "Find Design stays in Lisbon with Free cancellation, then open Casa Flora",
        targets, {"Destination": "Lisbon"},
    )
    initial = requirement_status(requirements, targets, [])
    assert {(item["control"], item["status"]) for item in initial} == {
        ("Destination", "PENDING"),
        ("Free cancellation", "PENDING"),
        ("Stay category", "PENDING"),
        ("View Casa Flora", "PENDING"),
        ("Find stays", "PENDING"),
    }
    history = [
        {"label": "Destination", "entered_text": "Lisbon"},
        {"label": "Stay category → Design", "selected_value": "Design"},
        {"label": "Free cancellation", "control_after": {"checked": "true"}},
        {"label": "Find stays", "operation": "CLICK"},
        {"label": "View Casa Flora", "operation": "CLICK"},
    ]
    completed = requirement_status(requirements, {}, history)
    assert all(item["status"] == "satisfied" for item in completed)


def test_text_value_parser_requires_explicit_nonempty_pairs():
    assert _parse_text_values(["Search=Gödel"]) == {"Search": "Gödel"}
    for invalid in ("Search", "=Gödel", "Search="):
        try:
            _parse_text_values([invalid])
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid value {invalid!r}")


def test_agent_passes_only_caller_supplied_text_to_browser():
    page = _page()

    class FakeBrowser:
        actions = []

        def observe(self):
            return page

        def act(self, action, observed):
            assert observed is page
            self.actions.append(action)
            page["marker"] = "two"

        def fresh(self, observed):
            return True

    class FakeModel:
        calls = 0

        def decide_many(self, requests, batch_size=None):
            self.calls += 1
            if self.calls == 2:
                transcript = requests[0].state["action_transcript"]
                assert transcript[-1]["entered_text"] == "Gödel"
                assert transcript[-1]["operation"] == "TYPE_TEXT"
            request = requests[0]
            winner = "TYPE_TEXT" if self.calls == 1 else "DONE"
            values = {key: float(key == winner) for key in request.choices}
            return [SimpleNamespace(choice=winner, confidence=1.0, probabilities=values)] + [
                SimpleNamespace(choice=next(iter(item.choices)), confidence=1.0,
                                probabilities={key: float(key == next(iter(item.choices))) for key in item.choices})
                for item in requests[1:]
            ]

    browser = FakeBrowser()
    result = BrowserAgent(browser, FakeModel(), "Search for Gödel", text_values={"Search": "Gödel"}).run()
    assert result["status"] == "done"
    assert browser.actions[0]["text"] == "Gödel"


def test_agent_reobserves_and_redecides_after_stale_target():
    page = _page()

    class FakeBrowser:
        attempts = 0

        def observe(self):
            return page

        def act(self, action, observed):
            self.attempts += 1
            if self.attempts == 1:
                raise StalePage("autocomplete changed")
            page["marker"] = "two"

        def fresh(self, observed):
            return True

    class FakeModel:
        calls = 0

        def decide_many(self, requests, batch_size=None):
            self.calls += 1
            winner = "TYPE_TEXT" if self.calls <= 2 else "DONE"
            answers = []
            for request in requests:
                choice = winner if request.id == "operation" else next(iter(request.choices))
                answers.append(SimpleNamespace(
                    choice=choice,
                    confidence=1.0,
                    probabilities={key: float(key == choice) for key in request.choices},
                ))
            return answers

    browser, model = FakeBrowser(), FakeModel()
    result = BrowserAgent(browser, model, "Search", text_values={"Search": "Gödel"}).run()
    assert result["status"] == "done"
    assert browser.attempts == 2
    assert result["decisions"][0]["stale"] is True
    assert len(result["actions"]) == 1
