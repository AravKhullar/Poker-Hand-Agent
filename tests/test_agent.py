import pytest

from pokeragent.agent import MAX_STEPS, route_after_critique


def test_route_retries_when_ungrounded_and_under_cap():
    state = {"steps": 1, "critique": "ungrounded: no support"}
    assert route_after_critique(state) == "retry"


def test_route_ends_when_grounded():
    state = {"steps": 1, "critique": "grounded"}
    assert route_after_critique(state) == "end"


def test_route_ends_when_max_steps_hit_even_if_ungrounded():
    state = {"steps": MAX_STEPS, "critique": "ungrounded: no support"}
    assert route_after_critique(state) == "end"


@pytest.mark.integration
def test_parse_hand_sets_facing_action_in_trainer_mode(trainer_hand_history):
    from pokeragent.agent import parse_hand

    state = parse_hand({"hand_history": trainer_hand_history})
    hand = state["parsed_actions"][-1]
    assert hand["facing_action"]  # non-empty — hand isn't complete


@pytest.mark.integration
def test_parse_hand_empty_facing_action_in_review_mode(review_hand_history):
    from pokeragent.agent import parse_hand

    state = parse_hand({"hand_history": review_hand_history})
    hand = state["parsed_actions"][-1]
    assert hand["facing_action"] == ""


@pytest.mark.integration
def test_full_graph_runs_end_to_end(review_hand_history):
    from pokeragent.agent import build_graph

    graph = build_graph()
    result = graph.invoke({
        "hand_history": review_hand_history,
        "mode": "review",
        "steps": 0,
    })
    assert result["draft"]
    assert result["critique"]