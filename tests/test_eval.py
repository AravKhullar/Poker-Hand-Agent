from pokeragent.eval import first_action, recommended_action


def test_recommended_action_reads_recommendation_line():
    draft = "- **Recommendation**: 3-bet to 9bb\n- **Why**: calling is too passive."
    assert recommended_action(draft) == "raise"


def test_recommended_action_does_not_match_call_to_raise():
    assert recommended_action("- **Recommendation**: Call\n- **Why**: ...") == "call"


def test_first_action_canonicalizes_labels():
    assert first_action("raise (3-bet)") == "raise"
    assert first_action("fold") == "fold"
    assert first_action("no action here") is None
