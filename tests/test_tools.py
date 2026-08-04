import pytest

from pokeragent.tools import ANY_TWO_CARDS, hand_equity


def test_any_two_cards_has_91_labels():
    assert len(ANY_TWO_CARDS.split(",")) == 91


def test_any_two_cards_contains_key_hands():
    labels = ANY_TWO_CARDS.split(",")
    assert "AA" in labels
    assert "22" in labels
    assert "AK" in labels


def test_hand_equity_sums_to_one():
    result = hand_equity("AhKs", "Kh 7d 2c", "any two cards")
    assert result.hero_equity == pytest.approx(1 - result.villain_equity, abs=1e-6)


def test_hand_equity_strong_hand_is_favored():
    # top pair, top kicker on this board should clearly beat a wide range
    result = hand_equity("AhKs", "Kh 7d 2c", "any two cards")
    assert result.hero_equity > 0.6


def test_hand_equity_accepts_explicit_range_notation():
    result = hand_equity("AhKs", "Kh 7d 2c", "22+,AJs+,KQs")
    assert 0 <= result.hero_equity <= 1