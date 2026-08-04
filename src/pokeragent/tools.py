"""Tools for AI Agent"""

from concurrent.futures import ProcessPoolExecutor
from pokerkit import Card, Deck, StandardHighHand, calculate_equities, parse_range
from pydantic import BaseModel, Field

SAMPLE_COUNT = 5000

_RANKS = "23456789TJQKA"

def _build_card_range() -> str:
    labels = []
    for i, low in enumerate(_RANKS):
        for high in _RANKS[i:]:
            labels.append(high + low)
    return ",".join(labels)

ANY_TWO_CARDS = _build_card_range()
 
_WIDE_ALIASES = {"any two cards", "any two", "random", "any"}

class EquityResult(BaseModel):
    hero_cards: str
    board: str
    hero_equity: float = Field(ge = 0, le = 1)
    villain_equity: float = Field(ge = 0, le = 1)

def hand_equity(hero_cards: str, board: str, villain_range: str) -> EquityResult:
    """Compute Hero Equity Given Villain Range and Board Cards."""
    range_str = (
        ANY_TWO_CARDS if villain_range.strip().lower() in _WIDE_ALIASES
        else villain_range
    )

    board_cards = Card.parse(board.replace(" ", "")) if board.strip() else ()

    hero_eq, villain_eq = calculate_equities(
        (parse_range(hero_cards.replace(" ", "")), parse_range(range_str)),
            board_cards,
            2,  # hole cards per player (hold'em)
            5,  # total board cards once the hand is complete
            Deck.STANDARD,
            (StandardHighHand,),
            sample_count=SAMPLE_COUNT,
            executor=ProcessPoolExecutor(),
    )

    return EquityResult(
        hero_cards=hero_cards,
        board=board,
        hero_equity=hero_eq,
        villain_equity=villain_eq,
    )