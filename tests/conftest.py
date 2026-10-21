import pytest


@pytest.fixture
def review_hand_history() -> str:
    """A complete hand — for testing review mode."""
    return (
        "6-max cash, 100bb effective. Hero on BTN with AhKs. "
        "Hero raises to 2.5bb, SB calls. "
        "Flop: Kh 7d 2c. SB checks, Hero bets 3bb, SB calls. "
        "Turn: 9s. SB checks, Hero bets 8bb, SB folds."
    )


@pytest.fixture
def trainer_hand_history() -> str:
    """A hand that stops mid-decision — for testing trainer/simulate mode."""
    return (
        "6-max cash, 100bb effective. Hero on BTN with AhKs. "
        "Hero raises to 2.5bb, SB calls. "
        "Flop: Kh 7d 2c. SB checks, Hero bets 3bb, SB calls. "
        "Turn: 9s. SB checks. Hero to act."
    )