"""Best-of-3 round controller wrapping src/game/match.py's single-round
Match. Match itself stays untouched (single KO = single round's business,
still fully unit-testable headless) -- this layer just keeps score across
rounds and decides when the whole fight is over: first to 2 round wins
takes the match; a 1-1 split forces a 3rd, deciding round.
"""
from src.game.match import Match

ROUNDS_TO_WIN = 2


class RoundMatch:
    def __init__(self, p1_name: str, p2_name: str):
        self.p1_name = p1_name
        self.p2_name = p2_name
        self.round_wins = {"p1": 0, "p2": 0}
        self.round_num = 1
        self.match = Match(p1_name, p2_name)
        self.match_winner = None  # "p1" / "p2" once someone hits ROUNDS_TO_WIN

    def side_of(self, player) -> str:
        return "p1" if player is self.match.p1 else "p2"

    def round_winner(self):
        """The Match's winner for the CURRENT round, or None if still going."""
        return self.match.winner

    def report_round_result(self):
        """Call once after self.match.winner becomes non-None. Records the
        round win and, if that clinches ROUNDS_TO_WIN, sets match_winner.
        Returns the winning side ("p1"/"p2") for convenience."""
        side = self.side_of(self.match.winner)
        self.round_wins[side] += 1
        if self.round_wins[side] >= ROUNDS_TO_WIN:
            self.match_winner = side
        return side

    def is_match_over(self) -> bool:
        return self.match_winner is not None

    def start_next_round(self):
        self.round_num += 1
        self.match = Match(self.p1_name, self.p2_name)
