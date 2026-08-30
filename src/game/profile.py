"""Local player identity/stats -- there is no backend/server/account system
in this project (it's a native desktop app, confirmed: no api/backend/auth
directories anywhere in the repo), so "player profile" means a small JSON
file on this machine, not a real account. Explicitly scoped that way per
user decision (local-profile-only, no login) rather than fabricating a fake
multi-user backend.

One profile per machine (not per-P1/P2) -- the person actually running the
game is the "player" the menu/profile screen is for; P1 vs P2 is a
per-match seat, not a separate identity, same as before this system existed.
"""
import json
import time
from pathlib import Path

SAVE_DIR = Path(__file__).resolve().parent.parent.parent / "save"
PROFILE_PATH = SAVE_DIR / "profile.json"

DEFAULT_USERNAME = "FIGHTER"


class PlayerProfile:
    def __init__(self, username: str, wins: int = 0, losses: int = 0,
                 fighter_counts: dict | None = None, created_at: float | None = None):
        self.username = username
        self.wins = wins
        self.losses = losses
        self.fighter_counts = fighter_counts or {}   # model_key -> times picked
        self.created_at = created_at or time.time()

    @property
    def matches_played(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        return (self.wins / self.matches_played) if self.matches_played else 0.0

    @property
    def favorite_fighter(self) -> str | None:
        if not self.fighter_counts:
            return None
        return max(self.fighter_counts, key=self.fighter_counts.get)

    def record_pick(self, fighter_key: str):
        self.fighter_counts[fighter_key] = self.fighter_counts.get(fighter_key, 0) + 1

    def record_result(self, won: bool):
        if won:
            self.wins += 1
        else:
            self.losses += 1
        self.save()

    def save(self):
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "username": self.username,
            "wins": self.wins,
            "losses": self.losses,
            "fighter_counts": self.fighter_counts,
            "created_at": self.created_at,
        }
        # atomic-ish write -- a crash mid-write leaves the .tmp file, not a
        # half-written profile.json that load() would choke on next launch.
        tmp = PROFILE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(PROFILE_PATH)

    @classmethod
    def load(cls):
        """Returns the saved profile, or None if this is a first run (no
        save file yet) -- caller (play_game.py's main()) shows the name-
        entry screen in that case, same idea as any game's first-launch
        "enter your name" prompt."""
        if not PROFILE_PATH.is_file():
            return None
        try:
            data = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None  # corrupt/unreadable save -- treat as first run rather than crash
        return cls(
            username=data.get("username", DEFAULT_USERNAME),
            wins=data.get("wins", 0),
            losses=data.get("losses", 0),
            fighter_counts=data.get("fighter_counts", {}),
            created_at=data.get("created_at"),
        )

    @classmethod
    def create(cls, username: str):
        profile = cls(username=username.strip() or DEFAULT_USERNAME)
        profile.save()
        return profile
