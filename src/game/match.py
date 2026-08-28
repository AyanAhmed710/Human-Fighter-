"""
Engine-agnostic match/damage/state logic -- no ursina import here on purpose,
so this can be unit-tested with plain `python -c` / asserts without a
graphical window (I can't see the game's rendered output, so the logic that
CAN be verified headless should be kept separate from the rendering code
that can't). src/game/entities.py (ursina-dependent) reads Player.state each
frame to decide what to render/animate; it never decides game rules itself.

Damage/timing numbers are a first-pass balance guess, not tuned from any
data -- expect to retune after real two-player playtesting, same as the
cooldown numbers in player_input.py.
"""
import random
import time

# Critical hits: a flat per-attack chance to land extra hard and stun the
# defender well past the normal hit-react window (see _apply_damage/update
# below -- the stun just reuses player.state == "hit"/state_until, so
# real_entities.py needs no changes: it already holds the HitReact clip's
# last frame + red tint for however long state_until says, normal hit or
# crit alike).
CRIT_CHANCE = 0.25
CRIT_DAMAGE_MULTIPLIER = 1.75
CRIT_STUN_DURATION = 3.0

ACTION_STATS = {
    # damage: HP off a 100-HP bar. anim_duration: how long the attacker is
    # locked in the attack animation (can't queue another action) -- set to
    # match each real Mixamo clip's native length at 30fps (Punching.fbx=39
    # frames, Kicking.fbx=66, Shooting.fbx=36; confirmed via inspect_fbx.py),
    # not the earlier procedural-rig guesses, so real_entities.py's Actor
    # playback isn't cut off mid-swing or left hanging past game-state idle.
    # impact_delay: seconds into the animation when damage actually lands --
    # still a guess (~40-60% through the swing) since I can't watch the clip
    # to find the actual strike frame; retune once you've seen it play.
    "punch":  {"damage": 8,  "anim_duration": 39 / 30, "impact_delay": 0.55},
    "kick":   {"damage": 14, "anim_duration": 66 / 30, "impact_delay": 1.10},
    "shoot":  {"damage": 10, "anim_duration": 36 / 30, "impact_delay": 0.70},
}

# Was a flat 0.25s guess from before any real hit-reaction clip existed.
# real_entities.py now plays a real HitReact clip on hit (49 frames,
# confirmed via getAnimControl().getNumFrames()) but only its first
# HIT_REACT_FRACTION=0.5 (24 frames) -- the rest is recovery-to-neutral we
# don't need since sync() snaps straight to the Idle guard pose once this
# window ends anyway. Left at the old 0.25s, that snap fired while the clip
# was still mid-recoil (24 frames/30fps=0.8s > 0.25s), popping the pose
# mid-motion instead of letting the flinch finish -- 0.8s matches it exactly.
# If HIT_REACT_FRACTION changes, this needs to change with it.
HIT_REACT_DURATION = 24 / 30
MAX_HEALTH = 100


class Player:
    def __init__(self, name: str):
        self.name = name
        self.health = MAX_HEALTH
        self.state = "idle"          # idle | attacking | hit | ko
        self.current_action = None   # "punch" | "kick" | "shoot" while attacking
        self.state_until = 0.0       # time.time() deadline for current state
        self.state_started_at = time.time()  # for the renderer to interpolate a
                                              # wind-up->strike->return pose curve
                                              # over the state's duration -- match
                                              # logic itself never reads this
        self.pending_damage = None   # (target_player, damage, apply_at_time, is_crit) or None
        self.was_crit_hit = False    # set True on this player by _apply_damage the frame a
                                      # crit lands on them; stays True through the whole stun
                                      # (state=="hit") for the renderer/commentary to read

    def is_locked(self) -> bool:
        """True while mid-attack-animation or mid-hit-reaction -- an action
        arriving from the classifier during this window is dropped, not
        queued (matches ActionSegmenter's own cooldown already preventing
        rapid-fire re-triggers; this is the game-state-level version of the
        same idea, in case timings ever drift apart)."""
        return self.state in ("attacking", "hit") and time.time() < self.state_until

    def is_ko(self) -> bool:
        return self.state == "ko"


class Match:
    def __init__(self, player1_name: str = "Player 1", player2_name: str = "Player 2"):
        self.p1 = Player(player1_name)
        self.p2 = Player(player2_name)
        self.winner = None   # None while in progress, else the winning Player

    def _opponent(self, player: Player) -> Player:
        return self.p2 if player is self.p1 else self.p1

    def try_action(self, player: Player, action: str) -> bool:
        """Called when a player's classifier fires a recognized action.
        Returns True if it was accepted (started an attack), False if
        dropped (player locked mid-animation, or match already over)."""
        if self.winner is not None or player.is_ko() or player.is_locked():
            return False

        stats = ACTION_STATS[action]
        now = time.time()
        player.state = "attacking"
        player.state_started_at = now
        player.current_action = action
        player.state_until = now + stats["anim_duration"]

        is_crit = random.random() < CRIT_CHANCE
        damage = round(stats["damage"] * CRIT_DAMAGE_MULTIPLIER) if is_crit else stats["damage"]
        opponent = self._opponent(player)
        player.pending_damage = (opponent, damage, now + stats["impact_delay"], is_crit)
        return True

    def update(self):
        """Call once per game-loop frame. Applies any due damage, clears
        expired states, checks KO/win condition."""
        now = time.time()
        for player in (self.p1, self.p2):
            if player.pending_damage is not None:
                target, damage, apply_at, is_crit = player.pending_damage
                if now >= apply_at:
                    self._apply_damage(target, damage, is_crit)
                    player.pending_damage = None

            if player.state in ("attacking", "hit") and now >= player.state_until:
                player.state = "ko" if player.health <= 0 else "idle"
                player.state_started_at = now
                player.current_action = None

        if self.winner is None:
            if self.p1.is_ko() and not self.p2.is_ko():
                self.winner = self.p2
            elif self.p2.is_ko() and not self.p1.is_ko():
                self.winner = self.p1

    def _apply_damage(self, target: Player, damage: int, is_crit: bool = False):
        now = time.time()
        target.health = max(0, target.health - damage)
        target.state_started_at = now
        target.was_crit_hit = is_crit
        if target.health <= 0:
            target.state = "ko"
            target.state_until = now + 1e9  # stays KO'd, no auto-clear
        else:
            target.state = "hit"
            target.state_until = now + (CRIT_STUN_DURATION if is_crit else HIT_REACT_DURATION)
