"""
Procedural low-poly fighter rig + rendering -- no external 3D models/rigs
needed (nothing was sourced or asked for), built entirely from ursina's
built-in primitives (cube/sphere) parented into a simple humanoid hierarchy
and posed by direct rotation each frame. Purely a function of
src.game.match.Player's state/current_action/timestamps -- this module never
decides game rules, only how to *show* whatever match.py already decided
(see match.py's docstring for that split).

Everything here is a deterministic function of (state, elapsed fraction of
the state's duration), not a library of pre-baked animation clips -- easiest
thing to get right without being able to see the rendered result myself.
"""
import math
import time

from ursina import Entity, color, curve, destroy

from src.game.match import ACTION_STATS

TORSO_COLOR = {"p1": color.azure, "p2": color.orange}
SKIN_COLOR = color.rgb(230, 190, 160)
PROJECTILE_COLOR = color.yellow

# rotation_z (degrees) swept forward for each attack's limb -- how far the
# arm/leg travels; sign is flipped per-fighter based on which way it faces
# (see FighterEntity.facing).
SWING_ANGLE = {"punch": 55, "kick": 65, "shoot": 80}


def _progress(started_at: float, until: float) -> float:
    now = time.time()
    total = max(until - started_at, 1e-6)
    return max(0.0, min(1.0, (now - started_at) / total))


def _swing_curve(frac: float) -> float:
    """0 -> 1 -> 0 triangular sweep: limb travels out then back over the
    attack's duration, peaking at the midpoint -- simple stand-in for a
    real wind-up/strike/recover animation curve."""
    return 1.0 - abs(2 * frac - 1.0)


class FighterEntity:
    def __init__(self, x: float, facing: int, side_key: str):
        """x: fixed horizontal position in the arena. facing: +1 or -1,
        which way this fighter's front points (toward the opponent) --
        determines which direction limbs swing forward. side_key: "p1"/"p2",
        just for color."""
        self.facing = facing
        self.body_color = TORSO_COLOR[side_key]
        self.root = Entity(position=(x, 0, 0))

        self.torso = Entity(parent=self.root, model="cube", color=self.body_color,
                             scale=(0.5, 0.9, 0.3), position=(0, 1.2, 0))
        self.head = Entity(parent=self.root, model="sphere", color=SKIN_COLOR,
                            scale=0.35, position=(0, 1.85, 0))

        # limbs pivot at their top (shoulder/hip) -- origin=(0,.5,0) in
        # ursina's unit-cube local space moves the rotation pivot to the top
        # of the mesh, so rotating swings it like a real limb instead of
        # spinning around its own center
        self.left_leg = Entity(parent=self.root, model="cube", color=self.body_color,
                                scale=(0.18, 0.75, 0.18), origin=(0, 0.5, 0),
                                position=(-0.15, 0.75, 0))
        self.right_leg = Entity(parent=self.root, model="cube", color=self.body_color,
                                 scale=(0.18, 0.75, 0.18), origin=(0, 0.5, 0),
                                 position=(0.15, 0.75, 0))
        self.left_arm = Entity(parent=self.root, model="cube", color=SKIN_COLOR,
                                scale=(0.15, 0.65, 0.15), origin=(0, 0.5, 0),
                                position=(-0.35, 1.55, 0))
        self.right_arm = Entity(parent=self.root, model="cube", color=SKIN_COLOR,
                                 scale=(0.15, 0.65, 0.15), origin=(0, 0.5, 0),
                                 position=(0.35, 1.55, 0))

        self._striking_arm = self.right_arm   # which arm/leg animates for
        self._striking_leg = self.right_leg   # punch/kick/shoot -- fixed
                                               # choice, not handedness-aware
                                               # (the classifier already
                                               # collapses L/R into one
                                               # "striking" signal, §3 of
                                               # MODEL_JOURNEY.md)
        self._last_action_start = None   # edge-detect a NEW attack beginning,
                                          # for one-shot effects like spawning
                                          # a shoot projectile

    def _reset_pose(self):
        self.left_leg.rotation_z = 0
        self.right_leg.rotation_z = 0
        self.left_arm.rotation_z = 0
        self.right_arm.rotation_z = 0
        self.torso.rotation_x = 0
        self.torso.color = self.body_color
        self.root.rotation_z = 0   # undoes the KO fall-over -- this was the
                                    # bug: a KO set this but nothing ever
                                    # reset it back on restart, so the loser
                                    # stayed lying down every match after

    def sync(self, player, opponent_root):
        """Call once per frame. player: src.game.match.Player. opponent_root:
        the opposing FighterEntity's .root, needed only to aim a shoot
        projectile at the opponent's current position."""
        if player.state == "ko":
            # fall over and stay down -- final, no further per-frame updates
            self.root.rotation_z = 80 * self.facing
            return

        if player.state == "idle":
            if player.guard_up:  # is_blocking AND not stamina-locked-out -- see match.py
                # bouncy guard stance -- both arms raised, faster/shorter
                # bob than the resting idle sway so it visibly reads as "on
                # guard" rather than just standing still. No separate clip
                # needed here (unlike real_entities.py's Block.glb) -- the
                # procedural rig is posed directly every frame already.
                self.left_arm.rotation_z = -55 * self.facing
                self.right_arm.rotation_z = 55 * self.facing
                self.left_leg.rotation_z = 0
                self.right_leg.rotation_z = 0
                self.torso.rotation_x = 0
                self.torso.color = self.body_color
                self.torso.y = 1.2 + 0.035 * math.sin(time.time() * 8)
                return
            self._reset_pose()
            self.torso.y = 1.2 + 0.02 * math.sin(time.time() * 2)  # idle bob
            return

        if player.state == "hit":
            frac = _progress(player.state_started_at, player.state_until)
            self.torso.rotation_x = -15 * (1 - frac)  # snap back, ease out
            self.torso.color = color.red if frac < 0.5 else self.body_color
            return

        if player.state == "attacking" and player.current_action:
            action = player.current_action
            stats = ACTION_STATS[action]
            frac = _progress(player.state_started_at, player.state_until)
            angle = SWING_ANGLE[action] * _swing_curve(frac) * self.facing

            if action == "kick":
                self._striking_leg.rotation_z = angle
            else:  # punch or shoot both use the arm
                self._striking_arm.rotation_z = angle

            is_new_action = self._last_action_start != player.state_started_at
            self._last_action_start = player.state_started_at
            if action == "shoot" and is_new_action:
                self._spawn_projectile(stats["impact_delay"], opponent_root)

    def _spawn_projectile(self, travel_time: float, opponent_root):
        origin = self.right_arm.world_position
        proj = Entity(model="sphere", color=PROJECTILE_COLOR, scale=0.12, position=origin)
        proj.animate_position(opponent_root.position + (0, 1.2, 0), duration=travel_time,
                               curve=curve.linear)
        destroy(proj, delay=travel_time + 0.05)
