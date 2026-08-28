"""
Real Mixamo character rendering -- loads assets/models/{warrok,vampire}.glb
(mesh + mixamorig: skeleton + Idle/Punch/Kick/Shoot animation clips, built by
tools/convert_character.py) via Panda3D's Actor class, which plays named
skeletal animation clips directly. src/game/entities.py's procedural rig is
kept as-is (not modified/removed) -- this is a separate, swappable renderer
driving real motion-captured animation instead of a coded triangular swing.

Uses Panda3D's Actor/NodePath API directly (setPos/setH/setColorScale), not
Ursina's Entity wrapper -- Ursina IS Panda3D underneath and shares the same
`render` scene graph, so a raw Actor parented to `render` renders fine
alongside Ursina's own Entities; Actor's animation-clip API (loop/play/pose)
has no Ursina equivalent, so there's no reason to fight Ursina's wrapper here.

Hit uses a real Mixamo clip now (assets/mixamo/Hit Reaction.fbx, converted
via tools/convert_hit_reaction.py into <prefix>_HitReact.glb) -- only its
first third plays (the flinch/impact snap), not the full ~1.6s clip: the
gameplay hit-stun window (match.py's HIT_REACT_DURATION) stays a snappy
0.25s so getting jabbed doesn't freeze you for a second and a half, but the
pose it holds once that window ends is a real recoiled-from-impact frame
instead of a flat color tint. No real Death/KO clip was downloaded -- KO
still uses a crude coded forward-pitch fall, same idea as entities.py's
fallback, until one is added.
"""
from pathlib import Path

from direct.actor.Actor import Actor
from panda3d.core import Filename, NodePath

MODEL_DIR = Path(r"C:\Data_Tekken\assets\models")

# Mixamo's exported forward axis needs a heading offset to make fighters face
# each other along the arena's x-axis. NOT visually confirmed -- I can't see
# the render. If a fighter faces the wrong way (backward/sideways) when you
# run this, that's the first thing to adjust: try 0/90/180/270 here, or flip
# which offset applies to facing>0 vs facing<0 below.
HEADING_OFFSET = 90.0

# fraction of the HitReact clip's frames to actually play -- see module
# docstring: the full clip is a ~1.6s flinch+recover arc, and we only want
# the front slice (the actual impact/recoil), not the long recovery-to-
# neutral tail -- that part's skipped by snapping straight to the Idle
# guard pose once match.py's HIT_REACT_DURATION ends. match.py's
# HIT_REACT_DURATION is derived FROM this fraction (24/30 = 0.8s for 0.5 of
# HitReact's 49 frames) -- change this, change that too, or the snap fires
# before the clip finishes playing and the pose visibly pops mid-motion
# (exactly what happened at the old 0.35/0.25s pairing).
HIT_REACT_FRACTION = 0.5


class RealFighterEntity:
    def __init__(self, model_prefix: str, x: float, facing: int, parent: NodePath):
        """model_prefix: e.g. "warrok" -- loads <prefix>_base.glb (mesh +
        skeleton + Idle) plus <prefix>_Punch.glb / _Kick.glb / _Shoot.glb as
        separate named anim files, Panda3D Actor's own standard multi-
        animation pattern (tools/convert_character_v2.py). v1's approach --
        bundling all 4 clips into one file via Blender NLA tracks -- exported
        files that looked correct (right names, right frame counts, verified
        via getAnimNames()/getNumFrames()) but the pose never actually
        applied at runtime: confirmed by rendering an actual screenshot after
        60 simulated frames of a looping "Idle" reporting isPlaying()=True --
        character stayed frozen in a collapsed rest pose regardless. Root
        cause not fully isolated (likely a skin/joint-binding issue specific
        to multi-track NLA export); switching to Actor's own documented
        pattern instead of debugging the custom one further."""
        self.facing = facing

        def p(suffix):
            # Panda3D's model loader needs its own Filename form for a
            # Windows absolute path -- a raw backslash string reliably fails
            # with "not found on model path" even though the file exists.
            return Filename.fromOsSpecific(str(MODEL_DIR / f"{model_prefix}_{suffix}.glb"))

        # "Idle" (embedded directly in the base file, not a separate anim
        # file) needs to be listed here too -- it did NOT auto-register
        # under its own name just by being the modelRoot's own animation
        # (confirmed: getAnimControl("Idle") returned None without this).
        self.actor = Actor(p("base"), {"Idle": p("base"), "Punch": p("Punch"),
                                        "Kick": p("Kick"), "Shoot": p("Shoot"),
                                        "HitReact": p("HitReact")})
        self.actor.reparentTo(parent)
        self.actor.setPos(x, 0, 0)
        self.actor.setH(HEADING_OFFSET if facing > 0 else HEADING_OFFSET + 180)

        self._current_clip = None
        self._last_action_start = None

    @property
    def root(self):
        return self.actor

    def sync(self, player, opponent_root):
        if player.state == "ko":
            if self._current_clip != "ko":
                self.actor.stop()
                self.actor.setColorScale(1, 1, 1, 1)
                self.actor.setP(80)  # crude forward-pitch fall -- no real
                                      # death clip downloaded, see docstring
                self._current_clip = "ko"
            return

        if player.state == "idle":
            if self._current_clip is None:
                # very first frame of the match -- show the real Idle clip
                # once so the fighter isn't stuck in the raw bind pose.
                self.actor.setP(0)
                self.actor.setColorScale(1, 1, 1, 1)
                self.actor.play("Idle")
                self._current_clip = "Idle"
            elif self._current_clip == "ko":
                # fresh match after a restart -- undo the KO fall and show
                # Idle again (this is a real "new match", not "attack ended").
                self.actor.setP(0)
                self.actor.setColorScale(1, 1, 1, 1)
                self.actor.play("Idle")
                self._current_clip = "Idle"
            elif self._current_clip in ("hit", "Shoot"):
                # Shoot's own last frame and HitReact's cut-off frame (see
                # HIT_REACT_FRACTION) are both awkward holds, not real
                # stances -- snap straight to the Idle-to-fight clip's FINAL
                # frame (the guard/fight-ready pose) instead. pose(), not
                # play(), so nothing visibly plays, it's an instant snap,
                # same as the "no idle rerun" rule for Punch/Kick above.
                self.actor.setColorScale(1, 1, 1, 1)
                last_frame = self.actor.getAnimControl("Idle").getNumFrames() - 1
                self.actor.pose("Idle", last_frame)
                self._current_clip = "settled"
            # else ("Idle", a finished Punch/Kick clip, or "settled"): do
            # nothing. Punch/Kick already hold their own last frame once they
            # finish (Panda3D's play() stops and holds by default) -- re-
            # triggering Idle here was the bug: it visibly played Idle's
            # transition motion over that held pose every single time an
            # attack ended, looking like "punch, then idle plays too". Now
            # pressing 1/2/3 plays *only* that attack clip, full stop.
            return

        if player.state == "hit":
            if self._current_clip != "hit":
                self.actor.setColorScale(1, 0.55, 0.55, 1)  # lighter tint now
                                                              # that a real
                                                              # recoil pose
                                                              # also reads as
                                                              # "just got hit"
                num_frames = self.actor.getAnimControl("HitReact").getNumFrames()
                end_frame = max(1, int(num_frames * HIT_REACT_FRACTION))
                self.actor.play("HitReact", fromFrame=0, toFrame=end_frame)
                self._current_clip = "hit"
            return

        if player.state == "attacking" and player.current_action:
            clip = player.current_action.capitalize()  # "punch" -> "Punch" --
                                                          # matches the clip
                                                          # names convert_character.py
                                                          # baked into the .glb
            is_new_action = self._last_action_start != player.state_started_at
            self._last_action_start = player.state_started_at
            if is_new_action:
                self.actor.setColorScale(1, 1, 1, 1)
                self.actor.play(clip)
                self._current_clip = clip
