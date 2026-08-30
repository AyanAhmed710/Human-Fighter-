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
import time

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
# HIT_REACT_DURATION is derived FROM this fraction (12/30 = 0.4s for 0.25 of
# HitReact's 49 frames) -- change this, change that too, or the snap fires
# before the clip finishes playing and the pose visibly pops mid-motion
# (exactly what happened at the old 0.35/0.25s pairing). Was 0.5 (0.8s) --
# per an explicit "getting stuck for ~1s on every hit" complaint, halved
# again to keep normal hits snappy; only the crit sequence should feel like
# a real stun now.
HIT_REACT_FRACTION = 0.25

# How long to hold the settled-Idle pose before actually starting a punch/
# kick/shoot clip that was thrown WHILE blocking -- purely a render-side
# beat so the swing doesn't look like it pops straight out of the raised
# guard pose. Deliberately small: match.py's own impact_delay timer (0.55-
# 1.10s depending on the move) already started counting the instant
# try_action() fired, unaffected by this -- too large a value here would
# make the visual swing start late enough to look out of sync with when the
# hit actually lands, so this stays a small fraction of the shortest move's
# impact_delay, not a real "recovery time" combat-mechanic change.
ATTACK_FROM_BLOCK_SETTLE = 0.08

# Exact frame to cut "Getting Up" at, for the crit-stun sequence -- picked
# directly by the user against Mixamo's own preview player (frame 154/258
# there was still down on hands/knees, not standing; 156 was their call).
CRIT_GETTING_UP_CUTOFF_FRAME = 156

# The 3 crit-stun clips (HitReact full + Stunned + GettingUp trimmed to
# CRIT_GETTING_UP_CUTOFF_FRAME) run 49+65+156=270 frames total at normal
# (1x) speed -- 270/30fps = 9.0s. Brought down in 2 steps per explicit user
# requests: first to 5s (1.8x), then to 3s (3.0x): 270/(30*3.0) = 3.0s
# exactly. match.py's CRIT_STUN_DURATION must match this real total or the
# stun would end while GettingUp is still mid-motion -- change one, change
# both. NOTE: 3x is a much bigger speedup than 1.8x was -- a "getting up"
# motion built for a real-time recovery can start looking unnaturally
# fast/sped-up-film at this rate rather than a quick, snappy recovery.
# Flagged, not blocked on -- dial back toward 1.8-2x if it looks off.
CRIT_PLAYBACK_RATE = 3.0


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

        # "Idle" needs to be listed here too even when it's the base file's
        # own embedded animation -- it did NOT auto-register under its own
        # name just by being the modelRoot's own animation (confirmed:
        # getAnimControl("Idle") returned None without this). Prefer a
        # dedicated <prefix>_Idle.glb (Mixamo's "Idle" clip, converted via
        # tools/convert_block_anims.py) if one's been converted -- falls
        # back to the base file's own embedded clip otherwise, so nothing
        # breaks for a character that hasn't had one made yet.
        idle_path = MODEL_DIR / f"{model_prefix}_Idle.glb"
        anim_files = {"Idle": p("Idle") if idle_path.is_file() else p("base"),
                      "Punch": p("Punch"), "Kick": p("Kick"),
                      "Shoot": p("Shoot"), "HitReact": p("HitReact")}
        # Block sequence: "IdleToFight" (Mixamo's "Standing Idle To Fight
        # Idle", one-shot transition into the guard) plays once, then
        # "Block" (Mixamo's "Bouncing Fight Idle", a real LOOP -- not
        # one-shot like Punch/Kick/Shoot) takes over for as long as the
        # guard's held (and allowed -- see match.py's Player.guard_up, which
        # folds in the 5s hold cap/3s cooldown, and this class's sync()
        # below). Converted via tools/convert_block_anims.py, same
        # anim-only pipeline as HitReact. Each is added to Actor's clip dict
        # only if its file actually exists -- asking Panda3D to load a path
        # that isn't there raises at construction, which would crash the
        # whole game over a missing decorative clip instead of just falling
        # back to holding the Idle pose (see sync()).
        self._has_idle_to_fight_clip = (MODEL_DIR / f"{model_prefix}_IdleToFight.glb").is_file()
        self._has_block_clip = (MODEL_DIR / f"{model_prefix}_Block.glb").is_file()
        if self._has_idle_to_fight_clip:
            anim_files["IdleToFight"] = p("IdleToFight")
        if self._has_block_clip:
            anim_files["Block"] = p("Block")
        # Crit-hit stun sequence: HitReact (full clip this time, not the
        # HIT_REACT_FRACTION cut normal hits use) -> Stunned (dazed hold) ->
        # GettingUp (trimmed to CRIT_GETTING_UP_CUTOFF_FRAME -- the full
        # clip is a real ground-to-standing recovery, 259 frames/~8.6s, way
        # more than wanted here; the cut point was picked directly by the
        # user against Mixamo's own preview, not tuned by me). Needs BOTH
        # clips to run the sequence -- falls back to the short normal-hit
        # behavior for crits too if either is missing, see sync().
        self._has_stunned_clip = (MODEL_DIR / f"{model_prefix}_Stunned.glb").is_file()
        self._has_getting_up_clip = (MODEL_DIR / f"{model_prefix}_GettingUp.glb").is_file()
        self._has_crit_clips = self._has_stunned_clip and self._has_getting_up_clip
        if self._has_stunned_clip:
            anim_files["Stunned"] = p("Stunned")
        if self._has_getting_up_clip:
            anim_files["GettingUp"] = p("GettingUp")
        self.actor = Actor(p("base"), anim_files)
        self.actor.reparentTo(parent)
        self.actor.setPos(x, 0, 0)
        self.actor.setH(HEADING_OFFSET if facing > 0 else HEADING_OFFSET + 180)

        self._current_clip = None
        self._last_action_start = None
        self._block_transition_start = None  # time.time() when IdleToFight
                                              # started, for the transition
                                              # ->loop handoff in sync()
        self._pending_attack = None  # (clip_name, time.time() to actually start
                                      # it) -- see sync()'s attacking branch: an
                                      # attack fired while blocking settles
                                      # through Idle for a beat first, purely
                                      # cosmetic (match.py's own damage timer
                                      # already started ticking the moment
                                      # try_action() fired, untouched by this)
        self._last_hit_start = None  # edge-detect a NEW hit-stun beginning,
                                      # independent of which crit sub-stage
                                      # clip is currently showing
        self._crit_stage_started_at = None  # time.time() when the current
                                             # crit sub-stage clip started,
                                             # for the hit->stunned->gettingup
                                             # handoffs in sync()

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
            # Block: holding the guard stance -- match.py never changes
            # player.state for this (a blocked hit is fully absorbed in
            # place, see _apply_damage), so this is checked first, every
            # idle frame, ahead of the settle/first-frame logic below.
            # Two stages: IdleToFight plays ONCE (one-shot transition into
            # the guard), then Block takes over as a real LOOP for as long
            # as the guard's held -- the handoff is timed off IdleToFight's
            # own frame count (assumed 30fps export, same convention as
            # match.py's ACTION_STATS comment) since Panda3D's Actor has no
            # built-in "on clip finished" callback wired up here.
            if player.guard_up and self._has_block_clip:  # is_blocking AND not
                                                           # stamina-locked-out -- see match.py
                if self._current_clip not in ("IdleToFight", "Block"):
                    self.actor.setColorScale(1, 1, 1, 1)
                    if self._has_idle_to_fight_clip:
                        self.actor.play("IdleToFight")
                        self._current_clip = "IdleToFight"
                        self._block_transition_start = time.time()
                    else:
                        # no transition clip converted -- straight into the
                        # loop, still correct, just less polished.
                        self.actor.loop("Block")
                        self._current_clip = "Block"
                elif self._current_clip == "IdleToFight":
                    num_frames = self.actor.getAnimControl("IdleToFight").getNumFrames()
                    transition_duration = num_frames / 30.0
                    if time.time() - self._block_transition_start >= transition_duration:
                        self.actor.loop("Block")
                        self._current_clip = "Block"
                # else: already looping Block -- nothing to do.
                return
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
            elif self._current_clip in ("hit", "Shoot", "Block", "IdleToFight", "critGettingUp"):
                # Shoot's own last frame, HitReact's cut-off frame (see
                # HIT_REACT_FRACTION), just having released the guard
                # (whether it fully reached the Block loop or the guard key
                # let go mid-transition), and the crit stun sequence finally
                # finishing (match.py's CRIT_STUN_DURATION expiring right as
                # GettingUp's trimmed playback ends) are all awkward holds,
                # not real stances -- snap straight to the Idle clip's own
                # FINAL frame (the guard/fight-ready pose) instead. pose(),
                # not play(), so nothing visibly plays, it's an instant
                # snap, same as the "no idle rerun" rule for Punch/Kick
                # above -- this is the "release the block key -> back to
                # idle" / "recovered from the crit stun -> back to idle" step.
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
            is_new_hit = self._last_hit_start != player.state_started_at
            self._last_hit_start = player.state_started_at
            if is_new_hit:
                self.actor.setColorScale(1, 0.55, 0.55, 1)  # lighter tint now
                                                              # that a real
                                                              # recoil pose
                                                              # also reads as
                                                              # "just got hit"
                if player.was_crit_hit and self._has_crit_clips:
                    # crit: play the FULL HitReact clip (not the fraction-
                    # cut normal hits use) at CRIT_PLAYBACK_RATE, then this
                    # same branch's "not a new hit" leg below advances
                    # through Stunned and GettingUp (same rate) as each
                    # stage's own (rate-adjusted) duration elapses.
                    self.actor.setPlayRate(CRIT_PLAYBACK_RATE, "HitReact")
                    self.actor.play("HitReact")
                    self._current_clip = "critHit"
                    self._crit_stage_started_at = time.time()
                else:
                    num_frames = self.actor.getAnimControl("HitReact").getNumFrames()
                    end_frame = max(1, int(num_frames * HIT_REACT_FRACTION))
                    self.actor.play("HitReact", fromFrame=0, toFrame=end_frame)
                    self._current_clip = "hit"
                return
            # not a new hit this frame -- advance the crit sub-stage machine
            # if we're mid-sequence (normal hits just hold their fraction-
            # cut HitReact pose, nothing more to do for those).
            if self._current_clip == "critHit":
                num_frames = self.actor.getAnimControl("HitReact").getNumFrames()
                if time.time() - self._crit_stage_started_at >= num_frames / (30.0 * CRIT_PLAYBACK_RATE):
                    self.actor.setPlayRate(CRIT_PLAYBACK_RATE, "Stunned")
                    self.actor.play("Stunned")
                    self._current_clip = "critStunned"
                    self._crit_stage_started_at = time.time()
            elif self._current_clip == "critStunned":
                num_frames = self.actor.getAnimControl("Stunned").getNumFrames()
                if time.time() - self._crit_stage_started_at >= num_frames / (30.0 * CRIT_PLAYBACK_RATE):
                    self.actor.setPlayRate(CRIT_PLAYBACK_RATE, "GettingUp")
                    self.actor.play("GettingUp", fromFrame=0, toFrame=CRIT_GETTING_UP_CUTOFF_FRAME)
                    self._current_clip = "critGettingUp"
            # else "hit" (normal) or "critGettingUp": already showing the
            # right thing -- just keep holding/playing until match.py's own
            # state_until flips state back to "idle" (the idle branch's
            # "elif ... in (..., critGettingUp): snap to Idle" handles the
            # final return-to-idle step from there).
            return

        if player.state == "attacking" and player.current_action:
            clip = player.current_action.capitalize()  # "punch" -> "Punch" --
                                                          # matches the clip
                                                          # names convert_character.py
                                                          # baked into the .glb
            is_new_action = self._last_action_start != player.state_started_at
            self._last_action_start = player.state_started_at
            if is_new_action:
                self._pending_attack = None
                if self._current_clip in ("Block", "IdleToFight"):
                    # was blocking/mid-transition-into-block -- settle
                    # through Idle first (instant pose snap, same as the
                    # release-the-guard case above) instead of popping
                    # straight from the raised guard into the swing; the
                    # real attack clip starts a beat later, see
                    # ATTACK_FROM_BLOCK_SETTLE's docstring for why this
                    # doesn't touch match.py's own damage timing.
                    last_frame = self.actor.getAnimControl("Idle").getNumFrames() - 1
                    self.actor.pose("Idle", last_frame)
                    self._current_clip = "settled"
                    self._pending_attack = (clip, time.time() + ATTACK_FROM_BLOCK_SETTLE)
                else:
                    self.actor.setColorScale(1, 1, 1, 1)
                    self.actor.play(clip)
                    self._current_clip = clip
            if self._pending_attack is not None and time.time() >= self._pending_attack[1]:
                pending_clip = self._pending_attack[0]
                self.actor.setColorScale(1, 1, 1, 1)
                self.actor.play(pending_clip)
                self._current_clip = pending_clip
                self._pending_attack = None
