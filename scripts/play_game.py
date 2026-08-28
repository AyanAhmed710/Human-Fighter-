"""
Gesture-controlled 2-player fighter -- Ursina app entrypoint. Wires together:
  - src/game/match.py       (engine-agnostic health/damage/state rules)
  - src/game/entities.py    (procedural low-poly rig, renders match.Player state)
  - src/game/player_input.py (background camera+classifier threads, one per player)

Two control modes:
  --keyboard    no cameras needed -- player 1: 1/2/3 = punch/kick/shoot,
                player 2: 8/9/0 = punch/kick/shoot. For testing the game
                logic/rendering itself without two people and two webcams.
  (default)     real camera-driven play: --camera1/--camera2 pick which
                webcam device index is which player (default 0 and 1).

Usage:
    python scripts/play_game.py --keyboard          # test without cameras
    python scripts/play_game.py                     # cameras 0 and 1
    python scripts/play_game.py --camera1 0 --camera2 2

Press R to restart the match after a KO. Press ESC/q to quit.
"""
import argparse
import math
import random
import sys
import time as _pytime  # stdlib time.time() for the low-health pulse's sin
                         # wave -- ursina's own `time` import below is a
                         # different object (exposes time.dt, not time.time())
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# mediapipe MUST be imported before ursina -- ursina/panda3d ships its own
# native DLLs that shadow one mediapipe's compiled bindings need on Windows;
# importing in this order avoids "DLL load failed while importing
# _framework_bindings" (reproducible, confirmed by import-order swap test).
import mediapipe  # noqa: F401  (import order side effect only, not used directly here)

import numpy as np
from panda3d.core import AmbientLight, DirectionalLight, Filename
from panda3d.core import Texture as PandaTexture
from ursina import (EditorCamera, Entity, Text, Texture, Ursina, Vec3, application, camera, color,
                     destroy, load_texture, scene, time)

from src.game.commentator import Commentator
from src.game.entities import FighterEntity
from src.game.lava_flow import LavaFlow
from src.game.match import MAX_HEALTH
from src.game.menu import CharacterSelect, MainMenu
from src.game.player_input import PlayerCameraInput
from src.game.real_entities import RealFighterEntity
from src.game.round_match import ROUNDS_TO_WIN, RoundMatch
from src.game import sfx

ARENA_HALF_WIDTH = 1.0  # was 3.0 -- fighters 6 units apart, way beyond
                         # punch/kick reach. 1.0 puts them 2 units apart,
                         # close enough that attacks visibly land.
P1_MODEL = "warrok"
P2_MODEL = "vampire"

# assets/arena/source/Lava_Stage.fbx, converted by tools/convert_arena.py.
# Numbers below came from measuring the actual model (tools/measure_floor.py,
# tools/screenshot_arena*.py), not guessed:
#   - 'base' mesh's bounding-box TOP (Y=0.1522 local) looked right in an
#     isolated screenshot but was wrong in real play -- confirmed by the
#     user's own in-game screenshot, fighters stood ~1.6 units above the
#     ground. Reason: 'base' isn't a flat disc, it's a bumpy rock outcrop,
#     and that bounding-box top was an unrelated peak off to the side, not
#     the surface actually underneath the fighters' feet.
#   - tools/measure_floor.py does a real triangle/ray test -- walks every
#     triangle of every arena mesh and interpolates the exact surface Y at
#     the fighters' actual XZ spot (+-1, 0), instead of trusting a nearby
#     vertex or a bounding box. That surface is local Y=0.072985; fighters'
#     feet land exactly on it now (verified: 0.0000 world-space gap).
#   - ARENA_SCALE=20 makes the platform ~5.3 units wide (comfortably fits
#     two fighters 2 units apart) and the surrounding crater rim ~15.6
#     units across.
ARENA_MODEL_PATH = r"C:\Data_Tekken\assets\models\arena.glb"
ARENA_FLOOR_LOCAL_Y = 0.072985
ARENA_SCALE = 20

KEYBOARD_BINDINGS = {
    "1": ("p1", "punch"), "2": ("p1", "kick"), "3": ("p1", "shoot"),
    "8": ("p2", "punch"), "9": ("p2", "kick"), "0": ("p2", "shoot"),
}

# Round-flow timing (seconds). Match logic (try_action/match.update) is
# frozen outside the "fight" phase -- see Game.update()'s phase machine --
# so nobody can land a punch during the "ROUND 2" title card or while the
# KO banner is still up.
INTRO_DURATION = 1.3      # "ROUND N" title card
GO_DURATION = 0.6         # "FIGHT!" flash
ROUND_END_DURATION = 2.6  # winner banner hangs before the next round/match-end

LOW_HEALTH_FRACTION = 0.25  # health bar pulses red + heartbeat sfx below this

# Fixed gameplay camera framing (main()'s own verified-safe numbers -- see
# main()'s camera setup for how these were measured). Named here so the
# camera-shake helper below can always snap back to the exact same spot
# instead of drifting frame to frame.
BASE_CAMERA_POS = (4.95, 6.213, 16.094)
BASE_CAMERA_ROT = (15.625, -161.805, -0.0)

# Hit-stop: a brief total freeze the instant a hit lands -- update() skips
# input/match/sync entirely for this long, which is what actually reads as
# "impact" in most fighting games (more than the animation itself). Kept
# short enough it doesn't mess with match.py's own (wall-clock, not frame-
# based) damage timing -- see update()'s hitstop check for why that's safe.
HITSTOP_DURATION_HIT = 0.06
HITSTOP_DURATION_KO = 0.16

# Camera shake -- decays linearly from full magnitude to 0 over duration,
# random jitter each frame, snaps back to BASE_CAMERA_POS the instant it ends.
SHAKE_DURATION_HIT = 0.12
SHAKE_MAGNITUDE_HIT = 0.05
SHAKE_DURATION_KO = 0.35
SHAKE_MAGNITUDE_KO = 0.18

# Critical hits (src/game/match.py: CRIT_CHANCE=25%, 1.75x damage, 3s stun)
# land harder than a normal hit but short of a KO -- shake/hitstop sized
# between the two so a crit visibly reads as "bigger than a normal hit"
# without stealing the KO's own punch.
HITSTOP_DURATION_CRIT = 0.10
SHAKE_DURATION_CRIT = 0.22
SHAKE_MAGNITUDE_CRIT = 0.11

# Webcam picture-in-picture size (camera.ui units, same space the health bars
# and banners already live in) -- width fixed, height derived per-frame from
# the camera's real aspect ratio so a 640x480 feed doesn't look squashed.
CAM_PREVIEW_WIDTH = 0.38


class WebcamPreview:
    """Live top-corner picture-in-picture of one player's own camera feed --
    lets a player see themselves so they can line up gestures the same way
    a real motion-capture booth would, instead of guessing where they are in
    frame. Reads PlayerCameraInput.latest_frame each real engine frame (a
    plain BGR numpy array, already horizontally mirrored by that thread) and
    pushes it straight into a panda3d Texture's RAM image -- no cv2 window,
    this renders inside the game's own UI layer.

    The panda3d Texture is built lazily off whatever resolution the camera
    actually reports on its first real frame (setup2dTexture needs a fixed
    width/height up front, and that isn't known until the capture thread has
    actually opened the device), then reused/mutated in place every frame
    after that -- rebuilding a fresh Texture (and Entity) every frame would
    mean a full GPU re-upload from scratch 60 times a second instead of the
    cheap "same texture, new pixels" update this does instead.
    """

    def __init__(self, corner: str, label: str):
        self.corner = corner  # "left" | "right" -- which side of the screen
        self.label = label
        self.panda_tex = None
        self.entity = None
        self.label_text = None
        self._size = None  # (w, h) of the camera frame the texture was built for

    def update(self, frame):
        if frame is None:
            return  # camera thread hasn't produced a frame yet (or never connected)

        h, w = frame.shape[:2]
        if self._size != (w, h):
            self._build(w, h)

        # Panda textures store row 0 as the BOTTOM row (OpenGL convention);
        # cv2 frames store row 0 as the TOP row -- flip vertically or the
        # preview renders upside down (confirmed against ursina.Texture's own
        # PIL path above, which does the same FLIP_TOP_BOTTOM before upload).
        # "BGR" format string skips the BGR->RGB channel swap cv2 frames need
        # everywhere else -- panda3d accepts the raw OpenCV channel order
        # directly, so no per-frame cvtColor cost here.
        self.panda_tex.setRamImageAs(np.ascontiguousarray(np.flipud(frame)).tobytes(), "BGR")

    def _build(self, w: int, h: int):
        self._size = (w, h)
        self.panda_tex = PandaTexture()
        self.panda_tex.setup2dTexture(w, h, PandaTexture.TUnsignedByte, PandaTexture.FRgb)

        height = CAM_PREVIEW_WIDTH * (h / w)
        if self.entity is None:
            if self.corner == "left":
                pos, origin = (-0.87, 0.48), (-0.5, 0.5)
                label_pos = (-0.87, 0.48 - height - 0.025)
            else:
                # nudged in from the hard-right edge (0.87 like the left one
                # would be) -- ursina's own always-on dev stats widget
                # ("entities:"/"colliders:") lives in that exact top-right
                # corner and would sit on top of the preview otherwise.
                pos, origin = (0.68, 0.48), (0.5, 0.5)
                label_pos = (0.68, 0.48 - height - 0.025)
            self.entity = Entity(parent=camera.ui, model="quad",
                                  texture=Texture(self.panda_tex),
                                  scale=(CAM_PREVIEW_WIDTH, height), position=pos, origin=origin)
            self.label_text = Text(self.label, position=label_pos, scale=0.7, color=color.white,
                                    origin=(0 if self.corner == "left" else 0, 0))
        else:
            # resolution changed mid-session (shouldn't normally happen, but
            # cheap to handle) -- same entity, fresh texture object + rescale.
            self.entity.texture = Texture(self.panda_tex)
            self.entity.scale = (CAM_PREVIEW_WIDTH, height)

    def destroy(self):
        if self.entity is not None:
            destroy(self.entity)
            destroy(self.label_text)
            self.entity = None
            self.label_text = None


class Game:
    """Owns one best-of-3 RoundMatch plus a small phase state machine
    (intro -> go -> fight -> round_end -> [intro again | match_end]) that
    gates src/game/match.py's action/damage logic to only the "fight"
    phase -- nobody can land a hit while the "ROUND 2" title card or a KO
    banner is on screen, same as a real fighting game freezing input during
    those beats."""

    def __init__(self, keyboard_mode: bool, camera1: int, camera2: int, procedural: bool,
                 model1: str = P1_MODEL, model2: str = P2_MODEL):
        self.keyboard_mode = keyboard_mode
        self.round_match = RoundMatch("Player 1", "Player 2")

        # fighter1 (p1/model1) placed on the RIGHT, fighter2 (p2/model2) on
        # the LEFT -- swapped per user request. Only the world x/facing swap
        # here; health_bar1/"PLAYER 1"/etc. still track match.p1 same as
        # before, so the on-screen model just moves side, stats stay wired
        # to whichever fighter object they always were.
        if procedural:
            # procedural rig has no swappable model -- character select's
            # choice only applies to the real Mixamo-model renderer below.
            self.fighter1 = FighterEntity(x=ARENA_HALF_WIDTH, facing=-1, side_key="p1")
            self.fighter2 = FighterEntity(x=-ARENA_HALF_WIDTH, facing=1, side_key="p2")
        else:
            self.fighter1 = RealFighterEntity(model1, x=ARENA_HALF_WIDTH, facing=-1, parent=scene)
            self.fighter2 = RealFighterEntity(model2, x=-ARENA_HALF_WIDTH, facing=1, parent=scene)

        self.input1 = self.input2 = None
        self.cam_preview1 = self.cam_preview2 = None
        if not keyboard_mode:
            self.input1 = PlayerCameraInput(camera_index=camera1).start()
            self.input2 = PlayerCameraInput(camera_index=camera2).start()
            self.cam_preview1 = WebcamPreview("left", "PLAYER 1 CAM")
            self.cam_preview2 = WebcamPreview("right", "PLAYER 2 CAM")

        self.rain = None  # rain disabled per user request

        # AI live commentary -- entirely optional, self-disables with no
        # OPENAI_API_KEY set (see commentator.py's own docstring). Named
        # after the actual chosen characters, not "Player 1"/"Player 2", so
        # commentary lines can call fighters by name like a real broadcast.
        self.commentator = Commentator(model1.capitalize(), model2.capitalize())
        self._last_commentary_path = None

        self.phase = "intro"
        self.phase_timer = 0.0
        self._low_health_active = False  # drives the red pulse + heartbeat loop

        self.hitstop_timer = 0.0   # while > 0, update() freezes everything
        self.shake_timer = 0.0     # while > 0, camera jitters around BASE_CAMERA_POS
        self.shake_duration = 1.0  # avoids a div-by-zero in the decay fraction
        self.shake_magnitude = 0.0
        self._round_end_pending = False  # True between "winner detected" and
                                          # "KO hitstop finished counting down"

        self._build_ui()
        sfx.play_round_announcement(self.round_match.round_num)

    @property
    def match(self):
        # RoundMatch swaps in a fresh Match each round (start_next_round) --
        # a property means every other method below always reads whichever
        # round is currently live without needing to know about rounds.
        return self.round_match.match

    def _build_ui(self):
        self.name_text1 = Text("PLAYER 1", position=(-0.85, 0.47), scale=1.3, color=color.azure)
        self.name_text2 = Text("PLAYER 2", position=(0.55, 0.47), scale=1.3, color=color.orange)

        Entity(parent=camera.ui, model="quad", color=color.dark_gray,
               scale=(0.4, 0.045), position=(-0.6, 0.43), origin=(-0.5, 0))
        Entity(parent=camera.ui, model="quad", color=color.dark_gray,
               scale=(0.4, 0.045), position=(0.2, 0.43), origin=(-0.5, 0))
        self.health_bar1 = Entity(parent=camera.ui, model="quad", color=color.lime,
                                   scale=(0.4, 0.045), position=(-0.6, 0.43), origin=(-0.5, 0))
        # health_bar2 anchored on its RIGHT edge (origin=0.5, position=the
        # bar's right end at 0.6) unlike bar1's left-edge anchor -- mirrored
        # on purpose. Before this fix both bars anchored left (origin=-0.5),
        # so bar1 correctly drained toward screen-center as p1 lost health,
        # but bar2 ALSO drained toward screen-center instead of draining
        # toward its own outer/right edge under the "PLAYER 2" label --
        # remaining green crept toward the middle instead of staying pinned
        # under P2's name, which read as "the bar's on the wrong side" even
        # though the label/scale_x were always wired to the correct player
        # (confirmed earlier via forced-state/forced-health screenshot tests).
        self.health_bar2 = Entity(parent=camera.ui, model="quad", color=color.lime,
                                   scale=(0.4, 0.045), position=(0.6, 0.43), origin=(0.5, 0))

        # round-win pips -- ROUNDS_TO_WIN small squares per player, filled in
        # as they win rounds. Sit just under each health bar.
        self.pips1 = [Entity(parent=camera.ui, model="quad", color=color.dark_gray,
                              scale=(0.03, 0.03), position=(-0.6 + i * 0.045, 0.395),
                              origin=(-0.5, 0)) for i in range(ROUNDS_TO_WIN)]
        self.pips2 = [Entity(parent=camera.ui, model="quad", color=color.dark_gray,
                              scale=(0.03, 0.03), position=(0.6 - i * 0.045, 0.395),
                              origin=(0.5, 0)) for i in range(ROUNDS_TO_WIN)]

        self.round_banner = Text("ROUND 1", position=(0, 0.12), scale=4, color=color.yellow,
                                  origin=(0, 0))
        self.sub_banner = Text("", position=(0, 0.0), scale=1.6, color=color.white,
                                origin=(0, 0))
        self.hint_text = Text("R to restart -- ESC/q to quit -- C for free camera",
                               position=(-0.2, -0.47), scale=0.8, color=color.gray)

        # full-screen red pulse, alpha driven every frame in update() while
        # either player is under LOW_HEALTH_FRACTION -- z between the
        # gameplay HUD (default z=0) and nothing else fight-phase draws
        # behind it, so it just tints the whole screen without hiding text.
        self.low_health_overlay = Entity(parent=camera.ui, model="quad",
                                          color=color.rgba32(255, 0, 0, 0),
                                          scale=(4, 4), position=(0, 0), z=0.9)

        self._refresh_hud()

    def _refresh_hud(self):
        self.health_bar1.scale_x = 0.4 * (self.match.p1.health / MAX_HEALTH)
        self.health_bar2.scale_x = 0.4 * (self.match.p2.health / MAX_HEALTH)
        self.health_bar1.color = _health_color(self.match.p1.health)
        self.health_bar2.color = _health_color(self.match.p2.health)
        for i, pip in enumerate(self.pips1):
            pip.color = color.azure if i < self.round_match.round_wins["p1"] else color.dark_gray
        for i, pip in enumerate(self.pips2):
            pip.color = color.orange if i < self.round_match.round_wins["p2"] else color.dark_gray

    def _trigger_impact(self, hitstop: float, shake_duration: float, shake_magnitude: float):
        """Called the instant a hit or KO actually lands -- freezes the game
        (see update()'s hitstop check) and kicks the camera. Fighting games
        sell "weight" mostly through this beat, more than the animation
        itself -- a punch that connects with zero freeze/shake reads as
        limp no matter how good the clip is."""
        self.hitstop_timer = hitstop
        self.shake_timer = shake_duration
        self.shake_duration = shake_duration
        self.shake_magnitude = shake_magnitude

    def _apply_camera_shake(self):
        if _debug_cam is not None and _debug_cam.enabled:
            return  # never fight the user's own free-fly camera controls
        if self.shake_timer > 0:
            self.shake_timer -= time.dt
            frac = max(0.0, self.shake_timer / self.shake_duration)
            jitter = Vec3(random.uniform(-1, 1), random.uniform(-1, 1),
                          random.uniform(-1, 1)) * self.shake_magnitude * frac
            camera.position = Vec3(*BASE_CAMERA_POS) + jitter
        else:
            camera.position = BASE_CAMERA_POS

    def _update_low_health(self):
        """Red screen pulse + heartbeat loop while either player is under
        LOW_HEALTH_FRACTION -- off (and silent) otherwise, and instantly
        cleared the moment a round ends (see the winner block above)."""
        threshold = MAX_HEALTH * LOW_HEALTH_FRACTION
        p1, p2 = self.match.p1, self.match.p2
        low = (0 < p1.health <= threshold) or (0 < p2.health <= threshold)
        if low:
            if not self._low_health_active:
                # edge-triggered -- fires once on the frame health first
                # crosses the threshold, not every frame like the looping
                # heartbeat below.
                sfx.play_critical_health_alert()
                low_side, low_hp = (("p1", p1.health) if 0 < p1.health <= threshold
                                    else ("p2", p2.health))
                self.commentator.notify_low_health(low_side, low_hp)
            # sin pulse, ~2.5 times/second -- fast enough to read as urgent
            # without being a seizure-risk strobe.
            pulse = 0.5 + 0.5 * math.sin(_pytime.time() * 8)
            alpha = int(70 * pulse)
            self.low_health_overlay.color = color.rgba32(255, 0, 0, alpha)
            sfx.start_low_health_alarm()
        elif self._low_health_active:
            self.low_health_overlay.color = color.rgba32(255, 0, 0, 0)
            sfx.stop_low_health_alarm()
        self._low_health_active = low

    def _sync_fighters(self):
        self.fighter1.sync(self.match.p1, self.fighter2.root)
        self.fighter2.sync(self.match.p2, self.fighter1.root)
        if self.rain is not None:
            self.rain.update(time.dt)

    def _update_webcam_previews(self):
        if self.cam_preview1 is not None:
            self.cam_preview1.update(self.input1.latest_frame)
        if self.cam_preview2 is not None:
            self.cam_preview2.update(self.input2.latest_frame)

    def _update_commentary(self):
        # non-blocking poll -- the actual OpenAI round-trips happen entirely
        # off-thread (see commentator.py); this just checks whether a line
        # finished synthesizing since last frame. A commentary line can
        # legitimately arrive a beat or two after the event it's reacting to
        # (real shoutcasters do too) -- it's fire-and-forget, decoupled from
        # hitstop/banner timing, so it's fine for it to land during whatever
        # phase happens to be showing by the time it's ready.
        path = self.commentator.poll()
        if path is not None:
            if self._last_commentary_path is not None:
                # safe to delete now, not right after playback starts --
                # loader.loadSfx() (called inside sfx.play_file) reads the
                # file fully before returning, so by the time poll() hands us
                # a NEW path, the previous file's bytes are already off disk
                # and into panda3d's own buffer.
                self._last_commentary_path.unlink(missing_ok=True)
            sfx.play_file(path)
            self._last_commentary_path = path

    def update(self):
        # runs every real frame no matter what phase/debug-mode we're in --
        # players should always be able to see themselves and check their
        # framing, including during the "ROUND 2" title card or the free-fly
        # debug camera, not just while punches are actually landing.
        self._update_webcam_previews()
        self._update_commentary()

        if _debug_cam is not None and _debug_cam.enabled:
            # free-fly camera mode -- see input()'s 'c'/'p' handling. Skip
            # normal gameplay updates so nothing keeps animating/attacking
            # while you're just lining up a shot.
            _update_debug_text(self)
            return

        self._apply_camera_shake()  # runs every real frame, every phase --
                                     # including through the hitstop freeze
                                     # below -- so the shake still visibly
                                     # plays through a freeze, not paused by it.

        if self.phase == "intro":
            self.phase_timer += time.dt
            if self.phase_timer >= INTRO_DURATION:
                self.phase, self.phase_timer = "go", 0.0
                self.round_banner.text = "FIGHT!"
                self.round_banner.color = color.red
                self.sub_banner.text = ""
                sfx.play("fight")
            self._sync_fighters()
            return

        if self.phase == "go":
            self.phase_timer += time.dt
            if self.phase_timer >= GO_DURATION:
                self.phase, self.phase_timer = "fight", 0.0
                self.round_banner.text = ""
            self._sync_fighters()
            return

        if self.phase == "round_end":
            self.phase_timer += time.dt
            self._sync_fighters()
            if self.phase_timer >= ROUND_END_DURATION:
                if self.round_match.is_match_over():
                    self.phase = "match_end"
                else:
                    self.round_match.start_next_round()
                    self.phase, self.phase_timer = "intro", 0.0
                    self.round_banner.text = _round_label(self.round_match.round_num)
                    self.round_banner.color = color.yellow
                    self.sub_banner.text = ""
                    self._refresh_hud()
                    sfx.play_round_announcement(self.round_match.round_num)
            return

        if self.phase == "match_end":
            self._sync_fighters()
            return

        # phase == "fight" from here on.
        if self.hitstop_timer > 0:
            # true freeze: no input processing, no match.update(), no sync --
            # only scoped to the fight phase (not intro/go/round_end/
            # match_end above) so a trailing hitstop from the finishing hit
            # can never stall the round/match-end banners from advancing.
            # match.py's own timers are wall-clock (time.time()), not frame-
            # based, so skipping a couple frames here doesn't desync damage
            # timing -- it just holds the last rendered frame on screen for
            # this long, which is the whole point.
            self.hitstop_timer -= time.dt
            return

        if self._round_end_pending:
            # hitstop from the finishing hit just finished counting down --
            # transition now, before touching input/match again this frame.
            self._round_end_pending = False
            self.phase, self.phase_timer = "round_end", 0.0
            return

        # normal live gameplay
        def _try(player, action):
            if self.match.try_action(player, action):
                sfx.play(action)  # action is "punch"/"kick"/"shoot" -- same
                                   # names as the sfx clips generated by
                                   # tools/generate_sfx.py

        if self.keyboard_mode:
            for key in list(_pressed_since_last_frame):
                side, action = KEYBOARD_BINDINGS[key]
                player = self.match.p1 if side == "p1" else self.match.p2
                _try(player, action)
            _pressed_since_last_frame.clear()  # one action per physical keypress,
                                                # not one per frame it's held
        else:
            action1 = self.input1.get_action_nowait()
            if action1 is not None:
                _try(self.match.p1, action1[0])
            action2 = self.input2.get_action_nowait()
            if action2 is not None:
                _try(self.match.p2, action2[0])

        health_before = (self.match.p1.health, self.match.p2.health)
        self.match.update()
        # a landed hit shows up as a health drop this frame -- match.py is
        # engine-agnostic (no ursina import) so it can't play sounds itself;
        # comparing health before/after here is how the fight-phase block
        # notices the moment impact_delay actually applies damage.
        if self.match.p1.health < health_before[0] or self.match.p2.health < health_before[1]:
            sfx.play("hit")
            # a crit on EITHER side this frame gets the bigger shake/hitstop --
            # simultaneous-trade crits are rare enough not to bother splitting
            # per-side feedback over it.
            crit_landed = self.match.p1.was_crit_hit or self.match.p2.was_crit_hit
            if crit_landed:
                self._trigger_impact(HITSTOP_DURATION_CRIT, SHAKE_DURATION_CRIT, SHAKE_MAGNITUDE_CRIT)
            else:
                self._trigger_impact(HITSTOP_DURATION_HIT, SHAKE_DURATION_HIT, SHAKE_MAGNITUDE_HIT)
            # whichever side's health just dropped is the defender -- the
            # attacker's current_action is still the move that landed
            # (impact_delay always lands before anim_duration ends, so the
            # attacker hasn't been cleared back to "idle" yet this frame).
            # Both can drop the same frame (a simultaneous trade) -- picking
            # p1-hit first unconditionally would always attribute those to
            # "p2 attacked", silently favoring one side's commentary over a
            # long session. random.choice breaks the tie instead.
            p1_hit = self.match.p1.health < health_before[0]
            p2_hit = self.match.p2.health < health_before[1]
            if p1_hit and p2_hit:
                defender_side = random.choice(("p1", "p2"))
            else:
                defender_side = "p1" if p1_hit else "p2"
            if defender_side == "p1":
                defender, attacker, attacker_side = self.match.p1, self.match.p2, "p2"
            else:
                defender, attacker, attacker_side = self.match.p2, self.match.p1, "p1"
            if attacker.current_action:
                self.commentator.notify_action(attacker_side, attacker.current_action,
                                                attacker.health, defender.health,
                                                is_crit=defender.was_crit_hit)
        self._sync_fighters()
        self._refresh_hud()
        self._update_low_health()

        if self.match.winner is not None and not self._round_end_pending:
            # not self._round_end_pending guards this from re-running every
            # frame during the KO hitstop freeze below -- match.winner stays
            # set from here on, but this block (round scoring, banners, sfx)
            # must fire exactly once. The actual phase transition to
            # "round_end" is deferred to the pending-flag check above, so
            # the freeze/shake from _trigger_impact plays out first instead
            # of the KO banner popping up instantly.
            self._round_end_pending = True
            side = self.round_match.report_round_result()
            winner_name = self.match.winner.name
            sfx.play("ko")
            sfx.play_ko_announcement()
            winner_side = "p1" if self.match.winner is self.match.p1 else "p2"
            self.commentator.notify_ko(winner_side, "p2" if winner_side == "p1" else "p1")
            self._trigger_impact(HITSTOP_DURATION_KO, SHAKE_DURATION_KO, SHAKE_MAGNITUDE_KO)
            if self.round_match.is_match_over():
                self.round_banner.text = "K.O.!"
                self.round_banner.color = color.red
                self.sub_banner.text = f"{winner_name.upper()} WINS THE MATCH"
                self.hint_text.text = "R to rematch -- ESC/q to quit"
                sfx.play("match_win")
            else:
                self.round_banner.text = "K.O.!"
                self.round_banner.color = color.red
                self.sub_banner.text = (f"{winner_name.upper()} WINS ROUND "
                                         f"{self.round_match.round_num}")
            self._refresh_hud()
            self.low_health_overlay.color = color.rgba32(255, 0, 0, 0)
            sfx.stop_low_health_alarm()
            self._low_health_active = False

    def restart(self):
        self.round_match = RoundMatch("Player 1", "Player 2")
        self.phase, self.phase_timer = "intro", 0.0
        self.round_banner.text = "ROUND 1"
        self.round_banner.color = color.yellow
        self.sub_banner.text = ""
        self.hint_text.text = "R to restart -- ESC/q to quit -- C for free camera"
        # force both fighters back to a fresh Idle pose -- sync()'s "very
        # first frame" branch only fires when _current_clip is None.
        self.fighter1._current_clip = None
        self.fighter2._current_clip = None
        self.low_health_overlay.color = color.rgba32(255, 0, 0, 0)
        sfx.stop_low_health_alarm()
        self._low_health_active = False
        self.hitstop_timer = 0.0
        self.shake_timer = 0.0
        self._round_end_pending = False
        camera.position = BASE_CAMERA_POS
        self._refresh_hud()
        sfx.play_round_announcement(self.round_match.round_num)


_pressed_since_last_frame = set()  # populated by the module-level input() below --
                                    # one physical keypress per entry, drained by
                                    # Game.update() each frame so a held key doesn't
                                    # repeat-fire every frame


def _round_label(round_num: int) -> str:
    # with ROUNDS_TO_WIN=2, round 3 only ever happens on a 1-1 split -- call
    # it out as the decider instead of just "ROUND 3".
    if round_num >= 2 * ROUNDS_TO_WIN - 1:
        return f"ROUND {round_num} -- FINAL ROUND"
    return f"ROUND {round_num}"


def _health_color(health: int):
    frac = health / MAX_HEALTH
    if frac > 0.5:
        return color.lime
    if frac > 0.2:
        return color.yellow
    return color.red


_game = None  # set once character select confirms -- ursina discovers
              # update()/input() below by scanning __main__'s module-level
              # globals, so they can't be nested inside main() as local
              # closures; a module global holding the Game instance is the
              # correct way to give them access to it. Stays None while the
              # main menu / character select screens are up, which is also
              # what gates match logic out of update()/input() below.
_lava = None  # set by main() -- animates the arena's lava (src/game/lava_flow.py)
_menu = None  # set by main() -- MainMenu screen, live until Start is clicked
_char_select = None  # set after Start -- CharacterSelect screen, live until confirmed
_preview_actors = {}  # model_key -> RealFighterEntity, live only during
                       # character select -- same arena spot the fighters
                       # fight at, idling + slowly turning like a showcase
                       # pedestal instead of standing frozen

# Debug free-fly camera (press C to toggle, P to print a snapshot) -- lets
# you line up the exact shot you want by hand (right-mouse-drag to rotate,
# WASD to move, scroll to zoom -- ursina's stock EditorCamera controls) and
# read back the real numbers instead of me guessing camera.position/
# rotation_x from screenshots and getting it wrong. Also prints both
# fighters' feet height vs. where the arena floor actually is, for the
# "standing in mid-air" report -- paste the console output back to me.
_debug_cam = None
_debug_text = None


def _update_debug_text(game):
    # world_position/world_rotation, NOT camera.position/camera.rotation --
    # while EditorCamera is active it parents `camera` under its own rig
    # entity, so camera.position/rotation read as LOCAL offsets within that
    # rig (this bit us: an earlier snapshot showed rotation=(0,0,0) simply
    # because EditorCamera resets that local value on enable, not because
    # the camera was actually facing forward). world_position/world_rotation
    # resolve through the parent chain to the real transform, which is what
    # actually needs to go into the fixed gameplay camera.position/rotation_x.
    lines = [
        f"[DEBUG CAM] hold RIGHT-MOUSE + WASD to move, E/Q = up/down, scroll = zoom",
        f"C=toggle off  P=print snapshot to console",
        f"camera.world_position = {tuple(round(v, 3) for v in camera.world_position)}",
        f"camera.world_rotation = {tuple(round(v, 3) for v in camera.world_rotation)}",
    ]
    if game is not None:
        lines.append(f"fighter1 feet Y = {game.fighter1.root.getY(scene):.3f}")
        lines.append(f"fighter2 feet Y = {game.fighter2.root.getY(scene):.3f}")
    _debug_text.text = "\n".join(lines)


def _print_debug_snapshot(game):
    print("=== camera/position snapshot ===")
    print(f"camera.world_position = {tuple(camera.world_position)}")
    print(f"camera.world_rotation = {tuple(camera.world_rotation)}")
    if game is not None:
        print(f"fighter1 ({P1_MODEL}) feet Y = {game.fighter1.root.getY(scene):.4f}, "
              f"X = {game.fighter1.root.getX(scene):.4f}")
        print(f"fighter2 ({P2_MODEL}) feet Y = {game.fighter2.root.getY(scene):.4f}, "
              f"X = {game.fighter2.root.getX(scene):.4f}")
    print("=================================")


def update():
    if _game is not None:
        _game.update()
    if _lava is not None:
        # keeps flowing even while the debug free-cam has paused match
        # logic -- it's ambient scenery, not gameplay state.
        _lava.update(time.dt)
    # preview actors (character select) are static -- see spawn_previews()
    # in main(): pose()'d once to a fixed frame, nothing to tick per-frame.


def input(key):
    debug_active = _debug_cam is not None and _debug_cam.enabled
    if key == "escape" or (key == "q" and not debug_active):
        # 'q' quits normally, but EditorCamera's own controls use q/e for
        # down/up while free-flying (held together with right-mouse-drag) --
        # this was silently quitting the whole game every time you tried to
        # go down in free-cam mode. Only escape quits while it's active.
        application.quit()
        return
    if key == "c" and _debug_cam is not None:
        _debug_cam.enabled = not _debug_cam.enabled
        _debug_text.enabled = _debug_cam.enabled
        return
    if key == "p" and debug_active:
        _print_debug_snapshot(_game)
        return
    if key == "r":
        if _game is not None:
            _game.restart()
        return
    if key.endswith(" up"):
        _pressed_since_last_frame.discard(key[:-len(" up")])
    elif key in KEYBOARD_BINDINGS:
        _pressed_since_last_frame.add(key)


def main():
    # BUG FIX: this was missing `_menu`/`_char_select` -- main()'s own
    # bottom-of-function `_menu = MainMenu(...)` was creating a LOCAL
    # variable shadowing the module global, invisible to the nested
    # show_char_select()/back_to_menu() closures below (they each declare
    # their OWN `global _menu`, which makes Python read/write the module
    # global exclusively inside them, not this function's local one). Net
    # effect: pressing Start called destroy(_menu) on the module global,
    # which was still None -- a silent no-op -- so the real on-screen
    # MainMenu never actually got destroyed and sat there overlapping the
    # new CharacterSelect screen underneath it.
    global _game, _debug_cam, _debug_text, _lava, _menu, _char_select
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyboard", action="store_true",
                     help="no cameras -- keyboard-controlled test mode (1/2/3 = "
                          "p1 punch/kick/shoot, 8/9/0 = p2 punch/kick/shoot)")
    ap.add_argument("--camera1", type=int, default=0, help="player 1's webcam device index")
    ap.add_argument("--camera2", type=int, default=1, help="player 2's webcam device index")
    ap.add_argument("--procedural", action="store_true",
                     help="use the old low-poly primitive rig instead of the real "
                          "Mixamo character models (fallback/debug option)")
    args = ap.parse_args()

    app = Ursina(fullscreen=True)

    sfx.start_music()  # loops for the whole session -- menu through fight

    if not args.procedural:
        # glTF PBR materials (the real character models) render unlit/black
        # without this -- panda3d-simplepbr is already an ursina dependency,
        # just never initialized until now since the procedural rig's flat
        # ursina colors didn't need it.
        import simplepbr
        simplepbr.init()

        # PBR needs actual light sources in the scene to shade anything --
        # ursina's default flat-color Entities didn't need this either.
        # Colors nudged warm (vs. neutral white) to match the lava arena's
        # glow -- confirmed via tools/screenshot_arena_full.py renders.
        ambient = AmbientLight("ambient")
        ambient.setColor((0.4, 0.35, 0.35, 1))
        ambient_np = scene.attachNewNode(ambient)
        scene.setLight(ambient_np)

        sun = DirectionalLight("sun")
        sun.setColor((1.0, 0.85, 0.7, 1))
        sun_np = scene.attachNewNode(sun)
        sun_np.setHpr(30, -60, 0)
        scene.setLight(sun_np)

    # Background image (assets/Background/arena_backdrop.png, converted from
    # the user's jpg -- Panda3D's texture loader errored on the .jpg outright)
    # replaces Sky(): Sky() wraps its texture around a full spherical dome,
    # which badly distorts a plain wide landscape image (it's not a 360
    # panorama) -- confirmed via tools/test_sky_bg.py, showed as a tiny
    # unrecognizable fragment. Instead this is the standard Panda3D
    # background-card trick: a flat quad parented to the camera, in the
    # 'background' render bin with depth WRITE off but depth TEST on, so it
    # always sits behind real 3D geometry (the arena) via normal depth
    # comparison, filling only the empty sky gap. setShaderOff/setLightOff
    # keep simplepbr's PBR pipeline from recoloring it -- confirmed via
    # tools/test_backdrop.py that without those two calls it rendered
    # visibly pink-tinted instead of its real orange/red.
    bg_tex = load_texture("arena_backdrop.png", path=r"C:\Data_Tekken\assets\Background")
    backdrop = Entity(parent=camera, model="quad", texture=bg_tex, unlit=True,
                       position=(0, 0, 80), scale=(200, 75))
    backdrop.setShaderOff(1)
    backdrop.setLightOff(1)
    backdrop.setBin("background", 0)
    backdrop.setDepthWrite(False)

    if args.procedural:
        Entity(model="plane", scale=20, color=color.dark_gray, texture="white_cube",
               texture_scale=(20, 20))
        camera.position = (0, 2.2, -10)
        camera.rotation_x = 8
    else:
        # assets/arena/source/Lava_Stage.fbx -> tools/convert_arena.py ->
        # assets/models/arena.glb. Scale/offset/camera numbers were measured
        # against the real model, not guessed -- see the comment by
        # ARENA_MODEL_PATH above and tools/screenshot_arena_full.py.
        arena = loader.loadModel(Filename.fromOsSpecific(ARENA_MODEL_PATH))
        arena.reparentTo(scene)
        arena.setScale(ARENA_SCALE)
        arena.setY(-ARENA_FLOOR_LOCAL_Y * ARENA_SCALE)
        # exact shot the user picked by hand with the free debug camera (C/P
        # in-game) -- assets/coordinates2.png, read via world_position/
        # world_rotation (not camera.position/rotation, which are local to
        # EditorCamera's rig while it's active and would be wrong here).
        camera.position = BASE_CAMERA_POS
        camera.rotation = BASE_CAMERA_ROT

        # continuous lava flow -- see src/game/lava_flow.py for why this is
        # floor-only UV scroll + arena-wide emissive pulse, not a full-arena
        # scroll (that broke the mountain walls' UVs, confirmed via
        # tools/test_lava_flow.py).
        _lava = LavaFlow(arena, arena.find("**/lower"))

    # Debug free-fly camera -- press C in-game to toggle, P to print the
    # current camera position/rotation + both fighters' feet height to the
    # console. See the module-level comment above _debug_cam for controls.
    _debug_cam = EditorCamera(enabled=False)
    _debug_text = Text(position=(-0.87, 0.3), scale=0.8, color=color.yellow, enabled=False)

    # Full-screen dark overlay, reused across the menu and character-select
    # screens at different opacities (darkest = empty moody arena behind the
    # main menu, lighter = character select so the live preview models read
    # clearly, off entirely during the actual fight). Same camera framing is
    # kept for all 3 screens on purpose -- one continuous shot of the arena,
    # not a jump-cut -- so what makes each screen its own "page" is what's
    # standing in it (nobody / 2 idling showcase models / the live match)
    # plus this overlay, not the camera moving around.
    _vignette = Entity(parent=camera.ui, model="quad", color=color.rgba32(0, 0, 0, 195),
                        scale=(4, 4), position=(0, 0), z=1)

    def spawn_previews():
        # only the real Mixamo renderer has swappable models -- the
        # procedural rig is a single fixed low-poly build, nothing to show
        # off on a pedestal.
        if args.procedural:
            return
        for key, x in (("warrok", -ARENA_HALF_WIDTH), ("vampire", ARENA_HALF_WIDTH)):
            preview = RealFighterEntity(key, x=x, facing=1, parent=scene)
            # standing still on purpose -- Idle's LAST frame (the settled
            # guard stance), same pose real_entities.py snaps Shoot back to.
            # pose(), not loop()/play(): no animation plays at all, no
            # per-frame rotation either (see module-level update() below --
            # it no longer touches preview actors).
            last_frame = preview.actor.getAnimControl("Idle").getNumFrames() - 1
            preview.actor.pose("Idle", last_frame)
            _preview_actors[key] = preview

    def despawn_previews():
        for preview in _preview_actors.values():
            preview.actor.cleanup()
            preview.actor.removeNode()
        _preview_actors.clear()

    def highlight_pick(chosen_key: str, other_key: str):
        # picking a fighter shows it off: brighten it and throw a real Punch
        # clip (play(), not loop() -- plays once and holds its last frame,
        # same "no auto-return-to-idle" rule as in-match attacks in
        # real_entities.py). The other card snaps back to its static
        # guard-stance pose in case it was mid-punch from a previous pick.
        for key, preview in _preview_actors.items():
            if key == chosen_key:
                preview.actor.setColorScale(1.25, 1.25, 1.1, 1)
                preview.actor.play("Punch")
            else:
                preview.actor.setColorScale(1, 1, 1, 1)
                last_frame = preview.actor.getAnimControl("Idle").getNumFrames() - 1
                preview.actor.pose("Idle", last_frame)

    def start_match(model1: str, model2: str):
        global _game, _char_select
        despawn_previews()
        destroy(_char_select)
        _char_select = None
        _vignette.enabled = False
        _game = Game(keyboard_mode=args.keyboard, camera1=args.camera1, camera2=args.camera2,
                     procedural=args.procedural, model1=model1, model2=model2)

    def back_to_menu():
        global _menu, _char_select
        despawn_previews()
        destroy(_char_select)
        _char_select = None
        _vignette.color = color.rgba32(0, 0, 0, 195)
        _menu = MainMenu(on_start=show_char_select, on_quit=application.quit)

    def show_char_select():
        global _menu, _char_select
        destroy(_menu)
        _menu = None
        _vignette.color = color.rgba32(0, 0, 0, 90)
        spawn_previews()
        _char_select = CharacterSelect(on_pick=highlight_pick, on_confirm=start_match,
                                        on_back=back_to_menu)

    _menu = MainMenu(on_start=show_char_select, on_quit=application.quit)

    app.run()


if __name__ == "__main__":
    main()
