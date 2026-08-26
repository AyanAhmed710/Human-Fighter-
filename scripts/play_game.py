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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# mediapipe MUST be imported before ursina -- ursina/panda3d ships its own
# native DLLs that shadow one mediapipe's compiled bindings need on Windows;
# importing in this order avoids "DLL load failed while importing
# _framework_bindings" (reproducible, confirmed by import-order swap test).
import mediapipe  # noqa: F401  (import order side effect only, not used directly here)

from panda3d.core import AmbientLight, DirectionalLight
from ursina import Entity, Sky, Text, Ursina, application, camera, color, scene

from src.game.entities import FighterEntity
from src.game.match import MAX_HEALTH, Match
from src.game.player_input import PlayerCameraInput
from src.game.real_entities import RealFighterEntity

ARENA_HALF_WIDTH = 3.0
P1_MODEL = "warrok"
P2_MODEL = "vampire"

KEYBOARD_BINDINGS = {
    "1": ("p1", "punch"), "2": ("p1", "kick"), "3": ("p1", "shoot"),
    "8": ("p2", "punch"), "9": ("p2", "kick"), "0": ("p2", "shoot"),
}


class Game:
    def __init__(self, keyboard_mode: bool, camera1: int, camera2: int, procedural: bool):
        self.keyboard_mode = keyboard_mode
        self.match = Match("Player 1", "Player 2")

        if procedural:
            self.fighter1 = FighterEntity(x=-ARENA_HALF_WIDTH, facing=1, side_key="p1")
            self.fighter2 = FighterEntity(x=ARENA_HALF_WIDTH, facing=-1, side_key="p2")
        else:
            self.fighter1 = RealFighterEntity(P1_MODEL, x=-ARENA_HALF_WIDTH, facing=1, parent=scene)
            self.fighter2 = RealFighterEntity(P2_MODEL, x=ARENA_HALF_WIDTH, facing=-1, parent=scene)

        self.input1 = self.input2 = None
        if not keyboard_mode:
            self.input1 = PlayerCameraInput(camera_index=camera1).start()
            self.input2 = PlayerCameraInput(camera_index=camera2).start()

        self._build_ui()

    def _build_ui(self):
        self.name_text1 = Text("PLAYER 1", position=(-0.85, 0.47), scale=1.3, color=color.azure)
        self.name_text2 = Text("PLAYER 2", position=(0.55, 0.47), scale=1.3, color=color.orange)

        bar_back1 = Entity(parent=camera.ui, model="quad", color=color.dark_gray,
                            scale=(0.4, 0.045), position=(-0.6, 0.43), origin=(-0.5, 0))
        bar_back2 = Entity(parent=camera.ui, model="quad", color=color.dark_gray,
                            scale=(0.4, 0.045), position=(0.2, 0.43), origin=(-0.5, 0))
        self.health_bar1 = Entity(parent=camera.ui, model="quad", color=color.lime,
                                   scale=(0.4, 0.045), position=(-0.6, 0.43), origin=(-0.5, 0))
        self.health_bar2 = Entity(parent=camera.ui, model="quad", color=color.lime,
                                   scale=(0.4, 0.045), position=(0.2, 0.43), origin=(-0.5, 0))

        self.status_text = Text("", position=(-0.15, 0.0), scale=2.5, color=color.white,
                                 origin=(0, 0))
        self.hint_text = Text("R to restart -- ESC/q to quit", position=(-0.2, -0.47), scale=0.8,
                               color=color.gray)

    def update(self):
        if self.keyboard_mode:
            for key in list(_pressed_since_last_frame):
                side, action = KEYBOARD_BINDINGS[key]
                player = self.match.p1 if side == "p1" else self.match.p2
                self.match.try_action(player, action)
            _pressed_since_last_frame.clear()  # one action per physical keypress,
                                                # not one per frame it's held
        else:
            action1 = self.input1.get_action_nowait()
            if action1 is not None:
                self.match.try_action(self.match.p1, action1[0])
            action2 = self.input2.get_action_nowait()
            if action2 is not None:
                self.match.try_action(self.match.p2, action2[0])

        self.match.update()

        self.fighter1.sync(self.match.p1, self.fighter2.root)
        self.fighter2.sync(self.match.p2, self.fighter1.root)

        self.health_bar1.scale_x = 0.4 * (self.match.p1.health / MAX_HEALTH)
        self.health_bar2.scale_x = 0.4 * (self.match.p2.health / MAX_HEALTH)
        self.health_bar1.color = _health_color(self.match.p1.health)
        self.health_bar2.color = _health_color(self.match.p2.health)

        if self.match.winner is not None:
            self.status_text.text = f"{self.match.winner.name} WINS"
        else:
            self.status_text.text = ""

    def restart(self):
        self.match = Match("Player 1", "Player 2")


_pressed_since_last_frame = set()  # populated by the module-level input() below --
                                    # one physical keypress per entry, drained by
                                    # Game.update() each frame so a held key doesn't
                                    # repeat-fire every frame


def _health_color(health: int):
    frac = health / MAX_HEALTH
    if frac > 0.5:
        return color.lime
    if frac > 0.2:
        return color.yellow
    return color.red


_game = None  # set by main() -- ursina discovers update()/input() below by
              # scanning __main__'s module-level globals, so they can't be
              # nested inside main() as local closures; a module global
              # holding the Game instance is the correct way to give them
              # access to it


def update():
    if _game is not None:
        _game.update()


def input(key):
    if key in ("escape", "q"):
        application.quit()
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
    global _game
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

    app = Ursina()

    if not args.procedural:
        # glTF PBR materials (the real character models) render unlit/black
        # without this -- panda3d-simplepbr is already an ursina dependency,
        # just never initialized until now since the procedural rig's flat
        # ursina colors didn't need it.
        import simplepbr
        simplepbr.init()

        # PBR needs actual light sources in the scene to shade anything --
        # ursina's default flat-color Entities didn't need this either.
        ambient = AmbientLight("ambient")
        ambient.setColor((0.35, 0.35, 0.4, 1))
        ambient_np = scene.attachNewNode(ambient)
        scene.setLight(ambient_np)

        sun = DirectionalLight("sun")
        sun.setColor((1.0, 0.97, 0.9, 1))
        sun_np = scene.attachNewNode(sun)
        sun_np.setHpr(30, -60, 0)
        scene.setLight(sun_np)

    Sky()
    Entity(model="plane", scale=20, color=color.dark_gray, texture="white_cube",
           texture_scale=(20, 20))
    camera.position = (0, 2.2, -10)
    camera.rotation_x = 8

    _game = Game(keyboard_mode=args.keyboard, camera1=args.camera1, camera2=args.camera2,
                 procedural=args.procedural)

    app.run()


if __name__ == "__main__":
    main()
