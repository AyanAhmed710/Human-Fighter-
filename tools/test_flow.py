"""Headless smoke test for the whole menu -> character select -> fight ->
round-end -> match-end -> restart flow. Drives the same functions the UI
buttons call (no real mouse), screenshotting at each stage, and force-KOs
both rounds via match.p1/p2.health so the full round machine actually runs
end to end instead of just booting.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mediapipe  # noqa: F401
import simplepbr
from panda3d.core import AmbientLight, DirectionalLight, Filename
from ursina import Entity, Text, Ursina, camera, color, destroy, load_texture, scene

from src.game.lava_flow import LavaFlow
from src.game.menu import CharacterSelect, MainMenu
from src.game.real_entities import RealFighterEntity

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
pg = importlib.import_module("scripts.play_game")

ARENA_MODEL_PATH = pg.ARENA_MODEL_PATH
ARENA_FLOOR_LOCAL_Y = pg.ARENA_FLOOR_LOCAL_Y
ARENA_SCALE = pg.ARENA_SCALE

app = Ursina()
simplepbr.init()
ambient = AmbientLight("ambient")
ambient.setColor((0.4, 0.35, 0.35, 1))
scene.setLight(scene.attachNewNode(ambient))
sun = DirectionalLight("sun")
sun.setColor((1.0, 0.85, 0.7, 1))
sun_np = scene.attachNewNode(sun)
sun_np.setHpr(30, -60, 0)
scene.setLight(sun_np)

arena = loader.loadModel(Filename.fromOsSpecific(ARENA_MODEL_PATH))  # noqa: F821
arena.reparentTo(scene)
arena.setScale(ARENA_SCALE)
arena.setY(-ARENA_FLOOR_LOCAL_Y * ARENA_SCALE)
camera.position = (4.95, 6.213, 16.094)
camera.rotation = (15.625, -161.805, -0.0)
pg._lava = LavaFlow(arena, arena.find("**/lower"))


def snap(name):
    for _ in range(5):
        taskMgr.step()  # noqa: F821
    result = base.win.saveScreenshot(  # noqa: F821
        Filename.fromOsSpecific(rf"C:\Data_Tekken\tools\flow_{name}.png"))
    print(f"[{name}] screenshot:", result)


# --- stage 1: main menu ---
menu = MainMenu(on_start=lambda: None, on_quit=lambda: None)
snap("1_menu")
destroy(menu)

# --- stage 2: character select w/ live previews ---
previews = {}
for key, x in (("warrok", -1.0), ("vampire", 1.0)):
    p = RealFighterEntity(key, x=x, facing=1, parent=scene)
    p.actor.loop("Idle")
    previews[key] = p
for _ in range(10):
    taskMgr.step()  # noqa: F821
    for p in previews.values():
        p.actor.setH(p.actor.getH() + 0.3)

picked = {}
def on_pick(chosen, other):
    picked["chosen"], picked["other"] = chosen, other
    for key, p in previews.items():
        p.actor.setColorScale((1.25, 1.25, 1.1, 1) if key == chosen else (1, 1, 1, 1))

confirmed = {}
def on_confirm(m1, m2):
    confirmed["m1"], confirmed["m2"] = m1, m2

select = CharacterSelect(on_pick=on_pick, on_confirm=on_confirm, on_back=lambda: None)
snap("2_select_before_pick")

select.pick("warrok")
snap("3_select_after_pick")
print("on_pick fired:", picked)

select._confirm()
print("on_confirm fired:", confirmed)
assert confirmed == {"m1": "warrok", "m2": "vampire"}, "Fight! button logic broken"

for p in previews.values():
    p.actor.cleanup()
    p.actor.removeNode()
destroy(select)

# --- stage 3: real Game, force through 2 rounds ---
# Phase transitions are gated by game.phase_timer vs real time.dt, which
# won't accumulate meaningfully across back-to-back update() calls with no
# real engine frame between them -- so this test skips time by setting
# phase_timer directly past each threshold instead of looping hundreds of
# real frames (deterministic, and avoids exercising unrelated real-time
# timing flakiness in a smoke test).
game = pg.Game(keyboard_mode=True, camera1=0, camera2=1, procedural=False,
               model1="warrok", model2="vampire")
pg._game = game
snap("4_fight_round1_intro")
assert game.phase == "intro" and game.round_banner.text == "ROUND 1"

game.phase_timer = pg.INTRO_DURATION + 0.01
game.update()
assert game.phase == "go" and game.round_banner.text == "FIGHT!", game.phase
snap("5_fight_go_flash")

game.phase_timer = pg.GO_DURATION + 0.01
game.update()
assert game.phase == "fight" and game.round_banner.text == "", game.phase
snap("6_fight_live")

# force a KO to end round 1
game.match.p1.health = 0
game.match.p1.state = "ko"
game.match.p1.state_until = 1e18
game.update()
assert game.phase == "fight" and game._round_end_pending, game.phase
# KO now defers the phase transition until the hitstop freeze finishes (see
# Game.update()'s _round_end_pending) -- skip the wait deterministically.
game.hitstop_timer = 0.0
game.update()
assert game.phase == "round_end", game.phase
print("round 1 winner:", game.round_match.round_winner().name)
print("phase:", game.phase, "round_wins:", game.round_match.round_wins,
      "banner:", repr(game.round_banner.text), repr(game.sub_banner.text))
snap("7_round1_end_banner")

game.phase_timer = pg.ROUND_END_DURATION + 0.01
game.update()
assert game.phase == "intro" and game.round_match.round_num == 2, \
    (game.phase, game.round_match.round_num)
print("phase after round 1 end:", game.phase, "round_num:", game.round_match.round_num)
snap("8_round2_intro")

game.phase_timer = pg.INTRO_DURATION + 0.01
game.update()
game.phase_timer = pg.GO_DURATION + 0.01
game.update()
assert game.phase == "fight", game.phase

# force a 2nd KO for the same loser (p1) -> p2 clinches the match 2-0
game.match.p1.health = 0
game.match.p1.state = "ko"
game.match.p1.state_until = 1e18
game.update()
game.hitstop_timer = 0.0  # see the round-1 KO comment above
game.update()
assert game.phase == "round_end" and game.round_match.match_winner == "p2", \
    (game.phase, game.round_match.match_winner)
print("match_winner:", game.round_match.match_winner, "phase:", game.phase,
      "banner:", repr(game.round_banner.text), repr(game.sub_banner.text))
snap("9_round2_end_banner")

game.phase_timer = pg.ROUND_END_DURATION + 0.01
game.update()
assert game.phase == "match_end", game.phase
print("final phase (match over):", game.phase)
snap("10_match_end_banner")

game.restart()
assert game.phase == "intro" and game.round_match.round_wins == {"p1": 0, "p2": 0}
print("after restart -- phase:", game.phase, "round_wins:", game.round_match.round_wins)
snap("11_after_restart")

print("ALL STAGES COMPLETED WITHOUT EXCEPTION")
