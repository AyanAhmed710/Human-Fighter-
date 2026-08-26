"""
Offscreen render of the actual play_game.py character setup (same ground
plane, same camera position, same RealFighterEntity construction/idle pose)
so the result can be inspected as an image instead of guessed at blind.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import simplepbr
from panda3d.core import AmbientLight, DirectionalLight
from ursina import Entity, Sky, Ursina, camera, color, scene

from src.game.real_entities import RealFighterEntity
from src.game.match import Match

app = Ursina()
simplepbr.init()

ambient = AmbientLight("ambient")
ambient.setColor((0.35, 0.35, 0.4, 1))
scene.attachNewNode(ambient)
ambient_np = scene.attachNewNode(ambient)
scene.setLight(ambient_np)

sun = DirectionalLight("sun")
sun.setColor((1.0, 0.97, 0.9, 1))
sun_np = scene.attachNewNode(sun)
sun_np.setHpr(30, -60, 0)
scene.setLight(sun_np)

Sky()
Entity(model="plane", scale=20, color=color.dark_gray, texture="white_cube", texture_scale=(20, 20))
camera.position = (0, 2.2, -10)
camera.rotation_x = 8

match = Match("Player 1", "Player 2")
fighter1 = RealFighterEntity("warrok", x=-3.0, facing=1, parent=scene)
fighter2 = RealFighterEntity("vampire", x=3.0, facing=-1, parent=scene)
fighter1.sync(match.p1, fighter2.root)
fighter2.sync(match.p2, fighter1.root)

# Settle on Idle BEFORE triggering the punch -- Panda3D's AnimControl clock
# is wall-time based, and the very first taskMgr.step() right after loading
# 8 large model files carries a huge real-elapsed dt, which single-steps the
# animation straight through to its end (confirmed: reached frame 38 doing
# this the naive way). The real game never hits this -- it's already
# running its main loop steadily, dt already small/stable, long before you
# ever press a key. This settle phase reproduces that same steady-state.
for _ in range(20):
    taskMgr.step()

match.try_action(match.p1, "punch")
fighter1.sync(match.p1, fighter2.root)
while fighter1.actor.getAnimControl("Punch").getFrame() < 20 and \
        fighter1.actor.getAnimControl("Punch").isPlaying():
    # taskMgr.step() -- not a raw graphicsEngine.renderFrame() call (tried
    # that first): it skips Panda3D's task manager entirely, which is what
    # actually runs simplepbr's per-frame shader-input updates (camera
    # position etc.), causing "Shader input camera_world_position is not
    # present". taskMgr.step() runs one full task-manager pass (including
    # rendering). `base`/`taskMgr` are injected into builtins by Panda3D's
    # ShowBase.__init__ (Ursina subclasses it), not imports.
    taskMgr.step()  # noqa: F821

print("final Punch frame reached:", fighter1.actor.getAnimControl("Punch").getFrame())
print("final Punch isPlaying:", fighter1.actor.getAnimControl("Punch").isPlaying())
print("match.p1.state:", match.p1.state, "current_action:", match.p1.current_action)

from panda3d.core import Filename
out_path = Filename.fromOsSpecific(r"C:\Data_Tekken\tools\screenshot_idle.png")
result = base.win.saveScreenshot(out_path)
print(f"saveScreenshot result: {result}  (True/valid Filename = success)")
