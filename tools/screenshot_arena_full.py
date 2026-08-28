"""Full test: real arena (scaled+positioned) + both fighters + camera, so
the framing can actually be judged from a screenshot instead of guessed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import simplepbr
from panda3d.core import AmbientLight, DirectionalLight, Filename
from ursina import Sky, Ursina, camera, scene

from src.game.real_entities import RealFighterEntity
from src.game.match import Match
from src.game.rain import LAVA_RAIN_KWARGS, RainEffect

# measured via tools/measure_floor.py doing an actual triangle/ray test (not
# a bounding box, not vertex-proximity -- both of those were misleading: a
# nearby vertex on the outcrop's peak isn't necessarily the triangle that's
# actually overhead at the fighter's XZ). This is the real 'base' mesh
# surface Y directly under (x=+-1, z=0), confirmed to land at exactly 0.0.
ARENA_FLOOR_LOCAL_Y = 0.072985
ARENA_SCALE = 20

app = Ursina()
simplepbr.init()

ambient = AmbientLight("ambient")
ambient.setColor((0.4, 0.35, 0.35, 1))  # slightly warm, lava-lit feel
scene.setLight(scene.attachNewNode(ambient))
sun = DirectionalLight("sun")
sun.setColor((1.0, 0.85, 0.7, 1))
sun_np = scene.attachNewNode(sun)
sun_np.setHpr(30, -60, 0)
scene.setLight(sun_np)

Sky()

arena_path = Filename.fromOsSpecific(r"C:\Data_Tekken\assets\models\arena.glb")
arena = loader.loadModel(arena_path)  # noqa: F821
arena.reparentTo(scene)
arena.setScale(ARENA_SCALE)
arena.setY(-ARENA_FLOOR_LOCAL_Y * ARENA_SCALE)

match = Match("Player 1", "Player 2")
fighter1 = RealFighterEntity("warrok", x=-1.0, facing=1, parent=scene)
fighter2 = RealFighterEntity("vampire", x=1.0, facing=-1, parent=scene)
fighter1.sync(match.p1, fighter2.root)
fighter2.sync(match.p2, fighter1.root)

camera.position = (0, 9.0, -18)
camera.rotation_x = 26

rain = RainEffect(area_size=4, height=8, parent=scene, **LAVA_RAIN_KWARGS)
for _ in range(60):
    rain.update(1 / 60)
    taskMgr.step()  # noqa: F821

fmin, fmax = fighter1.root.getTightBounds()
print("fighter1 bounds:", fmin, fmax, "height:", fmax[1] - fmin[1])
amin, amax = arena.getTightBounds()
print("arena (scaled+offset) bounds:", amin, amax, "size:", amax - amin)
outer = arena.find("**/Outer")
omin, omax = outer.getTightBounds()
print("Outer bounds:", omin, omax)

out_path = Filename.fromOsSpecific(r"C:\Data_Tekken\tools\screenshot_arena_full.png")
result = base.win.saveScreenshot(out_path)  # noqa: F821
print("saveScreenshot result:", result)
