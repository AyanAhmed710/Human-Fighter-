import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import simplepbr
from panda3d.core import AmbientLight, DirectionalLight, Filename
from ursina import Entity, Sky, Ursina, camera, color, scene

from src.game.match import Match
from src.game.real_entities import RealFighterEntity

app = Ursina()
simplepbr.init()

ambient = AmbientLight("ambient")
ambient.setColor((0.35, 0.35, 0.4, 1))
scene.setLight(scene.attachNewNode(ambient))
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

for _ in range(20):
    taskMgr.step()

action = sys.argv[1] if len(sys.argv) > 1 else "kick"
match.try_action(match.p2, action)  # test on p2 (Vampire) this time
fighter2.sync(match.p2, fighter1.root)
target_frame = 30 if action == "kick" else 18
while fighter2.actor.getAnimControl(action.capitalize()).getFrame() < target_frame and \
        fighter2.actor.getAnimControl(action.capitalize()).isPlaying():
    taskMgr.step()

print(f"{action} frame reached:", fighter2.actor.getAnimControl(action.capitalize()).getFrame())

out_path = Filename.fromOsSpecific(rf"C:\Data_Tekken\tools\screenshot_{action}.png")
result = base.win.saveScreenshot(out_path)
print(f"saveScreenshot result: {result}")
