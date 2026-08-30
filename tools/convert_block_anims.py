"""Adds the block-stance clips (idle-to-fight transition + the bouncy guard
loop) to already-converted characters, without re-touching the existing
(verified-good) base/Punch/Kick/Shoot/HitReact .glb files -- same anim-only
export pattern as convert_hit_reaction.py, just parameterized over which
source FBX/action-name/output-suffix to use, since this needs to run 4 times
(2 clips x 2 characters) instead of once.

Run via:
    blender.exe --background --python convert_block_anims.py -- <character.fbx> <out_prefix>

Outputs <out_prefix>_IdleToFight.glb and <out_prefix>_Block.glb -- skeleton +
action only, no mesh (see convert_hit_reaction.py's docstring for why: an
anim file that includes the mesh binds the clip to its own separate
Character instance instead of the modelRoot's, so it plays but never
visibly moves the actual rendered character).
"""
import sys
from pathlib import Path

import bpy

ROOT = Path(r"C:\Data_Tekken")
# (source FBX, action name to give it, output filename suffix)
CLIPS = [
    (ROOT / "assets/mixamo/Standing Idle To Fight Idle.fbx", "IdleToFight", "IdleToFight"),
    (ROOT / "assets/mixamo/Bouncing Fight Idle.fbx", "Block", "Block"),
]

argv = sys.argv[sys.argv.index("--") + 1:]
character_fbx, out_prefix = argv[0], argv[1]


def export_clip(clip_fbx: Path, action_name: str, out_suffix: str):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=character_fbx)
    char_armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")

    icosphere = bpy.data.objects.get("Icosphere")
    if icosphere is not None:
        bpy.data.objects.remove(icosphere, do_unlink=True)

    before_objs = set(bpy.data.objects.keys())
    bpy.ops.import_scene.fbx(filepath=str(clip_fbx))
    new_objs = [bpy.data.objects[n] for n in set(bpy.data.objects.keys()) - before_objs]
    temp_armature = next(o for o in new_objs if o.type == "ARMATURE")

    action = temp_armature.animation_data.action
    action.name = action_name
    action.use_fake_user = True

    for obj in new_objs:
        bpy.data.objects.remove(obj, do_unlink=True)

    if char_armature.animation_data is None:
        char_armature.animation_data_create()
    char_armature.animation_data.action = action

    bpy.ops.object.select_all(action="DESELECT")
    char_armature.select_set(True)  # anim-only export -- mesh NOT selected

    out_glb = f"{out_prefix}_{out_suffix}.glb"
    Path(out_glb).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=out_glb,
        export_format="GLB",
        use_selection=True,
        export_animation_mode="ACTIVE_ACTIONS",
        export_yup=True,
    )
    print(f"exported -> {out_glb}")


def main():
    for clip_fbx, action_name, out_suffix in CLIPS:
        export_clip(clip_fbx, action_name, out_suffix)


if __name__ == "__main__":
    main()
