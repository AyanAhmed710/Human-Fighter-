"""Adds the critical-hit stun-sequence clips (Stunned + Getting Up) to
already-converted characters -- same anim-only export pattern as
convert_hit_reaction.py/convert_block_anims.py, just scoped to this pair.

Run via:
    blender.exe --background --python convert_crit_anims.py -- <character.fbx> <out_prefix>

Outputs <out_prefix>_Stunned.glb and <out_prefix>_GettingUp.glb -- skeleton
+ action only, no mesh (see convert_hit_reaction.py's docstring for why).
"""
import sys
from pathlib import Path

import bpy

ROOT = Path(r"C:\Data_Tekken")
CLIPS = [
    (ROOT / "assets/mixamo/Stunned.fbx", "Stunned", "Stunned"),
    (ROOT / "assets/mixamo/Getting Up.fbx", "GettingUp", "GettingUp"),
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
