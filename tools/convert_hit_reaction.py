"""Adds the HitReact clip to an already-converted character without
re-touching the existing (verified-good) base/Punch/Kick/Shoot .glb files --
same anim-only export pattern as convert_character_v2.py's ANIM_FILES loop,
just scoped to one clip so there's no risk of regressing what already works.

Run via:
    blender.exe --background --python convert_hit_reaction.py -- <character.fbx> <out_prefix>

Outputs <out_prefix>_HitReact.glb (skeleton + action only, no mesh -- same
"anim-only files must not include the mesh" rule convert_character_v2.py
found the hard way: including it binds the clip to its own separate
Character instance instead of the modelRoot's, so it plays but never
visibly moves the actual rendered character).
"""
import sys
from pathlib import Path

import bpy

ROOT = Path(r"C:\Data_Tekken")
HIT_REACT_FBX = ROOT / "assets/mixamo/Hit Reaction.fbx"

argv = sys.argv[sys.argv.index("--") + 1:]
character_fbx, out_prefix = argv[0], argv[1]


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=character_fbx)
    char_armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")

    icosphere = bpy.data.objects.get("Icosphere")
    if icosphere is not None:
        bpy.data.objects.remove(icosphere, do_unlink=True)

    before_objs = set(bpy.data.objects.keys())
    bpy.ops.import_scene.fbx(filepath=str(HIT_REACT_FBX))
    new_objs = [bpy.data.objects[n] for n in set(bpy.data.objects.keys()) - before_objs]
    temp_armature = next(o for o in new_objs if o.type == "ARMATURE")

    action = temp_armature.animation_data.action
    action.name = "HitReact"
    action.use_fake_user = True

    for obj in new_objs:
        bpy.data.objects.remove(obj, do_unlink=True)

    if char_armature.animation_data is None:
        char_armature.animation_data_create()
    char_armature.animation_data.action = action

    bpy.ops.object.select_all(action="DESELECT")
    char_armature.select_set(True)  # anim-only export -- mesh NOT selected

    out_glb = f"{out_prefix}_HitReact.glb"
    Path(out_glb).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=out_glb,
        export_format="GLB",
        use_selection=True,
        export_animation_mode="ACTIVE_ACTIONS",
        export_yup=True,
    )
    print(f"exported -> {out_glb}")


if __name__ == "__main__":
    main()
