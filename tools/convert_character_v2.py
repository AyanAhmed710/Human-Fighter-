"""
Run via: blender.exe --background --python convert_character_v2.py -- <character.fbx> <out_prefix>

v2: exports ONE SEPARATE .glb per animation instead of bundling all 4 into a
single file's NLA tracks (v1 -- convert_character.py). The NLA-track-in-one-
file approach exported files that *looked* correct (right mesh, right
animation names, right frame counts) but the pose genuinely never applied at
runtime -- confirmed via Panda3D screenshot: character stayed frozen in a
collapsed rest pose regardless of which clip played or how many frames were
simulated. Root cause not fully isolated (likely a skin/joint-binding issue
specific to multi-track NLA export), but rather than keep debugging a custom
export path, this switches to Panda3D's own standard, well-documented Actor
multi-animation pattern: one base file (mesh+skeleton+Idle) plus one anim-only
file per additional clip, loaded via
    Actor({"modelRoot": base.glb}, {"Punch": punch.glb, ...})
-- the officially supported way Actor handles multiple animations, much more
likely to have correct skin/joint binding than a hand-rolled NLA export.

Outputs <out_prefix>_base.glb (mesh+skeleton+Idle) and <out_prefix>_<Clip>.glb
per other clip (skeleton+action only, mesh included too for simplicity -- not
optimized for file size yet).
"""
import sys
from pathlib import Path

import bpy

ROOT = Path(r"C:\Data_Tekken")
ANIM_FILES = {
    "Idle": ROOT / "assets/mixamo/Standing Idle To Fight Idle.fbx",
    "Punch": ROOT / "assets/mixamo/Punching.fbx",
    "Kick": ROOT / "assets/mixamo/Kicking.fbx",
    "Shoot": ROOT / "assets/mixamo/Shooting.fbx",
}

argv = sys.argv[sys.argv.index("--") + 1:]
character_fbx, out_prefix = argv[0], argv[1]


def export_one(char_armature, char_meshes, action, out_glb, include_mesh):
    if char_armature.animation_data is None:
        char_armature.animation_data_create()
    char_armature.animation_data.action = action

    # REVERTED: an nla.bake() step was added here to try to fix Punch/Kick/
    # Shoot not visibly animating, on the theory that Blender 5.2.1's new
    # "layered actions" data model wasn't being exported correctly. It did
    # NOT fix that (the hip joint after a baked Punch showed an actually
    # broken/twisted rotation, confirmed via exposeJoint -- not a real
    # recovery), and it also regressed the base/Idle export, which was
    # confirmed correct (both characters standing, correctly posed) BEFORE
    # this bake step existed. Reverted entirely rather than keep stacking
    # fixes on a theory that made things worse. Idle-only, unbaked, is the
    # last known-good state -- Punch/Kick/Shoot playback remains unresolved.

    # anim-only files (Punch/Kick/Shoot) must contain ONLY the skeleton, not
    # a duplicate copy of the mesh -- including the mesh in every export
    # (first attempt) meant Panda3D's Actor bound each anim to that file's
    # OWN separate Character/skeleton instance instead of the modelRoot's,
    # so the animation played (isPlaying()=True, no errors) but never
    # visibly affected the actual rendered character. Confirmed via
    # screenshot: identical pose at frame 35/38 of "Punch" as at Idle.
    bpy.ops.object.select_all(action="DESELECT")
    char_armature.select_set(True)
    if include_mesh:
        for m in char_meshes:
            m.select_set(True)

    Path(out_glb).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=out_glb,
        export_format="GLB",
        use_selection=True,
        export_animation_mode="ACTIVE_ACTIONS",
        export_yup=True,
    )
    print(f"  exported -> {out_glb}")


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=character_fbx)
    char_armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")

    icosphere = bpy.data.objects.get("Icosphere")
    if icosphere is not None:
        bpy.data.objects.remove(icosphere, do_unlink=True)
    char_meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    print(f"character armature: {char_armature.name}, meshes: {[m.name for m in char_meshes]}")

    for clip_name, anim_path in ANIM_FILES.items():
        before_objs = set(bpy.data.objects.keys())
        bpy.ops.import_scene.fbx(filepath=str(anim_path))
        new_objs = [bpy.data.objects[n] for n in set(bpy.data.objects.keys()) - before_objs]
        temp_armature = next(o for o in new_objs if o.type == "ARMATURE")

        action = temp_armature.animation_data.action
        action.name = clip_name
        action.use_fake_user = True

        for obj in new_objs:
            bpy.data.objects.remove(obj, do_unlink=True)

        suffix = "base" if clip_name == "Idle" else clip_name
        export_one(char_armature, char_meshes, action, f"{out_prefix}_{suffix}.glb",
                   include_mesh=(clip_name == "Idle"))


if __name__ == "__main__":
    main()
