"""
Run via: blender.exe --background --python convert_character.py -- <character.fbx> <out.glb>

Imports one Mixamo character FBX (mesh + mixamorig: skeleton + its own baked
2-frame idle pose), then imports the 4 separately-downloaded animation FBX
files, lifts each one's action off its (throwaway) imported armature, and
re-assigns it onto the real character's armature as a named NLA track --
this works because all 5 files share the same mixamorig: bone-name
convention (verified via inspect_fbx.py), so an action authored against one
Mixamo skeleton retargets directly onto another by bone name, no manual
retargeting needed. Exports one .glb per character with mesh + skeleton +
4 named animation clips (Idle/Punch/Kick/Shoot) that Ursina/panda3d-gltf can
load and play by name.
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
character_fbx, out_glb = argv[0], argv[1]

bpy.ops.wm.read_factory_settings(use_empty=True)

# 1. import the real character (mesh + skeleton)
bpy.ops.import_scene.fbx(filepath=character_fbx)
char_armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")

# Mixamo FBX exports bundle a stray "Icosphere" marker object at the origin
# (~2 units across -- not tiny, floats visibly in-game if exported), skinned
# to the same armature as the real character mesh. Excluding it from the
# export *selection* wasn't enough -- Blender's glTF exporter pulls in every
# mesh skinned to an exported armature regardless of selection (confirmed by
# two failed attempts: an ARMATURE-modifier check, then a selection-based
# name exclusion, both still exported it). Only actually deleting the object
# stops it from being included.
icosphere = bpy.data.objects.get("Icosphere")
if icosphere is not None:
    bpy.data.objects.remove(icosphere, do_unlink=True)

char_meshes = [o for o in bpy.data.objects if o.type == "MESH"]
print(f"character armature: {char_armature.name}, meshes: {[m.name for m in char_meshes]}")

if char_armature.animation_data is None:
    char_armature.animation_data_create()
nla = char_armature.animation_data.nla_tracks

# 2. for each animation file: import into the same scene (creates its own
# throwaway armature+mesh+action), steal the action, delete the throwaway
# objects, push the action onto the real character as a named NLA strip
for clip_name, anim_path in ANIM_FILES.items():
    before_objs = set(bpy.data.objects.keys())
    bpy.ops.import_scene.fbx(filepath=str(anim_path))
    new_objs = [bpy.data.objects[n] for n in set(bpy.data.objects.keys()) - before_objs]
    temp_armature = next(o for o in new_objs if o.type == "ARMATURE")

    action = temp_armature.animation_data.action
    action.name = clip_name
    action.use_fake_user = True  # survives the temp armature's deletion below

    for obj in new_objs:
        bpy.data.objects.remove(obj, do_unlink=True)

    # Every strip starts at frame 0 on its OWN track -- NOT staggered on a
    # shared/growing frame_cursor (first attempt at this): the glTF exporter's
    # NLA_TRACKS mode keeps each track's exported animation frame range
    # relative to the GLOBAL timeline, not the strip's own local start, so
    # staggering made e.g. "Punch" export as 80 frames (40 of held Idle pose
    # + the real 39-frame punch motion) instead of just its own 39 frames --
    # confirmed via Panda3D's actor.getAnimControl(name).getNumFrames()
    # exactly matching the old cumulative offsets. Separate tracks starting
    # at frame 0 each keep every clip's own local frame range correct.
    track = nla.new()
    track.name = clip_name
    strip = track.strips.new(clip_name, 0, action)
    strip.name = clip_name
    print(f"  added clip '{clip_name}': frames {action.frame_range}")

# 3. export mesh + skeleton + all 4 NLA-track clips as one glb
bpy.ops.object.select_all(action="DESELECT")
char_armature.select_set(True)
for m in char_meshes:
    m.select_set(True)

# Mixamo FBX exports bundle a stray "Icosphere" marker object; deleted above
# right after the character import so it's never part of char_meshes/export.
# NOTE: bpy.ops.import_scene.gltf (Blender's *importer*, not this exporter)
# synthesizes its OWN phantom "Icosphere" bone-display-widget object whenever
# it reimports a skinned armature -- purely an editor-side artifact, not
# something that's actually in the exported file. Verify output .glb files
# with pygltflib (raw glTF inspection) instead of Blender reimport, or the
# importer's own widget will look like a bug that isn't there.
Path(out_glb).parent.mkdir(parents=True, exist_ok=True)
bpy.ops.export_scene.gltf(
    filepath=out_glb,
    export_format="GLB",
    use_selection=True,
    export_animation_mode="NLA_TRACKS",
    export_yup=True,
)
print(f"exported -> {out_glb}")
