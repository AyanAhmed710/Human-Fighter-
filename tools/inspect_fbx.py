import sys
import bpy

path = sys.argv[sys.argv.index("--") + 1]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=path)

print(f"=== {path} ===")
for obj in bpy.data.objects:
    print(f"  object: {obj.name}  type={obj.type}")
    if obj.type == "ARMATURE":
        bones = list(obj.data.bones.keys())
        print(f"    bone count: {len(bones)}")
        print(f"    first 5 bones: {bones[:5]}")

print(f"  actions in file: {[a.name for a in bpy.data.actions]}")
for action in bpy.data.actions:
    print(f"    action '{action.name}': frame_range={action.frame_range}")
