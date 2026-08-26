import sys
import bpy

path = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=path)

print(f"=== {path} ===")
for obj in bpy.data.objects:
    print(f"  object: {obj.name}  type={obj.type}")
print(f"  actions: {[a.name for a in bpy.data.actions]}")
