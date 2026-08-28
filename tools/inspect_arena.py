"""Inspect the arena FBX before conversion -- object names, bounding box
(to figure out floor height / scale / camera framing), materials + which
texture images they reference (FBX from Maya often needs textures relinked
manually before glTF export), and whether it carries any animation."""
import sys
import bpy

path = sys.argv[sys.argv.index("--") + 1]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=path)

print(f"=== {path} ===")
min_co = [float("inf")] * 3
max_co = [float("-inf")] * 3

for obj in bpy.data.objects:
    print(f"  object: {obj.name}  type={obj.type}  loc={tuple(obj.location)}  scale={tuple(obj.scale)}")
    if obj.type == "MESH":
        for v in obj.bound_box:
            world_v = obj.matrix_world @ __import__("mathutils").Vector(v)
            for i in range(3):
                min_co[i] = min(min_co[i], world_v[i])
                max_co[i] = max(max_co[i], world_v[i])
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None:
                continue
            print(f"    material: {mat.name}")
            if mat.use_nodes:
                for node in mat.node_tree.nodes:
                    if node.type == "TEX_IMAGE" and node.image:
                        print(f"      texture node -> {node.image.name}  filepath={node.image.filepath!r}")

print(f"\n  bounding box min={min_co} max={max_co}")
print(f"  size = {[max_co[i]-min_co[i] for i in range(3)]}")
print(f"  actions in file: {[a.name for a in bpy.data.actions]}")
