"""
Run via: blender.exe --background --python convert_arena.py -- <arena.fbx> <out.glb> <textures_dir>

Converts the downloaded Lava_Stage.fbx into a static (no armature, no
animation needed) glb for the game arena. Two problems found by inspecting
the raw FBX (tools/inspect_arena*.py) that a plain re-export wouldn't fix:

1. Each material's Base Color image texture node points at a path baked in
   by the original exporter (assets/arena/source/<name>.png) that doesn't
   match where the files actually are on disk (assets/arena/textures/) --
   fixed by relinking each Image Texture node's filepath by basename.

2. lambert1_Emissive.png / Main_Base_Emissive.png sit on disk but were
   NEVER wired into the material graph at all (each material only had a
   single Base Color texture node, confirmed via node-graph inspection) --
   without this the lava glow textures are simply unused and the stage
   would render as flat/unlit rock. Added as a second Image Texture node
   per material, linked into the Principled BSDF's Emission Color input
   with Emission Strength turned up so it actually glows once exported
   (glTF's KHR_materials_emissive_strength extension, auto-emitted by
   Blender's exporter for Strength > 1).

The 'sky' mesh is dropped -- its own texture (Sky01_Base_color.png) was
never included in the download, so it would import broken/pink either way;
the game already has its own Sky()/lighting.
"""
import sys
from pathlib import Path

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
in_fbx, out_glb, textures_dir = argv[0], argv[1], Path(argv[2])

EMISSIVE_MAP = {
    "Main_Base": "Main_Base_Emissive.png",
    "lambert1": "lambert1_Emissive.png",
}
EMISSION_STRENGTH = 4.0

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=in_fbx)

sky = bpy.data.objects.get("sky")
if sky is not None:
    bpy.data.objects.remove(sky, do_unlink=True)

for mat in bpy.data.materials:
    nodes = mat.node_tree.nodes
    bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        continue

    # 1. relink the existing Base Color texture to its real file location
    base_color_node = next((n for n in nodes if n.type == "TEX_IMAGE"), None)
    if base_color_node is not None and base_color_node.image is not None:
        real_path = textures_dir / Path(base_color_node.image.filepath).name
        if real_path.exists():
            base_color_node.image.filepath = str(real_path)
            base_color_node.image.reload()
            print(f"relinked {mat.name} base color -> {real_path}")
        else:
            print(f"WARNING: {mat.name} base color file not found: {real_path}")

    # 2. add + wire the emissive (lava glow) texture, if this material has one
    emissive_file = EMISSIVE_MAP.get(mat.name)
    if emissive_file is None:
        continue
    emissive_path = textures_dir / emissive_file
    if not emissive_path.exists():
        print(f"WARNING: emissive file not found for {mat.name}: {emissive_path}")
        continue

    img = bpy.data.images.load(str(emissive_path))
    tex_node = nodes.new("ShaderNodeTexImage")
    tex_node.image = img
    tex_node.location = (base_color_node.location.x, base_color_node.location.y - 300)
    mat.node_tree.links.new(tex_node.outputs["Color"], bsdf.inputs["Emission Color"])
    bsdf.inputs["Emission Strength"].default_value = EMISSION_STRENGTH
    print(f"wired emissive lava glow -> {mat.name} (strength={EMISSION_STRENGTH})")

bpy.ops.object.select_all(action="SELECT")
Path(out_glb).parent.mkdir(parents=True, exist_ok=True)
bpy.ops.export_scene.gltf(
    filepath=out_glb,
    export_format="GLB",
    use_selection=True,
    export_yup=True,
)
print(f"exported -> {out_glb}")
