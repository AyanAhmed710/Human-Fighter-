import sys
import bpy

path = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=path)

ico = bpy.data.objects.get("Icosphere")
if ico:
    print(f"Icosphere: location={tuple(ico.location)} dimensions={tuple(ico.dimensions)} "
          f"vertices={len(ico.data.vertices)} parent={ico.parent}")
else:
    print("no Icosphere object found")
