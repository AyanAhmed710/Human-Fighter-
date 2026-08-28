"""Exact answer to 'what is the ground height directly under this point':
for each fighter's XZ, walk every triangle of every arena mesh, keep the
ones whose XZ projection actually contains that point, and barycentric-
interpolate the real surface Y there. Vertex-proximity sampling (an earlier
version of this script) can miss entirely if vertices are sparse relative
to the check radius, or pick up an unrelated nearby vertex -- this is the
same math a physics engine's ground-raycast would do, so it's the ground
truth for "does the mesh actually have a surface under the feet, and at
what height."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from panda3d.core import Filename, GeomVertexReader
from ursina import Ursina, scene

ARENA_MODEL_PATH = r"C:\Data_Tekken\assets\models\arena.glb"
ARENA_FLOOR_LOCAL_Y = 0.072985  # solved from the -1.5843 gap measured with
                                 # the old (wrong) 0.1522 offset: real
                                 # surface directly under the fighters, not
                                 # just a nearby vertex on an outcrop peak
ARENA_SCALE = 20
FIGHTER_XS = [-1.0, 1.0]  # ARENA_HALF_WIDTH * (-1, 1)
FIGHTER_Z = 0.0

app = Ursina()
arena = loader.loadModel(Filename.fromOsSpecific(ARENA_MODEL_PATH))  # noqa: F821
arena.reparentTo(scene)
arena.setScale(ARENA_SCALE)
arena.setY(-ARENA_FLOOR_LOCAL_Y * ARENA_SCALE)


def point_in_triangle_xz(px, pz, a, b, c):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[2] - p3[2]) - (p2[0] - p3[0]) * (p1[2] - p3[2])
    d1 = sign((px, 0, pz), a, b)
    d2 = sign((px, 0, pz), b, c)
    d3 = sign((px, 0, pz), c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def barycentric_y(px, pz, a, b, c):
    denom = (b[2] - c[2]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[2] - c[2])
    if abs(denom) < 1e-12:
        return None
    w1 = ((b[2] - c[2]) * (px - c[0]) + (c[0] - b[0]) * (pz - c[2])) / denom
    w2 = ((c[2] - a[2]) * (px - c[0]) + (a[0] - c[0]) * (pz - c[2])) / denom
    w3 = 1 - w1 - w2
    return w1 * a[1] + w2 * b[1] + w3 * c[1]


hits = {fx: [] for fx in FIGHTER_XS}

for mesh_name in ("base", "lower", "Outer"):
    node = arena.find(f"**/{mesh_name}")
    if node.isEmpty():
        continue
    geom_np = node.find("**/+GeomNode")
    if geom_np.isEmpty():
        continue
    geom_node = geom_np.node()
    mat_to_scene = geom_np.getMat(scene)

    for gi in range(geom_node.getNumGeoms()):
        geom = geom_node.getGeom(gi)
        vdata = geom.getVertexData()
        reader = GeomVertexReader(vdata, "vertex")
        verts = []
        while not reader.isAtEnd():
            v = reader.getData3()
            w = mat_to_scene.xformPoint(v)
            verts.append((w.x, w.y, w.z))

        for pi in range(geom.getNumPrimitives()):
            prim = geom.getPrimitive(pi)
            prim = prim.decompose()  # triangulate fans/strips
            for t in range(prim.getNumPrimitives()):
                s = prim.getPrimitiveStart(t)
                e = prim.getPrimitiveEnd(t)
                idxs = [prim.getVertex(k) for k in range(s, e)]
                if len(idxs) != 3:
                    continue
                a, b, c = (verts[i] for i in idxs)
                for fx in FIGHTER_XS:
                    if point_in_triangle_xz(fx, FIGHTER_Z, a, b, c):
                        y = barycentric_y(fx, FIGHTER_Z, a, b, c)
                        if y is not None:
                            hits[fx].append((mesh_name, y))

for fx in FIGHTER_XS:
    results = sorted(hits[fx], key=lambda h: -h[1])
    print(f"fighter X={fx}, Z={FIGHTER_Z}: {len(results)} triangle(s) directly overhead/underfoot")
    for mesh_name, y in results:
        print(f"    mesh={mesh_name:8s} Y={y:.4f}")
    if not results:
        print("    NOTHING found at this XZ -- fighter would have no ground at all!")
