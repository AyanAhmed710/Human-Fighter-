"""Makes the arena's lava look alive: continuous UV-scroll on the floor's
lava-crack texture (so the pattern visibly creeps rather than sitting
frozen) plus a slow sine-wave pulse on every lava material's emissive glow,
arena-wide (the lava "breathes" instead of holding one flat brightness). No
new assets needed -- just animating what tools/convert_arena.py already set
up on the existing materials.

UV-scroll is deliberately floor-only, not arena-wide: tested scrolling the
whole arena (tools/test_lava_flow.py) and the crater walls ('Outer' mesh)
turned into an ugly streaked mess -- that mesh's low-poly rock faces each
have their own unique, non-tiling UV unwrap, so shifting coordinates breaks
alignment at every seam between faces. The floor ('lower' mesh) has a
proper tileable crack pattern and scrolls cleanly. The emissive pulse has
no such risk (it's just a uniform color multiply, not a UV remap), so that
part is safe to apply everywhere.
"""
import math

from panda3d.core import Texture


class LavaFlow:
    def __init__(self, arena_root, floor_root, scroll_speed=(0.012, 0.008),
                 pulse_speed=1.1, pulse_amount=0.35):
        self.floor_root = floor_root
        self.scroll_speed = scroll_speed
        self.pulse_speed = pulse_speed
        self.pulse_amount = pulse_amount
        self.elapsed = 0.0

        self.floor_stages = list(floor_root.findAllTextureStages())
        for tex in floor_root.findAllTextures():
            tex.setWrapU(Texture.WMRepeat)
            tex.setWrapV(Texture.WMRepeat)

        # remember each material's ORIGINAL emission so the pulse multiplies
        # around a fixed base instead of drifting/compounding frame to
        # frame. tuple(...) matters here -- mat.getEmission() returns a
        # live reference into the material's own storage, not a snapshot;
        # storing that directly meant every update() call was reading back
        # a value already mutated by the previous call and multiplying
        # again on top of it (confirmed via tools/test_lava_flow.py: the
        # "base" value printed had visibly drifted between calls).
        self.materials = []
        for mat in arena_root.findAllMaterials():
            if mat.hasEmission():
                self.materials.append((mat, tuple(mat.getEmission())))

    def update(self, dt):
        self.elapsed += dt

        u = (self.scroll_speed[0] * self.elapsed) % 1.0
        v = (self.scroll_speed[1] * self.elapsed) % 1.0
        for stage in self.floor_stages:
            self.floor_root.setTexOffset(stage, u, v)

        pulse = 1.0 + self.pulse_amount * math.sin(self.elapsed * self.pulse_speed)
        for mat, base_emission in self.materials:
            mat.setEmission((
                base_emission[0] * pulse,
                base_emission[1] * pulse,
                base_emission[2] * pulse,
                base_emission[3],
            ))
