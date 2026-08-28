"""
Continuous rain effect -- a pool of thin streak Entities that fall and loop
back to the top, no external asset needed. Cheap enough (a few hundred
Entities) to just update every drop in Python each frame rather than reach
for Panda3D's native particle-effect/egg-file system.
"""
import random

from ursina import Entity, color, scene

# preset for the lava arena: bright glowing orange/red embers instead of
# blue-white water rain. Falling faster/smaller since it reads as sparks and
# droplets of molten rock rather than long streaks of water.
LAVA_RAIN_KWARGS = dict(
    drop_color=color.rgba32(255, 110, 30, 200),
    drop_scale=(0.02, 0.16, 0.02),
    fall_speed=9,
    count=140,
)


def _jitter(base_color):
    """Slight per-drop tint variance (orange -> yellow -> deep red) so the
    whole effect doesn't look like one flat copy-pasted color."""
    # base_color.r/g/b/a are 0-1 floats (ursina's Color convention) --
    # rgba32() is the 0-255 constructor, rgba() takes 0-1 and would clamp
    # anything above 1 to white (this bit us: the first version of this
    # function fed 0-255 numbers into rgba(), and every drop rendered as
    # solid white regardless of the intended color).
    r = min(255, max(0, base_color.r * 255 + random.uniform(-30, 30)))
    g = min(255, max(0, base_color.g * 255 + random.uniform(-30, 30)))
    b = min(255, max(0, base_color.b * 255 + random.uniform(-10, 10)))
    return color.rgba32(int(r), int(g), int(b), int(base_color.a * 255))


class RainEffect:
    def __init__(self, count=180, area_size=6, height=9, fall_speed=13, parent=None,
                 drop_color=None, drop_scale=(0.012, 0.3, 0.012)):
        self.area_size = area_size
        self.height = height
        self.fall_speed = fall_speed
        parent = parent or scene
        drop_color = drop_color or color.rgba32(190, 210, 255, 110)

        self.drops = []
        for _ in range(count):
            drop = Entity(
                parent=parent,
                model="cube",
                color=_jitter(drop_color),
                scale=drop_scale,
                position=(
                    random.uniform(-area_size, area_size),
                    random.uniform(0, height),
                    random.uniform(-area_size, area_size),
                ),
                unlit=True,  # rain streaks shouldn't be shaded by scene lights --
                             # for lava this also keeps them looking self-lit/glowing
            )
            # belt-and-suspenders: explicitly keep these off simplepbr's PBR
            # lighting pipeline (raw Panda3D NodePath calls -- Entity
            # subclasses NodePath) so they stay flat/self-lit regardless of
            # scene lights, matching unlit=True's intent.
            drop.setShaderOff(1)
            drop.setLightOff(1)
            self.drops.append(drop)

    def update(self, dt):
        for drop in self.drops:
            drop.y -= self.fall_speed * dt
            if drop.y < 0:
                # respawn at the top with a fresh random x/z, so it reads as
                # continuous rain rather than a fixed falling pattern
                drop.y = self.height
                drop.x = random.uniform(-self.area_size, self.area_size)
                drop.z = random.uniform(-self.area_size, self.area_size)
