"""Central visual design system for the game's 2D UI (camera.ui layer) --
one place owning colors/fonts/reusable widget-builders so menu.py,
play_game.py's HUD/VS/victory screens, and profile.py's screen all read as
one coherent "premium fighting game" look instead of each screen inventing
its own colors/spacing. Nothing here touches 3D rendering, match logic, or
combat -- purely presentational.

Original visual identity, not a Tekken reskin: the palette below was picked
to tie into what's *already* on screen (the lava arena's warm orange glow,
the existing azure P1 / orange P2 split from play_game.py's HUD) rather than
copying any specific game's UI -- deep charcoal base, molten-orange primary
accent, cold azure secondary, plus danger-red and victory-gold for state,
and a hot magenta reserved ONLY for crit callouts so it stays a rare "wow"
color instead of just another accent.
"""
from pathlib import Path

from panda3d.core import Filename
from ursina import Entity, Text, color, curve, invoke

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
VOID = color.hex("#08080b")            # emptiest background layer
PANEL = color.hex("#15151c")           # card/panel fill
PANEL_LIGHT = color.hex("#1f1f2a")     # raised panel / hover fill
BORDER = color.hex("#3a3a48")          # panel edges
BORDER_BRIGHT = color.hex("#5a5a6e")

TEXT = color.hex("#f2f1ec")            # primary text -- warm off-white, not pure white
TEXT_MUTED = color.hex("#9a97a6")

ACCENT_P1 = color.hex("#3fb6ff")       # azure -- player 1 identity color throughout
ACCENT_P2 = color.hex("#ff7a1a")       # molten orange -- player 2 identity color, matches lava arena
DANGER = color.hex("#ff2d3d")          # low health / KO
VICTORY = color.hex("#ffcf4a")         # gold -- wins, rank-up, round pips filled
CRIT = color.hex("#ff3df0")            # reserved for critical-hit callouts ONLY

# ---------------------------------------------------------------------------
# Fonts -- real system fonts (confirmed present on this Windows box), each
# picked for a distinct job. Resolved once at import time via _font(), which
# falls back to None (Ursina's own default) if a path doesn't exist -- so
# this module never hard-crashes the game over a missing font file, it just
# quietly looks less polished.
# ---------------------------------------------------------------------------
_FONT_DIR = Path("C:/Windows/Fonts")


def _font(filename: str):
    p = _FONT_DIR / filename
    if not p.is_file():
        return None
    # Ursina's Text.font setter hands the string straight to panda3d's
    # loader.loadFont(), which wants a Panda-style path ("/c/Windows/..."),
    # not a raw Windows path -- confirmed via loadFont() raising OSError on
    # "C:/Windows/Fonts/impact.ttf" directly, working once run through
    # Filename.fromOsSpecific(). Same conversion play_game.py already uses
    # for the arena model/backdrop.
    return str(Filename.fromOsSpecific(str(p)))


FONT_DISPLAY = _font("impact.ttf")      # huge dramatic headlines: VS, VICTORY, K.O., ROUND N
FONT_HEAVY = _font("bahnschrift.ttf")   # bold condensed: names, HUD labels, buttons, numbers
FONT_BODY = _font("segoeui.ttf")        # description/body/hint text


def font_kwargs(font):
    # Text(font=None, ...) is NOT the same as omitting font= -- passing the
    # literal None overrides Ursina's own default with nothing, so this
    # only injects the kwarg when a real path resolved.
    return {"font": font} if font else {}


# ---------------------------------------------------------------------------
# Reusable widget builders
# ---------------------------------------------------------------------------

def panel(parent, scale, position, origin=(0, 0), z=0.5, tint=PANEL, border=BORDER):
    """Layered card: a slightly-larger border-color quad behind a tinted
    fill quad, giving a 1-2%-of-width "stroke" without needing an actual
    outline shader. z=0.5 by default -- sits behind normal-z (0) text/
    buttons and in front of the z=1 full-screen vignette, same convention
    menu.py's character cards already established (see its own z=0.5 note).
    Returns the FILL entity -- parent further children to that."""
    sx, sy = scale
    edge = Entity(parent=parent, model="quad", color=border,
                   scale=(sx + 0.006, sy + 0.006), position=position, origin=origin, z=z + 0.001)
    fill = Entity(parent=parent, model="quad", color=tint,
                   scale=scale, position=position, origin=origin, z=z)
    fill.edge = edge  # keep the border alive/reachable from the fill (e.g. for destroy())
    return fill


def section_title(parent, text, position, scale=1.6, fg=TEXT, origin=(0, 0)):
    """A headline in the display font with a short accent underline beneath
    it -- the underline is what actually reads as "designed", not just a
    bigger font size."""
    t = Text(text, parent=parent, position=position, origin=origin, scale=scale,
             color=fg, **font_kwargs(FONT_DISPLAY))
    return t


def style_button_text(button, font=None, color_=TEXT):
    """Ursina's Button builds its own internal Text lazily (button.text_entity,
    only created once you assign button.text=...) so font can't be passed as
    a constructor kwarg -- this sets it after the fact. Safe no-op if the
    button has no text (icon-only buttons)."""
    font = font if font is not None else FONT_HEAVY
    if button.text_entity and font:
        button.text_entity.font = font
    if button.text_entity:
        button.text_entity.color = color_
    return button


def safe_animate(entity, method_name: str, value, **kwargs):
    """Wraps an Entity.animate_<x>() call (animate_scale, animate_x,
    animate_position, ...) in a try/except -- ANY entity whose position/
    scale/etc. gets animated is at risk of this if its own screen can be
    destroyed while the animation might still be in flight (screen
    transitions, Back/Quit/Rematch/Menu buttons, a round/phase change
    tearing down a HUD element, restart() firing mid-VS-intro, etc.): these
    animate_* calls' step mechanics read the entity's CURRENT value (e.g.
    scale_getter -> panda3d's getScale()) each time a step actually runs.
    If the underlying NodePath has already been destroyed by then, that
    read throws AssertionError: !is_empty() from deep inside panda3d --
    confirmed via a real crash right after NameEntry's submit button
    destroyed its own screen mid-click, and again via CharacterSelect's
    card-pick pulse still being mid-animation when FIGHT! destroyed the
    whole screen a moment later. Best-effort: skip silently instead of
    crashing the whole game over a decorative animation.

    IMPORTANT: if you need a delay before the animation starts, do NOT pass
    delay= here -- schedule it yourself instead:
        invoke(safe_animate, entity, "animate_scale", value, duration=..., delay=X)
    animate_*'s own delay= kwarg hands the actual work off to ursina's
    internal callback machinery, decoupled from this function's own call
    stack -- the failure (if the entity's gone by then) happens on that
    LATER frame, inside ursina's own scheduled callback, so wrapping just
    this call would not catch it. Scheduling the delay yourself via
    invoke() means THIS function (try/except and all) is what actually
    runs once the delay elapses -- see hover_button's on_click below for
    the pattern."""
    try:
        getattr(entity, method_name)(value, **kwargs)
    except Exception:
        pass


def safe_animate_scale(entity, value, **kwargs):
    """safe_animate(entity, "animate_scale", value, **kwargs) -- see that
    docstring for the full explanation. Kept as its own name since this is
    by far the most common case (every button's hover/press pulse)."""
    safe_animate(entity, "animate_scale", value, **kwargs)


def hover_button(button, sfx_module=None, pulse=1.06):
    """Wires a real Ursina Button with a snappy hover/press feel -- scale
    pulse on mouse-enter, settle on mouse-exit, quick squash on click. Kept
    to scale animation only (no color-swap fighting the button's own
    highlight_color) so this layers cleanly on top of whatever color a
    caller already gave the button. sfx_module: pass src/game/sfx if you
    want the existing quiet UI-hover blip (see sfx.play_hover); omitted by
    default so this stays usable in contexts without audio."""
    base_scale = button.scale
    orig_click = button.on_click

    def on_enter():
        safe_animate_scale(button, base_scale * pulse, duration=0.09, curve=curve.out_expo)
        if sfx_module is not None:
            sfx_module.play_hover()
    def on_exit():
        safe_animate_scale(button, base_scale, duration=0.12, curve=curve.out_expo)
    def on_click():
        safe_animate_scale(button, base_scale * 0.94, duration=0.05, curve=curve.out_expo)
        invoke(safe_animate_scale, button, base_scale, duration=0.12, curve=curve.out_expo, delay=0.05)
        if orig_click:
            orig_click()

    button.on_mouse_enter = on_enter
    button.on_mouse_exit = on_exit
    button.on_click = on_click
    return button


class GlowBar:
    """A "premium" health/progress bar: dark frame + colored fill + a thin
    lighter sheen strip riding the top of the fill -- three flat quads, no
    shader, but reads as a beveled/glossy bar instead of a flat rectangle.
    anchor="left"|"right" controls which edge stays fixed as the fill
    shrinks (see play_game.py's health_bar2 fix -- P2's bar must anchor
    right so it drains toward its own outer edge, not toward screen-center).
    """

    def __init__(self, position, width, height, anchor="left", z=0.4):
        self.width = width
        self.height = height
        self.anchor = anchor
        edge_x = -0.5 if anchor == "left" else 0.5
        self.origin = (edge_x, 0)
        # position passed in is the anchored edge's x -- same convention the
        # old ad hoc bars in play_game.py used, kept so callers don't need
        # to re-derive it.
        frame_pad = 0.006
        self.frame = Entity(parent=camera_ui(), model="quad", color=BORDER,
                             scale=(width + frame_pad, height + frame_pad), position=position,
                             origin=self.origin, z=z + 0.02)
        self.bg = Entity(parent=camera_ui(), model="quad", color=PANEL,
                          scale=(width, height), position=position, origin=self.origin, z=z + 0.01)
        self.fill = Entity(parent=camera_ui(), model="quad", color=color.lime,
                            scale=(width, height), position=position, origin=self.origin, z=z)
        sheen_h = height * 0.3
        sheen_y = position[1] + height * 0.32
        self.sheen = Entity(parent=camera_ui(), model="quad",
                             color=color.rgba32(255, 255, 255, 60),
                             scale=(width, sheen_h), position=(position[0], sheen_y),
                             origin=self.origin, z=z - 0.001)

    def set_fraction(self, frac: float, fg):
        frac = max(0.0, min(1.0, frac))
        self.fill.scale_x = self.width * frac
        self.fill.color = fg
        self.sheen.scale_x = self.width * frac

    def destroy(self):
        from ursina import destroy as _destroy
        for e in (self.frame, self.bg, self.fill, self.sheen):
            _destroy(e)


def camera_ui():
    # deferred import -- avoids a hard import-order dependency on ursina's
    # `camera` global existing yet at module-import time (theme.py can be
    # imported before Ursina() has run in some test/tooling contexts).
    from ursina import camera
    return camera.ui
