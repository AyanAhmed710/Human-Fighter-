"""Coded UI screens -- name entry, main menu, character select, and the
player profile screen. Built entirely from Ursina's stock Button/Entity/
Text/InputField primitives (no external UI art pack needed), styled through
src/game/theme.py so every screen shares one visual language instead of each
inventing its own colors/fonts.

CharacterSelect's Fight button is ALWAYS clickable (not enabled/disabled)
and validates on click instead -- an earlier version toggled
`button.enabled` on pick/no-pick, which could leave the button's collider in
a stale state and made it look "broken" (dead clicks with no feedback at
all). This version can never dead-click: unpicked click -> clear on-screen
warning; picked click -> always fires.

These classes only own the flat camera.ui overlay; the actual 3D
"showroom" -- the two live character models standing still in the arena,
each throwing a punch when its CHOOSE button is clicked -- is spawned/
despawned by play_game.py's main(), which also owns the shared 3D camera
and the dark vignette that makes each screen read as a genuinely different
place instead of the same view with different text on top.
"""
import random

from ursina import Button, Entity, Func, Text, camera, color, time
from ursina.prefabs.input_field import InputField

from src.game import sfx, theme


def _clicky(on_click):
    """Wraps a Button's on_click so every button in these screens gets the
    same short UI-click sfx for free, without repeating sfx.play("click")
    at every call site."""
    def wrapped():
        sfx.play("click", volume=0.5)
        on_click()
    return wrapped


def _styled_button(text, parent, position, scale, fg=theme.TEXT,
                    base=theme.PANEL_LIGHT, accent=None, on_click=None, text_scale=1.0):
    """One consistent button recipe used everywhere in these screens:
    themed fill/hover/press colors, hover_button's scale-pulse + UI-hover
    sfx, and the heavy HUD font on its label. accent=None -> neutral gray
    button (Settings/Back/Close); pass an accent color (theme.ACCENT_P2 for
    the primary CTA, theme.VICTORY for confirms, theme.DANGER for Quit) to
    make a button visually dominant/intentional rather than every button
    looking the same weight."""
    tint = accent if accent else base
    btn = Button(text, parent=parent, position=position, scale=scale,
                 color=tint, highlight_color=tint.tint(.25),
                 pressed_color=tint.tint(-.25), on_click=_clicky(on_click) if on_click else None)
    theme.style_button_text(btn, font=theme.FONT_HEAVY, color_=fg)
    if btn.text_entity:
        btn.text_entity.world_scale *= text_scale
    theme.hover_button(btn, sfx_module=sfx)
    return btn


class _EmberField(Entity):
    """Cheap ambient atmosphere for the main menu -- a handful of small
    warm-colored quads drifting upward and fading, looping forever. Pure
    2D camera.ui decoration (not 3D particles), kept small in count (16)
    specifically because this runs alongside a live mediapipe pipeline in
    camera mode elsewhere in the app -- menu screens shouldn't be the thing
    that burns the frame budget."""
    COUNT = 16

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.embers = []
        for _ in range(self.COUNT):
            e = Entity(parent=self, model="quad", color=theme.ACCENT_P2,
                       scale=random.uniform(0.004, 0.01), z=0.8,
                       position=(random.uniform(-0.95, 0.95), random.uniform(-0.55, 0.55)))
            e.speed = random.uniform(0.035, 0.09)
            e.drift = random.uniform(-0.015, 0.015)
            e.base_alpha = random.uniform(70, 160)
            self.embers.append(e)

    def update(self):
        for e in self.embers:
            e.y += e.speed * time.dt
            e.x += e.drift * time.dt
            if e.y > 0.6:
                e.y = -0.6
                e.x = random.uniform(-0.95, 0.95)
            # fades in near the bottom, fades out near the top -- avoids a
            # hard pop at either end of the loop.
            fade = min(1.0, (e.y + 0.6) / 0.25, (0.6 - e.y) / 0.25)
            e.color = color.rgba32(255, int(120 + 40 * fade), 20, int(e.base_alpha * fade))


class NameEntry(Entity):
    """First-launch-only screen: no saved profile yet (see profile.py) so
    the player types a name once before ever reaching the main menu. Not
    shown again after that -- play_game.py's main() only builds this when
    PlayerProfile.load() returned None."""

    def __init__(self, on_submit):
        super().__init__(parent=camera.ui)
        theme.section_title(self, "WHO'S FIGHTING?", position=(0, 0.16), scale=3.2)
        Text("Enter a name -- this is how you'll show up in-game", parent=self,
             position=(0, 0.075), origin=(0, 0), scale=1.05, color=theme.TEXT_MUTED,
             **theme.font_kwargs(theme.FONT_BODY))

        card = theme.panel(self, (0.5, 0.16), (0, -0.02))
        # InputField's own __init__ hardcodes highlight_color=color.black
        # internally (ursina/prefabs/input_field.py) -- passing it again
        # here collides as a duplicate kwarg, so only `color` is themed.
        self.field = InputField(parent=card, scale=(0.42, 0.08), position=(0, 0),
                                 character_limit=16, active=True, color=theme.PANEL)
        self.field.submit_on = ["enter"]
        self.field.on_submit = self._submit
        if self.field.text_field.text_entity:
            self.field.text_field.text_entity.font = theme.FONT_HEAVY

        self._on_submit = on_submit
        _styled_button("ENTER THE ARENA", self, (0, -0.22), (0.32, 0.09),
                       accent=theme.ACCENT_P2, on_click=self._submit, text_scale=1.05)

    def _submit(self):
        name = (self.field.text_field.text or "").strip()
        self._on_submit(name or "FIGHTER")


class MainMenu(Entity):
    def __init__(self, on_start, on_profile, on_online, on_quit, profile):
        super().__init__(parent=camera.ui)
        self.profile = profile

        self.embers = _EmberField(self)

        # player identity strip, top-left -- the player sees their own name
        # the instant they hit the menu, not buried in a submenu.
        Text(profile.username.upper(), parent=self, position=(-0.87, 0.46), origin=(-0.5, 0),
             scale=1.5, color=theme.ACCENT_P1, **theme.font_kwargs(theme.FONT_HEAVY))
        record = (f"{profile.wins}W - {profile.losses}L"
                  + (f"   {profile.win_rate*100:.0f}% WIN RATE" if profile.matches_played else "   NO MATCHES YET"))
        Text(record, parent=self, position=(-0.87, 0.415), origin=(-0.5, 0),
             scale=0.85, color=theme.TEXT_MUTED, **theme.font_kwargs(theme.FONT_BODY))

        # everything below is the "home" row -- grouped under one container
        # so opening Settings can disable the whole thing at once (see
        # toggle_settings). Without this, the settings panel drew BEHIND
        # these buttons (children share z=0 by default; the panel's own
        # z=0.3 put it further from camera, not closer) so FIGHT stayed
        # visible AND clickable right through the open settings panel --
        # confirmed via screenshot, this fixes it properly instead of just
        # painting over it.
        self.home = Entity(parent=self)
        theme.section_title(self.home, "HUMAN FIGHTER", position=(0, 0.34), scale=4.6)
        Entity(parent=self.home, model="quad", color=theme.ACCENT_P2, scale=(0.32, 0.006),
               position=(0, 0.255))
        Text("FIGHT.  COMPETE.  IMPROVE.", parent=self.home, position=(0, 0.21), origin=(0, 0),
             scale=1.15, color=theme.TEXT_MUTED, **theme.font_kwargs(theme.FONT_HEAVY))

        # primary CTA -- visually dominant: biggest button on screen, only
        # one in the aggressive orange accent color, dead center.
        _styled_button("FIGHT", self.home, (0, -0.1), (0.4, 0.14),
                       accent=theme.ACCENT_P2, on_click=on_start, text_scale=1.4)

        _styled_button("ONLINE 1V1", self.home, (0, -0.21), (0.28, 0.075),
                       accent=theme.ACCENT_P1, on_click=on_online)

        _styled_button("PROFILE", self.home, (-0.24, -0.32), (0.22, 0.075),
                       on_click=on_profile)
        _styled_button("SETTINGS", self.home, (0, -0.32), (0.22, 0.075),
                       on_click=self.toggle_settings)
        _styled_button("QUIT", self.home, (0.24, -0.32), (0.22, 0.075),
                       accent=theme.DANGER, on_click=on_quit)

        self.settings_panel = Entity(parent=self, enabled=False)
        theme.panel(self.settings_panel, (0.66, 0.52), (0, -0.05), z=0.3)
        theme.section_title(self.settings_panel, "CONTROLS", position=(-0.3, 0.15), scale=1.6)
        Text("Player 1: 1=punch  2=kick  3=shoot  4=hold to block", parent=self.settings_panel,
             position=(-0.3, 0.06), scale=1.1, color=theme.TEXT, **theme.font_kwargs(theme.FONT_BODY))
        Text("Player 2: 8=punch  9=kick  0=shoot  7=hold to block", parent=self.settings_panel,
             position=(-0.3, 0.0), scale=1.1, color=theme.TEXT, **theme.font_kwargs(theme.FONT_BODY))
        Text("R=restart   ESC/Q=quit   C=free camera", parent=self.settings_panel,
             position=(-0.3, -0.06), scale=1.1, color=theme.TEXT, **theme.font_kwargs(theme.FONT_BODY))
        Text("Best of 3 rounds -- 2 round wins takes the match",
             parent=self.settings_panel, position=(-0.3, -0.12), scale=1.0,
             color=theme.TEXT_MUTED, **theme.font_kwargs(theme.FONT_BODY))
        _styled_button("CLOSE", self.settings_panel, (0, -0.25), (0.2, 0.06),
                       on_click=self.toggle_settings)

    def toggle_settings(self):
        opening = not self.settings_panel.enabled
        self.settings_panel.enabled = opening
        self.home.enabled = not opening


# Only 2 real character models exist (assets/models/warrok*.glb,
# vampire*.glb -- see play_game.py's P1_MODEL/P2_MODEL) so "select your
# character" only has one real decision: which of the 2 Player 1 takes.
# Whoever Player 1 doesn't pick automatically goes to Player 2.
CHAR_INFO = {
    "warrok": {
        "label": "WARROK",
        "desc": "Heavy brawler. Slow but every hit lands hard.",
    },
    "vampire": {
        "label": "VAMPIRE",
        "desc": "Fast striker. Weaves in and punishes openings.",
    },
}

# Pulled straight from src/game/match.py's ACTION_STATS -- both fighters
# currently share the same numbers (no per-character balance pass yet), so
# this is shown as the roster's shared move set, not a per-character stat.
POWER_LINES = [("Punch", 8), ("Kick", 14), ("Shoot", 10)]


class CharacterSelect(Entity):
    def __init__(self, on_pick, on_confirm, on_back, profile):
        super().__init__(parent=camera.ui)
        self.on_pick = on_pick      # called (chosen_key, other_key) live, for
                                     # the 3D preview to react (spin/scale-pop)
        self.on_confirm = on_confirm  # called (p1_key, p2_key) when Fight! fires
        self.chosen = None

        theme.section_title(self, "SELECT YOUR FIGHTER", position=(0, 0.47), scale=2.6)
        self.hint = Text(f"{profile.username.upper()}, PLAYER 1 -- CHOOSE FIRST", parent=self,
                          position=(0, 0.4), origin=(0, 0), scale=1.05, color=theme.ACCENT_P1,
                          **theme.font_kwargs(theme.FONT_BODY))

        self.labels = {}
        self.cards = {}
        self.glows = {}
        self.choose_buttons = {}
        # card vertical layout, all relative to CARD_Y/CARD_H so nothing
        # here needs separately-tuned magic numbers per row -- was a real
        # bug earlier (card scale grew to 0.6 without moving CHOOSE/FIGHT!
        # to match, so CHOOSE sat on the card's bottom border and FIGHT!
        # got clipped off the bottom of the screen; confirmed via
        # screenshot, fixed by deriving every y from the card's own bounds).
        CARD_Y, CARD_H = -0.08, 0.5
        card_top, card_bottom = CARD_Y + CARD_H / 2, CARD_Y - CARD_H / 2
        # WARROK's card nudged further left (was -0.34, symmetric with
        # VAMPIRE's +0.34) -- its right edge was overlapping the VAMPIRE
        # preview model standing behind it in the arena gap, hiding it
        # (confirmed via screenshot). Explicit per-key x instead of the
        # derived x/x+=0.68 stepping so only WARROK's card moves --
        # VAMPIRE's stays exactly where it was, no reason to touch it too.
        CARD_X = {"warrok": -0.40, "vampire": 0.34}
        for key, info in CHAR_INFO.items():
            x = CARD_X[key]
            glow = Entity(parent=self, model="quad", color=color.rgba32(0, 0, 0, 0),
                           scale=(0.34, CARD_H + 0.02), position=(x, CARD_Y), z=0.55)
            self.glows[key] = glow
            card = theme.panel(self, (0.32, CARD_H), (x, CARD_Y), z=0.5, tint=theme.PANEL)
            self.cards[key] = card
            lbl = Text(info["label"], parent=self, position=(x, card_top - 0.09), origin=(0, 0),
                       scale=1.9, color=theme.TEXT, **theme.font_kwargs(theme.FONT_DISPLAY))
            self.labels[key] = lbl
            Text(info["desc"], parent=self, position=(x - 0.14, card_top - 0.16), scale=0.78,
                 wordwrap=22, color=theme.TEXT_MUTED, **theme.font_kwargs(theme.FONT_BODY))
            Entity(parent=self, model="quad", color=theme.BORDER, scale=(0.28, 0.003),
                   position=(x, card_top - 0.25))
            y = card_top - 0.29
            for stat_label, val in POWER_LINES:
                Text(f"{stat_label}", parent=self, position=(x - 0.13, y), scale=0.85,
                     color=theme.TEXT_MUTED, **theme.font_kwargs(theme.FONT_BODY))
                Text(f"{val}", parent=self, position=(x + 0.1, y), scale=0.85,
                     color=theme.TEXT, **theme.font_kwargs(theme.FONT_HEAVY))
                y -= 0.045
            btn = _styled_button("CHOOSE", self, (x, card_bottom + 0.045), (0.24, 0.06),
                                 on_click=Func(self.pick, key))
            self.choose_buttons[key] = btn

        _styled_button("FIGHT!", self, (0, card_bottom - 0.09), (0.26, 0.09),
                       accent=theme.VICTORY, on_click=self._confirm, text_scale=1.2)
        _styled_button("< BACK", self, (-0.85, 0.46), (0.15, 0.06), on_click=on_back)

    def pick(self, key):
        self.chosen = key
        other = [k for k in CHAR_INFO if k != key][0]
        for k, lbl in self.labels.items():
            selected = k == key
            lbl.color = theme.VICTORY if selected else theme.TEXT
            self.cards[k].color = theme.PANEL_LIGHT if selected else theme.PANEL
            self.cards[k].edge.color = theme.VICTORY if selected else theme.BORDER
            self.glows[k].color = color.rgba32(255, 207, 74, 45) if selected else color.rgba32(0, 0, 0, 0)
            theme.safe_animate_scale(
                self.choose_buttons[k], self.choose_buttons[k].scale * (1.08 if selected else 1.0),
                duration=0.12)
        self.hint.color = theme.ACCENT_P1
        self.hint.text = (f"P1: {CHAR_INFO[key]['label']}   P2: {CHAR_INFO[other]['label']}"
                           f"   -- hit FIGHT! when ready")
        self.on_pick(key, other)

    def _confirm(self):
        if self.chosen is None:
            self.hint.text = "PICK A FIGHTER FIRST!"
            self.hint.color = theme.DANGER
            return
        other = [k for k in CHAR_INFO if k != self.chosen][0]
        self.on_confirm(self.chosen, other)


class OnlineMenu(Entity):
    """LAN 1v1 setup screen -- reached from MainMenu's ONLINE 1V1 button.
    Two flows on one screen: HOST opens a listening socket right away (see
    src/game/netcode.py) and this screen just shows a "waiting" status;
    JOIN reveals an IP input field. This class has no idea how the actual
    networking works -- play_game.py's main() owns the NetHost/NetClient
    and pushes whatever status string is currently true in via set_status(),
    same separation match.py/entities.py already keep between simulation
    and rendering."""

    def __init__(self, on_host, on_join, on_back):
        super().__init__(parent=camera.ui)
        theme.section_title(self, "ONLINE 1V1", position=(0, 0.4), scale=2.6)
        Text("Both laptops must be on the same WiFi/router", parent=self,
             position=(0, 0.32), origin=(0, 0), scale=0.95, color=theme.TEXT_MUTED,
             **theme.font_kwargs(theme.FONT_BODY))

        theme.panel(self, (0.7, 0.42), (0, 0.02), z=0.5)

        _styled_button("HOST GAME", self, (-0.17, 0.12), (0.28, 0.09),
                       accent=theme.ACCENT_P2, on_click=on_host)
        Text("Hosts and shows your IP for the\nother player to type in.",
             parent=self, position=(-0.17, 0.02), origin=(0, 0), scale=0.72,
             color=theme.TEXT_MUTED, **theme.font_kwargs(theme.FONT_BODY))

        _styled_button("JOIN GAME", self, (0.17, 0.12), (0.28, 0.09),
                       on_click=self._show_join)
        Text("Type in the IP address shown\non the host's screen.",
             parent=self, position=(0.17, 0.02), origin=(0, 0), scale=0.72,
             color=theme.TEXT_MUTED, **theme.font_kwargs(theme.FONT_BODY))

        self.join_row = Entity(parent=self, enabled=False)
        # InputField hardcodes highlight_color=color.black internally --
        # passing it again collides as a duplicate kwarg (see NameEntry's
        # own field above), so only `color` is themed here too.
        self.ip_field = InputField(parent=self.join_row, scale=(0.34, 0.075),
                                    position=(-0.06, -0.14), character_limit=15,
                                    active=False, color=theme.PANEL)
        if self.ip_field.text_field.text_entity:
            self.ip_field.text_field.text_entity.font = theme.FONT_HEAVY
        self.ip_field.submit_on = ["enter"]
        self.ip_field.on_submit = self._submit_join
        self._on_join = on_join
        _styled_button("CONNECT", self.join_row, (0.23, -0.14), (0.16, 0.075),
                       accent=theme.VICTORY, on_click=self._submit_join)

        # Text's wordwrap setter reads self.raw_text, which only gets set the
        # first time self.text is assigned a non-empty string -- passing
        # wordwrap= alongside an initially-empty text="" crashes at
        # construction (confirmed: AttributeError 'Text' object has no
        # attribute 'raw_text'). Start with a real string so that
        # assignment runs, then clear it back to empty.
        self.status = Text(" ", parent=self, position=(0, -0.3), origin=(0, 0),
                            scale=1.0, color=theme.ACCENT_P1, wordwrap=40,
                            **theme.font_kwargs(theme.FONT_BODY))
        self.status.text = ""

        _styled_button("< BACK", self, (-0.85, 0.46), (0.15, 0.06), on_click=on_back)

    def _show_join(self):
        self.join_row.enabled = True
        self.ip_field.active = True

    def _submit_join(self):
        ip = (self.ip_field.text_field.text or "").strip()
        if ip:
            self._on_join(ip)

    def set_status(self, text, color_=None):
        self.status.text = text
        self.status.color = color_ or theme.ACCENT_P1


class ProfileScreen(Entity):
    """Standalone player-profile screen reachable from the main menu.
    Read-only -- there's nothing to edit here yet beyond the name chosen at
    first launch (NameEntry), consistent with the "local profile, no
    accounts system" scope this was built to."""

    def __init__(self, profile, on_back):
        super().__init__(parent=camera.ui)
        theme.section_title(self, "FIGHTER PROFILE", position=(0, 0.44), scale=2.6)

        card = theme.panel(self, (0.7, 0.65), (0, -0.02), z=0.5)

        # avatar placeholder -- no uploaded/generated avatar art exists in
        # this project, so this is a themed initial-in-a-frame instead of a
        # blank gap or a fabricated image.
        Entity(parent=self, model="quad", color=theme.PANEL_LIGHT, scale=(0.16, 0.16),
               position=(-0.24, 0.22), z=0.4)
        initial = profile.username[:1].upper() if profile.username else "?"
        Text(initial, parent=self, position=(-0.24, 0.22), origin=(0, 0), scale=3.2,
             color=theme.ACCENT_P1, **theme.font_kwargs(theme.FONT_DISPLAY))

        Text(profile.username.upper(), parent=self, position=(-0.1, 0.26), origin=(-0.5, 0),
             scale=2.0, color=theme.TEXT, **theme.font_kwargs(theme.FONT_DISPLAY))
        record_color = theme.VICTORY if profile.win_rate >= 0.5 and profile.matches_played else theme.TEXT_MUTED
        Text(f"{profile.matches_played} MATCHES PLAYED", parent=self, position=(-0.1, 0.18),
             origin=(-0.5, 0), scale=1.0, color=theme.TEXT_MUTED, **theme.font_kwargs(theme.FONT_BODY))

        stats = [
            ("WINS", str(profile.wins), theme.ACCENT_P1),
            ("LOSSES", str(profile.losses), theme.DANGER),
            ("WIN RATE", f"{profile.win_rate*100:.0f}%", record_color),
        ]
        x = -0.24
        for label, value, c in stats:
            Text(value, parent=self, position=(x, -0.05), origin=(0, 0), scale=2.4,
                 color=c, **theme.font_kwargs(theme.FONT_DISPLAY))
            Text(label, parent=self, position=(x, -0.15), origin=(0, 0), scale=0.85,
                 color=theme.TEXT_MUTED, **theme.font_kwargs(theme.FONT_BODY))
            x += 0.24

        fav = profile.favorite_fighter
        fav_label = CHAR_INFO.get(fav, {}).get("label", "--") if fav else "--"
        Text("FAVORITE FIGHTER", parent=self, position=(0, -0.26), origin=(0, 0),
             scale=0.9, color=theme.TEXT_MUTED, **theme.font_kwargs(theme.FONT_BODY))
        Text(fav_label, parent=self, position=(0, -0.32), origin=(0, 0),
             scale=1.6, color=theme.ACCENT_P2, **theme.font_kwargs(theme.FONT_HEAVY))

        _styled_button("< BACK TO MENU", self, (0, -0.42), (0.3, 0.075), on_click=on_back)
