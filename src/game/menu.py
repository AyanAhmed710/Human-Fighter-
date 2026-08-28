"""Coded UI screens -- main menu and character select. Built entirely from
Ursina's stock Button/Entity/Text primitives (no external UI art pack
needed). These classes only own the flat camera.ui overlay; the actual 3D
"showroom" -- the two live character models standing still in the arena,
each throwing a punch when its CHOOSE button is clicked -- is spawned/
despawned by play_game.py's main(), which also owns the shared 3D camera
and the dark vignette that makes each screen read as a genuinely different
place instead of the same view with different text on top.

CharacterSelect's Fight button is ALWAYS clickable (not enabled/disabled)
and validates on click instead -- an earlier version toggled
`button.enabled` on pick/no-pick, which could leave the button's collider in
a stale state and made it look "broken" (dead clicks with no feedback at
all). This version can never dead-click: unpicked click -> clear on-screen
warning; picked click -> always fires.
"""
from ursina import Button, Entity, Func, Text, camera, color

from src.game import sfx


def _clicky(on_click):
    """Wraps a Button's on_click so every button in these screens gets the
    same short UI-click sfx for free, without repeating sfx.play("click")
    at every call site."""
    def wrapped():
        sfx.play("click", volume=0.5)
        on_click()
    return wrapped

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


class MainMenu(Entity):
    def __init__(self, on_start, on_quit):
        super().__init__(parent=camera.ui)

        Text("HUMAN FIGHTER", parent=self, position=(0, 0.34), origin=(0, 0),
             scale=3.4, color=color.orange)
        Text("A GESTURE-CONTROLLED DUEL", parent=self, position=(0, 0.26), origin=(0, 0),
             scale=1.1, color=color.rgba32(255, 200, 150, 200))

        Button("Start", parent=self, position=(0, 0.03), scale=(0.3, 0.08),
               color=color.azure.tint(-.2), highlight_color=color.azure,
               pressed_color=color.azure.tint(.3), on_click=_clicky(on_start))
        Button("Settings", parent=self, position=(0, -0.1), scale=(0.3, 0.08),
               color=color.gray, highlight_color=color.light_gray,
               on_click=_clicky(self.toggle_settings))
        Button("Quit", parent=self, position=(0, -0.23), scale=(0.3, 0.08),
               color=color.red.tint(-.2), highlight_color=color.red,
               on_click=_clicky(on_quit))

        self.settings_panel = Entity(parent=self, enabled=False)
        Entity(parent=self.settings_panel, model="quad", color=color.rgba32(0, 0, 0, 190),
               scale=(0.62, 0.42), position=(0, -0.08))
        Text("CONTROLS", parent=self.settings_panel, position=(-0.28, 0.05),
             scale=1.4, color=color.yellow)
        Text("Player 1: 1=punch  2=kick  3=shoot", parent=self.settings_panel,
             position=(-0.28, -0.02), scale=1.1)
        Text("Player 2: 8=punch  9=kick  0=shoot", parent=self.settings_panel,
             position=(-0.28, -0.08), scale=1.1)
        Text("R=restart   ESC/Q=quit   C=free camera", parent=self.settings_panel,
             position=(-0.28, -0.14), scale=1.1)
        Text("Best of 3 rounds -- 2 round wins takes the match",
             parent=self.settings_panel, position=(-0.28, -0.2), scale=1.0,
             color=color.light_gray)
        Button("Close", parent=self.settings_panel, position=(0, -0.26),
               scale=(0.2, 0.06), color=color.gray, on_click=_clicky(self.toggle_settings))

    def toggle_settings(self):
        self.settings_panel.enabled = not self.settings_panel.enabled


class CharacterSelect(Entity):
    def __init__(self, on_pick, on_confirm, on_back):
        super().__init__(parent=camera.ui)
        self.on_pick = on_pick      # called (chosen_key, other_key) live, for
                                     # the 3D preview to react (spin/scale-pop)
        self.on_confirm = on_confirm  # called (p1_key, p2_key) when Fight! fires
        self.chosen = None

        Text("SELECT YOUR FIGHTER", parent=self, position=(0, 0.47), origin=(0, 0),
             scale=2.1, color=color.orange)
        self.hint = Text("Player 1: choose a fighter below", parent=self,
                          position=(0, 0.395), origin=(0, 0), scale=1.15, color=color.azure)

        self.labels = {}
        self.choose_buttons = {}
        x = -0.34
        for key, info in CHAR_INFO.items():
            # explicit z so this card backdrop always draws behind the text
            # sitting on top of it -- without a distinct z, this quad and
            # play_game.py's full-screen vignette (also semi-transparent,
            # also ~z=0) tie on depth, and Panda3D's cull-bin sort for tied
            # depth isn't guaranteed stable frame to frame -- alpha-blend
            # order could flip which one draws "on top" from one frame to
            # the next, reading as text flickering dim/bright. z=0.5 pins it
            # firmly between the vignette (z=1, furthest back) and the text
            # (default z=0, frontmost) so there's nothing left to tie on.
            Entity(parent=self, model="quad", color=color.rgba32(12, 12, 18, 195),
                   scale=(0.32, 0.36), position=(x, -0.25), z=0.5)
            lbl = Text(info["label"], parent=self, position=(x, -0.11), origin=(0, 0),
                       scale=1.4, color=color.white)
            self.labels[key] = lbl
            Text(info["desc"], parent=self, position=(x - 0.14, -0.165), scale=0.78,
                 wordwrap=22, color=color.white)
            y = -0.275
            for stat_label, val in POWER_LINES:
                Text(f"{stat_label}  {val}", parent=self, position=(x - 0.1, y), scale=0.85,
                     color=color.white)
                y -= 0.045
            btn = Button("CHOOSE", parent=self, position=(x, -0.46), scale=(0.24, 0.06),
                        color=color.azure.tint(-.25), highlight_color=color.azure,
                        on_click=_clicky(Func(self.pick, key)))
            self.choose_buttons[key] = btn
            x += 0.68

        Button("FIGHT!", parent=self, position=(0, -0.46), scale=(0.24, 0.085),
              color=color.lime.tint(-.2), highlight_color=color.lime,
              pressed_color=color.lime.tint(.3), on_click=_clicky(self._confirm))
        Button("< Back", parent=self, position=(-0.85, 0.46), scale=(0.15, 0.06),
              color=color.gray, highlight_color=color.light_gray, on_click=_clicky(on_back))

    def pick(self, key):
        self.chosen = key
        other = [k for k in CHAR_INFO if k != key][0]
        for k, lbl in self.labels.items():
            lbl.color = color.yellow if k == key else color.white
        self.choose_buttons[key].color = color.lime
        self.choose_buttons[other].color = color.azure.tint(-.25)
        self.hint.color = color.azure
        self.hint.text = (f"P1: {CHAR_INFO[key]['label']}   P2: {CHAR_INFO[other]['label']}"
                           f"   -- hit FIGHT! when ready")
        self.on_pick(key, other)

    def _confirm(self):
        if self.chosen is None:
            self.hint.text = "Pick a fighter first!"
            self.hint.color = color.red
            return
        other = [k for k in CHAR_INFO if k != self.chosen][0]
        self.on_confirm(self.chosen, other)
