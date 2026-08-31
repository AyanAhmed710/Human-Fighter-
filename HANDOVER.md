# Handover notes — for a fresh Claude Code session

Written by Claude at the end of a long session so a new session (different
machine, no shared conversation history) can pick this project up fast.
Read this before making changes — several of these are non-obvious and cost
real debugging time to find the first time.

## What this project is

A gesture-controlled 2-player fighting game. Python, Ursina (wraps
Panda3D) for the engine, MediaPipe + a trained XGBoost classifier for
webcam gesture recognition (punch/kick/shoot). Two real Mixamo-rigged
characters (Warrok, Vampire) with real mocap animation clips, a lava-arena
3D scene, and a from-scratch 2D UI (menu/character-select/HUD/VS/victory
screens) built on Ursina primitives, no external UI framework.

**Not a web app.** No React/HTML/backend/API/database anywhere — a single
desktop Python process. Don't assume web-dev patterns apply.

## How to run it

```
python scripts/play_game.py --keyboard   # no webcam needed, test mode
python scripts/play_game.py              # real gesture control, cams 0+1
```
`--keyboard`: P1 = 1/2/3=punch/kick/shoot, 4=hold-to-block. P2 = 8/9/0,
7=hold-to-block (only relevant for local 2-player testing on one keyboard
— online play only ever uses the P1 cluster, see netcode section below).
R = restart, ESC/Q = quit, C = free debug camera.

**Setup**: Python 3.10 (mediapipe has no wheel for newer versions on this
project's pinned `mediapipe==0.10.21`). `requirements.txt` covers the
ML-training side of the repo but is missing 3 packages the actual game
needs — also run:
```
pip install ursina==7.0.0 panda3d==1.10.16 panda3d-simplepbr==0.13.1
```
`.env` (gitignored, `OPENAI_API_KEY=...`) is optional — only enables AI
live commentary; the game runs fine without it (self-disables gracefully).
`save/profile.json` (gitignored) is the local player's name/win-loss
record — recreated fresh via a first-launch name-entry screen if absent.

## Current branch state

`feature/commentary-crit-multiplayer-prep`, well ahead of `main` (main
predates almost everything below). This session's work, roughly in order:
critical hits + AI commentary + round system + menu overhaul (base of the
branch) → local player profiles → full visual/theme overhaul (menu, HUD,
VS screen, victory screen) → **LAN 1v1 multiplayer** → **block stance**
→ **animated crit-hit stun sequence** → several balance/bugfix passes.
Nothing has been merged to `main` yet — that's a call for the human to make
whenever they're ready, not something to do unprompted.

## Architecture map

- `src/game/match.py` — **engine-agnostic** game rules (no ursina import,
  on purpose, so it's independently unit-testable). `Player`/`Match`
  classes: health, action stats, crit rolls, block-stamina, KO/win
  detection. Every renderer reads FROM this, never decides rules itself.
- `src/game/real_entities.py` — the real Mixamo-rig renderer
  (`RealFighterEntity`). Owns a Panda3D `Actor` directly (not Ursina's
  Entity wrapper — Actor's clip API has no Ursina equivalent). All the
  animation state machines (idle/attack/hit/block/crit-stun) live here,
  driven by `match.Player` state each frame via `.sync()`.
- `src/game/entities.py` — procedural low-poly fallback rig (`--procedural`
  flag), no external models. Kept in sync feature-wise with real_entities
  where cheap to do (e.g. block pose), not where it'd need real clips.
- `src/game/player_input.py` — one background thread per player: opens a
  webcam, runs MediaPipe pose+hands, feeds a trained XGBoost model,
  produces punch/kick/shoot events non-blocking-polled by the game loop.
  Also does the block-stance elbow-angle geometry (separate from the
  ML classifier — a plain 3-point-angle heuristic, not another model).
- `src/game/netcode.py` — LAN 1v1 multiplayer. **Replicated-input design,
  not client-server state broadcast**: both machines run the FULL game
  independently; only actions (punch/kick/shoot/block/restart) cross the
  wire. Read its module docstring before touching — explains why this
  approach was chosen over the more typical host-authoritative pattern.
  Host is always P1 (their real profile), joiner is always P2.
- `src/game/menu.py` — every 2D screen (NameEntry, MainMenu, CharacterSelect,
  OnlineMenu, ProfileScreen), all built from Ursina Button/Entity/Text.
- `src/game/theme.py` — shared design system (colors/fonts/widget
  builders) every screen pulls from, so they read as one coherent look.
- `src/game/profile.py` — local-machine-only player identity
  (`save/profile.json`). Explicitly NOT an accounts system — one profile
  per machine, no login, per an early design decision in this session.
- `scripts/play_game.py` — the actual Ursina app entrypoint. Wires
  everything above together; owns the phase state machine (intro → go →
  fight → round_end → match_end) and all the game-feel juice (hitstop,
  camera shake, hit-streak counter, crit banner).
- `tools/convert_*.py` — Blender scripts (run via a portable Blender at
  `tools/blender-5.2.1-windows-x64/blender.exe`, gitignored/not in the
  repo — download separately if converting more animations) that turn raw
  Mixamo FBX (`assets/mixamo/`, gitignored — raw sources, not needed to
  run the game) into the anim-only `.glb` files the game actually loads
  (`assets/models/*.glb`, tracked in git).

## Non-obvious gotchas (cost real debugging time to find)

- **mediapipe must be imported before ursina**, every entrypoint — ursina/
  panda3d's native DLLs shadow ones mediapipe needs on Windows, causing a
  cryptic `DLL load failed while importing _framework_bindings` otherwise.
- **Ursina z-axis**: LOWER z = closer to camera = renders in front. Easy
  to get backwards. Siblings at the same default z=0 can flicker on
  ordering (Panda3D's alpha-blend sort isn't stable frame-to-frame) — give
  overlapping UI layers explicit, different z values.
- **Never call `entity.animate_scale(...)`/`animate_x(...)` etc. raw.**
  Use `theme.safe_animate_scale(entity, ...)` / `theme.safe_animate(entity,
  "animate_x", ...)` instead. Raw calls crash the whole game
  (`AssertionError: !is_empty()` deep in panda3d) if the entity's own
  screen gets destroyed while the animation is still mid-flight — this bit
  us twice already from two different call sites before the pattern got
  applied everywhere. If you add a new animated UI element, use the safe
  wrapper from the start.
- **Panda3D `Actor` anim-only `.glb` exports must NOT include the mesh** —
  only the skeleton + action. Including it silently binds the clip to its
  own separate Character instance instead of the model's, so it "plays"
  (no error) but the actual rendered character never visibly moves. See
  `tools/convert_hit_reaction.py`'s docstring for the full story.
- **30fps is assumed throughout** the animation timing pipeline (frame
  counts ÷ 30 = seconds) — match.py's stun/lock durations are hand-derived
  from real clip frame counts on this assumption. If you add a clip
  exported at a different frame rate, the timing math needs adjusting.
- **`match.py`'s `CRIT_STUN_DURATION` must always equal the real total
  runtime of real_entities.py's crit animation sequence** (HitReact full +
  Stunned + GettingUp-trimmed-to-frame-156, currently played at 3.0x speed
  = 3.0s exactly). Change one, change the other, or the stun either cuts
  the animation off early or leaves an awkward extra hold.
- **`Player.is_blocking` vs `Player.guard_up`**: `is_blocking` is the raw
  external input (set every frame by whatever's driving that player —
  camera/keyboard/network). `guard_up` is the ENFORCED flag (raw input AND
  not currently locked out by the 5s-hold/3s-cooldown block-stamina rule)
  — `_apply_damage` and both renderers check `guard_up`, never
  `is_blocking` directly. Keep this split if you touch blocking logic.
- **The block elbow-angle heuristic is intentionally simple** (just a
  60-120° shoulder-elbow-wrist angle check, per an explicit user request
  to keep it that way) — no wrist-height gate on top. A real false
  positive (hands on hips, etc.) is a known, accepted tradeoff, not a bug.

## Known open/flagged items (not fixed, just noted)

- The crit-stun animation now plays at 3.0x speed (compressed from a real
  9.0s sequence down to 3.0s per explicit requests) — flagged to the user
  that 3x is a big speedup and might look unnaturally fast on a clip built
  for real-time pacing. Not yet confirmed good or bad by an actual
  playtest — worth asking if it still feels right.
- Cross-resolution UI clipping at narrower aspect ratios (1440×900
  specifically) was found during an earlier visual QA pass and explicitly
  deferred by the user ("skip these for now") — never resumed. A few
  elements sit near the ±0.85-0.9 edge of `camera.ui`, which clips at
  narrower-than-16:9 aspect ratios. Search history/git blame around
  `menu.py`'s username/back-button positions and `play_game.py`'s
  WebcamPreview/name-text positions if this comes up again.
- No merge to `main` has happened — everything above lives only on
  `feature/commentary-crit-multiplayer-prep`.
