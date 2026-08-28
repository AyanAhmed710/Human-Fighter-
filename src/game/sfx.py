"""Sound-effect playback.

punch/kick/shoot/click use the real clips handed over in assets/sound/
sound_effects/, and the round-1/2/3/critical-health/KO callouts use the real
narrator lines in assets/sound/Game Narrator sound/ (CLIP_FILES below). There
was also a "game start" narrator line at one point -- removed per user
request, both the CLIP_FILES entry and its two call sites in play_game.py.
Everything else (hit/ko/fight-flash/match_win/heartbeat) has no
user-provided clip yet, so those still fall back to the synthesized WAVs in
assets/sfx/ (see tools/generate_sfx.py -- sine/noise placeholders, no
external sample library). Drop a real file in and add it to CLIP_FILES to
replace any of those the same way punch/kick/shoot were.

Loads every clip via Panda3d's own loader instead of Ursina's Audio(name)
constructor. Audio(name) only accepts a bare filename and re-searches
application.asset_folder (Ursina's default is '.') by globbing
f'**/{name}' under it for a match -- handed a full absolute path, that glob
pattern (the literal path string, dots and all) never matches anything
under '.', so it silently fails to find files that plainly exist. Loading
the panda3d sound object ourselves and passing it to Audio() as
`sound_file_name` skips that lookup entirely: Audio's clip_setter stores
any non-str value as-is. This also means Panda3d's ffmpeg-backed decoder
handles the .mp3 clips fine, not just the synthesized .wav ones (confirmed:
loader.loadSfx() decodes both) -- no format conversion needed.

Each clip is loaded once and cached, then replayed via play() -- creating a
fresh Audio() per hit would re-decode the file from disk every frame an
attack lands.
"""
from pathlib import Path

from panda3d.core import Filename
from ursina import Audio

SFX_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "sfx"
SOUND_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "sound"

NARRATOR_DIR = SOUND_DIR / "Game Narrator sound"

CLIP_FILES = {
    "punch": SOUND_DIR / "sound_effects" / "punch_soundeffect.mp3",
    "kick": SOUND_DIR / "sound_effects" / "kicking.mp3",
    "shoot": SOUND_DIR / "sound_effects" / "shooting.mp3",
    "click": SOUND_DIR / "sound_effects" / "button_press.mp3",
    # filenames below were "Round 1.aac"/"Round 2.aac"/"Round 3.aac" -- the
    # user re-exported these as "round1 final.aac" etc. and the old files are
    # gone, same silent-breakage shape as the background-music swap (see
    # _find_music_file's docstring): loader.loadSfx() on a path that no
    # longer exists just fails quietly, no exception, so this needs to track
    # whatever the real current filenames are, not guess.
    "narrator_round_1": NARRATOR_DIR / "round1 final.aac",
    "narrator_round_2": NARRATOR_DIR / "round2 final.aac",
    "narrator_round_3": NARRATOR_DIR / "round3 final.aac",
    "narrator_critical_health": NARRATOR_DIR / "Critical Health.aac",
    "narrator_ko": NARRATOR_DIR / "KO.aac",
}
MUSIC_DIR = SOUND_DIR / "background_music"


def _find_music_file() -> Path | None:
    """Resolves whatever audio file is actually sitting in background_music/
    instead of a hardcoded filename -- the hardcoded exact name broke silently
    the moment the user swapped in a replacement track that saved under a
    different name (Windows appended " (1)" since the old file was still
    there at save time), and loader.loadSfx() on a path that doesn't exist
    just fails quietly rather than raising, so start_music() looked like it
    worked but played nothing. Picks the first .mp3/.wav/.aac/.ogg file found,
    sorted so the result is at least deterministic if more than one ever ends
    up in there."""
    if not MUSIC_DIR.is_dir():
        return None
    for ext in ("*.mp3", "*.wav", "*.aac", "*.ogg"):
        matches = sorted(MUSIC_DIR.glob(ext))
        if matches:
            return matches[0]
    return None

_ROUND_ANNOUNCEMENT_CLIPS = {1: "narrator_round_1", 2: "narrator_round_2", 3: "narrator_round_3"}

# Narrator voice lines were getting buried under the sfx/music -- panda3d's
# setVolume isn't clamped to 1.0 (unlike a lot of audio APIs), it's a plain
# linear gain multiplier straight through to OpenAL, so > 1.0 genuinely
# boosts the clip instead of doing nothing (confirmed: play() with volume=1.6
# runs and reaches PLAYING status same as 1.0, no error/clamp). Can't hear it
# myself to fine-tune for clipping -- if 1.6 is too loud/distorted, dial it
# back; if still not loud enough, push it higher.
NARRATOR_VOLUME = 1.6

# OpenAI's TTS output (commentator.py) is mastered noticeably quieter than
# the user's own pre-recorded narrator clips -- same 1.0-is-not-a-ceiling
# gain trick, just pushed further since this is a different, quieter source.
# Bump this one specifically if AI commentary is still too quiet/loud; it
# doesn't affect the real narrator lines above.
COMMENTARY_VOLUME = 6.5

_clips = {}            # name -> Audio, lazily loaded on first play()
_heartbeat = None       # the one looping low-health Audio, lazily created
_heartbeat_on = False
_music = None           # the one looping background-music Audio, lazily created
_music_on = False
_active_voice = None    # whichever "spoken line" Audio (narrator clip or AI
                         # commentary) is currently playing, if any -- see
                         # _play_voice() below


def _clip_path(name: str) -> Path:
    return CLIP_FILES.get(name, SFX_DIR / f"{name}.wav")


def _load_sound(path: Path):
    p = Filename.fromOsSpecific(str(path))
    return loader.loadSfx(p)  # noqa: F821 -- `loader` is injected into
                               # builtins by Panda3d's ShowBase (Ursina())
                               # once the app exists; by the time anything
                               # here actually runs (never at import time)
                               # that app is already up. Same pattern as
                               # play_game.py's own `loader.loadModel(...)`.


def _play_voice(clip: Audio):
    """Plays a "spoken line" clip (narrator lines, AI commentary) -- cuts off
    whatever spoken line is already playing first, so two voices never
    overlap/talk over each other. Percussive sfx (hit/ko impact, punch/kick/
    shoot, click, music) don't go through this -- those are meant to layer
    with everything, including with a voice line."""
    global _active_voice
    if _active_voice is not None:
        _active_voice.stop(destroy=False)
    clip.play()
    _active_voice = clip


def play_file(path, volume: float = COMMENTARY_VOLUME) -> Audio:
    """Plays an arbitrary audio file straight from disk, uncached, no name
    lookup -- for one-off dynamically generated clips (src/game/
    commentator.py's AI commentary lines) where every line is different text,
    so there's nothing sensible to cache under a fixed name like _get() does.
    Returns the Audio so the caller can hang onto it if it wants to."""
    clip = Audio(_load_sound(Path(path)), autoplay=False, loop=False, volume=volume)
    _play_voice(clip)
    return clip


def _get(name: str) -> Audio:
    clip = _clips.get(name)
    if clip is None:
        clip = Audio(_load_sound(_clip_path(name)), autoplay=False, loop=False)
        _clips[name] = clip
    return clip


def play(name: str, volume: float = 1.0):
    clip = _get(name)
    clip.volume = volume
    clip.play()


def play_round_announcement(round_num: int, volume: float = NARRATOR_VOLUME):
    """Real narrator voice line for rounds 1-3 (assets/sound/Game Narrator
    sound/); falls back to the synthesized bell for any round beyond that
    (can't happen with ROUNDS_TO_WIN=2 today, but stays correct if that
    ever changes). Goes through _play_voice -- see its docstring -- so it
    can't overlap with a KO/critical-health line or AI commentary."""
    clip = _get(_ROUND_ANNOUNCEMENT_CLIPS.get(round_num, "round_start"))
    clip.volume = volume
    _play_voice(clip)


def play_ko_announcement(volume: float = NARRATOR_VOLUME):
    """Real narrator "K.O.!" voice line. Deliberately NOT run through
    _play_voice's cutoff for the synthesized "ko" impact sfx (that's a
    percussive hit sound, meant to layer under this), but IS exclusive
    against any other spoken line -- see _play_voice's docstring."""
    clip = _get("narrator_ko")
    clip.volume = volume
    _play_voice(clip)


def play_critical_health_alert(volume: float = NARRATOR_VOLUME):
    """Real narrator low-health callout -- fired once on the edge transition
    into low health (see play_game.py's _update_low_health), not every frame
    like the looping heartbeat alarm. Exclusive against other spoken lines,
    same as the others above -- see _play_voice's docstring."""
    clip = _get("narrator_critical_health")
    clip.volume = volume
    _play_voice(clip)


def start_low_health_alarm():
    """Idempotent -- safe to call every frame while a player is low; only
    actually (re)starts the loop the first time."""
    global _heartbeat, _heartbeat_on
    if _heartbeat_on:
        return
    if _heartbeat is None:
        # was _load_sound("heartbeat") -- a bare string, not the resolved
        # SFX_DIR/heartbeat.wav path _clip_path() builds everywhere else, so
        # this would have failed to find the file the first time a player
        # actually dropped low (never exercised by any boot/flow test since
        # those force health straight to 0, skipping the low-health range).
        _heartbeat = Audio(_load_sound(_clip_path("heartbeat")), autoplay=False, loop=True, volume=0.55)
    _heartbeat.play()
    _heartbeat_on = True


def stop_low_health_alarm():
    global _heartbeat_on
    if _heartbeat is not None and _heartbeat_on:
        # destroy=False -- Audio.stop()'s default destroys the entity, which
        # would wipe our cached heartbeat clip and force a reload next time.
        _heartbeat.stop(destroy=False)
    _heartbeat_on = False


def start_music(volume: float = 0.22):
    """Idempotent -- safe to call from main() once at startup. Volume kept
    low so it sits under SFX/voice rather than competing with them; loops for
    the whole session (track length varies with whatever's actually dropped
    in background_music/)."""
    global _music, _music_on
    if _music_on:
        return
    if _music is None:
        music_file = _find_music_file()
        if music_file is None:
            # was a hard KeyError/silent no-op before -- print so a missing/
            # misnamed track shows up in the console instead of just "no
            # music, no idea why".
            print(f"[sfx] no background music file found in {MUSIC_DIR} -- skipping")
            return
        _music = Audio(_load_sound(music_file), autoplay=False, loop=True, volume=volume)
    _music.play()
    _music_on = True


def stop_music():
    global _music_on
    if _music is not None and _music_on:
        _music.stop(destroy=False)
    _music_on = False
