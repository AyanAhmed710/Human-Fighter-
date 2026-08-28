"""LLM-generated live fight commentary.

An OpenAI chat model writes one short, hype WWE-style line reacting to
something that just happened in the match (a landed hit, someone going
critical, a KO) -- factoring in current health and each player's recent
move history so it can call out a "style" ("keeps spamming kicks" etc.), not
just the single event. OpenAI's TTS then speaks that line, and it plays
through the same Audio pipeline sfx.py already uses for everything else.

Entirely optional and silently self-disabling: with no OPENAI_API_KEY in the
environment (or the openai package missing, or the API rejecting the key),
self.enabled stays False and every notify_*() call turns into a no-op --
the game plays exactly the same as before this file existed, just quieter.

Both real OpenAI calls here (a chat completion, then TTS) take real network
time -- likely 1-3 real seconds combined, not something Ursina's frame loop
can wait on. They run entirely on one background worker thread, same shape
as src/game/player_input.py's camera-capture thread: the main thread only
ever pushes small text events in and polls finished audio-file paths out,
never blocks on either. The one thing that must stay on the MAIN thread is
actually loading + playing that audio (Panda3D's loader/Audio aren't safe to
touch from a background thread) -- see play_game.py's Game._update_commentary.
"""
import os
import queue
import random
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

from dotenv import load_dotenv

# Reads C:\Data_Tekken\.env (repo root, two levels up from this file) into
# os.environ if it's not already set there -- doesn't override a real OS env
# var if one's already present. Without this, OPENAI_API_KEY only worked if
# set in the exact shell session play_game.py was launched from; a .env file
# next to the project is the more common way people actually hand over keys,
# so this needs to be read explicitly, nothing does that for us automatically.
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

MIN_INTERVAL_SECONDS = 2.5     # floor between commentary lines, however triggered -- lowered
                                # from 6.0 for more frequent lines; still enough to let a short
                                # line finish (see max_tokens/word-count below) before the next
                                # one can fire, and _play_voice still cuts off cleanly either way
ACTION_TRIGGER_CHANCE = 0.75   # raised from 0.4 -- most landed hits now get a line
HISTORY_LEN = 6                # recent-action memory per player, feeds the "keeps spamming
                                # kicks" style read into the prompt

# Both picked for low latency over max quality -- this is meant to feel like
# a live reaction, a few-hundred-ms slower model isn't worth it here. If
# these model names ever get retired, swap them for whatever's current;
# nothing else here depends on the exact name.
CHAT_MODEL = "gpt-4o-mini"
TTS_MODEL = "tts-1"
TTS_VOICE = "onyx"  # deep, hype-announcer-ish built-in voice

SYSTEM_PROMPT = (
    "You are a hype WWE-style live commentator calling a 2-player gesture-"
    "controlled fighting game. React to the single event you're given with "
    "ONE short, punchy, over-the-top burst -- under 8 words, like a real "
    "shoutcaster's quick reaction, not a full sentence. No stage "
    "directions, no emojis, no asterisks, no quotation marks -- plain "
    "spoken text only, it goes straight to text-to-speech. Use the "
    "fighters' names when it fits."
)


class Commentator:
    """One instance for the whole match. p1_name/p2_name are display names
    (the chosen character names, e.g. "Warrok"/"Vampire") used in prompts so
    the commentary can call fighters by name like a real broadcast would."""

    def __init__(self, p1_name: str, p2_name: str):
        self.p1_name = p1_name
        self.p2_name = p2_name
        self.enabled = bool(os.environ.get("OPENAI_API_KEY"))
        self._history = {"p1": deque(maxlen=HISTORY_LEN), "p2": deque(maxlen=HISTORY_LEN)}
        self._last_fire = 0.0
        # maxsize=1: only ever one commentary request in flight. A second
        # notify() arriving while the worker's still busy on the last one
        # just gets dropped (queue.Full below) rather than queuing up stale
        # reactions to things that already scrolled past.
        self._in_queue: "queue.Queue[str]" = queue.Queue(maxsize=1)
        self._out_queue: "queue.Queue[tuple[str, Path]]" = queue.Queue()
        self._client = None
        self._thread = None

        if not self.enabled:
            print("[commentator] OPENAI_API_KEY not set -- AI commentary disabled")
            return

        try:
            import openai
            self._client = openai.OpenAI()
        except Exception as exc:  # missing package, malformed key, etc. -- never let this
                                    # take the whole game down over an optional feature
            print(f"[commentator] disabled -- couldn't init OpenAI client: {exc}")
            self.enabled = False
            return

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _name(self, side: str) -> str:
        return self.p1_name if side == "p1" else self.p2_name

    def _style_note(self, side: str) -> str:
        hist = self._history[side]
        if len(hist) >= 3 and len(set(list(hist)[-3:])) == 1:
            return f"{self._name(side)} has thrown nothing but {hist[-1]}s the last few moves -- spamming it."
        if hist:
            return f"{self._name(side)}'s recent moves, oldest first: {', '.join(hist)}."
        return ""

    def notify_action(self, side: str, action: str, attacker_hp: int, defender_hp: int,
                       is_crit: bool = False):
        """side/action: whoever's attack just actually landed (dealt
        damage), not just whoever pressed a button -- called from
        play_game.py's hit-detection block, which already only fires on a
        real health drop. is_crit: true when match.py rolled a critical hit
        (1.75x damage, 3s stun) -- a crit always gets a line regardless of
        ACTION_TRIGGER_CHANCE, same as low-health/KO, since it's the rarer,
        more exciting event."""
        if not self.enabled:
            return
        self._history[side].append(action)
        if not is_crit and random.random() > ACTION_TRIGGER_CHANCE:
            return
        defender = "p2" if side == "p1" else "p1"
        if is_crit:
            context = (
                f"CRITICAL HIT! {self._name(side)} landed a huge {action.upper()} on "
                f"{self._name(defender)}, stunning them. {self._name(side)} health: "
                f"{attacker_hp}/100, {self._name(defender)} health: {defender_hp}/100."
            )
        else:
            context = (
                f"{self._name(side)} just landed a {action.upper()} on {self._name(defender)}. "
                f"{self._name(side)} health: {attacker_hp}/100, {self._name(defender)} health: "
                f"{defender_hp}/100. {self._style_note(side)}"
            )
        self._fire(context)

    def notify_low_health(self, side: str, hp: int):
        """Called once on the edge transition into low health -- see
        play_game.py's _update_low_health -- so this doesn't need its own
        probability gate, only the shared cooldown below."""
        if not self.enabled:
            return
        context = (
            f"{self._name(side)} is down to {hp}/100 health -- dangerously close to a KO. "
            f"{self._style_note(side)}"
        )
        self._fire(context)

    def notify_ko(self, winner_side: str, loser_side: str):
        if not self.enabled:
            return
        context = f"{self._name(winner_side)} just landed the finishing blow and KO'd {self._name(loser_side)}!"
        self._fire(context)

    def _fire(self, context: str):
        now = time.time()
        if now - self._last_fire < MIN_INTERVAL_SECONDS:
            return
        try:
            self._in_queue.put_nowait(context)
        except queue.Full:
            return
        self._last_fire = now

    def poll(self):
        """Call once a frame from the main thread. Returns a Path to a
        freshly synthesized line ready to play, or None."""
        try:
            text, path = self._out_queue.get_nowait()
        except queue.Empty:
            return None
        print(f"[commentator] {text}")
        return path

    def _worker(self):
        while True:
            context = self._in_queue.get()
            try:
                reply = self._client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": context}],
                    max_tokens=20,  # lowered from 60 -- matches the new under-8-words prompt
                    temperature=0.9,
                )
                line = (reply.choices[0].message.content or "").strip()
                if not line:
                    continue

                fd, tmp_name = tempfile.mkstemp(suffix=".mp3", prefix="commentary_")
                os.close(fd)
                tmp_path = Path(tmp_name)

                speech = self._client.audio.speech.create(
                    model=TTS_MODEL, voice=TTS_VOICE, input=line, response_format="mp3",
                )
                speech.write_to_file(tmp_path)

                self._out_queue.put((line, tmp_path))
            except Exception as exc:
                # network hiccup, rate limit, bad response -- log and move on,
                # the next notify() just tries again later. Never crash the
                # worker thread over one failed commentary line.
                print(f"[commentator] request failed: {exc}")
