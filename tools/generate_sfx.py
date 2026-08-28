"""Generates every sound effect the game uses as plain synthesized WAV files
-- sine/noise + envelope math, no external sample library, no download. That
sidesteps any licensing question entirely and means the whole audio set is
reproducible by re-running this script.

Run once (or after tweaking a sound):
    ./takken/Scripts/python.exe tools/generate_sfx.py
Writes into assets/sfx/*.wav. src/game/sfx.py loads those files at runtime.
"""
import math
import random
import struct
import wave
from pathlib import Path

SFX_DIR = Path(__file__).resolve().parent.parent / "assets" / "sfx"
SFX_DIR.mkdir(parents=True, exist_ok=True)
RATE = 44100


def _write(name, samples):
    """samples: list of floats roughly in [-1, 1]. Clips and writes 16-bit
    mono PCM."""
    path = SFX_DIR / f"{name}.wav"
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        frames = bytearray()
        for s in samples:
            s = max(-1.0, min(1.0, s))
            frames += struct.pack("<h", int(s * 32000))
        f.writeframes(bytes(frames))
    print(f"wrote {path} ({len(samples) / RATE:.2f}s)")


def _n(duration):
    return int(RATE * duration)


def _fade_edges(samples, fade_samples=64):
    """Short linear fade in/out so clips don't click at the start/end."""
    n = len(samples)
    fade_samples = min(fade_samples, n // 2)
    for i in range(fade_samples):
        g = i / fade_samples
        samples[i] *= g
        samples[n - 1 - i] *= g
    return samples


def thump(duration, freq, noise_amt, pitch_drop=0.4):
    """Low sine thud with a bit of noise on top -- punch/kick/hit impacts.
    pitch_drop: fraction the frequency sags by over the clip, for a heavier
    "thud" feel than a flat tone."""
    n = _n(duration)
    out = []
    for i in range(n):
        t = i / RATE
        decay = math.exp(-t * (7 / duration))
        f = freq * (1 - pitch_drop * (i / n))
        tone = math.sin(2 * math.pi * f * t)
        noise = random.uniform(-1, 1) * noise_amt * math.exp(-t * (20 / duration))
        out.append((tone * (1 - noise_amt) + noise) * decay)
    return _fade_edges(out)


def sweep(duration, f_start, f_end, noise_amt=0.0):
    n = _n(duration)
    out = []
    for i in range(n):
        t = i / RATE
        frac = i / n
        f = f_start + (f_end - f_start) * frac
        decay = math.exp(-t * (3 / duration))
        tone = math.sin(2 * math.pi * f * t)
        noise = random.uniform(-1, 1) * noise_amt
        out.append((tone * (1 - noise_amt) + noise) * decay)
    return _fade_edges(out)


def bell(duration, base_freq, partials=(1.0, 2.01, 3.03)):
    n = _n(duration)
    out = []
    for i in range(n):
        t = i / RATE
        decay = math.exp(-t * (2.2 / duration))
        s = sum(math.sin(2 * math.pi * base_freq * p * t) / (k + 1)
                for k, p in enumerate(partials))
        out.append(s * decay * 0.6)
    return _fade_edges(out)


def click(duration, freq, noise_amt=0.6):
    n = _n(duration)
    out = []
    for i in range(n):
        t = i / RATE
        decay = math.exp(-t * (30 / duration))
        tone = math.sin(2 * math.pi * freq * t)
        noise = random.uniform(-1, 1) * noise_amt
        out.append((tone * (1 - noise_amt) + noise) * decay)
    return _fade_edges(out)


def arpeggio(notes, note_duration):
    out = []
    for freq in notes:
        n = _n(note_duration)
        for i in range(n):
            t = i / RATE
            decay = math.exp(-t * (4 / note_duration))
            out.append(math.sin(2 * math.pi * freq * t) * decay * 0.8)
    return _fade_edges(out)


def heartbeat_loop():
    """Two thumps (lub-dub) with silence after, sized to loop seamlessly."""
    beat1 = thump(0.09, 55, noise_amt=0.15, pitch_drop=0.5)
    gap1 = [0.0] * _n(0.10)
    beat2 = thump(0.11, 45, noise_amt=0.15, pitch_drop=0.5)
    gap2 = [0.0] * _n(0.55)
    return beat1 + gap1 + beat2 + gap2


def main():
    _write("punch", thump(0.18, 95, noise_amt=0.35))
    _write("kick", thump(0.26, 65, noise_amt=0.45, pitch_drop=0.5))
    _write("shoot", sweep(0.22, 900, 180, noise_amt=0.25))
    _write("hit", click(0.10, 180, noise_amt=0.7))
    _write("ko", sweep(0.9, 500, 60, noise_amt=0.1))
    _write("round_start", bell(0.8, 300))
    _write("fight", sweep(0.28, 250, 900, noise_amt=0.3))
    _write("match_win", arpeggio([392.0, 523.25, 659.25], 0.16))
    _write("heartbeat", heartbeat_loop())
    _write("click", click(0.05, 700, noise_amt=0.2))


if __name__ == "__main__":
    main()
