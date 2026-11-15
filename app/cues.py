"""Audio cues for flashcard concepts: one short WAV per card, in the format sleep_cue.py's mixer expects
(44.1 kHz, 16-bit, stereo).

The concept is spoken with the macOS `say` voice. When speech is unavailable or fails for a card, the card gets
a two-note tone motif instead, different for each card index, and the cue is labelled "tone". Concept text is
only ever passed to `say` as a single argument after `--`; nothing goes through a shell.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

SAMPLE_RATE = 44100
CHANNELS = 2
SAMPLE_WIDTH = 2                       # bytes: 16-bit
SPEECH_PEAK_DBFS = -6.0
TONE_PEAK_DBFS = -12.0                 # a pure tone sounds much louder than speech at the same peak
TRIM_BELOW_PEAK_DB = -40.0             # leading/trailing audio quieter than this (relative to the peak) is cut
TRIM_PAD_S = 0.015
FADE_S = 0.020
MIN_SPEECH_S = 0.08                    # shorter than this after trimming means `say` produced nothing useful
MAX_SPOKEN_CHARS = 80
SLUG_MAX = 40
_TOOL_TIMEOUT_S = 30

_CUE_FILE = re.compile(r"^\d{2,}_[a-z0-9-]{1,%d}\.wav$" % SLUG_MAX)
# C major pentatonic from C4: any two of these sound consonant together.
_NOTES_HZ = (261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 587.33, 659.25)
_MOTIFS = tuple((a, b) for a in range(len(_NOTES_HZ)) for b in range(len(_NOTES_HZ)) if a != b)


def _tool(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    fallback = f"/usr/bin/{name}"
    return fallback if os.access(fallback, os.X_OK) else None


def tts_available() -> bool:
    """True when macOS `say` and `afconvert` are both present."""
    return _tool("say") is not None and _tool("afconvert") is not None


def slugify(text: str) -> str:
    """Lowercase ASCII letters, digits and hyphens; at most 40 characters; never empty."""
    ascii_text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower().replace("'", "")).strip("-")
    return slug[:SLUG_MAX].strip("-") or "cue"


def _spoken_text(text: str) -> str:
    """The text handed to `say`: one line, no control characters, none of say's embedded [[commands]]."""
    text = "".join(c if c.isprintable() else " " for c in str(text))
    text = " ".join(text.replace("[[", " ").replace("]]", " ").split())
    if len(text) > MAX_SPOKEN_CHARS:
        text = text[:MAX_SPOKEN_CHARS].rsplit(" ", 1)[0]
    return text if any(c.isalnum() for c in text) else ""


def _read_wav(path: Path) -> np.ndarray:
    """int16 samples, shape (n, 2). Raises ValueError if the file is not 44.1 kHz 16-bit stereo."""
    with wave.open(str(path), "rb") as w:
        if (w.getframerate(), w.getsampwidth(), w.getnchannels()) != (SAMPLE_RATE, SAMPLE_WIDTH, CHANNELS):
            raise ValueError("unexpected WAV format")
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype="<i2").reshape(-1, CHANNELS)


def _write_wav(path: Path, samples: np.ndarray) -> None:
    tmp = path.with_suffix(".wav.tmp")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(np.ascontiguousarray(samples, dtype="<i2").tobytes())
    os.replace(tmp, path)


def _fade(x: np.ndarray) -> np.ndarray:
    n = min(int(FADE_S * SAMPLE_RATE), len(x) // 2)
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x


def _to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x * 32767.0), -32768, 32767).astype("<i2")


def _polish(samples: np.ndarray) -> np.ndarray:
    """Trim leading/trailing silence, peak-normalise to -6 dBFS and add 20 ms fades.
    Raises ValueError when there is no audio to keep."""
    x = samples.astype(np.float64) / 32768.0
    level = np.abs(x).max(axis=1) if len(x) else np.zeros(0)
    peak = float(level.max()) if len(level) else 0.0
    if peak < 1e-3:
        raise ValueError("silent audio")
    loud = np.flatnonzero(level >= peak * 10 ** (TRIM_BELOW_PEAK_DB / 20))
    pad = int(TRIM_PAD_S * SAMPLE_RATE)
    x = x[max(0, loud[0] - pad):loud[-1] + 1 + pad]
    if len(x) < MIN_SPEECH_S * SAMPLE_RATE:
        raise ValueError("audio too short")
    x = x * (10 ** (SPEECH_PEAK_DBFS / 20) / peak)
    return _to_int16(_fade(x))


def _speak(text: str, out_path: Path) -> bool:
    """Write the spoken text to out_path. False (and no file) if speech could not be made."""
    say, afconvert = _tool("say"), _tool("afconvert")
    if not (say and afconvert and text):
        return False
    try:
        with tempfile.TemporaryDirectory(prefix="studycue_") as tmp:
            aiff, raw = Path(tmp) / "cue.aiff", Path(tmp) / "cue.wav"
            subprocess.run([say, "-o", str(aiff), "--", text], check=True, timeout=_TOOL_TIMEOUT_S,
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run([afconvert, "-f", "WAVE", "-d", f"LEI16@{SAMPLE_RATE}", "-c", str(CHANNELS),
                            str(aiff), str(raw)], check=True, timeout=_TOOL_TIMEOUT_S,
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            samples = _read_wav(raw)
        _write_wav(out_path, _polish(samples))
    except (OSError, subprocess.SubprocessError, ValueError, wave.Error, EOFError):
        return False
    return True


def _tone(index: int) -> np.ndarray:
    """A soft two-note motif for card `index` (0-based). The 56 ordered note pairs are all different; the
    stride spreads neighbouring cards across the scale."""
    a, b = _MOTIFS[(index * 23) % len(_MOTIFS)]
    note_s, gap_s = 0.28, 0.06
    t = np.arange(int(note_s * SAMPLE_RATE)) / SAMPLE_RATE
    envelope = np.minimum(t / 0.015, 1.0) * np.exp(-t / 0.12)

    def note(hz):
        return (np.sin(2 * np.pi * hz * t) + 0.2 * np.sin(4 * np.pi * hz * t)) * envelope

    mono = np.concatenate([note(_NOTES_HZ[a]), np.zeros(int(gap_s * SAMPLE_RATE)), note(_NOTES_HZ[b])])
    mono *= 10 ** (TONE_PEAK_DBFS / 20) / np.abs(mono).max()
    return _to_int16(_fade(np.repeat(mono[:, None], CHANNELS, axis=1)))


def _cue_text(card) -> str:
    concept = card.get("concept") if isinstance(card, dict) else None
    return " ".join(concept.split()) if isinstance(concept, str) else ""


def make_cues(cards: list[dict], out_dir: Path) -> list[dict]:
    """One WAV per card, in card order, named "<NN>_<slug>.wav". Returns [{"file", "text", "kind"}] where kind
    is "speech" (the concept spoken by `say`) or "tone" (the fallback motif).

    Cue files left in out_dir by an earlier call are removed first, because sleep_cue.py plays every .wav in
    the folder. Other files there are left alone.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.iterdir():
        if old.is_file() and _CUE_FILE.match(old.name):
            old.unlink()

    speech = tts_available()
    cues = []
    for n, card in enumerate(cards or [], 1):
        text = _cue_text(card)
        name = f"{n:02d}_{slugify(text)}.wav"
        path = out_dir / name
        if speech and _speak(_spoken_text(text), path):
            kind = "speech"
        else:
            _write_wav(path, _tone(n - 1))
            kind = "tone"
        cues.append({"file": name, "text": text, "kind": kind})
    return cues


def write_manifest(session: dict, cues: list[dict], out_dir: Path) -> Path:
    """Write out_dir/manifest.json: which session the cues belong to and which card each file is for.
    Cards are matched to cues by position (make_cues keeps card order)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = session if isinstance(session, dict) else {}
    cards = session.get("flashcards") if isinstance(session.get("flashcards"), list) else []
    if len(cards) != len(cues):
        cards = []                                     # can't pair them reliably: list the files on their own

    entries = []
    for i, cue in enumerate(cues):
        card = cards[i] if cards and isinstance(cards[i], dict) else {}
        entries.append({
            "index": i + 1, "file": cue.get("file"), "text": cue.get("text"), "kind": cue.get("kind"),
            "card_id": card.get("id"), "concept": card.get("concept", cue.get("text")),
            "front": card.get("front"), "back": card.get("back"),
            "section_idx": card.get("section_idx"), "from_missed": card.get("from_missed"),
        })
    manifest = {
        "session_id": session.get("id"), "title": session.get("title"), "subject": session.get("subject"),
        "mode": session.get("mode"), "created_at": datetime.now().isoformat(timespec="seconds"),
        "audio": {"sample_rate_hz": SAMPLE_RATE, "bit_depth": 8 * SAMPLE_WIDTH, "channels": CHANNELS},
        "count": len(entries), "cues": entries,
    }
    path = out_dir / "manifest.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
    return path
