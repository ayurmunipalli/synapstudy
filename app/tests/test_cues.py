"""Tests for app/cues.py. Audio is written only under tmp_path. The speech tests use the real macOS `say`
and skip (rather than fail) where `say` yields no audio, which is what happens inside a sandboxed shell."""
import json
import os
import re
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from app import cues

CARDS = [
    {"id": "a1", "front": "What preserves useful variations?", "back": "Natural selection.",
     "concept": "natural selection", "section_idx": 0, "from_missed": True},
    {"id": "b2", "front": "Who wrote it?", "back": "Darwin.", "concept": "Charles Darwin", "section_idx": 1,
     "from_missed": False},
    {"id": "c3", "front": "What limits numbers?", "back": "Climate.", "concept": "climate", "section_idx": 2,
     "from_missed": False},
]


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        params = (w.getframerate(), 8 * w.getsampwidth(), w.getnchannels())
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").reshape(-1, w.getnchannels())
    return params, data


def dbfs(samples):
    return 20 * np.log10(np.abs(samples.astype(float)).max() / 32768.0)


def no_speech(monkeypatch):
    monkeypatch.setattr(cues, "tts_available", lambda: False)


def test_tts_available_matches_the_tools_on_this_machine():
    expected = all(os.access(f"/usr/bin/{tool}", os.X_OK) for tool in ("say", "afconvert"))
    assert cues.tts_available() is expected
    assert expected, "this project runs on macOS: /usr/bin/say and /usr/bin/afconvert should exist"


@pytest.mark.parametrize("text,slug", [
    ("natural selection", "natural-selection"),
    ("  Natural   Selection!! ", "natural-selection"),
    ("Schrödinger's cat", "schrodingers-cat"),
    ("ATP → ADP + Pi", "atp-adp-pi"),
    ("../../etc/passwd", "etc-passwd"),
    ("C++ / C#", "c-c"),
    ("", "cue"), ("   ", "cue"), ("!!!", "cue"), ("日本語", "cue"),
    ("a" * 80, "a" * 40),
    ("x" * 39 + " yz", "x" * 39),                               # no trailing hyphen after the cut
])
def test_slugify(text, slug):
    result = cues.slugify(text)
    assert result == slug
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", result) and 1 <= len(result) <= 40


def test_tone_fallback_files_have_the_format_sleep_cue_expects(tmp_path, monkeypatch):
    no_speech(monkeypatch)
    result = cues.make_cues(CARDS, tmp_path / "cues")
    assert result == [
        {"file": "01_natural-selection.wav", "text": "natural selection", "kind": "tone"},
        {"file": "02_charles-darwin.wav", "text": "Charles Darwin", "kind": "tone"},
        {"file": "03_climate.wav", "text": "climate", "kind": "tone"},
    ]
    audio = []
    for cue in result:
        params, data = read_wav(tmp_path / "cues" / cue["file"])
        assert params == (44100, 16, 2)
        assert 0.3 < len(data) / 44100 < 1.0
        assert np.abs(data).max() > 1000                         # not silent
        assert dbfs(data) == pytest.approx(-12.0, abs=0.3)       # quieter than speech on purpose
        assert np.array_equal(data[:, 0], data[:, 1])
        assert abs(int(data[0, 0])) < 50 and abs(int(data[-1, 0])) < 50      # fades: no click at either end
        audio.append(data)
    assert not np.array_equal(audio[0], audio[1])
    assert not np.array_equal(audio[1], audio[2])
    assert not np.array_equal(audio[0], audio[2])
    assert sorted(p.name for p in (tmp_path / "cues").iterdir()) == [c["file"] for c in result]


def test_tone_motifs_differ_for_every_index_up_to_56():
    tones = [cues._tone(i).tobytes() for i in range(56)]
    assert len(set(tones)) == 56


def test_tone_motifs_use_two_different_notes():
    for index in (0, 1, 7, 30):
        mono = cues._tone(index)[:, 0].astype(float)
        half = len(mono) // 2

        def dominant(x):
            spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
            return np.fft.rfftfreq(len(x), 1 / 44100)[spectrum.argmax()]

        first, second = dominant(mono[:half]), dominant(mono[half:])
        assert abs(first - second) > 20
        for hz in (first, second):
            assert min(abs(hz - note) for note in cues._NOTES_HZ) < 6


def test_speech_cue_for_a_two_word_concept(tmp_path):
    if not cues.tts_available():
        pytest.skip("say/afconvert are not installed")
    (cue,) = cues.make_cues([{"concept": "natural selection"}], tmp_path)
    if cue["kind"] != "speech":
        pytest.skip("`say` produced no audio here (it does not inside a sandboxed shell); speech not verified")
    assert cue == {"file": "01_natural-selection.wav", "text": "natural selection", "kind": "speech"}
    params, data = read_wav(tmp_path / cue["file"])
    assert params == (44100, 16, 2)
    seconds = len(data) / 44100
    assert 0.3 < seconds < 3.0
    assert dbfs(data) == pytest.approx(-6.0, abs=0.2)            # peak-normalised
    rms = np.sqrt(np.mean(data.astype(float) ** 2))
    assert rms > 300                                             # speech, not a click or silence
    assert abs(int(data[0, 0])) < 50 and abs(int(data[-1, 0])) < 50         # 20 ms fades
    level = np.abs(data).max(axis=1)
    loud = np.flatnonzero(level >= 0.01 * level.max())
    assert loud[0] < 0.03 * 44100 and len(data) - loud[-1] < 0.03 * 44100   # silence trimmed at both ends
    different = cues.make_cues([{"concept": "sexual selection"}, {"concept": "climate"}], tmp_path / "b")
    assert [c["kind"] for c in different] == ["speech", "speech"]
    assert read_wav(tmp_path / "b" / different[0]["file"])[1].shape != read_wav(tmp_path / "b" / different[1]["file"])[1].shape


def test_text_is_never_passed_through_a_shell(tmp_path, monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0)               # writes nothing: the cue then falls back to a tone

    monkeypatch.setattr(cues.subprocess, "run", fake_run)
    monkeypatch.setattr(cues, "tts_available", lambda: True)
    nasty = ['x"; touch /tmp/pwned; echo "', "-v Fred --rate 900", "$(reboot) `id` && ls | cat",
             "pause [[slnc 60000]] here", "line one\nline two\x07"]
    result = cues.make_cues([{"concept": text} for text in nasty], tmp_path)

    say_calls = [(a, k) for a, k in calls if Path(a[0]).name == "say"]
    assert len(say_calls) == len(nasty)
    for (args, kwargs), text in zip(say_calls, nasty):
        assert isinstance(args, list) and all(isinstance(a, str) for a in args)
        assert not kwargs.get("shell")
        assert args[1] == "-o" and args[3] == "--" and len(args) == 5       # the text is one argument, after --
        assert kwargs.get("timeout")
    spoken = [args[4] for args, _ in say_calls]
    assert spoken[0] == nasty[0] and spoken[1] == nasty[1] and spoken[2] == nasty[2]   # passed literally
    assert "[[" not in spoken[3] and "]]" not in spoken[3] and "slnc 60000" in spoken[3]
    assert spoken[4] == "line one line two"
    assert [c["kind"] for c in result] == ["tone"] * len(nasty)
    for cue in result:
        assert re.fullmatch(r"\d{2}_[a-z0-9-]{1,40}\.wav", cue["file"])
        assert (tmp_path / cue["file"]).is_file()
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(c["file"] for c in result)   # nothing escaped the folder


@pytest.mark.parametrize("failure", [
    subprocess.CalledProcessError(1, "say"), subprocess.TimeoutExpired("say", 30), FileNotFoundError("say"),
], ids=["nonzero-exit", "timeout", "missing-binary"])
def test_a_failed_speech_command_falls_back_to_a_tone(failure, tmp_path, monkeypatch):
    def fake_run(args, **kwargs):
        raise failure

    monkeypatch.setattr(cues.subprocess, "run", fake_run)
    monkeypatch.setattr(cues, "tts_available", lambda: True)
    result = cues.make_cues(CARDS[:2], tmp_path)
    assert [c["kind"] for c in result] == ["tone", "tone"]
    for cue in result:
        params, data = read_wav(tmp_path / cue["file"])
        assert params == (44100, 16, 2) and np.abs(data).max() > 1000


def test_silent_speech_output_counts_as_failed(tmp_path, monkeypatch):
    """`say` can exit 0 and write an empty file; that must not become a silent cue."""
    def fake_run(args, **kwargs):
        if Path(args[0]).name == "afconvert":
            with wave.open(args[-1], "wb") as w:
                w.setnchannels(2)
                w.setsampwidth(2)
                w.setframerate(44100)
                w.writeframes(np.zeros((22050, 2), dtype="<i2").tobytes())
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(cues.subprocess, "run", fake_run)
    monkeypatch.setattr(cues, "tts_available", lambda: True)
    (cue,) = cues.make_cues(CARDS[:1], tmp_path)
    assert cue["kind"] == "tone"
    assert np.abs(read_wav(tmp_path / cue["file"])[1]).max() > 1000


def test_polish_trims_normalises_and_fades():
    rate = 44100
    t = np.arange(rate // 2) / rate
    burst = (0.1 * 32767 * np.sin(2 * np.pi * 220 * t)).astype("<i2")
    silence = np.zeros(rate // 2, dtype="<i2")
    stereo = np.repeat(np.concatenate([silence, burst, silence])[:, None], 2, axis=1)
    out = cues._polish(stereo)
    assert len(out) / rate == pytest.approx(0.5 + 2 * cues.TRIM_PAD_S, abs=0.01)
    assert dbfs(out) == pytest.approx(-6.0, abs=0.1)
    assert abs(int(out[0, 0])) < 50 and abs(int(out[-1, 0])) < 50
    with pytest.raises(ValueError):
        cues._polish(np.zeros((1000, 2), dtype="<i2"))
    with pytest.raises(ValueError):
        cues._polish(np.zeros((0, 2), dtype="<i2"))


def test_cards_without_a_usable_concept_still_get_a_cue(tmp_path, monkeypatch):
    no_speech(monkeypatch)
    result = cues.make_cues([{"concept": ""}, {}, {"concept": None}, "not a card"], tmp_path)
    assert [c["file"] for c in result] == ["01_cue.wav", "02_cue.wav", "03_cue.wav", "04_cue.wav"]
    assert all(c["text"] == "" and c["kind"] == "tone" for c in result)
    assert cues.make_cues([], tmp_path / "empty") == [] and (tmp_path / "empty").is_dir()


def test_old_cue_files_are_replaced_but_other_files_are_left_alone(tmp_path, monkeypatch):
    no_speech(monkeypatch)
    cues.make_cues(CARDS, tmp_path)
    (tmp_path / "notes.txt").write_text("keep me")
    (tmp_path / "My Recording.wav").write_bytes(b"keep me too")
    result = cues.make_cues(CARDS[2:], tmp_path)                 # a new, shorter deck
    assert [c["file"] for c in result] == ["01_climate.wav"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["01_climate.wav", "My Recording.wav", "notes.txt"]


def test_write_manifest_maps_cards_to_files(tmp_path, monkeypatch):
    no_speech(monkeypatch)
    made = cues.make_cues(CARDS, tmp_path)
    session = {"id": "0" * 32, "title": "Origin of Species", "subject": "Biology", "mode": "reading",
               "flashcards": CARDS}
    path = cues.write_manifest(session, made, tmp_path)
    assert path == tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    assert manifest["session_id"] == "0" * 32 and manifest["title"] == "Origin of Species"
    assert manifest["subject"] == "Biology" and manifest["count"] == 3
    assert manifest["audio"] == {"sample_rate_hz": 44100, "bit_depth": 16, "channels": 2}
    assert [(c["index"], c["file"], c["card_id"], c["concept"], c["kind"]) for c in manifest["cues"]] == [
        (1, "01_natural-selection.wav", "a1", "natural selection", "tone"),
        (2, "02_charles-darwin.wav", "b2", "Charles Darwin", "tone"),
        (3, "03_climate.wav", "c3", "climate", "tone")]
    assert manifest["cues"][0]["front"] == CARDS[0]["front"] and manifest["cues"][0]["from_missed"] is True
    assert "validated" not in path.read_text().lower() and "works" not in path.read_text().lower()
    # manifest.json is not a cue: regenerating the cues leaves it in place
    cues.make_cues(CARDS, tmp_path)
    assert path.exists()


def test_write_manifest_without_cards_on_the_session(tmp_path, monkeypatch):
    no_speech(monkeypatch)
    made = cues.make_cues(CARDS, tmp_path)
    manifest = json.loads(cues.write_manifest({"id": "1" * 32, "title": "T"}, made, tmp_path).read_text())
    assert [c["file"] for c in manifest["cues"]] == [c["file"] for c in made]
    assert [c["concept"] for c in manifest["cues"]] == ["natural selection", "Charles Darwin", "climate"]
    assert all(c["card_id"] is None for c in manifest["cues"])


def test_cue_files_load_in_the_pygame_mixer_that_sleep_cue_uses(tmp_path, monkeypatch):
    """sleep_cue.py does pygame.mixer.init(frequency=44100, size=-16, channels=2) then mixer.Sound(file)."""
    pygame = pytest.importorskip("pygame")
    no_speech(monkeypatch)
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")               # no sound device is opened, nothing is played
    made = cues.make_cues(CARDS[:1], tmp_path)
    try:
        pygame.mixer.init(frequency=44100, size=-16, channels=2)
    except pygame.error as exc:
        pytest.skip(f"pygame mixer could not start here: {exc}")
    try:
        sound = pygame.mixer.Sound(str(tmp_path / made[0]["file"]))
        assert 0.3 < sound.get_length() < 1.0
    finally:
        pygame.mixer.quit()
