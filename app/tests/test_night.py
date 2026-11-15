"""app/night.py with stand-in scripts (synthetic: no headband, nothing is recorded)."""
import sys
import time

from app import config, night

FAKE_CUE = r'''
import signal, sys, time
print("Warning: something harmless", flush=True)
print("Detector: personal model x. Snooze: double-tap, 15 min. Ctrl+C to stop.", flush=True)
print("23:01:00 cue 1: 01_neuron  (0.71)", flush=True)
print("23:01:08 SNOOZE 1: cues held 15 min", flush=True)
print("23:02:00 cue 2: 02_weights  (0.70)", flush=True)
try:
    time.sleep(30)
except KeyboardInterrupt:
    pass
print("2 cues played, 1 snoozes. Now run: python stage_nap.py results/cuenap-20261115-230000.npz", flush=True)
'''
FAKE_NO_STREAM = 'raise SystemExit("No Muse_EEG stream found. In another terminal run: OpenMuse stream")\n'
FAKE_STAGE = 'import sys\nprint("Stages: W 10 (5%), N2 120 (60%)")\nprint("Plot saved: results/cuenap-20261115-230000_stages.png")\n'


def _wait(n, phases, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = n.status()
        if s["phase"] in phases:
            return s
        time.sleep(0.05)
    raise AssertionError(f"phase never reached {phases}: {n.status()['phase']}")


def test_a_night_runs_counts_cues_stops_and_stages(tmp_path, monkeypatch):
    model = tmp_path / "model.npz"
    model.write_bytes(b"x")
    monkeypatch.setattr(config, "SLEEP_MODEL", model)
    monkeypatch.setattr(config, "RESULTS", tmp_path / "results")
    (tmp_path / "sleep_cue.py").write_text(FAKE_CUE)
    (tmp_path / "stage_nap.py").write_text(FAKE_STAGE)
    cues = tmp_path / "cues"
    cues.mkdir()
    (cues / "01_neuron.wav").write_bytes(b"RIFF")
    n = night.Night(root=tmp_path, python=sys.executable)
    s = n.start("a" * 32, cues, 0.9)
    assert s["volume"] == 0.5                       # capped: sleep cues stay quiet
    s = _wait(n, {"running"})
    deadline = time.time() + 5
    while n.status()["cues"] < 2 and time.time() < deadline:
        time.sleep(0.05)
    s = n.status()
    assert s["cues"] == 2 and s["snoozes"] == 1 and "prototype" in s["model_note"].lower()
    try:
        n.start(None, None, 0.1)
        raise AssertionError("a second night must be refused")
    except RuntimeError:
        pass
    s = n.stop()
    s = _wait(n, {"done"})
    assert s["recording"] == "results/cuenap-20261115-230000.npz"
    n.stage()
    deadline = time.time() + 10
    while n.status()["stage"]["status"] == "running" and time.time() < deadline:
        time.sleep(0.05)
    st = n.status()["stage"]
    assert st["status"] == "done" and st["image"] == "cuenap-20261115-230000_stages.png"
    assert any("N2" in line for line in st["summary"])
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / st["image"]).write_bytes(b"png")
    assert n.image_path(st["image"]) is not None
    assert n.image_path("../secret.png") is None and n.image_path("other_stages.png") is None


def test_no_headband_stream_is_reported_not_hidden(tmp_path, monkeypatch):
    model = tmp_path / "model.npz"
    model.write_bytes(b"x")
    monkeypatch.setattr(config, "SLEEP_MODEL", model)
    (tmp_path / "sleep_cue.py").write_text(FAKE_NO_STREAM)
    n = night.Night(root=tmp_path, python=sys.executable)
    n.start(None, None, 0.15)
    s = _wait(n, {"failed"})
    assert "No Muse_EEG stream" in s["error"] and s["recording"] is None
    try:
        n.stage()
        raise AssertionError("no recording: staging must be refused")
    except RuntimeError:
        pass


def test_no_trained_model_refuses_to_start(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SLEEP_MODEL", tmp_path / "missing.npz")
    n = night.Night(root=tmp_path, python=sys.executable)
    try:
        n.start(None, None, 0.15)
        raise AssertionError("must refuse")
    except RuntimeError as e:
        assert "model" in str(e)
