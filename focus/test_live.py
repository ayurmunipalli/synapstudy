"""Tests for the streaming detector (focus/live.py).

  .venv/bin/python -m pytest focus/test_live.py -q        (add -s to see the fs=None report)

The equivalence tests read the two real look-away recordings (read-only; nothing is written to results/).
Everything else uses synthetic signals, which is allowed in unit tests only.
"""
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from focus import live
from focus.live import LiveLookaway, LslRunner, iter_replay, replay_events, signal_quality, to_offline
from focus.lookaway import STEP, VARIANTS, features, parse_session, run_variant

ROOT = Path(__file__).resolve().parent.parent
# The two recorded look-away sessions. They are raw EEG recordings and are not shipped with the code, so the
# tests that replay them are skipped where the files are absent (the synthetic tests still run).
SESSIONS = [p for p in (ROOT / "results" / "lookaway-20261114-234828.npz", ROOT / "results" / "lookaway-20261115-000831.npz")
            if p.exists()]
needs_recordings = pytest.mark.skipif(len(SESSIONS) < 2, reason="the recorded sessions are not in this checkout")
FROZEN = json.loads((ROOT / "results" / "lookaway_frozen_params.json").read_text())["variants"]
KIND = {"lookaway": "look_away", "eyes_closed": "eyes_closed"}
P = dict(theta=10, k=0.75, q=0.5, r=1.5, N=4)      # for the synthetic tests (same values as the frozen set)

_cache = {}


def _offline(path):
    if path not in _cache:
        z = dict(np.load(path))
        sess = parse_session(z)
        _cache[path] = (sess, features(z, sess))
    return _cache[path]


def _assert_same_events(live_events, offline_events, t0, N):
    drifts = [e for e in live_events if e["type"] == "drift"]
    assert len(drifts) == len(offline_events)
    assert [e["type"] for e in live_events[::2]] == ["drift"] * len(drifts)       # drift, resume, drift, ...
    assert all(e["type"] == "resume" for e in live_events[1::2])
    for i, off in enumerate(offline_events):
        d = live_events[2 * i]
        assert abs((d["t"] - t0) - off["onset"]) < 1e-6
        assert d["kind"] == "+".join(KIND[k] for k in off["kind"].split("+"))
        assert abs(d["onset_t"] - (d["t"] - (N - 1) * STEP)) < 1e-9 and d["confidence"] is None
        if off["end"] is None:                     # still open at the session end: no resume
            assert 2 * i + 1 == len(live_events)
        else:
            r = live_events[2 * i + 1]
            assert set(r) == {"type", "t", "net_rot", "peak_angle"} and abs((r["t"] - t0) - off["end"]) < 1e-6


# ---------------------------------------------------------------- required: equivalence on the real recordings
@pytest.mark.parametrize("path", SESSIONS, ids=lambda p: p.stem)
def test_replay_events_equal_offline_combined(path):
    sess, F = _offline(path)
    p = FROZEN["combined"]["params"]
    offline, _ = run_variant(F, sess, "combined", p)
    assert len(offline) > 0                        # both sessions have events; an empty match would prove nothing
    _assert_same_events(replay_events(path), offline, sess["t0"], p["N"])


@pytest.mark.parametrize("path", SESSIONS, ids=lambda p: p.stem)
def test_replay_features_bit_identical_to_offline(path):
    sess, F = _offline(path)
    trace = []
    for _ in iter_replay(path, trace=trace):
        pass
    tr = np.array(trace)
    assert np.array_equal(tr[:, 0], F["t"])
    for i, name in enumerate(["head", "eye", "quiet", "alpha", "alpha_ratio"], start=1):
        assert np.array_equal(tr[:, i], F[name], equal_nan=True), name


@pytest.mark.parametrize("variant", [v for v in VARIANTS if v != "combined"])
@pytest.mark.parametrize("path", SESSIONS, ids=lambda p: p.stem)
def test_replay_events_equal_offline_other_variants(path, variant):
    sess, F = _offline(path)
    p = FROZEN[variant]["params"]
    offline, _ = run_variant(F, sess, variant, p)
    _assert_same_events(replay_events(path, variant=variant), offline, sess["t0"], p["N"])


@needs_recordings
@pytest.mark.parametrize("chunk_s", [0.013, 0.25, 1.7])
def test_chunk_size_does_not_change_events(chunk_s):
    path = SESSIONS[1]
    sess, F = _offline(path)
    p = FROZEN["combined"]["params"]
    offline, _ = run_variant(F, sess, "combined", p)
    events = [e for _, evs in iter_replay(path, chunk_s=chunk_s) for e in evs]
    _assert_same_events(events, offline, sess["t0"], p["N"])


@pytest.mark.parametrize("path", SESSIONS, ids=lambda p: p.stem)
def test_report_estimated_fs(path):
    """Report only (spec: print, don't assert): does estimating fs from the first 5 s change any event?"""
    sess, F = _offline(path)
    ref = to_offline(replay_events(path), sess["t0"])
    est_events = replay_events(path, fs=None)
    est = to_offline(est_events, sess["t0"])
    key = lambda evs: [(round(e["onset"], 6), e["kind"], None if e["end"] is None else round(e["end"], 6)) for e in evs]
    same = key(ref) == key(est)
    z = np.load(path)
    rel = z["eeg_ts"] - z["eeg_ts"][0]
    m = int(np.searchsorted(rel, 5.0, side="right"))
    fs_est = (m - 1) / (rel[m - 1] - rel[0])
    print(f"\n[fs=None report] {path.name}: recording fs {F['fs']:.4f} Hz, first-5-s estimate {fs_est:.4f} Hz; "
          f"events {'IDENTICAL' if same else 'DIFFERENT'} ({len(ref)} with recording fs, {len(est)} estimated)")
    if not same:
        print("   recording fs:", key(ref))
        print("   estimated fs:", key(est))


# ---------------------------------------------------------------- synthetic unit tests
FS, MFS = 256.0, 52.0


def _eeg(t_from, t_to, seed=0, level=700.0):
    ts = np.arange(t_from, t_to, 1 / FS)
    rng = np.random.default_rng(seed)
    return level + rng.normal(0, 5, (len(ts), 4)), ts


def _motion(t_from, t_to, turn=(None, None), rate=30.0):
    ts = np.arange(t_from, t_to, 1 / MFS)
    x = np.zeros((len(ts), 6))
    if turn[0] is not None:
        x[(ts >= turn[0]) & (ts < turn[1]), 4] = rate
    return x, ts


def test_head_turn_gives_drift_then_resume_with_spec_shapes():
    det = LiveLookaway(P, fs=FS)
    det.start(1000.0)
    det.push_eeg(*[a if i == 0 else a + 1000.0 for i, a in enumerate(_eeg(0, 12))])
    mx, mt = _motion(0, 12, turn=(5.0, 7.0))
    det.push_motion(mx, mt + 1000.0)
    ev = det.advance(1012.0)
    # 30 deg/s about the gyro's second axis: the 1-s angle passes 10 deg from the 5.5 s update to the 7.5 s one
    near = lambda v: pytest.approx(v, abs=0.7)     # the turn's edges fall between motion samples
    assert ev == [dict(type="drift", kind="look_away", t=1006.25, onset_t=1005.5, confidence=None,
                       angle=near(30.0), rot=[0.0, near(30.0), 0.0]),
                  # the whole 60 deg turn: from 1.25 s before the onset (4.25 s) to 2 s before the resume (7.5 s)
                  dict(type="resume", t=1009.5, net_rot=[0.0, near(60.0), 0.0], peak_angle=near(30.0))]
    assert ev[0]["angle"] == pytest.approx(float(np.linalg.norm(ev[0]["rot"])), abs=1e-12)
    json.dumps(ev, allow_nan=False)
    assert det.advance(1012.0) == []               # nothing runs twice


def test_short_turn_below_N_is_ignored():
    det = LiveLookaway(P, fs=FS)
    det.start(0.0)
    det.push_eeg(*_eeg(0, 10))
    # 12 deg in 0.3 s: the 1-s angle is over 10 deg at the 5.5, 5.75 and 6.0 s updates only (3 < N = 4)
    det.push_motion(*_motion(0, 10, turn=(5.05, 5.35), rate=40.0))
    trace = det._trace = []
    assert det.advance(10.0) == []
    assert [row[0] for row in trace if row[1] > P["theta"]] == [5.5, 5.75, 6.0]


def test_advance_only_runs_updates_up_to_the_given_time():
    det = LiveLookaway(P, fs=FS)
    det.start(0.0)
    det.push_eeg(*_eeg(0, 10))
    det.push_motion(*_motion(0, 10, turn=(5.0, 7.0)))
    assert det.advance(6.0) == [] and det.latest()["t"] == 6.0
    assert [e["t"] for e in det.advance(6.25)] == [6.25]
    assert [e["type"] for e in det.advance(9.49)] == []
    assert [e["type"] for e in det.advance(9.5)] == ["resume"]


def test_latest_is_json_safe_and_ready_after_full_history():
    det = LiveLookaway(P, fs=FS)
    assert det.latest() == dict(t=None, head=None, eye=None, quiet=None, alpha_ratio=None, ready=False)
    det.start(0.0)
    det.push_eeg(*_eeg(0, 25))
    det.push_motion(*_motion(0, 25))
    det.advance(10.0)
    mid = det.latest()
    assert mid["ready"] is False and mid["eye"] is None and mid["head"] is not None
    det.advance(24.0)
    late = det.latest()
    assert late["ready"] is True and set(late) == {"t", "head", "eye", "quiet", "alpha_ratio", "ready"}
    json.dumps(late, allow_nan=False)


def test_estimated_fs_holds_updates_until_five_seconds_then_catches_up():
    det = LiveLookaway(P)
    det.start(0.0)
    det.push_motion(*_motion(0, 8, turn=(2.0, 4.0)))
    x, ts = _eeg(0, 8)
    det.push_eeg(x[ts < 4.5], ts[ts < 4.5])
    assert det.fs is None and det.advance(4.4) == []
    det.push_eeg(x[ts >= 4.5], ts[ts >= 4.5])
    assert abs(det.fs - FS) < 0.01
    ev = det.advance(7.9)
    assert [(e["type"], e["t"]) for e in ev] == [("drift", 3.25), ("resume", 6.5)]


def test_memory_stays_bounded_with_and_without_advancing():
    det = LiveLookaway(P, fs=FS)
    det.start(0.0)
    for s in range(0, 300, 10):                    # never advanced: the hard cap applies
        det.push_eeg(*_eeg(s, s + 10, seed=s))
        det.push_motion(*_motion(s, s + 10))
    e, m = det._eeg, det._mot
    assert e.t[e.n - 1] - e.t[0] <= 90.0 and m.t[m.n - 1] - m.t[0] <= 90.0
    assert len(e.t) < 4 * 90 * FS

    det = LiveLookaway(P, fs=FS)
    det.start(0.0)
    for s in range(0, 300, 1):                     # advanced as data arrives: about one eye window is kept
        det.push_eeg(*_eeg(s, s + 1, seed=s))
        det.push_motion(*_motion(s, s + 1))
        det.advance(s + 1 - 1 / MFS)
    e, m = det._eeg, det._mot
    assert e.t[e.n - 1] - e.t[0] <= 25.0 and m.t[m.n - 1] - m.t[0] <= 15.0
    assert len(det._alpha_hist) == 240


def test_gap_in_eeg_gives_nan_not_a_crash():
    det = LiveLookaway(P, fs=FS)
    det.start(0.0)
    det.push_eeg(*_eeg(0, 30))
    det.push_motion(*_motion(0, 80))
    det.advance(29.0)
    assert det.latest()["eye"] is not None
    assert det.advance(60.0) == []                 # 30 s with no EEG: the 2-s and 20-s windows empty out
    gap = det.latest()
    assert gap["quiet"] is None and gap["eye"] is None and gap["ready"] is False
    det.push_eeg(*_eeg(70, 80, seed=1))
    det.advance(79.0)
    assert det.latest()["eye"] is not None


def test_no_motion_means_head_never_fires():
    det = LiveLookaway(P, fs=FS)
    det.start(0.0)
    det.push_eeg(*_eeg(0, 10))
    assert det.advance(9.0) == [] and det.latest()["head"] is None


def test_events_without_motion_data_carry_null_rotation_fields():
    """An eye-movement event with no motion stream at all: the rotation fields are there, and null."""
    det = LiveLookaway(dict(k=0.75, q=0.3, N=4), variant="eye_only", fs=FS)
    det.start(0.0)
    x, ts = _eeg(0, 40)
    x[(ts >= 30.0) & (ts < 32.0), 1] += 300.0      # a 2 s step in AF7 - AF8 (synthetic "eye movement")
    det.push_eeg(x, ts)
    ev = det.advance(39.9)
    drift = next(e for e in ev if e["type"] == "drift")
    resume = next(e for e in ev if e["type"] == "resume")
    assert drift["angle"] is None and drift["rot"] is None
    assert resume["net_rot"] is None and resume["peak_angle"] is None
    json.dumps(ev, allow_nan=False)


def test_an_event_too_early_for_its_history_has_no_net_rotation():
    """A turn in the first second or so: there is no gyro integral from 1.25 s before its onset."""
    det = LiveLookaway(dict(theta=10, N=2), variant="motion_only", fs=FS)
    det.start(0.0)
    det.push_eeg(*_eeg(0, 8))
    det.push_motion(*_motion(0, 8, turn=(0.0, 2.0)))
    ev = det.advance(7.9)
    assert [(e["type"], e["t"]) for e in ev] == [("drift", 1.25), ("resume", 4.5)]
    assert ev[0]["angle"] == pytest.approx(30.0, abs=0.7) and ev[1]["net_rot"] is None
    assert ev[1]["peak_angle"] == pytest.approx(30.0, abs=0.7)


def test_peak_angle_is_the_largest_one_second_angle_of_the_event():
    det = LiveLookaway(dict(theta=10, N=2), variant="motion_only", fs=FS)
    det.start(0.0)
    det.push_eeg(*_eeg(0, 12))
    mx, mt = _motion(0, 12, turn=(5.0, 6.0), rate=20.0)       # slow start, then faster
    mx[(mt >= 6.0) & (mt < 7.0), 4] = 50.0
    det.push_motion(mx, mt)
    drift, resume = det.advance(11.9)
    assert drift["angle"] < 20.0 and resume["peak_angle"] == pytest.approx(50.0, abs=1.0)
    assert resume["net_rot"] == [0.0, pytest.approx(70.0, abs=1.0), 0.0]


# ---------------------------------------------------------------- the rotation fields on the real recordings
@pytest.mark.parametrize("theta", [10, 20])
@pytest.mark.parametrize("path", SESSIONS, ids=lambda p: p.stem)
def test_replay_rotation_fields_match_the_recording(path, theta):
    """The events' rotation fields against the gyro integral computed in one piece, as focus/lookaway.py does.
    (theta 20 is not a frozen setting: this checks the arithmetic only, not how well it detects.)"""
    z = dict(np.load(path))
    sess, F = _offline(path)
    p = dict(theta=theta, N=2)
    offline, _ = run_variant(F, sess, "motion_only", p)
    events = replay_events(path, variant="motion_only", params=p)
    _assert_same_events(events, offline, sess["t0"], p["N"])           # same events as offline at this theta too
    mts = np.asarray(z["motion_ts"], float) - sess["t0"]
    gyro = np.asarray(z["motion"], float)[:, 3:6]
    cum = np.vstack([np.zeros(3), np.cumsum(0.5 * (gyro[1:] + gyro[:-1]) * np.diff(mts)[:, None], axis=0)])
    at = lambda t: cum[np.searchsorted(mts, t, side="right") - 1]
    head_at = lambda t: F["head"][int(round(t / STEP)) - 1]
    drifts = [e for e in events if e["type"] == "drift"]
    assert len(drifts) >= 10
    for i, d in enumerate(drifts):
        t, onset = d["t"] - sess["t0"], d["onset_t"] - sess["t0"]
        assert d["angle"] == head_at(t) and d["angle"] > theta
        assert np.allclose(d["rot"], at(t) - at(t - 1.0), rtol=0, atol=1e-9)
        if 2 * i + 1 < len(events):
            r = events[2 * i + 1]
            t_res = r["t"] - sess["t0"]
            assert np.allclose(r["net_rot"], at(t_res - 2.0) - at(onset - 1.25), rtol=0, atol=1e-9)
            during = F["head"][int(round(onset / STEP)) - 1:int(round(t_res / STEP))]
            assert r["peak_angle"] == during.max() >= d["angle"]
    json.dumps(events, allow_nan=False)


# ---------------------------------------------------------------- changing the thresholds while running
def test_set_params_applies_from_the_next_update_and_keeps_history():
    det = LiveLookaway(dict(theta=40, N=2), variant="motion_only", fs=FS)
    det.start(0.0)
    det.push_eeg(*_eeg(0, 30))
    mx, mt = _motion(0, 30, turn=(5.0, 7.0))                   # 30 deg/s for 2 s ...
    mx[(mt >= 15.0) & (mt < 17.0), 4] = 30.0                   # ... twice
    det.push_motion(mx, mt)
    assert det.advance(12.0) == []                             # 30 deg in 1 s is under theta 40
    det.set_params(dict(theta=10, N=2))
    assert det.params == dict(theta=10, N=2)
    ev = det.advance(29.9)                                     # no restart: the second turn is caught at once
    assert [(e["type"], e["t"]) for e in ev] == [("drift", 15.75), ("resume", 19.5)]
    det.set_params(dict(theta=40.0, N=2))
    assert det.params["theta"] == 40.0


@pytest.mark.parametrize("bad", [dict(N=2), dict(theta=None, N=2), dict(theta=float("nan"), N=2),
                                 dict(theta=True, N=2), dict(theta=10), dict(theta=10, N=0), dict(theta=10, N=2.5)])
def test_set_params_refuses_unusable_values_and_keeps_the_old_ones(bad):
    det = LiveLookaway(dict(theta=10, N=2), variant="motion_only", fs=FS)
    with pytest.raises(ValueError):
        det.set_params(bad)
    assert det.params == dict(theta=10, N=2)
    with pytest.raises(ValueError):
        LiveLookaway(bad, variant="motion_only", fs=FS)


def test_push_before_start_and_bad_shapes_are_errors():
    det = LiveLookaway(P, fs=FS)
    with pytest.raises(RuntimeError):
        det.push_eeg(*_eeg(0, 1))
    det.start(0.0)
    with pytest.raises(ValueError):
        det.push_eeg(np.zeros((5, 3)), np.arange(5.0))
    with pytest.raises(ValueError):
        det.push_motion(np.zeros((5, 6)), np.arange(4.0))
    det.push_eeg(np.empty((0, 4)), [])             # empty chunks are fine
    assert det.advance(100.0) == []                # and nothing runs without data... features are just NaN


def test_default_params_are_the_frozen_ones():
    assert LiveLookaway().params == FROZEN["combined"]["params"] == P
    assert LiveLookaway(variant="motion_only").params == FROZEN["motion_only"]["params"]


def test_signal_quality_flags_rail_and_muscle_noise():
    t = np.arange(0, 1, 1 / FS)
    rng = np.random.default_rng(3)
    x = 700 + rng.normal(0, 5, (len(t), 4))
    x[:, 1] = 1450.0                                               # railed
    x[:, 2] += 120 * np.sin(2 * np.pi * 70 * t)                    # 85 uV RMS at 70 Hz
    x[:, 3] += 120 * np.sin(2 * np.pi * 10 * t)                    # big alpha is not noise
    assert signal_quality(x, FS) == [True, False, False, True]
    assert signal_quality(x[:20], FS) == [False] * 4               # too little data to say


# ---------------------------------------------------------------- LslRunner against a fake LSL (logic only)
class _FakeSource:
    """Synthetic EEG + motion handed out 0.25 s per pull. Gyro turns from 8 s on; data stops at 12 s."""

    def __init__(self):
        self.lock = threading.Lock()
        self.available = True
        self.sessions = 0
        self.closed = 0
        self.pulls = {}
        self.stop_at = 12.0
        self.base = 5000.0

    def find_stream(self, name, timeout):
        return name if self.available else None

    def inlet(self, info, max_buflen=0, processing_flags=0):
        src = self
        with self.lock:
            self.sessions += 1
            self.pulls = {}

        class Inlet:
            def pull_chunk(self, timeout=0.0, max_samples=0):
                with src.lock:
                    i = src.pulls.get(info, 0)
                    a, b = i * 0.25, (i + 1) * 0.25
                    if a >= src.stop_at:
                        return [], []
                    src.pulls[info] = i + 1
                if info == "Muse_EEG":
                    x, ts = _eeg(a, b - 1e-9, seed=i)
                else:
                    x, ts = _motion(a, b - 1e-9, turn=(8.0, 1e9))
                return x.tolist(), (ts + src.base).tolist()

            def close_stream(self):
                with src.lock:
                    src.closed += 1
        return Inlet()


def _wait(cond, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_lsl_runner_connects_detects_stalls_and_reconnects(monkeypatch):
    src = _FakeSource()
    monkeypatch.setattr(live, "_lsl_backend", lambda: (src.find_stream, src.inlet, 0, lambda: 99999.0))
    events, statuses = [], []
    calls = {"n": 0}

    def on_event(ev):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("a callback error must not kill the runner")
        events.append(ev)

    runner = LslRunner(on_event, statuses.append, poll_s=0.001)
    runner.stall_s, runner.resolve_every_s, runner.signal_every_s = 0.3, 0.2, 0.02
    runner.start()
    try:
        # connected, a drift from the head turn, then the data stops: resume + disconnected
        assert _wait(lambda: any(s["connected"] for s in statuses))
        assert _wait(lambda: any(e["type"] == "drift" for e in events))
        assert _wait(lambda: any(not s["connected"] and "No EEG data" in s["detail"] for s in statuses))
        drift = next(e for e in events if e["type"] == "drift")
        assert drift["kind"] == "look_away" and 5008.0 < drift["t"] < 5010.0
        assert drift["angle"] > 10 and drift["rot"][1] == pytest.approx(drift["angle"])
        resumes = [e for e in events if e["type"] == "resume"]
        # emitted at the disconnect, at LSL "now": the movement's end was never seen, so no net rotation
        assert resumes == [dict(type="resume", t=99999.0, net_rot=None, peak_angle=pytest.approx(30.0, abs=1.0))]
        sig = [e for e in events if e["type"] == "signal"]
        assert sig and set(sig[-1]) == {"type", "t", "head", "eye", "quiet", "alpha_ratio", "ready", "quality"}
        assert sig[-1]["quality"] == [True] * 4 and isinstance(sig[-1]["t"], float)
        json.dumps(sig, allow_nan=False)
        assert src.closed == 2
        # it re-resolves and connects again with a fresh detector
        assert _wait(lambda: src.sessions >= 4)
        assert _wait(lambda: sum(e["type"] == "drift" for e in events) >= 2)
    finally:
        runner.stop()
    assert not runner.is_alive()
    assert statuses[-1] == dict(connected=False, detail="Detector stopped", ready=False)
    assert all(set(s) == {"connected", "detail", "ready"} for s in statuses)


def test_lsl_runner_reports_missing_stream_and_missing_backend(monkeypatch):
    src = _FakeSource()
    src.available = False
    monkeypatch.setattr(live, "_lsl_backend", lambda: (src.find_stream, src.inlet, 0, lambda: 0.0))
    statuses = []
    runner = LslRunner(lambda e: None, statuses.append, poll_s=0.001)
    runner.resolve_every_s = 0.05
    runner.start()
    assert _wait(lambda: any(s["detail"] == "No headband stream found" for s in statuses))
    runner.stop()
    assert not runner.is_alive() and not any(s["connected"] for s in statuses)

    def broken():
        raise SystemExit("Could not load pylsl")                   # what common._lsl does without pylsl

    monkeypatch.setattr(live, "_lsl_backend", broken)
    statuses = []
    runner = LslRunner(lambda e: None, statuses.append, poll_s=0.001)
    runner.resolve_every_s = 0.05
    runner.start()
    assert _wait(lambda: any("Could not load pylsl" in s["detail"] for s in statuses))
    assert runner.is_alive()                                       # still retrying, not dead
    runner.stop()
    assert not runner.is_alive()


def test_lsl_runner_set_params_reaches_the_running_detector_and_later_ones(monkeypatch):
    src = _FakeSource()
    monkeypatch.setattr(live, "_lsl_backend", lambda: (src.find_stream, src.inlet, 0, lambda: 99999.0))
    built = []

    class Spy(LiveLookaway):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            built.append(self)

    monkeypatch.setattr(live, "LiveLookaway", Spy)
    events = []
    runner = LslRunner(events.append, lambda s: None, variant="motion_only", poll_s=0.001)
    runner.stall_s, runner.resolve_every_s = 0.3, 0.2
    with pytest.raises(ValueError):
        runner.set_params(dict(theta=10))                          # no N: refused, nothing changes
    runner.set_params(dict(theta=45, N=2))                         # before start: the 30 deg/s turn is under it
    runner.start()
    try:
        assert _wait(lambda: len(built) >= 1)
        assert built[0].params == dict(theta=45, N=2)
        assert _wait(lambda: src.closed >= 2)                      # that session ran dry (data stops at 12 s)
        assert not any(e["type"] == "drift" for e in events)
        assert _wait(lambda: len(built) >= 2)
        assert built[1].params == dict(theta=45, N=2)              # a reconnection keeps the setting
        # Changed while running: the detector in use gets it, or (between two sessions) the next one built.
        runner.set_params(dict(theta=15, N=2))
        assert _wait(lambda: built[-1].params == dict(theta=15, N=2))
        assert _wait(lambda: any(e["type"] == "drift" for e in events))
        runner.set_params(None)                                    # back to the frozen parameters
        assert _wait(lambda: built[-1].params == FROZEN["motion_only"]["params"])
    finally:
        runner.stop()
    assert not runner.is_alive()


def test_lsl_runner_set_params_changes_the_detector_in_use():
    runner = LslRunner(lambda e: None, lambda s: None, variant="motion_only")
    det = runner._det = LiveLookaway(dict(theta=45, N=2), variant="motion_only", fs=FS)   # as _session() sets it
    runner.set_params(dict(theta=15, N=2))
    assert det.params == dict(theta=15, N=2)
    with pytest.raises(ValueError):
        runner.set_params(dict(theta="15", N=2))
    assert det.params == dict(theta=15, N=2)
    runner.set_params(None)
    assert det.params == FROZEN["motion_only"]["params"]


# ---------------------------------------------------------------- which headband to bind to (fake stream infos)
class _Info:
    """Stands in for pylsl.StreamInfo: only what the selection reads."""

    def __init__(self, name, host, uid=None):
        self._name, self._host, self._uid = name, host, uid or f"{name}@{host}"

    def name(self):
        return self._name

    def hostname(self):
        return self._host

    def uid(self):
        return self._uid


def test_choose_streams_is_a_preference_never_a_filter():
    mine_e, mine_m = _Info("Muse-EEG (AAAA)", "This-Mac.local"), _Info("Muse-ACCGYRO (AAAA)", "this-mac")
    other_e, other_m = _Info("Muse-EEG (BBBB)", "Other-Mac.local"), _Info("Muse-ACCGYRO (BBBB)", "Other-Mac.local")

    # one of each: nothing to choose, no note
    assert live.choose_streams([mine_e], [mine_m], "This-Mac") == (mine_e, mine_m, "")
    assert live.choose_streams([other_e], [other_m], "This-Mac") == (other_e, other_m, "")     # not filtered out

    # another headband answered first: this machine's streams are preferred (case and ".local" ignored)
    e, m, note = live.choose_streams([other_e, mine_e], [other_m, mine_m], "THIS-MAC.local")
    assert (e, m) == (mine_e, mine_m)
    assert note == ("2 EEG and 2 motion streams found; using Muse-EEG (AAAA) on This-Mac.local "
                    "with Muse-ACCGYRO (AAAA) on this-mac")

    # the hostname settles nothing: an EEG / motion pair from the same device is preferred to a mixed one
    e, m, note = live.choose_streams([other_e, mine_e], [mine_m, other_m], "Some-Other-Name")
    assert (e, m) == (other_e, other_m) and "Muse-EEG (BBBB) on Other-Mac.local" in note
    e, m, _ = live.choose_streams([_Info("Muse-EEG (X)", ""), _Info("Muse-EEG (Y)", "")],
                                  [_Info("Muse-ACCGYRO (Y)", "")], "")       # an empty hostname never matches
    assert (e.name(), m.name()) == ("Muse-EEG (Y)", "Muse-ACCGYRO (Y)")

    # nothing matches at all: the first stream of each type, as before the preference existed
    a, b = _Info("Muse_EEG", "Lab-PC"), _Info("Muse-EEG", "Lab-PC-2")
    c, d = _Info("Muse_ACCGYRO", "Lab-PC"), _Info("Muse-ACCGYRO", "Lab-PC-2")
    e, m, note = live.choose_streams([a, b], [c, d], "This-Mac")
    assert (e, m) == (a, c) and note.startswith("2 EEG and 2 motion streams found; using Muse_EEG on Lab-PC")

    # no motion stream; and things that are not StreamInfo objects (the runner tests' fakes) pass through
    e, m, note = live.choose_streams([other_e, mine_e], [], "this-mac")
    assert (e, m) == (mine_e, None) and note == "2 EEG and 0 motion streams found; using Muse-EEG (AAAA) on This-Mac.local"
    assert live.choose_streams(["Muse_EEG"], ["Muse_ACCGYRO"], "This-Mac") == ("Muse_EEG", "Muse_ACCGYRO", "")


def test_find_streams_listens_again_only_when_no_local_stream_answered(monkeypatch):
    import pylsl
    mine, other = _Info("Muse-EEG (AAAA)", "This-Mac.local"), _Info("Muse-EEG (BBBB)", "Other-Mac.local")
    stranger = _Info("SomeAmp", "This-Mac.local")                  # an EEG stream that is not a Muse
    calls = []
    monkeypatch.setattr(live.socket, "gethostname", lambda: "this-mac")

    def resolver(answers):
        def resolve_byprop(prop, value, minimum=1, timeout=0.0):
            calls.append((prop, value, minimum, timeout))
            return list(answers[len(calls) - 1])
        return resolve_byprop

    # this machine's stream answers: one quick resolve, as before
    monkeypatch.setattr(pylsl, "resolve_byprop", resolver([[stranger, mine]]))
    assert live._find_streams("Muse_EEG", 1.5) == [mine]
    assert calls == [("type", "EEG", 1, 1.5)]

    # only another machine's stream answers at first: listen for the whole timeout once more, keep the order
    calls.clear()
    monkeypatch.setattr(pylsl, "resolve_byprop", resolver([[other], [mine, other, stranger]]))
    assert live._find_streams("Muse_EEG", 1.5) == [other, mine]
    assert [c[2] for c in calls] == [1, 64]

    # nothing at all: no second wait
    calls.clear()
    monkeypatch.setattr(pylsl, "resolve_byprop", resolver([[]]))
    assert live._find_streams("Muse_ACCGYRO", 1.5) == [] and calls == [("type", "ACCGYRO", 1, 1.5)]


class _TwoBands(_FakeSource):
    """Discovery returns another machine's headband first, then this machine's."""

    def __init__(self):
        super().__init__()
        self.opened = []

    def find_stream(self, name, timeout):
        stype = name.split("_", 1)[1]
        return [_Info(f"Muse-{stype} (BBBB)", "Other-Mac.local"), _Info(f"Muse-{stype} (AAAA)", "This-Mac.local")]

    def inlet(self, info, max_buflen=0, processing_flags=0):
        self.opened.append((info.name(), info.hostname()))
        return super().inlet("Muse_ACCGYRO" if "ACCGYRO" in info.name() else "Muse_EEG", max_buflen, processing_flags)


def test_lsl_runner_binds_to_the_preferred_streams_and_says_which(monkeypatch):
    src = _TwoBands()
    monkeypatch.setattr(live, "_lsl_backend", lambda: (src.find_stream, src.inlet, 0, lambda: 0.0))
    monkeypatch.setattr(live.socket, "gethostname", lambda: "this-mac")
    statuses = []
    runner = LslRunner(lambda e: None, statuses.append, poll_s=0.001)
    runner.start()
    try:
        assert _wait(lambda: any(s["connected"] for s in statuses))
    finally:
        runner.stop()
    assert src.opened[:2] == [("Muse-EEG (AAAA)", "This-Mac.local"), ("Muse-ACCGYRO (AAAA)", "This-Mac.local")]
    detail = next(s["detail"] for s in statuses if s["connected"])
    assert detail == ("Headband connected: 2 EEG and 2 motion streams found; using Muse-EEG (AAAA) on "
                      "This-Mac.local with Muse-ACCGYRO (AAAA) on This-Mac.local")
