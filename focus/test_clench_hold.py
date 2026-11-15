"""Unit tests for focus/clench_hold.py. Synthetic signals only (allowed in unit tests; never results)."""
import numpy as np
from scipy.signal import butter, sosfiltfilt

from focus import clench_hold as H

FS = 256.0


def _sig(seconds, bursts, amp=60.0, seed=42):
    """725-uV offset + 5-uV noise; bursts = [(t_on, t_off)] of 20-45 Hz 'muscle' noise of ~amp uV RMS."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    ts = np.arange(n) / FS
    x = 725 + rng.normal(0, 5, (n, 4))
    sos = butter(4, [22, 43], btype="band", fs=FS, output="sos")
    emg = sosfiltfilt(sos, rng.normal(0, 1, (n, 4)), axis=0)
    emg *= amp / emg.std()
    for a, b in bursts:
        m = (ts >= a) & (ts < b)
        x[m] += emg[m]
    return x, ts


def test_held_clench_fires_once_at_3s():
    x, ts = _sig(30, [(10.0, 14.0)])
    r = H.run_offline(x, ts)
    assert len(r["triggers"]) == 1
    assert 12.9 <= r["triggers"][0] <= 13.4          # ~3 s after onset (+ up to one update / 0.15-s RMS window)
    assert r["progress"].max() == 1.0


def test_long_hold_not_lost_to_baseline_drift():
    """A 10-s hold must still be above threshold at the end (baseline frozen)."""
    x, ts = _sig(40, [(10.0, 20.0)], amp=40.0)
    r = H.run_offline(x, ts)
    i = np.searchsorted(r["t"], 19.5)
    assert r["above"][i] and len(r["triggers"]) == 1


def test_short_clench_does_not_fire_and_is_cancelled():
    x, ts = _sig(30, [(10.0, 12.0)])
    r = H.run_offline(x, ts)
    assert r["triggers"] == [] and len(r["cancels"]) == 1


def test_chewing_rhythm_does_not_fire():
    bursts = [(10.0 + k * 0.7, 10.0 + k * 0.7 + 0.3) for k in range(20)]   # 0.3 s on / 0.4 s off for 14 s
    x, ts = _sig(40, bursts)
    assert H.run_offline(x, ts)["triggers"] == []


def test_brief_dip_inside_hold_is_tolerated_long_gap_resets():
    x, ts = _sig(40, [(10.0, 11.5), (11.65, 14.0)])          # 0.15-s dip
    assert len(H.run_offline(x, ts)["triggers"]) == 1
    x, ts = _sig(40, [(10.0, 11.5), (12.2, 14.0)])           # 0.7-s gap: two short clenches
    assert H.run_offline(x, ts)["triggers"] == []


def test_refractory_and_quiet_signal():
    x, ts = _sig(40, [(10.0, 14.0), (14.5, 17.6)])           # a second 3-s hold completes < 5 s after the first trigger
    assert len(H.run_offline(x, ts)["triggers"]) == 1
    x, ts = _sig(60, [])
    assert H.run_offline(x, ts)["triggers"] == []


def test_live_wrapper_matches_offline_exactly():
    """LiveClench fed in irregular chunks gives the same triggers as run_offline at the same fs."""
    x, ts = _sig(40, [(10.0, 14.0), (20.0, 21.5), (25.0, 30.0)])
    ts = ts + 1000.0                                  # LSL-like absolute times
    params = dict(floor=15.0, channels="ears", gap_s=0.5)
    off = H.run_offline(x, ts, **params)
    fs = off["fs"]
    live = H.LiveClench(params)
    rng = np.random.default_rng(1)
    i, trig, prog = 0, [], []
    while i < len(ts):
        n = int(rng.integers(5, 40))
        live.push(x[i:i + n], ts[i:i + n])
        i += n
        for ev in live.advance(ts[min(i, len(ts)) - 1], fs):
            (trig if ev["type"] == "clench" else prog).append(ev)
    assert [round(e["t"] - 1000.0, 6) for e in trig] == [round(t, 6) for t in off["triggers"]]
    assert all(abs(e["onset_t"] - (e["t"] - 3.0)) < 1e-9 for e in trig)
    assert any(p["cancelled"] for p in prog)            # the 1.5-s clench is reported as cancelled
    assert prog[-1]["progress"] == 0.0                  # progress always returns to 0 after a hold


def test_live_wrapper_waits_for_fs_and_frozen_file():
    live = H.LiveClench(dict(floor=15.0, channels="ears", gap_s=0.5))
    x, ts = _sig(5, [])
    live.push(x, ts)
    assert live.advance(ts[-1], None) == []
    p = H.load_frozen_v2()
    assert p is None or (p["channels"] in ("all4", "ears") and p["floor"] > 0)
    assert H.load_frozen_v2("/nonexistent.json") is None
