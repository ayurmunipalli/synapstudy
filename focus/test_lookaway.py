"""Unit tests for the event logic and scoring (synthetic inputs, allowed in unit tests only).
  .venv/bin/python -m pytest focus/test_lookaway.py -q   (or run directly)
"""
import numpy as np

from focus.lookaway import drift_events, normal_reading_mask, score, wilson

T = np.arange(0.25, 200.0001, 0.25)


def _sess():
    return dict(t0=0, dur=200.0, cues=[dict(kind="lookaway", direction="LEFT", on=60.0, end=64.0),
                                       dict(kind="eyes_closed", direction="", on=120.0, end=126.0)], m=[170.0])


def test_onset_after_N_and_clear_after_8():
    la = (T >= 61.0) & (T < 63.0)
    ev = drift_events(T, la, None, 4)
    assert len(ev) == 1 and abs(ev[0]["onset"] - 61.75) < 1e-9 and ev[0]["kind"] == "lookaway"
    assert abs(ev[0]["end"] - (63.0 + 7 * 0.25)) < 1e-9


def test_short_blip_below_N_is_ignored():
    la = (T >= 80.0) & (T < 80.5)            # 2 updates
    assert drift_events(T, la, None, 4) == []
    assert len(drift_events(T, la, None, 2)) == 1


def test_scoring_hit_delay_fa_and_exclusions():
    s = _sess()
    la = ((T >= 61.0) & (T < 63.0)) | ((T >= 90.0) & (T < 91.0)) | ((T >= 165.0) & (T < 166.0))
    ec = (T >= 127.0) & (T < 128.0)          # within cue end + 2 s
    ev = drift_events(T, la, ec, 2)
    r = score(ev, s)
    assert r["hits"] == 2 and r["kind_correct"] == 2
    assert abs(r["delays"][0] - 1.25) < 1e-9
    assert r["fa"] == 1 and abs(r["fa_onsets"][0] - 90.25) < 1e-9   # 165 s is inside the M-press exclusion
    assert r["m_with_event"] == 1


def test_event_started_before_cue_is_not_a_hit():
    s = _sess()
    la = (T >= 58.0) & (T < 66.0)
    r = score(drift_events(T, la, None, 2), s)
    assert r["cues"][0]["hit"] is False


def test_normal_reading_mask():
    s = _sess()
    m = normal_reading_mask(np.array([10, 59.5, 68.9, 69.1, 161, 171.5, 172.5]), s)
    assert list(m) == [False, False, False, True, False, False, True]


def test_wilson():
    lo, hi = wilson(9, 10)
    assert 0.59 < lo < 0.6 and 0.98 < hi < 0.99


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
    print("all tests passed")
