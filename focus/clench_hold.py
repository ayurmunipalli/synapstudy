"""Hold-to-confirm jaw clench: "clench for 3 s" = "I just zoned out" (study-focus app, video and reading).

Built on the frozen jaw-clench settings (results/facecalib_frozen_v2.json clench_params: all 4 channels, x 3,
floor 30 uV; the muse-live logic in focus/face.py). Two changes, both needed for a held clench, fixed with the
plan in RESULTS.md ("Clench-hold trigger: plan") before any recording:
  1. The baseline is FROZEN while a clench could be in the window: it updates only if no update in the last
     WIN (3 s) was above threshold. Otherwise a held clench is absorbed into the EMA (0.95/0.05 per 0.1 s, so
     ~80% after 3 s), the threshold climbs to the clench level, and the hold is lost before 3 s.
  2. A hold = above threshold from its first update, with no gap longer than GAP_S (0.3 s) between above-threshold
     updates, for HOLD_S (3.0 s). It fires once, at the update where it reaches 3 s. After that the jaw must be
     below threshold for > GAP_S before a new hold can start, and no new trigger within REFRACTORY_S.

Per update the detector reports `progress` (0-1) for an on-screen ring, and `cancelled` when a hold that reached
at least 1 s ends before 3 s.
"""
import json
import os

import numpy as np
from scipy.signal import butter, sosfiltfilt

STEP = 0.1
WIN = 3.0
HOLD_S = 3.0
GAP_S = 0.3
REFRACTORY_S = 5.0
CLENCH_MULT = 3.0        # frozen v1/v2 clench_params
CLENCH_FLOOR = 30.0      # uV


class HoldDetector:
    def __init__(self, fs, mult=CLENCH_MULT, floor=CLENCH_FLOOR, hold_s=HOLD_S, gap_s=GAP_S, channels="all4"):
        self.fs = fs
        self.mult, self.floor, self.hold_s, self.gap_s = mult, floor, hold_s, gap_s
        self.cols = {"all4": [0, 1, 2, 3], "ears": [0, 3]}[channels]   # v2 grid option; default = v1
        self.sos = butter(4, [20, 45], btype="band", fs=fs, output="sos")
        self.base = None
        self.last_above = -np.inf      # time of the last above-threshold update
        self.start = None              # start time of the current hold candidate
        self.fired = False             # current candidate already fired
        self.last_trigger = -np.inf

    def rms(self, x):
        """(window median of per-sample RMS across channels, RMS of the last 0.15 s) on 20-45 Hz, chosen channels."""
        x = np.asarray(x, float)[:, self.cols]
        f = sosfiltfilt(self.sos, x - x.mean(axis=0), axis=0)
        per = np.sqrt((f ** 2).mean(axis=1))
        return float(np.median(per)), float(np.sqrt((f[-int(0.15 * self.fs):] ** 2).mean()))

    def update(self, x, t):
        """x: the last WIN s of EEG (samples x >=4, uV); t: time of the newest sample (s).
        Returns dict(above, thr, now, progress, trigger, cancelled)."""
        out = dict(above=False, thr=None, now=None, progress=0.0, trigger=False, cancelled=False)
        if len(x) < int(WIN * self.fs * 0.95):
            return out
        base_inst, now = self.rms(x)
        if self.base is None:
            self.base = base_inst
        thr = max(self.mult * self.base, self.floor)
        above = now > thr
        if above:
            if self.start is None or t - self.last_above > self.gap_s:
                # a new candidate only after a > GAP_S quiet spell (the previous candidate has ended)
                self.start, self.fired = t, False
            self.last_above = t
        elif self.start is not None and t - self.last_above > self.gap_s:
            if not self.fired and self.last_above - self.start >= 1.0:
                out["cancelled"] = True
            self.start, self.fired = None, False
        # freeze: update the baseline only when nothing in the last WIN s was above threshold
        if t - self.last_above > WIN:
            self.base = 0.95 * self.base + 0.05 * base_inst
        if self.start is not None and not self.fired:
            held = t - self.start
            out["progress"] = float(min(1.0, held / self.hold_s))
            if held >= self.hold_s - 1e-9 and t - self.last_trigger >= REFRACTORY_S:
                self.fired = True
                self.last_trigger = t
                out["trigger"] = True
        out.update(above=bool(above), thr=float(thr), now=now)
        return out


FROZEN_V2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "results", "clenchhold_frozen_v2.json")


def load_frozen_v2(path=FROZEN_V2):
    """The held-out-validated v2 settings ({floor, channels, gap_s}), or None if the file is missing/unreadable."""
    try:
        with open(path) as f:
            p = json.load(f)["params"]
        return dict(floor=float(p["floor"]), channels=str(p["channels"]), gap_s=float(p["gap_s"]))
    except (OSError, KeyError, ValueError, TypeError):
        return None


class LiveClench:
    """Streaming wrapper for the live app (focus/live.py): feed raw EEG chunks, call advance() as data arrives.
    Updates run on the same grid as run_offline (t0 + 3 s + k * 0.1 s, the last round(3 fs) samples with ts <= tick),
    so with the same fs it reproduces run_offline's triggers exactly (tested).

    advance() returns events (LSL times):
      {"type": "clench_progress", "t", "progress" 0-1, "cancelled": bool}  while a hold is building, on the update it
                                                                          drops back to 0, and on a cancel
      {"type": "clench", "t", "onset_t": t - hold_s}                     when a 3-s hold completes
    """

    def __init__(self, params, keep_s=4.0):
        self.params = dict(params)
        self.keep_s = keep_s
        self.det = None
        self.xs, self.ts = [], []
        self.t0 = None
        self.next_tick = None
        self.k = 0
        self.prev_progress = 0.0

    def push(self, x, ts):
        x = np.asarray(x, float)[:, :4]
        ts = np.asarray(ts, float)
        if not len(ts):
            return
        if self.t0 is None:
            self.t0 = float(ts[0])
            self.next_tick = self.t0 + WIN
        self.xs.append(x)
        self.ts.append(ts)
        # trim whole chunks older than keep_s before the newest sample
        while len(self.ts) > 1 and self.ts[1][0] < ts[-1] - self.keep_s:
            self.xs.pop(0)
            self.ts.pop(0)

    def advance(self, upto, fs):
        """Run every update tick <= upto. fs: the stream's estimated rate (None until known -> nothing yet)."""
        if fs is None or self.t0 is None:
            return []
        if self.det is None:
            self.det = HoldDetector(fs, **self.params)
        out = []
        if self.next_tick > upto:
            return out
        x = np.vstack(self.xs)
        t = np.concatenate(self.ts)
        n_win = int(round(WIN * fs))
        while self.next_tick <= upto + 1e-9:
            tick = self.next_tick
            self.k += 1
            self.next_tick = self.t0 + WIN + self.k * STEP
            j = int(np.searchsorted(t - self.t0, tick - self.t0, side="right"))
            if j < n_win:
                continue
            r = self.det.update(x[j - n_win:j], tick - self.t0)
            p = r["progress"]
            if p > 0 or self.prev_progress > 0 or r["cancelled"]:
                out.append(dict(type="clench_progress", t=float(tick), progress=float(p), cancelled=bool(r["cancelled"])))
            self.prev_progress = p
            if r["trigger"]:
                out.append(dict(type="clench", t=float(tick), onset_t=float(tick - self.det.hold_s)))
        return out


def run_offline(eeg, ts, t0=None, **kw):
    """Step a recording through HoldDetector exactly as live (every STEP s, last WIN s).
    Returns dict(t=update times rel. t0, progress, above, triggers=[times], cancels=[times])."""
    eeg = np.asarray(eeg, float)[:, :4]
    ts = np.asarray(ts, float)
    t0 = ts[0] if t0 is None else t0
    rel = ts - t0
    fs = (len(ts) - 1) / (ts[-1] - ts[0])
    det = HoldDetector(fs, **kw)
    n_win = int(round(WIN * fs))
    upd = np.arange(max(WIN, rel[0] + WIN), rel[-1] + 1e-9, STEP)
    prog, above, trig, canc = np.zeros(len(upd)), np.zeros(len(upd), bool), [], []
    for i, t in enumerate(upd):
        j = np.searchsorted(rel, t, side="right")
        if j < n_win:
            continue
        r = det.update(eeg[j - n_win:j], float(t))
        prog[i], above[i] = r["progress"], r["above"]
        if r["trigger"]:
            trig.append(float(t))
        if r["cancelled"]:
            canc.append(float(t))
    return dict(t=upd, progress=prog, above=above, triggers=trig, cancels=canc, fs=fs)
