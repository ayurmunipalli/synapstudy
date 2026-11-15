"""Streaming port of the frozen look-away / eyes-closed detector (APP_SPEC.md section 5).

Source of truth: focus/lookaway.py (features, detectors, drift_events) and results/lookaway_frozen_params.json.
This file changes nothing about the rule; it only computes it sample-by-sample. focus/test_live.py checks that
replaying the two real recordings through LiveLookaway gives the same per-update features and the same events
as the offline code.

  .venv/bin/python -m focus.live --replay results/lookaway-XXXX.npz [--speed 1]   # replay a recording
  .venv/bin/python -m focus.live                                                  # live events from the headband

How equivalence is kept (each of these would otherwise shift a window by one sample or one rounding step):
  - All time arithmetic is done on timestamps minus t0, exactly as the offline code does, and update times are
    k * 0.25 in that same relative frame.
  - The 5 Hz low-pass runs from the first EEG sample pushed, from zero state, with the first sample's AF7 - AF8
    subtracted, and its state is carried across pushes (same arithmetic as one sosfilt over the whole recording).
  - The gyro integral is one running sequential sum (same additions, in the same order, as np.cumsum).
  - Buffers are trimmed only behind what a future update can still reach (20 s of EEG, the last round(2 fs)
    samples, the motion sample just before t - 1 s).
Where the offline code would raise (an EEG gap longer than a feature window leaves that window empty), this
code gives NaN for that feature instead, which the detectors treat as negative.

Added 2026-11-15 (after the first live sessions), without changing which events fire or when:
  - Events also say how the head moved, so the app can tell a turn away from the turn back and so what
    triggered can be looked at later. A drift carries "angle" (the 1-s head angle in degrees at the update
    that declared it) and "rot" ([x, y, z] degrees: the gyro integral over that same 1 s, whose norm is the
    angle). A resume carries "net_rot" ([x, y, z] degrees: the gyro integral from 1.25 s before the event's
    onset to 2 s before the resume, i.e. over the whole movement) and "peak_angle" (the largest 1-s head angle
    during the event). Each is None when there is no motion data for it. These are descriptions, not a
    validated measurement of where the head points (no gyro-bias removal, as in the offline code).
  - set_params() changes the thresholds while running (the app's head-turn sensitivity setting). Only the
    frozen parameters were tested on a held-out session; any other setting is the caller's to label.
"""
import argparse
import hashlib
import json
import logging
import math
import re
import socket
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

from focus import clench_hold
from focus.lookaway import CLEAR_UPDATES, STEP, VARIANTS, parse_session, score

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "results" / "lookaway_frozen_params.json"

EYE_WIN_S = 20.0
QUIET_WIN_S = 2.0
HEAD_WIN_S = 1.0
ALPHA_HIST_UPDATES = int(round(60.0 / STEP))   # updates in (t - 60, t]
FS_ESTIMATE_S = 5.0
MAX_KEEP_S = 90.0          # hard cap on buffered samples, whatever the caller does
TRIM_MIN_SAMPLES = 512     # don't bother compacting for less than this
KIND_NAMES = {"lookaway": "look_away", "eyes_closed": "eyes_closed"}
_FEATURE_OF_PARAM = {"theta": "head", "k": "eye", "q": "quiet", "r": "alpha_ratio"}
_NAN = float("nan")
# The movement behind an event, for "net_rot": from NET_PRE_S before its onset to NET_POST_S before its resume
# (the resume comes CLEAR_UPDATES * STEP = 2 s after the last update that was still positive). Both are whole
# numbers of updates, so the gyro integral is simply remembered at each update for the last HIST_UPDATES.
NET_PRE_S = 1.25
NET_POST_S = 2.0
NET_PRE_UPDATES = int(round(NET_PRE_S / STEP))
NET_POST_UPDATES = int(round(NET_POST_S / STEP))
HIST_UPDATES = 64          # 16 s of per-update history: enough for N up to 59


def load_frozen_params(variant="combined"):
    """The frozen parameters for a variant, from results/lookaway_frozen_params.json."""
    with open(FROZEN) as f:
        return dict(json.load(f)["variants"][variant]["params"])


def _checked_params(params, variant):
    """A copy of params, after checking it has a usable number for every parameter the variant uses."""
    p = dict(params)
    for name in VARIANTS[variant][2]:
        v = p.get(name)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"detector parameter {name!r} must be a number for variant {variant!r}, got {v!r}")
    if p["N"] != int(p["N"]) or p["N"] < 1:
        raise ValueError(f"detector parameter N must be a whole number of updates >= 1, got {p['N']!r}")
    return p


def _vec(v):
    """A 3-vector as a JSON-safe list of floats, or None if there is none or it is not finite."""
    if v is None:
        return None
    out = [float(x) for x in v]
    return out if all(math.isfinite(x) for x in out) else None


class _Buf:
    """Append-only sample buffer with timestamps; drops from the front. Each column block stays a C-contiguous
    array so slices have the same memory layout (and so the same summation order) as the offline arrays."""

    def __init__(self, widths, cap=4096):
        self.t = np.empty(cap)
        self.cols = [np.empty(cap) if w == 0 else np.empty((cap, w)) for w in widths]
        self.n = 0

    def append(self, t, *cols):
        need = self.n + len(t)
        if need > len(self.t):
            cap = max(need, 2 * len(self.t))
            self.t = self._grown(self.t, cap)
            self.cols = [self._grown(c, cap) for c in self.cols]
        self.t[self.n:need] = t
        for dst, src in zip(self.cols, cols):
            dst[self.n:need] = src
        self.n = need

    def _grown(self, a, cap):
        b = np.empty((cap,) + a.shape[1:])
        b[:self.n] = a[:self.n]
        return b

    def drop(self, i):
        if i <= 0:
            return
        keep = self.n - i
        self.t[:keep] = self.t[i:self.n].copy()
        for c in self.cols:
            c[:keep] = c[i:self.n].copy()
        self.n = keep


class LiveLookaway:
    """The frozen look-away / eyes-closed rule, computed incrementally.

    start(t0), then push_eeg / push_motion as samples arrive, then advance(upto) with a time up to which every
    sample has been pushed. Not ready (all detectors negative) until its windows have full history, as offline.
    With fs=None the sampling rate is estimated from the first 5 s of EEG and no update runs until then.
    """

    def __init__(self, params=None, variant="combined", fs=None):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.params = _checked_params(load_frozen_params(variant) if params is None else params, variant)
        self._la_rule, self._ec_rule, used = VARIANTS[variant]
        self._ready_features = [_FEATURE_OF_PARAM[n] for n in used if n in _FEATURE_OF_PARAM]
        self._fs_given = None if fs is None else float(fs)
        self._lock = threading.RLock()
        self._trace = None      # tests set this to a list to collect (t, head, eye, quiet, alpha, alpha_ratio)
        self._t0 = None
        self._reset()

    # ------------------------------------------------------------ state
    def _reset(self):
        self.fs = None
        self._sos = self._zi = self._n2 = self._hann = self._band_a = self._band_t = None
        self._pending = []                     # (rel ts, eeg[:, :4]) held until fs is known
        self._heog_base = None
        self._eeg = _Buf((0, 2))               # filtered hEOG, [TP9, TP10]
        self._eeg_dropped = 0
        self._eeg_first = None                 # rel time of the first EEG sample ever pushed
        self._mot = _Buf((3,), cap=1024)       # cumulative gyro integral (deg)
        self._mot_first = None
        self._mot_last_t = self._mot_last_g = None
        self._mot_last_cum = np.zeros(3)
        self._k = 0                            # updates done; update k is at t0 + k * STEP
        self._alpha_hist = deque(maxlen=ALPHA_HIST_UPDATES)
        self._c_la = self._c_ec = self._clean = 0
        self._active = False
        self._rot = self._cum_now = None       # set by _features: 1-s rotation and the gyro integral at t
        self._hist = deque(maxlen=HIST_UPDATES)    # (gyro integral or None, head) at each recent update
        self._ev_cum0 = None                   # gyro integral NET_PRE_S before the open event's onset
        self._ev_peak = _NAN                   # largest 1-s head angle of the open event so far
        self._last = dict(t=None, head=_NAN, eye=_NAN, quiet=_NAN, alpha_ratio=_NAN)
        if self._fs_given is not None:
            self._set_fs(self._fs_given)

    def _set_fs(self, fs):
        if not (math.isfinite(fs) and fs > 60.0):
            raise ValueError(f"EEG sampling rate {fs!r} Hz is not usable (need > 60 Hz)")
        self.fs = float(fs)
        self._sos = butter(4, 5, btype="low", fs=self.fs, output="sos")
        self._zi = np.zeros((self._sos.shape[0], 2))
        self._n2 = int(round(2 * self.fs))
        self._hann = np.hanning(self._n2)[:, None]
        f = np.fft.rfftfreq(self._n2, 1 / self.fs)
        self._band_a = (f >= 8) & (f <= 12)
        self._band_t = (f >= 2) & (f <= 30)

    def start(self, t0):
        """Reset all state. Update k will be at LSL time t0 + k * 0.25 s."""
        with self._lock:
            self._t0 = float(t0)
            self._reset()

    def set_params(self, params):
        """Replace the thresholds, from the next update on. The features do not depend on them, so nothing
        is recomputed and no history is lost. Safe to call from another thread."""
        p = _checked_params(params, self.variant)
        with self._lock:
            self.params = p

    def _need_start(self):
        if self._t0 is None:
            raise RuntimeError("LiveLookaway.start(t0) must be called before pushing samples")

    # ------------------------------------------------------------ input
    def push_eeg(self, samples, ts):
        """samples (n, >=4) uV in the order TP9, AF7, AF8, TP10; ts LSL seconds."""
        x = np.asarray(samples, float)
        ts = np.asarray(ts, float)
        if len(ts) == 0:
            return
        if x.ndim != 2 or x.shape[1] < 4 or len(x) != len(ts):
            raise ValueError(f"push_eeg: expected samples (n, >=4) and n timestamps, got {x.shape} and {ts.shape}")
        with self._lock:
            self._need_start()
            rel = ts - self._t0
            x = x[:, :4]
            if self._eeg_first is None:
                self._eeg_first = float(rel[0])
                self._heog_base = x[0, 1] - x[0, 2]
            if self.fs is None:
                self._pending.append((rel, x))
                first, last = self._pending[0][0][0], rel[-1]
                if last - first < FS_ESTIMATE_S:
                    return
                allrel = np.concatenate([p[0] for p in self._pending])
                m = int(np.searchsorted(allrel, first + FS_ESTIMATE_S, side="right"))
                self._set_fs((m - 1) / (allrel[m - 1] - allrel[0]))
                chunks, self._pending = self._pending, []
                for r, c in chunks:
                    self._ingest_eeg(r, c)
            else:
                self._ingest_eeg(rel, x)
            self._trim_eeg()

    def _ingest_eeg(self, rel, x):
        heog, self._zi = sosfilt(self._sos, x[:, 1] - x[:, 2] - self._heog_base, zi=self._zi)
        self._eeg.append(rel, heog, x[:, [0, 3]])

    def push_motion(self, samples, ts):
        """samples (n, 6): accelerometer g in [0:3], gyro deg/s in [3:6]; ts LSL seconds."""
        x = np.asarray(samples, float)
        ts = np.asarray(ts, float)
        if len(ts) == 0:
            return
        if x.ndim != 2 or x.shape[1] < 6 or len(x) != len(ts):
            raise ValueError(f"push_motion: expected samples (n, 6) and n timestamps, got {x.shape} and {ts.shape}")
        with self._lock:
            self._need_start()
            rel = ts - self._t0
            gyro = x[:, 3:6]
            if self._mot_last_t is None:
                self._mot_first = float(rel[0])
                cum = np.zeros((1, 3))
                tt, g = rel, gyro
            else:
                cum = np.empty((0, 3))
                tt = np.concatenate([[self._mot_last_t], rel])
                g = np.vstack([self._mot_last_g, gyro])
            if len(tt) > 1:
                inc = 0.5 * (g[1:] + g[:-1]) * np.diff(tt)[:, None]
                # sequential sum continued from the running total: the same additions as one np.cumsum
                cum = np.vstack([cum, np.cumsum(np.vstack([self._mot_last_cum, inc]), axis=0)[1:]])
            self._mot.append(rel, cum)
            self._mot_last_t, self._mot_last_g, self._mot_last_cum = rel[-1], gyro[-1].copy(), cum[-1].copy()
            self._trim_motion()

    # ------------------------------------------------------------ bounded memory
    def _trim_eeg(self):
        b = self._eeg
        if b.n == 0:
            return
        ets = b.t[:b.n]
        hard = int(np.searchsorted(ets, ets[-1] - MAX_KEEP_S, side="right"))
        nxt = (self._k + 1) * STEP
        soft = min(int(np.searchsorted(ets, nxt - EYE_WIN_S, side="right")),
                   int(np.searchsorted(ets, nxt, side="right")) - self._n2)
        i = max(soft, hard)
        if hard > 0 or i >= TRIM_MIN_SAMPLES:
            b.drop(i)
            self._eeg_dropped += max(i, 0)

    def _trim_motion(self):
        b = self._mot
        if b.n == 0:
            return
        mts = b.t[:b.n]
        hard = int(np.searchsorted(mts, mts[-1] - MAX_KEEP_S, side="right"))
        # keep the last sample at or before (next update - 1 s): it is that update's window start
        soft = int(np.searchsorted(mts, (self._k + 1) * STEP - HEAD_WIN_S, side="right")) - 1
        i = max(soft, hard)
        if hard > 0 or i >= TRIM_MIN_SAMPLES:
            b.drop(i)

    # ------------------------------------------------------------ updates
    def advance(self, upto):
        """Run every update whose time is <= upto (LSL seconds), in order, and return the events they raise.
        Call it only with a time up to which all samples have already been pushed."""
        with self._lock:
            if self._t0 is None:
                return []
            return self._advance_rel(float(upto) - self._t0)

    def _advance_rel(self, rel_upto):
        events = []
        if self.fs is None:
            return events
        while (self._k + 1) * STEP <= rel_upto:
            self._k += 1
            events.extend(self._update(self._k))
        self._trim_eeg()
        self._trim_motion()
        return events

    def _features(self, t):
        head = eye = quiet = alpha = _NAN
        self._rot = self._cum_now = None
        # head: gyro integrated over the last 1 s
        m = self._mot
        if self._mot_first is not None and self._mot_first <= t:
            mts, cum = m.t[:m.n], m.cols[0]
            b = int(np.searchsorted(mts, t, side="right")) - 1
            if b >= 0:
                self._cum_now = cum[b].copy()
                if t >= HEAD_WIN_S and self._mot_first <= t - HEAD_WIN_S:
                    a = int(np.searchsorted(mts, t - HEAD_WIN_S, side="right")) - 1
                    if a >= 0:
                        self._rot = cum[b] - cum[a]
                        head = float(np.linalg.norm(self._rot))
        e = self._eeg
        ets, heog, tp = e.t[:e.n], e.cols[0], e.cols[1]
        j = int(np.searchsorted(ets, t, side="right"))             # samples with ts <= t (within the buffer)
        # eye: latest filtered hEOG against the last 20 s
        if self._eeg_first is not None and self._eeg_first <= t - EYE_WIN_S:
            w = heog[int(np.searchsorted(ets, t - EYE_WIN_S, side="right")):j]
            if w.size:
                med = np.median(w)
                p5, p95 = np.percentile(w, [5, 95])
                spread = max(p95 - p5, 1e-9)
                eye = float(abs(heog[j - 1] - med) / spread)
                w2 = heog[int(np.searchsorted(ets, t - QUIET_WIN_S, side="right")):j]
                if w2.size:
                    quiet = float((w2.max() - w2.min()) / spread)
        # alpha: last round(2 fs) samples of TP9 / TP10
        n2 = self._n2
        if j + self._eeg_dropped >= n2 and j >= n2:
            x = tp[j - n2:j] - tp[j - n2:j].mean(axis=0)
            x = x * self._hann
            p = np.abs(np.fft.rfft(x, axis=0)) ** 2
            rel = p[self._band_a].sum(0) / np.maximum(p[self._band_t].sum(0), 1e-12)
            alpha = float(rel.mean())
        return head, eye, quiet, alpha

    def _update(self, k):
        t = k * STEP
        head, eye, quiet, alpha = self._features(t)
        self._alpha_hist.append(alpha)
        alpha_ratio = _NAN
        if not math.isnan(alpha):
            h = np.array([a for a in self._alpha_hist if not math.isnan(a)])
            alpha_ratio = float(alpha / max(np.median(h), 1e-12))
        t_lsl = self._t0 + t
        self._last = dict(t=t_lsl, head=head, eye=eye, quiet=quiet, alpha_ratio=alpha_ratio)
        if self._trace is not None:
            self._trace.append((t, head, eye, quiet, alpha, alpha_ratio))
        self._hist.append((self._cum_now, head))

        # detectors (NaN is negative), then the variant's rules
        p, inf = self.params, math.inf
        nn = lambda v: -inf if math.isnan(v) else v
        d = dict(head=nn(head) > p.get("theta", inf), eye=nn(eye) > p.get("k", inf),
                 quiet=(inf if math.isnan(quiet) else quiet) < p.get("q", -inf),
                 alpha=nn(alpha_ratio) > p.get("r", inf))
        la = bool(self._la_rule(d)) if self._la_rule else False
        ec = bool(self._ec_rule(d)) if self._ec_rule else False

        # drift_events, one step
        N = p["N"]
        self._c_la = self._c_la + 1 if la else 0
        self._c_ec = self._c_ec + 1 if ec else 0
        if not self._active:
            kinds = [name for name, c in (("lookaway", self._c_la), ("eyes_closed", self._c_ec)) if c >= N]
            if kinds:
                self._active, self._clean = True, 0
                run = int(N)                               # the updates of the run that reached N
                self._ev_cum0 = self._cum_back(run - 1 + NET_PRE_UPDATES)
                self._ev_peak = max([h for _, h in list(self._hist)[-run:] if not math.isnan(h)], default=_NAN)
                return [dict(type="drift", kind="+".join(KIND_NAMES[x] for x in kinds), t=t_lsl,
                             onset_t=t_lsl - (N - 1) * STEP, confidence=None,
                             angle=None if math.isnan(head) else head, rot=_vec(self._rot))]
        else:
            if not math.isnan(head) and not head <= self._ev_peak:      # also true while the peak is NaN
                self._ev_peak = head
            self._clean = self._clean + 1 if not (la or ec) else 0
            if self._clean >= CLEAR_UPDATES:
                self._active, self._c_la, self._c_ec = False, 0, 0
                cum0, cum1 = self._ev_cum0, self._cum_back(NET_POST_UPDATES)
                net = None if cum0 is None or cum1 is None else _vec(cum1 - cum0)
                return [dict(type="resume", t=t_lsl, net_rot=net, peak_angle=self._peak())]
        return []

    def _cum_back(self, updates):
        """The gyro integral as it was `updates` updates before the current one, or None if there was no
        motion data then or that is further back than the history kept."""
        return self._hist[-1 - updates][0] if 0 <= updates < len(self._hist) else None

    def _peak(self):
        return None if math.isnan(self._ev_peak) else self._ev_peak

    def latest(self):
        """Feature values at the most recent update. Values not yet available are None (JSON-safe).
        ready = every feature this variant uses had full history at that update."""
        with self._lock:
            out = {k: (None if v is None or (isinstance(v, float) and math.isnan(v)) else v)
                   for k, v in self._last.items()}
            out["ready"] = out["t"] is not None and all(out[f] is not None for f in self._ready_features)
            return out

    def drift_open(self):
        with self._lock:
            return self._active

    def peak_angle(self):
        """The largest 1-s head angle (degrees) of the event that is open now; None if none is open or
        there has been no motion data."""
        with self._lock:
            return self._peak() if self._active else None


# ---------------------------------------------------------------- replay
def iter_replay(npz_path, variant="combined", params=None, fs="recording", chunk_s=0.1, speed=None, trace=None):
    """Feed a saved recording through LiveLookaway in chunk_s chunks, from its first sample, with
    t0 = the session_start mark. Yields (LSL time at the end of the chunk, [events]) for every chunk.
    speed: None = as fast as possible; 1 = real time; 20 = 20x. Updates stop at the session end, as offline."""
    with np.load(npz_path) as zf:
        missing = [k for k in ("eeg", "eeg_ts", "motion", "motion_ts", "ev_t", "ev_label") if k not in zf.files]
        if missing:
            raise ValueError(f"{npz_path} can't be replayed: it has no {', '.join(missing)}")
        z = {k: zf[k] for k in ("eeg", "eeg_ts", "motion", "motion_ts", "ev_t", "ev_label")}
    if not any(str(l).startswith("session_start") for l in z["ev_label"]):
        raise ValueError(f"{npz_path} can't be replayed: it has no session_start mark (not a lookaway_test recording)")
    sess = parse_session(z)
    t0 = float(sess["t0"])
    eeg, ets = np.asarray(z["eeg"], float), np.asarray(z["eeg_ts"], float)
    mot, mts = np.asarray(z["motion"], float), np.asarray(z["motion_ts"], float)
    rel_e, rel_m = ets - t0, mts - t0
    if fs == "recording":
        fs = (len(rel_e) - 1) / (rel_e[-1] - rel_e[0])       # as focus/lookaway.py:features
    det = LiveLookaway(params, variant, fs)
    det._trace = trace
    det.start(t0)
    last_update = len(np.arange(STEP, sess["dur"] + 1e-9, STEP)) * STEP

    start = min(rel_e[0], rel_m[0])
    stop = max(rel_e[-1], rel_m[-1])
    ie = im = 0
    i = 0
    wall0 = time.monotonic()
    while True:
        i += 1
        c_end = start + i * chunk_s
        je = int(np.searchsorted(rel_e, c_end, side="right"))
        jm = int(np.searchsorted(rel_m, c_end, side="right"))
        if je > ie:
            det.push_eeg(eeg[ie:je], ets[ie:je])
        if jm > im:
            det.push_motion(mot[im:jm], mts[im:jm])
        ie, im = je, jm
        if speed:
            lag = (c_end - start) / speed - (time.monotonic() - wall0)
            if lag > 0:
                time.sleep(lag)
        # every sample with relative time <= c_end is in, so updates up to c_end are safe to run
        yield t0 + c_end, det._advance_rel(min(c_end, last_update))
        if c_end >= stop:
            break
    tail = det._advance_rel(last_update)      # session_end mark later than the last sample
    if tail:
        yield t0 + last_update, tail


def replay_events(npz_path, variant="combined", params=None, fs="recording"):
    """All drift / resume events for a saved recording, as the live detector would have raised them.
    fs="recording" uses the same whole-recording rate as focus/lookaway.py:features; fs=None estimates it
    from the first 5 s as the live runner does; a number fixes it."""
    return [ev for _, evs in iter_replay(npz_path, variant, params, fs) for ev in evs]


def to_offline(events, t0):
    """Live events -> the offline list of dict(onset, kind, end), times relative to t0."""
    back = {v: k for k, v in KIND_NAMES.items()}
    out = []
    for ev in events:
        if ev["type"] == "drift":
            out.append(dict(onset=ev["t"] - t0, kind="+".join(back[x] for x in ev["kind"].split("+")), end=None))
        elif ev["type"] == "resume" and out and out[-1]["end"] is None:
            out[-1]["end"] = ev["t"] - t0
    return out


# ---------------------------------------------------------------- live
def signal_quality(eeg, fs):
    """Rough per-channel contact check on about the last 1 s of raw EEG (n, >=4) in uV: True when at most 1% of
    the samples sit at the amplifier rail (<= 1 or >= 1449) and the 40-100 Hz RMS is under 60 uV.
    UNVALIDATED: the thresholds are rules of thumb borrowed from contact_check.py, never tested against
    known good and bad contact. Use it as a hint for the fit dots, not as a measurement."""
    x = np.asarray(eeg, float)
    if x.ndim != 2 or x.shape[1] < 4 or len(x) < 0.5 * fs:
        return [False] * 4
    x = x[:, :4]
    n = len(x)
    rail = ((x <= 1) | (x >= 1449)).mean(axis=0)
    spec = np.fft.rfft(x - x.mean(axis=0), axis=0)
    f = np.fft.rfftfreq(n, 1 / fs)
    band = (f >= 40) & (f <= 100)
    rms = np.sqrt(2 * (np.abs(spec[band]) ** 2).sum(axis=0)) / n     # Parseval, one-sided
    return [bool(rail[i] <= 0.01 and rms[i] < 60.0) for i in range(4)]


# Which headband to bind to (added after review, 2026-11-15). LSL discovery answers for every outlet on the
# network, so at a shared venue the first "Muse-EEG" stream found can be someone else's headband. The choice
# below is a PREFERENCE, never a filter: when nothing matches, the first stream of each type is used, as
# before. UNTESTED against real streams (it needs the headband, and a second one to matter).
def _info_text(info, attr):
    """info.<attr>() as text; "" for anything that is not an LSL StreamInfo (the fakes in the tests)."""
    try:
        return str(getattr(info, attr)() or "")
    except Exception:  # noqa: BLE001
        return ""


def _host_key(host):
    """A hostname for comparing: lower case, without a trailing ".local"."""
    h = str(host or "").strip().lower()
    return h[:-len(".local")] if h.endswith(".local") else h


def _is_local(info, local_host):
    local = _host_key(local_host)
    return bool(local) and _host_key(_info_text(info, "hostname")) == local


def _device_id(info):
    """The "(<device id>)" at the end of an OpenMuse stream name, e.g. "Muse-EEG (ABCD)" -> "abcd"; else ""."""
    m = re.search(r"\(([^()]+)\)\s*$", _info_text(info, "name"))
    return m.group(1).strip().lower() if m else ""


def choose_streams(eeg, motion, local_host):
    """Pick the EEG and motion stream to bind to from all the candidates discovery returned (lists, in
    discovery order; motion may be empty). Returns (eeg_info, motion_info or None, note).

    Preference only: first the pair with the most streams published by this machine (StreamInfo.hostname()
    equal to local_host, ignoring case and a trailing ".local"), then a pair whose names end in the same
    "(<device id>)", then discovery order. So with no match at all the result is (eeg[0], motion[0]), which
    is what was used before. note is "" unless there was a choice to make; then it names what was chosen."""
    eeg, motion = list(eeg), list(motion or [])
    pairs = [(e, m) for e in eeg for m in (motion or [None])]

    def rank(pair):
        e, m = pair
        n_local = int(_is_local(e, local_host)) + int(m is not None and _is_local(m, local_host))
        same_device = m is not None and _device_id(e) != "" and _device_id(e) == _device_id(m)
        return n_local, same_device

    e, m = max(pairs, key=rank)          # max() keeps the first of equals: discovery order breaks ties
    note = ""
    if len(eeg) > 1 or len(motion) > 1:
        def label(info):
            return f"{_info_text(info, 'name') or info} on {_info_text(info, 'hostname') or 'unknown host'}"

        note = f"{len(eeg)} EEG and {len(motion)} motion streams found; using {label(e)}"
        if m is not None:
            note += f" with {label(m)}"
    return e, m, note


def _find_streams(name, timeout):
    """Every Muse stream of this type that LSL discovery returns, in discovery order (same name rule as
    common.find_stream, which returns only the first). Discovery normally returns as soon as one stream
    answers; if none of those is from this machine, it listens for the rest of `timeout` once more so that
    this machine's own stream can still be found and preferred."""
    import pylsl
    stype = name.split("_", 1)[1]             # "EEG" or "ACCGYRO"

    def muse(found):
        return [s for s in found if s.name() == name or s.name().startswith(f"Muse-{stype}")]

    found = muse(pylsl.resolve_byprop("type", stype, timeout=timeout))
    if found and not any(_is_local(s, socket.gethostname()) for s in found):
        seen = {s.uid() for s in found}
        found += [s for s in muse(pylsl.resolve_byprop("type", stype, minimum=64, timeout=timeout))
                  if s.uid() not in seen]
    return found


def _as_list(found):
    """find_stream may return a list of candidates, one stream, or None (the fakes in the tests)."""
    if found is None:
        return []
    return list(found) if isinstance(found, (list, tuple)) else [found]


def _lsl_backend():
    """(find_stream, StreamInlet, processing flags, local_clock). One place to swap in a fake for tests.
    find_stream(name, timeout) returns every candidate stream; LslRunner picks with choose_streams()."""
    import pylsl
    return _find_streams, pylsl.StreamInlet, pylsl.proc_clocksync | pylsl.proc_dejitter, pylsl.local_clock


class LslRunner(threading.Thread):
    """Runs the live detector on the headband's LSL streams in a background thread.

    on_event(dict) receives drift / resume events (LSL times) and, at 2 Hz, a signal message
    {"type": "signal", "t", "head", "eye", "quiet", "alpha_ratio", "ready", "quality": [bool x4]}.
    on_status({"connected": bool, "detail": str, "ready": bool}) is called whenever any of the three changes.
    Clench-hold events (focus/clench_hold.LiveClench, frozen v2 settings, if results/clenchhold_frozen_v2.json exists)
    also go to on_event: {"type": "clench_progress", "t", "progress", "cancelled"} and {"type": "clench", "t", "onset_t"}.
    Both are called from this thread. Exceptions from either are logged and swallowed.

    If a needed stream delivers nothing for 3 s the runner reports disconnected, emits a resume if a drift was
    open, and looks for the streams again every 5 s. Each reconnection starts a fresh detector (not ready
    until it has 20 s of history again). stop() ends the thread.

    set_params(dict) changes the detector's thresholds from its next update on and for every detector built
    after a reconnection; without it (or with None) the frozen parameters are used.

    "quality" comes from signal_quality(): rough and unvalidated.
    """

    stall_s = 3.0
    resolve_every_s = 5.0
    resolve_timeout_s = 1.5
    signal_every_s = 0.5

    def __init__(self, on_event, on_status, variant="combined", poll_s=0.05):
        super().__init__(daemon=True, name="LslRunner")
        self._on_event, self._on_status = on_event, on_status
        self.variant = variant
        self.poll_s = poll_s
        self._stop_evt = threading.Event()
        self._status = None
        self._params = None                  # None = the frozen parameters
        self._det = None                     # the detector of the session in progress
        self._params_lock = threading.Lock()

    def set_params(self, params):
        """Thresholds for the running detector (from its next update on) and for each one built later.
        None goes back to the frozen parameters. Safe to call from any thread, before or after start()."""
        p = _checked_params(load_frozen_params(self.variant) if params is None else params, self.variant)
        with self._params_lock:
            self._params = None if params is None else p
            if self._det is not None:
                self._det.set_params(p)

    def stop(self):
        self._stop_evt.set()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=self.resolve_timeout_s * 4 + 2.0)     # up to two resolves per stream type

    # ------------------------------------------------------------ callbacks
    def _emit(self, ev):
        try:
            self._on_event(ev)
        except Exception:  # noqa: BLE001
            log.exception("on_event callback failed")

    def _set_status(self, connected, detail, ready=False):
        s = dict(connected=bool(connected), detail=str(detail), ready=bool(ready))
        if s == self._status:
            return
        self._status = s
        try:
            self._on_status(dict(s))
        except Exception:  # noqa: BLE001
            log.exception("on_status callback failed")

    # ------------------------------------------------------------ thread
    def run(self):
        self._set_status(False, "Looking for the headband stream")
        while not self._stop_evt.is_set():
            began = time.monotonic()
            try:
                self._session()
            except (Exception, SystemExit) as e:  # noqa: BLE001  (a backend may raise SystemExit, as common._lsl does)
                log.exception("live detector session failed")
                self._set_status(False, f"Detector error: {e}")
            if self._stop_evt.wait(max(0.0, self.resolve_every_s - (time.monotonic() - began))):
                break
        self._set_status(False, "Detector stopped")

    def _session(self):
        """Resolve the streams and run until they stall or stop() is called."""
        find_stream, StreamInlet, flags, local_clock = _lsl_backend()
        need_motion = "theta" in VARIANTS[self.variant][2]
        eeg_found = _as_list(find_stream("Muse_EEG", self.resolve_timeout_s))
        if not eeg_found:
            self._set_status(False, "No headband stream found")
            return
        if self._stop_evt.is_set():
            return
        mot_found = _as_list(find_stream("Muse_ACCGYRO", self.resolve_timeout_s))
        # A preference among the candidates, never a filter (see choose_streams).
        eeg_info, mot_info, chosen = choose_streams(eeg_found, mot_found, socket.gethostname())
        if mot_info is None and need_motion:
            self._set_status(False, "Headband EEG stream found, but no motion stream: the look-away rule needs both")
            return
        with self._params_lock:                     # the current params (None = frozen); fs from the stream
            det = self._det = LiveLookaway(self._params, self.variant)
        inlets = {"eeg": StreamInlet(eeg_info, max_buflen=60, processing_flags=flags)}
        if mot_info is not None:
            inlets["motion"] = StreamInlet(mot_info, max_buflen=60, processing_flags=flags)
        needed = ["eeg", "motion"] if need_motion else ["eeg"]
        detail = "Headband connected" if mot_info is not None else "Headband connected (no motion stream)"
        if chosen:
            detail += f": {chosen}"
        try:
            self._pump(det, inlets, needed, detail, local_clock)
        finally:
            with self._params_lock:
                self._det = None
            if det.drift_open():        # the movement's end was not seen, so there is no net rotation for it
                self._emit(dict(type="resume", t=float(local_clock()), net_rot=None, peak_angle=det.peak_angle()))
            for inlet in inlets.values():
                try:
                    inlet.close_stream()
                except Exception:  # noqa: BLE001
                    pass

    def _pump(self, det, inlets, needed, detail, local_clock):
        now = time.monotonic()
        last_rx = {k: now for k in inlets}
        last_ts = {}
        started = False
        early_motion = []                    # motion that arrives before the first EEG sample
        tail_x, tail_t = np.empty((0, 4)), np.empty(0)     # about the last 1 s of raw EEG, for signal_quality
        # Clench-hold input (focus/clench_hold.py, frozen v2): its own raw-EEG buffer and 0.1-s grid. A failure here
        # disables the clench for this session only; it never stops the look-away detector.
        cparams = clench_hold.load_frozen_v2()
        clench = clench_hold.LiveClench(cparams) if cparams is not None else None
        next_signal = now + self.signal_every_s
        self._set_status(True, detail, False)
        while not self._stop_evt.is_set():
            for key, inlet in inlets.items():
                data, ts = inlet.pull_chunk(timeout=0.0, max_samples=2048)
                if not ts:
                    continue
                x = np.asarray(data, float)
                last_rx[key] = time.monotonic()
                last_ts[key] = float(ts[-1])
                if key == "eeg":
                    if not started:
                        det.start(float(ts[0]))
                        started = True
                        for mx, mt in early_motion:
                            det.push_motion(mx, mt)
                        early_motion = []
                    det.push_eeg(x, ts)
                    if clench is not None:
                        clench.push(x[:, :4], ts)
                    tail_x = np.vstack([tail_x, x[:, :4]])
                    tail_t = np.concatenate([tail_t, np.asarray(ts, float)])
                    keep = tail_t > tail_t[-1] - 1.0
                    tail_x, tail_t = tail_x[keep], tail_t[keep]
                elif started:
                    det.push_motion(x, ts)
                else:
                    early_motion.append((x, ts))
            if started and all(k in last_ts for k in needed):
                for ev in det.advance(min(last_ts[k] for k in needed)):
                    self._emit(ev)
            if started and clench is not None and "eeg" in last_ts:
                try:
                    cev = clench.advance(last_ts["eeg"], det.fs)
                except Exception:  # noqa: BLE001
                    log.exception("clench-hold detector failed; clench disabled until the stream reconnects")
                    clench, cev = None, []
                for ev in cev:
                    self._emit(ev)
            now = time.monotonic()
            stale = [k for k in needed if now - last_rx[k] > self.stall_s]
            if stale:
                name = "EEG" if "eeg" in stale else "motion"
                self._set_status(False, f"No {name} data from the headband for {self.stall_s:g} s")
                return
            latest = det.latest()
            self._set_status(True, detail, latest["ready"])
            if started and now >= next_signal:
                next_signal = now + self.signal_every_s
                if latest["t"] is None:
                    latest["t"] = last_ts["eeg"]
                self._emit(dict(type="signal", **latest, quality=signal_quality(tail_x, det.fs or 256.0)))
            self._stop_evt.wait(self.poll_s)


# ---------------------------------------------------------------- CLI
def _fmt(ev, t0):
    if ev["type"] == "drift":
        angle = "" if ev.get("angle") is None else f", head angle {ev['angle']:.0f} deg"
        return f"[{ev['t'] - t0:8.2f} s] DRIFT  {ev['kind']}  (run began {ev['onset_t'] - t0:.2f} s{angle})"
    peak = "" if ev.get("peak_angle") is None else f"  (peak head angle {ev['peak_angle']:.0f} deg)"
    return f"[{ev['t'] - t0:8.2f} s] resume{peak}"


def _cli_replay(path, speed, variant):
    with np.load(path) as zf:
        sess = parse_session({k: zf[k] for k in ("ev_label", "ev_t", "eeg_ts")})
    t0 = float(sess["t0"])
    with open(FROZEN) as f:
        frozen = json.load(f)
    params = frozen["variants"][variant]["params"]
    dev = hashlib.sha256(Path(path).read_bytes()).hexdigest() == frozen.get("dev_sha256")
    print(f"REPLAY of {path} ({variant}, frozen params {params}), "
          + (f"{speed:g}x real time" if speed else "as fast as possible"))
    if dev:
        print("Note: this is the DEVELOPMENT session the thresholds were tuned on, not a held-out one.")
    events = []
    for _, evs in iter_replay(path, variant, speed=speed or None):
        for ev in evs:
            events.append(ev)
            print(_fmt(ev, t0), flush=True)
    ev_off = to_offline(events, t0)
    kinds = ("eyes_closed",) if variant == "alpha_only" else ("lookaway", "eyes_closed")
    s = score(ev_off, sess, kinds)
    print(f"\n{len(ev_off)} drift events in {sess['dur']:.0f} s. Instructed cues "
          "(a hit is the first drift between the cue and 2 s after its end):")
    for c in s["cues"]:
        res = (f"drift after {c['delay']:.2f} s" + ("" if c["kind_correct"] else " (other kind)")
               if c["hit"] else "MISSED")
        print(f"  cue at {c['on']:6.1f} s  {c['kind']:11} {c['direction']:5}  {res}")
    print(f"  hits {s['hits']}/{s['n_cues']}; drifts during normal reading: {s['fa']} in {s['reading_min']:.1f} min")


def _cli_live(variant):
    t_first = {}

    def on_event(ev):
        if ev["type"] == "signal":
            return
        base = t_first.setdefault("t", ev["t"])
        print(_fmt(ev, base) + f"   (LSL {ev['t']:.2f})", flush=True)

    runner = LslRunner(on_event, lambda s: print(f"status: {s}", flush=True), variant)
    runner.start()
    print("Live look-away detector. Times are seconds since the first event. Ctrl-C to stop.")
    try:
        while runner.is_alive():
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop()


def main():
    ap = argparse.ArgumentParser(description="Live look-away detector: replay a recording or run on the headband.")
    ap.add_argument("--replay", help="a lookaway-*.npz recording to replay")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed (1 = real time; 0 = as fast as possible)")
    ap.add_argument("--variant", default="combined", choices=list(VARIANTS))
    a = ap.parse_args()
    if a.replay:
        _cli_replay(a.replay, a.speed, a.variant)
    else:
        _cli_live(a.variant)


if __name__ == "__main__":
    main()
