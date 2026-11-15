"""Rule-based look-away / eyes-closed detectors and scoring, exactly as pre-registered in RESULTS.md
("Look-away / eyes-closed check: evaluation plan", fixed 2026-11-14 ~21:40, before any session).

Everything is causal: the value at update time t uses only samples with timestamp <= t. Updates every 0.25 s.
Features are computed once per session; the parameter grid only changes the comparisons.

Implementation choices not spelled out in the plan (fixed before tuning, recorded in RESULTS.md):
  - Head: per-axis gyro integrated over the last 1.0 s (trapezoid on the motion timestamps), angle = vector norm.
    No gyro-bias removal (the plan doesn't have one).
  - Eye: hEOG = AF7 - AF8, 4th-order Butterworth 5 Hz low-pass run causally (sosfilt) over the whole session.
    Compared value = the latest filtered sample. Spread = P95 - P5 of the last 20 s (current sample included).
  - Eye-quiet: max - min of filtered hEOG over the last 2 s, divided by that same 20-s spread.
  - Alpha: last 2 s of TP9 and TP10, mean removed, Hann-windowed periodogram; 8-12 Hz power / 2-30 Hz power per
    channel, then the mean of the two. Ratio = that value / median of the alpha values at updates in the last 60 s.
  - A detector is negative until its window has full history (20 s eye, 2 s alpha, 1 s head).
  - Drift event: each detector (look-away, eyes-closed) keeps its own run count; an event starts at the update
    where either reaches N (onset = that update's time, i.e. when it would be shown). The event clears after
    8 consecutive updates with both detectors negative. Kind = the detector(s) that reached N at onset.
"""
import numpy as np
from scipy.signal import butter, sosfilt

STEP = 0.25
CLEAR_UPDATES = 8
GRID = dict(theta=[10, 15, 20, 30], k=[0.75, 1, 1.5, 2], q=[0.3, 0.5], r=[1.5, 2, 3], N=[2, 4])

# variant -> (look-away rule, eyes-closed rule, params it uses). Rules take the dict of boolean detector traces.
VARIANTS = {
    "combined":     (lambda d: d["head"] | d["eye"], lambda d: d["alpha"] & d["quiet"], ["theta", "k", "q", "r", "N"]),
    "motion_only":  (lambda d: d["head"],             None,                              ["theta", "N"]),
    "eye_only":     (lambda d: d["eye"],              lambda d: d["quiet"],              ["k", "q", "N"]),
    "eeg_no_motion": (lambda d: d["eye"],             lambda d: d["alpha"] & d["quiet"], ["k", "q", "r", "N"]),
    "alpha_only":   (None,                            lambda d: d["alpha"],              ["r", "N"]),
}


# ---------------------------------------------------------------- session parsing
def parse_session(z):
    lab, et = list(z["ev_label"]), np.asarray(z["ev_t"], float)
    t0 = et[[i for i, l in enumerate(lab) if l.startswith("session_start")][0]]
    ends = [et[i] for i, l in enumerate(lab) if l in ("session_end", "session_stopped_early")]
    t_end = ends[0] if ends else float(z["eeg_ts"][-1])
    cues, m = [], []
    for i, l in enumerate(lab):
        if l.startswith("cue|"):
            _, kind, direction, _dur = l.split("|")
            j = next(j for j in range(i + 1, len(lab)) if lab[j].startswith("cue_end|"))
            cues.append(dict(kind=kind, direction=direction, on=et[i] - t0, end=et[j] - t0))
        elif l == "self_drift":
            m.append(et[i] - t0)
    return dict(t0=t0, dur=t_end - t0, cues=cues, m=m)


# ---------------------------------------------------------------- features (parameter-free)
def features(z, sess):
    t0, dur = sess["t0"], sess["dur"]
    ets = np.asarray(z["eeg_ts"], float) - t0
    eeg = np.asarray(z["eeg"], float)[:, :4]
    fs = (len(ets) - 1) / (ets[-1] - ets[0])
    sos = butter(4, 5, btype="low", fs=fs, output="sos")
    heog = sosfilt(sos, eeg[:, 1] - eeg[:, 2] - (eeg[0, 1] - eeg[0, 2]))  # start near 0 to limit the transient
    tp = eeg[:, [0, 3]]
    mts = np.asarray(z["motion_ts"], float) - t0
    gyro = np.asarray(z["motion"], float)[:, 3:6]
    # cumulative integral of gyro (deg) at each motion sample, for 1-s window differences
    cum = np.vstack([np.zeros(3), np.cumsum(0.5 * (gyro[1:] + gyro[:-1]) * np.diff(mts)[:, None], axis=0)])

    ut = np.arange(STEP, dur + 1e-9, STEP)
    n = len(ut)
    head = np.full(n, np.nan); eye = np.full(n, np.nan); quiet = np.full(n, np.nan); alpha = np.full(n, np.nan)
    n2 = int(round(2 * fs))
    for i, t in enumerate(ut):
        j = np.searchsorted(ets, t, side="right")          # samples with ts <= t
        # head
        if t >= 1.0 and mts[0] <= t - 1.0:
            a = np.searchsorted(mts, t - 1.0, side="right") - 1
            b = np.searchsorted(mts, t, side="right") - 1
            head[i] = np.linalg.norm(cum[b] - cum[a])
        # eye
        if ets[0] <= t - 20.0:
            w = heog[np.searchsorted(ets, t - 20.0, side="right"):j]
            med = np.median(w)
            p5, p95 = np.percentile(w, [5, 95])
            spread = max(p95 - p5, 1e-9)
            eye[i] = abs(heog[j - 1] - med) / spread
            w2 = heog[np.searchsorted(ets, t - 2.0, side="right"):j]
            quiet[i] = (w2.max() - w2.min()) / spread
        # alpha
        if j >= n2:
            x = tp[j - n2:j] - tp[j - n2:j].mean(axis=0)
            x = x * np.hanning(n2)[:, None]
            p = np.abs(np.fft.rfft(x, axis=0)) ** 2
            f = np.fft.rfftfreq(n2, 1 / fs)
            rel = p[(f >= 8) & (f <= 12)].sum(0) / np.maximum(p[(f >= 2) & (f <= 30)].sum(0), 1e-12)
            alpha[i] = rel.mean()
    # alpha relative to its own median over the last 60 s of updates
    alpha_ratio = np.full(n, np.nan)
    for i in range(n):
        lo = np.searchsorted(ut, ut[i] - 60.0, side="right")
        h = alpha[lo:i + 1]
        h = h[~np.isnan(h)]
        if len(h) and not np.isnan(alpha[i]):
            alpha_ratio[i] = alpha[i] / max(np.median(h), 1e-12)
    return dict(t=ut, head=head, eye=eye, quiet=quiet, alpha=alpha, alpha_ratio=alpha_ratio, fs=fs,
                heog=heog, ets=ets)


def detectors(F, p):
    nn = lambda x: np.nan_to_num(x, nan=-np.inf)
    return dict(
        head=nn(F["head"]) > p.get("theta", np.inf),
        eye=nn(F["eye"]) > p.get("k", np.inf),
        quiet=np.nan_to_num(F["quiet"], nan=np.inf) < p.get("q", -np.inf),
        alpha=nn(F["alpha_ratio"]) > p.get("r", np.inf),
    )


# ---------------------------------------------------------------- events
def drift_events(t, la, ec, N):
    """la/ec: boolean arrays (or None). Returns list of dict(onset, kind, end)."""
    n = len(t)
    la = np.zeros(n, bool) if la is None else la
    ec = np.zeros(n, bool) if ec is None else ec
    ev, active, c_la, c_ec, clean = [], None, 0, 0, 0
    for i in range(n):
        c_la = c_la + 1 if la[i] else 0
        c_ec = c_ec + 1 if ec[i] else 0
        if active is None:
            kinds = [k for k, c in (("lookaway", c_la), ("eyes_closed", c_ec)) if c >= N]
            if kinds:
                active = dict(onset=t[i], kind="+".join(kinds), end=None)
                clean = 0
        else:
            clean = clean + 1 if not (la[i] or ec[i]) else 0
            if clean >= CLEAR_UPDATES:
                active["end"] = t[i]
                ev.append(active)
                active, c_la, c_ec = None, 0, 0
    if active is not None:
        ev.append(active)
    return ev


# ---------------------------------------------------------------- scoring
def normal_reading_mask(t, sess):
    ok = t >= 30.0
    for c in sess["cues"]:
        ok &= ~((t >= c["on"] - 1.0) & (t <= c["end"] + 5.0))
    for m in sess["m"]:
        ok &= ~((t >= m - 10.0) & (t <= m + 2.0))
    return ok


def score(ev, sess, cue_kinds=("lookaway", "eyes_closed")):
    cues = [c for c in sess["cues"] if c["kind"] in cue_kinds]
    per = []
    for c in cues:
        hits = [e for e in ev if c["on"] <= e["onset"] <= c["end"] + 2.0]
        h = hits[0] if hits else None
        per.append(dict(kind=c["kind"], direction=c["direction"], on=c["on"], hit=h is not None,
                        delay=None if h is None else h["onset"] - c["on"],
                        kind_correct=h is not None and c["kind"] in h["kind"].split("+")))
    grid = np.arange(0, sess["dur"], 0.01)
    ok = normal_reading_mask(grid, sess)
    reading_min = ok.sum() * 0.01 / 60
    fa_ev = [e for e in ev if normal_reading_mask(np.array([e["onset"]]), sess)[0]]
    m_caught = [any(m - 10 <= e["onset"] <= m + 2 for e in ev) for m in sess["m"]]
    return dict(cues=per, n_cues=len(per), hits=sum(p["hit"] for p in per),
                delays=[p["delay"] for p in per if p["hit"]], kind_correct=sum(p["kind_correct"] for p in per),
                fa=len(fa_ev), fa_onsets=[e["onset"] for e in fa_ev], reading_min=reading_min,
                m_presses=len(sess["m"]), m_with_event=sum(m_caught), n_events=len(ev))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def run_variant(F, sess, variant, p):
    la_rule, ec_rule, _ = VARIANTS[variant]
    d = detectors(F, p)
    ev = drift_events(F["t"], la_rule(d) if la_rule else None, ec_rule(d) if ec_rule else None, p["N"])
    kinds = ("eyes_closed",) if variant == "alpha_only" else ("lookaway", "eyes_closed")
    return ev, score(ev, sess, kinds)
