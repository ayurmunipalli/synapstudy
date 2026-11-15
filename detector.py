"""Deep-sleep detector shared by sleep_cue.py (live) and sleep_edf_check.py (offline replay).

Added 2026-11-14. Design fixed in RESULTS.md before testing ("Wake guard for the cue detector").

Each check looks at the last 30 s of forehead EEG:
  delta share = power 0.5-2 Hz / power 0.5-30 Hz        must be > delta_thr
  beta share  = power 15-30 Hz / power 0.5-30 Hz        must be < beta_thr   (wake guard 2; None = off)
and both must hold continuously for `persist` seconds (wake guard 1) before the detector says "deep".
persist=5, beta_thr=None reproduces the original sleep_cue.py rule (2 checks in a row, 5 s apart).
"""
import numpy as np

from common import rel_power

DELTA = (0.5, 2.0)
BETA = (15.0, 30.0)
TOTAL = (0.5, 30.0)
PERSIST_GUARDED = 120.0
PERSIST_ORIGINAL = 5.0


def shares(seg, fs):
    """seg: samples x channels (30 s). Returns (delta share, beta share), averaged over channels."""
    seg = seg - seg.mean(axis=0)
    return (float(rel_power(seg, fs, DELTA, TOTAL).mean()),
            float(rel_power(seg, fs, BETA, TOTAL).mean()))


class DeepSleepDetector:
    def __init__(self, delta_thr, beta_thr=None, persist=PERSIST_GUARDED):
        self.delta_thr, self.beta_thr, self.persist = delta_thr, beta_thr, persist
        self.run_start = None
        self.last = (np.nan, np.nan)

    def reset(self):
        self.run_start = None

    def update(self, seg, fs, t):
        """Feed the last 30 s at time t (seconds). Returns True while 'deep'."""
        d, b = shares(seg, fs)
        self.last = (d, b)
        ok = d > self.delta_thr and (self.beta_thr is None or b < self.beta_thr)
        if not ok:
            self.run_start = None
            return False
        if self.run_start is None:
            self.run_start = t
        return (t - self.run_start) >= self.persist


# ---------------------------------------------------------------------------------------------------------------
# Personal deep-sleep model (added 2026-11-14; design fixed in RESULTS.md before testing).
# Trained on one recording of a person, labelled by YASA (N3 vs everything else); used live on their next one.
# ---------------------------------------------------------------------------------------------------------------
BANDS = [(0.5, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 15.0), (15.0, 30.0)]
CONTEXT_S = 300.0  # trailing 5-min mean of each window feature
GRID_S = 5.0       # live checks and training windows both on a 5-s grid


def window_features(seg, fs):
    """seg: samples x channels (30 s). 7 features: log relative power per band (of 0.5-30 Hz) + log total power."""
    from scipy.signal import welch
    seg = seg - seg.mean(axis=0)
    f, p = welch(seg, fs=fs, nperseg=min(len(seg), int(fs * 4)), axis=0)
    p = p.mean(axis=1)  # average channels
    tot = p[(f >= 0.5) & (f <= 30)].sum()
    rel = [p[(f >= a) & (f < b)].sum() / max(tot, 1e-20) for a, b in BANDS]
    return np.log(np.r_[rel, tot] + 1e-12)


def _with_context(times, feats):
    out = np.empty((len(feats), feats.shape[1] * 2))
    for i, t in enumerate(times):
        m = (times > t - CONTEXT_S) & (times <= t)
        out[i] = np.r_[feats[i], feats[m].mean(axis=0)]
    return out


def grid_features(x, fs):
    """x: samples x channels. Features for 30-s windows ending every GRID_S s. Returns (end times s, n x 14)."""
    n30 = int(30 * fs)
    ends = np.arange(30.0, len(x) / fs + 1e-9, GRID_S)
    feats = np.array([window_features(x[int(t * fs) - n30:int(t * fs)], fs) for t in ends])
    return ends, _with_context(ends, feats)


def train_personal_model(x, fs, hyp, min_n3_epochs=10, target=("N3",)):
    """x: samples x channels (forehead), hyp: 30-s stage labels (W/N1/N2/N3/R) for the same recording.
    target: stages counted as positive (default N3; ("N2", "N3") for the stable-NREM fallback, added 2026-11-14).
    Returns a dict of model parameters, or None if there is too little of the target to learn from."""
    from sklearn.linear_model import LogisticRegression
    hyp = np.asarray(hyp)
    if np.sum(np.isin(hyp, target)) < min_n3_epochs:
        return None
    ends, X = grid_features(x, fs)
    ep = ((ends - 15.0) // 30).astype(int)  # label = stage of the epoch holding the window's centre
    ok = (ep >= 0) & (ep < len(hyp))
    ok &= np.isin(hyp[np.clip(ep, 0, len(hyp) - 1)], ["W", "N1", "N2", "N3", "R"])
    X, y = X[ok], np.isin(hyp[ep[ok]], target).astype(int)
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    m = LogisticRegression(C=1.0, max_iter=2000).fit((X - mu) / sd, y)
    return {"mean": mu, "std": sd, "coef": m.coef_[0], "intercept": float(m.intercept_[0]),
            "n_train": int(len(y)), "n_pos": int(y.sum())}


def save_model(model, path):
    np.savez(path, **model)


def load_model(path):
    d = np.load(path)
    return {k: (d[k] if d[k].ndim else d[k].item()) for k in d.files}


class PersonalModelDetector:
    """Same interface as DeepSleepDetector. 'Deep' = P(N3) > p_thr held for `persist` s."""

    def __init__(self, model, p_thr=0.5, persist=PERSIST_GUARDED):
        self.m, self.p_thr, self.persist = model, p_thr, persist
        self.hist_t, self.hist_f = [], []
        self.run_start = None
        self.last = (np.nan, np.nan)

    def reset(self):
        self.run_start = None  # context history is kept: it is still valid after a pause

    def update(self, seg, fs, t):
        f = window_features(seg, fs)
        if not self.hist_t or t - self.hist_t[-1] >= GRID_S - 1e-6:  # keep the 5-s grid used in training
            self.hist_t.append(t)
            self.hist_f.append(f)
            while self.hist_t and self.hist_t[0] <= t - CONTEXT_S:
                self.hist_t.pop(0)
                self.hist_f.pop(0)
        ctx = np.mean(self.hist_f, axis=0) if self.hist_f else f
        z = (np.r_[f, ctx] - self.m["mean"]) / self.m["std"]
        p = 1.0 / (1.0 + np.exp(-(z @ self.m["coef"] + self.m["intercept"])))
        self.last = (float(p), np.nan)
        if p <= self.p_thr:
            self.run_start = None
            return False
        if self.run_start is None:
            self.run_start = t
        return (t - self.run_start) >= self.persist


# ---------------------------------------------------------------------------------------------------------------
# Double-tap snooze (added 2026-11-14): two sharp accelerometer jolts 0.15-0.6 s apart.
# ---------------------------------------------------------------------------------------------------------------
TAP_GAP = (0.15, 0.6)


def find_taps(acc, ts, thr):
    """acc: samples x 3 (g). Tap times = jerk (|diff acc| per sample) above thr; peaks within 0.1 s merged."""
    if len(acc) < 3:
        return np.array([])
    jerk = np.linalg.norm(np.diff(acc, axis=0), axis=1)
    idx = np.where(jerk > thr)[0]
    taps, last_above = [], -np.inf
    for i in idx:
        t = ts[i + 1]
        if t - last_above > 0.1:  # a quiet gap of >0.1 s starts a new tap; one long jolt stays one tap
            taps.append(t)
        last_above = t
    return np.array(taps)


def find_double_taps(acc, ts, thr, gap=TAP_GAP):
    """Times of the second tap of each double tap."""
    taps = find_taps(acc, ts, thr)
    out, i = [], 0
    while i < len(taps) - 1:
        if gap[0] <= taps[i + 1] - taps[i] <= gap[1]:
            out.append(taps[i + 1])
            i += 2
        else:
            i += 1
    return np.array(out)
