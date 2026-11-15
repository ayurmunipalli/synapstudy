"""Test 7 analysis: score the nap's sleep stages with YASA and suggest a deep-sleep threshold.

  python stage_nap.py results/nap-XXXX.npz

YASA's model was trained on electrodes at the top of the head (C4); the Muse's forehead channels are
an approximation, so treat the stages as a rough guide.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import CH, effective_fs, load, rel_power  # noqa: E402
from detector import save_model, shares, train_personal_model  # noqa: E402

DELTA = (0.5, 2.0)


def delta_per_epoch(x, fs, sec=30):
    n = int(sec * fs)
    vals = []
    for i in range(0, len(x) - n + 1, n):
        seg = x[i:i + n, 1:3]  # AF7, AF8
        vals.append(rel_power(seg - seg.mean(0), fs, DELTA).mean())
    return np.array(vals)


def beta_per_epoch(x, fs, sec=30):
    """Beta (15-30 Hz) share per 30-s epoch on AF7/AF8, computed exactly as the live detector does."""
    n = int(sec * fs)
    return np.array([shares(x[i:i + n, 1:3], fs)[1] for i in range(0, len(x) - n + 1, n)])


def main(path):
    try:
        import mne
        import yasa
    except ImportError:
        raise SystemExit("Install first: pip install yasa mne")
    d = load(path)
    x, ts = d["eeg"], d["eeg_ts"]
    fs = effective_fs(ts)
    raw = mne.io.RawArray((x - x.mean(0)).T * 1e-6, mne.create_info(CH, fs, "eeg"), verbose=False)  # assumes µV
    raw.filter(0.3, 35, verbose=False)
    stages = {}
    for ch in ("AF8", "AF7"):
        try:
            pred = yasa.SleepStaging(raw, eeg_name=ch).predict()
            # YASA >= 0.7 returns a Hypnogram object labelled WAKE/N1/N2/N3/REM (fix 2026-11-14);
            # older versions return an array labelled W/N1/N2/N3/R. Normalise to the old labels.
            if hasattr(pred, "hypno"):
                pred = pred.hypno.astype(str).to_numpy()
            pred = np.asarray(pred, dtype=str)
            stages[ch] = np.array([{"WAKE": "W", "REM": "R"}.get(s, s) for s in pred])
        except Exception as e:  # noqa: BLE001
            print(f"YASA on {ch} failed: {e}")
    if not stages:
        raise SystemExit("Staging failed on both forehead channels.")
    ch, hyp = next(iter(stages.items()))
    delta = delta_per_epoch(x, fs)
    n = min(len(hyp), len(delta))
    hyp, delta = hyp[:n], delta[:n]

    print(f"Scored with {ch}: {n} epochs of 30 s ({n / 2:.0f} min)")
    for s in ("W", "N1", "N2", "N3", "R"):
        print(f"  {s:3} {np.sum(hyp == s) / 2:5.1f} min")
    beta = beta_per_epoch(x, fs)[:n]
    med = {s: np.median(delta[hyp == s]) for s in ("W", "N2", "N3") if np.any(hyp == s)}
    print("\nForehead delta share (0.5-2 Hz) by stage:", {k: round(v, 3) for k, v in med.items()})
    if "N3" in med and "N2" in med:
        thr = (med["N2"] + med["N3"]) / 2
        bthr = np.median(beta[hyp == "N2"])  # wake guard (added 2026-11-14, see detector.py)
        print("Forehead beta share (15-30 Hz) by stage:",
              {s: round(float(np.median(beta[hyp == s])), 3) for s in ("W", "N2", "N3") if np.any(hyp == s)})
        print(f"Suggested deep-sleep thresholds for sleep_cue.py: --threshold {thr:.3f} --beta-threshold {bthr:.3f}")
    else:
        print("No N3 scored, so no threshold yet. Try a longer nap, or a nap after a short night.")
    # Personal model (added 2026-11-14, design in RESULTS.md): trained on this recording's YASA stages, N3 vs rest.
    model = train_personal_model(x[:len(hyp) * int(round(30 * fs)), 1:3], fs, hyp)
    if model is not None:
        mpath = path.replace(".npz", "") + "_model.npz"
        save_model(model, mpath)
        print(f"Personal model saved ({model['n_pos']} N3 windows of {model['n_train']}): --model {mpath}")
    else:
        print("Too little N3 (< 5 min) to train a personal model.")
    print("PASS if N2 appears (ideally some N3) and the forehead signal stayed usable all nap.")

    # Where did cues / snoozes land? (test 8; added 2026-11-14)
    t0 = ts[0]
    lab = d["ev_label"]
    cue_t = np.array([t for t, l in zip(d["ev_t"], lab) if str(l).startswith("cue|")]) - t0
    snz_t = np.array([t for t, l in zip(d["ev_t"], lab) if str(l) == "snooze"]) - t0
    for name, tt in (("Cues", cue_t), ("Snoozes", snz_t)):
        if len(tt):
            ep = np.clip((tt // 30).astype(int), 0, n - 1)
            st = hyp[ep]
            print(f"{name}: {len(tt)}. By YASA stage: " +
                  ", ".join(f"{s} {np.sum(st == s)} ({np.mean(st == s) * 100:.0f}%)" for s in ("W", "N1", "N2", "N3", "R")))
    if len(cue_t):
        print("Test 8 PASS needs most cues in N2/N3 AND the sleeper not woken (check W after cues, and snoozes).")

    order = {"W": 0, "R": 1, "N1": 2, "N2": 3, "N3": 4}
    fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    t = np.arange(n) / 2
    ax[0].step(t, [order.get(s, 0) for s in hyp], where="post")
    ax[0].set_yticks(range(5), ["Wake", "REM", "N1", "N2", "N3"])
    ax[0].invert_yaxis()
    ax[0].set_title(f"Hypnogram (YASA on {ch})")
    for tt, col, lab_ in ((cue_t, "tab:red", "cue"), (snz_t, "tab:purple", "snooze")):
        if len(tt):
            ax[0].plot(tt / 60, np.full(len(tt), 4.4 if lab_ == "cue" else -0.4), "|", color=col, ms=10, label=lab_)
    if len(cue_t) or len(snz_t):
        ax[0].legend(loc="lower right", fontsize=8)
    ax[1].plot(t, delta)
    ax[1].set_ylabel("delta share")
    ax[1].set_xlabel("minutes")
    out = path.replace(".npz", "") + "_stages.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"Plot saved: {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
    else:
        main(sys.argv[1])
