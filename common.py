"""Shared helpers: find the Muse streams, record them in the background, mark events, filter."""
import os
import threading
import time

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

CH = ["TP9", "AF7", "AF8", "TP10"]
EEG_NAME = "Muse_EEG"
MOTION_NAME = "Muse_ACCGYRO"
FS = 256.0


def _lsl():
    try:
        import pylsl
        return pylsl
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Could not load pylsl ({e}). Run: pip install pylsl") from e


def local_clock():
    return _lsl().local_clock()


def find_stream(name, timeout):
    """Find a Muse stream. OpenMuse 0.1.8 names them 'Muse-EEG (<device id>)' and
    'Muse-ACCGYRO (<device id>)' (type 'EEG' / 'ACCGYRO'), despite its docstring saying 'Muse_EEG'.
    Accept the exact legacy name or the new 'Muse-<TYPE>' prefix."""
    pylsl = _lsl()
    stype = name.split("_", 1)[1]  # "EEG" or "ACCGYRO"
    found = pylsl.resolve_byprop("type", stype, timeout=timeout)
    ok = [s for s in found if s.name() == name or s.name().startswith(f"Muse-{stype}")]
    if len(ok) > 1:
        print(f"Warning: {len(ok)} {stype} streams found ({[s.name() for s in ok]}); using the first.")
    return ok[0] if ok else None


class Recorder:
    """Pulls the Muse EEG (and motion, if present) in a background thread.

    All timestamps share the laptop's LSL clock, so events marked with mark() line up with the EEG.
    """

    def __init__(self, path, need_motion=False, autosave_every=None):
        pylsl = _lsl()
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.events = []
        self.buf = {}
        self.inlets = {}
        self.lock = threading.Lock()
        self._stop = False
        self.autosave_every = autosave_every
        flags = pylsl.proc_clocksync | pylsl.proc_dejitter
        for key, name, timeout in [("eeg", EEG_NAME, 10), ("motion", MOTION_NAME, 4)]:
            info = find_stream(name, timeout)
            if info is None:
                if key == "eeg":
                    raise SystemExit(
                        "No Muse_EEG stream found. In another terminal run:\n"
                        "  OpenMuse stream --address <your-muse-address>")
                if need_motion:
                    raise SystemExit("No Muse_ACCGYRO stream found. Is OpenMuse streaming with the default preset?")
                continue
            self.inlets[key] = pylsl.StreamInlet(info, max_buflen=600, processing_flags=flags)
            self.buf[key] = ([], [])
            print(f"Connected to {name}: {info.channel_count()} channels at {info.nominal_srate():g} Hz")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        last_save = time.time()
        while not self._stop:
            for key, inlet in self.inlets.items():
                data, ts = inlet.pull_chunk(timeout=0.0, max_samples=2048)
                if ts:
                    with self.lock:
                        self.buf[key][0].extend(data)
                        self.buf[key][1].extend(ts)
            if self.autosave_every and time.time() - last_save > self.autosave_every:
                self.save(final=False)
                last_save = time.time()
            time.sleep(0.02)

    def mark(self, label, t=None):
        t = local_clock() if t is None else t
        with self.lock:
            self.events.append((t, label))
        return t

    def since(self, key, t0, max_samples=2000):
        """(data, timestamps) of samples stamped after t0 (from the last max_samples). Added 2026-11-14 for taps."""
        if key not in self.buf:
            return np.empty((0, 0)), np.empty(0)
        with self.lock:
            d = self.buf[key][0][-max_samples:]
            ts = self.buf[key][1][-max_samples:]
        d, ts = np.asarray(d, dtype=float), np.asarray(ts, dtype=float)
        m = ts > t0
        return d[m], ts[m]

    def latest(self, key, seconds, fs=FS):
        """Last `seconds` of data as (samples x channels) array, or None."""
        if key not in self.buf:
            return None
        n = int(seconds * fs)
        with self.lock:
            d = self.buf[key][0][-n:]
        if len(d) < n * 0.8:
            return None
        return np.asarray(d, dtype=float)

    def save(self, final=True):
        if final:
            self._stop = True
            self._thread.join(timeout=2)
        out = {}
        with self.lock:
            for key, (d, ts) in self.buf.items():
                out[key] = np.asarray(d, dtype=float)
                out[key + "_ts"] = np.asarray(ts, dtype=float)
            out["ev_t"] = np.asarray([e[0] for e in self.events], dtype=float)
            out["ev_label"] = np.asarray([e[1] for e in self.events], dtype=str)
        np.savez(self.path, **out)
        if final:
            n = len(out.get("eeg_ts", []))
            print(f"Saved {self.path}.npz  ({n} EEG samples, {n / FS / 60:.1f} min, {len(self.events)} events)")


def load(path):
    d = np.load(path if path.endswith(".npz") else path + ".npz", allow_pickle=False)
    eeg = d["eeg"]
    if eeg.ndim != 2 or eeg.shape[1] < 4:
        raise SystemExit(f"{path}: expected at least 4 EEG columns, got shape {eeg.shape}")
    return {
        "eeg": eeg[:, :4],  # TP9, AF7, AF8, TP10; any aux columns are dropped
        "eeg_ts": d["eeg_ts"],
        "motion": d["motion"] if "motion" in d.files else None,
        "motion_ts": d["motion_ts"] if "motion_ts" in d.files else None,
        "ev_t": d["ev_t"],
        "ev_label": d["ev_label"],
    }


def effective_fs(ts):
    if len(ts) < 2:
        return FS
    return (len(ts) - 1) / (ts[-1] - ts[0])


def bandpass(x, lo, hi, fs=FS, order=4):
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def rel_power(x, fs, band, total=(0.5, 30.0)):
    """Power in `band` divided by power in `total`, per channel (x: samples x channels)."""
    f, p = welch(x, fs=fs, nperseg=min(len(x), int(fs * 4)), axis=0)
    b = (f >= band[0]) & (f <= band[1])
    t = (f >= total[0]) & (f <= total[1])
    return p[b].sum(axis=0) / np.maximum(p[t].sum(axis=0), 1e-12)


def stamp():
    return time.strftime("%Y%m%d-%H%M%S")
