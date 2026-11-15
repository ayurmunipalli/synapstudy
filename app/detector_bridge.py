"""Where lapse events come from (APP_SPEC.md §8): the live headband, a replayed recording, or the keyboard.

    live       focus.live.LslRunner reports a connected headband
    replay     a saved results/lookaway-*.npz is being played back at real speed (refused while live)
    simulated  neither of the above; drift/resume come from sim() only

Every drift/resume message carries its source, so nothing downstream has to guess. `on_message(dict)` is
called from this module's own threads and from callers' threads; it must be thread-safe.

`t` / `onset_t` are in the source's own clock: LSL seconds (live), the recording's LSL seconds (replay) or
epoch seconds (simulated). `wall_ms` / `onset_wall_ms` are always epoch milliseconds on this Mac.

Live and replayed events also say how the head moved (degrees; null when the detector had no motion data):
a drift carries `angle` and `rot` [x, y, z], a resume carries `net_rot` [x, y, z] and `peak_angle`
(focus/live.py). Simulated events have none of these.
"""
import importlib
import math
import re
import threading
import time
from pathlib import Path

from app import config

REPLAY_FILE_RE = re.compile(r"^lookaway-[0-9-]+\.npz$")
SIM_ONSET_BACKDATE_MS = 750.0


class ReplayError(ValueError):
    """A replay request that was refused; the message is a plain sentence for the user."""


def _finite(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _deg(x) -> float | None:
    x = _finite(x)
    return None if x is None else round(x, 2)


def _deg3(v) -> list | None:
    """A rotation vector [x, y, z] in degrees, or None unless it is three finite numbers."""
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        return None
    out = [_deg(x) for x in v]
    return None if None in out else out


def _acted(kind) -> bool:
    return any(part in config.ACTED_KINDS for part in str(kind).split("+"))


class _Replay(threading.Thread):
    """Plays one recording's detector events at real speed, then ends."""

    def __init__(self, bridge: "DetectorBridge", path: Path, speed: float):
        super().__init__(name="replay", daemon=True)
        self.bridge, self.path, self.speed = bridge, path, max(float(speed), 1e-6)
        self.ready = False
        self._halt = threading.Event()

    def stop(self) -> None:
        self._halt.set()

    def run(self) -> None:
        b = self.bridge
        if self._halt.is_set():
            b._replay_done(self, None)
            return
        try:
            events, t_ref, dur = b._replay_loader(self.path)
            events = sorted((e for e in events if isinstance(e, dict) and e.get("type") in ("drift", "resume")
                             and _finite(e.get("t")) is not None), key=lambda e: float(e["t"]))
            t_ref = float(t_ref)
        except (Exception, SystemExit) as e:  # noqa: BLE001 - any failure just ends the replay
            b._replay_done(self, f"Replay of {self.path.name} failed ({type(e).__name__}).")
            return
        self.ready = True
        b._emit_status()

        start = time.monotonic()
        open_drift = False
        for ev in events:
            due = (float(ev["t"]) - t_ref) / self.speed
            if self._halt.wait(max(0.0, due - (time.monotonic() - start))):
                break
            b._emit(b._replay_message(ev))
            open_drift = ev["type"] == "drift"
        else:
            if _finite(dur) is not None:  # stay in replay until the recording itself would have ended
                self._halt.wait(max(0.0, float(dur) / self.speed - (time.monotonic() - start)))
        if open_drift:
            b._emit({"type": "resume", "source": "replay", "wall_ms": time.time() * 1000.0,
                     "t": t_ref + (time.monotonic() - start) * self.speed, "net_rot": None, "peak_angle": None})
        b._replay_done(self, None)


class DetectorBridge:
    def __init__(self, on_message, *, live_module=None, results_dir: Path | None = None,
                 replay_loader=None, replay_speed: float = 1.0, lsl_clock=None):
        """`on_message(dict)` receives ready-to-broadcast messages (§8 shapes) plus
        {"type": "status", ...status()} whenever the source or headband state changes.

        The keyword arguments exist for tests: `live_module` stands in for focus.live, `replay_loader(path)`
        returns (events, t_ref, duration_s), `lsl_clock()` stands in for pylsl.local_clock.
        """
        self._on_message = on_message
        self._live_module = live_module
        self._results_dir = Path(results_dir) if results_dir is not None else config.RESULTS
        self._replay_loader = replay_loader or self._load_replay
        self._replay_speed = replay_speed
        self._lsl_clock = lsl_clock
        self._lock = threading.RLock()
        self._runner = None
        self._live_connected = False
        self._live_ready = False
        self._live_detail = "The live detector is not running; lapses can only be simulated."
        self._replay: _Replay | None = None
        self._replay_error: str | None = None
        self._sim_open = False
        self._sim_clench_running = False
        self._params: dict | None = None      # detector thresholds; None = the frozen ones

    # ------------------------------------------------------------------ lifecycle
    def _live(self):
        if self._live_module is None:
            self._live_module = importlib.import_module("focus.live")
        return self._live_module

    def start(self) -> None:
        """Start the live detector. If focus.live or pylsl is missing the source just stays simulated."""
        with self._lock:
            if self._runner is not None:
                return
        try:
            runner = self._live().LslRunner(on_event=self._on_runner_event, on_status=self._on_runner_status,
                                             variant=config.DETECTOR_VARIANT)
            params = self.params()
            if params is not None:          # a runner starts with the frozen thresholds
                runner.set_params(params)
        except (Exception, SystemExit) as e:  # noqa: BLE001
            self._start_failed(e)
            return
        with self._lock:
            self._runner = runner
            self._live_detail = "Looking for the headband stream…"
        self._emit_status()
        try:
            runner.start()
        except (Exception, SystemExit) as e:  # noqa: BLE001
            self._start_failed(e)

    def _start_failed(self, error) -> None:
        with self._lock:
            self._runner = None
            self._live_connected = self._live_ready = False
            self._live_detail = (f"The live detector could not start ({type(error).__name__}: {error}); "
                                 "lapses can only be simulated.")
        self._emit_status()

    def stop(self) -> None:
        with self._lock:
            runner, self._runner = self._runner, None
            replay = self._replay
        for worker in (runner, replay):
            if worker is None:
                continue
            try:
                worker.stop()
                if worker is not threading.current_thread() and worker.is_alive():
                    worker.join(timeout=2.0)
            except Exception:  # noqa: BLE001
                pass
        with self._lock:
            self._live_connected = self._live_ready = False

    # ------------------------------------------------------------------ detector thresholds
    def set_params(self, params: dict | None) -> None:
        """The detector's thresholds (None = the frozen ones). They reach the live runner at once and apply
        to every replay started from now on; a replay already running keeps the ones it was made with.
        Raises ValueError if the runner refuses them (nothing is changed then)."""
        params = None if params is None else dict(params)
        with self._lock:
            setter = getattr(self._runner, "set_params", None)
            if setter is not None:
                setter(params)
            self._params = params

    def params(self) -> dict | None:
        with self._lock:
            return None if self._params is None else dict(self._params)

    # ------------------------------------------------------------------ state
    def _source(self) -> str:
        if self._live_connected:
            return "live"
        return "replay" if self._replay is not None else "simulated"

    def status(self) -> dict:
        with self._lock:
            source = self._source()
            if source == "live":
                return {"source": "live", "headband": True, "detector_ready": self._live_ready,
                        "detail": self._live_detail or "Headband connected"}
            if source == "replay":
                ready = self._replay.ready
                name = self._replay.path.name
                return {"source": "replay", "headband": False, "detector_ready": ready,
                        "detail": f"Replaying {name}" if ready else f"Preparing replay of {name}…"}
            return {"source": "simulated", "headband": False, "detector_ready": False,
                    "detail": self._replay_error or self._live_detail or "No headband stream found"}

    def _emit(self, message: dict) -> None:
        try:
            self._on_message(message)
        except Exception:  # noqa: BLE001 - a failing listener must not kill a detector thread
            pass

    def _emit_status(self) -> None:
        self._emit({"type": "status", **self.status()})

    # ------------------------------------------------------------------ live (LslRunner callbacks)
    def _wall_ms(self, t) -> float:
        """Epoch ms for an LSL timestamp: now, minus how long ago the LSL clock says it was."""
        now = time.time() * 1000.0
        t = _finite(t)
        if t is None:
            return now
        try:
            if self._lsl_clock is None:
                import pylsl
                self._lsl_clock = pylsl.local_clock
            return now - (float(self._lsl_clock()) - t) * 1000.0
        except (Exception, SystemExit):  # noqa: BLE001
            return now

    def _on_runner_status(self, st: dict) -> None:
        connected, ready = bool(st.get("connected")), bool(st.get("ready"))
        detail = str(st.get("detail") or ("Headband connected" if connected else "No headband stream found"))
        close_sim, replay = False, None
        with self._lock:
            before = (self._live_connected, self._live_ready, self._live_detail)
            self._live_connected, self._live_ready, self._live_detail = connected, connected and ready, detail
            changed = before != (self._live_connected, self._live_ready, self._live_detail)
            if connected and not before[0]:
                # going live ends anything that is not live
                self._replay_error = None
                replay = self._replay
                close_sim, self._sim_open = self._sim_open, False
        if close_sim:
            now = time.time() * 1000.0
            self._emit({"type": "resume", "source": "simulated", "wall_ms": now, "t": now / 1000.0})
        if replay is not None:
            replay.stop()
            if replay is not threading.current_thread():
                replay.join(timeout=2.0)
        if changed:
            self._emit_status()

    def _on_runner_event(self, ev: dict) -> None:
        kind = ev.get("type")
        if kind == "drift":
            self._emit({
                "type": "drift", "kind": str(ev.get("kind")), "source": "live", "acted": _acted(ev.get("kind")),
                "wall_ms": self._wall_ms(ev.get("t")), "onset_wall_ms": self._wall_ms(ev.get("onset_t")),
                "t": _finite(ev.get("t")), "onset_t": _finite(ev.get("onset_t")), "confidence": None,
                "angle": _deg(ev.get("angle")), "rot": _deg3(ev.get("rot")),
            })
        elif kind == "resume":
            self._emit({"type": "resume", "source": "live", "wall_ms": self._wall_ms(ev.get("t")),
                        "t": _finite(ev.get("t")),
                        "net_rot": _deg3(ev.get("net_rot")), "peak_angle": _deg(ev.get("peak_angle"))})
        elif kind == "clench_progress":
            # jaw-clench hold building up (focus/clench_hold.py, frozen v2); for the on-screen ring only, not stored
            self._emit({"type": "clench_progress", "source": "live", "wall_ms": self._wall_ms(ev.get("t")),
                        "progress": _finite(ev.get("progress")) or 0.0, "cancelled": bool(ev.get("cancelled"))})
        elif kind == "clench":
            # a completed 3-s jaw-clench hold: the wearer's own "I zoned out" signal (a jaw-muscle signal on the
            # EEG sensors, not brain activity). onset = when the hold started.
            self._emit({"type": "clench", "source": "live", "wall_ms": self._wall_ms(ev.get("t")),
                        "onset_wall_ms": self._wall_ms(ev.get("onset_t")),
                        "t": _finite(ev.get("t")), "onset_t": _finite(ev.get("onset_t"))})
        elif kind == "signal":
            quality = ev.get("quality")
            self._emit({
                "type": "signal", "wall_ms": self._wall_ms(ev.get("t")),
                # NaN is not valid JSON for the browser, so anything non-finite goes out as null
                "head": _finite(ev.get("head")), "eye": _finite(ev.get("eye")),
                "alpha_ratio": _finite(ev.get("alpha_ratio")),
                "quality": [bool(q) for q in quality] if quality is not None else [False] * 4,
                "ready": bool(ev.get("ready")),
            })

    # ------------------------------------------------------------------ simulated (keyboard D / R)
    def sim_clench(self) -> bool:
        """Simulated 3-s jaw-clench hold (key Z / the video's "Simulate clench" button): a progress ramp over
        3 s, then a clench event, all with source "simulated". Ignored (False) unless the source is `simulated`,
        or while a simulated hold is already running. The ramp stops if the source changes."""
        with self._lock:
            if self._source() != "simulated" or self._sim_clench_running:
                return False
            self._sim_clench_running = True

        def ramp():
            try:
                start = time.time() * 1000.0
                for k in range(1, 31):
                    time.sleep(0.1)
                    with self._lock:
                        if self._source() != "simulated":
                            self._emit({"type": "clench_progress", "source": "simulated",
                                        "wall_ms": time.time() * 1000.0, "progress": 0.0, "cancelled": True})
                            return
                    self._emit({"type": "clench_progress", "source": "simulated", "wall_ms": time.time() * 1000.0,
                                "progress": k / 30.0, "cancelled": False})
                now = time.time() * 1000.0
                self._emit({"type": "clench", "source": "simulated", "wall_ms": now, "onset_wall_ms": start,
                            "t": now / 1000.0, "onset_t": start / 1000.0})
                self._emit({"type": "clench_progress", "source": "simulated", "wall_ms": now,
                            "progress": 0.0, "cancelled": False})
            finally:
                with self._lock:
                    self._sim_clench_running = False

        threading.Thread(target=ramp, daemon=True, name="SimClench").start()
        return True

    def sim(self, action) -> bool:
        """Fire a simulated drift or resume. Ignored (False) unless the source is `simulated`; a resume
        with no simulated drift open is ignored too. A drift while one is still open here first closes
        that one with a simulated resume and then fires a fresh drift: the browser can end a simulated
        lapse without telling the bridge, and D must still work afterwards."""
        with self._lock:
            if self._source() != "simulated":
                return False
            reopen = False
            if action == "drift":
                reopen, self._sim_open = self._sim_open, True
            elif action == "resume" and self._sim_open:
                self._sim_open = False
            else:
                return False
        now = time.time() * 1000.0
        if action == "drift":
            if reopen:
                self._emit({"type": "resume", "source": "simulated", "wall_ms": now, "t": now / 1000.0})
            onset = now - SIM_ONSET_BACKDATE_MS
            self._emit({"type": "drift", "kind": "look_away", "source": "simulated", "acted": True,
                        "wall_ms": now, "onset_wall_ms": onset, "t": now / 1000.0, "onset_t": onset / 1000.0,
                        "confidence": None})
        else:
            self._emit({"type": "resume", "source": "simulated", "wall_ms": now, "t": now / 1000.0})
        return True

    # ------------------------------------------------------------------ replay
    def replay_path(self, filename) -> Path:
        """The recording to replay: a bare file name `lookaway-<digits and dashes>.npz` that exists
        directly inside results/. Anything else raises ReplayError. The file is only ever read."""
        if not isinstance(filename, str) or not REPLAY_FILE_RE.fullmatch(filename):
            raise ReplayError("Replay needs a file name like lookaway-20261115-000831.npz from results/.")
        base = self._results_dir.resolve()
        path = (base / filename).resolve()
        if path.parent != base or not path.is_file():
            raise ReplayError(f"There is no recording called {filename} in results/.")
        return path

    def _load_replay(self, path: Path):
        """Detector events for a recording (focus.live.replay_events with the thresholds in effect now:
        the frozen ones unless set_params changed them), the LSL time its session started at, and the
        session length in seconds."""
        events = self._live().replay_events(str(path), variant=config.DETECTOR_VARIANT, params=self.params())
        import numpy as np
        from focus.lookaway import parse_session
        with np.load(path, allow_pickle=False) as z:
            sess = parse_session(z)
        return events, float(sess["t0"]), float(sess["dur"])

    def _replay_message(self, ev: dict) -> dict:
        now = time.time() * 1000.0
        t = _finite(ev.get("t"))
        if ev["type"] == "resume":
            return {"type": "resume", "source": "replay", "wall_ms": now, "t": t,
                    "net_rot": _deg3(ev.get("net_rot")), "peak_angle": _deg(ev.get("peak_angle"))}
        onset_t = _finite(ev.get("onset_t"))
        lead_ms = max(0.0, (t - onset_t) * 1000.0) if onset_t is not None else 0.0
        return {"type": "drift", "kind": str(ev.get("kind")), "source": "replay", "acted": _acted(ev.get("kind")),
                "wall_ms": now, "onset_wall_ms": now - lead_ms, "t": t, "onset_t": onset_t, "confidence": None,
                "angle": _deg(ev.get("angle")), "rot": _deg3(ev.get("rot"))}

    def start_replay(self, filename) -> None:
        """Play a saved recording at real speed. Raises ReplayError if the name is not acceptable or the
        headband is live."""
        path = self.replay_path(filename)
        self.stop_replay()
        with self._lock:
            if self._live_connected:
                raise ReplayError("Replay is not available while the headband is live.")
            close_sim, self._sim_open = self._sim_open, False
            self._replay_error = None
            replay = self._replay = _Replay(self, path, self._replay_speed)
        if close_sim:
            now = time.time() * 1000.0
            self._emit({"type": "resume", "source": "simulated", "wall_ms": now, "t": now / 1000.0})
        self._emit_status()
        replay.start()

    def stop_replay(self) -> None:
        with self._lock:
            replay = self._replay
        if replay is None:
            return
        replay.stop()
        if replay.is_alive() and replay is not threading.current_thread():
            replay.join(timeout=2.0)
        self._replay_done(replay, None)

    def _replay_done(self, replay: _Replay, error: str | None) -> None:
        with self._lock:
            if self._replay is not replay:
                return
            self._replay = None
            self._replay_error = error
        self._emit_status()
