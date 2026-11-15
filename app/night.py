"""Night mode: run the existing sleep scripts from the app (added 2026-11-15, Ayur: "after the flashcards it should
go to a night screen that starts the sleep functionality and tracks my sleep data").

Nothing here changes how sleep is detected or cued. It starts `sleep_cue.py` (the existing live cueing script, with
the "N2 or deeper" model trained on the 2026-11-14 nap when it exists) as a child process, reads its printed lines
for the status, stops it with Ctrl+C (SIGINT, which makes it save the recording), and in the morning runs
`stage_nap.py` on that recording for the hypnogram. Both scripts are a PROTOTYPE: live cueing has never been run
on a sleeper, and the model is not validated (RESULTS.md, CLAUDE.md).
"""
from __future__ import annotations

import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import config

TAP_THRESHOLD = 0.356           # double-tap snooze threshold calibrated for Ayur (RESULTS.md, snooze check)
NIGHT_MINUTES = 600             # sleep_cue.py stops by itself after this; "I'm awake" stops it sooner
STAGE_TIMEOUT_S = 900
_CUE_RE = re.compile(r"\bcue (\d+): ")
_SNOOZE_RE = re.compile(r"\bSNOOZE (\d+):")
_SAVED_RE = re.compile(r"stage_nap\.py (\S+\.npz)")
_PLOT_RE = re.compile(r"Plot saved: (\S+\.png)")
_IMAGE_RE = re.compile(r"^cuenap-[0-9-]+_stages\.png$")


class Night:
    """One night at a time. All methods are safe to call from any thread."""

    def __init__(self, root: Path | None = None, python: str | None = None):
        self.root = Path(root or config.ROOT)
        self.python = python or sys.executable
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._state = self._fresh()

    @staticmethod
    def _fresh() -> dict:
        return {"phase": "idle", "session_id": None, "started_ms": None, "ended_ms": None, "cues": 0,
                "snoozes": 0, "movement_holds": 0, "recording": None, "error": None, "lines": [],
                "volume": None, "model": None, "stage": {"status": "none", "image": None, "summary": [],
                                                       "error": None}}

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        with self._lock:
            s = {**self._state, "lines": list(self._state["lines"][-8:]), "stage": dict(self._state["stage"])}
        note = config.SLEEP_NOTE_MODEL if config.SLEEP_MODEL.is_file() else config.SLEEP_NOTE
        s["model_note"] = note.replace("This command uses", "Tonight uses")
        return s

    # ------------------------------------------------------------------ start / stop
    def start(self, session_id: str | None, cues_dir: Path | None, volume: float) -> dict:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                raise RuntimeError("A night is already running.")
            volume = min(0.5, max(0.01, float(volume)))
            cmd = [self.python, "-u", str(self.root / "sleep_cue.py"), "--volume", f"{volume:.2f}",
                   "--minutes", str(NIGHT_MINUTES), "--tap-threshold", str(TAP_THRESHOLD)]
            if config.SLEEP_MODEL.is_file():
                cmd += ["--model", str(config.SLEEP_MODEL)]
                model = config.SLEEP_MODEL.name
            else:
                raise RuntimeError("There is no trained sleep model yet, so the night can't start from the app.")
            if cues_dir is not None and any(Path(cues_dir).glob("*.wav")):
                cmd += ["--cues", str(cues_dir)]
            self._state = {**self._fresh(), "phase": "starting", "session_id": session_id,
                           "started_ms": round(time.time() * 1000), "volume": volume, "model": model}
            self._proc = subprocess.Popen(cmd, cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          text=True, bufsize=1)
            proc = self._proc
        threading.Thread(target=self._read, args=(proc,), daemon=True, name="night-reader").start()
        return self.status()

    def stop(self) -> dict:
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                return self.status()
            self._state["phase"] = "stopping"
        proc.send_signal(signal.SIGINT)        # sleep_cue.py saves the recording on Ctrl+C
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        return self.status()

    def _read(self, proc: subprocess.Popen) -> None:
        for raw in proc.stdout:
            line = raw.rstrip()
            if not line:
                continue
            with self._lock:
                st = self._state
                st["lines"] = (st["lines"] + [line])[-40:]
                if line.startswith("Detector:"):
                    st["phase"] = "running"
                if m := _CUE_RE.search(line):
                    st["cues"] = int(m.group(1))
                if m := _SNOOZE_RE.search(line):
                    st["snoozes"] = int(m.group(1))
                if "movement: cues held" in line:
                    st["movement_holds"] += 1
                if m := _SAVED_RE.search(line):
                    st["recording"] = m.group(1)
        code = proc.wait()
        with self._lock:
            st = self._state
            st["ended_ms"] = round(time.time() * 1000)
            if st["recording"]:
                st["phase"] = "done"
            else:
                st["phase"] = "failed"
                # the script's own last words say why (e.g. "No Muse_EEG stream found ...")
                st["error"] = " ".join(st["lines"][-3:]) or f"sleep_cue.py exited with code {code}"

    # ------------------------------------------------------------------ morning: sleep stages
    def stage(self) -> dict:
        with self._lock:
            rec = self._state.get("recording")
            if not rec or self._state["phase"] != "done":
                raise RuntimeError("There is no finished night recording to stage yet.")
            if self._state["stage"]["status"] == "running":
                return self.status()
            self._state["stage"] = {"status": "running", "image": None, "summary": [], "error": None}
        threading.Thread(target=self._stage, args=(rec,), daemon=True, name="night-stage").start()
        return self.status()

    def _stage(self, rec: str) -> None:
        try:
            out = subprocess.run([self.python, str(self.root / "stage_nap.py"), rec], cwd=self.root,
                                 capture_output=True, text=True, timeout=STAGE_TIMEOUT_S)
            text = (out.stdout or "") + (out.stderr or "")
            plot = _PLOT_RE.search(text)
            lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("Plot saved")][-12:]
            result = {"status": "done" if plot else "failed", "image": Path(plot.group(1)).name if plot else None,
                      "summary": lines, "error": None if plot else (lines[-1] if lines else "Staging failed.")}
        except subprocess.TimeoutExpired:
            result = {"status": "failed", "image": None, "summary": [], "error": "Staging took too long."}
        except Exception as e:  # noqa: BLE001
            result = {"status": "failed", "image": None, "summary": [], "error": f"Staging failed ({type(e).__name__})."}
        with self._lock:
            self._state["stage"] = result

    def image_path(self, name: str) -> Path | None:
        """A staging plot of a night recording in results/ (cuenap-<digits>_stages.png only), else None."""
        if not isinstance(name, str) or not _IMAGE_RE.fullmatch(name):
            return None
        p = config.RESULTS / name
        return p if p.is_file() else None
