"""Session files on disk: one JSON file per session, written atomically and guarded by a lock.

Layout under the data directory (default `app_data/`):
    sessions/<id>.json      the session object (APP_SPEC.md §7)
    settings.json           app settings chosen on Home ({"theta": 20}); absent until one is changed
    cache/                  transcript and LLM caches (owned by app/content.py and app/llm.py)
    cues/<id>/              that session's audio cues and manifest
    shots/<id>/             that session's lapse screenshots

Every id and file name is checked against a strict pattern before it is turned into a path, and file paths
are resolved and confirmed to stay inside the session's own folder.
"""
import json
import os
import re
import shutil
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path

ID_RE = re.compile(r"^[0-9a-f]{32}$")
CUE_FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,100}\.wav$")
SHOT_FILE_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}\.jpg$")
LAPSE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

SOURCES = ("live", "replay", "simulated")
EVENT_MOTION_KEYS = {"drift": ("angle", "rot"), "resume": ("peak_angle", "net_rot")}


def new_id() -> str:
    return uuid.uuid4().hex


def valid_id(value) -> bool:
    return isinstance(value, str) and bool(ID_RE.fullmatch(value))


def now_iso() -> str:
    """Local time to the second, e.g. 2026-11-15T01:50:00."""
    return datetime.now().replace(microsecond=0).isoformat()


class Store:
    def __init__(self, data_dir: Path):
        self.root = Path(data_dir)
        self.sessions = self.root / "sessions"
        self.cache = self.root / "cache"
        self.cues = self.root / "cues"
        self.shots = self.root / "shots"
        self.settings_path = self.root / "settings.json"
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ paths
    def session_path(self, sid: str) -> Path:
        if not valid_id(sid):
            raise ValueError("not a session id")
        return self.sessions / f"{sid}.json"

    def cue_dir(self, sid: str) -> Path:
        if not valid_id(sid):
            raise ValueError("not a session id")
        return self.cues / sid

    def shot_dir(self, sid: str) -> Path:
        if not valid_id(sid):
            raise ValueError("not a session id")
        return self.shots / sid

    @staticmethod
    def file_in(folder: Path, name, pattern: re.Pattern) -> Path | None:
        """The path of `name` inside `folder`, or None unless the name matches `pattern` and the resolved
        path really is a file directly inside that folder (so no traversal and no symlink escape)."""
        if not isinstance(name, str) or not pattern.fullmatch(name):
            return None
        base = folder.resolve()
        path = (base / name).resolve()
        if path.parent != base or not path.is_file():
            return None
        return path

    def cue_file(self, sid: str, name) -> Path | None:
        return self.file_in(self.cue_dir(sid), name, CUE_FILE_RE) if valid_id(sid) else None

    def shot_file(self, sid: str, name) -> Path | None:
        return self.file_in(self.shot_dir(sid), name, SHOT_FILE_RE) if valid_id(sid) else None

    # ------------------------------------------------------------------ reading and writing
    @staticmethod
    def write_atomic(path: Path, data: bytes) -> None:
        """Write to a temp file in the same folder, then rename over the target, so a reader (or a crash)
        never sees a half-written file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _write(self, session: dict) -> None:
        data = json.dumps(session, ensure_ascii=False, indent=1).encode("utf-8")
        self.write_atomic(self.session_path(session["id"]), data)

    def _read(self, sid: str) -> dict | None:
        try:
            with open(self.session_path(sid), encoding="utf-8") as f:
                session = json.load(f)
        except FileNotFoundError:
            return None
        if not isinstance(session, dict) or session.get("id") != sid:
            raise ValueError(f"session file {sid}.json is damaged")
        return session

    # ------------------------------------------------------------------ settings
    def read_settings(self) -> dict:
        """The saved app settings; {} when there is no settings file yet or it cannot be read as an object.
        The caller checks each value before using it."""
        with self._lock:
            try:
                with open(self.settings_path, encoding="utf-8") as f:
                    settings = json.load(f)
            except (OSError, ValueError):
                return {}
        return settings if isinstance(settings, dict) else {}

    def write_settings(self, settings: dict) -> None:
        with self._lock:
            self.write_atomic(self.settings_path, json.dumps(settings, indent=1).encode("utf-8"))

    # ------------------------------------------------------------------ sessions
    def create(self, mode: str, subject: str, title: str, ai_mode: str, content: dict,
               key_ideas_status: str = "pending") -> dict:
        session = {
            "id": new_id(), "created_at": now_iso(), "mode": mode, "subject": subject, "title": title,
            "ai_mode": ai_mode,
            "content": content,
            "key_ideas": [], "key_ideas_status": key_ideas_status, "key_ideas_error": None,
            "log": None, "server_events": [], "summary": None, "flashcards": None, "cues": None,
            "simulated": False, "source_counts": {s: 0 for s in SOURCES}, "ended_at": None,
            # Which event source the server saw while the session ran; `measured` and `measure_note` are
            # decided from it when the session is finished (None until then).
            "source_log": [], "measured": None, "measure_note": None,
        }
        with self._lock:
            self._write(session)
        return session

    def load(self, sid: str) -> dict | None:
        """The session, or None if the id is malformed or no such session exists."""
        if not valid_id(sid):
            return None
        with self._lock:
            return self._read(sid)

    def update(self, sid: str, change) -> dict | None:
        """Read, apply `change(session)` (which edits it in place) and write back, all under the lock, so
        the request handlers and the background threads never lose each other's edits. Returns the new
        session, or None if it no longer exists."""
        if not valid_id(sid):
            return None
        with self._lock:
            session = self._read(sid)
            if session is None:
                return None
            change(session)
            self._write(session)
            return session

    def list(self) -> list[dict]:
        """All readable sessions, newest first. Damaged files are skipped, not deleted."""
        out = []
        with self._lock:
            if not self.sessions.is_dir():
                return out
            for path in self.sessions.iterdir():
                if path.suffix != ".json" or not valid_id(path.stem):
                    continue
                try:
                    session = self._read(path.stem)
                except (ValueError, OSError):
                    continue
                if session is not None:
                    out.append(session)
        out.sort(key=lambda s: str(s.get("created_at") or ""), reverse=True)
        return out

    def delete(self, sid: str) -> bool:
        """Delete this session's file, cues and screenshots, and nothing else."""
        if not valid_id(sid):
            return False
        with self._lock:
            path = self.session_path(sid)
            if not path.is_file():
                return False
            path.unlink()
            for folder in (self.cue_dir(sid), self.shot_dir(sid)):
                if folder.is_dir() and not folder.is_symlink():
                    shutil.rmtree(folder)
        return True

    def append_event(self, sid: str, event: dict) -> dict | None:
        """Add a drift/resume the server broadcast to the session's own event log. This log, not anything
        the client sends, is what marks a session as simulated. Finished sessions are left untouched."""
        entry = {
            "type": event.get("type"), "kind": event.get("kind"), "source": event.get("source"),
            "wall_ms": event.get("wall_ms"), "onset_wall_ms": event.get("onset_wall_ms"),
        }
        # What the head did (degrees), kept when the event says so, for tuning the trigger later.
        for key in EVENT_MOTION_KEYS.get(entry["type"], ()):
            if event.get(key) is not None:
                entry[key] = event[key]
        applied = []

        def change(session):
            if session.get("ended_at"):
                return
            session.setdefault("server_events", []).append(entry)
            if entry["source"] != "live":
                session["simulated"] = True
            if entry["type"] == "drift" and entry["source"] in SOURCES:
                counts = session.setdefault("source_counts", {s: 0 for s in SOURCES})
                counts[entry["source"]] = int(counts.get(entry["source"], 0)) + 1
            applied.append(True)

        session = self.update(sid, change)
        return session if applied else None

    def append_source(self, sid: str, source, wall_ms, only_if_changed: bool = False) -> dict | None:
        """Add the event source the server saw (live, replay or simulated) to the session's `source_log`.
        With only_if_changed, nothing is added when it equals the last entry. Like the event log, this is
        the server's own record: a replay entry marks the session simulated, and finished sessions are
        left untouched. Returns the session when an entry was added, otherwise None."""
        if source not in SOURCES:
            return None
        entry = {"source": source, "wall_ms": wall_ms}
        applied = []

        def change(session):
            if session.get("ended_at"):
                return
            log = session.setdefault("source_log", [])
            if only_if_changed and log and isinstance(log[-1], dict) and log[-1].get("source") == source:
                return
            log.append(entry)
            if source == "replay":
                session["simulated"] = True
            applied.append(True)

        session = self.update(sid, change)
        return session if applied else None
