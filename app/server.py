"""Local server for the SynapStudy app (APP_SPEC.md §6-§8, §13).

    .venv/bin/python -m app.server        # http://127.0.0.1:8770

Serves the static frontend, the JSON API under /api/ and one WebSocket at /ws. Honesty rules enforced here:
  - every drift/resume the server broadcasts while a session is active is written to that session's own
    event log, and that log decides whether the session is `simulated` (the client cannot clear it);
  - the event source in effect while a session runs (live, replay, simulated) is recorded by the server,
    and a session during which the headband was not live throughout is stored as not `measured`;
  - simulated and unmeasured sessions are left out of long-term insights unless asked for;
  - no sample sessions, lapses or statistics are ever created.
"""
import asyncio
import base64
import importlib
import json
import os
import re
import shlex
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.staticfiles import StaticFiles

from app import config, metrics
from app.detector_bridge import DetectorBridge, ReplayError
from app.store import CUE_FILE_RE, LAPSE_ID_RE, SOURCES, Store, new_id, now_iso, valid_id

SAMPLE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
NO_STORE = {"Cache-Control": "no-store"}
LOCAL_HOSTS = {"127.0.0.1", "localhost", "testserver"}   # "testserver" is Starlette's TestClient
ASK_NEEDS_KEY = "Questions need an Anthropic API key. Add ANTHROPIC_API_KEY to .env and restart."
FLASHCARD_IDEAS_WAIT_S = 30.0   # longest POST /flashcards waits for the covered sections' key ideas
NOTHING_COVERED = "Nothing was covered in this session, so there are no flashcards to make."
CONCEPT_WINDOW_S = 20.0       # caption window per concept in the clench popup
END_METHODS = ("movement_end", "turn_back", "manual", "simulated", "disconnect", "session_end", "clench")


class ApiError(Exception):
    """Turned into a JSON {"error": "<plain sentence>"} response."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


# ---------------------------------------------------------------------------------------------- helpers
def _pdf_clean(text: str) -> str:
    """PDF text as paragraphs: words split by an end-of-line hyphen are joined, single line breaks inside a
    paragraph become spaces, and blank lines (paragraph breaks) are kept."""
    text = text.replace("\r", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    out = []
    for block in re.split(r"\n\s*\n", text):
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        full = max(len(ln) for ln in lines)
        para = []
        for ln in lines:
            para.append(ln)
            # A paragraph (or heading) ends at a line clearly shorter than the block's full lines: one that ends a
            # sentence, or a short line with no closing punctuation (a title). PDFs often drop the blank lines.
            short = len(ln) < 0.8 * full
            if short and (ln.endswith((".", "!", "?", ":", '"', "\u201d")) or len(ln) < 60):
                out.append(" ".join(para))
                para = []
        if para:
            out.append(" ".join(para))
    return "\n\n".join(re.sub(r"\s+", " ", p).strip() for p in out if p.strip())


def _clean_str(value, limit: int) -> str:
    return " ".join(value.split())[:limit] if isinstance(value, str) else ""


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) != float("inf")


def _utf8(value):
    """JSON from a browser can carry lone surrogates, which cannot be stored or sent back as UTF-8.
    Replace them (with "?") everywhere in a parsed body."""
    if isinstance(value, str):
        try:
            value.encode("utf-8")
            return value
        except UnicodeEncodeError:
            return value.encode("utf-8", "replace").decode("utf-8")
    if isinstance(value, list):
        return [_utf8(v) for v in value]
    if isinstance(value, dict):
        return {_utf8(k): _utf8(v) for k, v in value.items()}
    return value


def _hide_secrets(text: str) -> str:
    """Belt and braces: the API key must never reach a response, whatever another module puts in a string."""
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        secret = os.environ.get(name) or ""
        if len(secret) >= 8 and secret in text:
            text = text.replace(secret, "[hidden]")
    return text


def _host_only(value: str) -> str:
    value = value.strip().lower()
    if value.startswith("["):
        return value[1:value.find("]")] if "]" in value else ""
    return value.rsplit(":", 1)[0] if ":" in value else value


class LocalOnly:
    """Refuses requests that do not come from this app's own window: the Host must be this machine, and a
    request that carries an Origin (other pages' scripts, any WebSocket) must come from this machine too.
    Without it any website open in a browser could talk to the local server."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            headers = dict(scope.get("headers") or [])
            host = _host_only(headers.get(b"host", b"").decode("latin-1"))
            origin = headers.get(b"origin")
            ok = host in LOCAL_HOSTS
            if ok and origin is not None:
                try:
                    ok = (urlsplit(origin.decode("latin-1")).hostname or "") in LOCAL_HOSTS
                except ValueError:
                    ok = False
            if not ok:
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                else:
                    response = JSONResponse({"error": "This app only accepts requests from its own window."},
                                            status_code=403)
                    await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


class NoStoreStatic(StaticFiles):
    """Static files that are never cached, so an edited frontend file is picked up on reload."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        return response


class Deps:
    """The modules other parts of the app provide (app.llm, app.content, app.cues), imported on first use
    so the server starts without them. Tests pass replacements: create_app(deps={"llm": fake, ...}).
    Other keys: "web_dir" (static root), "bridge_factory" (callable(on_message) -> bridge)."""

    MODULES = {"llm": "app.llm", "content": "app.content", "cues": "app.cues", "night": "app.night"}

    def __init__(self, overrides=None):
        self._overrides = dict(overrides or {})

    def get(self, name, default=None):
        return self._overrides.get(name, default)

    def module(self, name: str):
        if name in self._overrides:
            return self._overrides[name]
        try:
            return importlib.import_module(self.MODULES[name])
        except Exception as e:  # noqa: BLE001
            raise ApiError(503, f"The {name} part of the app could not be loaded ({type(e).__name__}: {e}).")


# ---------------------------------------------------------------------------------------------- WebSocket hub
class _Client:
    def __init__(self, hub: "Hub", ws: WebSocket, loop: asyncio.AbstractEventLoop):
        self.hub, self.ws, self.loop = hub, ws, loop
        self.lock = asyncio.Lock()      # one frame at a time per socket

    async def send_text(self, text: str) -> bool:
        try:
            async with self.lock:
                await asyncio.wait_for(self.ws.send_text(text), timeout=5.0)
            return True
        except Exception:  # noqa: BLE001 - the client went away (or stalled) mid-send: drop it
            self.hub.remove(self)
            return False


class Hub:
    """Connected WebSocket clients. publish() may be called from any thread; each send is scheduled on
    the client's own event loop, and a client that fails mid-broadcast is dropped without affecting the rest."""

    def __init__(self):
        self._clients: list[_Client] = []
        self._lock = threading.Lock()

    def add(self, ws: WebSocket) -> _Client:
        client = _Client(self, ws, asyncio.get_running_loop())
        with self._lock:
            self._clients.append(client)
        return client

    def remove(self, client: _Client) -> None:
        with self._lock:
            if client in self._clients:
                self._clients.remove(client)

    def count(self) -> int:
        with self._lock:
            return len(self._clients)

    @staticmethod
    def encode(message: dict) -> str | None:
        try:
            return json.dumps(message, allow_nan=False)
        except (TypeError, ValueError):
            print(f"[server] dropped a message that is not valid JSON: type={message.get('type')!r}",
                  file=sys.stderr)
            return None

    def publish(self, message: dict, skip: _Client | None = None) -> None:
        text = self.encode(message)
        if text is None:
            return
        with self._lock:
            clients = [c for c in self._clients if c is not skip]
        for client in clients:
            coro = client.send_text(text)
            try:
                asyncio.run_coroutine_threadsafe(coro, client.loop)
            except RuntimeError:        # that client's loop has closed
                coro.close()
                self.remove(client)


# ---------------------------------------------------------------------------------------------- app logic
class Core:
    def __init__(self, store: Store, deps: Deps):
        self.store, self.deps = store, deps
        self.hub = Hub()
        self._lock = threading.RLock()
        self.night = deps.module("night").Night()
        self._active: str | None = None          # session that broadcast events are logged against
        self._workers: dict[str, threading.Thread] = {}
        self._flash_locks: dict[str, threading.Lock] = {}    # one per session, handed out under _lock
        self._last_status: dict | None = None
        self._tts: bool | None = None
        self._frozen_params = self._load_frozen_params()
        self._theta = self._saved_theta()
        factory = deps.get("bridge_factory") or DetectorBridge
        self.bridge = factory(self.on_bridge_message)
        if self._frozen_params is not None and hasattr(self.bridge, "set_params"):
            self.bridge.set_params(self.detector_params())

    # ------------------------------------------------------------------ status
    @staticmethod
    def _load_frozen_params():
        try:
            with open(config.FROZEN_PARAMS, encoding="utf-8") as f:
                return json.load(f)["variants"][config.DETECTOR_VARIANT]["params"]
        except (OSError, ValueError, KeyError, TypeError):
            return None

    @staticmethod
    def _valid_theta(value) -> int | None:
        """The head-turn setting as one of config.HEAD_TURN_CHOICES, or None if it is not one of them."""
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value not in config.HEAD_TURN_CHOICES:
            return None
        return int(value)

    def _saved_theta(self) -> int:
        """The head-turn setting saved in <data>/settings.json, or the default if none (or a bad one) is."""
        theta = self._valid_theta(self.store.read_settings().get("theta"))
        return config.HEAD_TURN_DEFAULT if theta is None else theta

    def detector_params(self) -> dict | None:
        """The thresholds in effect: the frozen ones for this variant with theta replaced by the head-turn
        setting (N is left as frozen). None if the frozen parameters could not be read."""
        with self._lock:
            return None if self._frozen_params is None else {**self._frozen_params, "theta": self._theta}

    def set_theta(self, value) -> dict:
        """Change the head-turn setting: applied to the detector at once, saved for the next start, and
        sent to every window. Returns the new status body."""
        theta = self._valid_theta(value)
        if theta is None:
            choices = ", ".join(str(c) for c in config.HEAD_TURN_CHOICES[:-1])
            raise ApiError(400, f"The head-turn setting must be {choices} or {config.HEAD_TURN_CHOICES[-1]} degrees.")
        with self._lock:
            if self._frozen_params is None:
                raise ApiError(503, "The detector's frozen settings could not be read, so this cannot be changed.")
            if hasattr(self.bridge, "set_params"):
                try:
                    self.bridge.set_params({**self._frozen_params, "theta": theta})
                except Exception as e:  # noqa: BLE001 - refused: nothing is saved, and the old setting stays
                    raise ApiError(500, f"The detector did not accept that setting ({type(e).__name__}).")
            self._theta = theta
            self.store.write_settings({**self.store.read_settings(), "theta": theta})
        return self.broadcast_status(force=True)

    def ai_status(self) -> dict:
        try:
            st = self.deps.module("llm").status()
            mode = "claude" if st.get("mode") == "claude" else "extractive"
            return {"available": bool(st.get("available")), "model": str(st.get("model") or config.MODEL),
                    "mode": mode, "detail": _hide_secrets(str(st.get("detail") or ""))}
        except Exception as e:  # noqa: BLE001
            detail = e.message if isinstance(e, ApiError) else f"The AI module failed ({type(e).__name__})."
            return {"available": False, "model": config.MODEL, "mode": "extractive", "detail": _hide_secrets(detail)}

    def tts_available(self) -> bool:
        if self._tts is None:
            try:
                self._tts = bool(self.deps.module("cues").tts_available())
            except Exception:  # noqa: BLE001
                return False
        return self._tts

    def status_body(self) -> dict:
        body = dict(self.bridge.status())
        with self._lock:
            theta = self._theta
        body["detector"] = {"variant": config.DETECTOR_VARIANT, "params": self.detector_params(),
                            "acted_kinds": list(config.ACTED_KINDS), "theta": theta,
                            "choices": list(config.HEAD_TURN_CHOICES), "tested_theta": config.HEAD_TURN_TESTED,
                            "evidence": config.head_turn_evidence(theta)}
        body["ai"] = self.ai_status()
        body["audio"] = {"muse_speaker": False, "detail": config.AUDIO_DETAIL}
        body["tts"] = {"available": self.tts_available(), "engine": "say"}
        body["clients"] = self.hub.count()      # open WebSocket connections (app windows)
        body["build"] = config.BUILD            # the code this process is running (see config.code_stamp)
        return body

    def broadcast_status(self, force: bool = False, skip: _Client | None = None) -> dict:
        """Send the status to every client (except `skip`) if it changed, or always with force.
        Returns the status body either way."""
        body = self.status_body()
        with self._lock:
            if not force and body == self._last_status:
                return body
            self._last_status = body
        self.hub.publish({"type": "status", **body}, skip=skip)
        return body

    # ------------------------------------------------------------------ events
    def on_bridge_message(self, message: dict) -> None:
        """Called by the detector bridge from its threads. Drift/resume are written to the active
        session's event log BEFORE they are broadcast, so the server's record never lags the client's."""
        kind = message.get("type")
        if kind == "status":
            with self._lock:
                sid = self._active
            if sid:
                source = message.get("source")
                self._record_source(sid, source if source in SOURCES else None, only_if_changed=True)
            self.broadcast_status(force=True)
            return
        if kind in ("drift", "resume", "clench"):     # clench_progress is broadcast only (10 Hz, ring display)
            with self._lock:
                sid = self._active
            if sid:
                try:
                    self.store.append_event(sid, message)
                except Exception:  # noqa: BLE001
                    traceback.print_exc()
        self.hub.publish(message)

    def _record_source(self, sid: str, source=None, only_if_changed: bool = False) -> None:
        """Note in the session's source_log which event source is in effect now (the bridge's, unless
        the caller already has it from a status message). This log decides `measured` at finish."""
        try:
            if source is None:
                source = self.bridge.status()["source"]
            self.store.append_source(sid, source, round(time.time() * 1000), only_if_changed)
        except Exception:  # noqa: BLE001
            traceback.print_exc()

    def handle_client_message(self, message) -> dict | None:
        """One WebSocket message from the browser. Returns a reply for that client only, or None."""
        if not isinstance(message, dict):
            return None
        kind = message.get("type")
        if kind == "session":
            sid, state = message.get("id"), message.get("state")
            if not valid_id(sid):
                return {"type": "error", "error": "That is not a valid session id."}
            if state == "start":
                session = self.store.load(sid)
                if session is None:
                    return {"type": "error", "error": "No such session."}
                if not session.get("ended_at"):
                    with self._lock:
                        self._active = sid
                    self._record_source(sid)
            elif state == "end":
                with self._lock:
                    if self._active == sid:
                        self._active = None
        elif kind == "sim":
            if message.get("action") in ("drift", "resume"):
                self.bridge.sim(message["action"])     # ignored by the bridge unless the source is simulated
            elif message.get("action") == "clench":
                self.bridge.sim_clench()               # likewise; a 3-s simulated hold
        elif kind == "replay":
            try:
                if message.get("action") == "start":
                    self.bridge.start_replay(message.get("file"))
                elif message.get("action") == "stop":
                    self.bridge.stop_replay()
            except ReplayError as e:
                return {"type": "error", "error": str(e)}
        return None

    # ------------------------------------------------------------------ sessions
    def need(self, sid: str) -> dict:
        if not valid_id(sid):
            raise ApiError(400, "That is not a valid session id.")
        session = self.store.load(sid)
        if session is None:
            raise ApiError(404, "No such session.")
        return session

    def create_session(self, body: dict) -> dict:
        mode = body.get("mode")
        if mode not in ("reading", "video"):
            raise ApiError(400, 'The mode must be "reading" or "video".')
        subject = _clean_str(body.get("subject"), 80) or "General"
        title = _clean_str(body.get("title"), 200) or ("Untitled reading" if mode == "reading" else "Untitled video")
        content = self.deps.module("content")

        if mode == "reading":
            text = body.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ApiError(400, "There is no text to read. Paste or choose some text first.")
            if len(text) > config.TEXT_MAX_CHARS:
                raise ApiError(400, f"The text is too long: the limit is {config.TEXT_MAX_CHARS:,} characters.")
            sections = list(content.split_sections(text, config.SECTION_TARGET_WORDS, config.SECTION_MAX_WORDS))
            if not sections:
                raise ApiError(400, "There is no readable text in what was sent.")
            block = {"sections": sections}
        else:
            url = body.get("youtube_url")
            video_id = content.parse_youtube_id(url.strip()) if isinstance(url, str) and len(url) <= 500 else None
            if not isinstance(video_id, str) or not VIDEO_ID_RE.fullmatch(video_id):
                raise ApiError(400, "That does not look like a YouTube link.")
            transcript = content.fetch_transcript(video_id, self.store.cache)
            sections, concepts = [], {"status": "done", "source": "extractive", "items": []}
            if transcript.get("available") and transcript.get("snippets"):
                sections = list(content.transcript_sections(transcript["snippets"]))
                # Concept timeline for the clench popup: ~20-s caption windows with short labels. Extractive
                # labels now (instant); Claude labels replace them from a background thread when a key works.
                try:
                    windows = list(content.transcript_sections(transcript["snippets"], target_s=CONCEPT_WINDOW_S))
                    concepts = dict(self.deps.module("llm").concept_labels(windows, subject, title,
                                                                           self.store.cache, use_claude=False))
                    concepts["status"] = "pending" if self.ai_status()["available"] else "done"
                except Exception:  # noqa: BLE001 - concepts only serve the clench popup; never block a session
                    traceback.print_exc()
                    concepts = {"status": "done", "source": "extractive", "items": []}
            block = {"youtube_url": url.strip(), "video_id": video_id, "concepts": concepts,
                     "transcript_available": bool(sections),
                     "transcript_error": None if sections else str(
                         transcript.get("error") or "No captions were found for this video."),
                     "sections": sections}

        session = self.store.create(mode, subject, title, self.ai_status()["mode"], block,
                                    key_ideas_status="pending" if sections else "done")
        self.ensure_key_ideas(session)
        if mode == "video" and (session["content"].get("concepts") or {}).get("status") == "pending":
            threading.Thread(target=self._concepts_worker, args=(session["id"],), daemon=True,
                             name=f"concepts-{session['id'][:8]}").start()
        return session

    def _concepts_worker(self, sid: str) -> None:
        """Replace the extractive concept labels with Claude's (one call per video), then publish them."""
        try:
            session = self.store.load(sid)
            if session is None:
                return
            c = session["content"]
            transcript = self.deps.module("content").fetch_transcript(c["video_id"], self.store.cache)
            windows = list(self.deps.module("content").transcript_sections(transcript.get("snippets") or [],
                                                                           target_s=CONCEPT_WINDOW_S))
            result = dict(self.deps.module("llm").concept_labels(windows, session["subject"], session["title"],
                                                                 self.store.cache))
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            result = None

        def change(s):
            cur = s["content"].get("concepts") or {}
            if result is not None and result.get("items"):
                cur = {**result}
            cur["status"] = "done"
            s["content"]["concepts"] = cur
        updated = self.store.update(sid, change)
        if updated is not None:
            self.hub.publish({"type": "concepts", "session_id": sid, "concepts": updated["content"]["concepts"]})

    def get_session(self, sid: str) -> dict:
        session = self.need(sid)
        self.ensure_key_ideas(session)      # picks generation back up after a server restart
        return session

    def delete_session(self, sid: str) -> dict:
        self.need(sid)
        with self._lock:
            if self._active == sid:
                self._active = None
            self._flash_locks.pop(sid, None)
        self.store.delete(sid)
        return {"ok": True}

    # ------------------------------------------------------------------ key ideas (background thread)
    def ensure_key_ideas(self, session: dict) -> None:
        if session.get("key_ideas_status") not in ("pending", "partial"):
            return
        sid = session["id"]
        with self._lock:
            worker = self._workers.get(sid)
            if worker is not None and worker.is_alive():
                return
            worker = threading.Thread(target=self._key_ideas_worker, args=(sid,), daemon=True,
                                      name=f"key-ideas-{sid[:8]}")
            self._workers[sid] = worker
            worker.start()      # inside the lock, so a second caller cannot see it registered but not alive

    def wait_covered_ideas(self, sid: str, timeout: float | None = None) -> None:
        """Wait until the sections covered in the session have their key ideas (or generation is done,
        failed or not running), for at most `timeout` seconds (FLASHCARD_IDEAS_WAIT_S by default). No lock
        is held while waiting. Sections are written in order, so a covered section with no idea of its own
        counts as written once a later section has one. If the time runs out the caller carries on with
        the ideas there are."""
        deadline = time.monotonic() + (FLASHCARD_IDEAS_WAIT_S if timeout is None else timeout)
        while True:
            session = self.store.load(sid)
            if session is None or session.get("key_ideas_status") in ("done", "failed"):
                return
            have = {k.get("section_idx") for k in session.get("key_ideas", []) if _is_num(k.get("section_idx"))}
            reached = max(have, default=-1)
            if all(i in have or i < reached for i in self._covered_idx(session)):
                return
            with self._lock:
                worker = self._workers.get(sid)
            left = deadline - time.monotonic()
            if worker is None or worker is threading.current_thread() or not worker.is_alive() or left <= 0:
                return
            worker.join(min(0.25, left))

    def _key_ideas_worker(self, sid: str) -> None:
        try:
            self._generate_key_ideas(sid)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        finally:
            with self._lock:
                if self._workers.get(sid) is threading.current_thread():
                    del self._workers[sid]
            try:
                self.broadcast_status()     # the AI mode may have changed (for example a rejected key)
            except Exception:  # noqa: BLE001
                traceback.print_exc()

    def _generate_key_ideas(self, sid: str) -> None:
        session = self.store.load(sid)
        if session is None:
            return
        sections = [s for s in session["content"].get("sections", []) if isinstance(s, dict)]
        have = {k.get("section_idx") for k in session.get("key_ideas", [])}
        todo = [s for s in sections if s.get("idx") not in have]

        def finish(section_idx, ideas, status, error):
            def change(s):
                s["key_ideas"].extend(ideas)
                s["key_ideas_status"], s["key_ideas_error"] = status, error
            if self.store.update(sid, change) is None:
                return False            # the session was deleted meanwhile
            self.hub.publish({"type": "key_ideas", "session_id": sid, "section_idx": section_idx,
                              "ideas": ideas, "status": status, "error": error})
            return True

        if not todo:
            finish(sections[-1]["idx"] if sections else 0, [], "done", None)
            return
        for n, section in enumerate(todo):
            idx = section.get("idx")
            # Once the session has ended, sections the reader never reached are not written up: nothing
            # shows them, and with a key each one is a paid call (added after the review, 2026-11-15).
            current = self.store.load(sid)
            if current is None:
                return
            if current.get("ended_at") and idx not in self._covered_idx(current):
                if n == len(todo) - 1:
                    finish(idx, [], "done", None)
                continue
            try:
                result = self.deps.module("llm").key_ideas(section["text"], session["subject"],
                                                           session["title"], self.store.cache)
                ideas = [{"id": new_id(), "section_idx": idx, "text": str(i["text"]),
                          "keyword": str(i["keyword"]), "source": str(result["source"])}
                         for i in result["ideas"]]
            except Exception as e:  # noqa: BLE001
                reason = e.message if isinstance(e, ApiError) else type(e).__name__
                where = f"section {idx + 1}" if _is_num(idx) else "a section"
                finish(idx, [], "failed", f"Key ideas could not be written for {where} ({reason}).")
                return
            if not finish(idx, ideas, "done" if n == len(todo) - 1 else "partial", None):
                return

    # ------------------------------------------------------------------ client log, finish
    @staticmethod
    def clean_log(body, need_start: bool) -> dict:
        if not isinstance(body, dict):
            raise ApiError(400, "The session log must be a JSON object.")
        log = dict(body)
        for key in ("lapses", "progress"):
            value = body.get(key)
            log[key] = [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []
        log["gaze"] = body["gaze"] if isinstance(body.get("gaze"), dict) else {"mode": "none", "calibrated": False}
        log["video"] = body["video"] if isinstance(body.get("video"), dict) else {"duration_s": None, "watched_s": 0.0}
        for key in ("started_ms", "ended_ms"):
            log[key] = body.get(key) if _is_num(body.get(key)) else None
        if need_start and log["started_ms"] is None:
            raise ApiError(400, "The session log has no start time.")
        try:
            size = len(json.dumps(log, allow_nan=False))
        except (TypeError, ValueError):
            raise ApiError(400, "The session log contains values that are not valid JSON.")
        if size > config.LOG_MAX_BYTES:
            raise ApiError(413, "The session log is too large.")
        return log

    def save_log(self, sid: str, body) -> dict:
        session = self.need(sid)
        if session.get("ended_at"):
            raise ApiError(409, "This session is already finished.")
        log = self.clean_log(body, need_start=False)

        def change(s):
            if not s.get("ended_at"):
                s["log"] = log
        self.store.update(sid, change)
        return {"ok": True}

    def finish(self, sid: str, body) -> dict:
        session = self.need(sid)
        if session.get("ended_at"):
            return session              # a finished session is not rewritten
        log = self.clean_log(body, need_start=True)
        if log["ended_ms"] is None:
            log["ended_ms"] = round(time.time() * 1000)
        if log["ended_ms"] < log["started_ms"]:
            raise ApiError(400, "The session log ends before it starts.")

        # Stop logging events against this session first (the summaries below can take a while), and note
        # the event source in effect at this moment: it is the last entry of the session's source_log.
        with self._lock:
            was_active = self._active == sid
            if was_active:
                self._active = None
        last_source = {"source": self.bridge.status()["source"], "wall_ms": round(time.time() * 1000)}
        try:
            numbers = metrics.session_metrics(log, session["content"].get("sections", []), session["mode"])
            missed = self._missed(session, log)
        except Exception:
            with self._lock:        # not finished after all: keep logging against it
                if was_active and self._active is None:
                    self._active = sid
            raise

        def change(s):
            if s.get("ended_at"):
                return
            source_log = s.setdefault("source_log", [])
            source_log.append(last_source)
            events = s.get("server_events")
            replayed = "replay" in metrics.log_sources(source_log)
            simulated = metrics.is_simulated(events, log["lapses"]) or replayed
            measured, note = metrics.measurement(
                source_log, simulated, len(log["lapses"]),
                replay_seen=any(x.get("source") == "replay" for x in [*(events or []), *log["lapses"]]
                                if isinstance(x, dict)))
            summary = {**numbers, "missed": missed, "simulated": simulated, "measured": measured,
                       "generated_at": now_iso()}
            if not metrics.ever_live(source_log) and not log["lapses"]:
                # no headband at any point and nothing logged: nothing was measured, so no 100 % focus
                summary["focus_pct"] = summary["lapses_per_min"] = None
            s["log"] = log
            s["simulated"] = simulated
            s["measured"], s["measure_note"] = measured, note
            s["summary"] = summary
            s["ended_at"] = now_iso()

        updated = self.store.update(sid, change)
        if updated is None:
            raise ApiError(404, "No such session.")
        self.broadcast_status()
        return updated

    def _missed(self, session: dict, log: dict) -> list[dict]:
        """One entry per lapse in the client log: what the reader or viewer missed."""
        sid, video = session["id"], session["mode"] == "video"
        started = log["started_ms"]
        shot_re = re.compile(rf"^/api/shots/{sid}/[A-Za-z0-9_-]{{1,64}}\.jpg$")
        out = []
        for lapse in log["lapses"]:
            position = lapse.get("position") if isinstance(lapse.get("position"), dict) else None
            source = lapse.get("source") if lapse.get("source") in SOURCES else "simulated"
            shot = lapse.get("screenshot_url")
            duration = metrics.lapse_duration_s(lapse, log["ended_ms"])
            entry = {
                "lapse_id": lapse.get("id") if isinstance(lapse.get("id"), str) else None,
                "kind": str(lapse.get("kind") or "unknown"), "source": source, "label": "",
                "duration_s": None if duration is None else round(duration, 1),
                "end_method": lapse.get("end_method") if lapse.get("end_method") in END_METHODS else None,
                "excerpt": None, "summary": None, "summary_source": "none",
                "position": position,
                "screenshot_url": shot if isinstance(shot, str) and shot_re.fullmatch(shot) else None,
            }
            chosen = _clean_str(lapse.get("chosen_concept"), 120)
            if chosen:      # clench lapses (video): the concept the viewer picked as the last thing they remembered
                entry["chosen_concept"] = chosen
            if video:
                span = self._video_span(position)
                if span:
                    entry["label"] = f"{metrics.fmt_clock(span[0])}-{metrics.fmt_clock(span[1])}"
                entry["_span"] = span
            else:
                clock = metrics.fmt_clock((lapse["onset_ms"] - started) / 1000.0) if _is_num(lapse.get("onset_ms")) else ""
                pos = position or {}
                if _is_num(pos.get("page")):            # the browser builds the pages, so only it knows this
                    where = f"Page {int(pos['page'])}"
                elif _is_num(pos.get("section_idx")):
                    where = f"Section {int(pos['section_idx']) + 1}"
                else:
                    where = ""
                entry["label"] = " · ".join(x for x in (where, clock) if x)
                sentence = pos.get("sentence")
                if isinstance(sentence, str) and sentence.strip():
                    entry["excerpt"], entry["summary_source"] = sentence.strip()[:1000], "excerpt"
            out.append(entry)

        if video:
            self._fill_video_excerpts(session, out)
        return out

    @staticmethod
    def _video_span(position) -> tuple[float, float] | None:
        if not position or not _is_num(position.get("t_start")):
            return None
        a = float(position["t_start"])
        b = float(position["t_end"]) if _is_num(position.get("t_end")) else a
        return (max(0.0, min(a, b)), max(0.0, a, b))

    def _fill_video_excerpts(self, session: dict, missed: list[dict]) -> None:
        """Transcript text around each lapse, and a summary of it when the model is available."""
        block = session["content"]
        sections = block.get("sections") or []
        snippets, content = None, None
        if block.get("transcript_available"):
            try:
                content = self.deps.module("content")
                transcript = content.fetch_transcript(block["video_id"], self.store.cache)   # cached at creation
                if transcript.get("available"):
                    snippets = transcript.get("snippets")
            except Exception:  # noqa: BLE001
                traceback.print_exc()

        for entry in missed:
            span = entry.pop("_span", None)
            if not span or not block.get("transcript_available"):
                continue
            text = ""
            if snippets:
                try:
                    text = content.transcript_span(snippets, span[0], span[1], 5.0) or ""
                except Exception:  # noqa: BLE001
                    traceback.print_exc()
            else:   # transcript cache gone: fall back to the stored ~60 s sections that overlap the lapse
                text = " ".join(str(s.get("text", "")) for s in sections
                                if _is_num(s.get("start_s")) and _is_num(s.get("end_s"))
                                and s["start_s"] <= span[1] + 5.0 and s["end_s"] >= span[0] - 5.0)
            text = text.strip()
            if text:
                entry["excerpt"], entry["summary_source"] = text, "excerpt"

        excerpts = sorted({e["excerpt"] for e in missed if e["excerpt"]})
        if not excerpts:
            return
        try:
            llm = self.deps.module("llm")
        except ApiError:
            return

        def summarise(excerpt):
            try:
                return llm.summarize_missed(excerpt, self.store.cache)
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                return None

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = dict(zip(excerpts, pool.map(summarise, excerpts)))
        for entry in missed:
            result = results.get(entry["excerpt"])
            if isinstance(result, dict) and isinstance(result.get("summary"), str) and result["summary"].strip():
                entry["summary"] = result["summary"].strip()
                entry["summary_source"] = str(result.get("source") or "claude")

    # ------------------------------------------------------------------ flashcards and cues
    def _display_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(config.ROOT.resolve()))
        except ValueError:
            return str(path)

    def _cues_present(self, session: dict) -> bool:
        return all(isinstance(c.get("cue"), dict) and self.store.cue_file(session["id"], c["cue"].get("file"))
                   for c in session.get("flashcards") or [])

    @staticmethod
    def _covered_idx(session: dict) -> set:
        """Indexes of the sections actually covered in the session: sections the reader finished, plus
        any a lapse fell in. Empty when nothing was covered (the session was ended at once)."""
        sections = session["content"].get("sections", [])
        log = session.get("log") or {}
        covered = {p["section_idx"] for p in log.get("progress") or []
                   if isinstance(p, dict) and _is_num(p.get("section_idx"))}
        for lapse in log.get("lapses") or []:
            pos = lapse.get("position") if isinstance(lapse, dict) else None
            if not isinstance(pos, dict):
                continue
            if isinstance(pos.get("section_idx"), int):
                covered.add(pos["section_idx"])
            t0, t1 = pos.get("t_start"), pos.get("t_end")
            if _is_num(t0):
                t1 = t1 if _is_num(t1) else t0
                covered.update(sec["idx"] for sec in sections
                               if _is_num(sec.get("start_s")) and sec["end_s"] > t0 and sec["start_s"] <= t1)
        # the log comes from the browser: keep only indexes that really are sections of this session
        return {sec["idx"] for sec in sections
                if isinstance(sec, dict) and _is_num(sec.get("idx")) and sec["idx"] in covered}

    @classmethod
    def _covered(cls, session: dict) -> tuple[list, list]:
        """Key ideas and sections for the covered parts. Flashcards are made from these, never from
        material that was not reached."""
        covered = cls._covered_idx(session)
        return ([k for k in session["key_ideas"] if k.get("section_idx") in covered],
                [sec for sec in session["content"].get("sections", []) if sec.get("idx") in covered])

    def flashcards(self, sid: str, regenerate: bool = False) -> dict:
        session = self.need(sid)
        if not session.get("summary"):
            raise ApiError(409, "Finish the session before making flashcards.")
        if session.get("flashcards") is not None and not regenerate and self._cues_present(session):
            return session              # already made: no waiting and no lock
        group = self._study_sessions(session)
        if not any(self._covered_idx(s) for s in group):
            raise ApiError(409, NOTHING_COVERED)
        for s in group:
            self.ensure_key_ideas(s)
            self.wait_covered_ideas(s["id"])    # bounded, and outside any lock
        with self._lock:
            flash_lock = self._flash_locks.setdefault(sid, threading.Lock())
        with flash_lock:                # one maker per session; other sessions are not held up
            session = self.need(sid)
            if session.get("flashcards") is not None and not regenerate and self._cues_present(session):
                return session
            llm, cues = self.deps.module("llm"), self.deps.module("cues")
            group = [self.need(s["id"]) for s in self._study_sessions(session)]
            missed_excerpts = [m["excerpt"] for s in group for m in s["summary"].get("missed", []) if m.get("excerpt")]
            try:
                # A study plan's flashcards cover every resource studied: section indexes are offset per
                # resource so they stay distinct (cards from a plan carry no section index).
                ideas, sections = [], []
                for k, s in enumerate(group):
                    off = 10000 * k if len(group) > 1 else 0
                    i_s, sec_s = self._covered(s)
                    ideas += [{**i, "section_idx": i["section_idx"] + off} for i in i_s]
                    sections += [{**x, "idx": x["idx"] + off} for x in sec_s]
                title = session["title"] if len(group) == 1 else (session.get("study") or {}).get("subject") or session["subject"]
                n_max = 10 if len(group) == 1 else min(20, 8 + 4 * len(group))
                result = llm.make_flashcards(ideas, missed_excerpts, sections, session["subject"],
                                             title, self.store.cache, n_max=n_max)
                cards = [{"id": new_id(), "front": str(c["front"]), "back": str(c["back"]),
                          "concept": str(c["concept"]),
                          "section_idx": c.get("section_idx") if _is_num(c.get("section_idx")) and len(group) == 1 else None,
                          "from_missed": bool(c.get("from_missed")), "source": str(result["source"])}
                         for c in result["cards"]][:n_max]
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                raise ApiError(502, f"The flashcards could not be made ({type(e).__name__}).")

            out_dir = self.store.cue_dir(sid)
            out_dir.mkdir(parents=True, exist_ok=True)
            for old in out_dir.iterdir():       # cues from an earlier run of this same session
                if old.is_file() and (CUE_FILE_RE.fullmatch(old.name) or old.name == "manifest.json"):
                    old.unlink()
            try:
                made = list(cues.make_cues(cards, out_dir))
                if len(made) != len(cards):
                    raise ValueError("one cue per card expected")
                for card, cue in zip(cards, made):
                    if self.store.cue_file(sid, cue.get("file")) is None:
                        raise ValueError("cue file missing or badly named")
                    card["cue"] = {"file": cue["file"], "url": f"/api/cues/{sid}/{cue['file']}",
                                   "text": str(cue.get("text") or card["concept"]),
                                   "kind": "speech" if cue.get("kind") == "speech" else "tone"}
                manifest = cues.write_manifest({**session, "flashcards": cards}, made, out_dir)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                raise ApiError(500, f"The audio cues could not be made ({type(e).__name__}).")

            shown = self._display_path(out_dir)
            # Point the command at the N2-or-deeper model when it has been trained; otherwise leave a blank.
            if config.SLEEP_MODEL.is_file():
                model_arg, sleep_note = shlex.quote(self._display_path(config.SLEEP_MODEL)), config.SLEEP_NOTE_MODEL
            else:
                model_arg, sleep_note = "<your model from stage_nap.py>", config.SLEEP_NOTE
            cue_info = {
                "dir": shown, "count": len(cards), "manifest": self._display_path(Path(manifest)),
                "sleep_command": (f".venv/bin/python sleep_cue.py --model {model_arg} "
                                  f"--cues {shlex.quote(shown)} --volume 0.15"),
                "note": sleep_note,
            }

            def change(s):
                s["flashcards"], s["cues"] = cards, cue_info
            updated = self.store.update(sid, change)
        if updated is None:
            raise ApiError(404, "No such session.")
        self.broadcast_status()
        return updated

    # ------------------------------------------------------------------ night (sleep recording + quiet cues)
    def night_start(self, body: dict) -> dict:
        sid = body.get("session_id")
        cues_dir = None
        if valid_id(sid):
            session = self.store.load(sid)
            if session is not None and session.get("flashcards"):
                cues_dir = self.store.cue_dir(sid)
        volume = body.get("volume")
        try:
            return self.night.start(sid if valid_id(sid) else None, cues_dir, volume if _is_num(volume) else 0.15)
        except RuntimeError as e:
            raise ApiError(409, str(e))

    def night_stage(self) -> dict:
        try:
            return self.night.stage()
        except RuntimeError as e:
            raise ApiError(409, str(e))

    # ------------------------------------------------------------------ study plans (subject + several resources)
    def pdf_text(self, body: dict) -> dict:
        """Text of an uploaded PDF ({"name", "data_url": "data:application/pdf;base64,..."}), for a reading
        resource. Uses the PDF's own text layer (pypdf); a scanned PDF has none, and that is said plainly."""
        name = _clean_str(body.get("name"), 200) or "PDF"
        data_url = body.get("data_url")
        prefix = "base64,"
        if not isinstance(data_url, str) or prefix not in data_url[:80]:
            raise ApiError(400, "That is not a PDF upload.")
        try:
            raw = base64.b64decode(data_url.split(prefix, 1)[1], validate=True)
        except (ValueError, TypeError):
            raise ApiError(400, "That PDF upload is not valid base64.")
        if len(raw) > config.PDF_MAX_BYTES:
            raise ApiError(413, f"That PDF is too large (limit {config.PDF_MAX_BYTES // (1024 * 1024)} MB).")
        if not raw.startswith(b"%PDF"):
            raise ApiError(400, f"{name} is not a PDF file.")
        try:
            import io
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(raw))
            pages = [(pg.extract_text() or "") for pg in reader.pages]
        except Exception as e:  # noqa: BLE001
            raise ApiError(400, f"{name} could not be read ({type(e).__name__}).")
        text = _pdf_clean("\n\n".join(pages))
        title = re.sub(r"\.pdf$", "", name, flags=re.I)
        note = None
        if len(text) < 200:
            note = (f"{name} has almost no text that can be read (it may be a scanned image). "
                    "Only PDFs with a text layer work.")
        if len(text) > config.TEXT_MAX_CHARS:
            text = text[:config.TEXT_MAX_CHARS].rsplit(" ", 1)[0]
            note = f"{name} is long: only the first {config.TEXT_MAX_CHARS:,} characters are used."
        return {"title": title, "text": text, "pages": len(pages), "chars": len(text), "note": note}

    def create_study(self, body: dict) -> dict:
        """One study session = a subject and several resources (PDF text and/or YouTube videos), studied in
        order. Each resource is an ordinary reading or video session (unchanged); they are linked by a
        `study` block, and the flashcards at the end are made from all of them."""
        subject = _clean_str(body.get("subject"), 80)
        if not subject:
            raise ApiError(400, "Say what you want to study first.")
        resources = body.get("resources")
        if not isinstance(resources, list) or not resources:
            raise ApiError(400, "Add at least one PDF or YouTube link.")
        if len(resources) > config.STUDY_MAX_RESOURCES:
            raise ApiError(400, f"At most {config.STUDY_MAX_RESOURCES} resources per study session.")
        content = self.deps.module("content")
        bodies = []
        for i, r in enumerate(resources):
            if not isinstance(r, dict) or r.get("kind") not in ("reading", "video"):
                raise ApiError(400, f"Resource {i + 1} is not a PDF/text or a YouTube video.")
            if r["kind"] == "video":
                url = r.get("youtube_url")
                vid = content.parse_youtube_id(url.strip()) if isinstance(url, str) and len(url) <= 500 else None
                if not isinstance(vid, str) or not VIDEO_ID_RE.fullmatch(vid):
                    raise ApiError(400, f"Resource {i + 1} does not look like a YouTube link.")
                bodies.append({"mode": "video", "subject": subject, "youtube_url": url,
                               "title": _clean_str(r.get("title"), 200) or getattr(content, "video_title", lambda _v: None)(vid) or f"YouTube video ({vid})"})
            else:
                bodies.append({"mode": "reading", "subject": subject, "text": r.get("text"),
                               "title": _clean_str(r.get("title"), 200) or f"Reading {i + 1}"})
        study_id = new_id()
        made = [self.create_session(b) for b in bodies]
        ids = [m["id"] for m in made]
        out = []
        for i, m in enumerate(made):
            block = {"id": study_id, "subject": subject, "order": i, "count": len(ids), "session_ids": ids,
                     "titles": [x["title"] for x in made], "modes": [x["mode"] for x in made]}
            def change(s, block=block):
                s["study"] = block
            out.append(self.store.update(m["id"], change))
        return {"study_id": study_id, "subject": subject, "sessions": out}

    def _study_sessions(self, session: dict) -> list[dict]:
        """All finished sessions of this session's study plan (just this one when it has no plan)."""
        study = session.get("study")
        if not isinstance(study, dict) or not isinstance(study.get("session_ids"), list):
            return [session]
        out = []
        for sid in study["session_ids"]:
            s = session if sid == session["id"] else self.store.load(sid) if valid_id(sid) else None
            if s is not None and s.get("summary"):
                out.append(s)
        return out or [session]

    # ------------------------------------------------------------------ questions, screenshots, samples
    def ask(self, sid: str, body: dict) -> dict:
        session = self.need(sid)
        question = body.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ApiError(400, "Type a question first.")
        if len(question) > 2000:
            raise ApiError(400, "That question is too long (limit 2,000 characters).")
        if not self.ai_status()["available"]:
            raise ApiError(503, ASK_NEEDS_KEY)
        try:
            result = self.deps.module("llm").answer_question(question.strip(), session["content"].get("sections", []),
                                                             self.store.cache)
        except ApiError:
            raise
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            result = None
        self.broadcast_status()
        if not isinstance(result, dict) or not isinstance(result.get("answer"), str):
            if not self.ai_status()["available"]:
                raise ApiError(503, ASK_NEEDS_KEY)
            raise ApiError(503, "The question could not be answered just now: the AI call failed. Try again.")
        return {"answer": result["answer"], "source": str(result.get("source") or "claude")}

    def save_screenshot(self, sid: str, body: dict) -> dict:
        self.need(sid)
        lapse_id, data_url = body.get("lapse_id"), body.get("data_url")
        if not isinstance(lapse_id, str) or not LAPSE_ID_RE.fullmatch(lapse_id):
            raise ApiError(400, "That is not a valid lapse id.")
        prefix = "data:image/jpeg;base64,"
        if not isinstance(data_url, str) or not data_url.startswith(prefix):
            raise ApiError(400, "A screenshot must be a JPEG data URL.")
        encoded = data_url[len(prefix):]
        if len(encoded) > config.SCREENSHOT_MAX_BYTES * 4 // 3 + 8:
            raise ApiError(413, "That screenshot is larger than 2 MB.")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except ValueError:
            raise ApiError(400, "That screenshot is not valid base64.")
        if len(raw) > config.SCREENSHOT_MAX_BYTES:
            raise ApiError(413, "That screenshot is larger than 2 MB.")
        if not raw.startswith(b"\xff\xd8\xff"):
            raise ApiError(400, "That screenshot is not a JPEG image.")
        Store.write_atomic(self.store.shot_dir(sid) / f"{lapse_id}.jpg", raw)
        return {"url": f"/api/shots/{sid}/{lapse_id}.jpg"}

    def samples(self) -> list[dict]:
        return [{"id": s["id"], "title": s["title"], "words": s["words"]}
                for s in self.deps.module("content").list_samples()]

    def sample(self, sample_id: str) -> dict:
        if not SAMPLE_ID_RE.fullmatch(sample_id) or sample_id not in {s["id"] for s in self.samples()}:
            raise ApiError(404, "No such sample passage.")
        s = self.deps.module("content").load_sample(sample_id)
        return {"id": s["id"], "title": s["title"], "text": s["text"]}

    def shutdown(self) -> None:
        try:
            self.bridge.stop()
        except Exception:  # noqa: BLE001
            traceback.print_exc()


# ---------------------------------------------------------------------------------------------- the app
def create_app(data_dir: Path | None = None, start_detector: bool = True, deps: dict | None = None) -> FastAPI:
    """Build the server. `data_dir` defaults to app_data/. With start_detector=False nothing touches LSL
    (the source stays `simulated`). `deps` replaces collaborators for tests (see Deps)."""
    deps_ = Deps(deps)
    store = Store(Path(data_dir) if data_dir is not None else config.DATA)
    core = Core(store, deps_)
    web = Path(deps_.get("web_dir") or config.WEB)

    @asynccontextmanager
    async def lifespan(_app):
        if start_detector:
            await run_in_threadpool(core.bridge.start)
        yield
        await run_in_threadpool(core.shutdown)

    app = FastAPI(title="SynapStudy", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.core = core
    app.add_middleware(LocalOnly)

    # ------------------------------------------------------------ errors: always {"error": "<sentence>"}
    @app.exception_handler(ApiError)
    async def _api_error(_request, exc: ApiError):
        return JSONResponse({"error": exc.message}, status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def _bad_request(_request, _exc):
        return JSONResponse({"error": "The request was not understood: it needs a JSON object."}, status_code=400)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_request, exc: StarletteHTTPException):
        message = {404: "Not found.", 405: "That method is not allowed here."}.get(exc.status_code, str(exc.detail))
        return JSONResponse({"error": message}, status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def _server_error(_request, exc: Exception):
        traceback.print_exception(exc)
        return JSONResponse({"error": f"Something went wrong on the server ({type(exc).__name__})."}, status_code=500)

    # ------------------------------------------------------------ API (plain `def`: runs in a worker thread)
    @app.get("/api/status")
    def get_status():
        return core.status_body()

    @app.post("/api/detector")
    def post_detector(body: dict = Body(...)):
        """Head-turn sensitivity (added 2026-11-15): {"theta": one of config.HEAD_TURN_CHOICES}."""
        return core.set_theta(body.get("theta"))

    @app.get("/api/samples")
    def get_samples():
        return core.samples()

    @app.get("/api/replays")
    def get_replays():
        """Recorded look-away sessions that can be replayed for a rehearsal without the headband
        (added at integration, 2026-11-15). File names only; the replay itself is started over the
        WebSocket and is always labelled REPLAY."""
        from app.detector_bridge import REPLAY_FILE_RE
        try:
            names = sorted(p.name for p in config.RESULTS.iterdir() if REPLAY_FILE_RE.fullmatch(p.name))
        except OSError:
            names = []
        return [{"file": n} for n in names]

    @app.get("/api/samples/{sample_id}")
    def get_sample(sample_id: str):
        return core.sample(sample_id)

    @app.post("/api/sessions", status_code=201)
    def post_session(body: dict = Body(...)):
        return core.create_session(_utf8(body))

    @app.get("/api/sessions")
    def list_sessions():
        return [metrics.list_row(s) for s in store.list()]

    @app.get("/api/sessions/{sid}")
    def get_session(sid: str):
        return core.get_session(sid)

    @app.delete("/api/sessions/{sid}")
    def delete_session(sid: str):
        return core.delete_session(sid)

    @app.put("/api/sessions/{sid}/log")
    def put_log(sid: str, body: dict = Body(...)):
        return core.save_log(sid, _utf8(body))

    @app.post("/api/sessions/{sid}/finish")
    def post_finish(sid: str, body: dict = Body(...)):
        return core.finish(sid, _utf8(body))

    @app.post("/api/pdf_text")
    def post_pdf_text(body: dict = Body(...)):
        return core.pdf_text(_utf8(body))

    @app.post("/api/studies", status_code=201)
    def post_study(body: dict = Body(...)):
        return core.create_study(_utf8(body))

    @app.post("/api/sessions/{sid}/flashcards")
    def post_flashcards(sid: str, body: dict | None = Body(default=None)):
        return core.flashcards(sid, regenerate=bool((body or {}).get("regenerate")))

    @app.post("/api/sessions/{sid}/ask")
    def post_ask(sid: str, body: dict = Body(...)):
        return core.ask(sid, _utf8(body))

    @app.post("/api/sessions/{sid}/screenshots")
    def post_screenshot(sid: str, body: dict = Body(...)):
        return core.save_screenshot(sid, body)

    @app.get("/api/shots/{sid}/{file}")
    def get_shot(sid: str, file: str):
        path = store.shot_file(sid, file)
        if path is None:
            raise ApiError(404, "No such screenshot.")
        return FileResponse(path, media_type="image/jpeg", headers=NO_STORE)

    @app.get("/api/cues/{sid}/{file}")
    def get_cue(sid: str, file: str):
        path = store.cue_file(sid, file)
        if path is None:
            raise ApiError(404, "No such audio cue.")
        return FileResponse(path, media_type="audio/wav", headers=NO_STORE)

    @app.get("/api/night")
    def get_night():
        return core.night.status()

    @app.post("/api/night/start")
    def post_night_start(body: dict | None = Body(default=None)):
        return core.night_start(body or {})

    @app.post("/api/night/stop")
    def post_night_stop():
        return core.night.stop()

    @app.post("/api/night/stage")
    def post_night_stage():
        return core.night_stage()

    @app.get("/api/night/image/{file}")
    def get_night_image(file: str):
        path = core.night.image_path(file)
        if path is None:
            raise ApiError(404, "No such sleep plot.")
        return FileResponse(path, media_type="image/png", headers=NO_STORE)

    @app.get("/api/insights")
    def get_insights(include_simulated: str = "0"):
        return metrics.insights(store.list(), include_simulated=include_simulated == "1")

    # ------------------------------------------------------------ WebSocket
    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        client = core.hub.add(ws)
        try:
            # the other windows are told the connection count changed; this one gets the status directly
            status = await run_in_threadpool(core.broadcast_status, False, client)
            text = core.hub.encode({"type": "status", **status})
            if text is None or not await client.send_text(text):
                return
            while True:
                frame = await ws.receive()
                if frame["type"] == "websocket.disconnect":
                    break
                if not frame.get("text"):
                    continue
                try:
                    message = json.loads(frame["text"])
                except ValueError:
                    continue
                reply = await run_in_threadpool(core.handle_client_message, message)
                if reply is not None:
                    text = core.hub.encode(reply)
                    if text is not None and not await client.send_text(text):
                        break
        except (WebSocketDisconnect, RuntimeError):     # the socket closed under us
            pass
        except Exception:  # noqa: BLE001 - anything else ends this client only, but is not hidden
            traceback.print_exc()
        finally:
            core.hub.remove(client)
            try:
                await run_in_threadpool(core.broadcast_status)      # one connection fewer
            except Exception:  # noqa: BLE001
                traceback.print_exc()

    # ------------------------------------------------------------ static frontend (never cached)
    @app.get("/")
    def index():
        page = web / "index.html"
        if not page.is_file():
            raise ApiError(503, "The app's front page (app/web/index.html) is missing.")
        return FileResponse(page, media_type="text/html", headers=NO_STORE)

    for name in ("css", "js", "vendor"):
        app.mount(f"/{name}", NoStoreStatic(directory=web / name, check_dir=False), name=name)

    return app


def main() -> None:
    import uvicorn
    print(f"SynapStudy: http://{config.HOST}:{config.PORT}")
    # For test copies only (e.g. the demo recorder's dry run): a separate data folder, and no headband detector.
    data = os.environ.get("STUDY_FOCUS_DATA")
    no_detector = os.environ.get("STUDY_FOCUS_NO_DETECTOR") == "1"
    uvicorn.run(create_app(Path(data) if data else None, start_detector=not no_detector),
                host=config.HOST, port=config.PORT, log_level="warning")


if __name__ == "__main__":
    main()
