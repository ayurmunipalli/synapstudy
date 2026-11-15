"""HTTP API, WebSocket and detector bridge, through FastAPI's TestClient.

Nothing here touches the network, LSL, the real results/ folder or app_data/: the AI, content and cue
modules are replaced by the fakes below, and every file lives under pytest's temp dir. All passages, lapses
and events are synthetic and exist only in these tests.
"""
import asyncio
import base64
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app import config, metrics
from app import server as server_module
from app.detector_bridge import DetectorBridge, ReplayError
from app.server import ASK_NEEDS_KEY, NOTHING_COVERED, Hub, create_app
from app.store import new_id

T0 = 1_800_000_000_000
PASSAGE = "\n\n".join([" ".join(f"alpha{i}" for i in range(100)),
                       " ".join(f"beta{i}" for i in range(150)),
                       " ".join(f"gamma{i}" for i in range(50))])
VIDEO_URL = "https://www.youtube.com/watch?v=abcdefghijk"
NO_CAPTIONS_URL = "https://youtu.be/nocaptions0"
JPEG = b"\xff\xd8\xff\xe0" + b"synthetic-test-image" * 4


# ---------------------------------------------------------------------------------------------- fakes
class FakeLLM:
    def __init__(self, available=False):
        self.available = available
        self.fail_key_ideas_at = None
        self.gate = None            # a threading.Event the key-idea call waits on, when set

    def status(self):
        return {"available": self.available, "model": config.MODEL,
                "mode": "claude" if self.available else "extractive",
                "detail": "Fake key present" if self.available else "No ANTHROPIC_API_KEY"}

    def key_ideas(self, section_text, subject, title, cache_dir):
        if self.gate is not None:
            assert self.gate.wait(5)
        words = section_text.split()
        if self.fail_key_ideas_at is not None and words[0].startswith(self.fail_key_ideas_at):
            raise RuntimeError("synthetic failure")
        return {"ideas": [{"text": " ".join(words[:5]), "keyword": words[0]}],
                "source": "claude" if self.available else "extractive"}

    def summarize_missed(self, excerpt, cache_dir):
        return {"summary": f"Summary: {excerpt[:40]}", "source": "claude"} if self.available else None

    def concept_labels(self, windows, subject, title, cache_dir, use_claude=True):
        source = "claude" if (self.available and use_claude) else "extractive"
        return {"items": [{"idx": w["idx"], "start_s": w["start_s"], "end_s": w["end_s"],
                           "label": " ".join(w["text"].split()[:3])} for w in windows], "source": source}

    def make_flashcards(self, key_ideas, missed_excerpts, sections, subject, title, cache_dir, n_max=10):
        cards = [{"front": f"What is {k['keyword']}?", "back": k["text"], "concept": k["keyword"],
                  "section_idx": k["section_idx"],
                  "from_missed": any(k["keyword"] in e for e in missed_excerpts)} for k in key_ideas]
        cards.sort(key=lambda c: not c["from_missed"])
        return {"cards": cards[:n_max], "source": "claude" if self.available else "extractive"}

    def answer_question(self, question, context_sections, cache_dir):
        return {"answer": f"From {len(context_sections)} sections.", "source": "claude"} if self.available else None


class FakeContent:
    def __init__(self):
        self.fetches = 0

    def split_sections(self, text, target_words=170, max_words=220):
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        return [{"idx": i, "text": p, "word_count": len(p.split())} for i, p in enumerate(parts)]

    def parse_youtube_id(self, url):
        for marker in ("watch?v=", "youtu.be/"):
            if marker in url:
                return url.split(marker, 1)[1][:11]
        return None

    def fetch_transcript(self, video_id, cache_dir):
        self.fetches += 1
        if video_id.startswith("nocaptions"):
            return {"available": False, "error": "This video has no captions.", "snippets": []}
        return {"available": True, "error": None,
                "snippets": [{"start": 10.0 * i, "duration": 10.0, "text": f"line{i} about topic{i // 6}"}
                             for i in range(18)]}

    def transcript_sections(self, snippets, target_s=60.0):
        out = []
        for i in range(0, len(snippets), 6):
            chunk = snippets[i:i + 6]
            text = " ".join(s["text"] for s in chunk)
            out.append({"idx": len(out), "start_s": chunk[0]["start"], "end_s": chunk[-1]["start"] + chunk[-1]["duration"],
                        "text": text, "word_count": len(text.split())})
        return out

    def transcript_span(self, snippets, t_start, t_end, pad=5.0):
        return " ".join(s["text"] for s in snippets
                        if s["start"] < t_end + pad and s["start"] + s["duration"] > t_start - pad)

    def list_samples(self):
        return [{"id": f"darwin-{n}", "title": f"On the Origin of Species, part {n}", "words": 3} for n in (5, 6)]

    def load_sample(self, sample_id):
        return {"id": sample_id, "title": f"Sample {sample_id}", "text": "synthetic sample text"}


class FakeCues:
    def tts_available(self):
        return True

    def make_cues(self, cards, out_dir):
        out = []
        for i, card in enumerate(cards, 1):
            name = f"{i:02d}_{card['concept'].lower()}.wav"
            (out_dir / name).write_bytes(b"RIFF-synthetic-" + card["concept"].encode())
            out.append({"file": name, "text": card["concept"], "kind": "speech"})
        return out

    def write_manifest(self, session, cues, out_dir):
        path = out_dir / "manifest.json"
        path.write_text(json.dumps({"session_id": session["id"], "title": session["title"],
                                    "cards": [{"concept": c["concept"], "file": c["cue"]["file"]}
                                              for c in session["flashcards"]]}), encoding="utf-8")
        return path


class FakeRunner:
    """Stands in for focus.live.LslRunner: reports a connected headband as soon as it is started."""
    last = None

    def __init__(self, on_event, on_status, variant="combined", poll_s=0.05):
        self.on_event, self.on_status = on_event, on_status
        self.stopped = False
        self.params = "frozen"          # what set_params was last given ("frozen" = never called)
        FakeRunner.last = self

    def set_params(self, params):
        self.params = params

    def start(self):
        self.on_status({"connected": True, "detail": "Headband connected (fake)", "ready": True})

    def stop(self):
        self.stopped = True

    def is_alive(self):
        return False


class FakeLive:
    LslRunner = FakeRunner


class Env:
    def __init__(self, tmp_path, available=False, **create_kw):
        self.llm, self.content, self.cues = FakeLLM(available), FakeContent(), FakeCues()
        self.data = tmp_path / "data"
        deps = {"llm": self.llm, "content": self.content, "cues": self.cues, **create_kw.pop("deps", {})}
        self.app = create_app(self.data, start_detector=create_kw.pop("start_detector", False), deps=deps)
        self.core = self.app.state.core
        self.client = TestClient(self.app)

    def reading(self, subject="Biology", title="Synthetic passage", text=PASSAGE):
        r = self.client.post("/api/sessions", json={"mode": "reading", "subject": subject, "title": title, "text": text})
        assert r.status_code == 201, r.text
        return r.json()

    def video(self, url=VIDEO_URL):
        r = self.client.post("/api/sessions", json={"mode": "video", "subject": "Physics", "title": "Synthetic video",
                                                    "youtube_url": url})
        assert r.status_code == 201, r.text
        return r.json()

    def wait_ideas(self, sid, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            s = self.client.get(f"/api/sessions/{sid}").json()
            if s["key_ideas_status"] in ("done", "failed"):
                return s
            time.sleep(0.01)
        raise AssertionError("key ideas never finished")

    def finish(self, sid, log):
        r = self.client.post(f"/api/sessions/{sid}/finish", json=log)
        assert r.status_code == 200, r.text
        return r.json()


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


def reading_log(duration_s=600, lapses=(), progress=(0, 1), source="live"):
    return {"started_ms": T0, "ended_ms": T0 + duration_s * 1000,
            "gaze": {"mode": "mouse", "calibrated": False},
            "progress": [{"section_idx": i, "completed_ms": T0 + 1000 * (i + 1)} for i in progress],
            "lapses": [{"id": f"lapse-{n}", "kind": "look_away", "source": source, "onset_ms": T0 + a * 1000,
                        "resume_ms": T0 + b * 1000, "duration_s": b - a,
                        "position": {"section_idx": n % 3, "word_idx": 7, "sentence": f"Sentence {n} with beta0.",
                                     "method": "mouse"},
                        "screenshot_url": None} for n, (a, b) in enumerate(lapses)],
            "video": {"duration_s": None, "watched_s": 0.0}}


def recv(ws, kind=None, timeout=5.0):
    """Next WebSocket message (of the given type), failing instead of hanging if none arrives."""
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        while True:
            try:
                message = pool.submit(ws.receive_json).result(timeout)
            except FutureTimeout:
                raise AssertionError(f"no {kind or 'message'} within {timeout} s")
            if kind is None or message["type"] == kind:
                return message
    finally:
        pool.shutdown(wait=False)


# ---------------------------------------------------------------------------------------------- status, samples
def test_status_has_the_documented_shape_and_no_key(env, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key-000")
    body = env.client.get("/api/status").json()
    assert list(body) == ["source", "headband", "detector_ready", "detail", "detector", "ai", "audio", "tts",
                          "clients", "build"]
    assert body["clients"] == 0                 # no app window is connected in this test
    assert body["source"] == "simulated" and body["headband"] is False and body["detector_ready"] is False
    assert body["detector"]["variant"] == config.DETECTOR_VARIANT == "motion_only" and body["detector"]["acted_kinds"] == ["look_away"]
    # no settings file yet: the default head-turn setting, which is NOT the tested one, and says so
    assert body["detector"] == {
        "variant": "motion_only", "params": {"theta": 20, "N": 2}, "acted_kinds": ["look_away"], "theta": 20,
        "choices": [10, 15, 20, 30], "tested_theta": 10,
        "evidence": "Head-turn look-away (gyroscope, not EEG), 20\u00b0 setting: caught 10 of 11 cued look-aways in "
                    "the two recorded sessions (it missed one small downward look), with none during 13 min of "
                    "normal reading. Both sessions had been seen before this setting was chosen, so this is not "
                    "a held-out result."}
    assert list(body["detector"]) == ["variant", "params", "acted_kinds", "theta", "choices", "tested_theta", "evidence"]
    assert not (env.data / "settings.json").exists()        # reading the default writes nothing
    assert body["ai"] == {"available": False, "model": "claude-opus-5-5", "mode": "extractive",
                          "detail": "No ANTHROPIC_API_KEY"}
    assert body["audio"] == {"muse_speaker": False, "detail": config.AUDIO_DETAIL}
    assert body["tts"] == {"available": True, "engine": "say"}
    assert "sk-test-not-a-real-key-000" not in json.dumps(body)

    env.llm.status = lambda: {"available": True, "model": config.MODEL, "mode": "claude",
                              "detail": "leaky detail sk-test-not-a-real-key-000"}
    assert "sk-test-not-a-real-key-000" not in env.client.get("/api/status").text


def test_samples(env):
    assert env.client.get("/api/samples").json()[0] == {"id": "darwin-5", "title": "On the Origin of Species, part 5",
                                                        "words": 3}
    assert env.client.get("/api/samples/darwin-6").json() == {"id": "darwin-6", "title": "Sample darwin-6",
                                                              "text": "synthetic sample text"}
    for bad in ("darwin-99", "..%2Fx", "DARWIN-5"):
        r = env.client.get(f"/api/samples/{bad}")
        assert r.status_code == 404 and "error" in r.json()


# ---------------------------------------------------------------------------------------------- creating sessions
def test_create_reading_session_then_key_ideas_reach_done(env):
    env.llm.gate = threading.Event()        # hold generation so the 201 body is seen before any idea exists
    s = env.reading()
    assert s["mode"] == "reading" and s["subject"] == "Biology" and s["ai_mode"] == "extractive"
    assert [sec["word_count"] for sec in s["content"]["sections"]] == [100, 150, 50]
    assert s["key_ideas"] == [] and s["key_ideas_status"] == "pending" and s["key_ideas_error"] is None
    assert s["simulated"] is False and s["ended_at"] is None and s["summary"] is None and s["log"] is None
    assert env.client.get(f"/api/sessions/{s['id']}").json()["key_ideas_status"] == "pending"

    env.llm.gate.set()
    done = env.wait_ideas(s["id"])
    assert done["key_ideas_status"] == "done" and done["key_ideas_error"] is None
    assert [(k["section_idx"], k["keyword"], k["source"]) for k in done["key_ideas"]] == [
        (0, "alpha0", "extractive"), (1, "beta0", "extractive"), (2, "gamma0", "extractive")]
    assert all(len(k["id"]) == 32 and k["text"] for k in done["key_ideas"])
    assert (env.data / "sessions" / f"{s['id']}.json").is_file()


@pytest.mark.parametrize("body", [
    {"mode": "reading", "subject": "x", "title": "y", "text": ""},
    {"mode": "reading", "subject": "x", "title": "y", "text": "   \n "},
    {"mode": "reading", "subject": "x", "title": "y"},
    {"mode": "reading", "subject": "x", "title": "y", "text": "a" * (config.TEXT_MAX_CHARS + 1)},
    {"mode": "video", "subject": "x", "title": "y", "youtube_url": "not a link"},
    {"mode": "video", "subject": "x", "title": "y"},
    {"mode": "podcast", "subject": "x", "title": "y", "text": "hello"},
])
def test_create_rejects_bad_input_with_a_plain_error(env, body):
    r = env.client.post("/api/sessions", json=body)
    assert r.status_code == 400 and list(r.json()) == ["error"] and r.json()["error"]
    assert env.client.get("/api/sessions").json() == []


def test_bad_json_bodies_get_the_error_shape(env):
    r = env.client.post("/api/sessions", content=b"{nope", headers={"content-type": "application/json"})
    assert r.status_code == 400 and list(r.json()) == ["error"]
    r = env.client.post("/api/sessions", json=["a", "list"])
    assert r.status_code == 400 and list(r.json()) == ["error"]
    r = env.client.get("/api/nothing-here")
    assert r.status_code == 404 and r.json() == {"error": "Not found."}


def test_text_with_a_lone_surrogate_is_stored_and_served(env):
    raw = b'{"mode": "reading", "subject": "x", "title": "y", "text": "before \\ud83d after"}'
    r = env.client.post("/api/sessions", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 201
    assert r.json()["content"]["sections"][0]["text"] == "before ? after"
    assert env.client.get(f"/api/sessions/{r.json()['id']}").status_code == 200


def test_create_video_session_with_and_without_captions(env):
    s = env.video()
    c = s["content"]
    assert c["youtube_url"] == VIDEO_URL and c["video_id"] == "abcdefghijk"
    assert c["transcript_available"] is True and c["transcript_error"] is None
    assert [(x["idx"], x["start_s"], x["end_s"]) for x in c["sections"]] == [(0, 0.0, 60.0), (1, 60.0, 120.0), (2, 120.0, 180.0)]
    assert s["key_ideas_status"] == "pending"
    assert len(env.wait_ideas(s["id"])["key_ideas"]) == 3

    n = env.video(NO_CAPTIONS_URL)
    assert n["content"]["sections"] == [] and n["content"]["transcript_available"] is False
    assert n["content"]["transcript_error"] == "This video has no captions."
    assert n["key_ideas_status"] == "done" and n["key_ideas"] == []


def test_key_ideas_are_pushed_over_the_websocket_section_by_section(env):
    env.llm.gate = threading.Event()
    with env.client.websocket_connect("/ws") as ws:
        assert recv(ws)["type"] == "status"
        s = env.reading()
        env.llm.gate.set()
        messages = [recv(ws, "key_ideas") for _ in range(3)]
    assert [m["section_idx"] for m in messages] == [0, 1, 2]
    assert [m["status"] for m in messages] == ["partial", "partial", "done"]
    assert all(m["session_id"] == s["id"] and m["error"] is None and len(m["ideas"]) == 1 for m in messages)
    stored = env.wait_ideas(s["id"])["key_ideas"]
    assert [m["ideas"][0] for m in messages] == stored


def test_key_idea_failure_is_recorded_not_hidden(env):
    env.llm.fail_key_ideas_at = "beta"
    s = env.wait_ideas(env.reading()["id"])
    assert s["key_ideas_status"] == "failed"
    assert "section 2" in s["key_ideas_error"] and "RuntimeError" in s["key_ideas_error"]
    assert [k["section_idx"] for k in s["key_ideas"]] == [0]       # what was written before the failure is kept


def test_key_idea_generation_resumes_after_a_restart(env):
    """A session left `partial` on disk (as after a server restart) is picked up when it is next opened."""
    sections = env.content.split_sections(PASSAGE)
    s = env.core.store.create("reading", "Biology", "Left over", "extractive", {"sections": sections})
    env.core.store.update(s["id"], lambda x: (x["key_ideas"].append(
        {"id": new_id(), "section_idx": 0, "text": "kept", "keyword": "kept", "source": "extractive"}),
        x.update(key_ideas_status="partial")))
    done = env.wait_ideas(s["id"])
    assert [(k["section_idx"], k["keyword"]) for k in done["key_ideas"]] == [(0, "kept"), (1, "beta0"), (2, "gamma0")]


# ---------------------------------------------------------------------------------------------- log, finish
def test_log_autosave(env):
    sid = env.reading()["id"]
    log = reading_log(lapses=[(100, 110)])
    log["ended_ms"] = None
    assert env.client.put(f"/api/sessions/{sid}/log", json=log).json() == {"ok": True}
    saved = env.client.get(f"/api/sessions/{sid}").json()
    assert saved["log"]["lapses"][0]["id"] == "lapse-0" and saved["log"]["ended_ms"] is None
    assert saved["summary"] is None and saved["ended_at"] is None      # autosave never finishes a session

    log["lapses"] = []
    assert env.client.put(f"/api/sessions/{sid}/log", json=log).status_code == 200
    assert env.client.get(f"/api/sessions/{sid}").json()["log"]["lapses"] == []

    assert env.client.put(f"/api/sessions/{sid}/log", json=[1, 2]).status_code == 400
    assert env.client.put(f"/api/sessions/{new_id()}/log", json=log).status_code == 404
    env.finish(sid, reading_log())
    r = env.client.put(f"/api/sessions/{sid}/log", json=log)
    assert r.status_code == 409 and "error" in r.json()


def test_finish_reading_computes_the_metrics(tmp_path):
    env = live_env(tmp_path)            # a (fake) headband is connected, so the session counts as measured
    with env.client:
        sid = env.reading()["id"]
        log = reading_log(duration_s=600, lapses=[(100, 110), (300, 311.5)], progress=(0, 1, 0))
        log["lapses"][1]["position"]["page"] = 2
        log["lapses"][0]["end_method"] = "turn_back"
        log["lapses"][1]["end_method"] = "made up by the client"
        s = env.finish(sid, log)
        m = s["summary"]
        assert list(m) == ["duration_s", "lapse_count", "lapse_seconds", "focus_pct", "lapses_per_min", "reading",
                           "video", "missed", "simulated", "measured", "generated_at"]
        assert m["duration_s"] == 600.0 and m["lapse_count"] == 2 and m["lapse_seconds"] == 21.5
        assert m["focus_pct"] == 96.4 and m["lapses_per_min"] == 0.2
        assert m["reading"] == {"words_read": 250, "wpm": round(250 / ((600 - 21.5) / 60), 1)} and m["video"] is None
        assert m["simulated"] is False and s["simulated"] is False and s["ended_at"]
        assert m["measured"] is True and s["measured"] is True and s["measure_note"] is None
        assert [e["source"] for e in s["source_log"]] == ["live"]       # the sample taken at finish
        first, second = m["missed"]
        assert first == {"lapse_id": "lapse-0", "kind": "look_away", "source": "live", "label": "Section 1 · 01:40",
                         "duration_s": 10.0, "end_method": "turn_back", "excerpt": "Sentence 0 with beta0.",
                         "summary": None, "summary_source": "excerpt", "position": log["lapses"][0]["position"],
                         "screenshot_url": None}
        assert second["label"] == "Page 2 · 05:00" and second["duration_s"] == 11.5
        assert second["end_method"] is None         # only the documented end methods are passed on

        # finishing again (even with a different log) does not rewrite a finished session
        again = env.client.post(f"/api/sessions/{sid}/finish", json=reading_log(duration_s=60))
        assert again.status_code == 200 and again.json()["summary"] == m

        row = env.client.get("/api/sessions").json()[0]
        assert row == {"id": sid, "title": "Synthetic passage", "subject": "Biology", "mode": "reading",
                       "created_at": s["created_at"], "ended_at": s["ended_at"], "duration_s": 600.0,
                       "focus_pct": 96.4, "lapse_count": 2, "wpm": m["reading"]["wpm"], "simulated": False,
                       "measured": True}


@pytest.mark.parametrize("end_method", ["movement_end", "turn_back", "manual", "simulated", "disconnect",
                                        "session_end"])
def test_missed_entries_carry_the_lapses_end_method(env, end_method):
    sid = env.reading()["id"]
    log = reading_log(lapses=[(10, 12), (20, 22)], source="simulated")
    log["lapses"][0]["end_method"] = end_method
    missed = env.finish(sid, log)["summary"]["missed"]
    assert [m["end_method"] for m in missed] == [end_method, None]      # absent in the log -> null


def test_finish_needs_a_usable_log(env):
    sid = env.reading()["id"]
    assert env.client.post(f"/api/sessions/{sid}/finish", json={"lapses": []}).status_code == 400
    bad = reading_log()
    bad["ended_ms"] = T0 - 1
    assert env.client.post(f"/api/sessions/{sid}/finish", json=bad).status_code == 400
    assert env.client.get(f"/api/sessions/{sid}").json()["ended_at"] is None


def video_log(lapses):
    return {"started_ms": T0, "ended_ms": T0 + 180_000, "gaze": {"mode": "none", "calibrated": False}, "progress": [],
            "lapses": [{"id": f"v{n}", "kind": "look_away", "source": "live", "onset_ms": T0 + a * 1000,
                        "resume_ms": T0 + b * 1000, "duration_s": b - a, "position": {"t_start": a, "t_end": b},
                        "screenshot_url": None} for n, (a, b) in enumerate(lapses)],
            "video": {"duration_s": 180.0, "watched_s": 171.5}}


def test_finish_video_without_a_key_gives_transcript_excerpts_only(env):
    sid = env.video()["id"]
    m = env.finish(sid, video_log([(102, 114)]))["summary"]
    assert m["reading"] is None and m["video"] == {"duration_s": 180.0, "watched_s": 171.5}
    missed = m["missed"][0]
    # the fake transcript has one 10 s line per 10 s; [102 - 5, 114 + 5] overlaps lines 9, 10 and 11
    assert missed["excerpt"] == "line9 about topic1 line10 about topic1 line11 about topic1"
    assert missed["summary"] is None and missed["summary_source"] == "excerpt"
    assert missed["label"] == "01:42-01:54" and missed["position"] == {"t_start": 102, "t_end": 114}


def test_finish_video_with_the_model_adds_summaries(tmp_path):
    env = Env(tmp_path, available=True)
    sid = env.video()["id"]
    missed = env.finish(sid, video_log([(102, 114), (20, 25)]))["summary"]["missed"]
    assert missed[0]["summary"] == "Summary: line9 about topic1 line10 about topic1 l" and missed[0]["summary_source"] == "claude"
    assert missed[1]["excerpt"].startswith("line1 ") and missed[1]["summary_source"] == "claude"


def test_finish_video_without_captions_reports_none(env):
    sid = env.video(NO_CAPTIONS_URL)["id"]
    missed = env.finish(sid, video_log([(10, 20)]))["summary"]["missed"][0]
    assert missed["excerpt"] is None and missed["summary"] is None and missed["summary_source"] == "none"


# ---------------------------------------------------------------------------------------------- simulated flag
def test_simulated_flag_comes_from_the_servers_event_log_and_the_client_cannot_clear_it(env):
    sid = env.reading()["id"]
    env.wait_ideas(sid)                 # so no key_ideas message lands among the ones checked below
    with env.client.websocket_connect("/ws") as ws:
        assert recv(ws)["source"] == "simulated"
        ws.send_json({"type": "session", "id": sid, "state": "start"})
        ws.send_json({"type": "sim", "action": "drift"})
        drift = recv(ws, "drift")
        assert drift["kind"] == "look_away" and drift["source"] == "simulated" and drift["acted"] is True
        assert drift["confidence"] is None and drift["onset_wall_ms"] == pytest.approx(drift["wall_ms"] - 750)
        assert abs(drift["wall_ms"] - time.time() * 1000) < 5000
        ws.send_json({"type": "sim", "action": "drift"})        # already drifting: closed, then a fresh drift
        assert recv(ws)["type"] == "resume" and recv(ws)["type"] == "drift"
        ws.send_json({"type": "sim", "action": "resume"})
        assert recv(ws)["type"] == "resume"
        ws.send_json({"type": "sim", "action": "resume"})       # nothing open: ignored
        ws.send_json({"type": "session", "id": sid, "state": "end"})
        ws.send_json({"type": "sim", "action": "drift"})        # after "end": broadcast but not logged
        assert recv(ws, "drift")["source"] == "simulated"

    before = env.client.get(f"/api/sessions/{sid}").json()
    assert [(e["type"], e["kind"], e["source"]) for e in before["server_events"]] == [
        ("drift", "look_away", "simulated"), ("resume", None, "simulated"),
        ("drift", "look_away", "simulated"), ("resume", None, "simulated")]
    assert before["server_events"][0]["onset_wall_ms"] == drift["onset_wall_ms"]
    assert before["simulated"] is True and before["source_counts"] == {"live": 0, "replay": 0, "simulated": 2}
    assert [e["source"] for e in before["source_log"]] == ["simulated"]     # noted when the session became active
    assert before["measured"] is None and before["measure_note"] is None    # decided only at finish

    # The client now claims a clean, live session: no lapses, and its own "simulated": false everywhere.
    lying = reading_log(lapses=[(10, 12)], source="live")
    lying.update(simulated=False, server_events=[], source_counts={"live": 1, "replay": 0, "simulated": 0})
    assert env.client.put(f"/api/sessions/{sid}/log", json=lying).status_code == 200
    assert env.client.get(f"/api/sessions/{sid}").json()["simulated"] is True
    done = env.finish(sid, lying)
    assert done["simulated"] is True and done["summary"]["simulated"] is True
    assert len(done["server_events"]) == 4 and done["source_counts"]["simulated"] == 2
    assert done["measured"] is False and done["summary"]["measured"] is False
    assert done["measure_note"] == metrics.NOTE_SIMULATED
    row = env.client.get("/api/sessions").json()[0]
    assert row["simulated"] is True and row["measured"] is False


def test_a_lapse_the_client_reports_as_not_live_also_flags_the_session(env):
    sid = env.reading()["id"]       # no server events at all (the browser never said "session start")
    done = env.finish(sid, reading_log(lapses=[(10, 12)], source="replay"))
    assert done["simulated"] is True and done["summary"]["missed"][0]["source"] == "replay"


def live_env(tmp_path):
    """An app whose detector bridge believes a headband is connected (fake runner, fake LSL clock)."""
    def factory(on_message):
        return DetectorBridge(on_message, live_module=FakeLive, results_dir=tmp_path / "results",
                              lsl_clock=lambda: 1000.0)
    return Env(tmp_path, start_detector=True, deps={"bridge_factory": factory})


def test_sim_messages_are_ignored_unless_the_source_is_simulated(tmp_path):
    env = live_env(tmp_path)
    with env.client:                    # runs startup, which starts the (fake) live detector
        sid = env.reading()["id"]
        env.wait_ideas(sid)
        with env.client.websocket_connect("/ws") as ws:
            status = recv(ws)
            assert status["source"] == "live" and status["headband"] is True and status["detector_ready"] is True
            ws.send_json({"type": "session", "id": sid, "state": "start"})
            ws.send_json({"type": "sim", "action": "drift"})
            ws.send_json({"type": "sim", "action": "resume"})
            ws.send_json({"type": "replay", "action": "start", "file": "../../etc/passwd"})
            # the very next message is the replay refusal: the sim messages produced nothing
            assert recv(ws)["type"] == "error"
            assert env.client.get(f"/api/sessions/{sid}").json()["server_events"] == []

            # a real (here: fake-runner) live drift is broadcast, logged, and does not flag the session
            FakeRunner.last.on_event({"type": "drift", "kind": "look_away+eyes_closed", "t": 998.0, "onset_t": 997.25,
                                      "confidence": None})
            drift = recv(ws)
            assert drift["type"] == "drift" and drift["source"] == "live" and drift["acted"] is True
            assert drift["t"] == 998.0 and drift["onset_t"] == 997.25
            assert drift["wall_ms"] - drift["onset_wall_ms"] == pytest.approx(750, abs=50)
            assert time.time() * 1000 - drift["wall_ms"] == pytest.approx(2000, abs=500)   # LSL clock says 2 s ago
            FakeRunner.last.on_event({"type": "drift", "kind": "eyes_closed", "t": 999.0, "onset_t": 998.25})
            assert recv(ws)["acted"] is False                   # logged and broadcast, but not acted on
            FakeRunner.last.on_event({"type": "resume", "t": 999.5})
            assert recv(ws) == {"type": "resume", "source": "live", "wall_ms": pytest.approx(time.time() * 1000 - 500, abs=500),
                                "t": 999.5, "net_rot": None, "peak_angle": None}
            FakeRunner.last.on_event({"type": "signal", "t": 999.75, "head": 1.5, "eye": float("nan"), "quiet": 0.2,
                                      "alpha_ratio": 1.1, "ready": True, "quality": [True, False, True, True]})
            signal = recv(ws)
            assert signal["type"] == "signal" and signal["eye"] is None and signal["head"] == 1.5
            assert signal["quality"] == [True, False, True, True] and "quiet" not in signal

        s = env.finish(sid, reading_log(lapses=[(10, 12)]))
        assert [e["source"] for e in s["server_events"]] == ["live", "live", "live"]
        assert s["source_counts"] == {"live": 2, "replay": 0, "simulated": 0}
        assert s["simulated"] is False and s["summary"]["simulated"] is False
    assert FakeRunner.last.stopped is True      # shutdown stops the detector


# ---------------------------------------------------------------------------------------------- head-turn setting
def test_head_turn_evidence_names_the_setting_and_only_10_claims_a_held_out_test():
    assert config.HEAD_TURN_CHOICES == (10, 15, 20, 30)
    assert config.HEAD_TURN_TESTED == 10 and config.HEAD_TURN_DEFAULT == 20
    assert config.head_turn_evidence(10) == (
        "Head-turn look-away (gyroscope, not EEG), 10\u00b0 setting: 6/6 caught in 1 held-out session, 0 false "
        "alarms in 6 min of reading (RESULTS.md). This is the only setting tested on a session it was not tuned on.")
    assert config.head_turn_evidence(15) == (
        "Head-turn look-away (gyroscope, not EEG), 15\u00b0 setting: caught 11 of 11 cued look-aways in the two "
        "recorded sessions, with none during 13 min of normal reading. Both sessions had been seen before this "
        "setting was chosen, so this is not a held-out result.")
    assert config.head_turn_evidence(30) == config.head_turn_evidence(20).replace("20\u00b0", "30\u00b0")
    for theta in config.HEAD_TURN_CHOICES:
        text = config.head_turn_evidence(theta)
        assert f"{theta}\u00b0 setting" in text
        assert ("not a held-out result" in text) == (theta != config.HEAD_TURN_TESTED)
        assert ("held-out session" in text) == (theta == config.HEAD_TURN_TESTED)


def test_post_detector_changes_the_setting_saves_it_and_tells_every_window(env):
    with env.client.websocket_connect("/ws") as ws:
        assert recv(ws, "status")["detector"]["theta"] == 20
        r = env.client.post("/api/detector", json={"theta": 15})
        assert r.status_code == 200
        body = r.json()
        assert body["detector"]["theta"] == 15 and body["detector"]["params"] == {"theta": 15, "N": 2}
        assert body["detector"]["evidence"] == config.head_turn_evidence(15)
        assert list(body) == list(env.client.get("/api/status").json())        # the status body, whole
        pushed = recv(ws, "status")
        assert pushed["detector"] == body["detector"]
    assert json.loads((env.data / "settings.json").read_text()) == {"theta": 15}
    assert env.core.bridge.params() == {"theta": 15, "N": 2}                   # replays started now use it
    assert env.client.get("/api/status").json()["detector"]["theta"] == 15

    # the tested setting gives the frozen parameters back exactly
    body = env.client.post("/api/detector", json={"theta": 10}).json()
    frozen = json.loads(config.FROZEN_PARAMS.read_text())["variants"]["motion_only"]["params"]
    assert body["detector"]["params"] == frozen and body["detector"]["evidence"] == config.head_turn_evidence(10)


@pytest.mark.parametrize("bad", [{"theta": 12}, {"theta": "15"}, {"theta": None}, {"theta": True}, {"theta": 15.5},
                                 {"theta": [15]}, {}, {"degrees": 15}])
def test_post_detector_refuses_anything_that_is_not_one_of_the_choices(env, bad):
    r = env.client.post("/api/detector", json=bad)
    assert r.status_code == 400
    assert r.json() == {"error": "The head-turn setting must be 10, 15, 20 or 30 degrees."}
    assert env.client.get("/api/status").json()["detector"]["theta"] == 20
    assert not (env.data / "settings.json").exists() and env.core.bridge.params() == {"theta": 20, "N": 2}


def test_the_saved_head_turn_setting_is_read_at_start_and_a_bad_file_gives_the_default(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "settings.json").write_text('{"theta": 30, "something_else": 1}')
    env = Env(tmp_path)
    assert env.client.get("/api/status").json()["detector"]["params"] == {"theta": 30, "N": 2}
    assert env.core.bridge.params() == {"theta": 30, "N": 2}
    env.client.post("/api/detector", json={"theta": 10})
    assert json.loads((data / "settings.json").read_text()) == {"theta": 10, "something_else": 1}   # the rest is kept

    for n, bad in enumerate(['{"theta": 12}', '{"theta": "20"}', '{"theta": true}', "[10]", "not json", ""]):
        other = tmp_path / f"bad{n}"
        (other / "data").mkdir(parents=True)
        (other / "data" / "settings.json").write_text(bad)
        assert Env(other).client.get("/api/status").json()["detector"]["theta"] == config.HEAD_TURN_DEFAULT


def test_the_head_turn_setting_reaches_the_live_runner_at_start_and_when_changed(tmp_path):
    env = live_env(tmp_path)
    with env.client:
        assert FakeRunner.last.params == {"theta": 20, "N": 2}          # given before the runner was started
        assert env.client.post("/api/detector", json={"theta": 30}).status_code == 200
        assert FakeRunner.last.params == {"theta": 30, "N": 2}          # and at once when it is changed

        def refuse(_params):
            raise ValueError("refused")
        FakeRunner.last.set_params = refuse
        r = env.client.post("/api/detector", json={"theta": 15})
        assert r.status_code == 500 and "did not accept" in r.json()["error"]
        assert env.client.get("/api/status").json()["detector"]["theta"] == 30      # nothing changed or saved
        assert json.loads((env.data / "settings.json").read_text()) == {"theta": 30}


def test_a_replay_uses_the_head_turn_setting_in_effect_when_it_starts(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "lookaway-1.npz").write_bytes(b"synthetic")
    calls = []

    class Live:
        @staticmethod
        def replay_events(path, variant="combined", params=None):
            calls.append((variant, params))
            raise ValueError("stop here: the test only needs the arguments")

    done = threading.Event()
    bridge = DetectorBridge(lambda m: done.set() if m.get("source") == "simulated" else None,
                            live_module=Live, results_dir=results)
    bridge.start_replay("lookaway-1.npz")
    assert done.wait(5)
    done.clear()
    bridge.set_params({"theta": 20, "N": 2})
    bridge.start_replay("lookaway-1.npz")
    assert done.wait(5)
    assert calls == [("motion_only", None), ("motion_only", {"theta": 20, "N": 2})]     # None = the frozen ones


def test_drift_and_resume_messages_carry_what_the_head_did_and_the_session_keeps_it(tmp_path):
    env = live_env(tmp_path)
    with env.client:
        sid = env.reading()["id"]
        env.wait_ideas(sid)
        with env.client.websocket_connect("/ws") as ws:
            recv(ws)
            ws.send_json({"type": "session", "id": sid, "state": "start"})
            while not env.client.get(f"/api/sessions/{sid}").json()["source_log"]:
                time.sleep(0.01)
            FakeRunner.last.on_event({"type": "drift", "kind": "look_away", "t": 998.0, "onset_t": 997.75,
                                      "confidence": None, "angle": 23.456789, "rot": [1.0, -23.4321, 0.5]})
            drift = recv(ws, "drift")
            assert drift["angle"] == 23.46 and drift["rot"] == [1.0, -23.43, 0.5]
            FakeRunner.last.on_event({"type": "resume", "t": 999.5, "net_rot": [2.0, -51.239, 1.0], "peak_angle": 48.004})
            resume = recv(ws, "resume")
            assert resume["net_rot"] == [2.0, -51.24, 1.0] and resume["peak_angle"] == 48.0
            # no motion data, or something that is not three finite numbers: null, never NaN or a guess
            FakeRunner.last.on_event({"type": "drift", "kind": "look_away", "t": 1000.0, "onset_t": 999.75,
                                      "angle": float("nan"), "rot": [1.0, float("inf"), 0.0]})
            drift2 = recv(ws, "drift")
            assert drift2["angle"] is None and drift2["rot"] is None
            FakeRunner.last.on_event({"type": "resume", "t": 1001.0, "net_rot": [1.0, 2.0], "peak_angle": None})
            resume2 = recv(ws, "resume")
            assert resume2["net_rot"] is None and resume2["peak_angle"] is None
        events = env.client.get(f"/api/sessions/{sid}").json()["server_events"]
    assert [sorted(e) for e in events] == [
        ["angle", "kind", "onset_wall_ms", "rot", "source", "type", "wall_ms"],
        ["kind", "net_rot", "onset_wall_ms", "peak_angle", "source", "type", "wall_ms"],
        ["kind", "onset_wall_ms", "source", "type", "wall_ms"],
        ["kind", "onset_wall_ms", "source", "type", "wall_ms"]]
    assert events[0]["angle"] == 23.46 and events[0]["rot"] == [1.0, -23.43, 0.5]
    assert events[1]["net_rot"] == [2.0, -51.24, 1.0] and events[1]["peak_angle"] == 48.0


# ---------------------------------------------------------------------------------------------- bridge
def test_bridge_sim_only_when_simulated():
    seen = []
    bridge = DetectorBridge(seen.append, live_module=FakeLive)
    assert bridge.status()["source"] == "simulated"
    assert bridge.sim("resume") is False and bridge.sim("jump") is False and seen == []
    assert bridge.sim("drift") is True
    # The browser can end a simulated lapse without telling the bridge. A second drift must not be swallowed:
    # it closes the one the bridge still thinks is open and fires a fresh one.
    assert bridge.sim("drift") is True
    assert [m["type"] for m in seen] == ["drift", "resume", "drift"]
    assert seen[2]["onset_wall_ms"] == pytest.approx(seen[2]["wall_ms"] - 750) and seen[1]["wall_ms"] <= seen[2]["wall_ms"]
    assert bridge.sim("resume") is True
    assert bridge.sim("resume") is False        # none open: still a no-op
    assert [m["type"] for m in seen] == ["drift", "resume", "drift", "resume"]
    assert all(m["source"] == "simulated" for m in seen)

    bridge.sim("drift")
    bridge.start()                      # the fake runner reports a headband: source becomes live
    assert bridge.status() == {"source": "live", "headband": True, "detector_ready": True,
                               "detail": "Headband connected (fake)"}
    assert seen[-2] == {"type": "resume", "source": "simulated", "wall_ms": seen[-2]["wall_ms"], "t": seen[-2]["t"]}
    assert seen[-1]["type"] == "status" and seen[-1]["source"] == "live"
    assert [m["type"] for m in seen[-4:]] == ["drift", "status", "resume", "status"]
    n = len(seen)
    assert bridge.sim("drift") is False and bridge.sim("resume") is False and len(seen) == n
    FakeRunner.last.on_status({"connected": False, "detail": "No headband stream found", "ready": False})
    assert bridge.status() == {"source": "simulated", "headband": False, "detector_ready": False,
                               "detail": "No headband stream found"}
    assert bridge.sim("drift") is True
    bridge.stop()


def test_bridge_without_focus_live_stays_simulated_and_says_why():
    class Broken:
        @staticmethod
        def LslRunner(**_kw):
            raise ImportError("No module named 'pylsl'")

    seen = []
    bridge = DetectorBridge(seen.append, live_module=Broken)
    bridge.start()
    st = bridge.status()
    assert st["source"] == "simulated" and st["headband"] is False and "pylsl" in st["detail"]
    assert bridge.sim("drift") is True


def test_replay_accepts_only_a_bare_lookaway_file_inside_results(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "lookaway-20261115-000831.npz").write_bytes(b"synthetic")
    (results / "nap-20261114-170825.npz").write_bytes(b"synthetic")
    (tmp_path / "lookaway-1.npz").write_bytes(b"outside results")
    sub = results / "sub"
    sub.mkdir()
    (sub / "lookaway-2.npz").write_bytes(b"nested")
    bridge = DetectorBridge(lambda m: None, results_dir=results)
    assert bridge.replay_path("lookaway-20261115-000831.npz") == (results / "lookaway-20261115-000831.npz").resolve()
    for bad in ["../lookaway-1.npz", "sub/lookaway-2.npz", "nap-20261114-170825.npz", "lookaway-99.npz",
                "lookaway-20261115-000831.npz/", "/etc/passwd", "lookaway-.npz.exe", "lookaway-1.npz\n",
                "lookaway-20261115-000831_meta.json", "", None, 7, ["lookaway-20261115-000831.npz"]]:
        with pytest.raises(ReplayError):
            bridge.replay_path(bad)
    assert (results / "lookaway-20261115-000831.npz").read_bytes() == b"synthetic"     # read-only


def test_replay_plays_events_as_replay_and_is_refused_while_live(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "lookaway-1.npz").write_bytes(b"synthetic")
    seen, finished = [], threading.Event()

    def on_message(m):
        seen.append(m)
        if m["type"] == "status" and m["source"] == "simulated":
            finished.set()

    events = [{"type": "drift", "kind": "look_away", "t": 100.5, "onset_t": 99.75, "confidence": None},
              {"type": "resume", "t": 103.0},
              {"type": "drift", "kind": "eyes_closed", "t": 110.0, "onset_t": 109.25, "confidence": None}]
    bridge = DetectorBridge(on_message, live_module=FakeLive, results_dir=results, replay_speed=500.0,
                            replay_loader=lambda path: (events, 100.0, 12.0))
    bridge.start_replay("lookaway-1.npz")
    assert finished.wait(5)
    kinds = [(m["type"], m.get("source"), m.get("kind")) for m in seen]
    assert kinds == [("status", "replay", None), ("status", "replay", None),
                     ("drift", "replay", "look_away"), ("resume", "replay", None), ("drift", "replay", "eyes_closed"),
                     ("resume", "replay", None),            # a drift still open at the end is closed
                     ("status", "simulated", None)]
    assert seen[0]["detector_ready"] is False and seen[1]["detector_ready"] is True
    assert seen[2]["acted"] is True and seen[4]["acted"] is False
    # events without rotation fields (as here) go out with them set to null
    assert seen[2]["angle"] is None and seen[2]["rot"] is None
    assert seen[3] == {"type": "resume", "source": "replay", "wall_ms": seen[3]["wall_ms"], "t": 103.0,
                       "net_rot": None, "peak_angle": None}
    assert seen[5]["net_rot"] is None and seen[5]["peak_angle"] is None and set(seen[5]) == set(seen[3])
    assert seen[2]["wall_ms"] - seen[2]["onset_wall_ms"] == pytest.approx(750)
    assert bridge.sim("drift") is True          # back to simulated once the replay ends
    bridge.sim("resume")

    bridge.start()                              # headband (fake) connects
    with pytest.raises(ReplayError, match="live"):
        bridge.start_replay("lookaway-1.npz")
    bridge.stop()


def test_replay_that_cannot_be_read_falls_back_to_simulated(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "lookaway-1.npz").write_bytes(b"synthetic")
    done = threading.Event()

    def loader(_path):
        raise ValueError("not a recording")

    bridge = DetectorBridge(lambda m: done.set() if m.get("source") == "simulated" else None,
                            results_dir=results, replay_loader=loader)
    bridge.start_replay("lookaway-1.npz")
    assert done.wait(5)
    st = bridge.status()
    assert st["source"] == "simulated" and "failed" in st["detail"]


def test_replay_over_the_websocket(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "lookaway-1.npz").write_bytes(b"synthetic")
    events = [{"type": "drift", "kind": "look_away", "t": 0.2, "onset_t": 0.0, "confidence": None},
              {"type": "resume", "t": 0.4}]

    def factory(on_message):
        return DetectorBridge(on_message, results_dir=results, replay_speed=20.0,
                              replay_loader=lambda path: (events, 0.0, 0.6))

    env = Env(tmp_path, deps={"bridge_factory": factory})
    sid = env.reading()["id"]
    env.wait_ideas(sid)
    with env.client.websocket_connect("/ws") as ws:
        recv(ws)
        ws.send_json({"type": "session", "id": sid, "state": "start"})
        ws.send_json({"type": "replay", "action": "start", "file": "lookaway-1.npz"})
        assert recv(ws, "status")["source"] == "replay"
        ws.send_json({"type": "sim", "action": "drift"})        # ignored: the source is replay
        assert recv(ws, "drift")["source"] == "replay"
        assert recv(ws, "resume")["source"] == "replay"
        while recv(ws, "status")["source"] != "simulated":
            pass
    s = env.client.get(f"/api/sessions/{sid}").json()
    assert [(e["type"], e["source"]) for e in s["server_events"]] == [("drift", "replay"), ("resume", "replay")]
    assert s["simulated"] is True and s["source_counts"] == {"live": 0, "replay": 1, "simulated": 0}


# ---------------------------------------------------------------------------------------------- flashcards, ask
def test_flashcards_flow(env, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SLEEP_MODEL", tmp_path / "no_such_model.npz")     # no trained sleep model
    sid = env.reading()["id"]
    r = env.client.post(f"/api/sessions/{sid}/flashcards")
    assert r.status_code == 409 and "error" in r.json()            # needs a summary first

    env.finish(sid, reading_log(lapses=[(100, 110)]))
    s = env.client.post(f"/api/sessions/{sid}/flashcards").json()
    cards = s["flashcards"]
    # The missed concept comes first. The third section was never reached in this log, so it gets no card
    # (cards come only from the parts covered: changed at integration, 2026-11-15).
    assert [c["concept"] for c in cards] == ["beta0", "alpha0"]
    assert [c["from_missed"] for c in cards] == [True, False]
    first = cards[0]
    assert list(first) == ["id", "front", "back", "concept", "section_idx", "from_missed", "source", "cue"]
    assert first["source"] == "extractive" and first["section_idx"] == 1 and len(first["id"]) == 32
    assert first["cue"] == {"file": "01_beta0.wav", "url": f"/api/cues/{sid}/01_beta0.wav", "text": "beta0",
                            "kind": "speech"}

    cue_dir = env.data / "cues" / sid
    assert s["cues"] == {
        "dir": str(cue_dir), "count": 2, "manifest": str(cue_dir / "manifest.json"),
        "sleep_command": f".venv/bin/python sleep_cue.py --model <your model from stage_nap.py> --cues {cue_dir} --volume 0.15",
        "note": "Sleep replay is a prototype. The deep-sleep detector has no trained model for you yet and is not validated.",
    }
    manifest = json.loads((cue_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["session_id"] == sid and manifest["cards"][0] == {"concept": "beta0", "file": "01_beta0.wav"}

    audio = env.client.get(first["cue"]["url"])
    assert audio.status_code == 200 and audio.content == b"RIFF-synthetic-beta0"
    assert audio.headers["content-type"] == "audio/wav" and audio.headers["cache-control"] == "no-store"

    again = env.client.post(f"/api/sessions/{sid}/flashcards").json()
    assert [c["id"] for c in again["flashcards"]] == [c["id"] for c in cards]       # not regenerated
    fresh = env.client.post(f"/api/sessions/{sid}/flashcards", json={"regenerate": True}).json()
    assert [c["concept"] for c in fresh["flashcards"]] == ["beta0", "alpha0"]
    assert [c["id"] for c in fresh["flashcards"]] != [c["id"] for c in cards]
    assert sorted(p.name for p in cue_dir.iterdir()) == ["01_beta0.wav", "02_alpha0.wav", "manifest.json"]


def test_cue_command_points_at_the_trained_sleep_model_when_there_is_one(env, monkeypatch, tmp_path):
    model = tmp_path / "nap_model_n2n3.npz"
    model.write_bytes(b"synthetic")
    monkeypatch.setattr(config, "SLEEP_MODEL", model)
    sid = env.reading()["id"]
    env.finish(sid, reading_log(lapses=[(100, 110)]))
    cues = env.client.post(f"/api/sessions/{sid}/flashcards").json()["cues"]
    assert f"--model {model} " in cues["sleep_command"] and "<your model" not in cues["sleep_command"]
    assert cues["note"] == config.SLEEP_NOTE_MODEL
    assert "not been validated" in cues["note"] and "does not detect deep sleep (N3)" in cues["note"]


def test_flashcards_report_a_cue_failure_instead_of_inventing_cues(env):
    sid = env.reading()["id"]
    env.finish(sid, reading_log())
    env.cues.make_cues = lambda cards, out_dir: [{"file": "../../escape.wav", "text": "x", "kind": "speech"}] * len(cards)
    r = env.client.post(f"/api/sessions/{sid}/flashcards")
    assert r.status_code == 500 and "error" in r.json()
    assert env.client.get(f"/api/sessions/{sid}").json()["flashcards"] is None


def test_ask_is_503_without_a_key_and_answers_with_one(env, tmp_path):
    sid = env.reading()["id"]
    r = env.client.post(f"/api/sessions/{sid}/ask", json={"question": "What is beta0?"})
    assert r.status_code == 503
    assert r.json() == {"error": "Questions need an Anthropic API key. Add ANTHROPIC_API_KEY to .env and restart."}
    assert r.json()["error"] == ASK_NEEDS_KEY
    assert env.client.post(f"/api/sessions/{sid}/ask", json={"question": "  "}).status_code == 400
    assert env.client.post(f"/api/sessions/{sid}/ask", json={}).status_code == 400

    env.llm.available = True
    assert env.client.post(f"/api/sessions/{sid}/ask", json={"question": "What is beta0?"}).json() == {
        "answer": "From 3 sections.", "source": "claude"}
    env.llm.answer_question = lambda *a, **k: None          # key present but the call failed
    r = env.client.post(f"/api/sessions/{sid}/ask", json={"question": "Again?"})
    assert r.status_code == 503 and "API key" not in r.json()["error"]


# ---------------------------------------------------------------------------------------------- insights
def test_insights_aggregate_finished_sessions_and_exclude_simulated(tmp_path):
    env = live_env(tmp_path)            # sessions only count as measured with the (fake) headband connected
    with env.client:
        _insights_scenario(env)


def _insights_scenario(env):
    assert env.client.get("/api/insights").json() == {
        "overall": {"sessions": 0, "minutes": 0.0, "focus_pct": None, "lapses_per_min": None, "wpm": None},
        "by_subject": [], "excluded_simulated": 0, "excluded_unmeasured": 0}

    a = env.finish(env.reading(subject="Biology")["id"], reading_log(600, lapses=[(100, 160)], progress=(0, 1)))
    b = env.finish(env.reading(subject="Biology")["id"], reading_log(300, progress=()))
    c = env.finish(env.reading(subject="History")["id"], reading_log(120, lapses=[(10, 70)], source="simulated"))
    env.reading(subject="Biology")                      # never finished: must not count
    assert (a["summary"]["focus_pct"], b["summary"]["focus_pct"], c["simulated"]) == (90.0, 100.0, True)
    assert (a["measured"], b["measured"], c["measured"]) == (True, True, False)
    wpm_a = a["summary"]["reading"]["wpm"]
    assert wpm_a == round(250 / (540 / 60), 1) and b["summary"]["reading"]["wpm"] is None

    out = env.client.get("/api/insights").json()
    assert out["excluded_simulated"] == 1 and out["excluded_unmeasured"] == 0
    assert out["overall"] == {"sessions": 2, "minutes": 15.0, "focus_pct": round((90 * 600 + 100 * 300) / 900, 1),
                              "lapses_per_min": round(1 / 15, 2), "wpm": wpm_a}
    assert len(out["by_subject"]) == 1
    bio = out["by_subject"][0]
    assert bio["subject"] == "Biology" and bio["sessions"] == 2 and bio["focus_pct"] == 93.3
    assert {t["session_id"] for t in bio["trend"]} == {a["id"], b["id"]}
    assert all(t["date"] == a["created_at"][:10] for t in bio["trend"])

    for query in ("?include_simulated=0", "?include_simulated=false", "?include_simulated=yes"):
        assert env.client.get(f"/api/insights{query}").json()["overall"]["sessions"] == 2
    everything = env.client.get("/api/insights?include_simulated=1").json()
    assert everything["excluded_simulated"] == 0 and everything["overall"]["sessions"] == 3
    assert [s["subject"] for s in everything["by_subject"]] == ["Biology", "History"]
    assert everything["by_subject"][1]["focus_pct"] == 50.0


# ---------------------------------------------------------------------------------------------- ids, paths, files
def test_ids_are_validated_before_anything_else(env):
    sid = env.reading()["id"]
    for bad in ("nope", "A" * 32, sid[:-1], sid + "0", sid.upper()):
        for method, path in (("get", f"/api/sessions/{bad}"), ("delete", f"/api/sessions/{bad}"),
                             ("post", f"/api/sessions/{bad}/flashcards")):
            r = getattr(env.client, method)(path)
            assert r.status_code == 400 and list(r.json()) == ["error"], (method, path)
        assert env.client.put(f"/api/sessions/{bad}/log", json={}).status_code == 400
        assert env.client.post(f"/api/sessions/{bad}/finish", json={}).status_code == 400
        assert env.client.post(f"/api/sessions/{bad}/ask", json={"question": "q"}).status_code == 400
        assert env.client.get(f"/api/cues/{bad}/01_x.wav").status_code == 404
        assert env.client.get(f"/api/shots/{bad}/x.jpg").status_code == 404
    unknown = new_id()
    assert env.client.get(f"/api/sessions/{unknown}").status_code == 404
    assert env.client.delete(f"/api/sessions/{unknown}").status_code == 404
    assert env.client.get(f"/api/sessions/{sid}").status_code == 200

    env.wait_ideas(sid)
    with env.client.websocket_connect("/ws") as ws:
        recv(ws)
        ws.send_json({"type": "session", "id": "../../etc", "state": "start"})
        assert recv(ws)["type"] == "error"
        ws.send_json({"type": "session", "id": unknown, "state": "start"})
        assert recv(ws)["type"] == "error"
        ws.send_text("this is not json")
        ws.send_json(["not", "an", "object"])
        ws.send_json({"type": "mystery"})
        ws.send_json({"type": "replay", "action": "start", "file": "lookaway-0.npz"})     # not in results/
        assert recv(ws)["type"] == "error"       # the junk before it was ignored and the socket survived


def test_cue_and_screenshot_paths_cannot_leave_the_sessions_folder(env):
    sid = env.reading()["id"]
    other = env.reading()["id"]
    env.finish(sid, reading_log())
    env.client.post(f"/api/sessions/{sid}/flashcards")
    session_file = env.data / "sessions" / f"{sid}.json"
    assert session_file.is_file()
    for name in ("..%2F..%2Fsessions%2F" + sid + ".json", "%2e%2e%2f%2e%2e%2fsessions%2f" + sid + ".json",
                 "manifest.json", "..", "01_alpha0.wav%00.json", "nope.wav", ".wav", "01_alpha0.wav%0A"):
        r = env.client.get(f"/api/cues/{sid}/{name}")
        assert r.status_code == 404, name
        assert "alpha0" not in r.text
    assert env.client.get(f"/api/cues/{other}/01_alpha0.wav").status_code == 404    # another session's folder
    assert env.client.get(f"/api/cues/{sid}/01_alpha0.wav").status_code == 200

    shot = "data:image/jpeg;base64," + base64.b64encode(JPEG).decode()
    for lapse_id in ("../escape", "a/b", "", "x" * 65, "dot.dot", None):
        r = env.client.post(f"/api/sessions/{sid}/screenshots", json={"lapse_id": lapse_id, "data_url": shot})
        assert r.status_code == 400, lapse_id
    for data_url in ("data:image/png;base64,AAAA", "data:image/jpeg;base64,@@@", "nonsense", None,
                     "data:image/jpeg;base64," + base64.b64encode(b"GIF89a not a jpeg").decode()):
        r = env.client.post(f"/api/sessions/{sid}/screenshots", json={"lapse_id": "lapse-0", "data_url": data_url})
        assert r.status_code == 400, data_url
    big = "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8\xff" + b"0" * config.SCREENSHOT_MAX_BYTES).decode()
    assert env.client.post(f"/api/sessions/{sid}/screenshots", json={"lapse_id": "lapse-0", "data_url": big}).status_code == 413
    assert not (env.data / "shots").exists()

    r = env.client.post(f"/api/sessions/{sid}/screenshots", json={"lapse_id": "lapse-0", "data_url": shot})
    assert r.status_code == 200 and r.json() == {"url": f"/api/shots/{sid}/lapse-0.jpg"}
    got = env.client.get(r.json()["url"])
    assert got.content == JPEG and got.headers["content-type"] == "image/jpeg"
    assert env.client.get(f"/api/shots/{sid}/..%2F..%2Fsessions%2F{sid}.json").status_code == 404
    assert env.client.get(f"/api/shots/{other}/lapse-0.jpg").status_code == 404
    assert env.client.post(f"/api/sessions/{new_id()}/screenshots",
                           json={"lapse_id": "lapse-0", "data_url": shot}).status_code == 404


def test_a_screenshot_url_in_the_log_must_point_at_this_sessions_own_shots(env):
    sid = env.reading()["id"]
    log = reading_log(lapses=[(10, 12), (20, 22)])
    log["lapses"][0]["screenshot_url"] = f"/api/shots/{sid}/lapse-0.jpg"
    log["lapses"][1]["screenshot_url"] = "https://example.invalid/tracker.jpg"
    missed = env.finish(sid, log)["summary"]["missed"]
    assert [m["screenshot_url"] for m in missed] == [f"/api/shots/{sid}/lapse-0.jpg", None]


def test_delete_removes_that_session_only(env):
    keep, drop = env.reading()["id"], env.reading()["id"]
    for sid in (keep, drop):
        env.finish(sid, reading_log())
        env.client.post(f"/api/sessions/{sid}/flashcards")
        env.client.post(f"/api/sessions/{sid}/screenshots",
                        json={"lapse_id": "l1", "data_url": "data:image/jpeg;base64," + base64.b64encode(JPEG).decode()})
    assert env.client.delete(f"/api/sessions/{drop}").json() == {"ok": True}
    assert env.client.get(f"/api/sessions/{drop}").status_code == 404
    assert not (env.data / "cues" / drop).exists() and not (env.data / "shots" / drop).exists()
    assert [s["id"] for s in env.client.get("/api/sessions").json()] == [keep]
    assert (env.data / "cues" / keep / "manifest.json").is_file() and (env.data / "shots" / keep / "l1.jpg").is_file()


# ---------------------------------------------------------------------------------------------- static, origin
def test_static_files_are_served_with_no_store(tmp_path):
    web = tmp_path / "web"
    for folder in ("css", "js/lib", "vendor/webgazer"):
        (web / folder).mkdir(parents=True)
    (web / "index.html").write_text("<!doctype html><title>Study Focus</title>", encoding="utf-8")
    (web / "css" / "base.css").write_text(":root{}", encoding="utf-8")
    (web / "js" / "main.js").write_text("export {};", encoding="utf-8")
    (web / "js" / "lib" / "dom.js").write_text("export {};", encoding="utf-8")
    (web / "vendor" / "webgazer" / "webgazer.js").write_text("// stub", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("outside the web root", encoding="utf-8")
    env = Env(tmp_path, deps={"web_dir": web})

    for path, kind in (("/", "text/html"), ("/css/base.css", "text/css"), ("/js/main.js", "javascript"),
                       ("/js/lib/dom.js", "javascript"), ("/vendor/webgazer/webgazer.js", "javascript")):
        r = env.client.get(path)
        assert r.status_code == 200, path
        assert r.headers["cache-control"] == "no-store", path
        assert kind in r.headers["content-type"], path
    etag = env.client.get("/js/main.js").headers.get("etag")
    conditional = env.client.get("/js/main.js", headers={"if-none-match": etag} if etag else {})
    assert conditional.headers["cache-control"] == "no-store"

    for path in ("/js/..%2F..%2Fsecret.txt", "/js/%2e%2e/%2e%2e/secret.txt", "/css/../../secret.txt", "/js/missing.js",
                 "/index.html", "/app/server.py"):
        r = env.client.get(path)
        assert r.status_code == 404, path
        assert "outside the web root" not in r.text


def test_missing_front_page_is_reported_plainly(tmp_path):
    env = Env(tmp_path, deps={"web_dir": tmp_path / "no-web-here"})
    r = env.client.get("/")
    assert r.status_code == 503 and "index.html" in r.json()["error"]
    assert env.client.get("/api/status").status_code == 200


def test_requests_from_other_sites_are_refused(env):
    assert env.client.get("/api/status", headers={"host": "evil.example"}).status_code == 403
    assert env.client.get("/api/status", headers={"host": "127.0.0.1:8770"}).status_code == 200
    assert env.client.get("/api/status", headers={"host": "localhost:8770"}).status_code == 200
    r = env.client.post("/api/sessions", headers={"origin": "https://evil.example"},
                        json={"mode": "reading", "subject": "x", "title": "y", "text": "stolen write"})
    assert r.status_code == 403 and "error" in r.json()
    assert env.client.get("/api/sessions").json() == []
    ok = env.client.post("/api/sessions", headers={"origin": "http://127.0.0.1:8770"},
                         json={"mode": "reading", "subject": "x", "title": "y", "text": "own window"})
    assert ok.status_code == 201
    assert "access-control-allow-origin" not in {k.lower() for k in ok.headers}        # no CORS
    with pytest.raises(WebSocketDisconnect):
        with env.client.websocket_connect("/ws", headers={"origin": "https://evil.example"}):
            pass
    with env.client.websocket_connect("/ws", headers={"origin": "http://localhost:8770"}) as ws:
        assert recv(ws)["type"] == "status"


# ---------------------------------------------------------------------------------------------- hub
def test_hub_survives_a_client_that_dies_mid_broadcast():
    class Socket:
        def __init__(self, broken=False):
            self.broken, self.sent = broken, []

        async def send_text(self, text):
            if self.broken:
                raise RuntimeError("client went away")
            self.sent.append(json.loads(text))

    async def scenario():
        hub = Hub()
        good, dead, late = Socket(), Socket(broken=True), Socket()
        hub.add(dead), hub.add(good), hub.add(late)
        worker = threading.Thread(target=lambda: [hub.publish({"type": "signal", "n": n}) for n in range(3)])
        worker.start()                                  # published from a non-async thread
        worker.join()
        hub.publish({"type": "bad", "value": float("nan")})     # not valid JSON for a browser: dropped
        for _ in range(100):
            if len(good.sent) == 3 and len(late.sent) == 3:
                break
            await asyncio.sleep(0.01)
        return hub, good, late

    hub, good, late = asyncio.run(scenario())
    assert good.sent == late.sent == [{"type": "signal", "n": 0}, {"type": "signal", "n": 1}, {"type": "signal", "n": 2}]
    assert hub.count() == 2                             # the dead client was dropped, the others kept
    hub.publish({"type": "signal", "n": 9})             # its loop is closed now: no error, clients dropped
    assert hub.count() == 0


def test_status_is_rebroadcast_when_the_ai_mode_changes(env):
    with env.client.websocket_connect("/ws") as ws:
        assert recv(ws)["ai"]["mode"] == "extractive"
        env.core.broadcast_status(force=True)
        assert recv(ws, "status")["ai"]["mode"] == "extractive"
        env.llm.available = True
        env.core.broadcast_status()                     # changed since the last broadcast
        assert recv(ws, "status")["ai"] == {"available": True, "model": config.MODEL, "mode": "claude",
                                            "detail": "Fake key present"}


# ---------------------------------------------------------------------------------------------- measured or not
def test_a_session_with_no_headband_and_no_lapses_is_not_measured_and_not_100_percent(env):
    sid = env.reading()["id"]
    env.wait_ideas(sid)
    with env.client.websocket_connect("/ws") as ws:
        assert recv(ws)["source"] == "simulated"
        ws.send_json({"type": "session", "id": sid, "state": "start"})
        ws.send_json({"type": "session", "id": "not-an-id", "state": "start"})
        assert recv(ws)["type"] == "error"          # so the "start" before it has been handled
        started = env.client.get(f"/api/sessions/{sid}").json()
        assert [e["source"] for e in started["source_log"]] == ["simulated"]
        assert abs(started["source_log"][0]["wall_ms"] - time.time() * 1000) < 5000
        assert started["measured"] is None and started["measure_note"] is None

        s = env.finish(sid, reading_log(duration_s=600, lapses=[]))
    m = s["summary"]
    assert m["lapse_count"] == 0 and m["duration_s"] == 600.0
    assert m["focus_pct"] is None and m["lapses_per_min"] is None       # not 100 % and not 0 per minute
    assert m["reading"]["words_read"] == 250                             # what was read is still reported
    assert m["measured"] is False and s["measured"] is False
    assert s["measure_note"] == "No headband was connected: focus was not measured."
    assert s["simulated"] is False and m["simulated"] is False          # nothing was simulated either
    assert [e["source"] for e in s["source_log"]] == ["simulated", "simulated"]     # at start and at finish

    assert env.client.get("/api/sessions").json()[0]["measured"] is False
    out = env.client.get("/api/insights").json()
    assert out["overall"]["sessions"] == 0 and out["by_subject"] == []
    assert out["excluded_unmeasured"] == 1 and out["excluded_simulated"] == 0
    everything = env.client.get("/api/insights?include_simulated=1").json()
    assert everything["excluded_unmeasured"] == 0 and everything["overall"]["sessions"] == 1
    assert everything["overall"]["focus_pct"] is None                   # still no invented focus figure


def test_a_session_whose_start_message_never_arrived_is_still_not_measured(env):
    sid = env.reading()["id"]           # no WebSocket at all: only the sample taken at finish
    s = env.finish(sid, reading_log(lapses=[]))
    assert [e["source"] for e in s["source_log"]] == ["simulated"]
    assert s["measured"] is False and s["summary"]["focus_pct"] is None
    assert s["measure_note"] == metrics.NOTE_NO_HEADBAND


def test_simulated_lapses_without_a_headband_keep_their_numbers_and_say_so(env):
    sid = env.reading()["id"]
    s = env.finish(sid, reading_log(duration_s=600, lapses=[(100, 160)], source="simulated"))
    assert s["simulated"] is True and s["measured"] is False
    assert s["measure_note"] == "Lapses in this session were simulated; it is left out of long-term insights."
    assert s["summary"]["focus_pct"] == 90.0 and s["summary"]["lapses_per_min"] == 0.1
    out = env.client.get("/api/insights").json()
    assert out["excluded_simulated"] == 1 and out["excluded_unmeasured"] == 0   # counted once, as simulated


def test_a_replay_that_fired_nothing_still_marks_the_session(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "lookaway-1.npz").write_bytes(b"synthetic")

    def factory(on_message):            # a recording with no detector events in it
        return DetectorBridge(on_message, results_dir=results, replay_speed=1.0,
                              replay_loader=lambda path: ([], 0.0, 30.0))

    env = Env(tmp_path, deps={"bridge_factory": factory})
    sid = env.reading()["id"]
    env.wait_ideas(sid)
    with env.client.websocket_connect("/ws") as ws:
        recv(ws)
        ws.send_json({"type": "session", "id": sid, "state": "start"})
        ws.send_json({"type": "replay", "action": "start", "file": "lookaway-1.npz"})
        while not (st := recv(ws, "status"))["source"] == "replay" or not st["detector_ready"]:
            pass
        during = env.client.get(f"/api/sessions/{sid}").json()
        assert [e["source"] for e in during["source_log"]] == ["simulated", "replay"]
        assert during["simulated"] is True and during["server_events"] == []
        s = env.finish(sid, reading_log(lapses=[]))
        ws.send_json({"type": "replay", "action": "stop"})
    assert s["server_events"] == [] and s["summary"]["lapse_count"] == 0
    assert s["simulated"] is True and s["summary"]["simulated"] is True
    assert s["measured"] is False and s["summary"]["focus_pct"] is None
    assert s["measure_note"] == ("Lapses in this session came from a replayed recording, not from you; "
                                 "it is left out of long-term insights.")
    assert [e["source"] for e in s["source_log"]] == ["simulated", "replay", "replay"]
    assert env.client.get("/api/insights").json()["excluded_simulated"] == 1
    env.core.shutdown()


def test_a_headband_that_drops_mid_session_makes_it_partly_measured(tmp_path):
    env = live_env(tmp_path)
    with env.client:
        sid = env.reading()["id"]
        env.wait_ideas(sid)
        with env.client.websocket_connect("/ws") as ws:
            assert recv(ws)["source"] == "live"
            ws.send_json({"type": "session", "id": sid, "state": "start"})
            ws.send_json({"type": "session", "id": "not-an-id", "state": "start"})
            assert recv(ws)["type"] == "error"
            # a status message that does not change the source adds nothing to the log
            FakeRunner.last.on_status({"connected": True, "detail": "Headband connected, settling", "ready": False})
            assert recv(ws, "status")["source"] == "live"
            FakeRunner.last.on_status({"connected": False, "detail": "No headband stream found", "ready": False})
            assert recv(ws, "status")["source"] == "simulated"
            s = env.finish(sid, reading_log(duration_s=600, lapses=[(100, 160)]))
        assert [e["source"] for e in s["source_log"]] == ["live", "simulated", "simulated"]
        assert s["simulated"] is False and s["measured"] is False and s["summary"]["measured"] is False
        assert s["measure_note"] == ("The headband was connected for only part of this session; "
                                     "it is left out of long-term insights.")
        assert s["summary"]["focus_pct"] == 90.0        # the figure is kept; the session is labelled and left out
        out = env.client.get("/api/insights").json()
        assert out["overall"]["sessions"] == 0 and out["excluded_unmeasured"] == 1 and out["excluded_simulated"] == 0

        # a status change after the session is finished is not written into it
        FakeRunner.last.on_status({"connected": True, "detail": "Headband connected (fake)", "ready": True})
        assert len(env.client.get(f"/api/sessions/{sid}").json()["source_log"]) == 3


def test_sessions_saved_before_the_measured_flag_are_listed_as_unknown_and_still_counted(env):
    sections = env.content.split_sections(PASSAGE)
    old = env.core.store.create("reading", "Biology", "Saved earlier", "extractive", {"sections": sections},
                                key_ideas_status="done")

    def as_saved_before(s):
        for key in ("source_log", "measured", "measure_note"):
            del s[key]
        s["ended_at"] = "2026-11-15T02:00:00"
        s["summary"] = {"duration_s": 600.0, "lapse_count": 1, "lapse_seconds": 60.0, "focus_pct": 90.0,
                        "lapses_per_min": 0.1, "reading": {"words_read": 250, "wpm": 27.8}, "video": None,
                        "missed": [], "simulated": False, "generated_at": "2026-11-15T02:00:00"}
    env.core.store.update(old["id"], as_saved_before)
    assert env.client.get("/api/sessions").json()[0]["measured"] is None
    out = env.client.get("/api/insights").json()
    assert out["overall"]["sessions"] == 1 and out["excluded_unmeasured"] == 0 and out["excluded_simulated"] == 0


# ---------------------------------------------------------------------------------------------- connection count
def test_status_counts_open_connections_and_tells_the_other_windows(env):
    assert env.client.get("/api/status").json()["clients"] == 0
    with env.client.websocket_connect("/ws") as first:
        assert recv(first)["clients"] == 1
        assert env.client.get("/api/status").json()["clients"] == 1
        with env.client.websocket_connect("/ws") as second:
            assert recv(second)["clients"] == 2             # its own first message
            assert recv(first, "status")["clients"] == 2    # the window that was already open is told
        assert recv(first, "status")["clients"] == 1        # and told again when the other one closes
    deadline = time.monotonic() + 5
    while env.client.get("/api/status").json()["clients"] != 0:
        assert time.monotonic() < deadline
        time.sleep(0.01)


# ---------------------------------------------------------------------------------------------- flashcards: waiting
def _hold_key_ideas_for(env, word):
    """Make the fake key-idea call block on sections starting with `word` until the returned event is set."""
    release, plain = threading.Event(), env.llm.key_ideas

    def key_ideas(section_text, subject, title, cache_dir):
        if section_text.startswith(word):
            release.wait(20)
        return plain(section_text, subject, title, cache_dir)

    env.llm.key_ideas = key_ideas
    return release


def test_flashcards_do_not_wait_for_key_ideas_of_sections_that_were_not_covered(env):
    release = _hold_key_ideas_for(env, "gamma")         # the third section: never reached in this log
    try:
        sid = env.reading()["id"]
        env.finish(sid, reading_log(lapses=[(100, 110)], progress=(0, 1)))
        began = time.monotonic()
        r = env.client.post(f"/api/sessions/{sid}/flashcards")
        assert time.monotonic() - began < 5.0           # it used to wait up to 120 s for every section
        assert r.status_code == 200
        assert [c["concept"] for c in r.json()["flashcards"]] == ["beta0", "alpha0"]
        assert r.json()["key_ideas_status"] == "partial"        # the uncovered section is still being written
    finally:
        release.set()
    assert env.wait_ideas(sid)["key_ideas_status"] == "done"


def test_flashcards_wait_is_capped_and_does_not_hold_up_another_session(env, monkeypatch):
    monkeypatch.setattr(server_module, "FLASHCARD_IDEAS_WAIT_S", 1.0)
    release = _hold_key_ideas_for(env, "alpha")         # a covered section whose ideas never arrive in time
    try:
        slow = env.reading()["id"]
        env.finish(slow, reading_log(progress=(0,)))
        quick = env.reading(text="\n\n".join([" ".join(f"delta{i}" for i in range(40)),
                                              " ".join(f"epsilon{i}" for i in range(40))]))["id"]
        env.finish(quick, reading_log(progress=(0, 1)))
        env.wait_ideas(quick)

        pool = ThreadPoolExecutor(max_workers=1)
        began = time.monotonic()
        waiting = pool.submit(env.client.post, f"/api/sessions/{slow}/flashcards")
        other = env.client.post(f"/api/sessions/{quick}/flashcards")       # made while the first one waits
        assert other.status_code == 200 and not waiting.done()
        assert [c["concept"] for c in other.json()["flashcards"]] == ["delta0", "epsilon0"]

        r = waiting.result(10)
        assert 0.9 < time.monotonic() - began < 5.0     # gave up at the cap, not after 120 s
        # The fake builds cards from key ideas only and none exist yet for the covered section, so there are
        # no cards: nothing is taken from the sections that were not covered.
        assert r.status_code == 200 and r.json()["flashcards"] == []
        pool.shutdown(wait=False)
    finally:
        release.set()
    env.wait_ideas(slow)


def test_flashcards_are_refused_when_nothing_was_covered(env):
    sid = env.reading()["id"]
    env.wait_ideas(sid)
    env.finish(sid, reading_log(lapses=[], progress=()))        # ended at once: no section finished, no lapse
    r = env.client.post(f"/api/sessions/{sid}/flashcards")
    assert r.status_code == 409
    assert r.json() == {"error": "Nothing was covered in this session, so there are no flashcards to make."}
    assert r.json()["error"] == NOTHING_COVERED
    assert env.client.post(f"/api/sessions/{sid}/flashcards", json={"regenerate": True}).status_code == 409
    s = env.client.get(f"/api/sessions/{sid}").json()
    assert s["flashcards"] is None and s["cues"] is None
    assert not (env.data / "cues" / sid).exists()               # no cue files from material never reached

    # a section index the browser made up is not a covered section either
    other = env.reading()["id"]
    env.finish(other, reading_log(lapses=[], progress=(7,)))
    assert env.client.post(f"/api/sessions/{other}/flashcards").status_code == 409


# ---------------------------------------------------------------------------------------------- jaw-clench hold
def test_video_session_has_a_concept_timeline_for_the_clench_popup(env):
    c = env.video()["content"]
    items = c["concepts"]["items"]
    assert c["concepts"]["status"] == "done" and c["concepts"]["source"] == "extractive"   # no key in this env
    assert items and all(i["label"] for i in items)
    assert all(a["end_s"] == b["start_s"] for a, b in zip(items, items[1:]))              # contiguous windows
    assert env.video(NO_CAPTIONS_URL)["content"]["concepts"]["items"] == []


def test_bridge_forwards_live_clench_events_and_simulates_a_hold():
    seen = []
    bridge = DetectorBridge(seen.append, live_module=FakeLive)
    assert bridge.sim_clench() is True and bridge.sim_clench() is False      # one simulated hold at a time
    deadline = time.time() + 6
    while not any(m["type"] == "clench" for m in seen) and time.time() < deadline:
        time.sleep(0.05)
    prog = [m for m in seen if m["type"] == "clench_progress"]
    (clench,) = [m for m in seen if m["type"] == "clench"]
    assert clench["source"] == "simulated" and clench["wall_ms"] - clench["onset_wall_ms"] == pytest.approx(3000, abs=400)
    assert prog[0]["progress"] < prog[-2]["progress"] <= 1.0 and all(m["source"] == "simulated" for m in prog)
    time.sleep(0.2)
    assert bridge.sim_clench() is True           # the next simulated hold is allowed once the first is done

    seen.clear()
    bridge.start()                               # fake headband: source live, so simulation is refused
    assert bridge.sim_clench() is False
    FakeRunner.last.on_event({"type": "clench_progress", "t": 998.0, "progress": 0.5, "cancelled": False})
    FakeRunner.last.on_event({"type": "clench", "t": 999.0, "onset_t": 996.0})
    p, c = [m for m in seen if m["type"] in ("clench_progress", "clench")]
    assert p == {"type": "clench_progress", "source": "live", "wall_ms": p["wall_ms"], "progress": 0.5, "cancelled": False}
    assert c["source"] == "live" and c["t"] == 999.0 and c["onset_t"] == 996.0
    assert c["wall_ms"] - c["onset_wall_ms"] == pytest.approx(3000, abs=50)
    bridge.stop()


def test_clench_events_are_logged_and_clench_lapses_reach_the_summary(tmp_path):
    env = live_env(tmp_path)
    with env.client:
        sid = env.video()["id"]
        env.wait_ideas(sid)
        with env.client.websocket_connect("/ws") as ws:
            recv(ws)
            ws.send_json({"type": "session", "id": sid, "state": "start"})
            time.sleep(0.1)
            FakeRunner.last.on_event({"type": "clench_progress", "t": 998.0, "progress": 0.4, "cancelled": False})
            assert recv(ws, "clench_progress")["progress"] == 0.4
            FakeRunner.last.on_event({"type": "clench", "t": 999.0, "onset_t": 996.0})
            assert recv(ws, "clench")["source"] == "live"
        events = env.client.get(f"/api/sessions/{sid}").json()["server_events"]
        assert [e["type"] for e in events] == ["clench"]          # progress is broadcast only, never stored
        now = int(time.time() * 1000)
        lapse = {"id": "c1", "kind": "clench", "source": "live", "onset_ms": now - 30000, "resume_ms": now - 3000,
                 "duration_s": 27.0, "end_method": "clench", "position": {"t_start": 40.0, "t_end": 67.0},
                 "screenshot_url": None, "chosen_concept": "Variation under domestication"}
        log = {"started_ms": now - 120000, "ended_ms": now, "lapses": [lapse], "progress": [],
               "gaze": {"mode": "none", "calibrated": False}, "video": {"duration_s": 180.0, "watched_s": 120.0}}
        done = env.client.post(f"/api/sessions/{sid}/finish", json=log).json()
        (item,) = done["summary"]["missed"]
        assert item["kind"] == "clench" and item["end_method"] == "clench"
        assert item["chosen_concept"] == "Variation under domestication" and item["label"] == "00:40-01:07"


# ---------------------------------------------------------------------------------------------- study plans
def _tiny_pdf(text):
    """A one-page PDF whose text layer is `text` (synthetic, for tests)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1) + b"".join(b"%010d 00000 n \n" % x for x in offs)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return out


def test_pdf_text_reads_the_text_layer_and_says_when_there_is_none(env):
    import base64 as b64
    url = "data:application/pdf;base64," + b64.b64encode(_tiny_pdf("Natural selection preserves variation")).decode()
    r = env.client.post("/api/pdf_text", json={"name": "Chapter 4.pdf", "data_url": url}).json()
    assert r["title"] == "Chapter 4" and "Natural selection preserves variation" in r["text"] and r["pages"] == 1
    assert r["note"] and "scanned" in r["note"]           # under 200 characters: flagged as probably scanned
    bad = env.client.post("/api/pdf_text", json={"name": "x.pdf", "data_url": "data:application/pdf;base64,aGVsbG8="})
    assert bad.status_code == 400 and "not a PDF" in bad.json()["error"]


def test_a_study_plan_links_its_resources_and_makes_one_deck_from_all(env, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SLEEP_MODEL", tmp_path / "no_such_model.npz")
    r = env.client.post("/api/studies", json={"subject": "Evolution", "resources": [
        {"kind": "reading", "title": "Notes", "text": PASSAGE}, {"kind": "video", "youtube_url": VIDEO_URL}]})
    assert r.status_code == 201, r.text
    study = r.json()
    a, b = study["sessions"]
    assert a["mode"] == "reading" and b["mode"] == "video" and a["subject"] == b["subject"] == "Evolution"
    assert a["study"]["session_ids"] == [a["id"], b["id"]] and [a["study"]["order"], b["study"]["order"]] == [0, 1]
    assert a["study"]["titles"][0] == "Notes" and b["study"]["count"] == 2
    for s in (a, b):
        env.wait_ideas(s["id"])
    env.finish(a["id"], reading_log(lapses=[(100, 110)]))
    now = T0 + 120000
    env.finish(b["id"], {"started_ms": T0, "ended_ms": now, "lapses": [], "progress": [{"section_idx": 0, "completed_ms": T0 + 60000}],
                         "gaze": {"mode": "none", "calibrated": False}, "video": {"duration_s": 180.0, "watched_s": 120.0}})
    deck = env.client.post(f"/api/sessions/{b['id']}/flashcards").json()["flashcards"]
    concepts = {c["concept"] for c in deck}
    assert {"beta0", "alpha0"} <= concepts                      # from the reading resource
    assert len(concepts) > 2                                    # and something from the video resource
    assert all(c["section_idx"] is None for c in deck)          # a plan's cards carry no section index

    bad = env.client.post("/api/studies", json={"subject": "", "resources": [{"kind": "video", "youtube_url": VIDEO_URL}]})
    assert bad.status_code == 400
    bad = env.client.post("/api/studies", json={"subject": "X", "resources": [{"kind": "video", "youtube_url": "https://example.com"}]})
    assert bad.status_code == 400 and "YouTube" in bad.json()["error"]
