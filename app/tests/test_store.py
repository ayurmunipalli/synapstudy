"""Session store: atomic writes, locking, id and file-name validation. Synthetic data, temp dirs only."""
import json
import os
import threading

import pytest

from app.store import CUE_FILE_RE, SHOT_FILE_RE, Store, new_id, valid_id


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "data")


def make(store, **kw):
    args = dict(mode="reading", subject="Biology", title="Test passage", ai_mode="extractive",
                content={"sections": [{"idx": 0, "text": "one two three", "word_count": 3}]})
    args.update(kw)
    return store.create(**args)


def test_create_writes_the_documented_session_object(store):
    s = make(store)
    assert valid_id(s["id"])
    assert list(s) == ["id", "created_at", "mode", "subject", "title", "ai_mode", "content", "key_ideas",
                       "key_ideas_status", "key_ideas_error", "log", "server_events", "summary", "flashcards",
                       "cues", "simulated", "source_counts", "ended_at", "source_log", "measured", "measure_note"]
    assert s["source_log"] == [] and s["measured"] is None and s["measure_note"] is None
    assert s["key_ideas"] == [] and s["key_ideas_status"] == "pending" and s["key_ideas_error"] is None
    assert s["simulated"] is False and s["source_counts"] == {"live": 0, "replay": 0, "simulated": 0}
    assert s["log"] is None and s["summary"] is None and s["ended_at"] is None
    assert len(s["created_at"]) == 19 and s["created_at"][10] == "T"
    on_disk = json.loads((store.sessions / f"{s['id']}.json").read_text(encoding="utf-8"))
    assert on_disk == s
    assert store.load(s["id"]) == s


@pytest.mark.parametrize("bad", ["", "abc", "../etc/passwd", "A" * 32, "g" * 32, "0" * 31, "0" * 33,
                                 "0" * 31 + "/", "0" * 32 + "\n", None, 12])
def test_malformed_ids_never_reach_the_filesystem(store, bad):
    assert not valid_id(bad)
    assert store.load(bad) is None
    assert store.update(bad, lambda s: None) is None
    assert store.delete(bad) is False
    assert store.cue_file(bad, "01_x.wav") is None and store.shot_file(bad, "x.jpg") is None
    with pytest.raises(ValueError):
        store.session_path(bad)
    with pytest.raises(ValueError):
        store.cue_dir(bad)
    assert not store.root.exists()      # nothing was created along the way


def test_load_of_unknown_id_is_none(store):
    assert store.load(new_id()) is None


def test_update_is_read_modify_write_under_the_lock(store):
    sid = make(store)["id"]
    store.update(sid, lambda s: s.update(counter=0))

    def bump():
        for _ in range(25):
            store.update(sid, lambda s: s.update(counter=s["counter"] + 1))

    threads = [threading.Thread(target=bump) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert store.load(sid)["counter"] == 200
    assert [p.name for p in store.sessions.iterdir()] == [f"{sid}.json"]      # no temp files left behind


def test_failed_write_leaves_the_old_file_and_no_temp_file(store, monkeypatch):
    s = make(store)
    before = (store.sessions / f"{s['id']}.json").read_bytes()

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        store.update(s["id"], lambda x: x.update(title="changed"))
    monkeypatch.undo()
    assert (store.sessions / f"{s['id']}.json").read_bytes() == before
    assert [p.name for p in store.sessions.iterdir()] == [f"{s['id']}.json"]


def test_list_is_newest_first_and_skips_damaged_or_foreign_files(store):
    a, b, c = make(store, title="a"), make(store, title="b"), make(store, title="c")
    store.update(a["id"], lambda s: s.update(created_at="2026-11-01T10:00:00"))
    store.update(b["id"], lambda s: s.update(created_at="2026-11-03T10:00:00"))
    store.update(c["id"], lambda s: s.update(created_at="2026-11-02T10:00:00"))
    (store.sessions / f"{new_id()}.json").write_text("{not json", encoding="utf-8")
    (store.sessions / "notes.json").write_text("{}", encoding="utf-8")
    (store.sessions / ".tmpfile.tmp").write_text("x", encoding="utf-8")
    assert [s["title"] for s in store.list()] == ["b", "c", "a"]


def test_list_with_no_data_dir_is_empty(store):
    assert store.list() == []


def test_delete_removes_only_that_sessions_files(store):
    a, b = make(store), make(store)
    for s in (a, b):
        store.cue_dir(s["id"]).mkdir(parents=True)
        (store.cue_dir(s["id"]) / "01_x.wav").write_bytes(b"RIFF")
        store.shot_dir(s["id"]).mkdir(parents=True)
        (store.shot_dir(s["id"]) / "l1.jpg").write_bytes(b"\xff\xd8\xff")
    assert store.delete(a["id"]) is True
    assert store.load(a["id"]) is None
    assert not store.cue_dir(a["id"]).exists() and not store.shot_dir(a["id"]).exists()
    assert store.load(b["id"]) is not None
    assert (store.cue_dir(b["id"]) / "01_x.wav").is_file() and (store.shot_dir(b["id"]) / "l1.jpg").is_file()
    assert store.delete(a["id"]) is False


def test_append_event_logs_counts_and_flags_non_live(store):
    sid = make(store)["id"]
    live = {"type": "drift", "kind": "look_away", "source": "live", "acted": True, "wall_ms": 2000.0,
            "onset_wall_ms": 1250.0, "t": 5.0, "onset_t": 4.25, "confidence": None}
    s = store.append_event(sid, live)
    assert s["server_events"] == [{"type": "drift", "kind": "look_away", "source": "live", "wall_ms": 2000.0,
                                   "onset_wall_ms": 1250.0}]
    assert s["source_counts"] == {"live": 1, "replay": 0, "simulated": 0} and s["simulated"] is False

    s = store.append_event(sid, {"type": "resume", "source": "live", "wall_ms": 3000.0, "t": 6.0})
    assert s["server_events"][-1] == {"type": "resume", "kind": None, "source": "live", "wall_ms": 3000.0,
                                      "onset_wall_ms": None}
    assert s["source_counts"]["live"] == 1 and s["simulated"] is False     # resumes are not counted as lapses

    s = store.append_event(sid, {"type": "drift", "kind": "look_away", "source": "simulated", "wall_ms": 4000.0,
                                 "onset_wall_ms": 3250.0})
    assert s["simulated"] is True and s["source_counts"] == {"live": 1, "replay": 0, "simulated": 1}
    s = store.append_event(sid, {"type": "drift", "kind": "look_away", "source": "replay", "wall_ms": 5000.0,
                                 "onset_wall_ms": 4250.0})
    assert s["source_counts"]["replay"] == 1 and len(s["server_events"]) == 4


def test_append_source_records_what_the_server_saw(store):
    sid = make(store)["id"]
    s = store.append_source(sid, "simulated", 1000)
    assert s["source_log"] == [{"source": "simulated", "wall_ms": 1000}] and s["simulated"] is False
    assert store.append_source(sid, "simulated", 2000, only_if_changed=True) is None      # same as the last entry
    assert len(store.append_source(sid, "simulated", 3000)["source_log"]) == 2            # unless asked for anyway
    s = store.append_source(sid, "live", 4000, only_if_changed=True)
    assert [e["source"] for e in s["source_log"]] == ["simulated", "simulated", "live"] and s["simulated"] is False
    s = store.append_source(sid, "replay", 5000, only_if_changed=True)
    assert s["source_log"][-1] == {"source": "replay", "wall_ms": 5000}
    assert s["simulated"] is True                       # a replay ran during the session
    for bad in ("LIVE", "", None, 3, ["live"]):
        assert store.append_source(sid, bad, 6000) is None
    assert len(store.load(sid)["source_log"]) == 4 and store.load(sid)["measured"] is None
    assert store.append_source("not-an-id", "live", 1) is None


def test_append_source_leaves_finished_sessions_alone_and_copes_with_older_files(store):
    sid = make(store)["id"]
    store.update(sid, lambda s: s.pop("source_log"))            # a session file from before the field existed
    assert store.append_source(sid, "live", 1000)["source_log"] == [{"source": "live", "wall_ms": 1000}]
    store.update(sid, lambda s: s.update(ended_at="2026-11-15T02:00:00"))
    assert store.append_source(sid, "replay", 2000) is None
    s = store.load(sid)
    assert len(s["source_log"]) == 1 and s["simulated"] is False


def test_append_event_leaves_finished_sessions_alone(store):
    sid = make(store)["id"]
    store.update(sid, lambda s: s.update(ended_at="2026-11-15T02:00:00"))
    assert store.append_event(sid, {"type": "drift", "kind": "look_away", "source": "simulated",
                                    "wall_ms": 1.0, "onset_wall_ms": 0.0}) is None
    s = store.load(sid)
    assert s["server_events"] == [] and s["simulated"] is False


def test_file_lookup_rejects_traversal_bad_names_and_symlink_escapes(store, tmp_path):
    sid = make(store)["id"]
    folder = store.cue_dir(sid)
    folder.mkdir(parents=True)
    (folder / "01_natural-selection.wav").write_bytes(b"RIFF")
    (folder / "manifest.json").write_text("{}", encoding="utf-8")
    secret = tmp_path / "secret.wav"
    secret.write_bytes(b"nope")
    os.symlink(secret, folder / "02_link.wav")
    (folder / "sub.wav").mkdir()

    assert store.cue_file(sid, "01_natural-selection.wav") == (folder / "01_natural-selection.wav").resolve()
    for bad in ["../secret.wav", "..", "", "manifest.json", "01_x.wav/..", "a/b.wav", "/etc/passwd",
                "01_missing.wav", "02_link.wav", "sub.wav", ".hidden.wav", "x.WAV", "x.wav\n", None, 3]:
        assert store.cue_file(sid, bad) is None, bad
    assert store.shot_file(sid, "l1.jpg") is None       # no such file yet
    store.shot_dir(sid).mkdir(parents=True)
    (store.shot_dir(sid) / "l1.jpg").write_bytes(b"\xff\xd8\xff")
    assert store.shot_file(sid, "l1.jpg") is not None
    assert store.shot_file(sid, "../cues/x.jpg") is None and store.shot_file(sid, "l1.jpeg") is None
    assert CUE_FILE_RE.match("01_natural-selection.wav") and not CUE_FILE_RE.match("../x.wav")
    assert SHOT_FILE_RE.match("abc-DEF_1.jpg") and not SHOT_FILE_RE.match("a.b.jpg")


def test_append_event_keeps_what_the_head_did_when_the_event_says_so(store):
    sid = make(store)["id"]
    s = store.append_event(sid, {"type": "drift", "kind": "look_away", "source": "live", "wall_ms": 2000.0,
                                 "onset_wall_ms": 1750.0, "t": 5.0, "onset_t": 4.75, "confidence": None,
                                 "angle": 23.5, "rot": [1.0, -23.4, 0.5], "peak_angle": 99.0, "net_rot": [9, 9, 9]})
    assert s["server_events"][-1] == {"type": "drift", "kind": "look_away", "source": "live", "wall_ms": 2000.0,
                                      "onset_wall_ms": 1750.0, "angle": 23.5, "rot": [1.0, -23.4, 0.5]}
    s = store.append_event(sid, {"type": "resume", "source": "live", "wall_ms": 5000.0, "t": 8.0,
                                 "net_rot": [2.0, -51.2, 1.0], "peak_angle": 48.0, "angle": 1.0})
    assert s["server_events"][-1] == {"type": "resume", "kind": None, "source": "live", "wall_ms": 5000.0,
                                      "onset_wall_ms": None, "net_rot": [2.0, -51.2, 1.0], "peak_angle": 48.0}
    # null (no motion data) is not stored as a key at all
    s = store.append_event(sid, {"type": "resume", "source": "live", "wall_ms": 6000.0, "net_rot": None,
                                 "peak_angle": None})
    assert s["server_events"][-1] == {"type": "resume", "kind": None, "source": "live", "wall_ms": 6000.0,
                                      "onset_wall_ms": None}


def test_settings_file_round_trip_and_damage(store):
    assert store.read_settings() == {} and not store.settings_path.exists()     # nothing is created by reading
    store.write_settings({"theta": 20})
    assert store.settings_path == store.root / "settings.json"
    assert json.loads(store.settings_path.read_text()) == {"theta": 20} and store.read_settings() == {"theta": 20}
    assert [p.name for p in store.root.iterdir()] == ["settings.json"]          # no temp file left behind
    for damaged in ("not json", "[1, 2]", "", "null"):
        store.settings_path.write_text(damaged)
        assert store.read_settings() == {}
