"""Session numbers and long-term insights. All inputs here are synthetic and exist only in these tests."""
import pytest

from app import metrics

T0 = 1_800_000_000_000      # an arbitrary epoch-ms start


def lapse(onset_s, resume_s, source="live", **kw):
    return {"id": f"l{onset_s}", "kind": "look_away", "source": source, "onset_ms": T0 + onset_s * 1000,
            "resume_ms": None if resume_s is None else T0 + resume_s * 1000, **kw}


SECTIONS = [{"idx": 0, "text": "…", "word_count": 100}, {"idx": 1, "text": "…", "word_count": 150},
            {"idx": 2, "text": "…", "word_count": 50}]


def test_duration():
    assert metrics.duration_s(T0, T0 + 600_000) == 600.0
    assert metrics.duration_s(T0, T0 - 5) == 0.0
    assert metrics.duration_s(None, T0) == 0.0 and metrics.duration_s(T0, None) == 0.0


def test_lapse_seconds_sums_and_clips_to_the_session():
    end = T0 + 600_000
    assert metrics.lapse_seconds([lapse(100, 110), lapse(300, 311.5)], T0, end) == pytest.approx(21.5)
    # starts before the session, ends after it: only the part inside counts
    assert metrics.lapse_seconds([lapse(-5, 4), lapse(595, 620)], T0, end) == pytest.approx(9.0)
    # entirely outside
    assert metrics.lapse_seconds([lapse(-20, -10), lapse(700, 710)], T0, end) == 0.0
    # still open at the end: runs to the end of the session
    assert metrics.lapse_seconds([lapse(590, None)], T0, end) == pytest.approx(10.0)
    # no resume but a duration
    assert metrics.lapse_seconds([lapse(10, None, duration_s=4.0)], T0, end) == pytest.approx(4.0)
    # overlapping lapses are not double-counted, so the total can never exceed the session
    assert metrics.lapse_seconds([lapse(10, 30), lapse(20, 40), lapse(40, 45)], T0, end) == pytest.approx(35.0)
    assert metrics.lapse_seconds([lapse(-10, 700), lapse(0, 600)], T0, end) == pytest.approx(600.0)
    # junk entries are ignored, not guessed at
    assert metrics.lapse_seconds([{"onset_ms": "soon"}, None, 3, lapse(1, 2)], T0, end) == pytest.approx(1.0)
    assert metrics.lapse_seconds([lapse(1, 2)], None, end) == 0.0


def test_focus_and_rate_formulas():
    assert metrics.focus_pct(600.0, 21.5) == pytest.approx(100 * (1 - 21.5 / 600))
    assert metrics.focus_pct(600.0, 0.0) == 100.0
    assert metrics.focus_pct(0.0, 0.0) is None
    assert metrics.lapses_per_min(3, 600.0) == pytest.approx(0.3)
    assert metrics.lapses_per_min(3, 0.0) is None


def test_words_read_counts_each_completed_section_once():
    progress = [{"section_idx": 0, "completed_ms": T0 + 1}, {"section_idx": 1, "completed_ms": T0 + 2},
                {"section_idx": 0, "completed_ms": T0 + 3}, {"section_idx": 9}, {"section_idx": "2"}, "junk"]
    assert metrics.words_read(SECTIONS, progress) == 250
    assert metrics.words_read(SECTIONS, []) == 0 and metrics.words_read(SECTIONS, None) == 0


def test_wpm_excludes_lapse_time_and_is_none_when_not_computable():
    assert metrics.wpm(2400, 600.0, 21.5) == pytest.approx(2400 / ((600 - 21.5) / 60))
    assert metrics.wpm(0, 600.0, 0.0) is None                 # no completed section
    assert metrics.wpm(250, 19.9, 0.0) is None                # under 20 s of reading
    assert metrics.wpm(250, 60.0, 45.0) is None               # 15 s of reading once lapses are removed
    assert metrics.wpm(250, 20.0, 0.0) == pytest.approx(750.0)


def test_session_metrics_reading_matches_the_spec_example_numbers():
    log = {"started_ms": T0, "ended_ms": T0 + 600_000,
           "lapses": [lapse(100, 110), lapse(300, 311.5), lapse(500, 500)],
           "progress": [{"section_idx": 0, "completed_ms": T0 + 200_000},
                        {"section_idx": 1, "completed_ms": T0 + 400_000}]}
    m = metrics.session_metrics(log, SECTIONS, "reading")
    assert list(m) == ["duration_s", "lapse_count", "lapse_seconds", "focus_pct", "lapses_per_min", "reading", "video"]
    assert m["duration_s"] == 600.0 and m["lapse_count"] == 3 and m["lapse_seconds"] == 21.5
    assert m["focus_pct"] == 96.4 and m["lapses_per_min"] == 0.3
    assert m["reading"] == {"words_read": 250, "wpm": round(250 / ((600 - 21.5) / 60), 1)}
    assert m["video"] is None


def test_session_metrics_video_and_empty_session():
    log = {"started_ms": T0, "ended_ms": T0 + 120_000, "lapses": [lapse(30, 42)],
           "video": {"duration_s": 300.04, "watched_s": 118.26}}
    m = metrics.session_metrics(log, [], "video")
    assert m["reading"] is None and m["video"] == {"duration_s": 300.0, "watched_s": 118.3}
    assert m["focus_pct"] == 90.0 and m["lapses_per_min"] == 0.5

    empty = metrics.session_metrics({"started_ms": T0, "ended_ms": T0, "lapses": [], "progress": []}, SECTIONS, "reading")
    assert empty["duration_s"] == 0.0 and empty["focus_pct"] is None and empty["lapses_per_min"] is None
    assert empty["reading"] == {"words_read": 0, "wpm": None}


def test_is_simulated_comes_from_any_non_live_event_and_cannot_be_talked_down():
    live = {"type": "drift", "source": "live"}
    assert metrics.is_simulated([], []) is False
    assert metrics.is_simulated([live, {"type": "resume", "source": "live"}], [lapse(1, 2)]) is False
    assert metrics.is_simulated([live, {"type": "drift", "source": "simulated"}]) is True
    assert metrics.is_simulated([{"type": "drift", "source": "replay"}]) is True
    assert metrics.is_simulated([{"type": "drift"}]) is True                      # unknown source is not live
    # the server log says simulated; client lapses all claiming "live" (or no lapses) change nothing
    assert metrics.is_simulated([{"type": "drift", "source": "simulated"}], [lapse(1, 2, source="live")]) is True
    # a lapse the client itself reports as not live also sets it
    assert metrics.is_simulated([live], [lapse(1, 2, source="simulated")]) is True


def test_fmt_clock():
    assert metrics.fmt_clock(102) == "01:42" and metrics.fmt_clock(0) == "00:00"
    assert metrics.fmt_clock(3725.9) == "1:02:05" and metrics.fmt_clock(-3) == "00:00"


def finished(sid, subject, created, duration, focus, lapses, wpm, simulated=False, mode="reading", measured=True):
    return {"id": sid, "title": f"t-{sid}", "subject": subject, "mode": mode, "created_at": created,
            "ended_at": created, "simulated": simulated, "measured": measured,
            "summary": {"duration_s": duration, "lapse_count": lapses, "lapse_seconds": 0, "focus_pct": focus,
                        "lapses_per_min": None, "reading": {"words_read": 0, "wpm": wpm} if mode == "reading" else None,
                        "video": None, "missed": [], "simulated": simulated, "measured": measured,
                        "generated_at": created}}


def entries(*sources):
    return [{"source": s, "wall_ms": T0 + 1000 * n} for n, s in enumerate(sources)]


def test_measurement_is_true_only_for_a_live_unsimulated_session():
    assert metrics.measurement(entries("live"), False, 0) == (True, None)
    assert metrics.measurement(entries("live", "live"), False, 3) == (True, None)
    for log in ([], None, entries("simulated"), entries("replay"), entries("live", "simulated"),
                entries("simulated", "live"), [{"wall_ms": 1}], ["live"]):
        assert metrics.measurement(log, False, 0)[0] is False, log
    assert metrics.measurement(entries("live"), True, 1)[0] is False        # flagged simulated: never measured


def test_measurement_note_says_why_first_match_first():
    replay = "Lapses in this session came from a replayed recording, not from you; it is left out of long-term insights."
    partial = "The headband was connected for only part of this session; it is left out of long-term insights."
    simulated = "Lapses in this session were simulated; it is left out of long-term insights."
    nothing = "No headband was connected: focus was not measured."
    assert (metrics.NOTE_REPLAY, metrics.NOTE_PARTIAL, metrics.NOTE_SIMULATED, metrics.NOTE_NO_HEADBAND) == (
        replay, partial, simulated, nothing)
    note = lambda *a, **k: metrics.measurement(*a, **k)[1]      # noqa: E731
    # any replay entry wins, whatever else happened
    assert note(entries("replay"), True, 0) == replay
    assert note(entries("live", "replay", "live"), True, 2) == replay
    assert note(entries("simulated", "replay", "simulated"), True, 1) == replay
    # live for part of the session
    assert note(entries("live", "simulated"), False, 0) == partial
    assert note(entries("simulated", "live"), True, 4) == partial
    # never live, lapses logged
    assert note(entries("simulated"), True, 2) == simulated
    assert note([], True, 1) == simulated
    # never live (or nothing recorded), no lapses
    assert note(entries("simulated", "simulated"), False, 0) == nothing
    assert note([], False, 0) == nothing and note(None, False, 0) == nothing
    # Not spelled out in the contract: live throughout, yet a lapse or event was not live. The note follows
    # what that lapse or event was.
    assert note(entries("live"), True, 1) == simulated
    assert note(entries("live"), True, 1, replay_seen=True) == replay
    assert note(entries("simulated"), True, 1, replay_seen=True) == replay


def test_ever_live():
    assert metrics.ever_live(entries("simulated", "live")) is True
    assert metrics.ever_live(entries("simulated", "replay")) is False and metrics.ever_live([]) is False
    assert metrics.ever_live(None) is False and metrics.ever_live(["live", None]) is False


def test_list_row():
    row = metrics.list_row(finished("a" * 32, "Biology", "2026-11-15T01:50:00", 600.0, 96.4, 3, 251.0))
    assert row == {"id": "a" * 32, "title": "t-" + "a" * 32, "subject": "Biology", "mode": "reading",
                   "created_at": "2026-11-15T01:50:00", "ended_at": "2026-11-15T01:50:00", "duration_s": 600.0,
                   "focus_pct": 96.4, "lapse_count": 3, "wpm": 251.0, "simulated": False, "measured": True}
    unfinished = {"id": "b" * 32, "title": "x", "subject": "y", "mode": "video", "created_at": "2026-11-15T02:00:00",
                  "ended_at": None, "summary": None, "simulated": False, "measured": None}
    row = metrics.list_row(unfinished)
    assert row["duration_s"] is None and row["focus_pct"] is None and row["lapse_count"] is None and row["wpm"] is None
    assert row["measured"] is None
    unmeasured = finished("c" * 32, "Biology", "2026-11-15T01:50:00", 600.0, None, 0, None, measured=False)
    assert metrics.list_row(unmeasured)["measured"] is False and metrics.list_row(unmeasured)["focus_pct"] is None
    saved_earlier = finished("d" * 32, "Biology", "2026-11-15T01:50:00", 600.0, 96.4, 3, 251.0)
    del saved_earlier["measured"]               # a session file from before the field existed: unknown
    assert metrics.list_row(saved_earlier)["measured"] is None


def test_insights_are_time_weighted_and_leave_out_simulated_and_unfinished():
    sessions = [
        finished("1" * 32, "Biology", "2026-11-14T10:00:00", 600.0, 90.0, 4, 200.0),
        finished("2" * 32, "biology ", "2026-11-15T10:00:00", 1800.0, 98.0, 2, 260.0),
        finished("3" * 32, "History", "2026-11-15T11:00:00", 300.0, 80.0, 3, None, mode="video"),
        finished("4" * 32, "Biology", "2026-11-15T12:00:00", 900.0, 10.0, 30, 50.0, simulated=True),
        {"id": "5" * 32, "subject": "Biology", "created_at": "2026-11-15T13:00:00", "ended_at": None,
         "summary": None, "simulated": False},
    ]
    out = metrics.insights(sessions)
    assert out["excluded_simulated"] == 1 and out["excluded_unmeasured"] == 0
    assert out["overall"] == {
        "sessions": 3, "minutes": 45.0,
        "focus_pct": round((90 * 600 + 98 * 1800 + 80 * 300) / 2700, 1),
        "lapses_per_min": round(9 / 45, 2),
        "wpm": round((200 * 600 + 260 * 1800) / 2400, 1),     # only over sessions that have a wpm
    }
    assert [s["subject"] for s in out["by_subject"]] == ["Biology", "History"]
    bio, hist = out["by_subject"]
    assert bio["sessions"] == 2 and bio["minutes"] == 40.0
    assert bio["focus_pct"] == 96.0 and bio["lapses_per_min"] == 0.15 and bio["wpm"] == 245.0
    assert bio["trend"] == [
        {"date": "2026-11-14", "session_id": "1" * 32, "focus_pct": 90.0, "wpm": 200.0},
        {"date": "2026-11-15", "session_id": "2" * 32, "focus_pct": 98.0, "wpm": 260.0}]
    assert hist["wpm"] is None and hist["focus_pct"] == 80.0 and hist["trend"][0]["wpm"] is None

    with_sim = metrics.insights(sessions, include_simulated=True)
    assert with_sim["excluded_simulated"] == 0 and with_sim["overall"]["sessions"] == 4
    assert with_sim["by_subject"][0]["sessions"] == 3


def test_insights_leave_out_unmeasured_sessions_and_keep_ones_saved_before_the_flag():
    saved_earlier = finished("3" * 32, "Biology", "2026-11-13T10:00:00", 300.0, 80.0, 1, None)
    del saved_earlier["measured"], saved_earlier["summary"]["measured"]
    sessions = [
        finished("1" * 32, "Biology", "2026-11-14T10:00:00", 600.0, 90.0, 4, 200.0),
        finished("2" * 32, "Biology", "2026-11-15T10:00:00", 900.0, None, 0, 240.0, measured=False),
        saved_earlier,
        finished("4" * 32, "Biology", "2026-11-15T12:00:00", 900.0, 10.0, 30, 50.0, simulated=True, measured=False),
    ]
    out = metrics.insights(sessions)
    # the simulated one is counted as simulated only; "unmeasured" is for sessions left out for that reason alone
    assert out["excluded_simulated"] == 1 and out["excluded_unmeasured"] == 1
    assert out["overall"]["sessions"] == 2 and out["overall"]["minutes"] == 15.0
    assert out["overall"]["focus_pct"] == round((90 * 600 + 80 * 300) / 900, 1)
    assert {t["session_id"] for t in out["by_subject"][0]["trend"]} == {"1" * 32, "3" * 32}

    everything = metrics.insights(sessions, include_simulated=True)
    assert everything["excluded_simulated"] == 0 and everything["excluded_unmeasured"] == 0
    assert everything["overall"]["sessions"] == 4
    # the unmeasured session has no focus figure, so it adds minutes but nothing to the focus average
    assert everything["overall"]["focus_pct"] == round((90 * 600 + 80 * 300 + 10 * 900) / 1800, 1)


def test_insights_with_nothing_finished_reports_nothing():
    out = metrics.insights([])
    assert out == {"overall": {"sessions": 0, "minutes": 0.0, "focus_pct": None, "lapses_per_min": None, "wpm": None},
                   "by_subject": [], "excluded_simulated": 0, "excluded_unmeasured": 0}
