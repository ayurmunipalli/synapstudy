"""Session and long-term numbers (APP_SPEC.md §7). Pure functions: no I/O, no clock, no globals.

    duration_s      = (ended_ms - started_ms) / 1000
    lapse_seconds   = time covered by the lapses, clipped to the session (overlapping lapses count once)
    focus_pct       = 100 * (1 - lapse_seconds / duration_s)
    lapses_per_min  = lapse_count / (duration_s / 60)
    words_read      = words in the sections the reader completed
    wpm             = words_read / ((duration_s - lapse_seconds) / 60); None under 20 s of reading or with no
                      completed section

Anything that cannot be computed is None, never a guess.
"""
import math

MIN_READING_S = 20.0
SOURCES = ("live", "replay", "simulated")


def _num(x) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        return None
    return float(x)


def _round(x, digits=1):
    return None if x is None else round(float(x), digits)


def fmt_clock(seconds) -> str:
    """90 -> "01:30"; 3725 -> "1:02:05"."""
    s = max(0, int(_num(seconds) or 0))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


# ---------------------------------------------------------------------------- one session
def duration_s(started_ms, ended_ms) -> float:
    a, b = _num(started_ms), _num(ended_ms)
    if a is None or b is None:
        return 0.0
    return max(0.0, (b - a) / 1000.0)


def lapse_interval(lapse: dict, ended_ms) -> tuple[float, float] | None:
    """(onset_ms, resume_ms) of one lapse. A lapse still open at the end runs to the end of the session;
    one with no usable onset is skipped."""
    if not isinstance(lapse, dict):
        return None
    onset = _num(lapse.get("onset_ms"))
    if onset is None:
        return None
    resume = _num(lapse.get("resume_ms"))
    if resume is None:
        dur = _num(lapse.get("duration_s"))
        end = _num(ended_ms)
        resume = onset + dur * 1000.0 if dur is not None else (end if end is not None else onset)
    return (onset, max(onset, resume))


def lapse_duration_s(lapse: dict, ended_ms=None) -> float | None:
    iv = lapse_interval(lapse, ended_ms)
    if iv is None:
        return _num(lapse.get("duration_s")) if isinstance(lapse, dict) else None
    return (iv[1] - iv[0]) / 1000.0


def lapse_seconds(lapses, started_ms, ended_ms) -> float:
    """Seconds of the session spent in a lapse: each lapse clipped to [started_ms, ended_ms], and time
    covered by two overlapping lapses counted once (so it can never exceed the session)."""
    a, b = _num(started_ms), _num(ended_ms)
    if a is None or b is None or b <= a:
        return 0.0
    spans = []
    for lapse in lapses or []:
        iv = lapse_interval(lapse, b)
        if iv is None:
            continue
        lo, hi = max(iv[0], a), min(iv[1], b)
        if hi > lo:
            spans.append((lo, hi))
    spans.sort()
    total, cur_lo, cur_hi = 0.0, None, None
    for lo, hi in spans:
        if cur_hi is None or lo > cur_hi:
            if cur_hi is not None:
                total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
        else:
            cur_hi = max(cur_hi, hi)
    if cur_hi is not None:
        total += cur_hi - cur_lo
    return total / 1000.0


def focus_pct(duration: float, lapse_s: float) -> float | None:
    if duration <= 0:
        return None
    return min(100.0, max(0.0, 100.0 * (1.0 - lapse_s / duration)))


def lapses_per_min(count: int, duration: float) -> float | None:
    if duration <= 0:
        return None
    return count / (duration / 60.0)


def completed_sections(progress) -> set[int]:
    done = set()
    for p in progress or []:
        if isinstance(p, dict) and isinstance(p.get("section_idx"), int) and not isinstance(p["section_idx"], bool):
            done.add(p["section_idx"])
    return done


def words_read(sections, progress) -> int:
    """Words in the sections the log marks complete; each section counts once."""
    done = completed_sections(progress)
    total = 0
    for sec in sections or []:
        if isinstance(sec, dict) and sec.get("idx") in done:
            total += int(_num(sec.get("word_count")) or 0)
    return total


def wpm(words: int, duration: float, lapse_s: float) -> float | None:
    reading_s = duration - lapse_s
    if words <= 0 or reading_s < MIN_READING_S:
        return None
    return words / (reading_s / 60.0)


def session_metrics(log: dict, sections, mode: str) -> dict:
    """The numeric part of a session summary, in the spec's key order. `missed`, `simulated`, `measured`
    and `generated_at` are added by the server, which also blanks focus_pct and lapses_per_min when the
    headband was never connected and no lapse was logged (nothing was measured)."""
    lapses = [l for l in (log.get("lapses") or []) if isinstance(l, dict)]
    dur = duration_s(log.get("started_ms"), log.get("ended_ms"))
    lapse_s = lapse_seconds(lapses, log.get("started_ms"), log.get("ended_ms"))
    out = {
        "duration_s": _round(dur),
        "lapse_count": len(lapses),
        "lapse_seconds": _round(lapse_s),
        "focus_pct": _round(focus_pct(dur, lapse_s)),
        "lapses_per_min": _round(lapses_per_min(len(lapses), dur), 2),
        "reading": None,
        "video": None,
    }
    if mode == "video":
        video = log.get("video") if isinstance(log.get("video"), dict) else {}
        watched = _num(video.get("watched_s"))
        out["video"] = {"duration_s": _round(_num(video.get("duration_s"))),
                        "watched_s": _round(watched if watched is not None else 0.0)}
    else:
        words = words_read(sections, log.get("progress"))
        out["reading"] = {"words_read": words, "wpm": _round(wpm(words, dur, lapse_s))}
    return out


def is_simulated(server_events, lapses=None) -> bool:
    """True when the server's own event log has any event that did not come from the live headband.
    A lapse the client itself reports as not live also sets it. Nothing the client sends can clear it."""
    for ev in server_events or []:
        if not isinstance(ev, dict) or ev.get("source") != "live":
            return True
    for lapse in lapses or []:
        if not isinstance(lapse, dict) or lapse.get("source") != "live":
            return True
    return False


# ---------------------------------------------------------------------------- was the session measured?
NOTE_REPLAY = ("Lapses in this session came from a replayed recording, not from you; "
               "it is left out of long-term insights.")
NOTE_PARTIAL = "The headband was connected for only part of this session; it is left out of long-term insights."
NOTE_SIMULATED = "Lapses in this session were simulated; it is left out of long-term insights."
NOTE_NO_HEADBAND = "No headband was connected: focus was not measured."


def log_sources(source_log) -> list:
    """The `source` of each entry in a session's source_log (None for an entry that is not usable)."""
    return [e.get("source") if isinstance(e, dict) else None for e in source_log or []]


def ever_live(source_log) -> bool:
    return "live" in log_sources(source_log)


def measurement(source_log, simulated: bool, lapse_count: int, replay_seen: bool = False) -> tuple[bool, str | None]:
    """(measured, measure_note) for a finished session.

    `source_log` is the server's record of which event source was in effect while the session ran.
    measured is True only when that log is not empty, every entry is "live" and the session is not flagged
    simulated; the note is then None. Otherwise the note says why, first match first: a replay ran, the
    headband was connected for only part of the session, it was never connected but lapses were logged,
    or it was never connected and nothing was logged. `replay_seen` is True when an event or lapse of the
    session itself carries the source "replay"; it counts like a replay entry in the log."""
    sources = log_sources(source_log)
    live = [s == "live" for s in sources]
    if sources and all(live) and not simulated:
        return True, None
    if "replay" in sources or replay_seen:
        return False, NOTE_REPLAY
    if any(live) and not all(live):
        return False, NOTE_PARTIAL
    if lapse_count > 0 or any(live):    # all live but flagged simulated: a lapse or event was not live
        return False, NOTE_SIMULATED
    return False, NOTE_NO_HEADBAND


# ---------------------------------------------------------------------------- lists and long-term insights
def _flag(value):
    """True/False as stored, None for anything else (sessions saved before the field existed)."""
    return value if isinstance(value, bool) else None


def list_row(session: dict) -> dict:
    summary = session.get("summary") or {}
    reading = summary.get("reading") or {}
    return {
        "id": session.get("id"), "title": session.get("title"), "subject": session.get("subject"),
        "mode": session.get("mode"), "created_at": session.get("created_at"),
        "ended_at": session.get("ended_at"),
        "duration_s": summary.get("duration_s"), "focus_pct": summary.get("focus_pct"),
        "lapse_count": summary.get("lapse_count"), "wpm": reading.get("wpm"),
        "simulated": bool(session.get("simulated")),
        "measured": _flag(session.get("measured")),
    }


def _aggregate(rows: list[dict]) -> dict:
    """Time-weighted averages over finished sessions (weights = session duration)."""
    total_s = sum(r["duration_s"] for r in rows)
    focus_w = sum(r["focus_pct"] * r["duration_s"] for r in rows if r["focus_pct"] is not None)
    focus_t = sum(r["duration_s"] for r in rows if r["focus_pct"] is not None)
    wpm_w = sum(r["wpm"] * r["duration_s"] for r in rows if r["wpm"] is not None)
    wpm_t = sum(r["duration_s"] for r in rows if r["wpm"] is not None)
    lapses = sum(r["lapse_count"] for r in rows)
    return {
        "sessions": len(rows),
        "minutes": _round(total_s / 60.0),
        "focus_pct": _round(focus_w / focus_t) if focus_t > 0 else None,
        "lapses_per_min": _round(lapses / (total_s / 60.0), 2) if total_s > 0 else None,
        "wpm": _round(wpm_w / wpm_t) if wpm_t > 0 else None,
    }


def insights(sessions, include_simulated: bool = False) -> dict:
    """Long-term insights: overall and by subject. Only finished sessions count. Sessions flagged
    simulated are left out (and counted in `excluded_simulated`), and so are sessions whose `measured`
    flag is False (counted in `excluded_unmeasured` when that is the only reason), unless
    include_simulated is set. A session saved before `measured` existed is unknown, and is kept."""
    rows, excluded, unmeasured = [], 0, 0
    for s in sessions or []:
        summary = s.get("summary")
        if not s.get("ended_at") or not isinstance(summary, dict):
            continue
        if s.get("simulated") or summary.get("simulated"):
            if not include_simulated:
                excluded += 1
                continue
        elif s.get("measured") is False or summary.get("measured") is False:
            if not include_simulated:
                unmeasured += 1
                continue
        reading = summary.get("reading") or {}
        rows.append({
            "id": s.get("id"),
            "subject": str(s.get("subject") or "").strip() or "General",
            "created_at": str(s.get("created_at") or ""),
            "duration_s": _num(summary.get("duration_s")) or 0.0,
            "focus_pct": _num(summary.get("focus_pct")),
            "lapse_count": int(_num(summary.get("lapse_count")) or 0),
            "wpm": _num(reading.get("wpm")),
        })
    rows.sort(key=lambda r: r["created_at"])

    groups: dict[str, list[dict]] = {}
    names: dict[str, str] = {}
    for r in rows:
        key = r["subject"].casefold()
        names.setdefault(key, r["subject"])
        groups.setdefault(key, []).append(r)

    by_subject = []
    for key in sorted(groups):
        agg = _aggregate(groups[key])
        by_subject.append({
            "subject": names[key], **agg,
            "trend": [{"date": r["created_at"][:10], "session_id": r["id"],
                       "focus_pct": _round(r["focus_pct"]), "wpm": _round(r["wpm"])} for r in groups[key]],
        })
    return {"overall": _aggregate(rows), "by_subject": by_subject, "excluded_simulated": excluded,
            "excluded_unmeasured": unmeasured}
