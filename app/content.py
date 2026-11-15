"""Study content: splitting text into sections and sentences, YouTube ids and transcripts, sample passages.

Nothing here generates text. Every function only cuts up, joins or looks up words that are already in the
passage or transcript it was given.
"""
from __future__ import annotations

import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_DEFAULT_ROOT = Path(__file__).resolve().parent.parent


def _setting(name, default):
    """A value from app/config.py, or the same literal default when that module can't be imported."""
    try:
        from . import config
    except Exception:
        return default
    return getattr(config, name, default)


# --------------------------------------------------------------------------------------------------------------
# Sentences and sections

_OPENERS = "\"'“‘([{«"
_CLOSERS = "\"'”’)]}»"

# Never end a sentence: titles and Latin abbreviations that are always followed by more of the same sentence.
_ABBREV_ALWAYS = {
    "mr", "mrs", "ms", "messrs", "dr", "prof", "st", "mt", "sr", "jr", "vs", "e.g", "i.e", "cf", "viz",
    "gen", "col", "lt", "capt", "sgt", "rev", "hon", "gov", "sen", "rep",
}
# End a sentence unless a number follows ("No. 4", "Fig. 2", "Jan. 3").
_ABBREV_BEFORE_NUMBER = {
    "no", "nos", "fig", "figs", "eq", "eqs", "vol", "vols", "pp", "ch", "chap", "sec", "ca", "approx",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}
# Usually mid-sentence ("et al. found", "the U.S. Army", "Acme Inc. shares"): these, and dotted forms such as
# "U.S." or "a.m.", end a sentence only when a common sentence-opening word follows.
_ABBREV_UNLESS_STARTER = {"etc", "al", "inc", "ltd", "co", "corp", "est"}
_SENTENCE_STARTERS = {
    "the", "a", "an", "it", "he", "she", "they", "we", "you", "i", "in", "on", "at", "by", "to", "from", "with",
    "for", "this", "that", "these", "those", "there", "here", "but", "and", "so", "yet", "then", "thus",
    "however", "therefore", "because", "since", "when", "while", "if", "as", "after", "before", "although",
    "his", "her", "its", "their", "our", "my", "some", "many", "most", "one", "all", "each", "every", "such",
    "no", "not", "now", "what", "why", "how", "who", "where",
}
_DOTTED = re.compile(r"^(?:[a-z]\.)+[a-z]$")           # u.s, a.m, ph.d … (after the final period is removed)
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_CLAUSE_END = (",", ";", ":")


def _paragraphs(text: str) -> list[list[str]]:
    """Paragraphs (split on blank lines) as lists of whitespace-separated words; empty ones dropped."""
    paras = [p.split() for p in _PARAGRAPH_BREAK.split(text or "")]
    return [p for p in paras if p]


def _first_alnum(token: str) -> str:
    stripped = token.lstrip(_OPENERS)
    return stripped[:1]


def _ends_sentence(token: str, nxt: str | None, first_in_paragraph: bool) -> bool:
    core = token.rstrip(_CLOSERS)
    if not core or core[-1] not in ".!?…":
        return False
    if nxt is None:
        return True
    nxt_first = _first_alnum(nxt)
    if nxt_first.islower():
        return False                                   # '"Why?" he asked.' / an abbreviation we don't know
    if core[-1] in "!?":
        return True
    if core.endswith("...") or core[-1] == "…":
        return nxt_first.isupper()
    word = core[:-1].lstrip(_OPENERS)
    low = word.lower()
    if low in _ABBREV_ALWAYS:
        return False
    if low in _ABBREV_BEFORE_NUMBER and nxt_first.isdigit():
        return False
    if low in _ABBREV_UNLESS_STARTER or _DOTTED.match(low):
        nxt_word = nxt.strip(_OPENERS + _CLOSERS + ".,;:!?")
        return nxt_first.isupper() and nxt_word.lower() in _SENTENCE_STARTERS
    if len(word) == 1 and word.isalpha() and word.isupper() and word != "I":
        return False                                   # an initial: "J. K. Rowling", "Charles R. Darwin"
    if word.isdigit() and first_in_paragraph:
        return False                                   # a numbered list item: "1. First…"
    return True


def _paragraph_sentences(words: list[str]) -> list[list[str]]:
    out, cur = [], []
    for i, tok in enumerate(words):
        cur.append(tok)
        nxt = words[i + 1] if i + 1 < len(words) else None
        if _ends_sentence(tok, nxt, first_in_paragraph=(i == 0)):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def split_sentences(text: str) -> list[str]:
    """Sentences in order, whitespace collapsed to single spaces. Words are never dropped, added or changed."""
    return [" ".join(s) for para in _paragraphs(text) for s in _paragraph_sentences(para)]


def _split_long(words: list[str], max_words: int) -> list[list[str]]:
    """Cut one over-long sentence into chunks of at most max_words, at a clause boundary when one is near."""
    out, i = [], 0
    while len(words) - i > max_words:
        cut = i + max_words
        for j in range(cut, i + int(max_words * 0.6), -1):
            if words[j - 1].rstrip(_CLOSERS).endswith(_CLAUSE_END):
                cut = j
                break
        out.append(words[i:cut])
        i = cut
    out.append(words[i:])
    return out


def split_sections(text: str, target_words: int = 170, max_words: int = 220) -> list[dict]:
    """Cut a passage into sections of roughly target_words (never more than max_words), on paragraph and
    sentence boundaries. The sections' words, in order, are exactly the passage's words.

    A section's text keeps paragraph breaks as blank lines; all other whitespace becomes single spaces.
    """
    max_words = max(1, int(max_words))
    target_words = max(1, min(int(target_words), max_words))
    min_words = max(1, round(0.7 * target_words))

    units = []                                         # (words, paragraph index)
    for pi, para in enumerate(_paragraphs(text)):
        for sent in _paragraph_sentences(para):
            for chunk in (_split_long(sent, max_words) if len(sent) > max_words else [sent]):
                units.append((chunk, pi))

    groups, cur, cur_n = [], [], 0
    for i, (words, pi) in enumerate(units):
        if cur and cur_n + len(words) > max_words:
            groups.append(cur)
            cur, cur_n = [], 0
        cur.append((words, pi))
        cur_n += len(words)
        if i + 1 < len(units):
            paragraph_ends = units[i + 1][1] != pi
            if cur_n >= target_words or (paragraph_ends and cur_n >= min_words):
                groups.append(cur)
                cur, cur_n = [], 0
    if cur:
        groups.append(cur)

    def count(group):
        return sum(len(w) for w, _ in group)

    # A short tail reads badly as its own section: fold it into the one before when that still fits.
    if len(groups) > 1 and count(groups[-1]) < min_words // 2 and count(groups[-2]) + count(groups[-1]) <= max_words:
        groups[-2].extend(groups.pop())

    sections = []
    for idx, group in enumerate(groups):
        paras, last_pi = [], None
        for words, pi in group:
            if pi != last_pi:
                paras.append([])
                last_pi = pi
            paras[-1].extend(words)
        sections.append({"idx": idx, "text": "\n\n".join(" ".join(p) for p in paras), "word_count": count(group)})
    return sections


# --------------------------------------------------------------------------------------------------------------
# YouTube

_YT_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YT_HOSTS = {"youtube.com", "youtube-nocookie.com"}
_YT_PATH_PREFIXES = ("embed", "shorts", "live", "v")


def parse_youtube_id(url: str) -> str | None:
    """The 11-character video id from a YouTube URL (watch?v=, youtu.be/, /embed/, /shorts/) or a bare id."""
    if not isinstance(url, str):
        return None
    url = url.strip()
    if _YT_ID.match(url):
        return url
    if not url or any(c.isspace() for c in url):
        return None
    if "://" not in url:
        url = "https://" + url
    try:
        parts = urlparse(url)
        host = (parts.hostname or "").lower()
    except ValueError:
        return None
    if parts.scheme not in ("http", "https"):
        return None
    for prefix in ("www.", "m.", "music."):
        if host.startswith(prefix):
            host = host[len(prefix):]
            break
    segments = [s for s in parts.path.split("/") if s]
    candidate = None
    if host == "youtu.be":
        candidate = segments[0] if segments else None
    elif host in _YT_HOSTS:
        if segments and segments[0] == "watch" or not segments:
            candidate = (parse_qs(parts.query).get("v") or [None])[0]
        elif len(segments) >= 2 and segments[0] in _YT_PATH_PREFIXES:
            candidate = segments[1]
    return candidate if candidate and _YT_ID.match(candidate) else None


_TRANSCRIPT_LANGUAGES = ("en", "en-US", "en-GB")

# Library error class name -> what to tell the student. Checked along the exception's class hierarchy.
_TRANSCRIPT_ERRORS = {
    "TranscriptsDisabled": "Captions are turned off for this video.",
    "NoTranscriptFound": "This video has no captions.",
    "InvalidVideoId": "That does not look like a YouTube video id.",
    "VideoUnavailable": "This video is unavailable.",
    "VideoUnplayable": "This video cannot be played, so its captions cannot be read.",
    "AgeRestricted": "This video is age-restricted, so its captions cannot be read without signing in.",
    "IpBlocked": "YouTube is blocking caption requests from this network. Try again later or on another network.",
    "RequestBlocked": "YouTube is blocking caption requests from this network. Try again later or on another network.",
    "PoTokenRequired": "YouTube would not release this video's captions to the app.",
    "YouTubeRequestFailed": "YouTube returned an error when asked for the captions. Try again later.",
    "YouTubeDataUnparsable": "YouTube's reply could not be read. Try again later.",
    "FailedToCreateConsentCookie": "YouTube asked for cookie consent, so the captions could not be read.",
    "CouldNotRetrieveTranscript": "The captions for this video could not be retrieved.",
    "ConnectionError": "Could not reach YouTube. Check the internet connection.",
    "Timeout": "YouTube took too long to answer. Check the internet connection.",
    "RequestException": "Could not reach YouTube. Check the internet connection.",
    "OSError": "Could not reach YouTube. Check the internet connection.",
}


# Bounds on caption requests: (connect, read) seconds for each request, and seconds for all the requests one
# client makes. The library's own default session sets no timeout at all.
_TRANSCRIPT_TIMEOUT = (5.0, 15.0)
_TRANSCRIPT_BUDGET_S = 30.0


def _bounded_http_session(timeout=_TRANSCRIPT_TIMEOUT, budget_s: float = _TRANSCRIPT_BUDGET_S):
    """A requests.Session whose every request carries a timeout, and which refuses to start another request
    (requests.exceptions.Timeout) once `budget_s` seconds have passed since it was made. The read timeout is
    per socket read, so one request can outlast it if the server keeps trickling bytes."""
    import requests

    class _BoundedSession(requests.Session):
        def __init__(self):
            super().__init__()
            self._deadline = time.monotonic() + budget_s

        def request(self, method, url, **kwargs):
            if time.monotonic() >= self._deadline:
                raise requests.exceptions.Timeout("the caption requests took too long")
            if kwargs.get("timeout") is None:
                kwargs["timeout"] = timeout
            return super().request(method, url, **kwargs)

    return _BoundedSession()


def _new_transcript_api():
    """A youtube-transcript-api client with time-bounded requests. Imported here so the app still starts
    when the library is missing."""
    from youtube_transcript_api import YouTubeTranscriptApi
    return YouTubeTranscriptApi(http_client=_bounded_http_session())


def _error_names(exc: BaseException) -> list[str]:
    return [c.__name__ for c in type(exc).__mro__]


def _transcript_error(exc: BaseException) -> str:
    for name in _error_names(exc):
        if name in _TRANSCRIPT_ERRORS:
            return _TRANSCRIPT_ERRORS[name]
    return f"The captions for this video could not be retrieved ({type(exc).__name__})."


def _clean_snippets(fetched) -> list[dict]:
    out = []
    for s in fetched or []:
        get = s.get if isinstance(s, dict) else (lambda k, _s=s: getattr(_s, k, None))
        text = " ".join(str(get("text") or "").split())
        if not text:
            continue
        try:
            start, duration = float(get("start") or 0.0), float(get("duration") or 0.0)
        except (TypeError, ValueError):
            continue
        out.append({"start": start, "duration": max(0.0, duration), "text": text})
    return out


def _transcript_cache_file(video_id: str, cache_dir: Path) -> Path:
    return Path(cache_dir) / "transcripts" / f"{video_id}.json"


def _fetch_any_language(api, video_id: str):
    """English first; if the video has captions only in other languages, take the first one listed."""
    try:
        return api.fetch(video_id, languages=_TRANSCRIPT_LANGUAGES)
    except Exception as exc:
        if "NoTranscriptFound" not in _error_names(exc):
            raise
        for transcript in api.list(video_id):
            return transcript.fetch()
        raise


def fetch_transcript(video_id: str, cache_dir: Path) -> dict:
    """The video's captions: {"available", "error", "snippets": [{"start", "duration", "text"}]}.

    Successful fetches are cached as JSON under cache_dir/transcripts/. Never raises: no captions, a blocked
    request or no network all come back as available False with a plain-English error.
    """
    def failed(reason: str) -> dict:
        return {"available": False, "error": reason, "snippets": []}

    if not isinstance(video_id, str) or not _YT_ID.match(video_id):
        return failed(_TRANSCRIPT_ERRORS["InvalidVideoId"])

    cache_file = None
    try:
        cache_file = _transcript_cache_file(video_id, cache_dir)
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        snippets = _clean_snippets(cached.get("snippets"))
        if snippets:
            return {"available": True, "error": None, "snippets": snippets}
    except (OSError, ValueError, TypeError, AttributeError):
        pass                                           # no cache, or an unreadable one: fetch again

    try:
        api = _new_transcript_api()
    except ImportError:
        return failed("The caption library (youtube-transcript-api) is not installed.")
    except Exception as exc:
        return failed(_transcript_error(exc))

    try:
        snippets = _clean_snippets(_fetch_any_language(api, video_id))
    except Exception as exc:
        return failed(_transcript_error(exc))
    if not snippets:
        return failed("This video's captions are empty.")

    if cache_file is not None:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"video_id": video_id, "snippets": snippets}, ensure_ascii=False),
                           encoding="utf-8")
            os.replace(tmp, cache_file)
        except OSError:
            pass                                       # the transcript is still good; it just isn't cached
    return {"available": True, "error": None, "snippets": snippets}


def _snippet_ends(snippets: list[dict]) -> list[float]:
    """When each caption stops being the current one: its own end, or the next caption's start if sooner
    (auto-generated captions stay on screen while the next one appears, so their durations overlap)."""
    ends = []
    for i, s in enumerate(snippets):
        end = s["start"] + s.get("duration", 0.0)
        if i + 1 < len(snippets) and snippets[i + 1]["start"] > s["start"]:
            end = min(end, snippets[i + 1]["start"])
        ends.append(end)
    return ends


def video_title(video_id: str) -> str | None:
    """The video's title from YouTube's public oEmbed endpoint (no key), or None if it can't be had in 3 s."""
    try:
        import httpx
        r = httpx.get("https://www.youtube.com/oembed",
                      params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"}, timeout=3.0)
        title = r.json().get("title") if r.status_code == 200 else None
        return " ".join(str(title).split())[:200] or None if title else None
    except Exception:  # noqa: BLE001
        return None


def transcript_sections(snippets: list[dict], target_s: float = 60.0) -> list[dict]:
    """Group captions into sections of about target_s seconds. Sections are contiguous in time: each one ends
    where the next begins. Punctuated captions are cut at the first sentence end after target_s (within 25%)."""
    snippets = sorted(_clean_snippets(snippets), key=lambda s: s["start"])
    if not snippets:
        return []
    target_s = max(1.0, float(target_s))
    punctuated = any(s["text"].rstrip(_CLOSERS).endswith((".", "!", "?")) for s in snippets)
    slack = 0.25 * target_s if punctuated else 0.0

    groups, cur = [], []
    for i, s in enumerate(snippets):
        cur.append(s)
        if i + 1 == len(snippets):
            break
        elapsed = snippets[i + 1]["start"] - cur[0]["start"]
        at_sentence_end = s["text"].rstrip(_CLOSERS).endswith((".", "!", "?"))
        if elapsed >= target_s + slack or (elapsed >= target_s and at_sentence_end):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)

    def words(group):
        return sum(len(s["text"].split()) for s in group)

    # Fold a scrap at the end (a sign-off, a few seconds) into the section before it.
    if len(groups) > 1:
        tail = groups[-1]
        tail_s = tail[-1]["start"] + tail[-1]["duration"] - tail[0]["start"]
        if tail_s < 0.25 * target_s or words(tail) < 20:
            groups[-2].extend(groups.pop())

    last = snippets[-1]
    final_end = max(s["start"] + s["duration"] for s in snippets[-5:])
    sections = []
    for idx, group in enumerate(groups):
        end = groups[idx + 1][0]["start"] if idx + 1 < len(groups) else max(final_end, last["start"])
        sections.append({
            "idx": idx,
            "start_s": round(group[0]["start"], 2),
            "end_s": round(end, 2),
            "text": " ".join(s["text"] for s in group),
            "word_count": words(group),
        })
    return sections


def transcript_span(snippets: list[dict], t_start: float, t_end: float, pad: float = 5.0) -> str:
    """The caption text on screen during [t_start - pad, t_end + pad]; "" when there is none."""
    snippets = sorted(_clean_snippets(snippets), key=lambda s: s["start"])
    try:
        lo = float(t_start)
        hi = float(t_end) if t_end is not None else lo
    except (TypeError, ValueError):
        return ""
    if hi < lo:
        lo, hi = hi, lo
    lo, hi = lo - pad, hi + pad
    ends = _snippet_ends(snippets)
    return " ".join(s["text"] for s, end in zip(snippets, ends) if s["start"] < hi and end > lo)


# --------------------------------------------------------------------------------------------------------------
# Sample passages (the same Darwin text and the same partition as lookaway_test.load_part)

_TEXT_FIRST_LINE = 313        # "INTRODUCTION." of the main text (after the table of contents)
_WORDS_PER_PART = 3000
_SAMPLE_PARTS = range(5, 10)  # parts 1-4 were read in the look-away test sessions
_SAMPLE_ID = re.compile(r"^darwin-(\d{1,3})$")


@lru_cache(maxsize=2)
def _sample_parts(path: str, mtime: float) -> tuple[tuple[str, ...], ...]:
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    end = next(i for i, ln in enumerate(lines) if ln.startswith("*** END OF THE PROJECT GUTENBERG"))
    body = "\n".join(lines[_TEXT_FIRST_LINE - 1:end])
    paras = [re.sub(r"\s+", " ", p).strip() for p in body.split("\n\n")]
    paras = [p for p in paras if p]
    parts, cur, n = [], [], 0
    for p in paras:
        cur.append(p)
        n += len(p.split())
        if n >= _WORDS_PER_PART:
            parts.append(tuple(cur))
            cur, n = [], 0
    if cur:
        parts.append(tuple(cur))
    return tuple(parts)


def _load_parts() -> tuple[tuple[str, ...], ...]:
    path = Path(_setting("SAMPLE_TEXT", _DEFAULT_ROOT / "texts" / "origin_of_species_pg1228.txt"))
    try:
        return _sample_parts(str(path), path.stat().st_mtime)
    except (OSError, StopIteration, UnicodeDecodeError):
        return ()


def _sample(part: int, paras: tuple[str, ...]) -> dict:
    return {"id": f"darwin-{part}", "title": f"On the Origin of Species, part {part}",
            "words": sum(len(p.split()) for p in paras)}


def list_samples() -> list[dict]:
    """[{"id", "title", "words"}] for parts 5-9 of the sample text; [] if the text file is missing."""
    parts = _load_parts()
    return [_sample(n, parts[n - 1]) for n in _SAMPLE_PARTS if n <= len(parts)]


def load_sample(sample_id: str) -> dict:
    """{"id", "title", "text"} for one sample; paragraphs are separated by blank lines.
    Raises KeyError for an id that list_samples() does not offer."""
    m = _SAMPLE_ID.match(sample_id) if isinstance(sample_id, str) else None
    parts = _load_parts()
    part = int(m.group(1)) if m else 0
    if part not in _SAMPLE_PARTS or part > len(parts):
        raise KeyError(sample_id)
    info = _sample(part, parts[part - 1])
    return {"id": info["id"], "title": info["title"], "text": "\n\n".join(parts[part - 1])}
