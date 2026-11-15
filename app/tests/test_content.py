"""Tests for app/content.py. No network: the YouTube library is replaced by fakes. Synthetic text is used only
here; the Darwin sample is the real file in texts/ (read, never written). Files are written only under tmp_path."""
import ast
import inspect
import json
import random
import re
from pathlib import Path

import pytest

from app import content

ROOT = Path(__file__).resolve().parents[2]
VIDEO_ID = "dQw4w9WgXcQ"


# ---------------------------------------------------------------------------------------------- sentences

def test_split_sentences_copes_with_abbreviations_and_initials():
    text = ('Mr. Darwin sailed on H.M.S. Beagle in 1831. He visited, e.g., the Galapagos Islands, etc. and wrote '
            'notes. "Why?" he asked. J. B. S. Haldane and Dr. R. A. Fisher disagreed. The U.S. Army paid Smith '
            'et al. for it. It ended at 5 p.m. The next day came... Then nothing. See Fig. 3 and No. 7. I said '
            'no. So did I. Done!')
    assert content.split_sentences(text) == [
        "Mr. Darwin sailed on H.M.S. Beagle in 1831.",
        "He visited, e.g., the Galapagos Islands, etc. and wrote notes.",
        '"Why?" he asked.',
        "J. B. S. Haldane and Dr. R. A. Fisher disagreed.",
        "The U.S. Army paid Smith et al. for it.",
        "It ended at 5 p.m.",
        "The next day came...",
        "Then nothing.",
        "See Fig. 3 and No. 7.",
        "I said no.",
        "So did I.",
        "Done!",
    ]


def test_split_sentences_breaks_at_paragraphs_and_keeps_every_word():
    text = "A heading without a full stop\n\nFirst sentence here.  Second one,\nwrapped over a line.\n\n\nLast"
    sentences = content.split_sentences(text)
    assert sentences == ["A heading without a full stop", "First sentence here.",
                         "Second one, wrapped over a line.", "Last"]
    assert " ".join(sentences).split() == text.split()


def test_split_sentences_empty():
    assert content.split_sentences("") == []
    assert content.split_sentences("  \n\n \t ") == []


# ---------------------------------------------------------------------------------------------- sections

def _words_of(sections):
    return " ".join(s["text"] for s in sections).split()


def _check_sections(sections, text, max_words=220):
    assert _words_of(sections) == text.split()                 # nothing dropped, duplicated or reordered
    assert [s["idx"] for s in sections] == list(range(len(sections)))
    for s in sections:
        assert s["word_count"] == len(s["text"].split())
        assert 0 < s["word_count"] <= max_words
        assert set(s) == {"idx", "text", "word_count"}


def test_split_sections_on_the_real_sample_keeps_every_word_and_stays_in_range():
    samples = content.list_samples()
    assert samples, "texts/origin_of_species_pg1228.txt should be present"
    for info in samples:
        text = content.load_sample(info["id"])["text"]
        sections = content.split_sections(text)
        _check_sections(sections, text)
        counts = [s["word_count"] for s in sections]
        # "~120-220 words": every section but the last must reach the lower bound
        assert all(c >= 119 for c in counts[:-1]), counts
        # sections end at a sentence end (Darwin's sentences are all under 220 words in these parts)
        assert all(s["text"].rstrip("\"')”’")[-1] in ".!?" for s in sections[:-1])


def test_split_sections_random_synthetic_text_never_loses_words():
    rng = random.Random(42)
    vocab = ["alpha", "beta,", "gamma.", "Delta", "e.g.", "Mr.", "J.", "zeta!", "eta?", '"theta."', "iota;",
             "kappa", "3.5", "U.S.", "(lambda)", "mu...", "nu", "The", "xi:", "etc."]
    for _ in range(60):
        paras = []
        for _ in range(rng.randint(1, 8)):
            paras.append(" ".join(rng.choice(vocab) for _ in range(rng.randint(1, 400))))
        text = rng.choice(["\n\n", "\n \n", "\n\n\n"]).join(paras)
        target = rng.choice([20, 60, 170])
        max_words = target + rng.choice([0, 10, 50])
        _check_sections(content.split_sections(text, target_words=target, max_words=max_words), text, max_words)


def test_split_sections_cuts_a_sentence_longer_than_the_maximum():
    text = " ".join(f"word{i}" for i in range(700))            # one 700-word "sentence" with no punctuation
    sections = content.split_sections(text)
    _check_sections(sections, text)
    assert len(sections) >= 4


def test_split_sections_keeps_paragraph_breaks_and_prefers_paragraph_boundaries():
    para = lambda tag, n: " ".join([f"{tag}{i}" for i in range(n - 1)] + [f"{tag}end."])  # noqa: E731
    text = para("a", 130) + "\n\n" + para("b", 40) + "\n\n" + para("c", 100) + "\n\n" + para("d", 150)
    sections = content.split_sections(text)
    _check_sections(sections, text)
    assert sections[0]["text"].split()[-1] == "aend."           # 130 words >= 0.7 x 170: stops at the paragraph end
    assert "\n\n" in sections[1]["text"]                        # b and c share a section, break kept
    assert sections[1]["text"].startswith("b0 ") and sections[1]["text"].split()[-1] == "cend."


def test_split_sections_short_and_empty_text():
    assert content.split_sections("") == []
    assert content.split_sections(" \n\n ") == []
    one = content.split_sections("Just one short sentence.")
    assert one == [{"idx": 0, "text": "Just one short sentence.", "word_count": 4}]


def test_split_sections_folds_a_short_tail_into_the_previous_section():
    sentence = " ".join(["word"] * 9) + " end. "
    text = sentence * 17 + "A short closing remark."                # 170 words, then a 4-word tail
    sections = content.split_sections(text)
    _check_sections(sections, text)
    assert len(sections) == 1 and sections[0]["word_count"] == 174


# ---------------------------------------------------------------------------------------------- YouTube ids

@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10s",
    "https://www.youtube.com/watch?feature=share&v=dQw4w9WgXcQ",
    "http://youtube.com/watch?v=dQw4w9WgXcQ",
    "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://music.youtube.com/watch?v=dQw4w9WgXcQ",
    "youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ?si=abcdef&t=5",
    "youtu.be/dQw4w9WgXcQ",
    "https://www.youtube.com/embed/dQw4w9WgXcQ",
    "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ?rel=0",
    "https://www.youtube.com/shorts/dQw4w9WgXcQ",
    "https://www.youtube.com/live/dQw4w9WgXcQ",
    "  https://youtu.be/dQw4w9WgXcQ  ",
    "dQw4w9WgXcQ",
])
def test_parse_youtube_id_accepts(url):
    assert content.parse_youtube_id(url) == VIDEO_ID


@pytest.mark.parametrize("url", [
    "", "   ", None, 12345, "not a url", "https://example.com/watch?v=dQw4w9WgXcQ",
    "https://youtube.com.evil.example/watch?v=dQw4w9WgXcQ", "https://www.youtube.com/watch?v=tooshort",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ_too_long", "https://www.youtube.com/watch",
    "https://youtu.be/", "https://www.youtube.com/playlist?list=PL1234567890", "dQw4w9WgXc",
    "javascript:alert(1)//youtu.be/dQw4w9WgXcQ", "https://www.youtube.com/embed/../../etc/passwd",
])
def test_parse_youtube_id_rejects(url):
    assert content.parse_youtube_id(url) is None


# ---------------------------------------------------------------------------------------------- transcripts

class FakeApi:
    """Stands in for youtube_transcript_api.YouTubeTranscriptApi (same .fetch / .list call shapes)."""

    def __init__(self, fetch=None, listing=None):
        self._fetch, self._listing, self.calls = fetch, listing, []

    def fetch(self, video_id, languages=("en",), preserve_formatting=False):
        self.calls.append(("fetch", video_id, tuple(languages)))
        if isinstance(self._fetch, Exception):
            raise self._fetch
        return self._fetch

    def list(self, video_id):
        self.calls.append(("list", video_id))
        if isinstance(self._listing, Exception):
            raise self._listing
        return self._listing


def _fetched(rows):
    from youtube_transcript_api import FetchedTranscript, FetchedTranscriptSnippet
    return FetchedTranscript(snippets=[FetchedTranscriptSnippet(text=t, start=s, duration=d) for t, s, d in rows],
                             video_id=VIDEO_ID, language="English", language_code="en", is_generated=True)


def _use(monkeypatch, api):
    monkeypatch.setattr(content, "_new_transcript_api", lambda: api)
    return api


def test_the_library_api_matches_what_content_calls():
    """The real class exposes the instance methods and parameters fetch_transcript relies on (no request made)."""
    from youtube_transcript_api import YouTubeTranscriptApi, Transcript
    assert "languages" in inspect.signature(YouTubeTranscriptApi.fetch).parameters
    assert list(inspect.signature(YouTubeTranscriptApi.list).parameters) == ["self", "video_id"]
    assert callable(Transcript.fetch)
    assert isinstance(content._new_transcript_api(), YouTubeTranscriptApi)


def test_caption_requests_are_time_bounded():
    """Every request the caption client makes carries a timeout, and no new request is started once the
    client's overall budget is spent. No request leaves this test: the transport is replaced."""
    import requests
    from requests.adapters import BaseAdapter

    class Recorder(BaseAdapter):
        def __init__(self):
            super().__init__()
            self.timeouts = []

        def send(self, request, **kwargs):
            self.timeouts.append(kwargs.get("timeout"))
            response = requests.Response()
            response.status_code, response.url, response.request = 200, request.url, request
            response._content, response._content_consumed = b"", True
            return response

        def close(self):
            pass

    api = content._new_transcript_api()
    session = api._fetcher._http_client         # the session the library sends every request through
    assert isinstance(session, requests.Session)
    recorder = Recorder()
    session.mount("https://", recorder)
    session.get("https://www.youtube.com/watch?v=" + VIDEO_ID)
    session.post("https://www.youtube.com/youtubei/v1/player", json={})
    session.get("https://www.youtube.com/api/timedtext", timeout=3)     # an explicit timeout is kept
    assert recorder.timeouts == [content._TRANSCRIPT_TIMEOUT, content._TRANSCRIPT_TIMEOUT, 3]
    assert all(0 < t <= 15 for t in content._TRANSCRIPT_TIMEOUT) and 0 < content._TRANSCRIPT_BUDGET_S <= 30

    spent = content._bounded_http_session(budget_s=0.0)
    spent.mount("https://", recorder)
    with pytest.raises(requests.exceptions.Timeout):
        spent.get("https://www.youtube.com/watch?v=" + VIDEO_ID)
    assert len(recorder.timeouts) == 3          # refused before anything was sent
    assert "took too long" in content._transcript_error(requests.exceptions.Timeout("budget"))


def test_fetch_transcript_success_is_cleaned_and_cached(tmp_path, monkeypatch):
    api = _use(monkeypatch, FakeApi(fetch=_fetched([("hello\nthere", 0.0, 2.5), ("   ", 2.5, 1.0),
                                                     ("general kenobi", 3.5, 2.0)])))
    result = content.fetch_transcript(VIDEO_ID, tmp_path)
    assert result == {"available": True, "error": None, "snippets": [
        {"start": 0.0, "duration": 2.5, "text": "hello there"},
        {"start": 3.5, "duration": 2.0, "text": "general kenobi"}]}
    assert api.calls == [("fetch", VIDEO_ID, ("en", "en-US", "en-GB"))]
    cache_file = tmp_path / "transcripts" / f"{VIDEO_ID}.json"
    assert json.loads(cache_file.read_text())["snippets"] == result["snippets"]

    # second call: served from the cache, the library is not asked again (works offline)
    offline = _use(monkeypatch, FakeApi(fetch=OSError("no network")))
    assert content.fetch_transcript(VIDEO_ID, tmp_path) == result
    assert offline.calls == []


def _library_errors():
    import requests
    import youtube_transcript_api as yt
    return [
        (yt.TranscriptsDisabled(VIDEO_ID), "turned off"),
        (yt.VideoUnavailable(VIDEO_ID), "unavailable"),
        (yt.IpBlocked(VIDEO_ID), "blocking"),
        (yt.RequestBlocked(VIDEO_ID), "blocking"),
        (yt.AgeRestricted(VIDEO_ID), "age-restricted"),
        (yt.CouldNotRetrieveTranscript(VIDEO_ID), "could not be retrieved"),
        (requests.exceptions.ConnectionError("dns failure"), "internet connection"),
        (requests.exceptions.Timeout("slow"), "internet connection"),
        (RuntimeError("something unexpected"), "RuntimeError"),
    ]


@pytest.mark.parametrize("case", range(9))
def test_fetch_transcript_never_raises_and_gives_a_plain_reason(case, tmp_path, monkeypatch):
    exc, expected = _library_errors()[case]
    _use(monkeypatch, FakeApi(fetch=exc))
    result = content.fetch_transcript(VIDEO_ID, tmp_path)
    assert result["available"] is False and result["snippets"] == []
    assert expected in result["error"]
    assert "Traceback" not in result["error"] and "\n" not in result["error"]
    assert not (tmp_path / "transcripts").exists()             # failures are not cached


def test_fetch_transcript_falls_back_to_another_language_when_no_english(tmp_path, monkeypatch):
    import youtube_transcript_api as yt

    class OtherLanguage:
        def fetch(self):
            return _fetched([("bonjour tout le monde", 0.0, 2.0)])

    api = _use(monkeypatch, FakeApi(fetch=yt.NoTranscriptFound(VIDEO_ID, ["en"], None), listing=[OtherLanguage()]))
    result = content.fetch_transcript(VIDEO_ID, tmp_path)
    assert result["available"] is True and result["snippets"][0]["text"] == "bonjour tout le monde"
    assert [c[0] for c in api.calls] == ["fetch", "list"]


def test_fetch_transcript_with_no_captions_at_all(tmp_path, monkeypatch):
    import youtube_transcript_api as yt
    _use(monkeypatch, FakeApi(fetch=yt.NoTranscriptFound(VIDEO_ID, ["en"], None), listing=[]))
    result = content.fetch_transcript(VIDEO_ID, tmp_path)
    assert result == {"available": False, "error": "This video has no captions.", "snippets": []}


def test_fetch_transcript_empty_captions_and_bad_ids(tmp_path, monkeypatch):
    api = _use(monkeypatch, FakeApi(fetch=_fetched([("  ", 0.0, 1.0)])))
    assert content.fetch_transcript(VIDEO_ID, tmp_path)["error"] == "This video's captions are empty."
    for bad in ["", "../../etc/passwd", "short", None, "dQw4w9WgXcQ/.."]:
        result = content.fetch_transcript(bad, tmp_path)
        assert result["available"] is False and result["error"]
    assert len(api.calls) == 1                                  # bad ids never reach the library or the disk
    assert not (tmp_path / "transcripts").exists()


def test_fetch_transcript_when_the_library_is_missing(tmp_path, monkeypatch):
    def missing():
        raise ImportError("No module named 'youtube_transcript_api'")
    monkeypatch.setattr(content, "_new_transcript_api", missing)
    result = content.fetch_transcript(VIDEO_ID, tmp_path)
    assert result["available"] is False and "not installed" in result["error"]


def test_fetch_transcript_ignores_a_corrupt_cache_file(tmp_path, monkeypatch):
    (tmp_path / "transcripts").mkdir()
    (tmp_path / "transcripts" / f"{VIDEO_ID}.json").write_text("{not json")
    _use(monkeypatch, FakeApi(fetch=_fetched([("fresh text", 1.0, 2.0)])))
    result = content.fetch_transcript(VIDEO_ID, tmp_path)
    assert result["available"] is True and result["snippets"][0]["text"] == "fresh text"


def _auto_captions(n=120, step=3.0):
    """Unpunctuated, overlapping snippets like YouTube's auto-generated captions."""
    return [{"start": i * step, "duration": step * 2, "text": f"w{i}a w{i}b w{i}c w{i}d w{i}e"} for i in range(n)]


def test_transcript_sections_are_contiguous_and_keep_every_word():
    snippets = _auto_captions()
    sections = content.transcript_sections(snippets)
    assert " ".join(s["text"] for s in sections).split() == " ".join(s["text"] for s in snippets).split()
    assert [s["idx"] for s in sections] == list(range(len(sections)))
    assert sections[0]["start_s"] == 0.0
    for a, b in zip(sections, sections[1:]):
        assert a["end_s"] == b["start_s"]
    for s in sections[:-1]:
        assert s["end_s"] - s["start_s"] == pytest.approx(60.0)
    assert sections[-1]["end_s"] == pytest.approx(119 * 3.0 + 6.0)
    for s in sections:
        assert s["word_count"] == len(s["text"].split()) > 0
        assert set(s) == {"idx", "start_s", "end_s", "text", "word_count"}


def test_transcript_sections_cut_punctuated_captions_at_a_sentence_end():
    snippets = []
    for i in range(60):
        text = f"this is caption {i}" + ("." if i % 3 == 2 else "")   # a sentence ends every 12 s
        snippets.append({"start": i * 4.0, "duration": 4.0, "text": text})
    sections = content.transcript_sections(snippets, target_s=60.0)
    assert len(sections) >= 3
    for s in sections[:-1]:
        assert s["text"].endswith(".")
        assert 60.0 <= s["end_s"] - s["start_s"] < 75.0


def test_transcript_sections_cut_anyway_when_no_sentence_ends_within_the_slack():
    snippets = [{"start": i * 4.0, "duration": 4.0, "text": f"caption {i}" + ("." if i == 0 else "")}
                for i in range(60)]
    sections = content.transcript_sections(snippets, target_s=60.0)
    for s in sections[:-1]:
        assert 75.0 <= s["end_s"] - s["start_s"] < 75.0 + 4.0


def test_transcript_sections_fold_a_scrap_at_the_end_and_handle_empty_input():
    snippets = _auto_captions(n=21, step=3.0)                   # 63 s: one 60 s section plus a 3 s scrap
    sections = content.transcript_sections(snippets)
    assert len(sections) == 1 and sections[0]["word_count"] == 21 * 5
    assert content.transcript_sections([]) == []
    assert content.transcript_sections([{"start": 0, "duration": 1, "text": "  "}]) == []


def test_transcript_span_returns_the_captions_around_a_lapse():
    snippets = [{"start": i * 5.0, "duration": 5.0, "text": f"part{i}"} for i in range(20)]
    assert content.transcript_span(snippets, 42.0, 51.0) == "part7 part8 part9 part10 part11"
    assert content.transcript_span(snippets, 42.0, 51.0, pad=0.0) == "part8 part9 part10"
    assert content.transcript_span(snippets, 51.0, 42.0, pad=0.0) == "part8 part9 part10"   # swapped times
    assert content.transcript_span(snippets, 500.0, 510.0) == ""
    assert content.transcript_span([], 1.0, 2.0) == ""
    assert content.transcript_span(snippets, None, None) == ""


def test_transcript_span_does_not_repeat_overlapping_auto_captions():
    snippets = _auto_captions(n=10, step=3.0)                   # each stays on screen 6 s but is current for 3 s
    assert content.transcript_span(snippets, 9.0, 12.0, pad=0.0) == snippets[3]["text"]


# ---------------------------------------------------------------------------------------------- samples

def _reference_load_part():
    """lookaway_test.load_part and its constants, lifted from the source without importing the module
    (importing it would start pygame)."""
    tree = ast.parse((ROOT / "lookaway_test.py").read_text(encoding="utf-8"))
    wanted = [n for n in tree.body
              if (isinstance(n, ast.FunctionDef) and n.name == "load_part")
              or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in
                  ("TEXT", "TEXT_FIRST_LINE", "WORDS_PER_PART"))]
    namespace = {"re": re}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), "lookaway_test.py", "exec"), namespace)
    namespace["TEXT"] = str(ROOT / namespace["TEXT"])
    return namespace["load_part"]


def test_samples_are_parts_5_to_9_with_the_same_partition_as_the_lookaway_test():
    if not (ROOT / "lookaway_test.py").exists():
        pytest.skip("lookaway_test.py (the research script this is checked against) is not in this checkout")
    samples = content.list_samples()
    assert [s["id"] for s in samples] == [f"darwin-{n}" for n in range(5, 10)]
    load_part = _reference_load_part()
    for n, info in zip(range(5, 10), samples):
        assert info["title"] == f"On the Origin of Species, part {n}"
        sample = content.load_sample(info["id"])
        assert set(sample) == {"id", "title", "text"} and sample["id"] == info["id"]
        assert sample["text"].split("\n\n") == load_part(n)
        assert info["words"] == len(sample["text"].split())
        assert 3000 <= info["words"] < 3700


@pytest.mark.parametrize("bad", ["darwin-4", "darwin-10", "darwin-0", "darwin-", "5", "", None, "../darwin-5",
                                 "darwin-5/../../x"])
def test_load_sample_rejects_ids_it_does_not_offer(bad):
    with pytest.raises(KeyError):
        content.load_sample(bad)


def test_samples_when_the_text_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(content, "_setting", lambda name, default: tmp_path / "missing.txt")
    assert content.list_samples() == []
    with pytest.raises(KeyError):
        content.load_sample("darwin-5")
