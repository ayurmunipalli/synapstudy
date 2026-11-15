"""Shared constants for the SynapStudy app (APP_SPEC.md §6)."""
import os
from pathlib import Path

HOST, PORT = "127.0.0.1", 8770
# STUDY_FOCUS_PORT moves the server to another port (used to test a second copy next to a running app).
PORT = int(os.environ.get("STUDY_FOCUS_PORT") or PORT)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "app_data"
SESSIONS = DATA / "sessions"
CACHE = DATA / "cache"
CUES = DATA / "cues"
SHOTS = DATA / "shots"          # lapse screenshots, one folder per session (not named in the spec)

MODEL = "claude-opus-5-5"
EFFORT_FAST = "low"
EFFORT_MAIN = "medium"

ACTED_KINDS = ("look_away",)    # only validated kinds drive the UI
TEXT_MAX_CHARS = 200_000
PDF_MAX_BYTES = 40 * 1024 * 1024       # one uploaded PDF (study plans)
STUDY_MAX_RESOURCES = 12
# Reading sections are kept short so a whole section fits on one two-column page (integration, 2026-11-15).
SECTION_TARGET_WORDS, SECTION_MAX_WORDS = 110, 150
SAMPLE_TEXT = ROOT / "texts" / "origin_of_species_pg1228.txt"

# --- used by the server only -------------------------------------------------------------------------
WEB = ROOT / "app" / "web"
RESULTS = ROOT / "results"      # read-only: frozen detector parameters and recordings for replay
FROZEN_PARAMS = RESULTS / "lookaway_frozen_params.json"
DETECTOR_VARIANT = "motion_only"   # decided 2026-11-15, RESULTS.md "Demo live trigger"

# Head-turn sensitivity (added 2026-11-15, after the first live sessions felt far too sensitive): the 1-s head
# angle, in degrees, that counts as a look-away. It replaces theta in the frozen motion_only parameters; N is
# left alone. The choice is saved in <data>/settings.json as {"theta": 20} and read when the server starts.
# ONLY the tested value is the frozen one that was scored on a session it was not tuned on. The others were
# chosen after both recorded sessions had been looked at, and the sentence shown for them says so.
HEAD_TURN_CHOICES = (10, 15, 20, 30)
HEAD_TURN_TESTED = 10
HEAD_TURN_DEFAULT = 20
# The 10-degree wording follows RESULTS.md "Demo readiness" (2026-11-15): update it there first, then here.
# The other counts are the cued look-aways caught, and the drifts during normal reading, in the two recorded
# sessions (results/lookaway-20261114-234828.npz and -20261115-000831.npz) with motion_only, N 2, scored by
# focus/lookaway.py:score. Not held-out: do not reword them as if they were.
_SEEN = ("Both sessions had been seen before this setting was chosen, so this is not a held-out result.")
_HEAD_TURN_RESULT = {
    10: ("6/6 caught in 1 held-out session, 0 false alarms in 6 min of reading (RESULTS.md). "
         "This is the only setting tested on a session it was not tuned on."),
    15: ("caught 11 of 11 cued look-aways in the two recorded sessions, with none during 13 min of normal "
         "reading. " + _SEEN),
    20: ("caught 10 of 11 cued look-aways in the two recorded sessions (it missed one small downward look), "
         "with none during 13 min of normal reading. " + _SEEN),
    30: ("caught 10 of 11 cued look-aways in the two recorded sessions (it missed one small downward look), "
         "with none during 13 min of normal reading. " + _SEEN),
}


def head_turn_evidence(theta: int) -> str:
    """What is known about the head-turn trigger at this setting, as shown in the app (KeyError otherwise)."""
    return f"Head-turn look-away (gyroscope, not EEG), {theta}\u00b0 setting: {_HEAD_TURN_RESULT[theta]}"


AUDIO_DETAIL = "The Muse S Athena has no speaker; cues play through this Mac's sound output."
SLEEP_NOTE = ("Sleep replay is a prototype. The deep-sleep detector has no trained model for you yet "
              "and is not validated.")
# The "N2 or deeper" cue model trained on the nap of 2026-11-14 (Ayur's decision, RESULTS.md "Sleep cueing with
# an N2 or deeper model"). When this file exists the cue command points at it and the note below is shown.
SLEEP_MODEL = RESULTS / "nap-20261114-171747_model_n2n3.npz"
SLEEP_NOTE_MODEL = ("Sleep replay is a prototype. This command uses a detector trained on one nap to recognise "
                    "stable N2-or-deeper sleep. It has not been validated on another night, it does not detect "
                    "deep sleep (N3), and nothing yet shows that the cues improve memory.")
SCREENSHOT_MAX_BYTES = 2 * 1024 * 1024
LOG_MAX_BYTES = 5 * 1024 * 1024


def code_stamp() -> int:
    """Newest modification time (whole seconds) of the server's own code. The server reports the value it
    started with as "build" in /api/status; run_app.sh compares that with this function's current value and
    replaces a server that is running older code (added 2026-11-15 after an old server was reused all night)."""
    files = [p for p in (ROOT / "app").glob("*.py")] + [ROOT / "focus" / n for n in ("live.py", "lookaway.py", "clench_hold.py")]
    return int(max((p.stat().st_mtime for p in files if p.is_file()), default=0))


BUILD = code_stamp()      # fixed when this process imported the code
