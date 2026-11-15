# SynapStudy

Study with a Muse S Athena headband on. SynapStudy notices when you look away while reading (head turn, from the
headband's gyroscope) and lets you mark a zone-out by holding a 3-second jaw clench (a jaw-muscle signal on the EEG
sensors). It takes you back to where you were in a PDF, or shows "Here's What You Missed" for a YouTube lecture.
It then makes flashcards from all your resources and, at night, records your sleep and plays quiet audio cues in
stable sleep (a prototype).

## Run

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=..." > .env          # optional: without it, summaries and labels fall back to extracts
MUSE_ADDRESS=<your headband address> ./stream_forever.sh   # in Terminal.app; leave it running
./run_app.sh                                  # opens the app at http://127.0.0.1:8770
```

Without a headband the app still runs, with SIMULATED lapses (keys D / R, and Z for a clench).

## Test

```bash
.venv/bin/python -m pytest app/tests focus -q
node app/tests/js/tracker_test.mjs && node app/tests/js/gaze_departure_test.mjs
```

## Layout

- `app/`: server (FastAPI), study plans, AI (Claude) and content, night mode, web front end (`app/web/`, with
  WebGazer 3.5.3 vendored under its GPLv3 licence).
- `focus/`: live detectors: look-away (`lookaway.py`, `live.py`) and clench hold (`clench_hold.py`).
- `sleep_cue.py`, `stage_nap.py`, `detector.py`, `common.py`: night recording, cueing and sleep staging.
- `results/`: the frozen detector settings and the sleep model the app loads (no recordings).
- `texts/`: a public-domain sample passage (Darwin, *On the Origin of Species*, Project Gutenberg #1228).
