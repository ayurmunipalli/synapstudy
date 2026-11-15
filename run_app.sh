#!/bin/bash
# SynapStudy: start the local server and open the app in ONE window (Chrome app mode).
#   ./run_app.sh            start and open the window; about 5 s after the window is closed, the server
#                           (if this script started it) and the app's own Chrome instance are stopped
#   ./run_app.sh --server   server only (then open http://127.0.0.1:8770 yourself)
# The headband stream is separate: keep ./stream_forever.sh running in Terminal.app for live detection.
# Without it the app still works, with SIMULATED lapses (D / R keys).
set -u
cd "$(dirname "$0")" || exit 1
ROOT="$(pwd)"
PY="$ROOT/.venv/bin/python"
URL="http://127.0.0.1:8770"
LOG="$ROOT/app_data/server.log"
CHROME_APP="/Applications/Google Chrome.app"
CHROME="$CHROME_APP/Contents/MacOS/Google Chrome"
PROFILE="$ROOT/app_data/chrome-profile"   # the app window's own Chrome profile, separate from normal Chrome
mkdir -p "$ROOT/app_data"

if [ ! -x "$PY" ]; then echo "Missing $PY (the project virtual environment)."; exit 1; fi

# A server that is already running is reused only if it is running the CURRENT code. One started before the
# code last changed (it reports an older "build", or none) is stopped and replaced; otherwise fixes never
# take effect (2026-11-15: a server from the first launch was silently reused for hours).
if curl -s -o /dev/null --max-time 1 "$URL/api/status"; then
  WANT="$("$PY" -c 'from app import config; print(config.code_stamp())' 2>/dev/null)"
  HAVE="$(curl -s --max-time 2 "$URL/api/status" | "$PY" -c '
import json, sys
try:
    print(json.load(sys.stdin).get("build"))
except Exception:
    print("unknown")
')"
  if [ -n "$WANT" ] && [ "$HAVE" != "$WANT" ]; then
    echo "A SynapStudy server is already running older code. Replacing it."
    for pid in $(lsof -nP -t -iTCP:8770 -sTCP:LISTEN 2>/dev/null); do
      case "$(ps -o command= -p "$pid" 2>/dev/null)" in
        *app.server*) kill "$pid" 2>/dev/null ;;
        *) echo "Port 8770 is in use by another program (pid $pid); not touching it."; exit 1 ;;
      esac
    done
    for _ in $(seq 1 40); do
      curl -s -o /dev/null --max-time 1 "$URL/api/status" || break
      sleep 0.25
    done
  fi
fi

# Reuse a server that is already running (and current); otherwise start one.
STARTED=0
if ! curl -s -o /dev/null --max-time 1 "$URL/api/status"; then
  "$PY" -m app.server >>"$LOG" 2>&1 &
  SERVER_PID=$!
  STARTED=1
  for _ in $(seq 1 60); do
    curl -s -o /dev/null --max-time 1 "$URL/api/status" && break
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "The server stopped. See $LOG"; tail -20 "$LOG"; exit 1; fi
    sleep 0.25
  done
  if ! curl -s -o /dev/null --max-time 1 "$URL/api/status"; then echo "The server did not start. See $LOG"; exit 1; fi
fi
# Main process of the Chrome instance that runs on this app's own profile folder (helper processes carry
# --type=). Matched on its --user-data-dir argument, so the user's normal Chrome is never touched.
app_chrome_pids() {
  ps -axww -o pid=,command= 2>/dev/null | NEEDLE="--user-data-dir=$PROFILE" awk \
    'index($0, ENVIRON["NEEDLE"]) && index($0, "Google Chrome") && !index($0, "--type=") { print $1 }'
}

# Open WebSocket connections (app windows) as the server reports them: a number, "missing" when the server
# answers without a "clients" field (one started before that field existed), "down" when it does not answer.
clients() {
  curl -s --max-time 2 "$URL/api/status" | "$PY" -c '
import json, sys
try:
    n = json.load(sys.stdin).get("clients")
except Exception:
    print("down")
else:
    print(n if isinstance(n, int) and not isinstance(n, bool) and n >= 0 else "missing")
'
}

QUIT_CHROME=0
cleanup() {
  [ "$STARTED" = 1 ] && kill "$SERVER_PID" 2>/dev/null
  if [ "$QUIT_CHROME" = 1 ]; then
    for pid in $(app_chrome_pids); do kill "$pid" 2>/dev/null; done
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if [ "${1:-}" = "--server" ]; then
  echo "SynapStudy is running at $URL  (Ctrl+C to stop)"
  [ "$STARTED" = 1 ] && wait "$SERVER_PID"
  exit 0
fi

# Chrome is started through LaunchServices ("open"), not as a child of this script: launched from Finder, a
# child of SynapStudy.app is expected to be refused the camera without a prompt. Its own profile folder
# keeps the app window separate from normal Chrome.
if [ -x "$CHROME" ] && open -n -a "$CHROME_APP" --args --app="$URL" --user-data-dir="$PROFILE" \
    --no-first-run --no-default-browser-check --window-size=1512,945 \
    --autoplay-policy=no-user-gesture-required; then
  # Chrome on macOS keeps running after its last window is closed, so "the window was closed" is read from
  # the server: once a window has connected and the count of connections then stays at 0 for 5 s (long
  # enough for a page reload), stop. Quitting that Chrome instance (Cmd+Q) also ends the wait. A server
  # without the "clients" count, or one that is not answering, leaves only the Chrome process to wait on.
  QUIT_CHROME=1
  seen_chrome=0; chrome_gone=0; seen_client=0; idle=0; waited=0
  while :; do
    sleep 1
    if [ -n "$(app_chrome_pids)" ]; then seen_chrome=1; chrome_gone=0; else chrome_gone=$((chrome_gone + 1)); fi
    if [ "$seen_chrome" = 1 ] && [ "$chrome_gone" -ge 2 ]; then break; fi
    if [ "$seen_chrome" = 0 ] && [ "$seen_client" = 0 ] && [ "$chrome_gone" -ge 30 ]; then
      echo "Google Chrome did not start within 30 s."; break
    fi
    n="$(clients)"
    case "$n" in
      down)    seen_client=0; idle=0; waited=0 ;;      # e.g. the server is being restarted: start over
      missing) ;;
      0)
        if [ "$seen_client" = 1 ]; then
          idle=$((idle + 1))
          if [ "$idle" -ge 5 ]; then break; fi
        else
          waited=$((waited + 1))
          if [ "$waited" -ge 30 ]; then echo "No app window connected to $URL within 30 s."; break; fi
        fi ;;
      *)       seen_client=1; idle=0 ;;
    esac
  done
else
  echo "Google Chrome was not found or did not open. Opening $URL in your default browser instead (Ctrl+C here to stop)."
  open "$URL"
  [ "$STARTED" = 1 ] && wait "$SERVER_PID"
fi
