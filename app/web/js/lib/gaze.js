// Where the reader is looking: WebGazer (webcam), the mouse pointer (testing), or nothing.
// All three feed one 15-second history of {x, y, t} points in viewport pixels (t = epoch ms).
//
// Webcam gaze is coarse (typically a line or more of text off, and worse as the head moves). After the
// click calibration its error is measured on dots the reader only looks at, and reported in
// status().error_px, so the view can draw a mark as wide as the gaze really is. (The check and the
// wider mark were added on 2026-11-15 and exercised only against a fake WebGazer, not yet a camera.)
//
// WebGazer 3.5.3 is vendored at /vendor/webgazer/ and loaded only when webcam mode is chosen.
// Notes from reading its source:
//  - params.faceMeshSolutionPath is read when the face tracker first initialises, so it is set before
//    begin(); every model file then comes from this server, nothing from the internet.
//  - begin() alert()s on any plain-http host other than "localhost". This app is served from
//    127.0.0.1 (still a secure context, so the camera works); the alert is silenced for that one call.
//  - begin() also starts training on every click and mouse move. Those listeners are removed so that
//    only the explicit calibration clicks (recordScreenPosition) train the model.
//  - end() removes WebGazer's elements but leaves the camera on, so the stream is stopped here.
//  - params.applyKalmanFilter (on by default, set again here) makes each regression smooth its own
//    output; the gaze listener receives that filtered point. WebGazer's extra 4-frame average only
//    moves its own dot, which is hidden.
//  - recordScreenPosition() adds one training pair to the model; clearData() empties the model (and
//    resets the Kalman filter). "Add more points" therefore just records more pairs without clearing.

import { h } from './dom.js';

const WEBGAZER_SRC = '/vendor/webgazer/webgazer.js';
const FACE_MESH_PATH = '/vendor/webgazer/mediapipe/face_mesh';
const HISTORY_MS = 15000;         // covers the widest lost-focus window below, with room to spare
const CURRENT_MS = 700;           // current() is the median of this much of the most recent gaze
const FACE_TIMEOUT_MS = 1000;

// Where focus was lost. The detector's onset comes about 0.5-1.2 s AFTER the head began to turn (the
// 1-s rotation has to pass the threshold and persist), and the eyes leave the text before the head
// does. So the place is read from gaze samples taken before the movement, not up to the onset:
// Changed 2026-11-15 (Ayur: looking off to the LEFT from the right side of the page marked the left side):
// the window now ends 1.8 s before the onset (was 0.8 s), before the head starts turning. While the head
// turns, webcam gaze drifts toward the turn and can still land on the passage. The place may now be a few
// words earlier than where reading stopped, which is the safer error.
const LOST_FROM_MS = 3500;        // window start, this long before the reported onset (was 2500)
const LOST_TO_MS = 1800;          // window end, this long before the reported onset (was 800)
const LOST_WIDE_FROM_MS = 5500;   // start of the one wider window tried when the first has too few points (was 4000)
const LOST_MIN_SAMPLES = 8;       // fewer accepted points than this in a window: do not trust it
const LOST_PLAN = [
  { from: LOST_FROM_MS, to: LOST_TO_MS, min: LOST_MIN_SAMPLES },
  { from: LOST_WIDE_FROM_MS, to: LOST_TO_MS, min: LOST_MIN_SAMPLES },
];
// The mouse pointer (testing) does not turn with the head, and a simulated drift has no real onset:
// the pointer is read just before the event, as before.
const MOUSE_PLAN = [{ from: 1000, to: 200, min: 1 }];

// Calibration: 9 dots x 5 clicks, then an accuracy check on dots that are looked at but never clicked.
const CALIB_CLICKS = 5;
const CALIB_GRID = [8, 50, 92];   // percent of the calibration area, both axes: 3 x 3 = 9 dots
// "Add more points": 9 further dots x 3 clicks, added to the same model. Positions are fractions
// [x, y] of the passage area, where the reading happens. They are kept away from the first-pass dots
// and from the check dots: the check must not be measured at a point the model was trained on.
const EXTRA_CLICKS = 3;
const EXTRA_POINTS = [
  [0.13, 0.1], [0.36, 0.1], [0.87, 0.1],
  [0.13, 0.42], [0.36, 0.42], [0.87, 0.42],
  [0.13, 0.9], [0.36, 0.9], [0.87, 0.9],
];
// The accuracy check: the centre of the passage area and four off-centre points (the middle of each
// column, upper and lower), one at a time.
const CHECK_POINTS = [[0.5, 0.5], [0.22, 0.25], [0.78, 0.75], [0.78, 0.25], [0.22, 0.75]];
const CHECK_LEAD_MS = 2000;       // time to read the instruction before the first dot
const CHECK_DOT_MS = 2000;        // each dot is shown this long
const CHECK_SKIP_MS = 700;        // ...and its first part is ignored (the eyes are still travelling)
const CHECK_MIN_SAMPLES = 8;      // a dot with fewer predictions than this is not scored (face lost)
const CHECK_MIN_DOTS = 3;         // fewer scored dots than this: the check could not be done

let mode = 'none';
let calibrated = false;
let errorPx = null;         // gaze error measured by the last accuracy check (px), or null
let note = null;            // why webcam gaze is off, when it failed
let samples = [];
let lastFaceMs = 0;
let lastPositions = null;
let generation = 0;         // bumped by stop(); lets a slow start() notice it was abandoned
let pendingBegin = null;
let scriptPromise = null;
let dotEl = null;
let dotWanted = false;
let calib = null;           // { promise, cancel } while the calibration screen is up

// mouse mode
let pointer = null;
let pointerTimer = null;

function usable() {
  return mode === 'mouse' || (mode === 'webgazer' && calibrated);
}

function median(values) {
  const v = values.slice().sort((a, b) => a - b);
  const mid = v.length >> 1;
  return v.length % 2 ? v[mid] : (v[mid - 1] + v[mid]) / 2;
}

function medianPoint(from, to) {
  const xs = [];
  const ys = [];
  let t = 0;
  for (const s of samples) {
    if (s.t < from || s.t > to) continue;
    xs.push(s.x);
    ys.push(s.y);
    t = s.t;
  }
  return xs.length ? { x: median(xs), y: median(ys), t } : null;
}

// ---------- pure helpers (no DOM, no WebGazer; exported so they can be checked on their own) ----------

// Where the reader was before looking AWAY (added 2026-11-15 after Ayur's report: "if I'm on the right side
// of a passage and look off the screen to the left, it marks the left side"). The eyes leave before the
// head turns, and on the way out they sweep ACROSS the passage, so the fixed window before the detector's
// onset picked up points on the exit side. Instead: find the last gaze point on the passage (where the eyes
// left it), walk back over the sweep that led there (100-ms bins moving toward that exit point faster than
// SWEEP_SPEED_PX_S), and read the place from the gaze BEFORE the sweep began. The speed and windows are
// chosen from the size of a reading saccade versus a look-away, not from recorded gaze: untested on a camera.
const SWEEP_LOOKBACK_MS = 8000;    // how far back before the onset to look for the departure
const SWEEP_BIN_MS = 100;
const SWEEP_SPEED_PX_S = 600;      // faster than this toward the exit point = part of the sweep away
const DEPART_GAP_MS = 150;         // window end, this long before the sweep began
const DEPART_PLAN = [1500, 3000];  // window lengths tried in turn (ms before the window end)

/** {x, y, n, depart_ms} = the median of accepted points before the gaze swept off the passage, or null
 *  (no on-passage point in the last SWEEP_LOOKBACK_MS, or too few points before the sweep). */
export function departurePoint(history, onsetMs, accept, min = LOST_MIN_SAMPLES) {
  if (!Number.isFinite(onsetMs)) return null;
  const ok = (s) => typeof accept !== 'function' || accept(s);
  const recent = history.filter((s) => s.t >= onsetMs - SWEEP_LOOKBACK_MS && s.t <= onsetMs);
  let exit = null;
  for (let i = recent.length - 1; i >= 0; i -= 1) if (ok(recent[i])) { exit = recent[i]; break; }
  if (!exit) return null;
  // 100-ms bins of the on-passage gaze, newest first, ending at the exit point
  const bins = [];
  for (let end = exit.t; end > onsetMs - SWEEP_LOOKBACK_MS; end -= SWEEP_BIN_MS) {
    const pts = recent.filter((s) => s.t > end - SWEEP_BIN_MS && s.t <= end && ok(s));
    bins.push(pts.length ? { x: median(pts.map((p) => p.x)), y: median(pts.map((p) => p.y)), t: end } : null);
  }
  const dist = (b) => Math.hypot(b.x - exit.x, b.y - exit.y);
  let depart = exit.t;
  const step = SWEEP_SPEED_PX_S * SWEEP_BIN_MS / 1000;
  let prev = bins[0];
  for (let k = 1; k < bins.length; k += 1) {
    const b = bins[k];
    if (!b) continue;                                  // a gap in the samples: judge across it
    const gapBins = Math.round((prev.t - b.t) / SWEEP_BIN_MS);
    if (dist(b) - dist(prev) <= step * gapBins) break; // not moving toward the exit fast enough: sweep began here
    depart = b.t;
    prev = b;
  }
  const to = Math.min(depart - DEPART_GAP_MS, onsetMs - LOST_TO_MS);   // and never after the head began to turn
  for (const len of DEPART_PLAN) {
    const pts = history.filter((s) => s.t >= to - len && s.t <= to && ok(s));
    if (pts.length >= min) {
      return { x: median(pts.map((p) => p.x)), y: median(pts.map((p) => p.y)), n: pts.length, depart_ms: onsetMs - depart };
    }
  }
  return null;
}

/** Where the gaze was before a lapse. `history` is [{x, y, t}], `accept(point)` keeps only the points
 *  that count (the view passes "is on the passage"). Each window of `plan` is tried in turn; the first
 *  with at least `min` accepted points gives {x, y, n, from_ms} (medians). Null when none has enough. */
export function lostFocusPoint(history, onsetMs, accept, plan = LOST_PLAN) {
  if (!Number.isFinite(onsetMs)) return null;
  for (const { from, to, min } of plan) {
    const xs = [];
    const ys = [];
    for (const s of history) {
      if (s.t < onsetMs - from || s.t > onsetMs - to) continue;
      if (typeof accept === 'function' && !accept(s)) continue;
      xs.push(s.x);
      ys.push(s.y);
    }
    if (xs.length >= min) return { x: median(xs), y: median(ys), n: xs.length, from_ms: from };
  }
  return null;
}

/** Which word a gaze point belongs to, choosing the LINE first. `rects` are the word rectangles of the
 *  page in reading order ({left, right, top, bottom}). Words are grouped into lines, and lines into
 *  columns (lines whose horizontal extents overlap). The answer is the column nearest the point in x,
 *  then the line in it whose vertical centre is nearest in y, then the word on that line nearest in x.
 *  Returns the index into `rects`, or -1. (Nearest-word-in-2-D, used before, hops between lines.) */
export function snapToLine(rects, pt) {
  const lines = [];
  let line = null;
  let prev = null;
  rects.forEach((r, i) => {
    if (!r || !(r.right > r.left) || !(r.bottom > r.top)) return;   // not laid out
    const height = r.bottom - r.top;
    const cy = (r.top + r.bottom) / 2;
    // A new line starts when the word sits at another height, or jumps back to the left, or is far to
    // the right of the previous word (the first line of the next column).
    if (!line || Math.abs(cy - line.cy) > height / 2 || r.left < prev.left || r.left - prev.right > 1.5 * height) {
      line = { cy, left: r.left, right: r.right, words: [] };
      lines.push(line);
    }
    line.left = Math.min(line.left, r.left);
    line.right = Math.max(line.right, r.right);
    line.words.push({ i, left: r.left, right: r.right });
    prev = r;
  });
  if (!lines.length || !pt) return -1;

  const columns = [];
  for (const ln of lines) {
    let col = columns.find((c) => ln.left < c.right && ln.right > c.left);
    if (!col) {
      col = { left: ln.left, right: ln.right, lines: [] };
      columns.push(col);
    }
    col.left = Math.min(col.left, ln.left);
    col.right = Math.max(col.right, ln.right);
    col.lines.push(ln);
  }

  const offX = (a) => Math.max(a.left - pt.x, 0, pt.x - a.right);
  const nearest = (items, cost) => items.reduce((best, item) => (cost(item) < cost(best) ? item : best));
  const column = nearest(columns, offX);
  const chosen = nearest(column.lines, (ln) => Math.abs(ln.cy - pt.y));
  return nearest(chosen.words, offX).i;
}

/** The accuracy check's arithmetic. `dots` is [{x, y, samples: [{x, y}]}]: where each dot was, and the
 *  predictions made while the reader looked at it. A dot is scored when it has at least `minSamples`
 *  predictions: its error is the distance from the dot to the median prediction. The result is the
 *  median of those errors, or error_px null when fewer than `minDots` dots could be scored. */
export function gazeErrorFrom(dots, minSamples = CHECK_MIN_SAMPLES, minDots = CHECK_MIN_DOTS) {
  const distances = [];
  for (const d of dots) {
    if (!d || !Array.isArray(d.samples) || d.samples.length < minSamples) continue;
    const mx = median(d.samples.map((s) => s.x));
    const my = median(d.samples.map((s) => s.y));
    distances.push(Math.hypot(mx - d.x, my - d.y));
  }
  return {
    error_px: distances.length >= minDots ? Math.round(median(distances)) : null,
    used: distances.length,
    total: dots.length,
    distances,
  };
}

/** "about 110 px" : rounded so that it does not read as more exact than it is. */
export function roundGazeError(errorPx) {
  const step = errorPx >= 50 ? 10 : 5;
  return Math.max(step, Math.round(errorPx / step) * step);
}

/** Lines to step back up from the snapped word, to be safe (Ayur, 2026-11-15): half the gaze error in lines of
 *  text, rounded up. The error in lines is rounded to the nearest half line, as the accuracy message shows it. */
export function safetyLines(errorPx, lineHeightPx) {
  if (!(errorPx > 0) || !(lineHeightPx > 0)) return 0;
  const lines = Math.round((errorPx / lineHeightPx) * 2) / 2;
  return Math.ceil(lines / 2);
}

/** The word `n` lines above word `i`, in reading order (across columns: above the top line of a column is the
 *  bottom line of the column before), nearest to word i in x. Lines are grouped as in snapToLine. Returns an
 *  index into `rects`; i itself when n <= 0; the nearest word on the first line when there aren't n lines above. */
export function wordLinesAbove(rects, i, n) {
  if (!(n > 0) || !rects[i]) return i;
  const lines = [];
  let line = null;
  let prev = null;
  rects.forEach((r, k) => {
    if (!r || !(r.right > r.left) || !(r.bottom > r.top)) return;
    const height = r.bottom - r.top;
    const cy = (r.top + r.bottom) / 2;
    if (!line || Math.abs(cy - line.cy) > height / 2 || r.left < prev.left || r.left - prev.right > 1.5 * height) {
      line = { cy, words: [] };
      lines.push(line);
    }
    line.words.push(k);
    prev = r;
  });
  const at = lines.findIndex((ln) => ln.words.includes(i));
  if (at < 0) return i;
  const target = lines[Math.max(0, at - n)];
  const cx = (rects[i].left + rects[i].right) / 2;
  let best = target.words[0];
  for (const k of target.words) {
    const c = (rects[k].left + rects[k].right) / 2;
    if (Math.abs(c - cx) < Math.abs((rects[best].left + rects[best].right) / 2 - cx)) best = k;
  }
  return best;
}

/** "about 110 px, roughly 2.5 lines of text". Lines are left out when the line height is unknown. */
export function describeGazeError(errorPx, lineHeightPx) {
  const px = `about ${roundGazeError(errorPx)} px`;
  if (!(lineHeightPx > 0)) return px;
  const lines = Math.round((errorPx / lineHeightPx) * 2) / 2;
  if (lines < 1) return `${px}, less than 1 line of text`;
  return `${px}, roughly ${lines} ${lines === 1 ? 'line' : 'lines'} of text`;
}

function push(x, y, t) {
  samples.push({ x, y, t });
  const cutoff = t - HISTORY_MS;
  let drop = 0;
  while (drop < samples.length && samples[drop].t < cutoff) drop += 1;
  if (drop) samples.splice(0, drop);
  if (dotEl) placeDot();
}

function placeDot() {
  const p = gaze.current();
  dotEl.hidden = !p;
  if (p) dotEl.style.transform = `translate(${Math.round(p.x)}px, ${Math.round(p.y)}px)`;
}

function syncDot() {
  if (dotWanted && mode !== 'none' && !dotEl) {
    dotEl = h('div', { class: 'gaze-dot', 'aria-hidden': 'true', hidden: true });
    document.body.appendChild(dotEl);
    placeDot();
  } else if ((!dotWanted || mode === 'none') && dotEl) {
    dotEl.remove();
    dotEl = null;
  }
}

// ---------- WebGazer ----------

function loadWebgazer() {
  if (window.webgazer) return Promise.resolve(window.webgazer);
  if (!scriptPromise) {
    scriptPromise = new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = WEBGAZER_SRC;
      script.async = true;
      script.onload = () => {
        if (window.webgazer) resolve(window.webgazer);
        else reject(new Error('WebGazer loaded but did not start.'));
      };
      script.onerror = () => {
        scriptPromise = null;
        script.remove();
        reject(new Error('The WebGazer script could not be loaded from this app.'));
      };
      document.head.appendChild(script);
    });
  }
  return scriptPromise;
}

// Called by WebGazer once per camera frame; `data` is null when it has no prediction.
function onPrediction(data) {
  const now = Date.now();
  try {
    // The tracker builds a new landmark array for every frame in which it finds a face, and leaves the
    // old one in place otherwise, so a changed reference means "a face was seen in this frame".
    const positions = window.webgazer.getTracker().getPositions();
    if (positions && positions !== lastPositions) {
      lastPositions = positions;
      lastFaceMs = now;
    }
  } catch {
    /* tracker not ready yet */
  }
  if (data && Number.isFinite(data.x) && Number.isFinite(data.y)) push(data.x, data.y, now);
}

function faceSeen() {
  return Date.now() - lastFaceMs < FACE_TIMEOUT_MS;
}

function shutdownWebgazer() {
  const wg = window.webgazer;
  if (wg) {
    try {
      wg.pause();
      wg.clearGazeListener();
    } catch {
      /* never started */
    }
  }
  const video = document.getElementById('webgazerVideoFeed');
  const stream = video && video.srcObject;
  if (stream && typeof stream.getTracks === 'function') stream.getTracks().forEach((track) => track.stop());
  if (wg) {
    try {
      wg.end();   // throws if begin() never built its elements
    } catch {
      /* nothing to remove */
    }
  }
  for (const id of ['webgazerVideoContainer', 'webgazerGazeDot']) document.getElementById(id)?.remove();
  lastPositions = null;
  lastFaceMs = 0;
}

function describeFailure(err) {
  const name = err && err.name;
  if (name === 'NotAllowedError' || name === 'SecurityError') {
    return 'Camera access was refused, so webcam gaze is off.';
  }
  if (name === 'NotFoundError' || name === 'OverconstrainedError') return 'No camera was found, so webcam gaze is off.';
  if (name === 'NotReadableError') return 'The camera is in use by another app, so webcam gaze is off.';
  const detail = err && err.message ? err.message : String(err);
  return `Webcam gaze could not start (${detail}).`;
}

async function startWebgazer(mine) {
  // Let an earlier, abandoned begin() settle first so two camera streams never overlap.
  if (pendingBegin) await pendingBegin.catch(() => {});
  if (mine !== generation) return;

  const wg = await loadWebgazer();
  if (mine !== generation) return;

  wg.params.faceMeshSolutionPath = FACE_MESH_PATH;
  wg.saveDataAcrossSessions(false);
  wg.applyKalmanFilter(true);   // smooths each prediction (WebGazer's default, stated so it stays on)
  wg.showVideoPreview(false);
  wg.showPredictionPoints(false);
  wg.showFaceOverlay(false);
  wg.showFaceFeedbackBox(false);
  wg.setGazeListener(onPrediction);

  const realAlert = window.alert;
  window.alert = () => {};
  let begun;
  try {
    begun = Promise.resolve(wg.begin());
  } finally {
    window.alert = realAlert;
  }
  pendingBegin = begun;
  try {
    await begun;
  } finally {
    if (pendingBegin === begun) pendingBegin = null;
  }

  wg.removeMouseEventListeners();
  if (mine !== generation) {
    shutdownWebgazer();
    return;
  }
  mode = 'webgazer';
}

// ---------- mouse ----------

function onPointerMove(e) {
  pointer = { x: e.clientX, y: e.clientY };
  push(pointer.x, pointer.y, Date.now());
}

function startMouse() {
  mode = 'mouse';
  calibrated = true;   // the pointer needs no calibration
  errorPx = null;      // ...and is exact: the view keeps its tight word circle
  window.addEventListener('pointermove', onPointerMove, { passive: true });
  // A resting pointer sends no events; repeat its last position so the history has no holes.
  pointerTimer = setInterval(() => {
    if (pointer) push(pointer.x, pointer.y, Date.now());
  }, 100);
}

function stopMouse() {
  window.removeEventListener('pointermove', onPointerMove);
  clearInterval(pointerTimer);
  pointerTimer = null;
  pointer = null;
}

// ---------- calibration screen ----------
//
// One screen, four phases:
//   click   9 dots, 5 clicks each. The first click empties the model, so this replaces a calibration.
//   check   5 dots shown one at a time and only looked at; gives the gaze error (gazeErrorFrom).
//   result  the error in px and in lines of text, with: Calibrate again / Add more points / Continue /
//           Skip gaze.
//   extra   ("Add more points") 9 more dots, 3 clicks each, added to the same model; then check again.
// Only "Continue" resolves true. Esc, or "Skip gaze", cancels from any phase.

function buildCalibration(root, wg, opts, resolve) {
  const wasCalibrated = calibrated;
  const wasError = errorPx;
  let phase = 'click';
  let recorded = 0;          // training clicks taken since this screen opened
  let replaceModel = true;   // the next training click starts a new model
  let needed = CALIB_CLICKS; // clicks per dot in the current pass
  let remaining = 0;
  let outcome = null;        // result of the last accuracy check (gazeErrorFrom)
  let stepTimer = null;
  let closed = false;

  const hint = h('p', { class: 'calib-hint' });
  const left = h('p', { class: 'calib-left' });
  const skip = h('button', { class: 'btn btn-ghost', type: 'button', onClick: () => close(false) }, 'Skip gaze (Esc)');
  const layer = h('div', { class: 'calib', role: 'dialog', 'aria-label': 'Gaze calibration' });
  let keyTargets = [skip];   // the buttons Space and Enter may press

  const refresh = () => {
    if (phase === 'result') return;
    const face = faceSeen();
    hint.textContent = face ? 'Face found' : 'No face found yet: sit centred, facing the screen';
    hint.classList.toggle('is-warn', !face);
    if (phase !== 'check') left.textContent = `${remaining} ${remaining === 1 ? 'dot' : 'dots'} left`;
  };

  // The passage area in viewport pixels (asked of the view each time, so it follows a resize), or the
  // middle of the calibration screen when the view gave none.
  function geometry() {
    const base = layer.getBoundingClientRect();
    let area = null;
    try {
      area = typeof opts.area === 'function' ? opts.area() : null;
    } catch {
      area = null;
    }
    if (!area || !(area.width >= 240) || !(area.height >= 160)) {
      area = { left: base.left + base.width * 0.1, top: base.top + base.height * 0.15, width: base.width * 0.8, height: base.height * 0.7 };
    }
    return { base, area };
  }

  // Put an element's centre at fraction (fx, fy) of the passage area; returns that point (viewport px).
  function place(el, fx, fy) {
    const { base, area } = geometry();
    const x = Math.round(area.left + fx * area.width - base.left);
    const y = Math.round(area.top + fy * area.height - base.top);
    el.style.left = `${x}px`;
    el.style.top = `${y}px`;
    return { x: base.left + x, y: base.top + y };
  }

  const onDot = (e) => {
    const dot = e.currentTarget;
    if (closed || dot.dataset.done || (phase !== 'click' && phase !== 'extra')) return;
    if (e.detail === 0) return;          // keyboard "clicks" carry no gaze
    if (!faceSeen()) {
      // WebGazer silently drops a click it cannot pair with an eye image; don't count it either.
      refresh();
      hint.classList.remove('is-nudge');
      void hint.offsetWidth;
      hint.classList.add('is-nudge');
      return;
    }
    if (replaceModel) {
      // A new calibration replaces the old model (and any mouse-move data WebGazer gathered at start).
      replaceModel = false;
      try {
        Promise.resolve(wg.clearData()).catch(() => {});
      } catch {
        /* storage unavailable: the in-memory model was still reset */
      }
      samples = [];
    }
    // The model is changing: it counts as calibrated again only when this screen ends with Continue.
    calibrated = false;
    errorPx = null;
    const r = dot.getBoundingClientRect();
    wg.recordScreenPosition(r.left + r.width / 2, r.top + r.height / 2, 'click');
    recorded += 1;
    const count = Number(dot.dataset.count || 0) + 1;
    dot.dataset.count = String(count);
    dot.style.setProperty('--fill', String(count / needed));
    if (count >= needed) {
      dot.dataset.done = '1';
      dot.disabled = true;
      remaining -= 1;
    }
    refresh();
    if (remaining === 0) startCheck();
  };

  function makeDot(style) {
    return h('button', {
      class: 'calib-dot', type: 'button', tabindex: '-1', 'aria-label': 'Calibration dot',
      style: { ...style, '--fill': '0' }, onClick: onDot,
    });
  }

  // A one-line header for the phases whose dots lie inside the passage area (it sits above that area).
  function strip(title) {
    return h('div', { class: 'calib-strip' }, h('strong', null, title), hint, left, skip);
  }

  function startClicks() {
    clearTimeout(stepTimer);
    phase = 'click';
    replaceModel = true;
    needed = CALIB_CLICKS;
    outcome = null;
    const dots = [];
    for (const top of CALIB_GRID) {
      for (const leftPct of CALIB_GRID) dots.push(makeDot({ left: `${leftPct}%`, top: `${top}%` }));
    }
    remaining = dots.length;
    keyTargets = [skip];
    layer.replaceChildren(
      h('div', { class: 'calib-info' },
        h('p', { class: 'eyebrow' }, 'Gaze calibration'),
        h('h2', { class: 'h2' }, 'Look at each dot and click it five times'),
        h('p', { class: 'muted' }, 'Keep your head still and follow the pointer with your eyes. Any order.')),
      h('div', { class: 'calib-status' }, hint, left, skip),
      ...dots);
    refresh();
  }

  function startExtra() {
    clearTimeout(stepTimer);
    phase = 'extra';
    replaceModel = false;      // added to the model, not instead of it
    needed = EXTRA_CLICKS;
    const dots = EXTRA_POINTS.map(() => makeDot({}));
    remaining = dots.length;
    keyTargets = [skip];
    layer.replaceChildren(strip('More points: look at each dot and click it three times'), ...dots);
    dots.forEach((dot, i) => place(dot, EXTRA_POINTS[i][0], EXTRA_POINTS[i][1]));
    refresh();
  }

  // The reader looks at each dot in turn without clicking. Nothing is trained here.
  function startCheck() {
    clearTimeout(stepTimer);
    phase = 'check';
    const target = h('div', { class: 'calib-target', 'aria-hidden': 'true', hidden: true });
    const dots = [];
    let i = -1;
    let shownAt = 0;
    let at = null;
    keyTargets = [skip];
    layer.replaceChildren(strip('Accuracy check: look at each dot until it moves. Do not click.'), target);
    left.textContent = 'Get ready';
    refresh();

    const step = () => {
      if (closed) return;
      const now = Date.now();
      if (i >= 0) {
        const from = shownAt + CHECK_SKIP_MS;
        dots.push({ x: at.x, y: at.y, samples: samples.filter((s) => s.t >= from && s.t <= now) });
      }
      i += 1;
      if (i >= CHECK_POINTS.length) {
        showResult(gazeErrorFrom(dots));
        return;
      }
      at = place(target, CHECK_POINTS[i][0], CHECK_POINTS[i][1]);
      target.hidden = false;
      target.classList.remove('is-in');
      void target.offsetWidth;
      target.classList.add('is-in');
      shownAt = now;
      left.textContent = `Dot ${i + 1} of ${CHECK_POINTS.length}`;
      stepTimer = setTimeout(step, CHECK_DOT_MS);
    };
    stepTimer = setTimeout(step, CHECK_LEAD_MS);
  }

  function showResult(result) {
    phase = 'result';
    outcome = result;
    let lineHeight = null;
    try {
      lineHeight = typeof opts.lineHeight === 'function' ? opts.lineHeight() : null;
    } catch {
      lineHeight = null;
    }
    const measured = result.error_px !== null;
    const title = measured
      ? `Gaze accuracy: ${describeGazeError(result.error_px, lineHeight)}`
      : 'The gaze accuracy check could not be done';
    const detail = measured
      ? `Measured just now on ${result.used} of ${result.total} dots, sitting still; it can get worse as you move. The circle that marks a lapse is sized from this.`
      : `The camera lost your face on too many dots (${result.used} of ${result.total} could be measured). Calibrate again, or continue with a circle of a default width.`;
    const act = (label, cls, fn, tip) => h('button', { class: cls, type: 'button', title: tip, onClick: fn }, label);
    const again = act('Calibrate again', 'btn', startClicks, 'Start over: 9 dots, 5 clicks each');
    const more = act('Add more points', 'btn', startExtra, '9 more dots, 3 clicks each, added to this calibration; then the check runs again');
    const go = act('Continue', 'btn btn-primary', () => close(true));
    keyTargets = [again, more, go, skip];
    layer.replaceChildren(h('div', { class: 'calib-result' },
      h('p', { class: 'eyebrow' }, 'Gaze calibration'),
      h('h2', { class: 'h2' }, title),
      h('p', { class: 'muted' }, detail),
      h('div', { class: 'calib-actions' }, again, more, go, skip)));
    go.focus();
  }

  // While the calibration screen is up it owns the keyboard: Esc cancels, nothing reaches the view.
  const onKey = (e) => {
    e.stopPropagation();
    if (e.key === 'Escape') {
      e.preventDefault();
      close(false);
    } else if (e.key === ' ' || e.key === 'Enter') {
      if (!keyTargets.includes(e.target)) e.preventDefault();
    }
  };

  const timer = setInterval(refresh, 300);

  function close(done) {
    if (closed) return;
    closed = true;
    clearInterval(timer);
    clearTimeout(stepTimer);
    window.removeEventListener('keydown', onKey, true);
    layer.remove();
    if (done) {
      calibrated = true;
      errorPx = outcome ? outcome.error_px : null;
      samples = [];
    } else if (recorded === 0) {
      // Cancelled before any click: an earlier calibration is still intact.
      calibrated = wasCalibrated;
      errorPx = wasError;
    } else {
      // Cancelled after a click: the model was changed and never confirmed.
      calibrated = false;
      errorPx = null;
    }
    calib = null;
    resolve(done);
  }

  window.addEventListener('keydown', onKey, true);
  root.appendChild(layer);
  startClicks();
  return () => close(false);
}

// ---------- public API ----------

export const gaze = {
  /** Start a gaze source: "webgazer" | "mouse" | "none". Never rejects: if the camera is refused or
   *  WebGazer cannot load, the mode falls back to "none" and status().note says why. */
  async start(requested) {
    gaze.stop();
    const mine = generation;
    note = null;
    if (requested === 'mouse') {
      startMouse();
    } else if (requested === 'webgazer') {
      try {
        await startWebgazer(mine);
      } catch (err) {
        console.warn('webcam gaze failed to start', err);
        if (mine === generation) {
          shutdownWebgazer();
          mode = 'none';
          note = describeFailure(err);
        }
      }
    }
    syncDot();
    return gaze.status();
  },

  /** Show the calibration inside overlayRoot: 9 dots x 5 clicks, then the accuracy check and its
   *  result (see "calibration screen" above). Resolves true when the reader chooses Continue, false if
   *  cancelled (Esc, Skip gaze) or if there is nothing to calibrate. Mouse mode resolves true.
   *  options.area(): the passage rectangle in viewport px, where the check dots go;
   *  options.lineHeight(): the passage line height in px, to say the error in lines of text. */
  calibrate(overlayRoot, options = {}) {
    if (mode === 'mouse') return Promise.resolve(true);
    if (mode !== 'webgazer' || !window.webgazer) return Promise.resolve(false);
    if (calib) return calib.promise;
    const root = overlayRoot || document.getElementById('overlay-root') || document.body;
    const entry = { promise: null, cancel: null };
    entry.promise = new Promise((resolve) => {
      entry.cancel = buildCalibration(root, window.webgazer, options || {}, resolve);
    });
    // buildCalibration can finish synchronously only via cancel(), so this assignment is safe.
    calib = entry;
    return entry.promise;
  },

  /** {x, y, t}: the median of the last 0.7 s, or null (no source, not calibrated, or no recent point). */
  current() {
    if (!usable()) return null;
    const now = Date.now();
    return medianPoint(now - CURRENT_MS, now);
  },

  /** Where the gaze was before a lapse whose onset the detector put at wall_ms: {x, y, n}. Webcam: the
   *  points before the gaze swept off the passage and before the head turned (departurePoint), else the
   *  median of [wall_ms - 3500, wall_ms - 1800], widened once to [wall_ms - 5500, ...] when there are
   *  fewer than 8. Null when there are still too few, or no usable source. Mouse: [wall_ms - 1000, - 200]. */
  at(wall_ms, accept) {
    if (!usable()) return null;
    // Webcam gaze: the place before the eyes swept off the passage; the fixed window is the fallback.
    const p = (mode === 'mouse' ? null : departurePoint(samples, wall_ms, accept))
      || lostFocusPoint(samples, wall_ms, accept, mode === 'mouse' ? MOUSE_PLAN : LOST_PLAN);
    return p ? { x: p.x, y: p.y, n: p.n } : null;
  },

  /** The gaze points of the last `ms` (default 12 s), rounded, for diagnosing a lapse's place later. */
  trace(ms = 12000) {
    const from = Date.now() - ms;
    return samples.filter((s) => s.t >= from).map((s) => ({ x: Math.round(s.x), y: Math.round(s.y), t: s.t }));
  },

  /** {mode, calibrated, face} plus `calibrating` (the calibration screen is up), `note` (why webcam
   *  gaze is off, or null) and `error_px` (the gaze error measured after calibration, or null when it
   *  was not measured; always null for the mouse). */
  status() {
    return {
      mode,
      calibrated,
      face: mode === 'mouse' ? true : mode === 'webgazer' ? faceSeen() : false,
      calibrating: Boolean(calib),
      note,
      error_px: mode === 'webgazer' && calibrated ? errorPx : null,
    };
  },

  /** Debugging aid: draw the smoothed gaze point. Off by default. */
  showDot(on) {
    dotWanted = Boolean(on);
    syncDot();
  },

  stop() {
    generation += 1;
    if (calib) calib.cancel();
    if (mode === 'mouse') stopMouse();
    // Also covers a begin() that is still waiting on the camera prompt: when it settles, startWebgazer
    // sees the new generation and shuts WebGazer down again.
    if (mode === 'webgazer') shutdownWebgazer();
    mode = 'none';
    calibrated = false;
    errorPx = null;
    samples = [];
    syncDot();
  },
};
