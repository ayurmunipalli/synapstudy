// Shared app state and the session tracker.
//
// The tracker owns the client log of the session in progress (APP_SPEC §7): it turns the server's
// drift/resume messages into lapses, autosaves the log, and sends the final log when the session ends.
// It announces what it does on the bus: "lapse:start" {lapse, evt}, "lapse:end" {lapse, evt},
// "lapse:return" {lapse, evt} (the same lapse, extended to the turn back),
// "keyideas" {section_idx, ideas, status, error}, "clench:progress" {progress, cancelled, source},
// "clench:trigger" {source, wall_ms, onset_wall_ms} and "lapse:clench" {lapse}.

import { api } from './api.js';
import { bus } from './bus.js';
import { ws } from './ws.js';

export const state = {
  status: null,   // latest /api/status body (kept current by main.js)
  session: null,  // current session object, or null
};

const AUTOSAVE_MS = 10000;

// How a lapse ENDS (a heuristic, not validated: see TABLED.md). Rewritten 2026-11-15 after the first live
// sessions, where six head movements in a minute became ONE lapse pinned to the first one's word.
// The live trigger detects head MOVEMENT, not where the head is pointing. A look-away gives: drift
// (turning away) -> "resume" about 3 s later, while the reader may still be looking away -> a second
// drift when they turn back. The old rule took any drift within 20 s of a lapse's end as that turn back,
// so every movement in a busy minute was folded into the first lapse. Time alone cannot tell a turn back
// from a new look-away; direction can. Each drift now says which way the head rotated ("rot": gyro
// degrees [x, y, z] over the 1 s up to the drift), and each resume says the net rotation of the whole
// movement it ends ("net_rot"). A turn back is the reverse of the turn away.
//   - A drift opens a NEW lapse; its resume closes it at the moment the movement stopped ("movement_end").
//   - The one exception: the drift is the turn back from the previous lapse. Then the SAME lapse is
//     extended to this drift's onset ("turn_back") and "lapse:return" is announced, so the view can guide
//     the reader again. That needs ALL of:
//       * the previous lapse has the same source, is not simulated, and was not ended by hand;
//       * it has not been joined before (one turn away has one turn back);
//       * this drift began within JOIN_MS of that lapse's end and within JOIN_SPAN_MS of its start;
//       * both rotations are known: this drift's "rot", and the previous lapse's turn away (the "net_rot"
//         of the resume that closed it, or failing that its own drift's "rot");
//       * the two point in opposite directions: cosine <= JOIN_MAX_COS.
//     With no rotation information (an older server, or no motion data) nothing is ever joined.
//   - A movement that ended about where it began (net rotation under RETURNED_RATIO of its largest 1-s
//     angle) was a look away AND back inside one detector event. No turn back is still to come, so
//     nothing is joined to that lapse; without this its near-zero net rotation, which points nowhere
//     in particular, could pair with any later movement.
// What this is based on: the two recorded look-away sessions, both already looked at, so not a held-out
// result. All 10 clean away -> back pairs in them had a cosine of -0.98 to -1.00, and one stray movement
// followed 10 s later by a real look-away had +0.05. Net rotation over peak angle was 0.80 to 1.22 for
// every one-way movement and 0.12 for the one away-and-back event. The thresholds below were chosen from
// that; they have not been tested on a new session.
// Known limits: a turn back too slow for the detector leaves the lapse at "movement_end" ("at least N s"),
// and a later look-away the opposite way within JOIN_MS would then be taken for its turn back. All lapse
// lengths from the headband are approximate. Simulated events (D / R keys) simply end on the resume.
const CLEAR_MS = 2000;          // the detector reports "resume" 2 s after the movement stops
const JOIN_MS = 90000;
const JOIN_SPAN_MS = 120000;
const JOIN_MAX_COS = -0.5;
const RETURNED_RATIO = 0.5;
const REFRACTORY_MS = 2500;     // after a lapse is ended by hand, ignore the turn-back drift that follows

// The session in progress: { session, getPosition, log, open, held, timer, finishing } or null.
let cur = null;

function newId() {
  if (globalThis.crypto && typeof crypto.randomUUID === 'function') return crypto.randomUUID().replace(/-/g, '');
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}

function isNum(x) {
  return typeof x === 'number' && Number.isFinite(x);
}

function logPath(id) {
  return `/api/sessions/${id}/log`;
}

function save(c = cur) {
  if (!c) return Promise.resolve();
  return api.put(logPath(c.session.id), c.log).catch((err) => {
    console.warn('autosave failed:', err.message);
  });
}

function startAutosave(c) {
  clearInterval(c.timer);
  c.timer = setInterval(() => save(c), AUTOSAVE_MS);
}

/** A rotation vector from an event: three finite numbers that are not all zero, else null. */
function rotation(v) {
  if (!Array.isArray(v) || v.length !== 3 || !v.every(isNum)) return null;
  return Math.hypot(v[0], v[1], v[2]) > 0 ? v : null;
}

function cosine(a, b) {
  return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (Math.hypot(a[0], a[1], a[2]) * Math.hypot(b[0], b[1], b[2]));
}

/** Is this drift the turn back from the previous lapse? (The rule is spelled out at the top of this file.) */
function isTurnBack(c, source, at, evt) {
  const last = c.last;
  if (!last || source === 'simulated' || last.source !== source) return false;
  if (last.end_method === 'manual' || last.end_method === 'turn_back') return false;
  if (at - last.resume_ms > JOIN_MS || at - last.onset_ms > JOIN_SPAN_MS) return false;
  const back = rotation(evt.rot);
  if (!back || !c.lastAway) return false;
  return cosine(c.lastAway, back) <= JOIN_MAX_COS;
}

function positionFor(c, evt) {
  if (typeof c.getPosition !== 'function') return null;
  try {
    return c.getPosition(evt) ?? null;
  } catch (err) {
    console.error('getPosition failed', err);
    return null;
  }
}

function onDrift(evt) {
  // Only validated kinds drive the UI (acted === true); a drift while a lapse is open adds nothing.
  if (!cur || !evt || evt.acted !== true || cur.held || cur.finishing || cur.open) return;
  const now = Date.now();
  if (now < cur.quietUntil) return;
  const reported = isNum(evt.onset_wall_ms) ? evt.onset_wall_ms : isNum(evt.wall_ms) ? evt.wall_ms : now;
  const at = Math.round(Math.max(cur.log.started_ms, Math.min(reported, now)));
  const source = evt.source || (state.status && state.status.source) || 'simulated';

  // The reverse of the last lapse's turn away is the turn back: extend that lapse.
  if (isTurnBack(cur, source, at, evt)) {
    const last = cur.last;
    last.resume_ms = Math.max(last.resume_ms, at);
    last.duration_s = Math.round((last.resume_ms - last.onset_ms) / 10) / 100;
    last.end_method = 'turn_back';
    // Where the lapse began stays as it was; only a video's end time moves on.
    const position = positionFor(cur, { ...evt, type: 'resume' });
    if (position && isNum(position.t_end) && last.position && isNum(last.position.t_start)) {
      last.position = { ...last.position, t_end: Math.max(last.position.t_end ?? last.position.t_start, position.t_end) };
    }
    cur.swallowResume = true;      // the resume that ends this movement belongs to the same lapse
    save(cur);
    bus.emit('lapse:return', { lapse: last, evt });
    return;
  }

  const lapse = {
    id: newId(),
    kind: typeof evt.kind === 'string' ? evt.kind : 'look_away',
    // An event that does not say where it came from is never counted as live.
    source,
    onset_ms: at,
    resume_ms: null,       // null while the lapse is still open
    duration_s: null,
    end_method: null,      // how the end was decided (see the note at the top of this file)
    position: null,
    screenshot_url: null,
    angle: isNum(evt.angle) ? evt.angle : null,   // 1-s head angle (degrees) that set it off; null if not sent
  };
  cur.open = lapse;
  cur.openRot = rotation(evt.rot);
  cur.swallowResume = false;
  cur.log.lapses.push(lapse);
  lapse.position = positionFor(cur, evt);
  bus.emit('lapse:start', { lapse, evt });
}

function closeOpenLapse(c, wallMs, evt, method) {
  const lapse = c.open;
  if (!lapse) return;
  lapse.resume_ms = Math.round(Math.max(lapse.onset_ms, Math.min(wallMs, Date.now())));
  lapse.duration_s = Math.round((lapse.resume_ms - lapse.onset_ms) / 10) / 100;
  lapse.end_method = method || 'resume';
  const position = positionFor(c, evt);
  if (position && typeof position === 'object') lapse.position = { ...(lapse.position || {}), ...position };
  if (evt && isNum(evt.peak_angle)) lapse.peak_angle = evt.peak_angle;   // largest 1-s head angle, when known
  c.open = null;
  c.last = lapse;
  // Which way the head turned: the whole movement if the resume says so, else the drift's own second;
  // nothing if the head ended about where it began (away and back in one event: see the top of the file).
  const net = evt && Array.isArray(evt.net_rot) && evt.net_rot.length === 3 && evt.net_rot.every(isNum) ? evt.net_rot : null;
  const cameBack = Boolean(net) && isNum(evt.peak_angle) && Math.hypot(...net) < RETURNED_RATIO * evt.peak_angle;
  c.lastAway = cameBack ? null : rotation(net) || c.openRot;
  c.openRot = null;
  save(c);
  bus.emit('lapse:end', { lapse, evt });
}

function onResume(evt) {
  if (!cur) return;
  if (!cur.open) {                 // a resume with no open lapse: at most the end of a turn-back movement
    cur.swallowResume = false;
    return;
  }
  const at = evt && isNum(evt.wall_ms) ? evt.wall_ms : Date.now();
  const e = evt || { type: 'resume' };
  if (cur.open.source === 'simulated') closeOpenLapse(cur, at, e, 'simulated');
  else closeOpenLapse(cur, at - CLEAR_MS, e, 'movement_end');
}

function announce(c, sessionState) {
  ws.send({ type: 'session', id: c.session.id, state: sessionState });
}

/** Add key ideas to a session object without duplicating ones it already has. */
export function mergeKeyIdeas(session, ideas) {
  if (!session || !Array.isArray(ideas)) return;
  if (!Array.isArray(session.key_ideas)) session.key_ideas = [];
  const key = (k) => k.id ?? `${k.section_idx}:${k.text}`;
  const have = new Set(session.key_ideas.map(key));
  for (const idea of ideas) {
    if (!idea || have.has(key(idea))) continue;
    have.add(key(idea));
    session.key_ideas.push(idea);
  }
}

function onKeyIdeas(msg) {
  const session = state.session;
  if (!session || !msg || msg.session_id !== session.id) return;
  const ideas = Array.isArray(msg.ideas) ? msg.ideas : [];
  mergeKeyIdeas(session, ideas);
  if (typeof msg.status === 'string') session.key_ideas_status = msg.status;
  session.key_ideas_error = msg.error ?? null;
  bus.emit('keyideas', { section_idx: msg.section_idx, ideas, status: msg.status, error: msg.error ?? null });
}

// Jaw-clench hold (focus/clench_hold.py, frozen v2): the wearer's own "I zoned out" signal. A jaw-muscle signal
// picked up by the EEG sensors, not brain activity. Progress drives the on-screen ring; a completed hold is
// passed to the view as "clench:trigger", and the view decides where the lapse began (video: the concept the
// viewer picks; reading: gaze 10 s earlier) and calls tracker.addLapse. Ignored outside a running session.
function onClenchProgress(msg) {
  if (!cur || cur.finishing || !msg) return;
  bus.emit('clench:progress', { progress: isNum(msg.progress) ? msg.progress : 0, cancelled: Boolean(msg.cancelled),
    source: msg.source || 'simulated' });
}

function onClench(msg) {
  if (!cur || cur.finishing || !msg) return;
  const now = Date.now();
  bus.emit('clench:trigger', {
    source: msg.source || (state.status && state.status.source) || 'simulated',
    wall_ms: isNum(msg.wall_ms) ? Math.min(msg.wall_ms, now) : now,
    onset_wall_ms: isNum(msg.onset_wall_ms) ? Math.min(msg.onset_wall_ms, now) : now - 3000,
  });
}

bus.on('ws:drift', onDrift);
bus.on('ws:resume', onResume);
bus.on('ws:key_ideas', onKeyIdeas);
bus.on('ws:clench_progress', onClenchProgress);
bus.on('ws:clench', onClench);

// The server logs events against the session named in the last "session" message, so say it again
// after every reconnect.
bus.on('socket:open', () => {
  if (cur) announce(cur, 'start');
});

// If the connection drops while a lapse is open, its resume may never arrive. Close the lapse at the
// moment contact was lost: that is the longest it is known to have lasted. A resume that does arrive
// after reconnecting then finds no open lapse and is ignored.
bus.on('socket:close', () => {
  if (!cur || !cur.open) return;
  const now = Date.now();
  console.warn('connection lost during a lapse; closing it at the time of the disconnect');
  closeOpenLapse(cur, now, { type: 'resume', source: cur.open.source, wall_ms: now, reason: 'disconnect' }, 'disconnect');
});

export const tracker = {
  /** Start logging for a session and tell the server which session events belong to.
   *  getPosition(evt) returns the position object for the log; it is called at drift and at resume
   *  (the resume result is merged over the drift result). */
  begin(session, { getPosition } = {}) {
    if (!session || !session.id) throw new Error('tracker.begin needs a session');
    if (cur) this.cancel();
    state.session = session;
    cur = {
      session,
      getPosition,
      open: null,
      last: null,           // the most recently closed lapse (its turn back may extend it, once)
      lastAway: null,       // which way the head turned for that lapse: [x, y, z] degrees, or null
      openRot: null,        // the open lapse's drift rotation, or null
      swallowResume: false,
      quietUntil: 0,
      held: false,
      finishing: null,
      timer: null,
      log: {
        started_ms: Date.now(),
        ended_ms: null,
        gaze: { mode: 'none', calibrated: false },
        progress: [],
        lapses: [],
        video: { duration_s: null, watched_s: 0 },
      },
    };
    announce(cur, 'start');
    startAutosave(cur);
  },

  markProgress(section_idx) {
    if (!cur || !Number.isInteger(section_idx)) return;
    if (cur.log.progress.some((p) => p.section_idx === section_idx)) return;
    cur.log.progress.push({ section_idx, completed_ms: Date.now() });
  },

  setGaze({ mode, calibrated, error_px } = {}) {
    if (!cur) return;
    cur.log.gaze = { mode: mode || 'none', calibrated: Boolean(calibrated) };
    // The measured gaze error from the calibration check, when there is one (kept with the session).
    if (isNum(error_px)) cur.log.gaze.error_px = Math.round(error_px);
  },

  setVideo({ duration_s, watched_s } = {}) {
    if (!cur) return;
    if (duration_s !== undefined) cur.log.video.duration_s = isNum(duration_s) ? duration_s : null;
    if (isNum(watched_s)) cur.log.video.watched_s = watched_s;
  },

  attachScreenshot(lapse_id, url) {
    if (!cur) return;
    const lapse = cur.log.lapses.find((l) => l.id === lapse_id);
    if (lapse) lapse.screenshot_url = url || null;
  },

  /** A copy of the current client log (APP_SPEC §7), or null when no session is in progress. */
  log() {
    return cur ? JSON.parse(JSON.stringify(cur.log)) : null;
  },

  /** Close the session: send the final log, tell the server it ended, and return the session with
   *  its summary. If the server refuses, the session stays open so Finish can be pressed again. */
  finish() {
    if (!cur) return Promise.reject(new Error('No study session is in progress.'));
    const c = cur;
    if (c.finishing) return c.finishing;

    c.finishing = (async () => {
      const now = Date.now();
      // A lapse still open at the end is counted up to the end of the session, no further.
      if (c.open) closeOpenLapse(c, now, { type: 'resume', source: c.open.source, wall_ms: now, reason: 'finish' }, 'session_end');
      c.log.ended_ms = Date.now();
      clearInterval(c.timer);
      const id = c.session.id;
      try {
        await api.put(logPath(id), c.log).catch(() => {});   // keep a copy even if /finish fails
        const session = await api.post(`/api/sessions/${id}/finish`, c.log);
        announce(c, 'end');
        // The POST can take a while. If another session has been set up or begun meanwhile, it stays the
        // current one: the finished session is only returned.
        const newer = (cur && cur !== c) || (state.session && state.session.id !== id);
        if (cur === c) cur = null;
        if (!newer) state.session = session;
        return session;
      } catch (err) {
        c.log.ended_ms = null;
        c.finishing = null;
        if (cur === c) startAutosave(c);
        throw err;
      }
    })();
    return c.finishing;
  },

  /** Add a finished lapse that the viewer marked with a jaw clench (not a detector drift). The view decides its
   *  start (video: the end of the concept picked as the last thing remembered; reading: gaze 10 s before the
   *  clench) and its position. It does not touch the open look-away lapse or the turn-back state.
   *  Returns the lapse, or null when no session is being logged. Announces "lapse:clench" {lapse}. */
  addLapse({ source, onset_ms, resume_ms, position = null, chosen_concept = null } = {}) {
    if (!cur || cur.finishing) return null;
    const now = Date.now();
    const end = Math.round(Math.max(cur.log.started_ms, Math.min(isNum(resume_ms) ? resume_ms : now, now)));
    const start = Math.round(Math.max(cur.log.started_ms, Math.min(isNum(onset_ms) ? onset_ms : end, end)));
    const lapse = {
      id: newId(),
      kind: 'clench',
      source: source || 'simulated',
      onset_ms: start,
      resume_ms: end,
      duration_s: Math.round((end - start) / 10) / 100,
      end_method: 'clench',
      position,
      screenshot_url: null,
      chosen_concept: typeof chosen_concept === 'string' ? chosen_concept.slice(0, 120) : null,
      angle: null,
    };
    cur.log.lapses.push(lapse);
    save(cur);
    bus.emit('lapse:clench', { lapse });
    return lapse;
  },

  /** A clench the viewer dismissed as "not a zone-out" (counted in the log, for judging false triggers). */
  noteClenchDismissed() {
    if (!cur) return;
    cur.log.clench_dismissed = (cur.log.clench_dismissed || 0) + 1;
  },

  /** The open lapse, or null. */
  active() {
    return cur ? cur.open : null;
  },

  // ---- additions beyond APP_SPEC §10 (used by FE-A's own views; safe to ignore) ----

  /** End the open lapse now, by hand (the reader says they are back). The turn-back movement that
   *  follows is then not counted as a new lapse. Returns true if a lapse was open. */
  endLapse(method = 'manual') {
    if (!cur || !cur.open) return false;
    const now = Date.now();
    cur.quietUntil = now + REFRACTORY_MS;
    closeOpenLapse(cur, now, { type: 'resume', source: cur.open.source, wall_ms: now, reason: method }, method);
    return true;
  },

  /** True while a session is being logged. */
  running() {
    return Boolean(cur);
  },

  /** The session being logged, or null. */
  session() {
    return cur ? cur.session : null;
  },

  /** While held, new drifts are ignored (used during a mid-session gaze recalibration, when the
   *  reader is told to look around the screen). An open lapse still closes normally. */
  hold(on) {
    if (cur) cur.held = Boolean(on);
  },

  /** Abandon the session without finishing it: the log so far is saved, but no summary is made, so
   *  the session never counts towards long-term insights. */
  cancel() {
    if (!cur) return;
    const c = cur;
    if (c.open) {
      const now = Date.now();
      closeOpenLapse(c, now, { type: 'resume', source: c.open.source, wall_ms: now, reason: 'cancel' }, 'session_end');
    }
    clearInterval(c.timer);
    cur = null;
    save(c);
    announce(c, 'end');
  },
};
