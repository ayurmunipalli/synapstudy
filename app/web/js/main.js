// App entry: keeps the top bar in step with the server's status, handles the simulated D/R keys,
// registers the six views and shows the first screen.

import { api } from './api.js';
import { bus } from './bus.js';
import { ws } from './ws.js';
import { state, tracker } from './state.js';
import { navigate, registerView, startRouter } from './router.js';
import { h, isTyping } from './lib/dom.js';

const VIEWS = ['home', 'reading', 'video', 'summary', 'flashcards', 'history', 'night'];
const CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10'];
const SIGNAL_STALE_MS = 3000;
const NAV_CONFIRM_MS = 3500;
const SOURCES = {
  live: { text: 'LIVE', cls: 'badge-live', title: 'Lapses come from the headband, live.' },
  replay: { text: 'REPLAY', cls: 'badge-replay', title: 'Lapses come from a saved recording being replayed.' },
  simulated: { text: 'SIMULATED', cls: 'badge-sim', title: 'No headband: lapses are fired from the keyboard (D and R).' },
};

const badge = document.getElementById('source-badge');
const simKeys = document.getElementById('sim-keys');
const headband = document.getElementById('headband');
const headbandText = document.getElementById('headband-text');
const dots = [...document.querySelectorAll('#contact-dots i')];
const aiChip = document.getElementById('ai-chip');
const navLinks = [...document.querySelectorAll('.nav-link')];

let reachable = false;   // the server answered the last time we heard from it
let quality = null;      // latest signal.quality, [bool x4]
let signalAt = 0;

// ---------- top bar ----------

function contactSummary() {
  const live = reachable && quality && Date.now() - signalAt < SIGNAL_STALE_MS;
  dots.forEach((dot, i) => {
    dot.className = live ? (quality[i] ? 'ok' : 'bad') : '';
  });
  if (!live) return '';
  const parts = CHANNELS.map((name, i) => `${name} ${quality[i] ? 'ok' : 'poor'}`).join(', ');
  return `\nContact (rough estimate): ${parts}`;
}

function renderTopbar() {
  const s = reachable ? state.status : null;
  const source = s ? s.source : null;

  // The source badge: always shown, driven only by the server's status. Anything that is not
  // "live" is styled as not live, including a source name this build does not know.
  const known = SOURCES[source];
  if (known) {
    badge.className = `badge ${known.cls}`;
    badge.textContent = known.text;
    badge.title = known.title;
  } else if (s) {
    badge.className = 'badge badge-sim';
    badge.textContent = String(source || 'unknown').toUpperCase();
    badge.title = 'Unrecognised event source: treated as not live.';
  } else {
    badge.className = 'badge badge-unknown';
    badge.textContent = 'OFFLINE';
    badge.title = 'The app server is not answering, so no lapses can arrive.';
  }
  badge.hidden = false;
  simKeys.hidden = source !== 'simulated';

  if (!s) {
    headbandText.textContent = 'Server not reachable';
  } else if (!s.headband) {
    headbandText.textContent = 'No headband';
  } else {
    headbandText.textContent = s.detector_ready ? 'Headband on' : 'Headband warming up';
  }
  headband.title = (s && s.detail ? s.detail : '') + contactSummary();

  const ai = s && s.ai;
  if (!ai) {
    aiChip.hidden = false;
    aiChip.textContent = 'AI: unknown';
    aiChip.title = '';
  } else if (ai.mode === 'claude') {
    // Nothing to show when the AI is working normally (the "Claude" chip was removed, 2026-11-15).
    aiChip.hidden = true;
    return;
  } else {
    aiChip.hidden = false;
    const noKey = !ai.detail || /key/i.test(ai.detail);
    aiChip.textContent = noKey ? 'Extractive: no API key' : 'Extractive: AI unavailable';
    aiChip.title = ai.detail || '';
  }
}

function setStatus(body) {
  if (!body || typeof body !== 'object') return;
  reachable = true;
  state.status = body;
  renderTopbar();
  bus.emit('status', body);
}

bus.on('ws:status', (msg) => {
  const { type: _type, ...body } = msg;
  setStatus(body);
});
bus.on('ws:signal', (msg) => {
  quality = Array.isArray(msg.quality) ? msg.quality : null;
  signalAt = Date.now();
  renderTopbar();
});
bus.on('socket:open', () => {
  reachable = true;
  renderTopbar();
});
bus.on('socket:close', () => {
  reachable = false;
  renderTopbar();
});
setInterval(renderTopbar, 1000);   // lets the contact dots go grey when the signal stops

// ---------- navigation ----------

const navLabels = new Map(navLinks.map((btn) => [btn, btn.textContent]));
const navTimers = new Map();

function disarm(btn) {
  clearTimeout(navTimers.get(btn));
  navTimers.delete(btn);
  delete btn.dataset.confirm;
  btn.textContent = navLabels.get(btn);
}

for (const btn of navLinks) {
  btn.addEventListener('click', () => {
    btn.blur();
    // Leaving a running session ends it, so ask once (a second click within a few seconds confirms).
    if (tracker.running() && btn.dataset.confirm !== '1') {
      btn.dataset.confirm = '1';
      btn.textContent = 'End session?';
      navTimers.set(btn, setTimeout(() => disarm(btn), NAV_CONFIRM_MS));
      return;
    }
    disarm(btn);
    navigate(btn.dataset.nav);
  });
}

bus.on('route', ({ view }) => {
  for (const btn of navLinks) {
    disarm(btn);
    if (btn.dataset.nav === view) btn.setAttribute('aria-current', 'page');
    else btn.removeAttribute('aria-current');
  }
});

// ---------- simulated lapses: D = drift, R = resume, Z = 3-s jaw-clench hold ----------
// Handled here once for every view, only while the server says the source is "simulated" (the
// server checks again), and never while the user is typing.

window.addEventListener('keydown', (e) => {
  if (!state.status || state.status.source !== 'simulated') return;
  if (e.repeat || e.metaKey || e.ctrlKey || e.altKey) return;
  if (isTyping(e.target) || isTyping(document.activeElement)) return;
  const key = e.key.toLowerCase();
  if (key === 'd') {
    // The server keeps a simulated lapse open until it gets a resume. If this client has no open lapse
    // (it was closed by "I'm back", Finish, leaving the view or a disconnect), close the server's first,
    // so this D is not swallowed. With nothing open on the server the resume does nothing.
    if (!tracker.active()) ws.send({ type: 'sim', action: 'resume' });
    ws.send({ type: 'sim', action: 'drift' });
  } else if (key === 'r') {
    ws.send({ type: 'sim', action: 'resume' });
  } else if (key === 'z') {
    ws.send({ type: 'sim', action: 'clench' });     // a simulated 3-s jaw-clench hold
  }
});

// ---------- jaw-clench hold ring (all views) ----------
// Fills while a clench is being held (3 s marks a zone-out). Shown only once a hold is ~15% of the way,
// so the brief jaw activity of talking or chewing doesn't flash it (a display choice; the detector's
// frozen thresholds are unchanged). The signal is jaw muscle picked up by the EEG sensors, not brain activity.
const RING_SHOW_FROM = 0.15;
const ringEl = h('div', { class: 'clench-ring', hidden: true, role: 'status', 'aria-live': 'polite' });
ringEl.innerHTML = '<svg viewBox="0 0 44 44" aria-hidden="true"><circle class="clench-ring-bg" cx="22" cy="22" r="18"/>'
  + '<circle class="clench-ring-fg" cx="22" cy="22" r="18" pathLength="100" stroke-dasharray="0 100"/></svg>'
  + '<span class="clench-ring-text">Hold to mark a zone-out</span><span class="clench-ring-sim" hidden>SIMULATED</span>';
document.body.append(ringEl);
let ringHide = null;
let markedUntil = 0;     // after a completed hold, the progress drop back to 0 is not a "release"
function showRing(progress, text, source) {
  clearTimeout(ringHide);
  ringEl.hidden = false;
  ringEl.querySelector('.clench-ring-fg').setAttribute('stroke-dasharray', `${Math.round(progress * 100)} 100`);
  ringEl.querySelector('.clench-ring-text').textContent = text;
  ringEl.querySelector('.clench-ring-sim').hidden = source !== 'simulated';
}
bus.on('clench:progress', ({ progress, cancelled, source }) => {
  if (Date.now() < markedUntil && progress < RING_SHOW_FROM) return;
  if (progress >= RING_SHOW_FROM) {
    showRing(progress, 'Hold to mark a zone-out', source);
  } else if (!ringEl.hidden) {
    showRing(0, cancelled ? 'Released early: not marked' : 'Released', source);
    ringHide = setTimeout(() => { ringEl.hidden = true; }, cancelled ? 1200 : 400);
  }
});
bus.on('clench:trigger', ({ source }) => {
  markedUntil = Date.now() + 1500;
  showRing(1, 'Marked', source);
  ringHide = setTimeout(() => { ringEl.hidden = true; }, 1200);
});

// ---------- views ----------

function brokenView(name, err) {
  const reason = err && err.message ? err.message : String(err);
  return {
    mount(root) {
      root.append(h('div', { class: 'view-error' },
        h('h1', { class: 'h2' }, `The ${name} screen failed to load`),
        h('p', { class: 'notice notice-warn' }, reason),
        name === 'home' ? null : h('button', { class: 'btn', type: 'button', onClick: () => navigate('home') }, 'Back to Home')));
    },
  };
}

async function boot() {
  // Loaded one by one so that a fault in one screen cannot blank the whole app.
  const loaded = await Promise.allSettled(VIEWS.map((name) => import(`./views/${name}.js`)));
  loaded.forEach((result, i) => {
    const name = VIEWS[i];
    try {
      if (result.status !== 'fulfilled') throw result.reason;
      registerView(name, result.value);
    } catch (err) {
      console.error(`view "${name}" failed to load`, err);
      registerView(name, brokenView(name, err));
    }
  });
  startRouter();
}

api.get('/api/status').then(setStatus).catch(() => {
  if (!ws.connected()) renderTopbar();
});

boot();
