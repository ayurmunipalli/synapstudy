// View contract: each view module exports a named const `view = { mount(root, params) }`.
// mount fills `root` (the <main id="view"> element) and returns an unmount function, or nothing.
// A view must undo everything it started in unmount: listeners, timers, overlays, gaze, audio.
// (registerView also accepts a module's default export or a bare { mount } object, and mount may
// return a promise of the unmount function, so a small mismatch between builders still runs.)
//
// The router keeps one view mounted at a time. On every navigation it calls the old view's unmount,
// then empties <main id="view">, #topbar-session and #overlay-root as a safety net.

import { bus } from './bus.js';
import { h } from './lib/dom.js';

const views = new Map();
// Views that can be rebuilt from an id alone, so a reload of the window can return to them.
const RESTORABLE = new Set(['home', 'summary', 'flashcards', 'history']);

let currentName = null;
let unmountCurrent = null;
let generation = 0;

export function registerView(name, module) {
  const view = module && (module.view ?? module.default ?? module);
  if (!view || typeof view.mount !== 'function') {
    throw new Error(`View "${name}" must export view = { mount(root, params) }`);
  }
  views.set(name, view);
}

export function currentView() {
  return currentName;
}

function runUnmount(fn) {
  if (typeof fn !== 'function') return;
  try {
    fn();
  } catch (err) {
    console.error('view unmount failed', err);
  }
}

function showError(root, message) {
  root.replaceChildren(
    h('div', { class: 'view-error' },
      h('h1', { class: 'h2' }, 'This screen could not be shown'),
      h('p', { class: 'notice notice-warn' }, message),
      h('button', { class: 'btn', type: 'button', onClick: () => navigate('home') }, 'Back to Home')),
  );
}

function isPrimitive(v) {
  return typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean';
}

// Remember the screen in the URL fragment (no history entries) only when its params are plain values.
function rememberRoute(name, params) {
  let hash = '';
  const entries = Object.entries(params);
  if (RESTORABLE.has(name) && entries.every(([, v]) => isPrimitive(v))) {
    const query = new URLSearchParams(entries.map(([k, v]) => [k, String(v)])).toString();
    hash = `#/${name}${query ? `?${query}` : ''}`;
  }
  try {
    history.replaceState(null, '', hash || location.pathname + location.search);
  } catch {
    /* not essential */
  }
}

export function navigate(name, params) {
  const root = document.getElementById('view');
  const args = params && typeof params === 'object' ? params : {};

  const previous = unmountCurrent;
  unmountCurrent = null;
  runUnmount(previous);

  const mine = ++generation;
  currentName = name;
  root.replaceChildren();
  document.getElementById('topbar-session')?.replaceChildren();
  document.getElementById('overlay-root')?.replaceChildren();
  root.dataset.view = name;
  root.scrollTop = 0;
  rememberRoute(name, args);
  bus.emit('route', { view: name, params: args });

  const view = views.get(name);
  if (!view) {
    showError(root, `There is no "${name}" screen in this build.`);
    return;
  }

  let result;
  try {
    result = view.mount(root, args);
  } catch (err) {
    console.error(`view "${name}" failed to mount`, err);
    if (mine === generation) showError(root, err && err.message ? err.message : String(err));
    return;
  }

  // If mount itself navigated elsewhere, this view is already stale: undo it straight away.
  const adopt = (fn) => {
    if (typeof fn !== 'function') return;
    if (mine === generation) unmountCurrent = fn;
    else runUnmount(fn);
  };

  if (result && typeof result.then === 'function') {
    result.then(adopt, (err) => {
      console.error(`view "${name}" failed to mount`, err);
      if (mine === generation) showError(root, err && err.message ? err.message : String(err));
    });
  } else {
    adopt(result);
  }
}

/** Show the first screen: the one named in the URL fragment if it can be restored, otherwise Home. */
export function startRouter() {
  const match = /^#\/([a-z]+)(?:\?(.*))?$/.exec(location.hash);
  if (match && RESTORABLE.has(match[1]) && views.has(match[1])) {
    navigate(match[1], Object.fromEntries(new URLSearchParams(match[2] || '')));
  } else {
    navigate('home');
  }
}
