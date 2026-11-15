// Wrapper over the YouTube IFrame Player API, the one external script this app loads.

const API_SRC = 'https://www.youtube.com/iframe_api';
const API_TIMEOUT_MS = 15000;
const POLL_MS = 250;
// A step between two polls larger than this is a seek, not playback (covers 2x speed and a
// background tab whose timers are throttled to 1 Hz).
const MAX_STEP_S = 2.5;

const STATE_NAMES = { '-1': 'unstarted', 0: 'ended', 1: 'playing', 2: 'paused', 3: 'buffering', 5: 'cued' };

const ERROR_TEXT = {
  2: 'The video ID is not valid.',
  5: 'This video cannot be played in an embedded player.',
  100: 'The video was not found. It may be private or removed.',
  101: 'The owner of this video does not allow it to be played inside other apps.',
  150: 'The owner of this video does not allow it to be played inside other apps.',
  153: 'YouTube could not identify the page embedding the player (missing referrer).',
};

let apiPromise = null;

// Resolves with the global YT object. The script tag is added once; a failed load can be retried.
export function loadApi() {
  if (window.YT && window.YT.Player) return Promise.resolve(window.YT);
  if (apiPromise) return apiPromise;
  apiPromise = new Promise((resolve, reject) => {
    const script = document.createElement('script');
    const fail = () => {
      clearTimeout(timeout);
      script.remove();
      apiPromise = null;
      reject(new Error('The YouTube player could not be loaded. Check the internet connection.'));
    };
    const timeout = setTimeout(fail, API_TIMEOUT_MS);
    const previous = window.onYouTubeIframeAPIReady;
    window.onYouTubeIframeAPIReady = () => {
      clearTimeout(timeout);
      if (typeof previous === 'function') previous();
      resolve(window.YT);
    };
    script.src = API_SRC;
    script.async = true;
    script.onerror = fail;
    document.head.append(script);
  });
  return apiPromise;
}

export function describeError(code) {
  const text = ERROR_TEXT[code] || 'The video cannot be played.';
  return code == null ? text : `YouTube error ${code}: ${text}`;
}

// loadPlayer(el, videoId, { start, autoplay }) -> player object, returned at once.
// Calls made before the player is ready are queued. Extra to the spec'd methods: state(), watched(),
// onTick(fn), onError(fn) and `ready` (a Promise<boolean> that never rejects).
export function loadPlayer(el, videoId, opts = {}) {
  const listeners = { state: new Set(), tick: new Set(), error: new Set() };
  const queue = [];
  let player = null;
  let ready = false;
  let destroyed = false;
  let timer = null;
  let stateName = 'unstarted';
  let lastTime = Math.max(0, Number(opts.start) || 0);
  let lastDuration = 0;
  let prevTick = null;
  let watched = 0;
  let lastError = null;

  // YT.Player replaces its target element with the iframe, so give it a child and keep `el`.
  const slot = document.createElement('div');
  el.replaceChildren(slot);

  const emit = (type, payload) => {
    for (const fn of [...listeners[type]]) {
      try {
        fn(payload);
      } catch (err) {
        console.error(err);
      }
    }
  };
  const subscribe = (type, fn) => {
    listeners[type].add(fn);
    return () => listeners[type].delete(fn);
  };

  function fail(code, message) {
    lastError = { code, message: message || describeError(code) };
    emit('error', lastError);
  }

  function tick() {
    if (!player || destroyed) return;
    let t = lastTime;
    try {
      t = Number(player.getCurrentTime()) || 0;
      lastDuration = Number(player.getDuration()) || lastDuration;
    } catch {
      return;
    }
    if (prevTick != null) {
      const step = t - prevTick;
      if (step > 0 && step <= MAX_STEP_S) watched += step;
    }
    prevTick = t;
    lastTime = t;
    emit('tick', t);
  }

  function setPolling(on) {
    if (on && timer == null) {
      prevTick = null;
      tick();
      timer = setInterval(tick, POLL_MS);
    } else if (!on && timer != null) {
      clearInterval(timer);
      timer = null;
      tick();
      prevTick = null;
    }
  }

  function onStateChange(event) {
    stateName = STATE_NAMES[event.data] || 'unstarted';
    setPolling(stateName === 'playing');
    if (stateName === 'ended' && lastDuration > 0) {
      lastTime = lastDuration;
      emit('tick', lastTime);
    }
    emit('state', stateName);
  }

  const readyPromise = loadApi().then(
    (YT) =>
      new Promise((resolve) => {
        if (destroyed) return resolve(false);
        const playerVars = {
          playsinline: 1,
          origin: window.location.origin,
          enablejsapi: 1,
          rel: 0,
        };
        if (lastTime > 0) playerVars.start = Math.floor(lastTime);
        if (opts.autoplay) playerVars.autoplay = 1;
        try {
          player = new YT.Player(slot, {
            videoId,
            width: '100%',
            height: '100%',
            playerVars,
            events: {
              onReady: () => {
                if (destroyed) return resolve(false);
                ready = true;
                try {
                  lastDuration = Number(player.getDuration()) || 0;
                } catch {
                  lastDuration = 0;
                }
                for (const [name, args] of queue.splice(0)) player[name](...args);
                resolve(true);
              },
              onStateChange,
              onError: (event) => {
                fail(event.data);
                resolve(false);
              },
            },
          });
        } catch (err) {
          fail(null, `The YouTube player could not start: ${err.message}`);
          resolve(false);
        }
      }),
    (err) => {
      if (!destroyed) fail(null, err.message);
      return false;
    },
  );

  function call(name, ...args) {
    if (destroyed) return;
    if (!ready) {
      queue.push([name, args]);
      return;
    }
    try {
      player[name](...args);
    } catch (err) {
      console.error(err);
    }
  }

  return {
    ready: readyPromise,
    play: () => call('playVideo'),
    pause: () => call('pauseVideo'),
    seek(seconds) {
      const t = Math.max(0, Number(seconds) || 0);
      lastTime = t;
      prevTick = null;
      call('seekTo', t, true);
    },
    time() {
      if (ready && !destroyed) {
        try {
          lastTime = Number(player.getCurrentTime()) || 0;
        } catch {
          /* keep the last known time */
        }
      }
      return lastTime;
    },
    duration() {
      if (ready && !destroyed) {
        try {
          lastDuration = Number(player.getDuration()) || lastDuration;
        } catch {
          /* keep the last known duration */
        }
      }
      return lastDuration;
    },
    state: () => stateName,
    watched: () => watched,
    onState: (fn) => subscribe('state', fn),
    onTick: (fn) => subscribe('tick', fn),
    onError(fn) {
      if (lastError) fn(lastError);
      return subscribe('error', fn);
    },
    destroy() {
      if (destroyed) return;
      // Keep the last position readable after teardown (a late getPosition call must not throw).
      if (ready) {
        try {
          lastTime = Number(player.getCurrentTime()) || lastTime;
        } catch {
          /* ignore */
        }
      }
      destroyed = true;
      if (timer != null) clearInterval(timer);
      timer = null;
      queue.length = 0;
      for (const set of Object.values(listeners)) set.clear();
      try {
        if (player && typeof player.destroy === 'function') player.destroy();
      } catch {
        /* the iframe may already be gone */
      }
      player = null;
      el.replaceChildren();
    },
  };
}
