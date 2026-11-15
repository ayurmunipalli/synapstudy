// Sound: the soft lapse chime (synthesised, no audio file) and playback of cue files.
// The chime has its own fixed, gentle level. volume() is the level for cue playback only, so turning
// cues down to "barely audible" for the night never silences the daytime chime.

let ctx = null;
let sinkId = '';       // '' = the system default output
let master = 0.6;      // cue playback volume, 0..1
let current = null;    // { el, done } for the cue that is playing

function clamp01(v) {
  return Math.min(1, Math.max(0, Number(v)));
}

function context() {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtx) return null;
  if (!ctx) {
    ctx = new AudioCtx();
    if (sinkId && typeof ctx.setSinkId === 'function') ctx.setSinkId(sinkId).catch(() => {});
  }
  if (ctx.state === 'suspended') ctx.resume().catch(() => {});
  return ctx;
}

function note(c, freq, at, length, peak) {
  const osc = c.createOscillator();
  const gain = c.createGain();
  osc.type = 'sine';
  osc.frequency.value = freq;
  gain.gain.setValueAtTime(0.0001, at);
  gain.gain.exponentialRampToValueAtTime(peak, at + 0.02);
  gain.gain.exponentialRampToValueAtTime(0.0001, at + length);
  osc.connect(gain).connect(c.destination);
  osc.start(at);
  osc.stop(at + length + 0.05);
}

export const audio = {
  /** A soft two-note chime. Silent (never throws) if the browser has no Web Audio or blocks it. */
  chime() {
    try {
      const c = context();
      if (!c) return;
      const t = c.currentTime + 0.02;
      note(c, 659.25, t, 0.7, 0.11);         // E5
      note(c, 880.0, t + 0.16, 0.9, 0.09);   // A5
    } catch (err) {
      console.warn('chime failed:', err);
    }
  },

  /** Play an audio file. Stops whatever cue was playing. The promise resolves when playback ends or
   *  is stopped, and rejects if the file cannot be played. */
  play(url, { volume } = {}) {
    audio.stop();
    const el = new Audio();
    el.preload = 'auto';
    el.volume = clamp01(volume === undefined || volume === null ? master : volume);
    const item = { el, done: null };
    current = item;

    return new Promise((resolve, reject) => {
      let settled = false;
      item.done = (err) => {
        if (settled) return;
        settled = true;
        el.onended = null;
        el.onerror = null;
        if (current === item) current = null;
        if (err) reject(err);
        else resolve();
      };
      el.onended = () => item.done();
      el.onerror = () => item.done(new Error('That sound could not be played.'));
      el.src = url;

      const start = () => {
        if (settled) return;
        const playing = el.play();
        if (playing && typeof playing.catch === 'function') {
          playing.catch((err) => {
            // pause() during loading rejects play() with AbortError; that is a normal stop.
            if (err && err.name === 'AbortError') item.done();
            else item.done(err instanceof Error ? err : new Error(String(err)));
          });
        }
      };
      if (sinkId && typeof el.setSinkId === 'function') el.setSinkId(sinkId).catch(() => {}).then(start);
      else start();
    });
  },

  stop() {
    const item = current;
    if (!item) return;
    current = null;
    try {
      item.el.pause();
    } catch {
      /* nothing was playing */
    }
    if (item.done) item.done();
  },

  /** Selectable sound outputs as [{id, label}], or [] when the browser cannot switch outputs or does
   *  not reveal device names (Chrome only names them once a camera or microphone has been allowed). */
  async outputs() {
    const canSwitch = typeof HTMLMediaElement !== 'undefined' && typeof HTMLMediaElement.prototype.setSinkId === 'function';
    if (!canSwitch || !navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return [];
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      return devices
        .filter((d) => d.kind === 'audiooutput' && d.label)
        .map((d) => ({ id: d.deviceId, label: d.label }));
    } catch {
      return [];
    }
  },

  /** Route cue playback (and the chime, where the browser allows) to an output from outputs().
   *  Resolves true if the output was applied to what is playing now, false otherwise. */
  async setOutput(id) {
    sinkId = id || '';
    let ok = true;
    if (current && typeof current.el.setSinkId === 'function') {
      try {
        await current.el.setSinkId(sinkId);
      } catch {
        ok = false;
      }
    }
    if (ctx && typeof ctx.setSinkId === 'function') ctx.setSinkId(sinkId).catch(() => {});
    return ok;
  },

  /** Get or set the cue playback volume (0..1). Applies at once to a cue that is playing. */
  volume(v) {
    if (v !== undefined && v !== null && Number.isFinite(Number(v))) {
      master = clamp01(v);
      if (current) current.el.volume = master;
    }
    return master;
  },
};
