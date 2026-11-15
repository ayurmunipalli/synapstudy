// Night (added 2026-11-15): after the flashcards, start the sleep recording with quiet cues for tonight's concepts,
// then in the morning show the sleep stages. The work is done by the existing scripts (server: app/night.py runs
// sleep_cue.py, then stage_nap.py). PROTOTYPE: live cueing has never run on a sleeper and the detector is not
// validated; the screen says so.
import { api } from '../api.js';
import { navigate } from '../router.js';
import { h } from '../lib/dom.js';
import { audio } from '../lib/audio.js';

const POLL_MS = 3000;
const DEFAULT_VOLUME = 0.15;

const two = (n) => String(n).padStart(2, '0');
function clock() {
  const d = new Date();
  return `${d.getHours()}:${two(d.getMinutes())}`;
}
function duration(ms) {
  const m = Math.max(0, Math.round(ms / 60000));
  return m >= 60 ? `${Math.floor(m / 60)} h ${two(m % 60)} min` : `${m} min`;
}

export const view = {
  mount(root, params = {}) {
    let dead = false;
    let session = null;
    let status = null;
    let volume = DEFAULT_VOLUME;
    let busy = false;
    let error = null;
    const page = h('div', { class: 'night-page' });
    root.replaceChildren(page);

    const act = async (fn) => {
      if (busy) return;
      busy = true;
      error = null;
      render();
      try {
        status = await fn();
      } catch (err) {
        error = err.message;
      } finally {
        busy = false;
        if (!dead) render();
      }
    };

    function cuesList() {
      const cards = (session && session.flashcards) || [];
      return cards.filter((c) => c.cue && c.cue.url);
    }

    function setupScreen() {
      const cues = cuesList();
      const readout = h('span', { class: 'night-vol-value' }, volume.toFixed(2));
      const slider = h('input', { class: 'fc-slider', type: 'range', min: '0.02', max: '0.5', step: '0.01', 'aria-label': 'Cue volume' });
      slider.value = String(volume);
      slider.addEventListener('input', () => {
        volume = Number(slider.value);
        readout.textContent = volume.toFixed(2);
        audio.volume(volume);
      });
      const preview = h('button', { class: 'btn', type: 'button', disabled: !cues.length,
        onClick: () => { audio.play(cues[Math.floor(Math.random() * cues.length)].cue.url, { volume }); } }, 'Play a cue at this volume');
      const start = h('button', { class: 'btn btn-primary btn-lg', type: 'button', disabled: busy,
        onClick: () => act(() => api.post('/api/night/start', { session_id: session ? session.id : null, volume })) },
      busy ? 'Starting…' : 'Start sleep');
      const failed = status && status.phase === 'failed';
      return h('div', { class: 'night-card' },
        h('p', { class: 'eyebrow' }, 'Tonight'),
        h('h1', { class: 'night-title' }, 'Sleep on what you studied'),
        h('p', { class: 'night-lede' },
          `While you sleep, the headband records your sleep, and when it detects stable sleep it quietly replays `
          + `${cues.length ? `the ${cues.length} concepts from your flashcards` : 'a soft tone (no flashcard cues were made)'}.`),
        h('div', { class: 'night-field' },
          h('div', { class: 'night-label' }, 'Cue volume: barely audible, set it while awake'),
          h('div', { class: 'night-vol' }, slider, readout, preview)),
        h('ul', { class: 'night-check' },
          h('li', null, 'Headband on and charged, ear sensors wet, stream_forever.sh running in Terminal.app'),
          h('li', null, 'Mac plugged in, lid open, sound on, Do Not Disturb on'),
          h('li', null, 'Double-tap the forehead pod during the night to pause cues for 15 minutes')),
        h('p', { class: 'night-note' }, status ? status.model_note : ''),
        failed ? h('p', { class: 'notice notice-warn' }, `The last night stopped: ${status.error || 'no recording was saved.'}`) : null,
        error ? h('p', { class: 'notice notice-warn' }, error) : null,
        h('div', { class: 'night-actions' }, start,
          h('button', { class: 'btn btn-ghost', type: 'button', onClick: () => navigate('home') }, 'Not tonight')));
    }

    function sleepScreen() {
      const s = status;
      const elapsed = s.started_ms ? Date.now() - s.started_ms : 0;
      const stopping = s.phase === 'stopping' || busy;
      return h('div', { class: 'night-dark' },
        h('div', { class: 'night-clock' }, clock()),
        h('div', { class: 'night-state' }, s.phase === 'starting' ? 'Connecting to the headband…' : 'Sleep mode'),
        h('div', { class: 'night-stats' },
          h('span', null, `${duration(elapsed)} recorded`),
          h('span', null, `${s.cues} cue${s.cues === 1 ? '' : 's'} played`),
          h('span', null, `${s.snoozes} snooze${s.snoozes === 1 ? '' : 's'}`)),
        h('div', { class: 'night-hint' }, 'Double-tap the forehead pod to pause cues for 15 minutes.'),
        error ? h('p', { class: 'night-err' }, error) : null,
        h('button', { class: 'btn night-wake', type: 'button', disabled: stopping,
          onClick: () => act(() => api.post('/api/night/stop', {})) }, stopping ? 'Saving the recording…' : "I'm awake"));
    }

    function morningScreen() {
      const s = status;
      const st = s.stage || {};
      const parts = [
        h('p', { class: 'eyebrow' }, 'Good morning'),
        h('h1', { class: 'night-title' }, 'Your night'),
        h('div', { class: 'night-stats night-stats-day' },
          h('span', null, `${duration((s.ended_ms || Date.now()) - (s.started_ms || Date.now()))} recorded`),
          h('span', null, `${s.cues} cue${s.cues === 1 ? '' : 's'} played`),
          h('span', null, `${s.snoozes} snooze${s.snoozes === 1 ? '' : 's'}`)),
      ];
      if (st.status === 'none') {
        parts.push(h('div', { class: 'night-actions' },
          h('button', { class: 'btn btn-primary btn-lg', type: 'button', disabled: busy,
            onClick: () => act(() => api.post('/api/night/stage', {})) }, 'Show my sleep stages')));
      } else if (st.status === 'running') {
        parts.push(h('p', { class: 'muted' }, h('span', { class: 'spinner' }), ' Working out the sleep stages from the recording (about a minute)…'));
      } else if (st.status === 'done') {
        parts.push(h('img', { class: 'night-plot', src: `/api/night/image/${encodeURIComponent(st.image)}`, alt: 'Hypnogram of the night with cue marks' }));
        parts.push(h('pre', { class: 'night-summary' }, (st.summary || []).join('\n')));
        parts.push(h('p', { class: 'night-note' }, 'Stages are YASA’s estimate from the forehead sensors, not a sleep lab. Cue marks show where cues landed; nothing here shows they helped memory.'));
      } else {
        parts.push(h('p', { class: 'notice notice-warn' }, `The sleep stages could not be worked out: ${st.error || 'unknown error'}`));
      }
      if (s.recording) parts.push(h('p', { class: 'muted night-small' }, `Recording: ${s.recording}`));
      if (error) parts.push(h('p', { class: 'notice notice-warn' }, error));
      parts.push(h('div', { class: 'night-actions' }, h('button', { class: 'btn btn-ghost', type: 'button', onClick: () => navigate('home') }, 'Home')));
      return h('div', { class: 'night-card' }, ...parts);
    }

    function render() {
      if (dead) return;
      const phase = status ? status.phase : 'idle';
      page.classList.toggle('is-dark', phase === 'starting' || phase === 'running' || phase === 'stopping');
      if (!status) page.replaceChildren(h('div', { class: 'night-card' }, h('span', { class: 'spinner' })));
      else if (phase === 'starting' || phase === 'running' || phase === 'stopping') page.replaceChildren(sleepScreen());
      else if (phase === 'done') page.replaceChildren(morningScreen());
      else page.replaceChildren(setupScreen());
    }

    async function poll() {
      try {
        status = await api.get('/api/night');
      } catch (err) {
        error = err.message;
      }
      render();
    }

    (async () => {
      if (params.id) {
        try {
          session = await api.get(`/api/sessions/${params.id}`);
        } catch {
          session = null;
        }
      }
      await poll();
    })();
    const timer = setInterval(poll, POLL_MS);

    return () => {
      dead = true;
      clearInterval(timer);
      audio.stop();
    };
  },
};
