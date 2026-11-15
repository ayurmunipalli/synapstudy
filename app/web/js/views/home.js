// Home (APP_SPEC §11): one large "Start study session" button, then the session setup.

import { api } from '../api.js';
import { bus } from '../bus.js';
import { navigate } from '../router.js';
import { state } from '../state.js';
import { ws } from '../ws.js';
import { h, fmtNum } from '../lib/dom.js';

const MAX_FILE_BYTES = 2_000_000;

function countWords(text) {
  const t = text.trim();
  return t ? t.split(/\s+/).length : 0;
}

function firstWords(text, n = 8) {
  const words = text.trim().split(/\s+/);
  return words.slice(0, n).join(' ') + (words.length > n ? '…' : '');
}

// "10° (most sensitive; the tested setting)", "15°", "20°", "30° (least sensitive)"
function thetaLabel(theta, choices, tested) {
  const notes = [];
  if (theta === Math.min(...choices)) notes.push('most sensitive');
  if (theta === Math.max(...choices)) notes.push('least sensitive');
  if (theta === tested) notes.push('the tested setting');
  return `${theta}°${notes.length ? ` (${notes.join('; ')})` : ''}`;
}

function mount(root) {
  let dead = false;
  let mode = 'reading';
  let autoTitle = '';   // a title this form filled in itself (so it may replace it again)
  let busy = false;

  // ---- hero ------------------------------------------------------------------------------------
  const startBtn = h('button', { class: 'btn btn-primary btn-lg home-start', type: 'button', onClick: openSetup }, 'Start study session');
  const statusBox = h('div', { class: 'home-status' });

  // ---- setup: one study session = a subject and its resources, studied in order ---------------------
  // (Ayur, 2026-11-15: "ask me what I want to study", then PDFs and/or YouTube links, all one session.)
  // Each resource still runs in the unchanged reading or video screen; the server links them (POST
  // /api/studies), and the flashcards at the end are made from all of them.
  const subjects = h('datalist', { id: 'setup-subjects' });
  const subjectEl = h('input', { class: 'input study-subject', id: 'setup-subject', list: 'setup-subjects',
    placeholder: 'e.g. Neural networks for my ML midterm', autocomplete: 'off', maxlength: 80 });

  let resources = [];   // [{kind: 'pdf'|'text'|'video', title, file?, text?, url?}] in study order
  const listEl = h('ol', { class: 'study-list' });
  const fileEl = h('input', { type: 'file', accept: '.pdf,application/pdf,.txt,.md,text/plain,text/markdown', multiple: true, hidden: true, onChange: onFiles });
  const fileBtn = h('button', { class: 'btn', type: 'button', onClick: () => fileEl.click() }, 'Add PDFs');
  const urlEl = h('input', { class: 'input', id: 'setup-url', type: 'text', inputmode: 'url',
    placeholder: 'Paste a YouTube link', autocomplete: 'off', spellcheck: 'false',
    onKeydown: (e) => { if (e.key === 'Enter') { e.preventDefault(); addUrl(); } } });
  const urlBtn = h('button', { class: 'btn', type: 'button', onClick: addUrl }, 'Add YouTube video');
  const sampleEl = h('select', { class: 'input', 'aria-label': 'Add a sample passage', onChange: onSample },
    h('option', { value: '' }, 'Add a sample passage'));

  const gazeChoice = (value, label, hint, checked) => h('label', { class: 'choice' },
    h('input', { type: 'radio', name: 'setup-gaze', value, checked }),
    h('span', null, label, h('small', null, hint)));
  const shotEl = h('input', { type: 'checkbox' });
  const options = h('details', { class: 'study-options' },
    h('summary', null, 'Options'),
    h('div', { class: 'field' },
      h('span', { class: 'label' }, 'Gaze while reading PDFs'),
      gazeChoice('webgazer', 'Calibrate webcam gaze', 'The webcam finds where on the page you were reading. Calibration takes about a minute.', true),
      gazeChoice('none', 'Skip gaze', 'A lapse then marks the top of the page instead.', false),
      gazeChoice('mouse', 'Use the mouse as gaze (testing)', 'Treats the pointer as your gaze, for trying the app without a webcam.', false)),
    h('label', { class: 'choice' }, shotEl, h('span', null, 'Videos: capture a screenshot at each lapse (asks to share this window)')));

  const errorEl = h('span', { class: 'form-error', role: 'alert', hidden: true });
  const beginBtn = h('button', { class: 'btn btn-primary', type: 'submit' }, 'Start studying');

  const setup = h('form', { class: 'card setup', hidden: true, novalidate: true, onSubmit: (e) => { e.preventDefault(); begin(); } },
    h('div', { class: 'field' }, h('label', { for: 'setup-subject', class: 'study-q' }, 'What do you want to study?'), subjectEl, subjects),
    h('div', { class: 'field' },
      h('span', { class: 'label' }, 'Your resources, in the order you want to study them'),
      listEl,
      h('div', { class: 'setup-sources' }, fileBtn, fileEl, sampleEl),
      h('div', { class: 'setup-sources' }, urlEl, urlBtn)),
    options,
    h('div', { class: 'setup-actions' }, beginBtn, errorEl));

  const home = h('div', { class: 'home' },
    h('p', { class: 'eyebrow' }, 'Study session'),
    h('h1', { class: 'home-title' }, 'Pick up exactly where you left off.'),
    h('p', { class: 'home-lede' },
      'Tell it what you are studying and add your PDFs and YouTube videos. Work through them with the headband on: ',
      'it notes the moments you drift, shows you what you missed, and ends with flashcards from everything.'),
    startBtn,
    setup,
    statusBox);
  root.append(home);

  // ---- behaviour -------------------------------------------------------------------------------
  function openSetup() {
    home.classList.add('is-setup');
    setup.hidden = false;
    subjectEl.focus();
  }

  function showError(message) {
    errorEl.textContent = message || '';
    errorEl.hidden = !message;
  }

  function updateCount() {}   // kept for the sample loader below

  const KIND_LABEL = { pdf: 'PDF', text: 'Text', video: 'YouTube' };
  function renderList() {
    listEl.replaceChildren(...resources.map((r, i) => h('li', { class: 'study-item' },
      h('span', { class: `chip study-kind study-kind-${r.kind}` }, KIND_LABEL[r.kind]),
      h('span', { class: 'study-title' }, r.title),
      h('span', { class: 'study-btns' },
        h('button', { class: 'btn btn-ghost', type: 'button', disabled: i === 0, 'aria-label': `Move ${r.title} up`,
          onClick: () => { [resources[i - 1], resources[i]] = [resources[i], resources[i - 1]]; renderList(); } }, '↑'),
        h('button', { class: 'btn btn-ghost', type: 'button', 'aria-label': `Remove ${r.title}`,
          onClick: () => { resources.splice(i, 1); renderList(); } }, '×')))));
    if (!resources.length) listEl.append(h('li', { class: 'study-empty muted' }, 'Nothing added yet.'));
  }

  function onFiles() {
    const files = Array.from(fileEl.files || []);
    fileEl.value = '';
    showError(null);
    for (const file of files) {
      const isPdf = /\.pdf$/i.test(file.name) || file.type === 'application/pdf';
      resources.push({ kind: isPdf ? 'pdf' : 'text', title: file.name.replace(/\.[^.]+$/, ''), file });
    }
    renderList();
  }

  function addUrl() {
    const url = urlEl.value.trim();
    if (!url) return;
    if (!/(youtube\.com|youtu\.be)\//i.test(url)) {
      showError('That is not a YouTube link (youtube.com or youtu.be).');
      return;
    }
    showError(null);
    resources.push({ kind: 'video', title: url.replace(/^https?:\/\/(www\.)?/, ''), url });
    urlEl.value = '';
    renderList();
  }

  async function onSample() {
    const id = sampleEl.value;
    if (!id) return;
    showError(null);
    sampleEl.disabled = true;
    try {
      const sample = await api.get(`/api/samples/${encodeURIComponent(id)}`);
      if (dead) return;
      resources.push({ kind: 'text', title: sample.title || 'Sample passage', text: sample.text || '' });
      renderList();
    } catch (err) {
      if (!dead) showError(err.message);
    } finally {
      sampleEl.disabled = false;
      sampleEl.value = '';   // it works as a menu, not as a setting
    }
  }

  const readDataUrl = (file) => new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result);
    fr.onerror = () => reject(new Error(`${file.name} could not be read.`));
    fr.readAsDataURL(file);
  });

  async function begin() {
    if (busy) return;
    showError(null);
    const subject = subjectEl.value.trim();
    if (!subject) {
      showError('Say what you want to study first.');
      subjectEl.focus();
      return;
    }
    if (!resources.length) {
      showError('Add at least one PDF or YouTube video.');
      return;
    }
    busy = true;
    beginBtn.disabled = true;
    const progress = (msg) => beginBtn.replaceChildren(h('span', { class: 'spinner' }), msg);
    try {
      const body = [];
      const notes = [];
      for (const [i, r] of resources.entries()) {
        if (r.kind === 'video') {
          body.push({ kind: 'video', youtube_url: r.url });
        } else if (r.kind === 'pdf') {
          progress(`Reading ${r.title} (${i + 1}/${resources.length})`);
          const pdf = await api.post('/api/pdf_text', { name: r.file.name, data_url: await readDataUrl(r.file) });
          if (pdf.note) notes.push(pdf.note);
          if (!pdf.text || pdf.text.length < 50) throw new Error(pdf.note || `${r.file.name} has no readable text.`);
          body.push({ kind: 'reading', title: pdf.title || r.title, text: pdf.text });
        } else {
          const text = r.text ?? await r.file.text();
          if (!text.trim()) throw new Error(`${r.title} is empty.`);
          body.push({ kind: 'reading', title: r.title, text });
        }
        if (dead) return;
      }
      progress('Preparing your study session');
      const study = await api.post('/api/studies', { subject, resources: body });
      if (dead) return;
      const picked = setup.querySelector('input[name="setup-gaze"]:checked');
      const params = { gaze: picked ? picked.value : 'none', capture: shotEl.checked, screenshots: shotEl.checked };
      // The options for every resource in this study (the summary screen uses them for "Next").
      state.studyParams = params;
      const first = study.sessions[0];
      state.session = first;
      if (notes.length) console.warn(notes.join(' '));
      navigate(first.mode, { id: first.id, session: first, ...params });
    } catch (err) {
      if (dead) return;
      showError(err.message);
      busy = false;
      beginBtn.disabled = false;
      beginBtn.textContent = 'Start studying';
    }
  }

  let replays = [];   // recorded sessions the server can replay (file names)
  let thetaBusy = false;    // a head-turn setting is being sent
  let thetaError = null;

  // Head-turn sensitivity: the server applies it to the detector, saves it and tells every window.
  async function setTheta(theta) {
    if (thetaBusy || !Number.isFinite(theta)) return;
    thetaBusy = true;
    thetaError = null;
    try {
      const body = await api.post('/api/detector', { theta });
      // The new status also arrives over the socket; take the detector part now in case that is late.
      if (body && body.detector && state.status) state.status = { ...state.status, detector: body.detector };
    } catch (err) {
      thetaError = err.message;
    } finally {
      thetaBusy = false;
      if (!dead) renderStatus();
    }
  }

  // The current event source and AI mode, in plain words. Everything here comes from /api/status.
  function renderStatus() {
    const s = state.status;
    const rows = [];
    if (!s) {
      rows.push(h('div', { class: 'notice' }, 'Connecting to the app server.'));
    } else {
      const detail = s.detail ? ` (${s.detail})` : '';
      // The head-turn setting, when the server has one (an older server does not: then the control is
      // left out and the evidence sentence stays in the "Headband connected" line, as before).
      const det = s.detector || {};
      const choices = Array.isArray(det.choices) ? det.choices.filter((c) => Number.isFinite(c)) : [];
      const hasSetting = choices.length > 0 && choices.includes(det.theta);
      if (s.source === 'live') {
        const kinds = (det.acted_kinds || ['look_away']).map((k) => String(k).replace(/_/g, '-')).join(', ');
        const evidence = det.evidence && !hasSetting ? ` ${det.evidence}` : '';
        rows.push(h('div', { class: 'notice' },
          h('span', null, h('b', null, 'Headband connected. '), `Lapses acted on: ${kinds}.${evidence}`)));
      } else if (s.source === 'replay') {
        rows.push(h('div', { class: 'notice notice-warn' },
          h('span', null, h('b', null, 'Replaying a saved recording. '),
            'Lapses come from that file, not from you. Sessions run now are marked and left out of long-term insights.')));
      } else {
        rows.push(h('div', { class: 'notice notice-warn' },
          h('span', null, h('b', null, 'No headband connected: lapses can only be simulated with D/R. '),
            `Simulated sessions are marked and left out of long-term insights.${detail}`)));
      }
      if (hasSetting) {
        const pick = h('select', { class: 'input replay-pick', id: 'head-turn', disabled: thetaBusy,
          onChange: () => setTheta(Number(pick.value)) },
        ...choices.map((c) => h('option', { value: String(c) }, thetaLabel(c, choices, det.tested_theta))));
        pick.value = String(det.theta);
        rows.push(h('div', { class: 'notice replay-row' },
          h('label', { for: 'head-turn' }, h('b', null, 'Head-turn sensitivity: '),
            'how far your head must turn within a second to count as looking away.'),
          pick,
          det.evidence ? h('span', { style: 'flex-basis:100%' }, det.evidence) : null,
          thetaError ? h('span', { class: 'form-error', role: 'alert', style: 'flex-basis:100%' }, thetaError) : null));
      }
      // Rehearsal without the headband: replay the lapses of a recorded session at real speed.
      if (s.source === 'replay') {
        rows.push(h('div', { class: 'notice' },
          h('button', { class: 'btn', type: 'button', onClick: () => ws.send({ type: 'replay', action: 'stop' }) }, 'Stop replay')));
      } else if (s.source === 'simulated' && replays.length) {
        const pick = h('select', { class: 'input replay-pick', 'aria-label': 'Recorded session to replay' },
          ...replays.map((r) => h('option', { value: r.file }, r.file)));
        pick.value = replays[replays.length - 1].file;
        rows.push(h('div', { class: 'notice replay-row' },
          h('span', null, h('b', null, 'Rehearse: '), 'replay the head-turn lapses of a recorded session at real speed (shown as REPLAY).'),
          pick,
          h('button', { class: 'btn', type: 'button',
            onClick: () => ws.send({ type: 'replay', action: 'start', file: pick.value }) }, 'Start replay')));
      }

      const ai = s.ai || {};
      if (ai.mode !== 'claude') {     // the "AI: Claude" notice was removed (Ayur, 2026-11-15); the fallback warning stays
        rows.push(h('div', { class: 'notice notice-warn' },
          h('span', null, h('b', null, 'AI: extractive. '),
            `Key ideas and flashcards are sentences picked from your text; video summaries and questions are off.${ai.detail ? ` (${ai.detail})` : ''}`)));
      }
    }
    statusBox.replaceChildren(...rows);
  }

  // ---- data ------------------------------------------------------------------------------------
  api.get('/api/replays').then((list) => {
    if (dead || !Array.isArray(list)) return;
    replays = list.filter((r) => r && typeof r.file === 'string');
    renderStatus();
  }).catch(() => {});

  api.get('/api/samples').then((list) => {
    if (dead) return;
    if (!Array.isArray(list) || !list.length) {
      sampleEl.hidden = true;
      return;
    }
    for (const s of list) {
      const words = Number.isFinite(s.words) ? ` · ${fmtNum(s.words)} words` : '';
      sampleEl.append(h('option', { value: s.id }, `${s.title}${words}`));
    }
  }).catch(() => {
    sampleEl.hidden = true;
  });

  api.get('/api/sessions').then((list) => {
    if (dead || !Array.isArray(list)) return;
    const names = [...new Set(list.map((s) => (s.subject || '').trim()).filter(Boolean))].sort((a, b) => a.localeCompare(b));
    subjects.replaceChildren(...names.map((name) => h('option', { value: name })));
  }).catch(() => {});

  const offStatus = bus.on('status', renderStatus);
  renderStatus();
  renderList();

  return () => {
    dead = true;
    offStatus();
  };
}

export const view = { mount };
