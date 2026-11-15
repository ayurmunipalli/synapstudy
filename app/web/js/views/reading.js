// Reading tool (APP_SPEC §2, §11): the passage in two columns on the left, Insights on the right,
// the webcam indicator top centre. When the event source reports a look-away, Insights shows the STOP
// message, the place where focus was lost is circled in blue, and on resume a flowing dashed arrow leads
// from the reader's gaze back to it. The place comes from webcam gaze, which is coarse: the circle is
// drawn as wide as the gaze error measured after calibration, and the STOP card says so.

import { api } from '../api.js';
import { bus } from '../bus.js';
import { navigate } from '../router.js';
import { state, tracker, mergeKeyIdeas } from '../state.js';
import { h, fmtClock, isTyping } from '../lib/dom.js';
import { audio } from '../lib/audio.js';
import { gaze, snapToLine, describeGazeError, roundGazeError, safetyLines, wordLinesAbove } from '../lib/gaze.js';
import { createLapseOverlay, radiiForError, nearMark } from '../lib/lapse_overlay.js';

const GUIDE_MAX_MS = 8000;     // the arrow, circle and STOP card clear after this long at most
// A held jaw clench marks the place the gaze was this long before the clench finished (gaze.js keeps 15 s).
const CLENCH_LOOKBACK_MS = 10000;
const NEAR_PX = 140;           // ...or once the gaze stays this close to the circle's centre
const NEAR_MARGIN_PX = 40;     // ...or, for a wide circle, inside it or within this much of its edge
const NEAR_HOLD_MS = 600;      // ...for this long
const WIDE_MARK_ABOVE_PX = 60; // a measured gaze error above this gets a circle as wide as the error
const DEFAULT_GAZE_ERROR_PX = 120;   // the error assumed for the circle when it could not be measured
const PASSAGE_MARGIN_PX = 36;  // a gaze point this far outside the text still counts as "on the passage"
const IDEA_POLL_MS = 5000;
const STATUS_RANK = { pending: 0, partial: 1, done: 2, failed: 2 };
const MODIFIER_KEYS = new Set(['Shift', 'Control', 'Alt', 'Meta', 'CapsLock']);
const ABBREVIATIONS = new Set(['mr', 'mrs', 'ms', 'dr', 'prof', 'st', 'vs', 'etc', 'e.g', 'i.e', 'cf', 'fig', 'no', 'vol', 'pp']);

// Shown once, under the feed, as soon as a headband (or replayed) lapse is listed.
const ESTIMATE_NOTE = 'Lapse lengths and focus % are estimates: the headband senses the head turning, not the moment you look back.';

// "In or near", not "inside": the circle's half-width is the MEDIAN gaze error, so the true place falls
// outside it often (about two times in three if the error is the same in every direction).
const NOTE_APPROXIMATE = 'Gaze is approximate: you were somewhere in or near the circle.';
const NOTE_NOT_CALIBRATED = 'Gaze is not calibrated: a lapse will mark the top of the page.';
const NOTE_SKIPPED = 'Gaze was skipped: a lapse will mark the top of the page.';

// ---------- text model ----------

function endsSentence(word, next) {
  const m = /^(.*?)([.!?]+)["'”’)\]]*$/.exec(word);
  if (!m) return false;
  if (m[2] === '.') {
    const stem = m[1].replace(/^["'“‘(\[]+/, '').toLowerCase();
    if (ABBREVIATIONS.has(stem) || /^[a-z]$/.test(stem)) return false;   // "Dr." or an initial
  }
  return /^["'“‘(\[]*[A-Z0-9]/.test(next);
}

// Words get one global index across the whole passage (section 0's words first), which is what
// data-w and the log's word_idx carry. Sentences are found here so one can be highlighted and quoted.
function buildModel(sections) {
  const secs = [];
  const sentences = [];   // { from, to (global word range), text }
  const sentOf = [];      // sentOf[global word index] = index into sentences
  let total = 0;

  sections.forEach((section, pos) => {
    const words = [];
    String(section.text || '').split(/\n\s*\n+/).forEach((para, p) => {
      for (const t of para.split(/\s+/)) if (t) words.push({ t, para: p });
    });
    const sec = { idx: Number.isInteger(section.idx) ? section.idx : pos, start: total, words, ends: [] };
    let from = 0;
    words.forEach((w, k) => {
      const next = words[k + 1];
      sentOf.push(sentences.length);
      if (!next || next.para !== w.para || endsSentence(w.t, next.t)) {
        sentences.push({ from: total + from, to: total + k + 1, text: words.slice(from, k + 1).map((x) => x.t).join(' ') });
        sec.ends.push(k + 1);
        from = k + 1;
      }
    });
    total += words.length;
    secs.push(sec);
  });
  return { secs, sentences, sentOf, total };
}

function pickSession(params) {
  if (params.session && params.session.id) return params.session;
  const s = state.session;
  return s && (!params.id || s.id === params.id) ? s : null;
}

// ---------- the view ----------

function mount(root, params = {}) {
  const session = pickSession(params);
  const sections = session && session.content && Array.isArray(session.content.sections) ? session.content.sections : [];
  const model = buildModel(sections);

  if (!session || session.mode !== 'reading' || !model.total) {
    root.append(h('div', { class: 'view-error' },
      h('h1', { class: 'h2' }, 'No reading session is open'),
      h('p', { class: 'muted' }, 'Start a study session from Home to read a passage here.'),
      h('button', { class: 'btn btn-primary', type: 'button', onClick: () => navigate('home') }, 'Back to Home')));
    return undefined;
  }

  state.session = session;
  const overlayRoot = document.getElementById('overlay-root');
  const overlay = createLapseOverlay(overlayRoot);
  const wantedGaze = ['webgazer', 'mouse', 'none'].includes(params.gaze) ? params.gaze : 'none';

  let dead = false;
  let started = false;
  let startedAt = 0;
  let busy = false;          // the webcam is starting or the calibration screen is up
  let finishing = false;
  let finished = false;
  let confirming = false;
  let pages = [];            // each page: [{ si, a, b }] = words a..b of section number si
  let pageIdx = 0;
  let vis = null;            // what is on screen for a lapse: { lapse, wordIdx, phase, size, timer, prox, nearSince }
  let lastPos = null;
  let skipGazeStart = null;
  let clockTimer = null;
  let camTimer = null;
  let pollTimer = null;
  let resizeTimer = null;
  let confirmTimer = null;

  const completed = new Set();     // section idx values the reader has finished
  const feed = [];                 // Insights, oldest first: {kind:'section', idx} | {kind:'lapse', lapse, page}
  const shown = new Set();         // feed entries that have already faded in
  const reported = new Set();      // sections the server has sent a key_ideas message for
  const lapsePages = new Map();    // lapse id -> page number it happened on
  const pageHistory = [];          // [{page, t}] each time a page is shown, for the clench look-back
  const lined = new Set();         // lapse ids already listed in the feed
  const offs = [];

  // ---- skeleton --------------------------------------------------------------------------------
  const camLabel = h('span', null, 'Webcam off');
  const cam = h('div', { class: 'cam-indicator', role: 'status', title: '', dataset: { state: 'off' } },
    h('span', { class: 'cam-lens', 'aria-hidden': 'true' }), camLabel);
  // Blurred after use like every other control here: a focused input would swallow the reading keys and D/R.
  const dotToggle = h('input', { type: 'checkbox', onChange: (e) => { gaze.showDot(e.target.checked); e.currentTarget.blur(); } });
  const dotLabel = h('label', { hidden: true, title: 'Debugging aid: draws the smoothed gaze point' }, dotToggle, 'Show gaze dot');
  const recalBtn = h('button', {
    class: 'btn btn-ghost', type: 'button', hidden: true, title: 'Calibrate webcam gaze (C)',
    onClick: (e) => { e.currentTarget.blur(); recalibrate(); },
  }, 'Recalibrate');

  const pageEl = h('div', { class: 'page', lang: 'en' });
  const coverTitle = h('p', { class: 'h3' });
  const coverText = h('p');
  const coverBtn = h('button', { class: 'btn', type: 'button', onClick: () => skipGazeStart && skipGazeStart() }, 'Skip gaze');
  const cover = h('div', { class: 'passage-cover', hidden: true }, h('span', { class: 'spinner' }), coverTitle, coverText, coverBtn);
  const bodyEl = h('div', { class: 'passage-body' }, pageEl, cover);

  const prevBtn = h('button', { class: 'btn', type: 'button', onClick: (e) => { e.currentTarget.blur(); goTo(pageIdx - 1); } }, '← Previous');
  const nextBtn = h('button', { class: 'btn', type: 'button', onClick: (e) => { e.currentTarget.blur(); goTo(pageIdx + 1); } }, 'Next →');
  const pageCount = h('span', { class: 'page-count' });
  const footError = h('span', { class: 'form-error', hidden: true });
  const finishBtn = h('button', { class: 'btn', type: 'button', onClick: (e) => { e.currentTarget.blur(); finish(); } }, 'Finish reading');

  const passage = h('section', { class: 'panel passage-panel is-preparing' },
    h('header', { class: 'passage-head' },
      h('div', { class: 'passage-heading' },
        h('span', { class: 'eyebrow' }, 'Passage'),
        h('span', { class: 'passage-title', title: session.title || '' }, session.title || '')),
      cam,
      h('div', { class: 'passage-tools' }, dotLabel, recalBtn)),
    bodyEl,
    h('footer', { class: 'passage-foot' },
      prevBtn, pageCount, nextBtn,
      h('span', { class: 'keys' }, h('kbd', null, '←'), ' ', h('kbd', null, '→'), ' or ', h('kbd', null, 'Space')),
      h('span', { class: 'spacer' }), footError, finishBtn));

  const stopBadge = h('span', { class: 'badge', hidden: true });
  const stopNote = h('p', { class: 'stop-note', hidden: true }, 'Gaze unavailable: marking the top of the page');
  const stopApprox = h('p', { class: 'stop-note stop-approx', hidden: true }, NOTE_APPROXIMATE);
  const stopCard = h('div', { class: 'stop-card', role: 'alert', hidden: true },
    // The sketch's words, exactly.
    h('p', { class: 'stop-text' }, h('strong', null, 'STOP!'), " You lost focus, go back to where it's circled"),
    stopApprox, stopBadge, stopNote,
    // The headband notices the head turning, not when you are looking at the page again. If the turn
    // back isn't picked up, the reader can end the lapse here (tracker.endLapse).
    h('button', { class: 'btn stop-back', type: 'button',
      onClick: (e) => { e.stopPropagation(); e.currentTarget.blur(); backNow(); } }, "I'm back"),
    h('p', { class: 'stop-hint' }, 'or press ', h('kbd', null, 'Enter')));

  const gazeNoticeText = h('span');
  const gazeNoticeBtn = h('button', { class: 'btn', type: 'button', onClick: (e) => { e.currentTarget.blur(); recalibrate(); } }, 'Calibrate webcam gaze');
  const gazeNotice = h('div', { class: 'notice notice-warn', hidden: true }, gazeNoticeText, gazeNoticeBtn);
  const extractiveNote = h('p', { class: 'notice', hidden: true }, 'Extractive: add an API key for AI key ideas');
  const estimateNote = h('p', { class: 'estimate-note', hidden: true }, ESTIMATE_NOTE);
  const failNote = h('p', { class: 'notice notice-warn', hidden: true });
  const emptyNote = h('p', { class: 'insights-empty' }, 'Key ideas appear here as you finish each page.');
  const feedEl = h('ul', { class: 'idea-feed' });
  const insightsBody = h('div', { class: 'insights-body' },
    gazeNotice, stopCard,
    h('div', { class: 'insights-feed-wrap' },
      h('h3', { class: 'insights-title' }, 'Key ideas so far'), extractiveNote, failNote, emptyNote, feedEl, estimateNote));
  const insights = h('aside', { class: 'panel insights-panel' },
    h('header', { class: 'insights-head' }, h('span', { class: 'eyebrow' }, 'Insights')), insightsBody);

  root.append(h('div', { class: 'reading' }, passage, insights));

  const clock = h('span', { class: 'session-clock' }, '0:00');
  const slot = document.getElementById('topbar-session');
  if (slot) {
    const label = [session.subject, session.title].filter(Boolean).join(' · ');
    slot.replaceChildren(h('span', { class: 'session-meta', title: label }, label), clock);
  }

  // ---- pages -----------------------------------------------------------------------------------
  function fragmentNodes(frag) {
    const sec = model.secs[frag.si];
    const nodes = [];
    let p = null;
    let para = -1;
    for (let k = frag.a; k < frag.b; k += 1) {
      const w = sec.words[k];
      if (w.para !== para) {
        para = w.para;
        // A paragraph continued from the previous page is not indented again.
        const continued = k === frag.a && k > 0 && sec.words[k - 1].para === w.para;
        p = h('p', { class: continued ? 'para cont' : 'para', dataset: { s: sec.idx } });
        nodes.push(p);
      } else {
        p.append(' ');
      }
      p.append(h('span', { class: 'w', dataset: { w: sec.start + k, s: sec.idx } }, w.t));
    }
    return nodes;
  }

  function lastSentenceEnd(sec, a, b) {
    let best = 0;
    for (const end of sec.ends) {
      if (end > b) break;
      if (end > a) best = end;
    }
    return best;
  }

  // Build pages by measuring real layout.
  function paginate() {
    const measure = h('div', { class: 'page page-measure', 'aria-hidden': 'true' });
    bodyEl.appendChild(measure);
    const box = measure.getBoundingClientRect();
    const order = model.secs.map((sec, si) => ({ si, n: sec.words.length })).filter((s) => s.n > 0);

    if (box.width < 240 || box.height < 120) {
      // No usable layout to measure (window far too small): one scrolling section per page.
      measure.remove();
      pageEl.classList.add('is-unmeasured');
      return order.map(({ si, n }) => [{ si, a: 0, b: n }]);
    }
    pageEl.classList.remove('is-unmeasured');

    const fits = (frags) => {
      measure.replaceChildren(...frags.flatMap((f) => fragmentNodes(f)));
      const words = measure.getElementsByClassName('w');
      const last = words[words.length - 1];
      if (!last) return true;
      if (measure.scrollWidth > measure.clientWidth + 1) return false;   // text spilled into a third column
      const r = last.getBoundingClientRect();
      return r.right <= box.right + 1 && r.bottom <= box.bottom + 1;
    };

    // Fill each page like a book (changed at integration, 2026-11-15: keeping whole sections together
    // left the second column half empty). When the rest of a section does not fit, as much of it as fits
    // goes on this page, ending at a sentence when one is close, and it continues on the next page.
    const built = [];
    let page = [];
    for (const { si, n } of order) {
      let a = 0;
      while (a < n) {
        const rest = { si, a, b: n };
        if (fits([...page, rest])) {
          page.push(rest);
          a = n;
          continue;
        }
        let lo = a;            // most words of this section that still fit on the page (a = none)
        let hi = n - 1;
        while (lo < hi) {
          const mid = Math.ceil((lo + hi) / 2);
          if (fits([...page, { si, a, b: mid }])) lo = mid;
          else hi = mid - 1;
        }
        let b = lo;
        if (b > a) {
          const sentenceEnd = lastSentenceEnd(model.secs[si], a, b);
          if (sentenceEnd && sentenceEnd - a >= 0.5 * (b - a)) b = sentenceEnd;
        }
        if (b - a >= (page.length ? 12 : 1)) {
          page.push({ si, a, b });
          a = b;
        } else if (!page.length) {
          b = a + 1;           // always make progress, even in a tiny window
          page.push({ si, a, b });
          a = b;
        }
        built.push(page);
        page = [];
      }
    }
    if (page.length) built.push(page);
    measure.remove();
    return built;
  }

  function pageOfWord(wordIdx) {
    const i = pages.findIndex((p) => p.some((f) => {
      const base = model.secs[f.si].start;
      return wordIdx >= base + f.a && wordIdx < base + f.b;
    }));
    return i < 0 ? 0 : i;
  }

  function resetFinishConfirm() {
    clearTimeout(confirmTimer);
    confirming = false;
    if (!finishing) finishBtn.textContent = 'Finish reading';
  }

  function renderPage() {
    // Every page shown, from the first render on (also before Start), so the clench look-back knows which
    // page was up at any earlier moment.
    if (!pageHistory.length || pageHistory[pageHistory.length - 1].page !== pageIdx) pageHistory.push({ page: pageIdx, t: Date.now() });
    pageEl.replaceChildren(...(pages[pageIdx] || []).flatMap((f) => fragmentNodes(f)));
    pageEl.scrollTop = 0;
    const last = pageIdx >= pages.length - 1;
    pageCount.textContent = `Page ${pageIdx + 1} of ${pages.length}`;
    prevBtn.disabled = pageIdx === 0;
    nextBtn.disabled = last;
    finishBtn.classList.toggle('btn-primary', last);
    resetFinishConfirm();
    if (vis) applyMark();
  }

  // A section is complete once the reader leaves a page that holds its last word.
  function completePage(i) {
    let changed = false;
    for (const frag of pages[i] || []) {
      const sec = model.secs[frag.si];
      if (frag.b !== sec.words.length || completed.has(sec.idx)) continue;
      completed.add(sec.idx);
      tracker.markProgress(sec.idx);
      feed.push({ kind: 'section', idx: sec.idx, key: `s${sec.idx}` });
      changed = true;
    }
    if (changed) renderFeed();
  }

  function goTo(i) {
    if (!started || busy || vis || finishing) return;
    const to = Math.min(pages.length - 1, Math.max(0, i));
    if (to === pageIdx) return;
    completePage(pageIdx);
    pageIdx = to;
    renderPage();
  }

  function repaginate() {
    if (dead || !pages.length) return;
    const first = pages[pageIdx] && pages[pageIdx][0];
    const firstWord = first ? model.secs[first.si].start + first.a : 0;
    // Stay with the circled word if there is one, otherwise with the top of the current page.
    const anchor = vis && vis.wordIdx !== null ? vis.wordIdx : firstWord;
    pages = paginate();
    pageIdx = pageOfWord(anchor);
    renderPage();
  }

  // ---- Insights: key ideas and lapse lines ---------------------------------------------------------
  function ideasFor(idx) {
    return (session.key_ideas || []).filter((k) => k.section_idx === idx);
  }

  function ideasSettled(idx) {
    return reported.has(idx) || STATUS_RANK[session.key_ideas_status] === 2;
  }

  function fresh(key) {
    if (shown.has(key)) return false;
    shown.add(key);
    return true;
  }

  // A lapse from the headband (live or replayed) has an estimated length: the detector sees the head turn,
  // not the look back (state.js). "about" when the turn back was seen, "at least" when only the turn away
  // was. Simulated lapses, and any other or unknown end_method, keep the plain form.
  function isEstimated(lapse) {
    return lapse.source === 'live' || lapse.source === 'replay';
  }

  function lapseLength(lapse) {
    const seconds = Math.max(1, Math.round(lapse.duration_s || 0));
    if (isEstimated(lapse) && lapse.end_method === 'turn_back') return `about ${seconds} s`;
    if (isEstimated(lapse) && lapse.end_method === 'movement_end') return `at least ${seconds} s`;
    return `${seconds} s`;
  }

  function lapseLine(item) {
    const { lapse, page } = item;
    const clench = lapse.kind === 'clench';
    // A clench lapse is the reader's own mark: its length is the fixed 10-s look-back, not a measurement.
    const where = clench ? (page ? `You marked a zone-out on page ${page}` : 'You marked a zone-out')
      : (page ? `Lost focus on page ${page}` : 'Lost focus');
    const line = h('li', { class: fresh(item.key) ? 'lapse-line is-fresh' : 'lapse-line' },
      clench ? `${where} (jaw clench)` : `${where} · ${lapseLength(lapse)}`);
    if (lapse.source !== 'live') {
      // Honesty rule: a lapse that did not come from the headband is marked wherever it is listed.
      line.append(h('span', { class: lapse.source === 'replay' ? 'badge badge-replay' : 'badge badge-sim' }, String(lapse.source).toUpperCase()));
    }
    return line;
  }

  function renderFeed() {
    const atBottom = insightsBody.scrollHeight - insightsBody.scrollTop - insightsBody.clientHeight < 40;
    const rows = [];
    let extractive = false;
    for (const item of feed) {
      if (item.kind === 'lapse') {
        rows.push(lapseLine(item));
        continue;
      }
      const ideas = ideasFor(item.idx);
      if (ideas.length) {
        for (const idea of ideas) {
          if (idea.source === 'extractive') extractive = true;
          const key = `i${idea.id ?? `${idea.section_idx}:${idea.text}`}`;
          rows.push(h('li', { class: fresh(key) ? 'idea is-fresh' : 'idea' },
            h('p', { class: 'idea-text' }, idea.text),
            idea.keyword ? h('span', { class: 'chip' }, idea.keyword) : null));
        }
      } else if (!ideasSettled(item.idx)) {
        rows.push(h('li', { class: 'idea-writing' }, h('span', { class: 'spinner' }), 'writing…'));
      }
    }
    feedEl.replaceChildren(...rows);
    emptyNote.hidden = feed.length > 0;
    extractiveNote.hidden = !extractive;
    estimateNote.hidden = !feed.some((item) => item.kind === 'lapse' && isEstimated(item.lapse));
    const failed = session.key_ideas_status === 'failed';
    failNote.hidden = !failed;
    if (failed) failNote.textContent = `Key ideas could not be written${session.key_ideas_error ? `: ${session.key_ideas_error}` : '.'}`;
    if (atBottom) insightsBody.scrollTop = insightsBody.scrollHeight;
  }

  async function reloadIdeas() {
    try {
      const latest = await api.get(`/api/sessions/${session.id}`);
      if (dead || !latest) return;
      mergeKeyIdeas(session, latest.key_ideas);
      // Never step backwards: a WebSocket message may have overtaken this reply.
      if ((STATUS_RANK[latest.key_ideas_status] ?? -1) >= (STATUS_RANK[session.key_ideas_status] ?? -1)) {
        session.key_ideas_status = latest.key_ideas_status;
        session.key_ideas_error = latest.key_ideas_error ?? null;
      }
      renderFeed();
    } catch (err) {
      console.warn('could not reload key ideas:', err.message);
    }
  }

  // ---- gaze ------------------------------------------------------------------------------------
  // The passage's line pitch in px: used to say the gaze error in lines of text.
  function lineHeightPx() {
    const cs = getComputedStyle(pageEl);
    const lh = parseFloat(cs.lineHeight);
    return Number.isFinite(lh) && lh > 0 ? lh : 1.9 * (parseFloat(cs.fontSize) || 23);
  }

  // What the calibration screen needs from this view: where the passage is, and its line height.
  const calibOptions = { area: () => pageEl.getBoundingClientRect(), lineHeight: lineHeightPx };

  function refreshCam() {
    const s = gaze.status();
    let stateName = 'off';
    let text = 'Webcam off';
    let tip = '';
    if (s.calibrating) {
      stateName = 'calibrating';
      text = 'Calibrating';
    } else if (busy) {
      stateName = 'calibrating';
      text = 'Starting webcam';
    } else if (s.mode === 'mouse') {
      stateName = 'mouse';
      text = 'MOUSE (testing)';
    } else if (s.mode === 'webgazer' && s.calibrated) {
      stateName = s.face ? 'tracking' : 'lost';
      text = s.face ? 'Tracking gaze' : 'Face lost';
      // The measured gaze error is shown here, where it can always be seen.
      if (s.error_px === null) {
        text += ' · not checked';
        tip = 'Gaze accuracy was not measured (the check after calibration could not be done). The lost-focus circle uses a default width.';
      } else {
        text += ` · ~${roundGazeError(s.error_px)} px`;
        tip = `Gaze accuracy: ${describeGazeError(s.error_px, lineHeightPx())}. Measured right after calibration, sitting still; it can get worse as you move.`;
      }
    }
    if (cam.dataset.state !== stateName) cam.dataset.state = stateName;
    if (camLabel.textContent !== text) camLabel.textContent = text;
    if (cam.title !== tip) cam.title = tip;
  }

  function showGazeNotice(text) {
    gazeNoticeText.textContent = text;
    gazeNotice.hidden = false;
  }

  function syncGazeUi() {
    const s = gaze.status();
    dotLabel.hidden = s.mode === 'none';
    recalBtn.hidden = wantedGaze !== 'webgazer' || s.mode !== 'webgazer';
    if (started) tracker.setGaze({ mode: s.mode, calibrated: s.calibrated, error_px: s.error_px });
    refreshCam();
  }

  // Start a gaze source and, for the webcam, run the calibration. Every failure ends in mode "none"
  // with a visible note; reading always goes ahead.
  async function setupGaze(requested) {
    busy = true;
    refreshCam();
    try {
      let result;
      if (requested === 'webgazer') {
        coverTitle.textContent = 'Starting the webcam';
        coverText.textContent = 'Allow camera access if you are asked. The picture is processed on this Mac and is not shown or recorded.';
        cover.hidden = false;
        const skipped = new Promise((resolve) => { skipGazeStart = () => resolve(null); });
        result = await Promise.race([gaze.start('webgazer'), skipped]);
        skipGazeStart = null;
        cover.hidden = true;
        if (dead) return;
        if (result === null) {
          gaze.stop();
          showGazeNotice(NOTE_SKIPPED);
          return;
        }
      } else {
        result = await gaze.start(requested);
        if (dead) return;
      }

      if (result.mode === 'webgazer') {
        refreshCam();
        const ok = await gaze.calibrate(overlayRoot, calibOptions);
        if (dead) return;
        if (ok) {
          gazeNotice.hidden = true;
        } else {
          gaze.stop();   // no calibration, no use for the camera
          showGazeNotice(NOTE_NOT_CALIBRATED);
        }
      } else if (requested === 'webgazer') {
        showGazeNotice(`${result.note || 'Webcam gaze could not start.'} A lapse will mark the top of the page.`);
      } else {
        gazeNotice.hidden = true;
      }
    } finally {
      busy = false;
      if (!dead) syncGazeUi();
    }
  }

  async function recalibrate() {
    if (!started || busy || vis || finishing || wantedGaze !== 'webgazer') return;
    // The reader is about to look all over the screen: don't log that as lapses.
    tracker.hold(true);
    try {
      if (gaze.status().mode === 'webgazer') {
        busy = true;
        refreshCam();
        try {
          const ok = await gaze.calibrate(overlayRoot, calibOptions);
          if (dead) return;
          if (ok) {
            gazeNotice.hidden = true;
          } else if (!gaze.status().calibrated) {
            gaze.stop();
            showGazeNotice(NOTE_NOT_CALIBRATED);
          }
        } finally {
          busy = false;
          if (!dead) syncGazeUi();
        }
      } else {
        await setupGaze('webgazer');
      }
    } finally {
      if (!dead) tracker.hold(false);
    }
  }

  // ---- where focus was lost ----------------------------------------------------------------------
  function onPassage(pt) {
    const r = pageEl.getBoundingClientRect();
    const m = PASSAGE_MARGIN_PX;
    return pt.x >= r.left - m && pt.x <= r.right + m && pt.y >= r.top - m && pt.y <= r.bottom + m;
  }

  // The word a gaze point belongs to: the nearest line first, then the nearest word on it (gaze.js). For webcam
  // gaze it then steps back up half the measured gaze error in lines, rounded up (Ayur, 2026-11-15: "to be safe,
  // bring me back to the word half that many lines above"), so the mark errs before where reading stopped.
  function nearestWord(pt) {
    const spans = Array.from(pageEl.getElementsByClassName('w'));
    const rects = spans.map((el) => el.getBoundingClientRect());
    let i = snapToLine(rects, pt);
    if (i < 0) return null;
    if (gaze.status().mode === 'webgazer') {
      const err = gaze.status().error_px ?? DEFAULT_GAZE_ERROR_PX;
      i = wordLinesAbove(rects, i, safetyLines(err, lineHeightPx()));
    }
    return spans[i];
  }

  // The first word of the first section on this page that is not yet complete.
  function fallbackWord() {
    const frags = pages[pageIdx] || [];
    const frag = frags.find((f) => !completed.has(model.secs[f.si].idx)) || frags[0];
    if (!frag) return null;
    return pageEl.querySelector(`.w[data-w="${model.secs[frag.si].start + frag.a}"]`);
  }

  function positionOf(span, method) {
    const wordIdx = Number(span.dataset.w);
    const sentence = model.sentences[model.sentOf[wordIdx]];
    // `page` is extra to the §7 shape: only the browser knows the pages, and the server uses it for
    // the "Page 2 · 01:42" label of a missed item.
    return { section_idx: Number(span.dataset.s), word_idx: wordIdx, sentence: sentence ? sentence.text : '', method, page: pageIdx + 1 };
  }

  function locate(evt) {
    const onset = evt && Number.isFinite(evt.onset_wall_ms) ? evt.onset_wall_ms : evt && evt.wall_ms;
    // The gaze from before the head moved, using only points on the passage. Null without a usable
    // (calibrated) gaze source, or when too few such points were seen.
    const pt = gaze.at(onset, onPassage);
    if (pt) {
      const span = nearestWord(pt);
      if (span) return positionOf(span, gaze.status().mode === 'mouse' ? 'mouse' : 'gaze');
    }
    const span = fallbackWord();
    return span ? positionOf(span, 'fallback_top') : null;
  }

  // Called by the tracker at drift and again at resume; the place is decided once, at the drift.
  function getPosition(evt) {
    if (!evt || evt.type !== 'resume' || !lastPos) lastPos = locate(evt);
    return lastPos;
  }

  // ---- lapse visuals -----------------------------------------------------------------------------
  function clearHighlight() {
    for (const el of pageEl.querySelectorAll('.sent-hl, .lost')) el.classList.remove('sent-hl', 'lost');
  }

  // How wide to draw the circle. A place taken from webcam gaze is only as exact as the gaze: the
  // circle is sized from the error measured after calibration (a default when it was not measured).
  // The mouse and the top-of-page fallback are exact places and keep the tight circle (null).
  function markSize(pos) {
    if (!pos || pos.method !== 'gaze') return null;
    const err = gaze.status().error_px ?? DEFAULT_GAZE_ERROR_PX;
    return err > WIDE_MARK_ABOVE_PX ? radiiForError(err) : null;
  }

  // Put up the STOP card for a lapse and remember what is on screen for it.
  function openVisuals(lapse, wordIdx, phase) {
    const pos = lapse.position;
    vis = { lapse, wordIdx, phase, size: markSize(pos), timer: null, prox: null, nearSince: 0 };
    const live = lapse.source === 'live';
    stopBadge.hidden = live;
    stopBadge.className = lapse.source === 'replay' ? 'badge badge-replay' : 'badge badge-sim';
    stopBadge.textContent = String(lapse.source).toUpperCase();
    stopNote.hidden = Boolean(pos) && pos.method !== 'fallback_top';
    stopApprox.hidden = !vis.size;
    stopCard.hidden = false;
    insights.classList.add('is-stopped');
    insightsBody.scrollTop = 0;
    applyMark();
  }

  function applyMark() {
    clearHighlight();
    if (!vis || vis.wordIdx === null) return;
    const span = pageEl.querySelector(`.w[data-w="${vis.wordIdx}"]`);
    if (!span) {
      overlay.clear();
      return;
    }
    const sentence = model.sentences[model.sentOf[vis.wordIdx]];
    if (sentence) {
      for (const el of pageEl.getElementsByClassName('w')) {
        const i = Number(el.dataset.w);
        if (i >= sentence.from && i < sentence.to) el.classList.add('sent-hl');
      }
    }
    span.classList.add('lost');
    overlay.mark(span, vis.size);
    if (vis.phase === 'guide') overlay.point(arrowStart);
  }

  // The arrow starts at the reader's gaze; with no gaze it starts from the STOP card.
  function arrowStart() {
    const p = gaze.current();
    if (p) return p;
    const r = stopCard.getBoundingClientRect();
    return { x: r.left - 6, y: r.top + r.height / 2 };
  }

  function addLapseLine(lapse) {
    if (lined.has(lapse.id)) return;
    lined.add(lapse.id);
    feed.push({ kind: 'lapse', lapse, page: lapsePages.get(lapse.id), key: `l${lapse.id}` });
    renderFeed();
  }

  function endVisuals() {
    if (!vis) return;
    const { lapse, phase, timer, prox } = vis;
    clearTimeout(timer);
    clearInterval(prox);
    vis = null;
    overlay.clear();
    clearHighlight();
    stopCard.hidden = true;
    insights.classList.remove('is-stopped');
    // If the lapse is still open (dismissed with Esc), its line is added when it ends.
    if (phase === 'guide') addLapseLine(lapse);
  }

  function checkProximity() {
    if (!vis || vis.phase !== 'guide') return;
    // Back = the gaze is inside the circle or just outside it (for a tight circle: within NEAR_PX).
    if (nearMark(gaze.current(), overlay.geometry(), NEAR_MARGIN_PX, NEAR_PX)) {
      if (!vis.nearSince) vis.nearSince = Date.now();
      else if (Date.now() - vis.nearSince >= NEAR_HOLD_MS) endVisuals();
    } else {
      vis.nearSince = 0;
    }
  }

  // The reader says they are back: end the lapse by hand (guidance then shows as for any lapse end).
  function backNow() {
    if (!vis || vis.phase !== 'open') return;
    if (!tracker.endLapse('manual')) endVisuals();
  }

  function onLapseStart({ lapse }) {
    if (dead || finishing || !lapse) return;
    // Keep the gaze trace and the passage box with the lapse, to check later where the mark should have been.
    if (gaze.status().mode === 'webgazer') {
      const r = pageEl.getBoundingClientRect();
      lapse.gaze_trace = { passage: [Math.round(r.left), Math.round(r.top), Math.round(r.right), Math.round(r.bottom)],
        samples: gaze.trace() };
    }
    endVisuals();   // guidance from the previous lapse, if it is still up
    lapsePages.set(lapse.id, pageIdx + 1);
    audio.chime();

    const pos = lapse.position;
    openVisuals(lapse, pos && Number.isInteger(pos.word_idx) ? pos.word_idx : null, 'open');
  }

  function onLapseEnd({ lapse }) {
    if (dead || !lapse) return;
    if (!vis || vis.lapse.id !== lapse.id) {
      addLapseLine(lapse);
      return;
    }
    vis.phase = 'guide';
    if (finishing) {
      endVisuals();
      return;
    }
    overlay.point(arrowStart);
    vis.timer = setTimeout(endVisuals, GUIDE_MAX_MS);
    vis.prox = setInterval(checkProximity, 100);
  }

  // A second head movement soon after a lapse is taken as the reader turning back (state.js). They are
  // looking at the screen again now, so show the circle and the arrow once more.
  function onLapseReturn({ lapse }) {
    if (dead || finishing || !lapse) return;
    endVisuals();
    const pos = lapse.position;
    const wordIdx = pos && Number.isInteger(pos.word_idx) ? pos.word_idx : null;
    if (wordIdx === null) return;
    const page = pageOfWord(wordIdx);
    // Jump to the lapse's page directly, not through goTo(): that would mark the page being left as read.
    if (page !== pageIdx && started && !busy) {
      pageIdx = page;
      renderPage();
    }
    openVisuals(lapse, wordIdx, 'guide');
    vis.timer = setTimeout(endVisuals, GUIDE_MAX_MS);
    vis.prox = setInterval(checkProximity, 100);
    renderFeed();    // the lapse's line now shows its full length
  }

  // A held jaw clench = "I zoned out" (state.js). Go back to where the gaze was CLENCH_LOOKBACK_MS before the
  // clench finished, on the page that was showing then, and circle it as a lapse the reader marked.
  function pageAt(t) {
    let p = null;
    for (const e of pageHistory) if (e.t <= t) p = e.page;
    return p ?? (pageHistory.length ? pageHistory[0].page : pageIdx);
  }

  function onClench(trig) {
    if (dead || finishing || !started || busy || !trig) return;
    const target = trig.wall_ms - CLENCH_LOOKBACK_MS;
    endVisuals();
    const page = pageAt(target);
    // Jump straight to that page, not through goTo(): that would mark the page being left as read.
    if (page !== pageIdx) {
      pageIdx = page;
      renderPage();
    }
    const pt = gaze.at(target, onPassage);
    let span = pt ? nearestWord(pt) : null;
    const method = span ? (gaze.status().mode === 'mouse' ? 'mouse' : 'gaze') : 'fallback_top';
    if (!span) span = fallbackWord();
    const position = span ? positionOf(span, method) : null;
    const lapse = tracker.addLapse({ source: trig.source, onset_ms: target, resume_ms: trig.onset_wall_ms, position });
    if (!lapse) return;
    lapsePages.set(lapse.id, pageIdx + 1);
    audio.chime();
    openVisuals(lapse, position && Number.isInteger(position.word_idx) ? position.word_idx : null, 'guide');
    vis.timer = setTimeout(endVisuals, GUIDE_MAX_MS);
    vis.prox = setInterval(checkProximity, 100);
  }

  // ---- input -------------------------------------------------------------------------------------
  function isSimKey(e) {
    return Boolean(state.status && state.status.source === 'simulated') && /^[drz]$/i.test(e.key);
  }

  function onKey(e) {
    if (dead || e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey || isTyping(e.target)) return;
    if (vis) {
      if (vis.phase === 'open') {
        // The lapse is still going on: Enter means "I'm back"; Esc takes the STOP card down.
        if (e.key === 'Enter') {
          e.preventDefault();
          backNow();
        } else if (e.key === 'Escape') {
          e.preventDefault();
          endVisuals();
        }
        return;
      }
      // After the resume, any key clears the guidance (and does nothing else).
      if (e.repeat || MODIFIER_KEYS.has(e.key) || isSimKey(e)) return;
      e.preventDefault();
      endVisuals();
      return;
    }
    const onControl = e.target instanceof Element && e.target.closest('button, a, summary');
    if (onControl && (e.key === ' ' || e.key === 'Enter')) return;   // let the focused control act
    if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {
      e.preventDefault();
      goTo(pageIdx + 1);
    } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
      e.preventDefault();
      goTo(pageIdx - 1);
    } else if (e.key === 'c' || e.key === 'C') {
      recalibrate();
    }
  }

  // After the resume, any click clears the guidance; the click itself is used up.
  function onDocClick(e) {
    if (!vis || vis.phase !== 'guide') return;
    e.preventDefault();
    e.stopPropagation();
    endVisuals();
  }

  function onResize() {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(repaginate, 200);
  }

  // ---- finish ------------------------------------------------------------------------------------
  async function finish() {
    if (!started || finishing || busy) return;
    const last = pageIdx >= pages.length - 1;
    if (!last && !confirming) {
      confirming = true;
      finishBtn.textContent = `Finish on page ${pageIdx + 1} of ${pages.length}?`;
      confirmTimer = setTimeout(resetFinishConfirm, 4000);
      return;
    }
    resetFinishConfirm();
    finishing = true;
    endVisuals();
    if (last) completePage(pageIdx);
    footError.hidden = true;
    finishBtn.disabled = true;
    finishBtn.replaceChildren(h('span', { class: 'spinner' }), 'Finishing');
    try {
      const done = await tracker.finish();
      finished = true;
      if (!dead) navigate('summary', { id: (done && done.id) || session.id });
    } catch (err) {
      finishing = false;
      if (dead) return;
      finishBtn.disabled = false;
      finishBtn.textContent = 'Finish reading';
      footError.textContent = `Could not finish: ${err.message}`;
      footError.title = err.message;
      footError.hidden = false;
    }
  }

  // ---- start -------------------------------------------------------------------------------------
  async function start() {
    pages = paginate();
    renderPage();
    // Gaze is set up before the session clock starts, so calibration time is not counted as reading.
    if (wantedGaze !== 'none') await setupGaze(wantedGaze);
    if (dead) return;

    tracker.begin(session, { getPosition });
    started = true;
    startedAt = Date.now();
    passage.classList.remove('is-preparing');
    syncGazeUi();
    renderFeed();
    reloadIdeas();

    clockTimer = setInterval(() => {
      clock.textContent = fmtClock((Date.now() - startedAt) / 1000);
    }, 1000);
    // Key ideas normally arrive over the WebSocket; this catches any that were missed while it was down.
    pollTimer = setInterval(() => {
      const waiting = feed.some((item) => item.kind === 'section' && !ideasFor(item.idx).length && !ideasSettled(item.idx));
      if (waiting) reloadIdeas();
    }, IDEA_POLL_MS);
  }

  offs.push(
    bus.on('lapse:start', onLapseStart),
    bus.on('lapse:end', onLapseEnd),
    bus.on('lapse:return', onLapseReturn),
    bus.on('clench:trigger', onClench),
    bus.on('keyideas', (msg) => {
      if (msg && Number.isInteger(msg.section_idx)) reported.add(msg.section_idx);
      renderFeed();
    }),
  );
  window.addEventListener('keydown', onKey);
  window.addEventListener('resize', onResize);
  document.addEventListener('click', onDocClick, true);
  camTimer = setInterval(refreshCam, 400);

  start().catch((err) => {
    console.error('reading view failed to start', err);
    if (dead) return;
    cover.hidden = true;
    passage.classList.remove('is-preparing');
    footError.textContent = `The session could not start: ${err.message}`;
    footError.hidden = false;
  });

  return () => {
    dead = true;
    for (const off of offs) off();
    window.removeEventListener('keydown', onKey);
    window.removeEventListener('resize', onResize);
    document.removeEventListener('click', onDocClick, true);
    for (const t of [clockTimer, camTimer, pollTimer]) clearInterval(t);
    clearTimeout(resizeTimer);
    clearTimeout(confirmTimer);
    if (vis) {
      clearTimeout(vis.timer);
      clearInterval(vis.prox);
      vis = null;
    }
    overlay.clear();
    gaze.showDot(false);
    gaze.stop();
    // Leaving the view ends the session with what was actually read (same rule as the video view).
    if (started && !finished && !finishing && tracker.session() === session) {
      tracker.finish().catch((err) => console.warn('could not close the session:', err.message));
    }
  };
}

export const view = { mount };
