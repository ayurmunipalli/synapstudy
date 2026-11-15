// End-of-session Insights, full screen. Reading: "Where you lost focus". Video: "Here's What You Missed".
import { api } from '../api.js';
import { bus } from '../bus.js';
import { state } from '../state.js';
import { navigate } from '../router.js';
import { h, fmtClock, fmtPct, fmtNum } from '../lib/dom.js';
import { loadPlayer } from '../lib/youtube.js';
import { timeline } from '../lib/charts.js';

const REPLAY_LEAD_S = 3;
const ESTIMATE_NOTE =
  'Lapse lengths and focus % are estimates: the headband senses the head turning, not the moment you look back.';

const on = (el, type, fn) => {
  el.addEventListener(type, fn);
  return el;
};

function dateLabel(iso) {
  const d = new Date(iso);
  if (!iso || Number.isNaN(d.getTime())) return '';
  return d.toLocaleString(undefined, { year: 'numeric', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
}

// Every place a lapse is listed marks it when it did not come from the headband.
function sourceBadge(source) {
  if (source === 'simulated') return h('span', { class: 'badge badge-sim' }, 'SIMULATED');
  if (source === 'replay') return h('span', { class: 'badge badge-replay' }, 'REPLAY');
  return null;
}

// A lapse from the headband (live) or from a replayed recording has an estimated length: the tracker
// sees the head turning, not the moment the reader looks back (state.js). Simulated lapses are exact.
const isEstimated = (source) => source === 'live' || source === 'replay';

// An estimated length in whole seconds: "at least N s" (rounded down) or "about N s".
function estimatedLength(seconds, atLeast) {
  if (atLeast) return `at least ${seconds < 1 ? fmtNum(seconds, 1) : fmtNum(Math.floor(seconds))} s`;
  return seconds < 1 ? 'under 1 s' : `about ${fmtNum(Math.round(seconds))} s`;
}

// One lapse's length as shown on screen. end_method "movement_end": the head turn was seen, the return
// was not, so the length is a minimum; any other end of a headband lapse is approximate.
function lapseLength(duration_s, source, endMethod) {
  if (typeof duration_s !== 'number' || !Number.isFinite(duration_s)) return null;
  if (!isEstimated(source)) return `${fmtNum(duration_s, 1)} s`;
  return estimatedLength(duration_s, endMethod === 'movement_end');
}

// Session-level badge. Sessions saved before "measured" / "measure_note" existed lack both fields:
// they then get the SIMULATED badge as before, or none.
function sessionBadge(session) {
  const note = typeof session.measure_note === 'string' ? session.measure_note : '';
  if (/replayed recording/.test(note)) return h('span', { class: 'badge badge-replay' }, 'REPLAY');
  if (session.simulated || (session.summary && session.summary.simulated)) {
    return h('span', { class: 'badge badge-sim' }, 'SIMULATED');
  }
  if (session.measured === false) return h('span', { class: 'badge badge-unknown' }, 'NOT MEASURED');
  return null;
}

function tile(value, label, sub, textValue = false) {
  const node = h(
    'div',
    { class: 'tile' },
    h('div', { class: textValue ? 'tile-value sum-tile-text' : 'tile-value' }, value),
    h('div', { class: 'tile-label' }, label),
  );
  if (sub) node.append(h('div', { class: 'sum-tile-sub muted' }, sub));
  return node;
}

function section(title, ...children) {
  const node = h('section', { class: 'sum-section' }, h('h2', { class: 'h2 sum-section-title' }, title));
  node.append(...children.filter(Boolean));
  return node;
}

export const view = {
  mount(root, params = {}) {
    const offs = [];
    let dead = false;
    let session = null;
    let closeModal = null;

    const page = h('div', { class: 'sum-page' }, h('div', { class: 'sum-loading' }, h('div', { class: 'spinner' })));
    root.replaceChildren(page);

    function fail(message) {
      const home = on(h('button', { class: 'btn', type: 'button' }, 'Back to home'), 'click', () => navigate('home'));
      page.replaceChildren(h('div', { class: 'notice notice-warn' }, message), h('div', { class: 'sum-actions' }, home));
    }

    // ---- header and tiles -------------------------------------------------------------------------
    function header() {
      const meta = h('div', { class: 'sum-meta' });
      if (session.subject) meta.append(h('span', { class: 'chip' }, session.subject));
      meta.append(h('span', { class: 'muted' }, session.mode === 'video' ? 'Video' : 'Reading'));
      const when = dateLabel(session.created_at);
      if (when) meta.append(h('span', { class: 'muted' }, when));

      const head = h(
        'header',
        { class: 'sum-head' },
        h('div', { class: 'sum-eyebrow muted' }, 'Insights'),
        h('h1', { class: 'h1 sum-title' }, session.title || 'Untitled session'),
        meta,
      );
      // The server's measure_note is shown as it is. Sessions saved before it existed keep the old line.
      const badge = sessionBadge(session);
      const note = measureNote();
      if (badge || note) {
        const simulated = session.simulated || session.summary.simulated;
        const fallback = simulated
          ? 'This session used simulated lapses; it is excluded from long-term insights'
          : 'The headband was not connected throughout this session; it is left out of long-term insights.';
        head.append(h('div', { class: 'notice notice-warn sum-sim' }, badge, ' ', note || fallback));
      }
      return head;
    }

    function measureNote() {
      return typeof session.measure_note === 'string' && session.measure_note ? session.measure_note : null;
    }

    // Lapses as the client logged them, or (older or partial sessions) the summary's missed list.
    function lapseList() {
      const logged = (session.log && session.log.lapses) || [];
      return logged.length ? logged : session.summary.missed || [];
    }

    function endMethodOf(item) {
      if (item.end_method) return item.end_method;
      const lapse = ((session.log && session.log.lapses) || []).find((l) => l.id === item.lapse_id);
      return (lapse && lapse.end_method) || null;
    }

    // True when the screen shows a headband (or replayed) lapse length or a headband focus %.
    function showsEstimates() {
      if (lapseList().some((l) => l.kind !== 'clench' && isEstimated(l.source))) return true;
      const measured = session.measured ?? session.summary.measured;
      return measured === true && session.summary.focus_pct != null;
    }

    function tiles() {
      const s = session.summary;
      const estimated = showsEstimates();
      // Nothing was measured (no headband, no lapses): never show that as 100% focus or as zero lapses.
      const unmeasured = s.focus_pct == null;
      let focus;
      if (unmeasured) {
        focus = tile('Not measured', 'Focus', null, true);
      } else if (estimated) {
        const atLeast = lapseList().some((l) => isEstimated(l.source) && endMethodOf(l) === 'movement_end');
        const secs = s.lapse_seconds;
        const sub = secs == null ? null : secs > 0 ? `${estimatedLength(secs, atLeast)} in lapses` : '0 s in lapses';
        // Whole percent, rounded down: a tenth of a percent would overstate how well this is known.
        focus = tile(`${fmtNum(Math.floor(s.focus_pct))}%`, 'Focus (estimate)', sub);
      } else {
        focus = tile(fmtPct(s.focus_pct), 'Focus', s.lapse_seconds != null ? `${fmtNum(s.lapse_seconds, 1)} s in lapses` : null);
      }
      const lapses = unmeasured && !s.lapse_count
        ? tile('Not measured', 'Lapses', null, true)
        : tile(fmtNum(s.lapse_count), 'Lapses', s.lapses_per_min != null ? `${fmtNum(s.lapses_per_min, 2)} per minute` : null);
      const wrap = h('div', { class: 'sum-tiles' }, tile(fmtClock(s.duration_s), 'Time'), focus, lapses);
      if (session.mode === 'video') {
        const logVideo = (session.log && session.log.video) || {};
        const watched = (s.video && s.video.watched_s) ?? logVideo.watched_s;
        const length = (s.video && s.video.duration_s) ?? logVideo.duration_s;
        wrap.append(tile(fmtClock(watched), 'Watched', length ? `of ${fmtClock(length)}` : null));
      } else {
        const r = s.reading || {};
        const sub = r.wpm == null ? 'Too little reading to measure' : `${fmtNum(r.words_read)} words read`;
        wrap.append(tile(fmtNum(r.wpm), 'Reading speed (wpm)', sub));
      }
      return wrap;
    }

    // ---- timeline ---------------------------------------------------------------------------------
    function timelineSection() {
      const s = session.summary;
      const log = session.log || {};
      const lapses = (log.lapses || []).map((l, i) => {
        const start = ((l.onset_ms || 0) - (log.started_ms || 0)) / 1000;
        const dur = l.duration_s ?? (l.resume_ms && l.onset_ms ? (l.resume_ms - l.onset_ms) / 1000 : 0);
        const live = l.source === 'live';
        return {
          start_s: start,
          duration_s: dur,
          live,
          title: `Lapse ${i + 1} at ${fmtClock(Math.max(0, start))}`,
          detail: `${lapseLength(dur, l.source, l.end_method) ?? `${fmtNum(dur, 1)} s`}${live ? '' : ` · ${l.source || 'not live'}`}`,
        };
      });
      if (!(s.duration_s > 0)) return null;

      const legend = h('div', { class: 'sum-legend muted' });
      if (lapses.some((l) => l.live)) {
        legend.append(h('span', { class: 'sum-legend-item' }, h('span', { class: 'sum-swatch' }), 'Lapse (headband)'));
      }
      if (lapses.some((l) => !l.live)) {
        legend.append(
          h('span', { class: 'sum-legend-item' }, h('span', { class: 'sum-swatch sum-swatch-sim' }), 'Lapse, not live (simulated or replay)'),
        );
      }
      if (!lapses.length) legend.append('No lapses on the timeline.');
      return section('Session timeline', timeline({ duration_s: s.duration_s, lapses }), legend);
    }

    // ---- key ideas --------------------------------------------------------------------------------
    const ideasBody = h('div', { class: 'sum-ideas' });

    function renderIdeas() {
      const sections = (session.content && session.content.sections) || [];
      const status = session.key_ideas_status;
      const rows = [];

      // Only the parts actually covered in this session (sections finished, plus any a lapse fell in):
      // key ideas are about what was studied, not the whole document or video.
      const log = session.log || {};
      const covered = new Set((log.progress || []).map((p) => p.section_idx));
      for (const lapse of log.lapses || []) {
        const pos = lapse.position || {};
        if (Number.isInteger(pos.section_idx)) covered.add(pos.section_idx);
        if (typeof pos.t_start === 'number') {
          for (const sec of sections) {
            if (sec.start_s != null && sec.end_s > pos.t_start && sec.start_s < (pos.t_end ?? pos.t_start) + 0.001) covered.add(sec.idx);
          }
        }
      }
      const all = session.key_ideas || [];
      const ideas = all.filter((i) => covered.has(i.section_idx));
      if (sections.length) {
        rows.push(h('div', { class: 'muted sum-ideas-scope' },
          `From the parts you covered: ${covered.size} of ${sections.length} sections.`));
      }

      const anyExtractive = ideas.some((i) => i.source === 'extractive');
      const mixed = anyExtractive && ideas.some((i) => i.source !== 'extractive');
      if (anyExtractive) {
        rows.push(
          h(
            'div',
            { class: 'notice sum-ideas-note' },
            'Extractive: these are sentences taken from the material, not written by AI. Add an API key for AI key ideas.',
          ),
        );
      }
      if (status === 'failed') {
        const why = session.key_ideas_error ? `: ${session.key_ideas_error}` : '.';
        rows.push(h('div', { class: 'notice notice-warn' }, `Key ideas could not be written${why}`));
      }

      const groups = new Map();
      for (const idea of ideas) {
        if (!groups.has(idea.section_idx)) groups.set(idea.section_idx, []);
        groups.get(idea.section_idx).push(idea);
      }
      for (const idx of [...groups.keys()].sort((a, b) => a - b)) {
        const sec = sections.find((x) => x.idx === idx);
        const label =
          session.mode === 'video' && sec && sec.start_s != null
            ? `${fmtClock(sec.start_s)}–${fmtClock(sec.end_s)}`
            : `Section ${idx + 1}`;
        const group = h('div', { class: 'sum-idea-group' }, h('div', { class: 'sum-idea-label muted' }, label));
        for (const idea of groups.get(idx)) {
          const row = h('div', { class: 'sum-idea' }, h('div', { class: 'sum-idea-text' }, idea.text || ''));
          const tags = h('div', { class: 'sum-idea-tags' });
          if (idea.keyword) tags.append(h('span', { class: 'chip' }, idea.keyword));
          if (mixed && idea.source === 'extractive') tags.append(h('span', { class: 'sum-tag muted' }, 'Extractive'));
          row.append(tags);
          group.append(row);
        }
        rows.push(group);
      }

      if (status === 'pending' || status === 'partial') {
        rows.push(h('div', { class: 'sum-writing muted' }, h('span', { class: 'spinner' }), ' writing…'));
      } else if (!ideas.length && status !== 'failed') {
        const noCaptions = session.mode === 'video' && session.content && session.content.transcript_available === false;
        rows.push(
          h('div', { class: 'muted' }, noCaptions ? 'No captions, so there are no key ideas for this video.'
            : all.length ? 'No section was finished in this session, so there are no key ideas to show.' : 'No key ideas for this session.'),
        );
      }
      ideasBody.replaceChildren(...rows);
    }

    function onIdeas(msg) {
      if (!session || !msg || msg.session_id !== session.id) return;
      const have = session.key_ideas || (session.key_ideas = []);
      for (const idea of msg.ideas || []) {
        if (!have.some((x) => x.id === idea.id)) have.push({ ...idea, section_idx: idea.section_idx ?? msg.section_idx });
      }
      if (msg.status) session.key_ideas_status = msg.status;
      if (msg.error) session.key_ideas_error = msg.error;
      renderIdeas();
    }

    // ---- missed list ------------------------------------------------------------------------------
    function metaLine(item, text) {
      const line = h('div', { class: 'sum-missed-meta muted' }, text);
      const badge = sourceBadge(item.source);
      if (badge) line.append(' ', badge);
      return line;
    }

    function readingItem(item, n) {
      const body = h('div', { class: 'sum-missed-body' });
      if (item.excerpt) body.append(h('blockquote', { class: 'sum-quote' }, `“${item.excerpt}”`));
      else body.append(h('div', { class: 'muted' }, 'The sentence was not recorded for this lapse.'));
      const clench = item.kind === 'clench';
      // A clench lapse is the reader's own mark; its place is the gaze 10 s before the clench, not a measurement.
      const bits = clench ? [item.label, 'you marked this with a jaw clench']
        : [item.label, lapseLength(item.duration_s, item.source, endMethodOf(item))];
      body.append(metaLine(item, bits.filter(Boolean).join(' · ')));
      const method = item.position && item.position.method;
      if (method === 'fallback_top') {
        body.append(h('div', { class: 'sum-missed-note muted' }, 'Gaze unavailable: this is the top of the page, not where you were looking.'));
      } else if (method === 'mouse') {
        body.append(h('div', { class: 'sum-missed-note muted' }, 'Position taken from the mouse (testing), not from gaze.'));
      }
      return h('li', { class: 'sum-missed-item' }, h('div', { class: 'sum-num' }, String(n)), body);
    }

    function shotBox(item, range) {
      const videoId = session.content.video_id;
      const box = h('button', { class: 'sum-shot', type: 'button', 'aria-label': `Replay ${range}`, title: `Replay ${range}` });
      const caption = h('span', { class: 'sum-shot-caption' }, `Replay ${range}`);
      const img = h('img', { class: 'sum-shot-img', alt: '' });

      const useThumbnail = () => {
        caption.textContent = `Replay ${range}`;
        img.onerror = () => img.remove();
        img.src = `https://img.youtube.com/vi/${videoId}/mqdefault.jpg`;
      };
      if (item.screenshot_url) {
        caption.textContent = `Screenshot · ${range}`;
        img.onerror = useThumbnail;
        img.src = item.screenshot_url;
      } else {
        useThumbnail();
      }
      box.append(img, caption);
      return on(box, 'click', () => openReplay(item, range));
    }

    function videoItem(item, n) {
      const pos = item.position || {};
      const hasSpan = pos.t_start != null && pos.t_end != null;
      const range = hasSpan ? `${fmtClock(pos.t_start)}-${fmtClock(pos.t_end)}` : item.label || '';
      const body = h('div', { class: 'sum-missed-body' });
      const clench = item.kind === 'clench';
      const length = clench ? null : lapseLength(item.duration_s, item.source, endMethodOf(item));
      // A clench lapse's span is the viewer's own choice (from the concept they picked to the clench).
      const bits = [range, length ? `${length} away` : null, clench ? 'you marked this with a jaw clench' : null].filter(Boolean);
      body.append(metaLine(item, bits.join(' · ')));
      if (clench && item.chosen_concept) {
        body.append(h('div', { class: 'sum-missed-note muted' }, `Last thing you remembered: “${item.chosen_concept}”`));
      }

      if (item.summary_source === 'claude' && item.summary) {
        body.append(
          h('p', { class: 'sum-missed-text' }, item.summary),
          h('div', { class: 'sum-tag muted' }, 'AI summary of the transcript for this part'),
        );
      } else if (item.summary_source !== 'none' && item.excerpt) {
        body.append(h('div', { class: 'sum-tag muted' }, 'Transcript excerpt'), h('p', { class: 'sum-missed-text sum-excerpt' }, item.excerpt));
      } else {
        body.append(h('p', { class: 'sum-missed-text muted' }, 'No captions for this part'));
      }

      const row = h('li', { class: 'sum-missed-item' }, h('div', { class: 'sum-num' }, String(n)), body);
      if (hasSpan && session.content && session.content.video_id) row.append(shotBox(item, range));
      return row;
    }

    function missedSection() {
      const missed = session.summary.missed || [];
      const isVideo = session.mode === 'video';
      const title = isVideo ? "Here's What You Missed" : 'Where you lost focus';
      if (!missed.length) {
        // When nothing was measured, "No lapses caught." would read as a clean result: say why instead.
        const why = session.summary.focus_pct == null ? measureNote() : null;
        const empty = h('div', { class: 'sum-empty' }, h('div', {}, why || 'No lapses caught.'));
        if (state.status && state.status.source === 'simulated') {
          empty.append(
            h(
              'div',
              { class: 'muted' },
              'No headband is connected, so lapses can only be simulated: during a session press D for a drift and R to resume, or Z for a jaw clench.',
            ),
          );
        }
        return section(title, empty);
      }
      const list = h('ol', { class: 'sum-missed' });
      missed.forEach((item, i) => list.append(isVideo ? videoItem(item, i + 1) : readingItem(item, i + 1)));
      return section(title, list);
    }

    // ---- replay modal (video) ---------------------------------------------------------------------
    function openReplay(item, range) {
      if (closeModal) closeModal();
      const from = Math.max(0, item.position.t_start - REPLAY_LEAD_S);
      const to = Math.max(item.position.t_end, from + 1);

      const host = h('div', { class: 'sum-modal-player' });
      const status = h('div', { class: 'sum-modal-status muted' }, `Plays from ${fmtClock(from)} and stops at ${fmtClock(to)}.`);
      const again = h('button', { class: 'btn', type: 'button' }, 'Play again');
      again.hidden = true;
      const close = h('button', { class: 'btn btn-ghost', type: 'button' }, 'Close');
      const panel = h(
        'div',
        { class: 'sum-modal-panel card', role: 'dialog', 'aria-modal': 'true', 'aria-label': `Replay ${range}` },
        h('div', { class: 'sum-modal-head' }, h('div', { class: 'h3' }, `Replay ${range}`), close),
        h('div', { class: 'sum-modal-stage' }, host),
        h('div', { class: 'sum-modal-foot' }, status, again),
      );
      const modal = h('div', { class: 'sum-modal' }, panel);
      (document.getElementById('overlay-root') || document.body).append(modal);

      const player = loadPlayer(host, session.content.video_id, { start: from, autoplay: true });
      // Ignore the stop point until playback is seen before it (a seek reports the old time briefly).
      let armed = false;
      const run = () => {
        armed = false;
        again.hidden = true;
        status.textContent = `Plays from ${fmtClock(from)} and stops at ${fmtClock(to)}.`;
        player.seek(from);
        player.play();
      };
      player.onTick((t) => {
        if (!armed) {
          armed = t < to;
          return;
        }
        if (t >= to) {
          armed = false;
          player.pause();
          status.textContent = 'Replay finished.';
          again.hidden = false;
        }
      });
      player.onError(({ message }) => {
        status.textContent = message;
        status.classList.add('sum-modal-error');
      });
      player.ready.then((ok) => {
        if (ok) run();
      });

      const onKey = (e) => {
        if (e.key === 'Escape') shut();
      };
      function shut() {
        window.removeEventListener('keydown', onKey);
        player.destroy();
        modal.remove();
        if (closeModal === shut) closeModal = null;
      }
      window.addEventListener('keydown', onKey);
      on(close, 'click', shut);
      on(again, 'click', run);
      on(modal, 'click', (e) => {
        if (e.target === modal) shut();
      });
      closeModal = shut;
      close.focus();
    }

    // ---- questions (video) ------------------------------------------------------------------------
    function questionsSection() {
      const input = h('input', {
        class: 'input sum-ask-input',
        type: 'text',
        placeholder: 'TYPE HERE',
        'aria-label': 'Ask a question about this session',
        autocomplete: 'off',
        maxlength: '500',
      });
      const send = h('button', { class: 'btn', type: 'submit' }, 'Ask');
      const form = h('form', { class: 'sum-ask' }, input, send);
      const error = h('div', { class: 'notice notice-warn sum-ask-error' }, '');
      error.hidden = true;
      const waiting = h('div', { class: 'sum-writing muted' }, h('span', { class: 'spinner' }), ' thinking…');
      waiting.hidden = true;
      const answers = h('div', { class: 'sum-answers' });

      on(form, 'submit', async (e) => {
        e.preventDefault();
        const question = input.value.trim();
        if (!question || input.disabled) return;
        input.disabled = true;
        send.disabled = true;
        error.hidden = true;
        waiting.hidden = false;
        try {
          const res = await api.post(`/api/sessions/${session.id}/ask`, { question });
          if (dead) return;
          const by = res.source === 'claude' ? 'Answered by Claude from this session’s material' : `Source: ${res.source}`;
          answers.prepend(
            h(
              'div',
              { class: 'sum-qa' },
              h('div', { class: 'sum-q' }, question),
              h('div', { class: 'sum-a' }, res.answer || ''),
              h('div', { class: 'sum-tag muted' }, by),
            ),
          );
          input.value = '';
        } catch (err) {
          if (dead) return;
          error.textContent = err.message || 'The question could not be answered.';
          error.hidden = false;
        } finally {
          input.disabled = false;
          send.disabled = false;
          waiting.hidden = true;
          if (!dead) input.focus();
        }
      });
      return section('Questions?', form, error, waiting, answers);
    }

    // ---- actions ----------------------------------------------------------------------------------
    function actions() {
      // Part of a study plan with resources still to go: the main button moves on to the next one.
      const study = session.study;
      const nextIdx = study && Number.isInteger(study.order) ? study.order + 1 : null;
      if (study && nextIdx < study.count && Array.isArray(study.session_ids)) {
        const nextTitle = (study.titles && study.titles[nextIdx]) || 'the next resource';
        const nextBtn = h('button', { class: 'btn btn-primary btn-lg sum-review', type: 'button' },
          `Next (${nextIdx + 1} of ${study.count}): ${nextTitle}`);
        const err = h('div', { class: 'notice notice-warn' }, '');
        err.hidden = true;
        on(nextBtn, 'click', async () => {
          nextBtn.disabled = true;
          try {
            const next = await api.get(`/api/sessions/${study.session_ids[nextIdx]}`);
            if (dead) return;
            state.session = next;
            navigate(next.mode, { id: next.id, session: next, ...(state.studyParams || { gaze: 'webgazer' }) });
          } catch (e) {
            if (dead) return;
            nextBtn.disabled = false;
            err.textContent = `The next resource could not be opened: ${e.message}`;
            err.hidden = false;
          }
        });
        const homeBtn = on(h('button', { class: 'btn btn-ghost', type: 'button' }, 'Back to home'), 'click', () => navigate('home'));
        return h('div', { class: 'sum-actions-wrap' }, err, h('div', { class: 'sum-actions' }, nextBtn, homeBtn));
      }
      const many = study && study.count > 1;
      const label = h('span', {}, many ? `Flashcards from all ${study.count} resources` : 'Review flashcards');
      const spin = h('span', { class: 'spinner' });
      spin.hidden = true;
      const go = h('button', { class: 'btn btn-primary btn-lg sum-review', type: 'button' }, spin, label);
      const home = on(h('button', { class: 'btn btn-ghost', type: 'button' }, 'Back to home'), 'click', () => navigate('home'));
      const error = h('div', { class: 'notice notice-warn' }, '');
      error.hidden = true;

      on(go, 'click', async () => {
        if (go.disabled) return;
        if (session.flashcards) {
          navigate('flashcards', { id: session.id });
          return;
        }
        go.disabled = true;
        spin.hidden = false;
        error.hidden = true;
        label.textContent = 'Making flashcards and audio cues…';
        try {
          session = await api.post(`/api/sessions/${session.id}/flashcards`, {});
          if (!dead) navigate('flashcards', { id: session.id });
        } catch (err) {
          if (dead) return;
          go.disabled = false;
          spin.hidden = true;
          label.textContent = many ? `Flashcards from all ${study.count} resources` : 'Review flashcards';
          error.textContent = `Flashcards could not be made: ${err.message}`;
          error.hidden = false;
        }
      });
      return h('div', { class: 'sum-actions-wrap' }, error, h('div', { class: 'sum-actions' }, go, home));
    }

    function render() {
      const main = h('div', { class: 'sum-main' }, missedSection());
      if (session.mode === 'video') main.append(questionsSection());
      // Video: no key-ideas panel here (Ayur, 2026-11-15); the ideas are still made and used for the flashcards.
      let body = main;
      if (session.mode !== 'video') {
        body = h('div', { class: 'sum-columns' }, main, h('div', { class: 'sum-side' }, section('Key ideas', ideasBody)));
        renderIdeas();
      }
      const top = h('div', { class: 'sum-top' }, tiles());
      if (showsEstimates()) top.append(h('div', { class: 'sum-estimate muted' }, ESTIMATE_NOTE));
      const parts = [header(), top, timelineSection(), body, actions()];
      page.replaceChildren(...parts.filter(Boolean));
    }

    async function init() {
      const id = params.id || (state.session && state.session.id);
      if (!id) {
        fail('No session was chosen.');
        return;
      }
      try {
        session = await api.get(`/api/sessions/${id}`);
      } catch (err) {
        if (!dead) fail(`This session could not be opened: ${err.message}`);
        return;
      }
      if (dead) return;
      if (!session.summary) {
        fail('This session was not finished, so there are no insights for it.');
        return;
      }
      offs.push(bus.on('ws:key_ideas', onIdeas));
      render();
    }

    init().catch((err) => {
      if (!dead) fail(err.message);
    });

    return () => {
      dead = true;
      if (closeModal) closeModal();
      for (const off of offs) {
        if (typeof off === 'function') off();
      }
    };
  },
};
