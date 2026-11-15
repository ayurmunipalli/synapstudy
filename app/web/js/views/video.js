// Video aid: just the player. Lapses are logged silently and reviewed at the end (summary view).
import { api } from '../api.js';
import { bus } from '../bus.js';
import { state, tracker } from '../state.js';
import { ws } from '../ws.js';
import { navigate } from '../router.js';
import { h, fmtClock } from '../lib/dom.js';
import { loadPlayer } from '../lib/youtube.js';
import { capture } from '../lib/capture.js';

const VIDEO_SYNC_MS = 1000;
// The detector declares a drift a little after it began; never back-date the video time by more than this.
const MAX_ONSET_LAG_S = 10;

const on = (el, type, fn) => {
  el.addEventListener(type, fn);
  return el;
};
const round1 = (x) => Math.round(x * 10) / 10;

export const view = {
  mount(root, params = {}) {
    const offs = [];
    let dead = false;
    let session = null;
    let player = null;
    let started = false;
    let finishing = false;
    let finished = false;
    let startedMs = 0;
    let clockTimer = null;
    let lastSync = 0;
    let lapseStart = null;
    let shotsNoted = false;
    // Jaw-clench "I zoned out" (see state.js): concept timeline for the popup, and a video-time -> wall-clock
    // history so the lapse's start (the end of the concept picked) gets a real wall time even after seeks.
    let concepts = [];
    let conceptsSource = 'extractive';
    let conceptsStatus = 'done';
    const timeHistory = [];            // [{t: video s, wall: ms}] every tick while playing, last ~20 min
    let clenchPop = null;              // the open popup, or null

    const sections = [];
    const passed = new Set();
    const ideasBySection = new Map();
    let ideasStatus = 'pending';
    let ideasError = null;

    const wantShots = Boolean(params.screenshots ?? params.capture ?? params.screenshot ?? params.shots ?? false);

    // ---- skeleton -------------------------------------------------------------------------------
    const playerHost = h('div', { class: 'vid-player' });
    const gate = h('div', { class: 'vid-gate' }, h('div', { class: 'spinner' }));
    const stage = h('div', { class: 'vid-stage' }, playerHost, gate);
    const titleEl = h('div', { class: 'vid-title h2' }, '');
    const subEl = h('div', { class: 'vid-sub muted' }, '');
    const ideasChip = on(h('button', { class: 'chip vid-ideas-chip', type: 'button' }, 'Key ideas: 0'), 'click', () =>
      setDrawer(drawer.hidden),
    );
    ideasChip.hidden = true;
    const meta = h('div', { class: 'vid-meta' }, h('div', { class: 'vid-meta-text' }, titleEl, subEl), ideasChip);
    const notes = h('div', { class: 'vid-notes' });

    const drawerList = h('div', { class: 'vid-drawer-list' });
    const drawerClose = on(
      h('button', { class: 'btn btn-ghost vid-drawer-close', type: 'button', 'aria-label': 'Close key ideas' }, 'Close'),
      'click',
      () => setDrawer(false),
    );
    const drawer = h(
      'aside',
      { class: 'vid-drawer panel', 'aria-label': 'Key ideas' },
      h('div', { class: 'vid-drawer-head' }, h('div', { class: 'h3' }, 'Key ideas so far'), drawerClose),
      drawerList,
    );
    drawer.hidden = true;

    const page = h('div', { class: 'vid-page' }, stage, meta, notes, drawer);
    root.replaceChildren(page);

    const clockEl = h('span', { class: 'vid-clock' }, '0:00');
    const finishBtn = on(h('button', { class: 'btn btn-primary', type: 'button' }, 'Finish'), 'click', () => finish());

    // Simulated lapses by button. A click inside the YouTube player moves keyboard focus into its iframe,
    // and the D / R keys then never reach this page; these buttons work regardless, and a click on one
    // brings focus back to the page. Each is blurred after its click so focus rests on the page body.
    const driftBtn = h('button', { class: 'btn vid-sim-btn', type: 'button' }, 'Lose focus (D)');
    const resumeBtn = h('button', { class: 'btn vid-sim-btn', type: 'button' }, 'Resume (R)');
    const clenchBtn = h('button', { class: 'btn vid-sim-btn', type: 'button' }, 'Clench (Z)');
    on(clenchBtn, 'click', () => {
      ws.send({ type: 'sim', action: 'clench' });
      clenchBtn.blur();
    });
    on(driftBtn, 'click', () => {
      // The server ignores a drift while it still holds an earlier simulated lapse open; clear that first.
      if (!tracker.active()) ws.send({ type: 'sim', action: 'resume' });
      ws.send({ type: 'sim', action: 'drift' });
      driftBtn.blur();
    });
    on(resumeBtn, 'click', () => {
      ws.send({ type: 'sim', action: 'resume' });
      resumeBtn.blur();
    });

    // Shown only while the server says the source is simulated (it can change during a session).
    function syncSimButtons() {
      const show = started && !finishing && !finished && !dead && Boolean(state.status) && state.status.source === 'simulated';
      if (show && !driftBtn.isConnected && finishBtn.isConnected) finishBtn.before(driftBtn, resumeBtn, clenchBtn);
      else if (!show) {
        driftBtn.remove();
        resumeBtn.remove();
        clenchBtn.remove();
      }
    }

    function setDrawer(open) {
      drawer.hidden = !open;
      ideasChip.setAttribute('aria-expanded', String(open));
    }

    function note(text, warn = false) {
      notes.append(h('div', { class: warn ? 'notice notice-warn vid-note' : 'notice vid-note' }, text));
    }

    function shotsOff(reason) {
      capture.stop();
      if (shotsNoted) return;
      shotsNoted = true;
      note(`Screenshots are off: ${reason}. The session continues without them.`);
    }

    function gateMessage(title, lines, button) {
      const inner = h('div', { class: 'vid-gate-inner' }, h('div', { class: 'h2' }, title));
      for (const line of lines) inner.append(line);
      if (button) inner.append(button);
      gate.replaceChildren(inner);
      gate.hidden = false;
    }

    function homeButton() {
      return on(h('button', { class: 'btn', type: 'button' }, 'Back to home'), 'click', () => navigate('home'));
    }

    // ---- key ideas ------------------------------------------------------------------------------
    function addIdeas(list) {
      for (const idea of list || []) {
        const idx = idea.section_idx;
        if (idx == null) continue;
        const have = ideasBySection.get(idx) || [];
        if (!have.some((x) => x.id === idea.id)) have.push(idea);
        ideasBySection.set(idx, have);
      }
    }

    function onIdeas(msg) {
      if (!session || !msg) return;
      if (msg.session_id && msg.session_id !== session.id) return;
      if (msg.section_idx != null && Array.isArray(msg.ideas)) {
        addIdeas(msg.ideas.map((idea) => ({ ...idea, section_idx: idea.section_idx ?? msg.section_idx })));
      }
      if (msg.status) ideasStatus = msg.status;
      if (msg.error) ideasError = msg.error;
      renderIdeas();
    }

    function renderIdeas() {
      const shown = sections.filter((s) => passed.has(s.idx));
      const count = shown.reduce((n, s) => n + (ideasBySection.get(s.idx) || []).length, 0);
      ideasChip.textContent = `Key ideas: ${count}`;

      const rows = [];
      const writing = ideasStatus === 'pending' || ideasStatus === 'partial';
      const extractive = shown.some((s) => (ideasBySection.get(s.idx) || []).some((i) => i.source === 'extractive'));
      if (extractive) {
        rows.push(h('div', { class: 'vid-idea-label muted' }, 'Extractive: add an API key for AI key ideas'));
      }
      if (ideasStatus === 'failed') {
        rows.push(
          h('div', { class: 'notice notice-warn' }, `Key ideas could not be written${ideasError ? `: ${ideasError}` : '.'}`),
        );
      }
      if (!sections.length) {
        rows.push(h('div', { class: 'muted' }, 'No captions, so there are no key ideas for this video.'));
      } else if (!shown.length) {
        rows.push(h('div', { class: 'muted' }, 'Key ideas appear here as you watch.'));
      }
      for (const s of shown) {
        const block = h(
          'div',
          { class: 'vid-idea-block' },
          h('div', { class: 'vid-idea-time muted' }, `${fmtClock(s.start_s)}–${fmtClock(s.end_s)}`),
        );
        const ideas = ideasBySection.get(s.idx) || [];
        for (const idea of ideas) {
          const row = h('div', { class: 'vid-idea' }, h('div', { class: 'vid-idea-text' }, idea.text || ''));
          if (idea.keyword) row.append(h('span', { class: 'chip' }, idea.keyword));
          block.append(row);
        }
        if (!ideas.length && writing) {
          block.append(h('div', { class: 'vid-idea-writing muted' }, h('span', { class: 'spinner' }), ' writing…'));
        }
        if (ideas.length || writing) rows.push(block);
      }
      drawerList.replaceChildren(...rows);
    }

    // ---- progress and the client log --------------------------------------------------------------
    function syncVideo() {
      if (!started || !player) return;
      const duration = player.duration();
      tracker.setVideo({ duration_s: duration > 0 ? round1(duration) : null, watched_s: round1(player.watched()) });
    }

    function markPassed(upTo) {
      let changed = false;
      for (const s of sections) {
        if (!passed.has(s.idx) && upTo >= s.end_s) {
          passed.add(s.idx);
          tracker.markProgress(s.idx);
          changed = true;
        }
      }
      if (changed) renderIdeas();
    }

    function onTick(t) {
      if (!started) return;
      if (player && player.state() === 'playing') {
        timeHistory.push({ t, wall: Date.now() });
        if (timeHistory.length > 5000) timeHistory.splice(0, 1000);
      }
      markPassed(t);
      const now = Date.now();
      if (now - lastSync >= VIDEO_SYNC_MS) {
        lastSync = now;
        syncVideo();
      }
    }

    // Called by the tracker at drift and again at resume. Times are positions in the video.
    function getPosition(evt) {
      const now = round1(player ? player.time() : 0);
      const isResume = (evt && evt.type === 'resume') || (!(evt && evt.type === 'drift') && lapseStart != null);
      if (!isResume) {
        let lag = 0;
        if (player && player.state() === 'playing' && evt && evt.onset_wall_ms) {
          lag = Math.min(MAX_ONSET_LAG_S, Math.max(0, (Date.now() - evt.onset_wall_ms) / 1000));
        }
        lapseStart = Math.max(0, round1(now - lag));
        return { t_start: lapseStart, t_end: now };
      }
      const start = lapseStart ?? now;
      lapseStart = null;
      return { t_start: start, t_end: Math.max(start, now) };
    }

    // ---- jaw clench: "I zoned out" -> pick the last thing remembered ------------------------------
    // Wall time at which the video was at position `t`: the newest history entry at or just before t, moved
    // on by the difference; `fallback` (ms) if the history has nothing (e.g. before playback started).
    function wallAt(t, fallback) {
      for (let i = timeHistory.length - 1; i >= 0; i -= 1) {
        const e = timeHistory[i];
        if (e.t <= t + 0.3 && t - e.t < 30) return e.wall + (t - e.t) * 1000;
      }
      return fallback;
    }

    function onConcepts(msg) {
      if (!session || !msg || msg.session_id !== session.id || !msg.concepts) return;
      concepts = Array.isArray(msg.concepts.items) ? msg.concepts.items : concepts;
      conceptsSource = msg.concepts.source || conceptsSource;
      conceptsStatus = msg.concepts.status || conceptsStatus;
    }

    function closeClenchPop(resume) {
      if (!clenchPop) return;
      clenchPop.remove();
      clenchPop = null;
      tracker.hold(false);
      if (resume && player && started && !finishing && !finished && !dead) player.play();
    }

    function onClench(trig) {
      if (!started || finishing || finished || dead || clenchPop || !player || !trig) return;
      const wasPlaying = player.state() === 'playing';
      player.pause();
      tracker.hold(true);              // no look-away lapses while the popup is up (the video is paused)
      const now = round1(player.time());
      // Where the hold began: the clench was held ~3 s while the video ran on.
      const heldS = Math.max(0, Math.min(10, (trig.wall_ms - trig.onset_wall_ms) / 1000));
      const holdT = Math.max(0, round1(wasPlaying ? now - heldS : now));
      const holdWall = trig.onset_wall_ms;

      const choose = (tStart, label) => {
        const t0 = Math.max(0, Math.min(round1(tStart), holdT));
        tracker.addLapse({
          source: trig.source,
          onset_ms: wallAt(t0, holdWall - (holdT - t0) * 1000),
          resume_ms: holdWall,
          position: { t_start: t0, t_end: holdT },
          chosen_concept: label,
        });
        closeClenchPop(true);
      };

      const listed = concepts.filter((c) => c.start_s <= holdT).slice(-5).reverse();   // newest first
      const opts = [];
      for (const c of listed) {
        opts.push(on(h('button', { class: 'clench-opt', type: 'button' },
          h('span', { class: 'clench-opt-label' }, c.label),
          h('span', { class: 'clench-opt-time muted' }, `${fmtClock(c.start_s)}–${fmtClock(Math.min(c.end_s, holdT))}`)),
        'click', () => choose(Math.min(c.end_s, holdT), c.label)));
      }
      if (listed.length) {
        const oldest = listed[listed.length - 1];
        opts.push(on(h('button', { class: 'clench-opt clench-opt-earlier', type: 'button' },
          h('span', { class: 'clench-opt-label' }, 'Earlier than these'),
          h('span', { class: 'clench-opt-time muted' }, `from ${fmtClock(oldest.start_s)}`)),
        'click', () => choose(oldest.start_s, 'Earlier than the listed concepts')));
      } else {
        // No captions (or none yet): offer times instead, and say why.
        for (const back of [30, 60, 120]) {
          opts.push(on(h('button', { class: 'clench-opt', type: 'button' },
            h('span', { class: 'clench-opt-label' }, back < 60 ? `About ${back} s ago` : `About ${back / 60} min ago`)),
          'click', () => choose(holdT - back, `About ${back} s before the clench`)));
        }
      }
      const dismiss = on(h('button', { class: 'btn btn-ghost', type: 'button' }, "Not a zone-out, keep watching"),
        'click', () => {
          tracker.noteClenchDismissed();
          closeClenchPop(true);
        });
      const sourceNote = listed.length
        ? (conceptsSource === 'claude' ? 'Concepts written by Claude from the captions.'
          : 'Concepts are the captions\' own words (extractive): add an API key for AI labels.')
        : 'No caption concepts for this part, so pick roughly when you lost track.';
      clenchPop = h('div', { class: 'clench-pop', role: 'dialog', 'aria-modal': 'true', 'aria-label': 'Where did you lose track?' },
        h('div', { class: 'clench-pop-card panel' },
          h('div', { class: 'h2' }, 'What\'s the last thing you remember?'),
          h('div', { class: 'muted clench-pop-sub' },
            trig.source === 'simulated' ? 'SIMULATED clench. ' : 'You held a jaw clench. ',
            'The video is paused; the part after your choice goes into "Here\'s What You Missed".'),
          h('div', { class: 'clench-opts' }, ...opts),
          h('div', { class: 'muted clench-pop-note' }, sourceNote),
          h('div', { class: 'clench-pop-actions' }, dismiss)));
      (document.getElementById('overlay-root') || document.body).append(clenchPop);
      (opts[0] || dismiss).focus();
    }

    async function shoot(lapse) {
      if (!lapse || !capture.active()) return;
      try {
        const dataUrl = await capture.grab(stage);
        if (!dataUrl) throw new Error('no frame was available');
        const res = await api.post(`/api/sessions/${session.id}/screenshots`, {
          lapse_id: lapse.id,
          data_url: dataUrl,
        });
        if (!dead && res && res.url) tracker.attachScreenshot(lapse.id, res.url);
      } catch (err) {
        if (!dead) shotsOff(`a screenshot could not be saved (${err.message})`);
      }
    }

    // ---- start / finish ---------------------------------------------------------------------------
    async function start(button) {
      if (started || dead) return;
      button.disabled = true;
      // getDisplayMedia needs the click's user activation, so it is the first thing awaited.
      if (wantShots) {
        const res = await capture.request();
        if (dead) {
          capture.stop();
          return;
        }
        if (res.ok) offs.push(capture.onEnded(() => shotsOff('sharing was stopped')));
        else shotsOff(res.reason);
      }
      try {
        tracker.begin(session, { getPosition });
      } catch (err) {
        button.disabled = false;
        capture.stop();
        note(`The session could not start: ${err.message}`, true);
        return;
      }
      started = true;
      startedMs = Date.now();
      gate.hidden = true;
      gate.replaceChildren();
      ideasChip.hidden = false;

      const slot = document.getElementById('topbar-session');
      if (slot) slot.append(clockEl, finishBtn);
      syncSimButtons();
      clockTimer = setInterval(() => {
        clockEl.textContent = fmtClock((Date.now() - startedMs) / 1000);
      }, 1000);

      player.play();
    }

    async function finish() {
      if (!started || finishing || finished) return;
      closeClenchPop(false);           // a clench left unanswered at Finish is not logged
      finishing = true;
      syncSimButtons();
      finishBtn.disabled = true;
      finishBtn.textContent = 'Finishing…';
      player.pause();
      syncVideo();
      capture.stop();
      try {
        const done = await tracker.finish();
        finished = true;
        if (!dead) navigate('summary', { id: (done && done.id) || session.id });
      } catch (err) {
        finishing = false;
        if (dead) return;
        finishBtn.disabled = false;
        finishBtn.textContent = 'Finish';
        syncSimButtons();
        note(`The session could not be finished: ${err.message}`, true);
      }
    }

    function showReady() {
      const lines = [];
      if (!session.content.transcript_available) {
        const reason = session.content.transcript_error || 'No captions were found for this video.';
        lines.push(h('div', { class: 'notice notice-warn' }, `${reason} Summaries and key ideas need captions.`));
      }
      if (wantShots) {
        lines.push(
          h(
            'div',
            { class: 'muted vid-gate-hint' },
            'When you start, the browser asks to share this tab so one frame can be saved at each lapse.',
          ),
        );
      }
      const source = state.status && state.status.source;
      let hint = 'No webcam in this view: lapses come from the headband alone. Zoned out? Clench your jaw and hold it for 3 s to pick where you lost track.';
      if (source === 'simulated') {
        hint = 'No headband connected: press D to simulate a lapse and R to resume, Z for a jaw clench ("I zoned out"), or use the buttons shown in the top bar during the session.';
      } else if (source === 'replay') {
        hint = 'No webcam in this view: lapses come from the replayed recording, not from you.';
      }
      lines.push(h('div', { class: 'muted vid-gate-hint' }, hint));
      const button = h('button', { class: 'btn btn-primary btn-lg', type: 'button' }, 'Start video');
      on(button, 'click', () => start(button));
      gateMessage(session.title || 'Video', lines, button);
      button.focus();
    }

    async function init() {
      const hint =
        params.session || (state.session && (!params.id || state.session.id === params.id) ? state.session : null);
      const id = params.id || (hint && hint.id);
      if (!id) {
        gateMessage('No video session', [h('div', { class: 'muted' }, 'Start a study session from Home.')], homeButton());
        return;
      }
      try {
        session = await api.get(`/api/sessions/${id}`);
      } catch (err) {
        if (dead) return;
        if (!hint) {
          gateMessage('This session could not be opened', [h('div', { class: 'notice notice-warn' }, err.message)], homeButton());
          return;
        }
        session = hint;
      }
      if (dead) return;
      if (session.ended_at) {
        navigate('summary', { id: session.id });
        return;
      }
      const content = session.content || {};
      if (session.mode !== 'video' || !content.video_id) {
        gateMessage('Not a video session', [h('div', { class: 'muted' }, 'This session has no video to play.')], homeButton());
        return;
      }

      titleEl.textContent = session.title || 'Untitled video';
      subEl.textContent = session.subject || '';
      sections.push(...(content.sections || []));
      if (content.concepts) {
        concepts = Array.isArray(content.concepts.items) ? content.concepts.items : [];
        conceptsSource = content.concepts.source || 'extractive';
        conceptsStatus = content.concepts.status || 'done';
      }
      addIdeas(session.key_ideas);
      ideasStatus = session.key_ideas_status || 'pending';
      ideasError = session.key_ideas_error || null;
      renderIdeas();

      player = loadPlayer(playerHost, content.video_id);
      offs.push(
        player.onError(({ message }) => {
          if (dead) return;
          if (!started) {
            gateMessage(
              "This video can't be played here",
              [h('div', { class: 'notice notice-warn' }, message)],
              homeButton(),
            );
          } else {
            note(`${message} Press Finish to see what was recorded.`, true);
          }
        }),
        player.onTick(onTick),
        player.onState((name) => {
          if (dead || !started) return;
          if (name === 'ended') {
            const duration = player.duration();
            for (const s of sections) {
              if (!passed.has(s.idx) && s.start_s < duration) {
                passed.add(s.idx);
                tracker.markProgress(s.idx);
              }
            }
            renderIdeas();
            finish();
          }
        }),
      );
      showReady();
    }

    offs.push(
      bus.on('lapse:start', ({ lapse } = {}) => {
        if (!started) return;
        stage.classList.add('vid-lapse');
        shoot(lapse);
      }),
      bus.on('lapse:end', () => stage.classList.remove('vid-lapse')),
      bus.on('ws:key_ideas', onIdeas),
      bus.on('ws:concepts', onConcepts),
      bus.on('clench:trigger', onClench),
      bus.on('status', syncSimButtons),
    );
    const onKey = (e) => {
      if (e.key === 'Escape' && clenchPop) {
        tracker.noteClenchDismissed();
        closeClenchPop(true);
        return;
      }
      if (e.key === 'Escape' && !drawer.hidden) setDrawer(false);
    };
    window.addEventListener('keydown', onKey);

    init().catch((err) => {
      if (!dead) gateMessage('Something went wrong', [h('div', { class: 'notice notice-warn' }, err.message)], homeButton());
    });

    return () => {
      dead = true;
      window.removeEventListener('keydown', onKey);
      if (clockTimer != null) clearInterval(clockTimer);
      // Leaving the view ends the session with what was actually watched (same rule as the reading view).
      if (started && !finished && !finishing) {
        syncVideo();
        tracker.finish().catch((err) => console.warn('could not close the session:', err.message));
      }
      for (const off of offs) {
        if (typeof off === 'function') off();
      }
      capture.stop();
      if (player) player.destroy();
      clockEl.remove();
      finishBtn.remove();
      driftBtn.remove();
      resumeBtn.remove();
      clenchBtn.remove();
      if (clenchPop) {
        clenchPop.remove();
        clenchPop = null;
      }
    };
  },
};
