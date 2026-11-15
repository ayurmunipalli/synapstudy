// Flashcards with one audio cue per concept, then "Tonight's cues" (the exported cue set).
import { api } from '../api.js';
import { state } from '../state.js';
import { navigate } from '../router.js';
import { h } from '../lib/dom.js';
import { audio } from '../lib/audio.js';

const REVIEW_VOLUME = 0.8;
const DEFAULT_SLEEP_VOLUME = 0.15;

const on = (el, type, fn) => {
  el.addEventListener(type, fn);
  return el;
};

function speakerIcon() {
  const NS = 'http://www.w3.org/2000/svg';
  const icon = document.createElementNS(NS, 'svg');
  icon.setAttribute('viewBox', '0 0 20 20');
  icon.setAttribute('class', 'fc-icon');
  icon.setAttribute('aria-hidden', 'true');
  const path = document.createElementNS(NS, 'path');
  path.setAttribute('d', 'M3 7.5h3l4-3.5v12l-4-3.5H3z M13 7.2a4 4 0 0 1 0 5.6 M15.2 4.8a7.4 7.4 0 0 1 0 10.4');
  icon.append(path);
  return icon;
}

function isTyping(target) {
  if (!target || !target.tagName) return false;
  return ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName) || target.isContentEditable;
}

export const view = {
  mount(root, params = {}) {
    let dead = false;
    let session = null;
    let cards = [];
    let mode = 'loading'; // loading | deck | tonight
    let queue = [];
    let cleared = 0;
    let flipped = false;
    let playToken = 0;
    let deckKeys = null; // { flip, again, gotIt, cardEl } while a card is shown

    const page = h('div', { class: 'fc-page' });
    root.replaceChildren(page);

    function loading(text) {
      page.replaceChildren(h('div', { class: 'fc-loading' }, h('div', { class: 'spinner' }), h('div', { class: 'muted' }, text)));
    }

    function navButtons() {
      const back = on(h('button', { class: 'btn', type: 'button' }, 'Back to insights'), 'click', () =>
        navigate('summary', { id: session ? session.id : params.id }),
      );
      const done = on(h('button', { class: 'btn btn-primary', type: 'button' }, 'Done'), 'click', () => navigate('home'));
      return { back, done };
    }

    // The session line under each heading; "You missed this" tags come from lapses, so say when those were
    // simulated, replayed or not measured. The server's measure_note is shown as it is; sessions saved
    // before it existed keep the old line.
    function sessionLine() {
      // A study plan's deck covers all its resources: show the subject and how many resources, not one title.
      const study = session.study;
      const heading = study && study.count > 1 ? `${study.subject} · all ${study.count} resources` : (session.title || '');
      const line = h('div', { class: 'fc-session muted' }, heading);
      const note = typeof session.measure_note === 'string' && session.measure_note ? session.measure_note : null;
      let badge = null;
      if (note && /replayed recording/.test(note)) badge = h('span', { class: 'badge badge-replay' }, 'REPLAY');
      else if (session.simulated) badge = h('span', { class: 'badge badge-sim' }, 'SIMULATED');
      else if (session.measured === false) badge = h('span', { class: 'badge badge-unknown' }, 'NOT MEASURED');
      if (badge) line.append(badge);
      if (note) line.append(note);
      else if (session.simulated) line.append('This session used simulated lapses');
      return line;
    }

    function fail(message) {
      mode = 'loading';
      const home = on(h('button', { class: 'btn', type: 'button' }, 'Back to home'), 'click', () => navigate('home'));
      page.replaceChildren(h('div', { class: 'notice notice-warn' }, message), h('div', { class: 'fc-actions' }, home));
    }

    // Plays a card's cue. If the browser refuses (no user gesture yet) onBlocked is called instead of failing.
    function playCue(card, volume, onBlocked) {
      const token = ++playToken;
      if (!card || !card.cue || !card.cue.url) return;
      let playing;
      try {
        playing = audio.play(card.cue.url, { volume });
      } catch (err) {
        playing = Promise.reject(err);
      }
      Promise.resolve(playing).catch((err) => {
        // A cue cut short by the next one is not a failure.
        if (dead || token !== playToken || (err && err.name === 'AbortError')) return;
        if (onBlocked) onBlocked(err);
      });
    }

    function stopAudio() {
      playToken += 1;
      try {
        audio.stop();
      } catch {
        /* nothing playing */
      }
    }

    function cueLabel(card) {
      if (!card.cue) return 'No audio cue for this card';
      return card.cue.kind === 'tone' ? 'Cue: a tone (speech was not available)' : `Cue: “${card.cue.text || card.concept || ''}”`;
    }

    // ---- deck -------------------------------------------------------------------------------------
    function startDeck() {
      mode = 'deck';
      queue = cards.map((_, i) => i);
      cleared = 0;
      showCard();
    }

    function showCard() {
      if (!queue.length) {
        // Deck done: on to the night screen (Ayur, 2026-11-15), which starts the sleep recording and cues.
        stopAudio();
        navigate('night', { id: session.id });
        return;
      }
      stopAudio();
      const card = cards[queue[0]];
      const total = cards.length;
      flipped = false;

      const bar = h('div', { class: 'fc-bar-fill' });
      bar.style.width = `${(cleared / total) * 100}%`;
      const top = h(
        'div',
        { class: 'fc-top' },
        h('div', { class: 'fc-top-text' }, h('div', { class: 'h2' }, 'Flashcards'), sessionLine()),
        h('div', { class: 'fc-progress' }, h('div', { class: 'fc-count' }, `${Math.min(cleared + 1, total)} / ${total}`), h('div', { class: 'fc-bar' }, bar)),
      );

      const tags = h('div', { class: 'fc-tags' });
      if (card.from_missed) tags.append(h('span', { class: 'chip fc-missed' }, 'You missed this'));
      if (card.source === 'extractive') {
        tags.append(h('span', { class: 'fc-tag muted' }, 'Extractive: a sentence from the material with its keyword blanked, not written by AI'));
      }
      const sideLabel = h('div', { class: 'fc-side muted' }, 'Question');
      const text = h('div', { class: 'fc-text' }, card.front || '');
      const concept = h('div', { class: 'fc-concept' });
      const hint = h('div', { class: 'fc-hint muted' }, 'Click the card or press Space to see the answer');
      const cardEl = h('div', { class: 'fc-card card', role: 'button', tabindex: '0', 'aria-label': 'Flashcard: flip' }, tags, sideLabel, text, concept, hint);

      const replay = h('button', { class: 'btn btn-ghost fc-cue-btn', type: 'button', 'aria-label': 'Play the audio cue again' }, speakerIcon(), 'Play cue');
      const blocked = h('button', { class: 'btn fc-cue-blocked', type: 'button' }, 'Click to play cue');
      blocked.hidden = true;
      const cueRow = h('div', { class: 'fc-cue' }, replay, blocked, h('span', { class: 'muted' }, cueLabel(card)));
      if (!card.cue || !card.cue.url) replay.hidden = true;
      const play = () => {
        blocked.hidden = true;
        playCue(card, REVIEW_VOLUME, () => {
          blocked.hidden = false;
        });
      };
      on(replay, 'click', play);
      on(blocked, 'click', play);

      const reveal = h('button', { class: 'btn btn-primary btn-lg', type: 'button' }, 'Show answer');
      const againBtn = h('button', { class: 'btn btn-lg', type: 'button' }, 'Again ', h('span', { class: 'fc-key' }, '1'));
      const gotBtn = h('button', { class: 'btn btn-primary btn-lg', type: 'button' }, 'Got it ', h('span', { class: 'fc-key' }, '2'));
      againBtn.hidden = true;
      gotBtn.hidden = true;
      const controls = h('div', { class: 'fc-controls' }, reveal, againBtn, gotBtn);

      const skip = on(h('button', { class: 'btn btn-ghost', type: 'button' }, 'Skip to tonight'), 'click', () => {
        stopAudio();
        navigate('night', { id: session.id });
      });
      const foot = h('div', { class: 'fc-foot' }, h('span', { class: 'muted' }, 'Space flips · 1 Again · 2 Got it'), skip);

      function flip() {
        flipped = !flipped;
        cardEl.classList.toggle('fc-flipped', flipped);
        sideLabel.textContent = flipped ? 'Answer' : 'Question';
        text.textContent = flipped ? card.back || '' : card.front || '';
        concept.replaceChildren();
        if (flipped && card.concept) concept.append(h('span', { class: 'muted' }, 'Concept '), h('span', { class: 'chip' }, card.concept));
        hint.hidden = flipped;
        reveal.hidden = flipped;
        againBtn.hidden = !flipped;
        gotBtn.hidden = !flipped;
      }
      function again() {
        queue.push(queue.shift());
        showCard();
      }
      function gotIt() {
        queue.shift();
        cleared += 1;
        showCard();
      }
      on(cardEl, 'click', flip);
      on(reveal, 'click', flip);
      on(againBtn, 'click', again);
      on(gotBtn, 'click', gotIt);
      deckKeys = { flip, again, gotIt, cardEl };

      page.replaceChildren(top, cardEl, cueRow, controls, foot);
      play();
    }

    function onKey(e) {
      if (mode !== 'deck' || !deckKeys || e.metaKey || e.ctrlKey || e.altKey || isTyping(e.target)) return;
      if (e.code === 'Space' || e.key === ' ') {
        e.preventDefault();
        deckKeys.flip();
      } else if (e.key === 'Enter' && e.target === deckKeys.cardEl) {
        deckKeys.flip();
      } else if (flipped && e.key === '1') {
        deckKeys.again();
      } else if (flipped && e.key === '2') {
        deckKeys.gotIt();
      }
    }

    // ---- tonight's cues ---------------------------------------------------------------------------
    function copyBox(text) {
      const code = h('pre', { class: 'fc-code' }, text);
      const copy = h('button', { class: 'btn btn-ghost fc-copy', type: 'button' }, 'Copy');
      on(copy, 'click', async () => {
        try {
          await navigator.clipboard.writeText(text);
          copy.textContent = 'Copied';
        } catch {
          // Clipboard access can be refused; select the text so Cmd+C works.
          const range = document.createRange();
          range.selectNodeContents(code);
          const sel = window.getSelection();
          sel.removeAllRanges();
          sel.addRange(range);
          copy.textContent = 'Selected: press ⌘C';
        }
      });
      return h('div', { class: 'fc-copybox' }, code, copy);
    }

    function showTonight() {
      mode = 'tonight';
      deckKeys = null;
      stopAudio();
      const cues = session.cues || {};
      const match = /--volume\s+([0-9]*\.?[0-9]+)/.exec(cues.sleep_command || '');
      let volume = match ? Math.min(1, Math.max(0, Number(match[1]))) : DEFAULT_SLEEP_VOLUME;
      let lastPlayed = cards.find((c) => c.cue && c.cue.url) || null;

      // Concepts with play buttons.
      const list = h('ol', { class: 'fc-cue-list' });
      for (const card of cards) {
        const row = h('li', { class: 'fc-cue-item' });
        if (card.cue && card.cue.url) {
          const btn = h('button', { class: 'btn btn-ghost fc-cue-play', type: 'button', 'aria-label': `Play the cue for ${card.concept || 'this card'}` }, speakerIcon());
          const blockedNote = h('span', { class: 'muted' }, 'Could not play');
          blockedNote.hidden = true;
          on(btn, 'click', () => {
            blockedNote.hidden = true;
            lastPlayed = card;
            playCue(card, volume, () => {
              blockedNote.hidden = false;
            });
          });
          row.append(btn, h('span', { class: 'fc-cue-concept' }, card.concept || card.cue.text || ''));
          if (card.cue.kind === 'tone') row.append(h('span', { class: 'fc-tag muted' }, 'tone'));
          if (card.from_missed) row.append(h('span', { class: 'chip fc-missed' }, 'You missed this'));
          row.append(blockedNote);
        } else {
          row.append(h('span', { class: 'fc-cue-concept' }, card.concept || ''), h('span', { class: 'fc-tag muted' }, 'no cue file'));
        }
        list.append(row);
      }
      const listCard = h('div', { class: 'card fc-panel' }, h('div', { class: 'h3' }, 'Concepts'), cards.length ? list : h('div', { class: 'muted' }, 'No cues were made.'));

      // Volume.
      const readout = h('span', { class: 'fc-vol-value' }, volume.toFixed(2));
      const slider = h('input', { class: 'fc-slider', type: 'range', min: '0', max: '1', step: '0.01', 'aria-label': 'Cue preview volume' });
      slider.value = String(volume);
      on(slider, 'input', () => {
        volume = Number(slider.value);
        readout.textContent = volume.toFixed(2);
        audio.volume(volume); // also turns down a cue that is playing right now
      });
      on(slider, 'change', () => {
        if (lastPlayed) playCue(lastPlayed, volume, null);
      });
      const volumeCard = h(
        'div',
        { class: 'card fc-panel' },
        h('div', { class: 'h3' }, 'Volume'),
        h('div', { class: 'fc-vol-row' }, slider, readout),
        h('div', { class: 'fc-note' }, 'Sleep cues should be barely audible: set this while awake'),
        h('div', { class: 'muted fc-small' }, 'The slider sets the preview volume in this app only. The command below carries its own --volume value; edit it there.'),
      );

      // Output.
      const outputCard = h('div', { class: 'card fc-panel' }, h('div', { class: 'h3' }, 'Sound output'));
      const detail = h('div', { class: 'fc-note' }, '');
      outputCard.append(detail);
      const setDetail = (status) => {
        detail.textContent = (status && status.audio && status.audio.detail) || '';
        detail.hidden = !detail.textContent;
      };
      if (state.status) setDetail(state.status);
      else {
        setDetail(null);
        api.get('/api/status').then(
          (status) => {
            if (!dead) setDetail(status);
          },
          () => {},
        );
      }
      Promise.resolve()
        .then(() => audio.outputs())
        .then((outs) => {
          if (dead || mode !== 'tonight' || !Array.isArray(outs) || !outs.length) return;
          const select = h('select', { class: 'input fc-output', 'aria-label': 'Output device' });
          const placeholder = h('option', { value: '' }, 'Choose an output device…');
          placeholder.disabled = true;
          placeholder.selected = true;
          select.append(placeholder, h('option', { value: '' }, 'System default'));
          for (const out of outs) select.append(h('option', { value: String(out.id) }, out.label || 'Unnamed device'));
          const problem = h('div', { class: 'muted fc-small' }, '');
          problem.hidden = true;
          on(select, 'change', async () => {
            problem.hidden = true;
            let ok = false;
            try {
              ok = (await audio.setOutput(select.value)) !== false;
            } catch {
              ok = false;
            }
            if (!ok && !dead) {
              problem.textContent = 'That output could not be selected.';
              problem.hidden = false;
            }
          });
          outputCard.append(select, problem);
        })
        .catch(() => {});

      // Export.
      const exportCard = h('div', { class: 'card fc-panel' }, h('div', { class: 'h3' }, 'Cue files'));
      if (cues.dir) {
        exportCard.append(h('div', { class: 'muted fc-small' }, `Folder${cues.count != null ? ` (${cues.count} cues)` : ''}`), copyBox(cues.dir));
      }
      if (cues.sleep_command) exportCard.append(h('div', { class: 'muted fc-small' }, 'Command'), copyBox(cues.sleep_command));
      if (!cues.dir && !cues.sleep_command) exportCard.append(h('div', { class: 'muted' }, 'No cue set was exported.'));
      if (cues.note) exportCard.append(h('div', { class: 'notice notice-warn fc-cues-note' }, cues.note));

      const { back, done } = navButtons();
      const again = on(h('button', { class: 'btn btn-ghost', type: 'button' }, 'Review the cards again'), 'click', startDeck);
      if (!cards.length) again.hidden = true;

      page.replaceChildren(
        h('div', { class: 'fc-top' }, h('div', { class: 'fc-top-text' }, h('h1', { class: 'h1' }, 'Tonight’s cues'), sessionLine())),
        h('div', { class: 'fc-tonight' }, h('div', { class: 'fc-col' }, listCard), h('div', { class: 'fc-col' }, volumeCard, outputCard, exportCard)),
        h('div', { class: 'fc-actions' }, back, again, done),
      );
    }

    async function init() {
      const id = params.id || (state.session && state.session.id);
      if (!id) {
        fail('No session was chosen.');
        return;
      }
      loading('Loading flashcards…');
      session = await api.get(`/api/sessions/${id}`);
      if (dead) return;
      if (!session.flashcards) {
        if (!session.summary) {
          fail('This session was not finished, so there are no flashcards for it.');
          return;
        }
        loading('Making flashcards and audio cues…');
        session = await api.post(`/api/sessions/${id}/flashcards`, {});
        if (dead) return;
      }
      cards = session.flashcards || [];
      if (!cards.length) {
        showTonight();
        page.prepend(h('div', { class: 'notice' }, 'No flashcards could be made from this session.'));
        return;
      }
      startDeck();
    }

    window.addEventListener('keydown', onKey);
    init().catch((err) => {
      if (!dead) fail(`Flashcards could not be loaded: ${err.message}`);
    });

    return () => {
      dead = true;
      window.removeEventListener('keydown', onKey);
      stopAudio();
    };
  },
};
