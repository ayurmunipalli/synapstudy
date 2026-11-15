// Long-term insights: overall, by subject, and the list of sessions.
import { api } from '../api.js';
import { tracker } from '../state.js';
import { navigate } from '../router.js';
import { h, fmtClock, fmtPct, fmtNum } from '../lib/dom.js';
import { sparkline } from '../lib/charts.js';

const on = (el, type, fn) => {
  el.addEventListener(type, fn);
  return el;
};
const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

const ESTIMATE_NOTE =
  'Lapse lengths and focus % are estimates: the headband senses the head turning, not the moment you look back.';
// A session ended by leaving it for this screen is still being finished when the lists are first loaded.
const RECENT_MS = 5 * 60 * 1000;
const RELOAD_AFTER_MS = 6000;

// Session-level badge for a row. Rows from before "measured" existed (or from a server not yet
// restarted) lack the field: they are then badged SIMULATED as before, or not at all.
function rowBadge(s) {
  if (typeof s.measure_note === 'string' && /replayed recording/.test(s.measure_note)) {
    return h('span', { class: 'badge badge-replay' }, 'REPLAY');
  }
  if (s.simulated) return h('span', { class: 'badge badge-sim' }, 'SIMULATED');
  if (s.measured === false) return h('span', { class: 'badge badge-unknown' }, 'NOT MEASURED');
  return null;
}

function dateLabel(iso) {
  const d = new Date(iso);
  if (!iso || Number.isNaN(d.getTime())) return '';
  return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
}

function tile(value, label) {
  return h('div', { class: 'tile' }, h('div', { class: 'tile-value' }, value), h('div', { class: 'tile-label' }, label));
}

function stat(value, label) {
  return h('div', { class: 'hist-stat' }, h('div', { class: 'hist-stat-value' }, value), h('div', { class: 'hist-stat-label muted' }, label));
}

export const view = {
  mount(root) {
    let dead = false;
    let includeSimulated = false;
    let loadToken = 0;
    let reloaded = false;
    let reloadTimer = null;

    const toggle = h('input', { type: 'checkbox', class: 'hist-toggle-box' });
    const toggleLabel = h('label', { class: 'hist-toggle' }, toggle, h('span', {}, 'Include simulated sessions'));
    const excludedLine = h('div', { class: 'hist-excluded muted' }, '');
    const body = h('div', { class: 'hist-body' }, h('div', { class: 'hist-loading' }, h('div', { class: 'spinner' })));
    const page = h(
      'div',
      { class: 'hist-page' },
      h(
        'header',
        { class: 'hist-head' },
        h('div', {}, h('h1', { class: 'h1' }, 'Long-term insights'), h('div', { class: 'muted' }, 'Reading fluency and engagement, by subject and overall')),
        h('div', { class: 'hist-head-side' }, toggleLabel, excludedLine),
      ),
      body,
    );
    root.replaceChildren(page);

    on(toggle, 'change', () => {
      includeSimulated = toggle.checked;
      load();
    });

    function overallSection(overall, knowsUnmeasured) {
      const node = h('section', { class: 'hist-section' }, h('h2', { class: 'h2' }, 'Overall'));
      if (!overall || !overall.sessions) {
        node.append(h('div', { class: 'hist-empty' }, 'No finished sessions yet.'));
        return node;
      }
      if (includeSimulated) {
        node.append(
          h(
            'div',
            { class: 'notice notice-warn hist-sim-note' },
            h('span', { class: 'badge badge-sim' }, 'SIMULATED'),
            knowsUnmeasured
              ? ' These numbers include sessions with simulated or replayed lapses and sessions where focus was not measured.'
              : ' These numbers include sessions with simulated lapses.',
          ),
        );
      }
      node.append(
        h(
          'div',
          { class: 'hist-tiles' },
          tile(String(overall.sessions), overall.sessions === 1 ? 'Session' : 'Sessions'),
          tile(fmtNum(overall.minutes, 1), 'Minutes'),
          tile(fmtPct(overall.focus_pct), 'Focus'),
          tile(fmtNum(overall.lapses_per_min, 2), 'Lapses per minute'),
          tile(fmtNum(overall.wpm), 'Reading speed (wpm)'),
        ),
      );
      return node;
    }

    function trendBlock(title, points, format) {
      const withValue = points.filter((p) => p.value != null);
      if (!withValue.length) return null;
      const last = withValue[withValue.length - 1];
      return h(
        'div',
        { class: 'hist-trend' },
        h('div', { class: 'hist-trend-head' }, h('span', { class: 'muted' }, title), h('span', { class: 'hist-trend-last' }, `latest ${format(last.value)}`)),
        sparkline(points, { format, ariaLabel: `${title}, ${plural(withValue.length, 'session')}` }),
      );
    }

    function subjectCard(subject, tags) {
      const trend = subject.trend || [];
      const label = (p) => `${p.date || ''}${tags.has(p.session_id) ? ` · ${tags.get(p.session_id)}` : ''}`;
      const card = h(
        'div',
        { class: 'card hist-subject' },
        h('div', { class: 'h3' }, subject.subject || 'No subject'),
        h(
          'div',
          { class: 'hist-stats' },
          stat(fmtNum(subject.sessions), subject.sessions === 1 ? 'session' : 'sessions'),
          stat(fmtNum(subject.minutes, 1), 'minutes'),
          stat(fmtPct(subject.focus_pct), 'focus'),
          stat(fmtNum(subject.lapses_per_min, 2), 'lapses/min'),
          stat(fmtNum(subject.wpm), 'wpm'),
        ),
      );
      const focus = trendBlock('Focus by session', trend.map((p) => ({ label: label(p), value: p.focus_pct })), fmtPct);
      const speed = trendBlock('Reading speed by session', trend.map((p) => ({ label: label(p), value: p.wpm })), (v) => `${fmtNum(v)} wpm`);
      const trends = h('div', { class: 'hist-trends' });
      if (focus) trends.append(focus);
      if (speed) trends.append(speed);
      if (focus || speed) card.append(trends);
      return card;
    }

    function deleteCell(session) {
      const cell = h('td', { class: 'hist-cell-actions' });
      const ask = () => {
        const del = h('button', { class: 'btn btn-ghost hist-delete', type: 'button', 'aria-label': `Delete ${session.title || 'session'}` }, 'Delete');
        on(del, 'click', (e) => {
          e.stopPropagation();
          const yes = h('button', { class: 'btn hist-delete-yes', type: 'button' }, 'Delete');
          const no = h('button', { class: 'btn btn-ghost', type: 'button' }, 'Cancel');
          on(no, 'click', (ev) => {
            ev.stopPropagation();
            ask();
          });
          on(yes, 'click', async (ev) => {
            ev.stopPropagation();
            yes.disabled = true;
            no.disabled = true;
            try {
              await api.del(`/api/sessions/${session.id}`);
              if (!dead) load();
            } catch (err) {
              if (!dead) cell.replaceChildren(h('span', { class: 'hist-error' }, `Not deleted: ${err.message}`));
            }
          });
          cell.replaceChildren(h('span', { class: 'muted' }, 'Delete this session? '), yes, no);
        });
        cell.replaceChildren(del);
      };
      ask();
      on(cell, 'click', (e) => e.stopPropagation());
      return cell;
    }

    function sessionsSection(sessions) {
      const node = h('section', { class: 'hist-section' }, h('h2', { class: 'h2' }, 'Sessions'));
      if (!sessions.length) {
        node.append(h('div', { class: 'muted' }, 'Sessions appear here once you start one.'));
        return node;
      }
      const head = h('tr', {});
      for (const name of ['Date', 'Title', 'Subject', 'Type', 'Time', 'Focus', 'Lapses', 'wpm', '']) head.append(h('th', {}, name));
      const rows = h('tbody', {});
      for (const s of sessions) {
        const finished = Boolean(s.ended_at);
        const title = h('td', { class: 'hist-cell-title' }, s.title || 'Untitled');
        const badge = rowBadge(s);
        if (badge) title.append(' ', badge);
        const row = h(
          'tr',
          { class: finished ? 'hist-row hist-row-open' : 'hist-row' },
          h('td', { class: 'muted' }, dateLabel(s.created_at)),
          title,
          h('td', {}, s.subject || ''),
          h('td', { class: 'muted' }, s.mode === 'video' ? 'Video' : 'Reading'),
        );
        if (finished) {
          // No headband and no lapses: nothing was measured, so neither a focus % nor "0 lapses" is shown.
          const unmeasured = s.measured === false && s.focus_pct == null;
          row.append(
            h('td', { class: 'hist-num' }, fmtClock(s.duration_s)),
            h('td', { class: unmeasured ? 'muted' : 'hist-num' }, unmeasured ? 'Not measured' : fmtPct(s.focus_pct)),
            h('td', { class: 'hist-num' }, unmeasured && !s.lapse_count ? fmtNum(null) : fmtNum(s.lapse_count)),
            h('td', { class: 'hist-num' }, fmtNum(s.wpm)),
          );
          row.setAttribute('tabindex', '0');
          row.setAttribute('title', 'Open insights');
          const open = () => navigate('summary', { id: s.id });
          on(row, 'click', open);
          on(row, 'keydown', (e) => {
            if (e.key === 'Enter' && e.target === row) open();
          });
        } else {
          row.append(h('td', { class: 'muted', colspan: '4' }, 'Not finished'));
        }
        row.append(deleteCell(s));
        rows.append(row);
      }
      node.append(h('div', { class: 'card hist-table-wrap' }, h('table', { class: 'hist-table' }, h('thead', {}, head), rows)));
      return node;
    }

    async function load() {
      const token = ++loadToken;
      let insights;
      let sessions;
      try {
        [insights, sessions] = await Promise.all([
          api.get(`/api/insights?include_simulated=${includeSimulated ? 1 : 0}`),
          api.get('/api/sessions'),
        ]);
      } catch (err) {
        if (dead || token !== loadToken) return;
        body.replaceChildren(h('div', { class: 'notice notice-warn' }, `Insights could not be loaded: ${err.message}`));
        return;
      }
      if (dead || token !== loadToken) return;
      sessions = Array.isArray(sessions) ? sessions : [];
      // Trend points of sessions that are only there because the toggle is on say so in their label.
      const tags = new Map();
      for (const s of sessions) {
        if (s.simulated) tags.set(s.id, 'simulated');
        else if (s.measured === false) tags.set(s.id, 'not measured');
      }

      // An older server does not send excluded_unmeasured: the line then reads as it did before.
      const knowsUnmeasured = typeof insights.excluded_unmeasured === 'number';
      const excluded = insights.excluded_simulated || 0;
      const unmeasured = knowsUnmeasured ? insights.excluded_unmeasured : 0;
      if (includeSimulated) {
        excludedLine.textContent = knowsUnmeasured ? 'Simulated and unmeasured sessions are included.' : 'Simulated sessions are included.';
      } else if (!unmeasured) {
        excludedLine.textContent = `${plural(excluded, 'simulated session')} ${excluded === 1 ? 'is' : 'are'} excluded.`;
      } else if (!excluded) {
        excludedLine.textContent = `${plural(unmeasured, 'unmeasured session')} ${unmeasured === 1 ? 'is' : 'are'} excluded.`;
      } else {
        excludedLine.textContent = `${excluded} simulated and ${plural(unmeasured, 'unmeasured session')} are excluded.`;
      }

      const parts = [overallSection(insights.overall, knowsUnmeasured)];
      const subjects = insights.by_subject || [];
      if (insights.overall && insights.overall.sessions && subjects.length) {
        const grid = h('div', { class: 'hist-subjects' });
        for (const subject of subjects) grid.append(subjectCard(subject, tags));
        parts.push(h('section', { class: 'hist-section' }, h('h2', { class: 'h2' }, 'By subject'), grid));
      }
      parts.push(sessionsSection(sessions));
      // Focus % from a headband session is an estimate. A row without "measured" (saved before the
      // field existed) may be one too, unless it is marked simulated.
      const headband = (s) => s.ended_at && s.focus_pct != null && !s.simulated && s.measured !== false;
      if (sessions.some(headband)) parts.push(h('div', { class: 'hist-footnote muted' }, ESTIMATE_NOTE));
      body.replaceChildren(...parts);

      // One more look, once: a session that was just ended may still be listed as "Not finished".
      const now = Date.now();
      const pending = (s) => !s.ended_at && (tracker.running() || now - new Date(s.created_at).getTime() < RECENT_MS);
      if (!reloaded && sessions.some(pending)) {
        reloaded = true;
        reloadTimer = setTimeout(() => {
          if (!dead) load();
        }, RELOAD_AFTER_MS);
      }
    }

    load();

    return () => {
      dead = true;
      clearTimeout(reloadTimer);
    };
  },
};
