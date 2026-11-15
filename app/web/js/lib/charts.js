// Small inline-SVG charts. Colours come from the design tokens through the .chart-* classes in session.css.
import { fmtClock } from './dom.js';

const NS = 'http://www.w3.org/2000/svg';
let uid = 0;

function svg(tag, attrs = {}, ...children) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value != null) node.setAttribute(key, String(value));
  }
  node.append(...children);
  return node;
}

function div(className, text) {
  const node = document.createElement('div');
  node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

// A chart is a positioned wrapper holding the SVG and one tooltip that marks share.
function frame(className) {
  const wrap = div(`chart ${className}`);
  const tip = div('chart-tip');
  tip.hidden = true;
  const tipTitle = div('chart-tip-title');
  const tipDetail = div('chart-tip-detail');
  tip.append(tipTitle, tipDetail);

  function show(target, title, detail) {
    tipTitle.textContent = title;
    tipDetail.textContent = detail || '';
    tipDetail.hidden = !detail;
    tip.hidden = false;
    const box = wrap.getBoundingClientRect();
    const at = target.getBoundingClientRect();
    tip.style.left = `${at.left + at.width / 2 - box.left}px`;
    tip.style.top = `${at.top - box.top}px`;
  }
  function hide() {
    tip.hidden = true;
  }
  function bind(target, title, detail) {
    target.addEventListener('pointerenter', () => show(target, title, detail));
    target.addEventListener('pointerleave', hide);
    target.addEventListener('focus', () => show(target, title, detail));
    target.addEventListener('blur', hide);
  }
  return { wrap, tip, bind };
}

// One series by session. points: [{ label, value }] in order; null values leave a gap.
// opts: { width, height, format(value) -> string, ariaLabel }
export function sparkline(points, opts = {}) {
  const width = opts.width || 240;
  const height = opts.height || 64;
  const format = opts.format || ((v) => String(v));
  const pad = 12;

  const values = points.map((p) => p.value).filter((v) => typeof v === 'number' && Number.isFinite(v));
  if (!values.length) return div('chart chart-empty muted', 'No data yet');

  let lo = Math.min(...values);
  let hi = Math.max(...values);
  if (hi - lo < 1e-9) {
    lo -= 1;
    hi += 1;
  }
  const x = (i) => (points.length === 1 ? width / 2 : pad + (i * (width - 2 * pad)) / (points.length - 1));
  const y = (v) => height - pad - ((v - lo) / (hi - lo)) * (height - 2 * pad);

  const { wrap, tip, bind } = frame('chart-spark');
  const root = svg('svg', {
    viewBox: `0 0 ${width} ${height}`,
    width,
    height,
    role: 'img',
    'aria-label': opts.ariaLabel || 'Trend by session',
  });

  // Draw each run of consecutive sessions that have a value as its own line.
  let run = [];
  const flush = () => {
    if (run.length > 1) {
      const d = run.map(([px, py], i) => `${i ? 'L' : 'M'}${px.toFixed(1)} ${py.toFixed(1)}`).join(' ');
      root.append(svg('path', { d, class: 'chart-line' }));
    }
    run = [];
  };
  const marks = [];
  points.forEach((p, i) => {
    if (typeof p.value !== 'number' || !Number.isFinite(p.value)) {
      flush();
      return;
    }
    const px = x(i);
    const py = y(p.value);
    run.push([px, py]);
    marks.push({ px, py, point: p });
  });
  flush();

  for (const { px, py, point } of marks) {
    const dot = svg('circle', { cx: px, cy: py, r: 4, class: 'chart-dot' });
    const hit = svg('circle', { cx: px, cy: py, r: 12, class: 'chart-hit', tabindex: 0 });
    bind(hit, format(point.value), point.label);
    root.append(dot, hit);
  }

  wrap.append(root, tip);
  return wrap;
}

// The session as one bar with a mark per lapse.
// lapses: [{ start_s, duration_s, live, title, detail }]
export function timeline({ duration_s, lapses }) {
  const total = Math.max(1, Number(duration_s) || 0);
  const W = 1000;
  const H = 32;
  const hatchId = `chart-hatch-${++uid}`;
  const { wrap, tip, bind } = frame('chart-timeline');

  const root = svg('svg', {
    viewBox: `0 0 ${W} ${H}`,
    preserveAspectRatio: 'none',
    role: 'img',
    'aria-label': `Session timeline with ${lapses.length} lapse${lapses.length === 1 ? '' : 's'}`,
  });
  root.append(
    svg(
      'defs',
      {},
      svg(
        'pattern',
        { id: hatchId, width: 7, height: 7, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' },
        svg('rect', { width: 7, height: 7, class: 'chart-hatch-bg' }),
        svg('line', { x1: 0, y1: 0, x2: 0, y2: 7, class: 'chart-hatch-line' }),
      ),
    ),
    svg('rect', { x: 0, y: 8, width: W, height: H - 16, rx: 4, class: 'chart-tl-track' }),
  );

  for (const lapse of lapses) {
    const start = Math.min(Math.max(0, lapse.start_s), total);
    const w = Math.max(5, (Math.max(0, lapse.duration_s) / total) * W);
    const x0 = Math.min((start / total) * W, W - w);
    const mark = svg('rect', {
      x: x0,
      y: 3,
      width: w,
      height: H - 6,
      rx: 2,
      class: lapse.live ? 'chart-tl-mark' : 'chart-tl-mark-sim',
      fill: lapse.live ? null : `url(#${hatchId})`,
    });
    // The hit area is wider than a short mark so it can be hovered.
    const hitW = Math.max(w, 16);
    const hit = svg('rect', {
      x: Math.max(0, x0 - (hitW - w) / 2),
      y: 0,
      width: hitW,
      height: H,
      class: 'chart-hit',
      tabindex: 0,
    });
    bind(hit, lapse.title, lapse.detail);
    root.append(mark, hit);
  }

  const axis = div('chart-axis');
  axis.append(div('chart-axis-label', '0:00'), div('chart-axis-label', fmtClock(total)));
  wrap.append(root, axis, tip);
  return wrap;
}
