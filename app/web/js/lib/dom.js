// Small DOM and formatting helpers. Text always goes in as text nodes: nothing here parses HTML,
// so passage text, titles and AI output can never inject markup.

const SVG_NS = 'http://www.w3.org/2000/svg';
const SVG_TAGS = new Set([
  'svg', 'g', 'path', 'circle', 'ellipse', 'line', 'polyline', 'polygon', 'rect', 'text', 'tspan',
  'defs', 'marker', 'linearGradient', 'radialGradient', 'stop', 'clipPath', 'mask', 'pattern', 'use', 'symbol',
]);
// Set as properties (after children exist, so <select value> works) rather than attributes.
const PROPS = new Set(['value', 'checked', 'selected', 'disabled', 'hidden', 'indeterminate', 'readOnly', 'htmlFor']);

function append(el, child) {
  if (child == null || child === false || child === true) return;
  if (Array.isArray(child)) {
    for (const c of child) append(el, c);
  } else if (child instanceof Node) {
    el.appendChild(child);
  } else {
    el.appendChild(document.createTextNode(String(child)));
  }
}

/**
 * h(tag, attrs, ...children)
 * attrs: class (string or array), dataset ({k: v}), on<Event> (function), style (string or object),
 * and plain attributes (null/undefined/false are skipped, true becomes an empty attribute).
 * Children: nodes, strings, numbers, nested arrays; null/false are skipped.
 * SVG tag names create SVG elements.
 */
export function h(tag, attrs, ...children) {
  const isSvg = SVG_TAGS.has(tag);
  const el = isSvg ? document.createElementNS(SVG_NS, tag) : document.createElement(tag);

  // h('p', 'text') and h('div', node): the attrs argument was skipped.
  if (attrs != null && (typeof attrs !== 'object' || attrs instanceof Node || Array.isArray(attrs))) {
    children.unshift(attrs);
    attrs = null;
  }

  const props = [];
  for (const [key, val] of Object.entries(attrs || {})) {
    if (val == null || val === false) continue;
    if (key === 'class' || key === 'className') {
      const cls = Array.isArray(val) ? val.filter(Boolean).join(' ') : String(val);
      if (cls) el.setAttribute('class', cls);
    } else if (key === 'dataset') {
      for (const [k, v] of Object.entries(val)) if (v != null) el.dataset[k] = v;
    } else if (key === 'style') {
      if (typeof val === 'string') {
        el.setAttribute('style', val);
      } else {
        for (const [k, v] of Object.entries(val)) {
          if (v == null) continue;
          if (k.includes('-')) el.style.setProperty(k, String(v));
          else el.style[k] = v;
        }
      }
    } else if (/^on[A-Za-z]/.test(key) && typeof val === 'function') {
      el.addEventListener(key.slice(2).toLowerCase(), val);
    } else if (!isSvg && PROPS.has(key)) {
      props.push([key, val]);
    } else {
      el.setAttribute(key, val === true ? '' : String(val));
    }
  }

  append(el, children);
  for (const [key, val] of props) el[key] = val;
  return el;
}

const DASH = '–';

function finite(x) {
  return typeof x === 'number' && Number.isFinite(x);
}

/** Seconds as a clock: 102 -> "1:42", 3723 -> "1:02:03". Missing values give "–". */
export function fmtClock(seconds) {
  if (!finite(seconds)) return DASH;
  const total = Math.max(0, Math.floor(seconds));
  const hrs = Math.floor(total / 3600);
  const min = Math.floor((total % 3600) / 60);
  const sec = String(total % 60).padStart(2, '0');
  return hrs ? `${hrs}:${String(min).padStart(2, '0')}:${sec}` : `${min}:${sec}`;
}

/** A percentage already on the 0-100 scale: 96.4 -> "96.4%", 100 -> "100%". */
export function fmtPct(x) {
  if (!finite(x)) return DASH;
  const rounded = Math.round(x * 10) / 10;
  return `${Number.isInteger(rounded) ? rounded : rounded.toFixed(1)}%`;
}

/** A number with a fixed count of decimals and thousands separators: fmtNum(2400) -> "2,400". */
export function fmtNum(x, digits = 0) {
  if (!finite(x)) return DASH;
  return x.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

/** True when a key press belongs to something the user is typing into. */
export function isTyping(target) {
  if (!(target instanceof Element)) return false;
  if (target.isContentEditable) return true;
  return target.matches('input, textarea, select, [contenteditable=""], [contenteditable="true"]');
}
