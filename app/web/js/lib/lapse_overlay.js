// The sketch's "go back here" drawing: a blue circle around the word where focus was lost, and a
// dashed blue arrow that flows from where the reader is looking now to that circle.
// Drawn as one SVG layer in #overlay-root, in viewport pixels. The gaze point itself is never drawn:
// the arrow simply starts there.
//
// The circle is only as exact as the gaze behind it. A place taken from webcam gaze is drawn as a wide
// ellipse sized from the measured gaze error (radiiForError); a place that is known exactly (the mouse,
// the top-of-page fallback) keeps the tight circle round the word.
//
//   const overlay = createLapseOverlay(overlayRoot);
//   overlay.mark(wordElement);            // tight circle (it follows the element if layout moves)
//   overlay.mark(wordElement, {rx, ry});  // wide ellipse centred on the word, radii in px
//   overlay.point(() => ({x, y}) | null); // show the arrow; the start point is re-read at most 10x/s
//   overlay.center();                     // {x, y} of the circle, or null
//   overlay.geometry();                   // {cx, cy, rx, ry} of the circle, or null
//   overlay.clear();                      // remove everything

import { h } from './dom.js';

const TICK_MS = 100;          // how often the start point and the circle position are re-read
const EASE = 0.2;             // fraction of the remaining distance the arrow tail moves per frame
const GAP = 9;                // space between the arrowhead and the circle
const MIN_LENGTH = 46;        // closer than this to the circle's edge, the arrow is not drawn

// A gaze-sized mark: as wide as the gaze error, and 0.6 of that tall (so it spans the lines the
// reader may really have been on), within limits that keep it visible and on the page.
const WIDE_RX = [60, 220];
const WIDE_RY = [30, 110];

let markerCount = 0;

function clamp(x, lo, hi) {
  return Math.min(hi, Math.max(lo, x));
}

/** Radii {rx, ry} in px of the mark for a gaze error of errorPx. */
export function radiiForError(errorPx) {
  return { rx: clamp(errorPx, WIDE_RX[0], WIDE_RX[1]), ry: clamp(0.6 * errorPx, WIDE_RY[0], WIDE_RY[1]) };
}

/** Is a point inside the mark, grown by `margin` px on each side? Each radius is at least `minRadius`,
 *  so a tight circle still counts everything within minRadius of its centre. */
export function nearMark(pt, geom, margin = 0, minRadius = 0) {
  if (!pt || !geom) return false;
  const rx = Math.max(geom.rx + margin, minRadius);
  const ry = Math.max(geom.ry + margin, minRadius);
  return ((pt.x - geom.cx) / rx) ** 2 + ((pt.y - geom.cy) / ry) ** 2 <= 1;
}

export function createLapseOverlay(root) {
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

  let layer = null;
  let ring = null;
  let pulse = null;
  let arrow = null;
  let target = null;
  let radii = null;     // {rx, ry} of a wide mark, or null for the tight circle round the word
  let getStart = null;
  let goal = null;      // where the arrow tail is heading
  let tail = null;      // where it is drawn right now
  let geom = null;
  let ticker = null;
  let frame = 0;

  function build() {
    const id = `lapse-arrowhead-${++markerCount}`;
    ring = h('ellipse', { class: 'lapse-ring' });
    pulse = h('ellipse', { class: 'lapse-pulse' });
    arrow = h('path', { class: 'lapse-arrow', 'marker-end': `url(#${id})` });
    arrow.style.display = 'none';
    layer = h('svg', { class: radii ? 'lapse-layer is-wide' : 'lapse-layer', 'aria-hidden': 'true' },
      h('defs', null,
        h('marker', {
          id, markerUnits: 'userSpaceOnUse', markerWidth: 26, markerHeight: 26,
          viewBox: '0 0 26 26', refX: 21, refY: 13, orient: 'auto', overflow: 'visible',
        }, h('path', { class: 'lapse-arrowhead', d: 'M7 4 L22 13 L7 22' }))),
      pulse, ring, arrow);
    root.appendChild(layer);
  }

  function measure() {
    if (!target || !target.isConnected) return null;
    const r = target.getClientRects()[0] || target.getBoundingClientRect();
    if (!r || (!r.width && !r.height)) return null;
    const rx = Math.max(r.width / 2 + 14, 28);
    const ry = Math.max(r.height / 2 + 8, 22);
    return {
      cx: r.left + r.width / 2,
      cy: r.top + r.height / 2,
      // A wide mark never shrinks inside the word it is centred on.
      rx: radii ? Math.max(radii.rx, rx) : rx,
      ry: radii ? Math.max(radii.ry, ry) : ry,
    };
  }

  function placeRing() {
    for (const el of [ring, pulse]) {
      el.setAttribute('cx', geom.cx);
      el.setAttribute('cy', geom.cy);
      el.setAttribute('rx', geom.rx);
      el.setAttribute('ry', geom.ry);
    }
  }

  function drawArrow() {
    if (!arrow) return;
    if (!geom || !tail) {
      arrow.style.display = 'none';
      return;
    }
    const dx = geom.cx - tail.x;
    const dy = geom.cy - tail.y;
    const dist = Math.hypot(dx, dy);
    if (dist < Math.max(geom.rx, geom.ry) + MIN_LENGTH) {
      arrow.style.display = 'none';
      return;
    }
    // A quadratic curve that bows upward (toward the top of the screen) by about a fifth of its length.
    let nx = -dy / dist;
    let ny = dx / dist;
    if (ny > 0) {
      nx = -nx;
      ny = -ny;
    }
    const bow = Math.min(150, Math.max(28, dist * 0.2));
    const qx = (tail.x + geom.cx) / 2 + nx * bow;
    const qy = (tail.y + geom.cy) / 2 + ny * bow;
    // End on the circle's edge, on the side the curve arrives from, so the head points at the centre.
    const ux = qx - geom.cx;
    const uy = qy - geom.cy;
    const ulen = Math.hypot(ux, uy) || 1;
    const ex = ux / ulen;
    const ey = uy / ulen;
    const reach = 1 / Math.hypot(ex / (geom.rx + GAP), ey / (geom.ry + GAP));
    const endX = geom.cx + ex * reach;
    const endY = geom.cy + ey * reach;
    arrow.setAttribute('d', `M${tail.x.toFixed(1)} ${tail.y.toFixed(1)} Q${qx.toFixed(1)} ${qy.toFixed(1)} ${endX.toFixed(1)} ${endY.toFixed(1)}`);
    arrow.style.display = '';
  }

  function animate() {
    frame = 0;
    if (!getStart || !layer) return;
    if (goal) {
      if (!tail) {
        tail = { x: goal.x, y: goal.y };
      } else {
        tail.x += (goal.x - tail.x) * EASE;
        tail.y += (goal.y - tail.y) * EASE;
      }
    }
    drawArrow();
    frame = requestAnimationFrame(animate);
  }

  function tick() {
    geom = measure();
    if (!geom) {
      // The marked word left the page (it was re-rendered): nothing to point at.
      if (layer) layer.style.display = 'none';
      return;
    }
    layer.style.display = '';
    placeRing();
    if (getStart) {
      let p = null;
      try {
        p = getStart();
      } catch {
        p = null;
      }
      if (p && Number.isFinite(p.x) && Number.isFinite(p.y)) goal = { x: p.x, y: p.y };
    }
  }

  const overlay = {
    /** Circle this element: tightly, or with the given radii {rx, ry} (px) centred on it.
     *  Replaces any earlier circle and arrow. */
    mark(el, size = null) {
      overlay.clear();
      if (!el) return;
      target = el;
      radii = size && size.rx > 0 && size.ry > 0 ? { rx: size.rx, ry: size.ry } : null;
      build();
      tick();
      ticker = setInterval(tick, TICK_MS);
    },

    /** Draw the flowing arrow from getStart() to the circle. Returns false when nothing is circled
     *  or the reader asked for reduced motion (then only the circle shows). */
    point(startFn) {
      if (!layer || typeof startFn !== 'function' || reducedMotion.matches) return false;
      getStart = startFn;
      goal = null;
      tail = null;
      tick();
      if (!frame) frame = requestAnimationFrame(animate);
      return true;
    },

    center() {
      const g = measure();
      return g ? { x: g.cx, y: g.cy } : null;
    },

    /** {cx, cy, rx, ry} of the circle as drawn now, or null. */
    geometry() {
      return measure();
    },

    clear() {
      clearInterval(ticker);
      ticker = null;
      if (frame) cancelAnimationFrame(frame);
      frame = 0;
      if (layer) layer.remove();
      layer = ring = pulse = arrow = null;
      target = null;
      radii = null;
      getStart = null;
      goal = tail = geom = null;
    },
  };
  return overlay;
}
