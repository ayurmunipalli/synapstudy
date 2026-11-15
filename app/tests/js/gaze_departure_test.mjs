// The look-away place (gaze.js departurePoint). Synthetic gaze traces only.
//   node app/tests/js/gaze_departure_test.mjs        (from the project root)
import { register } from 'node:module';
register('./loader.mjs', import.meta.url);
globalThis.document = { createElement: () => ({}) };
const { departurePoint } = await import('../../web/js/lib/gaze.js');

let failed = 0;
const check = (name, ok) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}`); if (!ok) failed += 1; };
const onPassage = (p) => p.x >= 100 && p.x <= 1100 && p.y >= 100 && p.y <= 800;
// 30 Hz gaze: reading on the RIGHT column (x ~ 850-950, slow drift along a line), then at t0 a 400-ms
// sweep to the LEFT edge and off screen; the head-turn onset is declared 900 ms after the sweep began.
function trace(seed = 1) {
  let r = seed;
  const noise = () => { r = (r * 16807) % 2147483647; return (r / 2147483647 - 0.5) * 60; };
  const h = [];
  const t0 = 10000;
  for (let t = 0; t < t0; t += 33) h.push({ x: 850 + (t % 2000) / 20 + noise(), y: 400 + noise(), t });
  for (let t = t0; t < t0 + 400; t += 33) h.push({ x: 900 - (t - t0) * 2.5 + noise(), y: 400 + noise(), t });
  for (let t = t0 + 400; t < t0 + 1500; t += 33) h.push({ x: -50 + noise(), y: 420 + noise(), t });
  return { h, onset: t0 + 900 };
}
for (const seed of [1, 7, 42]) {
  const { h, onset } = trace(seed);
  const p = departurePoint(h, onset, onPassage);
  check(`seed ${seed}: place is on the right column where reading was (x=${p && Math.round(p.x)})`, p && p.x > 800);
  // the old fixed window [onset-2500, onset-800] would include the sweep; the new place must not lean left
}
check('no on-passage gaze: null', departurePoint([{ x: -50, y: 400, t: 9000 }], 10000, onPassage) === null);
check('too few points before the sweep: null', departurePoint([{ x: 900, y: 400, t: 9990 }], 10000, onPassage) === null);
// Reading slowly right-to-left within the passage (not a look-away) is not a sweep: the place is the reading
// from before the head could have started turning (window ends 1.8 s before the onset), not trimmed further.
{
  const h = [];
  for (let t = 0; t < 5000; t += 33) h.push({ x: 1000 - t / 10, y: 300, t });
  const p = departurePoint(h, 5000, onPassage);
  check(`slow movement is not trimmed past the pre-turn window (x=${p && Math.round(p.x)})`, p && p.x > 700 && p.x < 850);
}
// Head-turn drift: webcam gaze slides left SLOWLY (300 px/s, below the sweep speed) during the 1.5 s before the
// onset while still on the passage. The pre-turn cap keeps the place on the right column.
{
  const h = [];
  for (let t = 0; t < 8000; t += 33) h.push({ x: 900 + ((t * 7) % 50), y: 400, t });
  for (let t = 8000; t < 9500; t += 33) h.push({ x: 900 - (t - 8000) * 0.3, y: 400, t });
  const p = departurePoint(h, 9500, onPassage);
  check(`slow head-turn drift does not move the place left (x=${p && Math.round(p.x)})`, p && p.x > 850);
}
// Safety step back up: half the gaze error in lines, rounded up.
{
  const { safetyLines, wordLinesAbove } = await import('../../web/js/lib/gaze.js');
  check('error 3 lines -> 2 up', safetyLines(150, 50) === 2);
  check('error 2.5 lines -> 2 up', safetyLines(125, 50) === 2);
  check('error 2 lines -> 1 up', safetyLines(100, 50) === 1);
  check('error 0.5 line -> 1 up', safetyLines(25, 50) === 1);
  check('unknown -> 0', safetyLines(null, 50) === 0 && safetyLines(100, 0) === 0);
  // two columns of 4 lines x 3 words; word = 60 px wide, line pitch 50 px
  const rects = [];
  for (const colLeft of [0, 400]) for (let ln = 0; ln < 4; ln++) for (let w = 0; w < 3; w++) {
    const left = colLeft + w * 70;
    rects.push({ left, right: left + 60, top: 100 + ln * 50, bottom: 130 + ln * 50 });
  }
  check('2 up, same x', wordLinesAbove(rects, 3 * 3 + 1, 2) === 1 * 3 + 1);
  check('0 up = same word', wordLinesAbove(rects, 7, 0) === 7);
  check('above the top of column 2 is the end of column 1\'s last line', wordLinesAbove(rects, 12, 1) === 11);
  check('not enough lines above -> first line', wordLinesAbove(rects, 4, 5) === 1);
}
if (failed) { console.log(`${failed} FAILED`); process.exit(1); }
console.log('all gaze departure tests passed');
