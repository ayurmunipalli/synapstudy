// How the tracker decides when a lapse ends (app/web/js/state.js). Synthetic events only.
//   node app/tests/js/tracker_test.mjs        (from the project root)
import { register } from 'node:module';
register('./loader.mjs', import.meta.url);

let now = 1_000_000;
Date.now = () => now;
const timers = [];
globalThis.setTimeout = (fn, ms) => { timers.push({ fn, at: now + ms, dead: false }); return timers.length; };
globalThis.clearTimeout = (id) => { if (id && timers[id - 1]) timers[id - 1].dead = true; };
globalThis.setInterval = () => 0;
globalThis.clearInterval = () => {};
const advance = (ms) => {
  now += ms;
  for (const t of timers) if (!t.dead && t.at <= now) { t.dead = true; t.fn(); }
};

const { bus } = await import('../../web/js/bus.js');
const { state, tracker } = await import('../../web/js/state.js');
let failed = 0;
const check = (ok, msg) => { if (ok) console.log('ok  ', msg); else { failed += 1; console.error('FAIL', msg); } };
const starts = [];
const ends = [];
bus.on('lapse:start', (e) => starts.push(e.lapse));
bus.on('lapse:end', (e) => ends.push(e.lapse));
// Rotation vectors (gyro degrees, x y z). RIGHT and LEFT are opposite; DOWN is at right angles to both.
const RIGHT = [1, -40, 3];
const LEFT = [-2, 41, -1];
const DOWN = [30, 2, 1];
const drift = (source, rot = null, { acted = true, angle = null } = {}) => bus.emit('ws:drift',
  { type: 'drift', kind: 'look_away', source, acted, wall_ms: now, onset_wall_ms: now - 250, angle, rot });
const resume = (source, net_rot = null, peak_angle = null) => bus.emit('ws:resume',
  { type: 'resume', source, wall_ms: now, net_rot, peak_angle });
// One whole head movement: drift, then the detector's resume 3 s later (2 s after the movement stopped).
const move = (rot, net = rot, source = 'live') => { drift(source, rot); advance(3000); resume(source, net); };
const lapses = () => tracker.log().lapses;

tracker.begin({ id: 's1' }, { getPosition: () => ({ section_idx: 0 }) });
advance(10000);

const returns = [];
bus.on('lapse:return', (e) => returns.push(e.lapse));

// 1. A look-away as recorded: drift, "resume" ~3 s later (still away), then a second drift the other way.
drift('live', RIGHT, { angle: 31.5 }); advance(3000); resume('live', [2, -55, 4], 48.2);
check(!tracker.active() && ends.length === 1 && ends[0].end_method === 'movement_end' && Math.abs(ends[0].duration_s - 1.25) < 0.05,
  `the resume closes the lapse where the movement stopped (${ends[0] && ends[0].duration_s} s)`);
check(lapses()[0].angle === 31.5 && lapses()[0].peak_angle === 48.2, 'the lapse keeps the head angle that set it off and its peak');
advance(700); drift('live', LEFT);
check(starts.length === 1 && returns.length === 1 && returns[0] === ends[0] && ends[0].end_method === 'turn_back',
  'a drift in the opposite direction extends the same lapse (turn_back) and announces lapse:return');
check(Math.abs(ends[0].duration_s - 3.7) < 0.05, `its length now runs from onset to the turn back (${ends[0].duration_s} s)`);
advance(3000); resume('live', LEFT);
check(starts.length === 1 && ends.length === 1 && lapses().length === 1, 'the resume after the turn back changes nothing');

// 2. A lapse is joined at most once: another opposite movement right after is a new lapse.
advance(4000); move(LEFT);
check(lapses().length === 2 && returns.length === 1 && lapses()[0].end_method === 'turn_back'
  && Math.abs(lapses()[0].duration_s - 3.7) < 0.05, 'a lapse that already has its turn back is not extended again');
advance(60000);

// 3. A stray movement, then a real look-away 10 s later: the stray stays its own short lapse.
let n = lapses().length;
move(DOWN);                                   // stray
advance(7000); move(RIGHT);                   // turn away: not the reverse of the stray
check(lapses().length === n + 2 && starts.length === n + 2 && returns.length === 1,
  'a movement in an unrelated direction opens a new lapse');
advance(1000); move(LEFT);                    // turn back
let l = lapses();
check(l.length === n + 2 && returns.length === 2 && returns[1].id === l[n + 1].id && l[n + 1].end_method === 'turn_back'
  && l[n].end_method === 'movement_end' && l[n].duration_s < 2 && Math.abs(l[n + 1].duration_s - 4) < 0.1,
  `the turn back joins the look-away (${l[n + 1].duration_s} s), not the stray (${l[n].duration_s} s)`);
advance(60000);

// 4. Two look-aways the same way with no turn back seen in between: two lapses (the old rule made them one).
n = lapses().length;
move(RIGHT); advance(5000); move(RIGHT);
check(lapses().length === n + 2 && returns.length === 2, 'a movement in the same direction opens a new lapse');
check(lapses().slice(n).every((x) => x.end_method === 'movement_end'), 'and neither is marked as having a turn back');
advance(120000);

// 5. No rotation information (an older server, or no motion data): never joined.
n = lapses().length;
move(null); advance(1000); move(null);
check(lapses().length === n + 2 && returns.length === 2, 'drifts with no rotation each open a new lapse');
advance(120000);
n = lapses().length;
move(RIGHT); advance(1000); drift('live', null); advance(3000); resume('live', null);
check(lapses().length === n + 2 && returns.length === 2, 'a drift with no rotation is not taken as a turn back');
advance(120000);
n = lapses().length;
move(null, null); advance(1000); move(LEFT);
check(lapses().length === n + 2 && returns.length === 2, 'nor is anything joined to a lapse whose own rotation is unknown');
for (const bad of [[0, 0, 0], [1, 2], [1, NaN, 3], 'left']) {
  advance(120000);
  n = lapses().length;
  move(RIGHT); advance(1000); move(bad);
  check(lapses().length === n + 2, `a rotation of ${JSON.stringify(bad)} is treated as unknown`);
}
advance(120000);

// 6. When the resume has no net rotation, the drift's own rotation stands in for the turn away.
n = lapses().length;
move(RIGHT, null); advance(1000); move(LEFT);
check(lapses().length === n + 1 && lapses()[n].end_method === 'turn_back', 'the drift rotation is the fallback for the turn away');
advance(120000);
// ... and when both are sent, the whole movement (net_rot) decides, not the first second (a wind-up).
n = lapses().length;
move(LEFT, RIGHT); advance(1000); move(LEFT);
check(lapses().length === n + 1 && lapses()[n].end_method === 'turn_back', 'the net rotation of the movement decides when both are known');
advance(120000);

// A look away AND back inside one detector event (net rotation far below its peak): nothing is joined to it,
// though the later drift happens to be the reverse of its small leftover rotation.
n = lapses().length;
drift('live', RIGHT); advance(6000); resume('live', [-0.6, -5.6, 1.2], 48.9);
advance(20000); move([1, 40, -2]);
check(lapses().length === n + 2 && lapses()[n].end_method === 'movement_end' && Math.abs(lapses()[n].duration_s - 4.25) < 0.05,
  `an away-and-back event is complete in itself (${lapses()[n].duration_s} s) and a later movement is a new lapse`);
advance(120000);
n = lapses().length;
drift('live', RIGHT); advance(6000); resume('live', [0, 0, 0], 48.9);
advance(1000); move(LEFT);
check(lapses().length === n + 2, 'the same when the net rotation is exactly zero (no falling back to the drift rotation)');
advance(120000);

// 7. A 40 s look-away with an opposite return is ONE lapse of about 40 s.
n = lapses().length;
move(RIGHT); advance(36000); move(LEFT);
l = lapses();
check(l.length === n + 1 && l[n].end_method === 'turn_back' && Math.abs(l[n].duration_s - 39) < 1.5,
  `a long look-away and its turn back are one lapse (${l[n].duration_s} s)`);
advance(120000);

// 8. Too long ago to be a turn back: more than 90 s after the lapse ended.
n = lapses().length;
move(RIGHT); advance(89000); move(LEFT);
check(lapses().length === n + 2 && lapses()[n].end_method === 'movement_end', 'an opposite movement more than 90 s later opens a new lapse');
check(lapses().every((x) => x.duration_s <= 120.01), `no lapse is longer than 120 s (${Math.max(...lapses().map((x) => x.duration_s))} s)`);
advance(120000);

// 9. Replayed events follow the same rule, but a lapse is never joined across sources.
n = lapses().length;
move(RIGHT, RIGHT, 'replay'); advance(1000); move(LEFT, LEFT, 'replay');
check(lapses().length === n + 1 && lapses()[n].source === 'replay' && lapses()[n].end_method === 'turn_back', 'a replayed look-away joins its turn back');
advance(120000);
n = lapses().length;
move(RIGHT, RIGHT, 'replay'); advance(1000); move(LEFT, LEFT, 'live');
check(lapses().length === n + 2, 'a live drift is not the turn back of a replayed lapse');
advance(120000);

// 10. Ended by hand; the turn-back movement that follows is not a new lapse and does not extend it.
drift('live', RIGHT); advance(1000);
const before = lapses().length;
check(tracker.endLapse('manual') === true && ends.at(-1).end_method === 'manual', 'ended by hand');
advance(800); drift('live', LEFT);
check(!tracker.active() && lapses().length === before && ends.at(-1).end_method === 'manual', 'a drift right after a manual end is ignored');
advance(3000); resume('live', LEFT);
advance(5000); move(LEFT);
check(lapses().length === before + 1 && ends.at(-2).end_method === 'manual', 'and a later opposite movement does not extend a lapse ended by hand');
advance(120000);

// 11. Simulated events end on the resume itself and are never joined.
drift('simulated'); advance(2000); resume('simulated');
check(ends.at(-1).end_method === 'simulated' && Math.abs(ends.at(-1).duration_s - 2.25) < 0.05, 'a simulated resume ends the lapse');
check(ends.at(-1).angle === null && !('peak_angle' in ends.at(-1)), 'a simulated lapse has no head angle');
n = lapses().length;
advance(3000); drift('simulated', LEFT); advance(1000); resume('simulated');
advance(1000); drift('simulated', RIGHT); advance(1000); resume('simulated');
check(lapses().length === n + 2, 'simulated lapses are each their own lapse, whatever rotation they claim');

// 12. Kinds that are not acted on never open a lapse.
advance(30000); drift('live', RIGHT, { acted: false });
check(!tracker.active(), 'a drift that is not acted on is ignored');

const log = tracker.log();
check(log.lapses.every((l) => l.resume_ms !== null && typeof l.end_method === 'string'), 'every closed lapse has an end and an end_method');

// 13. A finish that returns after another session has begun does not replace that session in state.session.
const slow = tracker.finish();                       // s1: the POST is still pending
tracker.begin({ id: 's2' });
await slow;
check(state.session && state.session.id === 's2' && tracker.session().id === 's2',
  'a late finish leaves the newer running session as state.session');
// ...nor one that Home has already set up but the tracker has not begun yet.
const slow2 = tracker.finish();
state.session = { id: 's3' };
await slow2;
check(state.session.id === 's3' && !tracker.running(), 'a late finish leaves a newer set-up session as state.session');
// With nothing newer, the finished session (with its summary) becomes state.session, as before.
tracker.begin({ id: 's1' });
const done = await tracker.finish();
check(state.session === done && Boolean(done.summary) && !tracker.running(), 'otherwise finish() sets state.session to the finished session');

console.log(failed ? `${failed} FAILED` : 'all tracker tests passed');
process.exitCode = failed ? 1 : 0;
