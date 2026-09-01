// The simulation runs here, not on the main thread.
//
// A tick at a million organisms takes tens of milliseconds, so running it on
// the UI thread would drop the frame rate to the tick rate and make panning
// feel broken. In a worker the two are decoupled: the page renders at display
// rate from whatever frame the worker last produced, however slowly the world
// is actually advancing.
//
// Frames are copied out of WebAssembly memory into plain ArrayBuffers and
// transferred, not shared. A view into wasm memory cannot be transferred, and
// posting it without copying would hand the main thread a buffer the worker is
// about to overwrite. Two buffers are recycled so a frame is never allocated in
// the steady state.

import { Borscht, ColorMode } from './borscht.js';
import { PARAMS, STAT_NAMES } from './params.js';

let sim = null;
let running = false;
let colorMode = ColorMode.species;
let ticksPerFrame = 1;
let pending = false;

// Recycled frame buffers, returned by the main thread after upload.
const spare = [];

let tickMsAverage = 0;

// ---- rewind ---------------------------------------------------------------
//
// Periodic snapshots kept in a ring, so the timeline can be scrubbed backwards.
// Seeking loads the newest checkpoint at or before the target and re-ticks
// forward from there; because a snapshot carries the generator state, replaying
// reproduces exactly the run that was already seen rather than a new one.
//
// The ring is bounded by bytes rather than by count: a snapshot scales with the
// population, so a fixed count would quietly consume a gigabyte in a big world.
const CHECKPOINT_BUDGET = 96 * 1024 * 1024;
let checkpointEvery = 250;
let checkpoints = [];
let lineageEvery = 12;
let framesSincelineages = 0;

function checkpoint(force = false) {
  if (!sim) return;
  const tick = sim.tick(0);
  if (!force && checkpoints.length && tick - checkpoints[checkpoints.length - 1].tick < checkpointEvery) {
    return;
  }
  if (checkpoints.length && checkpoints[checkpoints.length - 1].tick === tick) return;
  checkpoints.push({ tick, bytes: sim.save() });
  let total = checkpoints.reduce((n, c) => n + c.bytes.length, 0);
  while (checkpoints.length > 2 && total > CHECKPOINT_BUDGET) {
    total -= checkpoints.shift().bytes.length;
  }
}

function seek(target) {
  if (!sim || !checkpoints.length) return;
  let at = checkpoints[0];
  for (const c of checkpoints) {
    if (c.tick <= target) at = c;
    else break;
  }
  sim.load(at.bytes);
  const forward = Math.max(0, Math.min(target - at.tick, 20000));
  if (forward > 0) sim.tick(forward);
  // Everything after the seek point is a future that has not happened yet.
  checkpoints = checkpoints.filter((c) => c.tick <= sim.tick(0));
  if (!checkpoints.length) checkpoint(true);
}

function history() {
  return {
    oldest: checkpoints.length ? checkpoints[0].tick : 0,
    checkpoints: checkpoints.map((c) => c.tick),
  };
}

function frame() {
  const { count, stride, plants, bytes } = sim.render(colorMode);
  const needed = count * stride;
  let buffer = spare.pop();
  if (!buffer || buffer.byteLength < needed) buffer = new ArrayBuffer(Math.max(needed, 1 << 16));
  new Uint8Array(buffer, 0, needed).set(bytes);
  return { buffer, count, stride, plants };
}

function publish(extra = {}) {
  const { buffer, count, stride, plants } = frame();
  const stats = {};
  const raw = sim.stats();
  for (let i = 0; i < STAT_NAMES.length; i += 1) stats[STAT_NAMES[i]] = raw[i];

  // The tree changes far more slowly than the world does, and sending hundreds
  // of branches every frame is wasted bandwidth.
  let tree = null;
  framesSincelineages += 1;
  if (framesSincelineages >= lineageEvery || extra.treeNow) {
    framesSincelineages = 0;
    tree = {
      lineages: sim.lineages(2, true),
      total: sim.lineageTotal(true),
      dropped: sim.lineageDropped(true),
    };
  }

  self.postMessage(
    {
      type: 'frame',
      buffer,
      count,
      stride,
      plants,
      stats,
      species: sim.species(24, true),
      plantSpecies: sim.species(12, false),
      worldSize: sim.worldSize,
      tickMs: tickMsAverage,
      running,
      tree,
      history: history(),
      ...extra,
    },
    [buffer],
  );
}

function step() {
  pending = false;
  if (!sim) return;
  if (running) {
    const start = performance.now();
    sim.tick(ticksPerFrame);
    checkpoint();
    const elapsed = (performance.now() - start) / ticksPerFrame;
    // Smoothed so the readout is legible rather than jittering every frame.
    tickMsAverage = tickMsAverage === 0 ? elapsed : tickMsAverage * 0.9 + elapsed * 0.1;
  }
  publish();
  if (running) schedule();
}

function schedule() {
  if (pending) return;
  pending = true;
  // setTimeout rather than a tight loop: it yields, so control messages
  // (pause, parameter changes) are handled between ticks instead of after.
  setTimeout(step, 0);
}

function build(seed, overrides) {
  sim.resetParams();
  if (overrides) sim.configure(PARAMS, overrides);
  sim.create(seed);
  tickMsAverage = 0;
  checkpoints = [];
  checkpoint(true);
}

self.onmessage = async (event) => {
  const msg = event.data;
  try {
    // Everything except 'init' needs a world. Messages can arrive during the
    // asynchronous module load, and a null dereference here surfaces as an
    // unhelpful "cannot read properties of null" on the page.
    if (msg.type !== 'init' && !sim) return;
    switch (msg.type) {
      case 'init': {
        sim = await Borscht.load(msg.wasm);
        build(msg.seed, msg.overrides);
        self.postMessage({ type: 'ready', params: PARAMS, statNames: STAT_NAMES });
        publish();
        break;
      }
      case 'reset':
        running = false;
        build(msg.seed, msg.overrides);
        publish();
        break;
      case 'play':
        running = true;
        schedule();
        break;
      case 'pause':
        running = false;
        publish();
        break;
      case 'step':
        running = false;
        sim.tick(msg.count ?? 1);
        checkpoint();
        publish({ treeNow: true });
        break;
      case 'seek':
        running = false;
        seek(msg.tick);
        publish({ treeNow: true });
        break;
      case 'checkpointEvery':
        checkpointEvery = Math.max(25, msg.value | 0);
        break;
      case 'speed':
        ticksPerFrame = Math.max(1, msg.value | 0);
        break;
      case 'color':
        colorMode = msg.value;
        if (!running) publish();
        break;
      case 'param':
        sim.setParam(msg.id, msg.value);
        break;
      case 'recycle':
        // Frame buffers come back after the main thread uploads them.
        if (spare.length < 3) spare.push(msg.buffer);
        break;
      case 'inspect':
        self.postMessage({ type: 'inspected', organism: sim.inspect(msg.x, msg.y, msg.radius) });
        break;
      case 'save':
        self.postMessage({ type: 'saved', bytes: sim.save() });
        break;
      case 'load':
        running = false;
        sim.load(new Uint8Array(msg.bytes));
        checkpoints = [];
        checkpoint(true);
        publish({ treeNow: true });
        break;
      default:
        break;
    }
  } catch (error) {
    self.postMessage({ type: 'error', message: String(error?.message ?? error) });
  }
};
