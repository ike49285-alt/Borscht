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
        publish();
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
        publish();
        break;
      default:
        break;
    }
  } catch (error) {
    self.postMessage({ type: 'error', message: String(error?.message ?? error) });
  }
};
