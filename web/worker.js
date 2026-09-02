// The simulation engine, and its worker entry point.
//
// The engine is a factory rather than a module bound to `self`, because it has
// two homes. In the multi-file app it runs in a Worker: a tick at a large
// muster takes long enough that running it on the UI thread would drop the
// frame rate to the tick rate and make panning feel broken. In the single-file
// build it runs on the main thread, because that build has no second file to
// load a worker from.
//
// The two differ in exactly two things, which is why they are the two
// arguments:
//
// * `post` delivers a message outward. In a worker that is `self.postMessage`
//   with a transfer list; in-page it is a direct call and the transfer list is
//   ignored, since there is no boundary to transfer across.
// * `schedule` runs the next step. A worker uses `setTimeout`, which yields so
//   control messages land between ticks rather than after them. On the main
//   thread that would starve rendering, so the page passes
//   `requestAnimationFrame` and the two interleave.
//
// Frames are copied out of WebAssembly memory rather than viewed in place. A
// view into wasm memory cannot be transferred, and growing that memory detaches
// every view over it. Two buffers are recycled so a frame is never allocated in
// the steady state.

import { Borscht, ColorMode } from './borscht.js';
import { PARAMS, STAT_NAMES } from './params.js';

export function createEngine({ post, schedule }) {
  let sim = null;
  let running = false;
  let colorMode = ColorMode.species;
  let ticksPerFrame = 1;
  let pending = false;

  // Recycled frame buffers, returned by the host after upload.
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

    post(
      {
        type: 'frame',
        buffer,
        count,
        stride,
        plants,
        stats,
        red: sim.teamCount(0),
        blue: sim.teamCount(1),
        outcome: sim.outcome,
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
    if (running) scheduleStep();
  }

  function scheduleStep() {
    if (pending) return;
    pending = true;
    schedule(step);
  }

  function build(seed, overrides) {
    sim.resetParams();
    if (overrides) sim.configure(PARAMS, overrides);
    sim.create(seed);
    tickMsAverage = 0;
  }

  // Messages that arrived before there was a world to apply them to.
  //
  // Loading the module is asynchronous, and the host sends its opening
  // configuration immediately after 'init'. Dropping those messages was a
  // standing trap: it produced a page whose controls read one way and whose
  // world ran another, with no error anywhere. They queue and are drained once
  // the world exists.
  let queued = [];

  async function handle(msg) {
    try {
      if (msg.type !== 'init' && !sim) {
        // 'recycle' returns a frame buffer, which is meaningless without a
        // world; everything else is configuration worth keeping.
        if (msg.type !== 'recycle') queued.push(msg);
        return;
      }
      switch (msg.type) {
        case 'init': {
          sim = await Borscht.load(msg.wasm);
          build(msg.seed, msg.overrides);
          post({ type: 'ready', params: PARAMS, statNames: STAT_NAMES });
          // Autoplay belongs here rather than in a 'play' the host sends after
          // 'init'. Loading the module is asynchronous, so that message would
          // arrive while `sim` was still null and be dropped by the guard
          // above -- leaving a page whose button says it is running and a world
          // that is not.
          running = Boolean(msg.autoplay);
          const waiting = queued;
          queued = [];
          for (const held of waiting) await handle(held);
          publish();
          if (running) scheduleStep();
          break;
        }
        case 'reset':
          running = false;
          build(msg.seed, msg.overrides);
          publish();
          break;
        case 'play':
          running = true;
          scheduleStep();
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
          // Frame buffers come back after the host uploads them.
          if (spare.length < 3) spare.push(msg.buffer);
          break;
        case 'inspect':
          post({ type: 'inspected', organism: sim.inspect(msg.x, msg.y, msg.radius) });
          break;
        default:
          break;
      }
    } catch (error) {
      post({ type: 'error', message: String(error?.message ?? error) });
    }
  }

  return { handle };
}

// Worker entry point. Guarded so the single-file build can import this module
// on the main thread without it trying to bind to a `self` that is the window.
if (typeof WorkerGlobalScope !== 'undefined' && self instanceof WorkerGlobalScope) {
  const engine = createEngine({
    post: (msg, transfer) => self.postMessage(msg, transfer),
    // setTimeout rather than a tight loop: it yields, so control messages
    // (pause, parameter changes) are handled between ticks instead of after.
    schedule: (fn) => setTimeout(fn, 0),
  });
  self.onmessage = (event) => engine.handle(event.data);
}
