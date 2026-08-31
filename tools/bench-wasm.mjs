// Measure the WebAssembly build the browser actually runs.
//
// Node instantiates the same raw module the page does, so these numbers are the
// real single-threaded browser figures rather than a native proxy for them.

import { readFile } from 'node:fs/promises';
import { Borscht } from '../web/borscht.js';
import { PARAMS, STAT_NAMES } from '../web/params.js';

const wasmPath = new URL('../target/wasm32-unknown-unknown/release/borscht_wasm.wasm', import.meta.url);
const targets = process.argv.slice(2).map(Number).filter(Boolean);
const populations = targets.length ? targets : [100_000, 500_000, 1_000_000];
const TICKS = Number(process.env.TICKS ?? 60);

const wasm = await readFile(wasmPath);
console.log(`module: ${(wasm.length / 1024).toFixed(0)} KiB\n`);
console.log(
  ['target', 'organisms', 'ms/tick', 'ticks/s', 'render ms', 'MB'].map((h) => h.padStart(11)).join(''),
);

// Density is held constant as the world is scaled, matching Config::for_population.
const defaults = Object.fromEntries(PARAMS.map((p) => [p.name, p.value]));
const baseTotal = defaults.max_plants + defaults.max_animals;

for (const target of populations) {
  const borscht = await Borscht.load(wasm.buffer.slice(wasm.byteOffset, wasm.byteOffset + wasm.length));
  const scale = target / baseTotal;
  borscht.resetParams();
  borscht.configure(PARAMS, {
    max_plants: Math.round(defaults.max_plants * scale),
    max_animals: Math.round(defaults.max_animals * scale),
    initial_plants: Math.round(defaults.initial_plants * scale),
    initial_animals: Math.round(defaults.initial_animals * scale),
    world_size: defaults.world_size * Math.sqrt(scale),
    grid_dim: Math.max(32, Math.round(defaults.grid_dim * Math.sqrt(scale))),
  });
  borscht.create(1);

  borscht.tick(10); // warm up: first-touch page faults are not per-tick cost
  const start = performance.now();
  borscht.tick(TICKS);
  const perTick = (performance.now() - start) / TICKS;

  const renderStart = performance.now();
  for (let i = 0; i < 10; i += 1) borscht.render();
  const renderMs = (performance.now() - renderStart) / 10;

  const stats = borscht.statsObject(STAT_NAMES);
  const mb = borscht.exports.memory.buffer.byteLength / (1024 * 1024);
  console.log(
    [
      String(target),
      String(borscht.population),
      perTick.toFixed(2),
      (1000 / perTick).toFixed(1),
      renderMs.toFixed(2),
      mb.toFixed(0),
    ]
      .map((c) => c.padStart(11))
      .join(''),
  );
  if (!Number.isFinite(stats.total_matter) || stats.total_matter <= 0) {
    throw new Error('world did not initialise');
  }
}
