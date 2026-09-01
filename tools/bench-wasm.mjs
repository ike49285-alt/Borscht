// Measure the WebAssembly build the browser actually runs.
//
// Node instantiates the same raw module the page does, so these numbers are the
// real single-threaded browser figures rather than a native proxy for them.

import { readFile } from 'node:fs/promises';
import { Borscht } from '../web/borscht.js';
import { PARAMS, STAT_NAMES } from '../web/params.js';

const wasmPath = new URL('../target/wasm32-unknown-unknown/release/borscht_wasm.wasm', import.meta.url);
const targets = process.argv.slice(2).map(Number).filter(Boolean);
const populations = targets.length ? targets : [20_000, 100_000, 500_000, 1_000_000];
const TICKS = Number(process.env.TICKS ?? 60);

const wasm = await readFile(wasmPath);
console.log(`module: ${(wasm.length / 1024).toFixed(0)} KiB\n`);
console.log(
  ['target', 'on field', 'ms/tick', 'ticks/s', 'render ms', 'MB'].map((h) => h.padStart(11)).join(''),
);

// Density is held constant as the field is scaled, matching Config::for_muster.
const defaults = Object.fromEntries(PARAMS.map((p) => [p.name, p.value]));
const baseTotal = defaults.units_per_side * 2;

for (const target of populations) {
  const borscht = await Borscht.load(wasm.buffer.slice(wasm.byteOffset, wasm.byteOffset + wasm.length));
  const scale = target / baseTotal;
  borscht.resetParams();
  const root = Math.sqrt(scale);
  const want = defaults.grid_dim * root;
  const lo = Math.max(1, Math.pow(2, Math.floor(Math.log2(Math.max(want, 1)))));
  borscht.configure(PARAMS, {
    units_per_side: Math.max(1, Math.round(defaults.units_per_side * scale)),
    max_units: Math.round(target * 1.05) + 16,
    field_size: defaults.field_size * root,
    grid_dim: Math.min(4096, Math.max(8, want >= lo * Math.SQRT2 ? lo * 2 : lo)),
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
  if (!Number.isFinite(stats.red) || stats.red + stats.blue <= 0) {
    throw new Error('the armies did not muster');
  }
}
