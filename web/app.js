// Page controller: owns the view transform, the charts and the UI, and talks to
// the simulation worker. It never touches WebAssembly directly.

import { Renderer } from './renderer.js';
import { PARAMS, ANIMAL_GENES, PLANT_GENES } from './params.js';

const $ = (id) => document.getElementById(id);
const canvas = $('view');
const banner = $('banner');

function fail(message) {
  banner.textContent = message;
  banner.style.display = 'block';
}

let renderer;
try {
  renderer = new Renderer(canvas);
} catch (error) {
  fail(`${error.message}. Try a current Chrome, Firefox or Safari.`);
  throw error;
}

const worker = new Worker('./worker.js', { type: 'module' });

// --------------------------------------------------------------- view state --

// Pan is in normalised world coordinates; zoom is how much of the world spans
// the shorter screen axis.
const view = { x: 0.5, y: 0.5, zoom: 1 };
let latest = null;
let worldSize = 1;
// Imported directly rather than awaited from the worker: the boot sequence
// needs the table to compute the initial scale, and waiting for it created an
// ordering bug where a reset arrived before the module had finished loading.
const params = PARAMS;

// -------------------------------------------------------------------- input --

let dragging = false;
let last = { x: 0, y: 0 };
let moved = 0;

canvas.addEventListener('pointerdown', (e) => {
  dragging = true;
  moved = 0;
  last = { x: e.clientX, y: e.clientY };
  canvas.classList.add('dragging');
  canvas.setPointerCapture(e.pointerId);
});

canvas.addEventListener('pointermove', (e) => {
  if (!dragging) return;
  const dx = e.clientX - last.x;
  const dy = e.clientY - last.y;
  moved += Math.abs(dx) + Math.abs(dy);
  last = { x: e.clientX, y: e.clientY };
  const shorter = Math.min(canvas.width, canvas.height) / (window.devicePixelRatio || 1);
  view.x -= dx / (shorter * view.zoom);
  view.y -= dy / (shorter * view.zoom);
  // The world is a torus; the view wraps with it rather than hitting an edge.
  view.x -= Math.floor(view.x);
  view.y -= Math.floor(view.y);
});

const endDrag = (e) => {
  if (!dragging) return;
  dragging = false;
  canvas.classList.remove('dragging');
  // A press that barely moved is a click, not a drag.
  if (moved < 4) inspectAt(e);
};
canvas.addEventListener('pointerup', endDrag);
canvas.addEventListener('pointercancel', endDrag);

canvas.addEventListener(
  'wheel',
  (e) => {
    e.preventDefault();
    const factor = Math.exp(-e.deltaY * 0.0015);
    view.zoom = Math.min(400, Math.max(0.6, view.zoom * factor));
  },
  { passive: false },
);

/** Screen position to world coordinates, accounting for pan, zoom and wrap. */
function toWorld(event) {
  const rect = canvas.getBoundingClientRect();
  const shorter = Math.min(rect.width, rect.height);
  const nx = (event.clientX - rect.left - rect.width / 2) / (shorter * view.zoom) + view.x;
  const ny = (event.clientY - rect.top - rect.height / 2) / (shorter * view.zoom) + view.y;
  return { x: (nx - Math.floor(nx)) * worldSize, y: (ny - Math.floor(ny)) * worldSize };
}

function inspectAt(event) {
  const { x, y } = toWorld(event);
  const rect = canvas.getBoundingClientRect();
  const shorter = Math.min(rect.width, rect.height);
  // Ten screen pixels' worth of world, so the hit area feels the same at any zoom.
  const radius = (10 / (shorter * view.zoom)) * worldSize;
  worker.postMessage({ type: 'inspect', x, y, radius: Math.max(radius, worldSize / 2000) });
}

// ------------------------------------------------------------------- charts --

class History {
  constructor(limit = 400) {
    this.limit = limit;
    this.series = new Map();
  }

  push(values) {
    for (const [key, value] of Object.entries(values)) {
      let arr = this.series.get(key);
      if (!arr) this.series.set(key, (arr = []));
      arr.push(value);
      if (arr.length > this.limit) arr.shift();
    }
  }

  clear() {
    this.series.clear();
  }
}

const history = new History();

function drawChart(canvasEl, keys, colors, { stacked = false } = {}) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(canvasEl.clientWidth * dpr));
  const height = Math.max(1, Math.floor(canvasEl.clientHeight * dpr));
  if (canvasEl.width !== width || canvasEl.height !== height) {
    canvasEl.width = width;
    canvasEl.height = height;
  }
  const ctx = canvasEl.getContext('2d');
  ctx.clearRect(0, 0, width, height);

  const series = keys.map((k) => history.series.get(k) ?? []);
  const n = Math.max(...series.map((s) => s.length), 0);
  if (n < 2) return;

  // A stacked chart is scaled by the sum, a layered one by the largest single
  // series, or the layers would clip.
  let peak = 0;
  for (let i = 0; i < n; i += 1) {
    let acc = 0;
    for (const s of series) {
      const v = s[i] ?? 0;
      if (stacked) acc += v;
      else peak = Math.max(peak, v);
    }
    if (stacked) peak = Math.max(peak, acc);
  }
  if (peak <= 0) return;

  const x = (i) => (i / (n - 1)) * width;
  const y = (v) => height - (v / peak) * (height - 2) - 1;

  if (stacked) {
    const running = new Array(n).fill(0);
    series.forEach((s, si) => {
      ctx.beginPath();
      ctx.moveTo(0, y(running[0]));
      for (let i = 0; i < n; i += 1) ctx.lineTo(x(i), y(running[i] + (s[i] ?? 0)));
      for (let i = n - 1; i >= 0; i -= 1) ctx.lineTo(x(i), y(running[i]));
      ctx.closePath();
      ctx.fillStyle = `${colors[si]}55`;
      ctx.fill();
      for (let i = 0; i < n; i += 1) running[i] += s[i] ?? 0;
    });
  }

  series.forEach((s, si) => {
    ctx.beginPath();
    for (let i = 0; i < s.length; i += 1) {
      const px = x(i + (n - s.length));
      const py = y(s[i]);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.strokeStyle = colors[si];
    ctx.lineWidth = 1.5 * dpr;
    ctx.stroke();
  });
}

// --------------------------------------------------------------------- HUD --

const fmt = (v) => {
  if (!Number.isFinite(v)) return '–';
  if (v >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(1)}k`;
  return String(Math.round(v));
};

const TRAITS = [
  ['mean_size', 'size', (v) => v.toFixed(2)],
  ['mean_max_speed', 'speed', (v) => v.toFixed(2)],
  ['mean_diet', 'carnivory', (v) => v.toFixed(3)],
  ['mean_vision', 'vision', (v) => v.toFixed(2)],
  ['mean_lifespan', 'lifespan', (v) => fmt(v)],
  ['mean_mutation_rate', 'mutation', (v) => v.toFixed(4)],
  ['mean_temp_opt', 'temp pref', (v) => v.toFixed(2)],
  ['mean_plant_toxicity', 'plant toxin', (v) => v.toFixed(3)],
  ['mean_plant_growth', 'plant growth', (v) => v.toFixed(3)],
  ['kills', 'kills/tick', (v) => v.toFixed(0)],
  ['productivity', 'productivity', (v) => v.toFixed(2)],
  ['drought_fraction', 'in drought', (v) => `${(v * 100).toFixed(0)}%`],
  ['temp_anomaly', 'temp anomaly', (v) => v.toFixed(2)],
  ['disturbances', 'disturbances', (v) => v.toFixed(0)],
];

function updatePanel(stats, species) {
  $('h-tick').textContent = fmt(stats.tick);
  $('h-plants').textContent = fmt(stats.plants);
  $('h-animals').textContent = fmt(stats.animals);
  $('h-species').textContent = `${stats.animal_species | 0} / ${stats.plant_species | 0}`;
  $('h-carn').textContent = `${(stats.carnivore_fraction * 100).toFixed(1)}%`;

  $('c-plants').textContent = fmt(stats.plants);
  $('c-animals').textContent = fmt(stats.animals);
  $('c-biomass').textContent = fmt(stats.plant_biomass);
  $('c-soil').textContent = fmt(stats.soil);
  $('c-bodies').textContent = fmt(stats.animal_mass);

  $('traits').innerHTML = TRAITS.map(
    ([key, label, format]) =>
      `<tr><td class="k">${label}</td><td>${Number.isFinite(stats[key]) ? format(stats[key]) : '–'}</td></tr>`,
  ).join('');

  const total = species.reduce((a, s) => a + s.population, 0) || 1;
  $('species-list').innerHTML = species
    .slice(0, 12)
    .map((s) => {
      const color = `hsl(${(s.hue * 360).toFixed(0)} 75% 62%)`;
      const share = (s.population / total) * 100;
      return `<div class="item">
        <i class="swatch" style="background:${color}"></i>
        <span style="width:34px">#${s.id}</span>
        <span class="bar"><i style="width:${share.toFixed(1)}%;background:${color}"></i></span>
        <span style="width:46px;text-align:right">${fmt(s.population)}</span>
      </div>`;
    })
    .join('') || '<div class="item" style="color:var(--muted)">no species yet</div>';
}

// Generated from the Rust gene definitions, not written out here: the
// inspector labels traits positionally, so a hand-maintained list mislabels
// every gene the moment one is renamed or reordered.
const GENE_NAMES = {
  animal: ANIMAL_GENES.map((n) => n.replace(/_/g, ' ')),
  plant: PLANT_GENES.map((n) => n.replace(/_/g, ' ')),
};

function showInspector(organism) {
  const box = $('inspector');
  if (!organism) {
    box.style.display = 'none';
    return;
  }
  box.style.display = 'block';
  const names = GENE_NAMES[organism.kind];
  const rows = [
    ['kind', organism.kind],
    ['id', organism.id],
    ['species', `#${organism.species}`],
    [organism.kind === 'animal' ? 'energy' : 'biomass', organism.level.toFixed(1)],
    ['age', Math.round(organism.age)],
    ...names.map((n, i) => [n, organism.traits[i].toFixed(3)]),
  ];
  $('inspect-table').innerHTML = rows
    .map(([k, v]) => `<tr><td class="k">${k}</td><td>${v}</td></tr>`)
    .join('');
}

// ------------------------------------------------------------------ worker --

let running = false;
let frames = 0;
let lastFpsAt = performance.now();

worker.onmessage = (event) => {
  const msg = event.data;
  switch (msg.type) {
    case 'ready':
      buildParamUI();
      break;
    case 'frame': {
      worldSize = msg.worldSize || 1;
      renderer.upload(new Uint8Array(msg.buffer, 0, msg.count * msg.stride), msg.count, msg.plants);
      // Hand the buffer straight back so the worker can reuse it.
      worker.postMessage({ type: 'recycle', buffer: msg.buffer }, [msg.buffer]);
      latest = msg;
      history.push({
        plants: msg.stats.plants,
        animals: msg.stats.animals,
        biomass: msg.stats.plant_biomass,
        soil: msg.stats.soil,
        bodies: msg.stats.animal_mass,
      });
      updatePanel(msg.stats, msg.species);
      $('h-ms').textContent = msg.tickMs ? msg.tickMs.toFixed(1) : '–';
      break;
    }
    case 'inspected':
      showInspector(msg.organism);
      break;
    case 'saved': {
      const blob = new Blob([msg.bytes], { type: 'application/octet-stream' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `borscht-${Date.now()}.borscht`;
      a.click();
      URL.revokeObjectURL(url);
      break;
    }
    case 'error':
      fail(msg.message);
      break;
    default:
      break;
  }
};

// Density is held constant as the world is scaled, so the ecology behaves the
// same at every setting and only the size of the map changes.
function scaleOverrides(target) {
  const byName = Object.fromEntries(params.map((p) => [p.name, p.value]));
  const base = (byName.max_plants ?? 700000) + (byName.max_animals ?? 300000);
  const scale = target / base;
  const root = Math.sqrt(scale);
  return {
    max_plants: Math.round((byName.max_plants ?? 700000) * scale),
    max_animals: Math.round((byName.max_animals ?? 300000) * scale),
    initial_plants: Math.round((byName.initial_plants ?? 300000) * scale),
    initial_animals: Math.round((byName.initial_animals ?? 12000) * scale),
    world_size: (byName.world_size ?? 2048) * root,
    grid_dim: Math.max(32, Math.round((byName.grid_dim ?? 256) * root)),
  };
}

function currentSetup() {
  return {
    seed: Math.max(0, Number($('seed').value) | 0),
    overrides: scaleOverrides(Number($('scale').value)),
  };
}

$('play').addEventListener('click', () => {
  running = !running;
  $('play').textContent = running ? 'Pause' : 'Play';
  worker.postMessage({ type: running ? 'play' : 'pause' });
});

$('step').addEventListener('click', () => {
  running = false;
  $('play').textContent = 'Play';
  worker.postMessage({ type: 'step', count: 1 });
});

$('reset').addEventListener('click', () => {
  running = false;
  $('play').textContent = 'Play';
  history.clear();
  showInspector(null);
  worker.postMessage({ type: 'reset', ...currentSetup() });
});

$('scale').addEventListener('change', () => $('reset').click());

$('speed').addEventListener('input', (e) => {
  const value = Number(e.target.value);
  $('speed-val').textContent = `${value}×`;
  worker.postMessage({ type: 'speed', value });
});

$('color').addEventListener('change', (e) => {
  worker.postMessage({ type: 'color', value: Number(e.target.value) });
});

$('save').addEventListener('click', () => worker.postMessage({ type: 'save' }));
$('load').addEventListener('click', () => $('file').click());
$('file').addEventListener('change', async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  running = false;
  $('play').textContent = 'Play';
  history.clear();
  const bytes = await file.arrayBuffer();
  worker.postMessage({ type: 'load', bytes }, [bytes]);
  e.target.value = '';
});

function buildParamUI() {
  const container = $('params');
  const groups = new Map();
  for (const p of params) {
    if (!groups.has(p.group)) groups.set(p.group, []);
    groups.get(p.group).push(p);
  }
  container.innerHTML = '';
  for (const [group, list] of groups) {
    const heading = document.createElement('h2');
    heading.textContent = group;
    heading.style.marginTop = '10px';
    container.append(heading);
    for (const p of list) {
      const wrap = document.createElement('div');
      wrap.className = 'param';
      const label = document.createElement('label');
      const span = document.createElement('span');
      span.style.flex = '1';
      span.textContent = p.name;
      const input = document.createElement('input');
      input.type = 'range';
      input.min = p.lo;
      input.max = p.hi;
      input.step = (p.hi - p.lo) / 200;
      input.value = p.value;
      const val = document.createElement('span');
      val.className = 'val';
      val.textContent = Number(p.value).toPrecision(3);
      input.addEventListener('input', () => {
        val.textContent = Number(input.value).toPrecision(3);
        worker.postMessage({ type: 'param', id: p.id, value: Number(input.value) });
      });
      label.append(span, input, val);
      const doc = document.createElement('div');
      doc.className = 'doc';
      doc.textContent = p.doc;
      wrap.append(label, doc);
      container.append(wrap);
    }
  }
}

// ------------------------------------------------------------------- frame --

function loop() {
  renderer.draw(view);
  if (latest) {
    drawChart($('chart-pop'), ['plants', 'animals'], ['#4ade80', '#60a5fa']);
    drawChart($('chart-matter'), ['biomass', 'soil', 'bodies'], ['#4ade80', '#a78bfa', '#fbbf24'], {
      stacked: true,
    });
  }
  frames += 1;
  const now = performance.now();
  if (now - lastFpsAt > 500) {
    $('h-fps').textContent = (frames / ((now - lastFpsAt) / 1000)).toFixed(0);
    frames = 0;
    lastFpsAt = now;
  }
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

// Boot: fetch the module here rather than in the worker so a failure surfaces
// with a readable message on the page.
(async () => {
  try {
    const response = await fetch('./borscht.wasm');
    if (!response.ok) {
      throw new Error(`could not load borscht.wasm (${response.status}). Run tools/build-web.sh first.`);
    }
    const wasm = await response.arrayBuffer();
    // The world is built inside 'init'. Posting a separate 'reset' here would
    // race the worker's asynchronous module load and arrive before there is a
    // simulation to reset.
    worker.postMessage({ type: 'init', wasm, ...currentSetup() }, [wasm]);
  } catch (error) {
    fail(error.message);
  }
})();
