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

// ---------------------------------------------------------- tree of life --
//
// Each lineage is a horizontal segment from the tick it split off to the tick it
// died out, with a connector back to its parent. Rows are ordered by walking the
// tree depth-first from the roots, so a lineage sits directly under the one it
// came from and a radiation reads as a block rather than as scattered lines.
//
// Extinct branches are most of the tree and are what makes it a history rather
// than a snapshot, so they stay in; the min-peak control is what keeps the
// picture legible.

const treeCanvas = $('tree');
const treeTip = $('tree-tip');
let treeData = { lineages: [], total: 0, dropped: 0 };
let treeLayout = [];
let treeMinPeak = 3;

function layoutTree(lineages) {
  const kept = lineages.filter((l) => l.peak >= treeMinPeak);
  if (!kept.length) return [];
  const byId = new Map(kept.map((l) => [l.id, l]));
  const all = new Map(lineages.map((l) => [l.id, l]));

  // A kept lineage whose parent was filtered out attaches to its nearest kept
  // ancestor, so pruning thins the tree instead of shattering it.
  const anchor = (l) => {
    let p = l.parent;
    const seen = new Set();
    while (p !== null && p !== undefined && !seen.has(p)) {
      if (byId.has(p)) return p;
      seen.add(p);
      p = all.get(p)?.parent ?? null;
    }
    return null;
  };

  const children = new Map();
  const roots = [];
  for (const l of kept) {
    const a = anchor(l);
    if (a === null) roots.push(l);
    else {
      if (!children.has(a)) children.set(a, []);
      children.get(a).push(l);
    }
  }
  const byBirth = (a, b) => a.birthTick - b.birthTick || a.id - b.id;
  roots.sort(byBirth);
  for (const list of children.values()) list.sort(byBirth);

  const rows = [];
  const visit = (l, depth) => {
    rows.push({ ...l, depth, row: rows.length, anchor: anchor(l) });
    for (const c of children.get(l.id) ?? []) visit(c, depth + 1);
  };
  for (const r of roots) visit(r, 0);
  return rows;
}

// Bottom padding clears the time bar, which floats over this canvas.
const TREE_PAD = { l: 18, r: 18, t: 34, b: 76 };

// Row geometry, computed once and used by both the draw and the hover test.
//
// The row height is *not* clamped upward. It used to have a 1.2px floor, which
// meant a few thousand rows needed several times the height of the canvas and
// everything past the first few hundred was painted below the bottom edge --
// while the footer still reported them all as shown. Letting rows go to a
// hairline keeps every one of them on the canvas by construction.
function treeRows(H, count) {
  const avail = H - TREE_PAD.t - TREE_PAD.b;
  const rowH = Math.min(26, avail / Math.max(1, count));
  const top = TREE_PAD.t + Math.max(0, (avail - count * rowH) / 2);
  return { rowH, top, y: (i) => top + i * rowH + rowH / 2 };
}

function drawTree(now) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, Math.floor(treeCanvas.clientWidth * dpr));
  const h = Math.max(1, Math.floor(treeCanvas.clientHeight * dpr));
  if (treeCanvas.width !== w || treeCanvas.height !== h) {
    treeCanvas.width = w;
    treeCanvas.height = h;
  }
  const ctx = treeCanvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = w / dpr;
  const H = h / dpr;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#06080e';
  ctx.fillRect(0, 0, W, H);

  const rows = treeLayout;
  if (!rows.length) {
    ctx.fillStyle = '#8b97ad';
    ctx.font = '13px ui-monospace, monospace';
    ctx.fillText('No lineages above the threshold yet.', 20, 30);
    return;
  }

  const pad = TREE_PAD;
  const tMax = Math.max(now, ...rows.map((r) => r.extinctTick ?? now), 1);
  const x = (t) => pad.l + (t / tMax) * (W - pad.l - pad.r);
  // Rows grow to fill the space when there are few lineages and shrink to a
  // hairline when there are thousands; the block is centred so a small tree
  // does not sit in a corner of an empty canvas.
  const { rowH, y } = treeRows(H, rows.length);
  const index = new Map(rows.map((r, i) => [r.id, i]));

  // Recessive time grid, labelled at the top so it never crosses the branches.
  ctx.strokeStyle = '#1a2130';
  ctx.fillStyle = '#5d6a80';
  ctx.font = '10px ui-monospace, monospace';
  ctx.lineWidth = 1;
  const step = Math.max(1, Math.pow(10, Math.floor(Math.log10(tMax / 5))) * (tMax / 5 > 5 * Math.pow(10, Math.floor(Math.log10(tMax / 5))) ? 5 : 1));
  for (let t = 0; t <= tMax; t += step) {
    const px = Math.round(x(t)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(px, pad.t - 6);
    ctx.lineTo(px, H - pad.b + 4);
    ctx.stroke();
    ctx.fillText(t >= 1000 ? `${(t / 1000).toFixed(t >= 10000 ? 0 : 1)}k` : String(t), px + 3, pad.t - 10);
  }

  // Connectors first, so branches sit on top of them. Opaque, they merge into a
  // solid block once there are thousands of them spanning hundreds of pixels
  // each; fading with the row height keeps them legible on a small tree and
  // turns them into texture on a large one.
  ctx.strokeStyle = `rgba(61, 74, 99, ${Math.max(0.15, Math.min(0.7, rowH * 0.35)).toFixed(3)})`;
  ctx.lineWidth = 1;
  for (const r of rows) {
    if (r.anchor === null || !index.has(r.anchor)) continue;
    const px = Math.round(x(r.birthTick)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(px, y(index.get(r.anchor)));
    ctx.lineTo(px, y(r.row));
    ctx.stroke();
  }

  for (const r of rows) {
    const end = r.extinctTick ?? now;
    const x0 = x(r.birthTick);
    const x1 = Math.max(x0 + 1.5, x(end));
    ctx.strokeStyle = `hsl(${(r.hue * 360).toFixed(0)} 72% 60%)`;
    ctx.lineWidth = Math.max(0.6, Math.min(3.5, rowH * 0.8));
    // Below a pixel a row cannot have its own line, so overlapping rows shade
    // rather than saturate and the density is readable.
    ctx.globalAlpha = rowH < 1 ? Math.max(0.25, rowH) : 1;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(x0, y(r.row));
    ctx.lineTo(x1, y(r.row));
    ctx.stroke();
    ctx.globalAlpha = 1;
    // A living lineage gets an emphasised end; an extinct one just stops. At
    // hairline heights thousands of dots become a block of their own, so they
    // only appear once a row is tall enough to own one.
    if (r.extinctTick === null && rowH >= 2) {
      ctx.fillStyle = '#e6ebf5';
      ctx.beginPath();
      ctx.arc(x1, y(r.row), Math.max(1.5, ctx.lineWidth * 0.5), 0, Math.PI * 2);
      ctx.fill();
    }
  }

  ctx.fillStyle = '#8b97ad';
  ctx.font = '11px ui-monospace, monospace';
  const living = rows.filter((r) => r.extinctTick === null).length;
  ctx.fillText(
    `${rows.length} lineages shown · ${living} living · ${treeData.total} ever recorded` +
      (treeData.dropped > 0 ? ` · ${treeData.dropped} not recorded` : ''),
    pad.l,
    H - pad.b + 22,
  );
  treeLayout = rows;
  // Exposed so the browser check can test the drawn geometry rather than
  // trusting the footer's count.
  window.__treeLayout = rows;
}

treeCanvas.addEventListener('pointermove', (e) => {
  if (!treeLayout.length) { treeTip.style.display = 'none'; return; }
  const rect = treeCanvas.getBoundingClientRect();
  // Same geometry the draw used, or the tooltip names the wrong lineage.
  const { rowH, top } = treeRows(rect.height, treeLayout.length);
  const i = Math.floor((e.clientY - rect.top - top) / rowH);
  const r = treeLayout[i];
  if (!r || e.clientY - rect.top < top) { treeTip.style.display = 'none'; return; }
  const end = r.extinctTick === null ? 'living' : `died ${Math.round(r.extinctTick)}`;
  treeTip.textContent = `lineage ${Math.round(r.id)} · born ${Math.round(r.birthTick)} · ${end} · peak ${Math.round(r.peak)}`;
  treeTip.style.display = 'block';
  treeTip.style.left = `${Math.min(e.clientX - rect.left + 14, rect.width - treeTip.offsetWidth - 8)}px`;
  treeTip.style.top = `${e.clientY - rect.top + 14}px`;
});
treeCanvas.addEventListener('pointerleave', () => { treeTip.style.display = 'none'; });

let showTree = false;
function setView(tree) {
  showTree = tree;
  treeCanvas.hidden = !tree;
  canvas.style.visibility = tree ? 'hidden' : 'visible';
  $('hud').style.display = tree ? 'none' : '';
  $('tree-card').hidden = !tree;
  $('view-tree').classList.toggle('on', tree);
  $('view-world').classList.toggle('on', !tree);
  $('view-tree').setAttribute('aria-pressed', String(tree));
  $('view-world').setAttribute('aria-pressed', String(!tree));
}
const matterSlider = $('matter');
function showMatter() {
  $('matter-val').textContent = `${Number(matterSlider.value).toFixed(2)}\u00d7`;
}
matterSlider.addEventListener('input', () => {
  showMatter();
  worker.postMessage({ type: 'matter', value: Number(matterSlider.value) });
});

$('view-world').addEventListener('click', () => setView(false));
$('view-tree').addEventListener('click', () => setView(true));
// A big tree needs thinning by hundreds, a small one by ones, so the slider is
// linear where the detail is and geometric where it is not.
function minPeakFor(step) {
  return step <= 10 ? step : Math.round(10 * Math.pow(50, (step - 10) / 20));
}
$('tree-min').addEventListener('input', (e) => {
  treeMinPeak = minPeakFor(Number(e.target.value));
  $('tree-min-val').textContent = String(treeMinPeak);
  treeLayout = layoutTree(treeData.lineages);
});

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

      if (msg.tree) {
        treeData = msg.tree;
        treeLayout = layoutTree(treeData.lineages);
        const living = treeData.lineages.filter((l) => l.extinctTick === null).length;
        $('tree-summary').textContent =
          `${treeData.total} lineages recorded · ${living} living` +
          (treeData.dropped > 0 ? ` · ${treeData.dropped} beyond the history limit` : '');
      }

      latestTick = msg.stats.tick;
      if (msg.history) oldestTick = msg.history.oldest;
      if (!scrubbing) {
        const scrub = $('t-scrub');
        scrub.min = String(Math.floor(oldestTick));
        scrub.max = String(Math.ceil(latestTick));
        scrub.value = String(Math.round(latestTick));
        $('t-tick').textContent = fmt(latestTick);
      }
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
  const base = byName.max_plants + byName.max_animals;
  const scale = target / base;
  const root = Math.sqrt(scale);
  // Nearest power of two, not the next one up. Cell size is what every per-cell
  // budget is denominated in -- how far mate search reaches, whether a cell has
  // anybody in it -- so rounding up would change the ecology at some scales and
  // not others. Mirrors Config::for_population.
  const want = byName.grid_dim * root;
  const lo = Math.max(1, Math.pow(2, Math.floor(Math.log2(Math.max(want, 1)))));
  const grid = Math.min(4096, Math.max(8, want >= lo * Math.SQRT2 ? lo * 2 : lo));
  return {
    max_plants: Math.round(byName.max_plants * scale),
    max_animals: Math.round(byName.max_animals * scale),
    initial_plants: Math.round(byName.initial_plants * scale),
    initial_animals: Math.round(byName.initial_animals * scale),
    world_size: byName.world_size * root,
    grid_dim: grid,
  };
}

function currentSetup() {
  return {
    seed: Math.max(0, Number($('seed').value) | 0),
    overrides: scaleOverrides(Number($('scale').value)),
  };
}

let scrubbing = false;
let latestTick = 0;
let oldestTick = 0;

function setRunning(next) {
  running = next;
  $('t-play').textContent = running ? '❚❚' : '▶';
  $('t-play').setAttribute('aria-label', running ? 'Pause' : 'Play');
  worker.postMessage({ type: running ? 'play' : 'pause' });
}

$('t-play').addEventListener('click', () => setRunning(!running));

$('t-step').addEventListener('click', () => {
  if (running) setRunning(false);
  worker.postMessage({ type: 'step', count: 1 });
});

// Scrubbing pauses: the worker has to load a checkpoint and re-tick forward, and
// racing that against a running simulation would fight over the same world.
$('t-scrub').addEventListener('pointerdown', () => {
  scrubbing = true;
  if (running) setRunning(false);
});
const endScrub = () => {
  if (!scrubbing) return;
  scrubbing = false;
  worker.postMessage({ type: 'seek', tick: Number($('t-scrub').value) });
};
$('t-scrub').addEventListener('pointerup', endScrub);
$('t-scrub').addEventListener('pointercancel', endScrub);
$('t-scrub').addEventListener('change', endScrub);
$('t-scrub').addEventListener('input', (e) => {
  $('t-tick').textContent = fmt(Number(e.target.value));
});

$('t-speed').addEventListener('input', (e) => {
  const value = Number(e.target.value);
  $('t-speed-val').textContent = String(value);
  worker.postMessage({ type: 'speed', value });
});

$('reset').addEventListener('click', () => {
  running = false;
  setRunning(false);
  history.clear();
  showInspector(null);
  // A new world is founded with its own stock of matter, so the control that
  // says how much of that stock is present goes back to all of it.
  matterSlider.value = '1';
  showMatter();
  worker.postMessage({ type: 'reset', ...currentSetup() });
});

$('scale').addEventListener('change', () => $('reset').click());

$('color').addEventListener('change', (e) => {
  worker.postMessage({ type: 'color', value: Number(e.target.value) });
});

$('save').addEventListener('click', () => worker.postMessage({ type: 'save' }));
$('load').addEventListener('click', () => $('file').click());
$('file').addEventListener('change', async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  setRunning(false);
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
  if (showTree) drawTree(latestTick);
  else renderer.draw(view);
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
