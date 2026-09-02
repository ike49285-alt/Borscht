// Page controller: owns the view transform, the charts and the UI, and talks to
// the simulation worker. It never touches WebAssembly directly.

import { Renderer } from './renderer.js';
import { PARAMS } from './params.js';

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

// The engine's home is a decision the host makes, not this file.
//
// `window.__borschtEngineHost` lets the single-file build hand over a
// same-thread shim with the same postMessage/onmessage surface, so every
// message site below is identical in both builds and there is only one viewer
// to maintain. Absent it -- which is the multi-file app -- this is a real
// Worker, exactly as before.
const worker = globalThis.__borschtEngineHost ?? new Worker('./worker.js', { type: 'module' });

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

// Every pointer currently down, so a second one can turn a drag into a pinch.
// Touch devices have no wheel, so without this they cannot zoom at all.
const pointers = new Map();
let pinch = null;

const spread = () => {
  const [a, b] = [...pointers.values()];
  return Math.hypot(a.x - b.x, a.y - b.y);
};

canvas.addEventListener('pointerdown', (e) => {
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  canvas.setPointerCapture(e.pointerId);
  if (pointers.size === 2) {
    // A pinch is not a click and not a drag; drop out of both.
    dragging = false;
    canvas.classList.remove('dragging');
    pinch = { from: spread(), zoom: view.zoom };
    return;
  }
  if (pointers.size > 2) return;
  dragging = true;
  moved = 0;
  last = { x: e.clientX, y: e.clientY };
  canvas.classList.add('dragging');
});

canvas.addEventListener('pointermove', (e) => {
  if (pointers.has(e.pointerId)) pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (pinch && pointers.size === 2) {
    const now = spread();
    if (pinch.from > 0) {
      view.zoom = Math.min(400, Math.max(0.6, pinch.zoom * (now / pinch.from)));
    }
    return;
  }
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
  pointers.delete(e.pointerId);
  if (pointers.size < 2) pinch = null;
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
  // Fourteen screen pixels' worth of world, so the hit area feels the same at
  // any zoom and is reachable with a fingertip rather than a mouse pixel.
  const radius = (14 / (shorter * view.zoom)) * worldSize;
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

/// Matches `Outcome` in the core, and the `outcome` export in the wasm ABI.
const OUTCOME = [
  'both lines holding',
  'red holds the field',
  'blue holds the field',
  'both armies broke',
];

// --------------------------------------------------------------------- HUD --

const fmt = (v) => {
  if (!Number.isFinite(v)) return '–';
  if (v >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(1)}k`;
  return String(Math.round(v));
};

function updatePanel(stats, red, blue) {
  $('h-tick').textContent = fmt(stats.tick ?? 0);
  $('h-plants').textContent = fmt(red);
  $('h-animals').textContent = fmt(blue);
  $('h-species').textContent = fmt((stats.red_holding ?? 0) + (stats.blue_holding ?? 0));
  const alive = red + blue;
  const routing = alive - ((stats.red_holding ?? 0) + (stats.blue_holding ?? 0));
  $('h-carn').textContent = alive > 0 ? `${((routing / alive) * 100).toFixed(1)}%` : '0%';
  $('c-plants').textContent = fmt(red);
  $('c-animals').textContent = fmt(blue);
  $('c-biomass').textContent = fmt(stats.red_holding ?? 0);
  $('c-soil').textContent = fmt(stats.blue_holding ?? 0);
  $('c-bodies').textContent = fmt(routing);
}

/// What `inspect` returns, in order, matching INSPECT_FIELDS in the wasm crate.
const INSPECT_FIELDS = ['side', 'type', 'health', 'max health', 'nerve', 'speed', 'facing', 'routing'];

function showInspector(unit) {
  const card = $('inspector');
  const table = $('inspect-table');
  if (!unit || !unit.length) {
    card.style.display = 'none';
    return;
  }
  const rows = INSPECT_FIELDS.map((name, i) => {
    let v = unit[i];
    if (name === 'side') v = v === 0 ? 'red' : 'blue';
    else if (name === 'routing') v = v > 0.5 ? 'yes' : 'no';
    else if (name === 'type') v = `#${Math.round(v)}`;
    else v = Number(v).toFixed(2);
    return `<tr><td class="k">${name}</td><td>${v}</td></tr>`;
  });
  table.innerHTML = rows.join('');
  card.style.display = 'block';
}

// The second view. Where the ecology had a phylogeny, a battle has an order of
// battle: who is left, who is still fighting, and how the strength of the two
// sides has moved against each other over the engagement.
const treeCanvas = $('tree');
const treeTip = $('tree-tip');

function drawReport() {
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

  const red = history.series.get('red') ?? [];
  const blue = history.series.get('blue') ?? [];
  const n = Math.max(red.length, blue.length);
  if (n < 2) {
    ctx.fillStyle = '#8b97ad';
    ctx.font = '13px ui-monospace, monospace';
    ctx.fillText('The armies have not yet met.', 24, 34);
    return;
  }

  const pad = { l: 56, r: 24, t: 40, b: 84 };
  const peak = Math.max(...red, ...blue, 1);
  const x = (i) => pad.l + (i / (n - 1)) * (W - pad.l - pad.r);
  const y = (v) => H - pad.b - (v / peak) * (H - pad.t - pad.b);

  // A baseline and a scale, so the numbers are readable rather than decorative.
  ctx.strokeStyle = '#1a2130';
  ctx.fillStyle = '#5d6a80';
  ctx.font = '10px ui-monospace, monospace';
  for (let g = 0; g <= 4; g += 1) {
    const v = (peak * g) / 4;
    const py = Math.round(y(v)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(pad.l, py);
    ctx.lineTo(W - pad.r, py);
    ctx.stroke();
    ctx.fillText(fmt(v), 8, py + 3);
  }

  const line = (series, colour) => {
    ctx.beginPath();
    for (let i = 0; i < series.length; i += 1) {
      const px = x(i + (n - series.length));
      const py = y(series[i]);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.strokeStyle = colour;
    ctx.lineWidth = 2;
    ctx.stroke();
  };
  line(red, '#eb4e46');
  line(blue, '#4e96f5');

  const lastRed = red[red.length - 1] ?? 0;
  const lastBlue = blue[blue.length - 1] ?? 0;
  ctx.font = '13px ui-monospace, monospace';
  ctx.fillStyle = '#eb4e46';
  ctx.fillText(`red  ${fmt(lastRed)}`, pad.l, 26);
  ctx.fillStyle = '#4e96f5';
  ctx.fillText(`blue ${fmt(lastBlue)}`, pad.l + 140, 26);
  ctx.fillStyle = '#8b97ad';
  ctx.font = '11px ui-monospace, monospace';
  ctx.fillText(
    'Men still on the field. The gap between the lines is the battle.',
    pad.l,
    H - pad.b + 24,
  );
}

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
$('view-world').addEventListener('click', () => setView(false));
$('view-tree').addEventListener('click', () => setView(true));

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
      renderer.upload(
        new Uint8Array(msg.buffer, 0, msg.count * msg.stride),
        msg.count,
        msg.plants,
        msg.worldSize,
      );
      // Hand the buffer straight back so the worker can reuse it.
      worker.postMessage({ type: 'recycle', buffer: msg.buffer }, [msg.buffer]);
      latest = msg;
      // Nothing stops the engine on its own any more, but it does start itself
      // when the page is built to autoplay. The button follows it rather than
      // the other way round, so the two cannot disagree.
      if (typeof msg.running === 'boolean' && msg.running !== running) {
        showRunning(msg.running);
      }
      history.push({
        red: msg.red,
        blue: msg.blue,
        red_holding: msg.stats.red_holding,
        blue_holding: msg.stats.blue_holding,
      });
      updatePanel(msg.stats, msg.red, msg.blue);
      $('h-ms').textContent = msg.tickMs ? msg.tickMs.toFixed(1) : '–';

      latestTick = msg.stats.tick;
      $('t-tick').textContent = fmt(latestTick);
      // Report how it stands, and keep running. The clock used to stop here,
      // which is what made a battle look hung: a mutual collapse counted as a
      // decision, so the moment both lines gave way the page froze on a field
      // still covered in running men. The pursuit is where most of the killing
      // happens and it is the half worth watching, so the battle ends when the
      // viewer pauses it, not when the engine decides it has seen enough.
      $('tree-summary').textContent = OUTCOME[msg.outcome] ?? OUTCOME[0];
      break;
    }
    case 'terrain':
      // The ground arrives once per battle rather than per frame, so it is
      // uploaded here and then simply drawn under every subsequent frame.
      renderer.uploadTerrain(new Uint8Array(msg.buffer), msg.dim);
      break;
    case 'inspected':
      showInspector(msg.organism);
      break;
    case 'error':
      fail(msg.message);
      break;
    default:
      break;
  }
};

// Density is held constant as the field is scaled, so the fighting behaves the
// same at every setting and only the size of the map changes. Mirrors
// Config::for_muster.
function scaleOverrides(target) {
  const byName = Object.fromEntries(params.map((p) => [p.name, p.value]));
  const base = byName.units_per_side * 2;
  const scale = target / base;
  const root = Math.sqrt(scale);
  // Nearest power of two, not the next one up. Cell size is what every per-cell
  // budget is denominated in -- whether a cell holds anybody, how many bodies
  // target selection has to scan -- so rounding up would change the combat at
  // some scales and not others.
  const want = byName.grid_dim * root;
  const lo = Math.max(1, Math.pow(2, Math.floor(Math.log2(Math.max(want, 1)))));
  const grid = Math.min(4096, Math.max(8, want >= lo * Math.SQRT2 ? lo * 2 : lo));
  return {
    units_per_side: Math.max(1, Math.round(byName.units_per_side * scale)),
    max_units: Math.round(target * 1.05) + 16,
    field_size: byName.field_size * root,
    grid_dim: grid,
  };
}

function currentSetup() {
  return {
    seed: Math.max(0, Number($('seed').value) | 0),
    overrides: scaleOverrides(Number($('scale').value)),
  };
}

let latestTick = 0;

/** Paint the play button to match whatever the engine is actually doing. */
function showRunning(next) {
  running = next;
  $('t-play').textContent = running ? '\u275a\u275a' : '\u25b6';
  $('t-play').setAttribute('aria-label', running ? 'Pause' : 'Play');
}

function setRunning(next) {
  showRunning(next);
  worker.postMessage({ type: running ? 'play' : 'pause' });
}

/** Send the speed control's current position to the engine, and label it. */
function applySpeed() {
  const value = Number($('t-speed').value);
  $('t-speed-val').textContent = String(value);
  worker.postMessage({ type: 'speed', value });
}

$('t-play').addEventListener('click', () => setRunning(!running));

$('t-step').addEventListener('click', () => {
  if (running) setRunning(false);
  worker.postMessage({ type: 'step', count: 1 });
});

$('t-speed').addEventListener('input', applySpeed);

$('reset').addEventListener('click', () => {
  running = false;
  setRunning(false);
  history.clear();
  showInspector(null);
  worker.postMessage({ type: 'reset', ...currentSetup() });
});

$('scale').addEventListener('change', () => $('reset').click());

$('color').addEventListener('change', (e) => {
  worker.postMessage({ type: 'color', value: Number(e.target.value) });
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
  if (showTree) drawReport();
  else renderer.draw(view);
  if (latest) {
    drawChart($('chart-pop'), ['red', 'blue'], ['#eb4e46', '#4e96f5']);
    drawChart($('chart-matter'), ['red_holding', 'blue_holding'], ['#eb4e46', '#4e96f5'], {
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

// Boot: get the module bytes here rather than in the engine so a failure
// surfaces with a readable message on the page.
//
// The single-file build has no file to fetch, so it leaves the bytes in
// `__borschtWasm` instead. Everything after that is the same in both builds.
(async () => {
  try {
    let wasm = globalThis.__borschtWasm;
    if (!wasm) {
      const response = await fetch('./borscht.wasm');
      if (!response.ok) {
        throw new Error(`could not load borscht.wasm (${response.status}). Run tools/build-web.sh first.`);
      }
      wasm = await response.arrayBuffer();
    }
    // The world is built inside 'init'. Posting a separate 'reset' here would
    // race the engine's asynchronous module load and arrive before there is a
    // simulation to reset.
    worker.postMessage(
      { type: 'init', wasm, autoplay: Boolean(globalThis.__borschtAutoplay), ...currentSetup() },
      [wasm],
    );
    // 'input' does not fire on load, so without this the engine keeps its own
    // default of one tick a frame however the control is set -- which made a
    // page built to open at speed look identical to one built at 1x.
    applySpeed();
  } catch (error) {
    fail(error.message);
  }
})();
