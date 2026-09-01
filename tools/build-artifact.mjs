// Build the viewer as one self-contained HTML file.
//
// The published Artifact is a single page under a policy that admits no
// external resources: no second file to load a Worker from, and nothing to
// fetch the WebAssembly module from. So the module is embedded as base64 and
// the engine runs on the main thread -- which costs nothing now that the
// largest world the viewer offers is 60k organisms at about 1.3 ms a tick.
//
// Everything is generated from web/, never hand-maintained, so the bundle
// cannot drift from the app the browser check actually tests. Every
// substitution below asserts that it matched: a silent no-op replacement
// produces a page that looks right and quietly runs the wrong world.
//
// Usage: node tools/build-artifact.mjs [--out FILE] [--seed N] [--scale N] [--speed N]
import { readFileSync, writeFileSync } from 'node:fs';

const root = new URL('..', import.meta.url).pathname;
const arg = (name, fallback) => {
  const at = process.argv.indexOf(`--${name}`);
  return at >= 0 && process.argv[at + 1] ? process.argv[at + 1] : fallback;
};
const OUT = arg('out', `${root}out/borscht-artifact.html`);
const SEED = arg('seed', '7');
const SCALE = arg('scale', '100000');
// One tick a frame. A battle at a hundred thousand men already costs several
// milliseconds a tick, so the frame is the budget rather than the tick, and
// asking for more per frame buys nothing but a chunkier picture. The control
// goes to 64 for anyone watching a small muster.
const SPEED = arg('speed', '1');

const read = (f) => readFileSync(`${root}web/${f}`, 'utf8');

/** Replace once, and fail loudly if the pattern has moved. */
function sub(text, from, to, what) {
  const count = text.split(from).length - 1;
  if (count !== 1) {
    throw new Error(`${what}: expected exactly one match, found ${count}. web/ has changed shape.`);
  }
  return text.replace(from, to);
}

// ---------------------------------------------------------------- markup --

let html = read('index.html');

// The Artifact host supplies the document skeleton, so keep only what sits
// between <body> and </body>, plus the <style> and <title> from the head.
const styleAt = html.indexOf('<style>');
const styleEnd = html.indexOf('</style>') + '</style>'.length;
if (styleAt < 0 || styleEnd < styleAt) throw new Error('no <style> block in index.html');
const style = html.slice(styleAt, styleEnd);
const bodyAt = html.indexOf('<body>') + '<body>'.length;
const bodyEnd = html.indexOf('</body>');
if (bodyAt < 6 || bodyEnd < bodyAt) throw new Error('no <body> in index.html');
let body = html.slice(bodyAt, bodyEnd);

// The module tag is replaced by the inlined bundle at the end.
body = sub(body, /\s*<script type="module" src="\.\/app\.js"><\/script>/, '', 'module script tag');

// Opening world.
body = sub(
  body,
  '<input id="seed" type="number" value="1"',
  `<input id="seed" type="number" value="${SEED}"`,
  'seed default',
);
// Clear whatever the page selects by default, then select the requested scale.
// Doing it in that order means the builder does not care whether the two happen
// to agree, which they did not the moment web/ was rescaled.
const selectedCount = body.split(' selected>').length - 1;
if (selectedCount !== 1) {
  throw new Error(`expected exactly one selected option in index.html, found ${selectedCount}`);
}
body = body.replace(' selected>', '>');
body = sub(
  body,
  `<option value="${SCALE}">`,
  `<option value="${SCALE}" selected>`,
  `scale option ${SCALE}`,
);
body = sub(
  body,
  'id="t-speed" type="range" min="1" max="64" value="1"',
  `id="t-speed" type="range" min="1" max="64" value="${SPEED}"`,
  'speed default',
);

// ------------------------------------------------------------------ code --

// Concatenation order resolves the imports, so the import statements go.
const strip = (src) =>
  src
    .replace(/^\s*import\s[^;]*;\s*$/gm, '')
    .replace(/^export\s+(const|let|function|class)\s/gm, '$1 ')
    .replace(/^export\s*\{[^}]*\};?\s*$/gm, '');

const params = strip(read('params.js'));
const borscht = strip(read('borscht.js'));
const renderer = strip(read('renderer.js'));
const engine = strip(read('worker.js'));
const app = strip(read('app.js'));

const wasm = readFileSync(`${root}web/borscht.wasm`).toString('base64');

// The shim standing in for a Worker. Same surface as the real one, so app.js
// does not know or care which it is talking to.
const host = `
// ---- engine host -------------------------------------------------------
//
// A Worker's interface, served from this thread. The transfer list every
// postMessage carries is ignored: there is no boundary to transfer across, and
// the 'recycle' message already returns frame buffers to the pool, so nothing
// about the allocation behaviour changes.
const __engineHost = {
  onmessage: null,
  postMessage(msg) { __engine.handle(msg); },
  addEventListener(type, fn) { if (type === 'message') this.onmessage = fn; },
};
const __engine = createEngine({
  post: (msg) => __engineHost.onmessage?.({ data: msg }),
  // requestAnimationFrame, not setTimeout: on the main thread a yielding timer
  // loop would starve the renderer, and here the two want to interleave.
  schedule: (fn) => requestAnimationFrame(fn),
});
globalThis.__borschtEngineHost = __engineHost;
globalThis.__borschtAutoplay = true;

// The module, inlined. A published page has nothing to fetch it from.
globalThis.__borschtWasm = Uint8Array.from(atob(__WASM_B64), (c) => c.charCodeAt(0)).buffer;

if (typeof WebAssembly !== 'object') {
  document.getElementById('banner').textContent =
    'This browser will not run WebAssembly, which is what the simulation is. Nothing below will move.';
  document.getElementById('banner').style.display = 'block';
}
`;

const script = [
  '<script type="module">',
  `const __WASM_B64 = "${wasm}";`,
  params,
  borscht,
  renderer,
  engine,
  host,
  app,
  '</' + 'script>',
].join('\n');

// The host's <head> carries only a charset, a viewport tag and a small reset,
// so the title and the stylesheet are ours to supply. The title is the name in
// the browser tab and the artifact gallery, and it must not change between
// deploys or the page reads as a different one.
const title = '<title>Borscht</title>';

writeFileSync(OUT, `${title}\n${style}\n${body}\n${script}\n`);
const kb = (Buffer.byteLength(readFileSync(OUT)) / 1024).toFixed(0);
console.log(`wrote ${OUT}  ${kb} KiB  (seed ${SEED}, scale ${SCALE}, speed ${SPEED}x)`);
