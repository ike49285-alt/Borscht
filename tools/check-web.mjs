// Drive the real page in a real browser.
//
// WebGL2 works in headless Chromium via SwiftShader, so this exercises the
// whole path -- module load, worker, shader compile, buffer upload, draw -- and
// not just the parts that are convenient to unit test.

import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { mkdirSync, readFileSync } from 'node:fs';
import { inflateSync } from 'node:zlib';

/**
 * Decode a truecolour PNG to raw RGB.
 *
 * The screenshot is the authoritative evidence that something was drawn.
 * `readPixels` cannot be used for this: the context is created with
 * `preserveDrawingBuffer: false`, so outside the frame that drew it the buffer
 * reads back empty and every run would look like a blank canvas.
 */
function decodePng(bytes) {
  let at = 8; // skip signature
  let width = 0;
  let height = 0;
  let channels = 3;
  const idat = [];
  while (at < bytes.length) {
    const len = bytes.readUInt32BE(at);
    const kind = bytes.toString('ascii', at + 4, at + 8);
    const body = bytes.subarray(at + 8, at + 8 + len);
    if (kind === 'IHDR') {
      width = body.readUInt32BE(0);
      height = body.readUInt32BE(4);
      const colorType = body[9];
      if (body[8] !== 8) throw new Error('expected 8-bit PNG');
      channels = colorType === 6 ? 4 : colorType === 2 ? 3 : 0;
      if (!channels) throw new Error(`unsupported PNG colour type ${colorType}`);
    } else if (kind === 'IDAT') {
      idat.push(body);
    } else if (kind === 'IEND') {
      break;
    }
    at += 12 + len;
  }
  const raw = inflateSync(Buffer.concat(idat));
  const stride = width * channels;
  const out = Buffer.alloc(width * height * channels);
  let prev = Buffer.alloc(stride);
  for (let y = 0; y < height; y += 1) {
    const filter = raw[y * (stride + 1)];
    const line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    const cur = Buffer.alloc(stride);
    for (let i = 0; i < stride; i += 1) {
      const a = i >= channels ? cur[i - channels] : 0;
      const b = prev[i];
      const c = i >= channels ? prev[i - channels] : 0;
      let value = line[i];
      if (filter === 1) value += a;
      else if (filter === 2) value += b;
      else if (filter === 3) value += (a + b) >> 1;
      else if (filter === 4) {
        const p = a + b - c;
        const pa = Math.abs(p - a);
        const pb = Math.abs(p - b);
        const pc = Math.abs(p - c);
        value += pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      }
      cur[i] = value & 0xff;
    }
    cur.copy(out, y * stride);
    prev = cur;
  }
  return { width, height, channels, data: out };
}

/** Fraction of the simulation viewport that is not background. */
function litFraction(path, region) {
  const png = decodePng(readFileSync(path));
  const { width, channels, data } = png;
  let lit = 0;
  let total = 0;
  for (let y = region.y0; y < region.y1; y += 2) {
    for (let x = region.x0; x < region.x1; x += 2) {
      const o = (y * width + x) * channels;
      total += 1;
      // Background is (6, 8, 14).
      if (data[o] > 20 || data[o + 1] > 24 || data[o + 2] > 32) lit += 1;
    }
  }
  return { lit, total, fraction: lit / Math.max(total, 1) };
}

const PORT = 8099;
const OUT = new URL('../out/web/', import.meta.url).pathname;
mkdirSync(OUT, { recursive: true });

const server = spawn('python3', ['-m', 'http.server', '-d', 'web', String(PORT), '--bind', '127.0.0.1'], {
  cwd: new URL('..', import.meta.url).pathname,
  stdio: 'ignore',
});
const stop = () => server.kill();
process.on('exit', stop);

// Wait for the server to accept connections.
for (let i = 0; i < 50; i += 1) {
  try {
    const r = await fetch(`http://127.0.0.1:${PORT}/index.html`);
    if (r.ok) break;
  } catch {
    await new Promise((r) => setTimeout(r, 100));
  }
}

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || '/opt/pw-browsers/chromium',
  args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader'],
});
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });

const problems = [];
page.on('console', (m) => {
  if (m.type() === 'error') problems.push(`console: ${m.text()}`);
});
page.on('pageerror', (e) => problems.push(`pageerror: ${e.message}`));

await page.goto(`http://127.0.0.1:${PORT}/index.html`, { waitUntil: 'load' });

/// The HUD abbreviates large numbers ("1.2k"), so stripping non-digits turns a
/// thousand into a one. Parse the suffix rather than discarding it.
function parseReadout(text) {
  const m = /([0-9.]+)\s*([kM])?/.exec(text ?? '');
  if (!m) return NaN;
  const scale = m[2] === 'k' ? 1e3 : m[2] === 'M' ? 1e6 : 1;
  return Number(m[1]) * scale;
}

const readTick = () => page.textContent('#h-tick');
const readAnimals = () => page.textContent('#h-animals');

// The world must populate before anything is asked to run.
await page.waitForFunction(() => document.getElementById('h-plants').textContent !== '0', null, {
  timeout: 30000,
});

const banner = await page.$eval('#banner', (el) => (el.style.display === 'none' ? '' : el.textContent));
if (banner) throw new Error(`page reported: ${banner}`);

await page.click('#t-play');
await page.waitForFunction(() => Number(document.getElementById('h-tick').textContent) > 5, null, {
  timeout: 30000,
});

// Let it run long enough for the charts to have a shape.
await page.waitForTimeout(6000);
const tickAfter = await readTick();
await page.screenshot({ path: `${OUT}/running.png` });

// The world square is centred in the stage; sample inside it so the letterboxed
// margins do not dilute the measurement.
const stage = await page.$eval('#stage', (el) => {
  const r = el.getBoundingClientRect();
  return { w: r.width, h: r.height };
});
const side = Math.min(stage.w, stage.h);
const region = {
  x0: Math.round((stage.w - side) / 2 + side * 0.1),
  x1: Math.round((stage.w - side) / 2 + side * 0.9),
  y0: Math.round((stage.h - side) / 2 + side * 0.1),
  y1: Math.round((stage.h - side) / 2 + side * 0.9),
};
const drawn = litFraction(`${OUT}/running.png`, region);

await page.click('#t-play'); // pause
const fps = await page.textContent('#h-fps');
const ms = await page.textContent('#h-ms');
const species = await page.textContent('#h-species');
const carn = await page.textContent('#h-carn');

// The tree of life must render real branches, not an empty canvas.
await page.click('#view-tree');
await page.waitForTimeout(900);
const treeDrawn = await page.evaluate(() => {
  const c = document.getElementById('tree');
  const ctx = c.getContext('2d');
  const { data } = ctx.getImageData(0, 0, c.width, c.height);
  let lit = 0;
  for (let i = 0; i < data.length; i += 4) {
    // Tree background is (6, 8, 14).
    if (data[i] > 24 || data[i + 1] > 28 || data[i + 2] > 36) lit += 1;
  }
  return { lit, total: data.length / 4 };
});
await page.screenshot({ path: `${OUT}/tree.png` });
const treeSummary = await page.textContent('#tree-summary');
await page.click('#view-world');
await page.waitForTimeout(300);

// Rewind: drag the timeline back and confirm the world actually goes back.
// Read the scrubber, which carries the raw tick rather than an abbreviation.
const tickBefore = Number(await page.inputValue('#t-scrub'));
const scrub = await page.$('#t-scrub');
const box = await scrub.boundingBox();
await page.mouse.move(box.x + box.width * 0.9, box.y + box.height / 2);
await page.mouse.down();
await page.mouse.move(box.x + box.width * 0.1, box.y + box.height / 2, { steps: 8 });
await page.mouse.up();
await page.waitForTimeout(1500);
const tickAfterSeek = Number(await page.inputValue('#t-scrub'));

// Colour modes must all render without error.
for (const mode of ['1', '2', '3', '4']) {
  await page.selectOption('#color', mode);
  await page.waitForTimeout(250);
}
await page.selectOption('#color', '0');

// Zoom in and click something to exercise the inspector.
await page.mouse.move(700, 450);
await page.mouse.wheel(0, -900);
await page.waitForTimeout(400);
// A world can be sparse, so try a few points rather than assuming an organism
// sits under one particular pixel.
let inspected = false;
for (const [x, y] of [[700, 450], [640, 400], [760, 500], [700, 380], [620, 520]]) {
  await page.mouse.click(x, y);
  await page.waitForTimeout(400);
  inspected = await page.$eval('#inspector', (el) => el.style.display !== 'none');
  if (inspected) break;
}
await page.screenshot({ path: `${OUT}/zoomed.png` });

console.log(`ticks:      ${tickAfter}`);
console.log(`animals:    ${await readAnimals()}`);
console.log(`species:    ${species}   carnivores: ${carn}`);
console.log(`ms/tick:    ${ms}    fps: ${fps}`);
console.log(`pixels lit: ${(drawn.fraction * 100).toFixed(1)}% of the world square`);
console.log(`inspector:  ${inspected ? 'opened' : 'did not open'}`);
console.log(`tree:       ${((treeDrawn.lit / treeDrawn.total) * 100).toFixed(1)}% drawn — ${treeSummary}`);
console.log(`rewind:     tick ${tickBefore} -> ${tickAfterSeek}`);

await browser.close();
stop();

const failures = [];
if (parseReadout(tickAfter) <= 5) failures.push('simulation did not advance');
// A blank-canvas detector, not a density assertion: how much of the square is
// lit depends on the world size, the zoom and how well the run happens to be
// doing, none of which this check is here to police.
if (drawn.fraction < 0.01) {
  failures.push(`canvas is effectively empty (${(drawn.fraction * 100).toFixed(2)}% lit)`);
}
if (!inspected) failures.push('inspector did not open on click');
if (treeDrawn.lit < treeDrawn.total * 0.002) failures.push('tree of life rendered nothing');
if (!(tickAfterSeek < tickBefore)) {
  failures.push(`rewind did not go back: ${tickBefore} -> ${tickAfterSeek}`);
}
if (problems.length) failures.push(...problems);
if (failures.length) {
  console.error(`\nFAILED:\n  ${failures.join('\n  ')}`);
  process.exit(1);
}
console.log('\nOK');
