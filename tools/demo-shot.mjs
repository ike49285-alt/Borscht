// Capture the viewer after it has run long enough to look like an ecosystem.
//
// A freshly seeded world is uniform noise: herds, patchiness and species
// clustering only appear after thousands of ticks, so this runs the page
// forward before taking the picture. Writes four images -- the world square,
// the whole page, the tree of life, and the world again after the biomass
// control has taken most of the matter out, which is the only way to see what
// that control actually does.
//
// Env: SEED, SCALE (organisms), UNTIL (ticks to run to), STARVE (biomass factor),
//      MINPEAK (tree threshold slider step, 1-30).
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { mkdirSync } from 'node:fs';

const PORT = 8098;
const OUT = new URL('../out/web/', import.meta.url).pathname;
mkdirSync(OUT, { recursive: true });
const server = spawn('python3', ['-m', 'http.server', '-d', 'web', String(PORT), '--bind', '127.0.0.1'], {
  cwd: new URL('..', import.meta.url).pathname, stdio: 'ignore',
});
process.on('exit', () => server.kill());
for (let i = 0; i < 50; i += 1) {
  try { if ((await fetch(`http://127.0.0.1:${PORT}/index.html`)).ok) break; } catch { await new Promise(r => setTimeout(r, 100)); }
}

const browser = await chromium.launch({
  executablePath: '/opt/pw-browsers/chromium',
  args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader'],
});
const page = await browser.newPage({ viewport: { width: 1500, height: 1000 } });
await page.goto(`http://127.0.0.1:${PORT}/index.html`, { waitUntil: 'load' });
await page.waitForFunction(() => document.getElementById('h-plants').textContent !== '0', null, { timeout: 30000 });

const tick = async () => {
  const text = await page.textContent('#h-tick');
  return Number(text.replace(/[^0-9.]/g, '')) * (text.includes('k') ? 1000 : 1);
};
const readout = async () => ({
  tick: await page.textContent('#h-tick'),
  plants: await page.textContent('#h-plants'),
  animals: await page.textContent('#h-animals'),
  species: await page.textContent('#h-species'),
  carn: await page.textContent('#h-carn'),
});

await page.selectOption('#scale', process.env.SCALE ?? '60000');
await page.waitForTimeout(1200);
await page.fill('#seed', process.env.SEED ?? '3');
await page.click('#reset');
await page.waitForTimeout(1200);
await page.fill('#t-speed', '32');
await page.dispatchEvent('#t-speed', 'input');
await page.click('#t-play');

// Long enough for species to radiate and structure to form.
const until = Number(process.env.UNTIL ?? 9000);
const deadline = Date.now() + 240000;
while (Date.now() < deadline) {
  if (await tick() > until) break;
  await page.waitForTimeout(2000);
}
await page.click('#t-play');
await page.waitForTimeout(800);

console.log('after %d ticks:', await tick(), await readout());
await page.locator('#stage').screenshot({ path: `${OUT}/stage.png` });
await page.screenshot({ path: `${OUT}/world.png` });

// Zoomed in, which is where individual creatures either read as bodies or do
// not. The whole-world view cannot answer that: at that scale everything is a
// pixel whatever it is drawn as.
await page.mouse.move(700, 500);
// Each notch is about 1.2x, so a dozen lands near 10x -- a view about twenty
// world units across, which holds a few dozen organisms. Far more than that and
// the view is usually empty space between plants.
const NOTCHES = Number(process.env.ZOOM ?? 12);
for (let i = 0; i < NOTCHES; i += 1) {
  await page.mouse.wheel(0, -120);
  await page.waitForTimeout(30);
}
await page.waitForTimeout(900);
await page.locator('#stage').screenshot({ path: `${OUT}/zoomed.png` });
for (let i = 0; i < NOTCHES; i += 1) await page.mouse.wheel(0, 120);
await page.waitForTimeout(600);

await page.click('#view-tree');
if (process.env.MINPEAK) {
  await page.fill('#tree-min', process.env.MINPEAK);
  await page.dispatchEvent('#tree-min', 'input');
}
await page.waitForTimeout(1500);
await page.screenshot({ path: `${OUT}/tree.png` });
console.log('tree:', await page.textContent('#tree-summary'));
await page.click('#view-world');
await page.waitForTimeout(400);

// Now take the matter out and let the world run on with it gone.
const starve = process.env.STARVE ?? '0.35';
await page.fill('#matter', starve);
await page.dispatchEvent('#matter', 'input');
await page.waitForTimeout(800);
console.log('at %s biomass:', starve, await readout());
await page.click('#t-play');
await page.waitForTimeout(12000);
await page.click('#t-play');
await page.waitForTimeout(600);
console.log('after running on:', await readout());
await page.locator('#stage').screenshot({ path: `${OUT}/starved.png` });

await browser.close();
server.kill();
