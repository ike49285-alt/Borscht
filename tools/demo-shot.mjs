// Capture a battle at the moment it turns.
//
// A mass battle looks like almost nothing for four hundred ticks and then
// decides itself in about a hundred and fifty, so a screenshot taken at a fixed
// time is usually of two blocks standing still. This watches the routing share
// and fires when the line actually gives.
//
// Env: SEED, SCALE (men), COLOR (team|kind|health|morale), ZOOM (wheel notches).
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
const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
await page.goto(`http://127.0.0.1:${PORT}/index.html`, { waitUntil: 'load' });
await page.waitForFunction(() => document.getElementById('h-plants').textContent !== '0', null, { timeout: 30000 });

const read = async () => ({
  tick: (await page.textContent('#h-tick')).trim(),
  red: (await page.textContent('#h-plants')).trim(),
  blue: (await page.textContent('#h-animals')).trim(),
  holding: (await page.textContent('#h-species')).trim(),
  routing: (await page.textContent('#h-carn')).trim(),
});
const routingShare = async () =>
  parseFloat((await page.textContent('#h-carn')).replace('%', '')) || 0;

await page.selectOption('#scale', process.env.SCALE ?? '20000');
await page.waitForTimeout(800);
await page.fill('#seed', process.env.SEED ?? '5');
await page.selectOption('#color', { team: '0', kind: '1', health: '2', morale: '3' }[process.env.COLOR ?? 'morale']);
await page.click('#reset');
await page.waitForTimeout(800);
await page.fill('#t-speed', '8');
await page.dispatchEvent('#t-speed', 'input');
await page.click('#t-play');

// The line holding, just before it gives.
const deadline = Date.now() + 240000;
let held = false;
while (Date.now() < deadline) {
  const share = await routingShare();
  if (share > 2) break;
  if (!held && Number((await page.textContent('#h-tick')).replace(/[^0-9.]/g, '')) > 300) {
    await page.locator('#stage').screenshot({ path: `${OUT}/line.png` });
    console.log('holding:', await read());
    held = true;
  }
  await page.waitForTimeout(400);
}
await page.locator('#stage').screenshot({ path: `${OUT}/breaking.png` });
console.log('breaking:', await read());

// And once it has gone.
while (Date.now() < deadline) {
  if ((await routingShare()) > 60) break;
  await page.waitForTimeout(300);
}
await page.locator('#stage').screenshot({ path: `${OUT}/rout.png` });
console.log('routed:', await read());

await browser.close();
server.kill();
