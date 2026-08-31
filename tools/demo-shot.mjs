// Capture the viewer after it has run long enough to look like an ecosystem.
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
const page = await browser.newPage({ viewport: { width: 1500, height: 940 } });
await page.goto(`http://127.0.0.1:${PORT}/index.html`, { waitUntil: 'load' });
await page.waitForFunction(() => document.getElementById('h-plants').textContent !== '0', null, { timeout: 30000 });

await page.selectOption('#scale', '50000');
await page.waitForTimeout(1500);
await page.fill('#seed', '4');
await page.click('#reset');
await page.waitForTimeout(1200);
await page.fill('#speed', '24');
await page.dispatchEvent('#speed', 'input');
await page.click('#play');

// Long enough for species to radiate and structure to form.
const deadline = Date.now() + 150000;
while (Date.now() < deadline) {
  const tick = Number((await page.textContent('#h-tick')).replace(/[^0-9.]/g, '')) * ((await page.textContent('#h-tick')).includes('k') ? 1000 : 1);
  if (tick > 9000) break;
  await page.waitForTimeout(2000);
}
await page.click('#play');
await page.waitForTimeout(800);
await page.screenshot({ path: `${OUT}/world.png` });

console.log('tick    ', await page.textContent('#h-tick'));
console.log('plants  ', await page.textContent('#h-plants'));
console.log('animals ', await page.textContent('#h-animals'));
console.log('species ', await page.textContent('#h-species'));
console.log('carn    ', await page.textContent('#h-carn'));

await browser.close();
server.kill();
