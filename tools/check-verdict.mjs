// Exercise the verdict control against a stand-in store.
//
// check-web.mjs proves the card disappears when there is nowhere to record to.
// This proves the other half: given a store, the card appears, the buttons wake
// up once the lines have met, and the document that reaches `add()` carries
// everything the trainer needs to rebuild the battle that was judged.
import { chromium } from 'playwright';
import { readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { execFileSync } from 'node:child_process';

// What the command-line build says it ships. A verdict names the commander it
// was passed on, and the whole point of that name is that the trainer can match
// it against a commander it actually has. If the module's answer and the
// binary's answer ever drift apart -- a change to the weights that only one side
// picked up, a mistake joining the two halves of the 64-bit name in JavaScript
// -- every verdict would silently attach to the wrong commander. So the two are
// compared rather than assumed equal.
let expectedCommander = null;
try {
  expectedCommander = execFileSync('./target/release/borscht', ['commander'], {
    encoding: 'utf8',
  }).trim();
} catch {
  console.log('note: no built borscht binary, skipping the commander cross-check');
}

const page_html = readFileSync(process.argv[2] ?? 'out/borscht-artifact.html', 'utf8');
const server = createServer((_, res) => {
  res.writeHead(200, { 'content-type': 'text/html' });
  res.end(`<!doctype html><html><head><meta charset="utf-8"></head><body>${page_html}</body></html>`);
}).listen(8137);

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || '/opt/pw-browsers/chromium',
  args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader'],
});
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });

// The stand-in has exactly the surface the page uses, and records what it is
// given instead of storing it.
await page.addInitScript(() => {
  window.__written = [];
  window.claude = {
    use: async (name) =>
      name === 'db'
        ? {
            collection: (path) => ({
              add: async (body) => {
                window.__written.push({ path, body });
                return { id: 'stub' };
              },
            }),
          }
        : null,
  };
});

const problems = [];
page.on('pageerror', (e) => problems.push(`page error: ${e.message}`));
await page.goto('http://127.0.0.1:8137/', { waitUntil: 'load' });

await page.waitForSelector('#verdict-card:not([hidden])', { timeout: 20000 });
const beforeContact = await page.evaluate(() => document.getElementById('verdict-good').disabled);

// A small muster at speed, because this check is about the control and not
// about how long a hundred thousand men take to meet under SwiftShader.
await page.selectOption('#scale', '20000');
await page.fill('#t-speed', '64');
await page.dispatchEvent('#t-speed', 'input');
// Changing the scale resets, and a reset pauses.
await page.waitForFunction(() => document.getElementById('t-play').textContent === '\u25b6', null, {
  timeout: 20000,
});
await page.click('#t-play');

// Run until the lines meet, which is when there is something to judge.
await page.waitForFunction(() => !document.getElementById('verdict-good').disabled, null, {
  timeout: 120000,
});

await page.fill('#verdict-note', 'left flank never closed');
await page.click('#verdict-sides .seg[data-side="1"]');
await page.click('#verdict-bad');
await page.waitForFunction(() => window.__written.length > 0, null, { timeout: 10000 });

const written = await page.evaluate(() => window.__written);
const status = await page.textContent('#verdict-status');
const noteCleared = await page.inputValue('#verdict-note');
await browser.close();
server.close();

console.log(`card appeared, buttons ${beforeContact ? 'disabled' : 'ENABLED'} before contact`);
console.log(`wrote ${written.length} document(s) to ${written[0]?.path}`);
console.log(JSON.stringify(written[0]?.body, null, 2));
console.log(`status: ${status}`);
console.log(
  `commander: page says ${body_preview(written)}, binary says ${expectedCommander ?? 'n/a'}`,
);

const body = written[0]?.body ?? {};
function body_preview(w) {
  return w[0]?.body?.commander ?? 'nothing';
}
const failures = [...problems];
if (!beforeContact) failures.push('a verdict could be recorded before the armies met');
if (written[0]?.path !== 'verdicts') failures.push(`wrote to ${written[0]?.path}, not verdicts`);
for (const key of ['verdict', 'side', 'seed', 'overrides', 'commander', 'tick', 'at']) {
  if (body[key] === undefined || body[key] === null) failures.push(`no ${key} in the document`);
}
if (body.side !== 'blue') failures.push(`side is ${body.side}, not the one selected`);
if (body.verdict !== 'badly fought') failures.push(`verdict is ${body.verdict}`);
if (body.note !== 'left flank never closed') failures.push('the note did not travel');
if (!body.overrides?.units_per_side) failures.push('the overrides cannot rebuild the battle');
if (noteCleared !== '') failures.push('the note field was not cleared after recording');
if (!/recorded/.test(status ?? '')) failures.push(`status never confirmed: ${status}`);
if (expectedCommander && body.commander !== expectedCommander) {
  failures.push(
    `the page names commander ${body.commander}, the binary ships ${expectedCommander}`,
  );
}

if (failures.length) {
  console.error(`\nFAILED:\n  ${failures.join('\n  ')}`);
  process.exit(1);
}
console.log('\nOK');
