// Stage 6: walk the demo path in a headless browser and save figures/ui.png.
// Usage (API running on :8000): node tgn_trace/ui/screenshot.js [url] [out.png]
const { chromium } = require('playwright');

(async () => {
  const url = process.argv[2] || 'http://127.0.0.1:8000/';
  const out = process.argv[3] || 'figures/ui.png';
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto(url);
  await page.waitForSelector('#agreement .grid2', { timeout: 300000 });
  // select greedy explainer, wait for it
  await page.selectOption('#explSel', 'greedy');
  await page.waitForFunction(() => document.querySelector('#explStatus').textContent.startsWith('greedy: status'), null, { timeout: 300000 });
  // select the top-ranked candidate and the second one, remove both, replay
  for (const i of [1, 2]) {
    await page.click(`#candTable tbody tr:nth-child(${i})`);
    await page.click('#btnToggle');
  }
  await page.click('#candTable tbody tr:nth-child(1)');
  await page.click('#btnReplay');
  await page.waitForSelector('#cfOut .cf', { timeout: 300000 });
  await page.screenshot({ path: out, fullPage: true });
  const summary = await page.evaluate(() => ({
    target: document.querySelector('#hUV').textContent,
    base: document.querySelector('#hBase').textContent,
    expl: document.querySelector('#explStatus').textContent,
    flag: document.querySelector('#flagBadge').textContent,
    cf: document.querySelector('#cfOut').innerText.replace(/\s+/g, ' ').slice(0, 400),
  }));
  console.log(JSON.stringify({ summary, errors }, null, 2));
  await browser.close();
})();
