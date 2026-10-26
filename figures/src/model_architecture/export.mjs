// Render index.html in headless Chrome, run layout checks, and export
// figures/model_architecture.{svg,pdf,png}.
//   node export.mjs            -> checks + export
//   node export.mjs --preview  -> checks + preview.png only
import { chromium } from 'playwright-core';
import { readFileSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const outDir = resolve(here, '../..');
const preview = process.argv.includes('--preview');
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

const browser = await chromium.launch({ executablePath: CHROME });
const page = await browser.newPage({ deviceScaleFactor: 4 });
page.on('console', m => console.log('[page]', m.text()));
page.on('pageerror', async e => { console.error('[page error]', e.stack || e.message); await browser.close(); process.exit(1); });
await page.goto(pathToFileURL(resolve(here, 'index.html')).href);
await page.waitForFunction(() => window.FIGURE_READY === true, null, { timeout: 30000 });

// ---------- layout checks (all in figure units = pt) ----------
const report = await page.evaluate(() => {
  const svg = document.getElementById('fig');
  const k = svg.getScreenCTM().a;                       // px per pt
  const o = svg.getBoundingClientRect();
  const toPt = r => ({ x: (r.left - o.left) / k, y: (r.top - o.top) / k, r: (r.right - o.left) / k, b: (r.bottom - o.top) / k });
  const issues = [];
  const lines = [...svg.querySelectorAll('g.tl')].map(g => ({ g, bb: toPt(g.getBoundingClientRect()), txt: g.textContent || '(math)' }));
  const name = l => (l.txt.trim() || [...l.g.querySelectorAll('.math')].length + ' math').slice(0, 40);
  // 1. text inside its box
  for (const l of lines) {
    const id = l.g.dataset.box; if (!id) continue;
    const b = toPt(document.getElementById(id).getBoundingClientRect());
    const tol = 0.6;
    if (l.bb.x < b.x + tol || l.bb.r > b.r - tol || l.bb.y < b.y + tol - 1 || l.bb.b > b.b - tol + 1)
      issues.push(`clip: "${name(l)}" [${l.bb.x.toFixed(1)},${l.bb.r.toFixed(1)}] in box ${id} [${b.x.toFixed(1)},${b.r.toFixed(1)}]`);
  }
  // 2. text lines overlapping each other
  // line boxes include the font's full ascent/descent, so stacked lines at 8.4 pt pitch
  // touch by design; require > 1.6 pt of vertical intrusion to count as a collision
  const inter = (a, b, mx, my) => a.x < b.r - mx && b.x < a.r - mx && a.y < b.b - my && b.y < a.b - my;
  for (let i = 0; i < lines.length; i++) for (let j = i + 1; j < lines.length; j++)
    if (inter(lines[i].bb, lines[j].bb, 0.3, 0.8)) issues.push(`overlap: "${name(lines[i])}" / "${name(lines[j])}"`);
  // 3. arrows crossing text
  const segs = [];
  for (const p of svg.querySelectorAll('path.arrow')) {
    const pts = p.getAttribute('d').slice(1).split('L').map(s => s.split(',').map(Number));
    for (let i = 1; i < pts.length; i++) segs.push([pts[i - 1], pts[i]]);
  }
  for (const l of lines) for (const [[x1, y1], [x2, y2]] of segs) {
    const bb = l.bb, m = 0.4;
    const sx = [Math.min(x1, x2), Math.max(x1, x2)], sy = [Math.min(y1, y2), Math.max(y1, y2)];
    if (sx[1] > bb.x + m && sx[0] < bb.r - m && sy[1] > bb.y + m && sy[0] < bb.b - m)
      issues.push(`arrow crosses text: "${name(l)}"`);
  }
  // 4. font sizes (text and math glyphs)
  let minText = Infinity, minMath = Infinity;
  for (const t of svg.querySelectorAll('text')) minText = Math.min(minText, parseFloat(t.getAttribute('font-size')));
  for (const p of svg.querySelectorAll('svg.math path[data-c]')) {
    const s = p.getScreenCTM().a / k * 1000; minMath = Math.min(minMath, s);
  }
  // 5. everything inside canvas
  const W = window.FIGURE.W, H = window.FIGURE.H;
  for (const l of lines) if (l.bb.x < 0 || l.bb.y < 0 || l.bb.r > W || l.bb.b > H) issues.push(`outside canvas: "${name(l)}"`);
  return { issues, minText, minMath: +minMath.toFixed(2), W, H };
});
console.log(`size ${report.W} x ${report.H} pt (${(report.W / 72).toFixed(2)} x ${(report.H / 72).toFixed(2)} in)`);
console.log(`min text size ${report.minText} pt, min math glyph size ${report.minMath} pt`);
console.log(report.issues.length ? report.issues.join('\n') : 'layout checks: OK');

const { W, H } = report;
await page.setViewportSize({ width: Math.ceil(W * 4 / 3), height: Math.ceil(H * 4 / 3) });
if (preview) {
  await page.locator('#fig').screenshot({ path: resolve(here, 'preview.png') });
  await browser.close();
  process.exit(0);
}

// ---------- standalone SVG with embedded fonts ----------
const font = (w, s) => readFileSync(resolve(here, `node_modules/@fontsource/inter/files/inter-latin-${w}-${s}.woff2`)).toString('base64');
const fontCss = [['400', 'normal'], ['400', 'italic'], ['600', 'normal']].map(([w, s]) =>
  `@font-face{font-family:'Inter';font-weight:${w};font-style:${s};src:url(data:font/woff2;base64,${font(w, s)}) format('woff2');}`).join('');
const svgText = await page.evaluate(css => {
  const svg = document.getElementById('fig').cloneNode(true);
  svg.removeAttribute('id');
  svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  const st = document.createElementNS('http://www.w3.org/2000/svg', 'style');
  st.textContent = css + 'text{white-space:pre}';
  svg.insertBefore(st, svg.firstChild);
  return new XMLSerializer().serializeToString(svg);
}, fontCss);
const svgPath = resolve(outDir, 'model_architecture.svg');
writeFileSync(svgPath, '<?xml version="1.0" encoding="UTF-8"?>\n' + svgText);

// ---------- PDF from the standalone SVG ----------
const p2 = await browser.newPage();
await p2.setContent(`<!DOCTYPE html><html><head><style>@page{size:${W}pt ${H}pt;margin:0}html,body{margin:0;padding:0}svg{display:block}</style></head><body>${svgText}</body></html>`);
await p2.evaluate(() => document.fonts.ready);
const pdfPath = resolve(outDir, 'model_architecture.pdf');
await p2.pdf({ path: pdfPath, width: `${W / 72}in`, height: `${H / 72}in`, printBackground: true, preferCSSPageSize: true,
  margin: { top: 0, right: 0, bottom: 0, left: 0 } });
await browser.close();

// ---------- PNG at 300 dpi from the PDF ----------
execFileSync('pdftoppm', ['-r', '300', '-png', '-singlefile', pdfPath, resolve(outDir, 'model_architecture')]);
console.log('wrote', svgPath, pdfPath, resolve(outDir, 'model_architecture.png'));
