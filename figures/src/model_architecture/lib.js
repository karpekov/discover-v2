'use strict';
// Drawing helpers for the architecture figure (layout lives in figure.js).
// All units are PostScript points (1/72 in); the figure is drawn at acmsmall
// \textwidth and included at width=\linewidth.

const NS = 'http://www.w3.org/2000/svg';
const XML_NS = 'http://www.w3.org/XML/1998/namespace';

// ---------- grid ----------
const W = 394;
const GUT = 6, NCOL = 12;
const COLW = (W - (NCOL - 1) * GUT) / NCOL;
const colL = c => (c - 1) * (COLW + GUT);
const colR = c => colL(c) + COLW;
const ROW_GAP = 14;

// ---------- type ----------
const FS = 7;          // body text (minimum size)
const FS_TITLE = 8;    // stage titles
const MFS = 8;         // math base size; scripts are clamped to FS
const LH = 8.4;        // line pitch
const ASC = 5.3, DESC = 1.9;
const PAD = 3.3;       // box padding
const PPAD = 5;        // panel padding
const TITLE_H = 27;    // panel title band

// ---------- strokes / colors (Okabe-Ito) ----------
const SW = 0.6, RAD = 2.5;
const ROLE = {
  sensor: { stroke: '#0072B2', fill: '#E3EEF7' },
  lang:   { stroke: '#E69F00', fill: '#FDF0D5' },
  llm:    { stroke: '#CC79A7', fill: '#F8E6F0' },
  label:  { stroke: '#009E73', fill: '#DCF1EA' },
  det:    { stroke: '#8C8C8C', fill: '#EFEFEF' },
  plain:  { stroke: '#8C8C8C', fill: '#FFFFFF' },
  panel:  { stroke: '#C4C4C4', fill: '#FAFAFA' },
};
const INK = '#1A1A1A', ARROW = '#3A3A3A', MUTED = '#555555';

const svg = document.getElementById('fig');
let L = {};  // layers

function el(tag, attrs = {}, parent = svg) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null) continue;
    if (k === 'xml:space') e.setAttributeNS(XML_NS, k, v); else e.setAttribute(k, v);
  }
  parent.appendChild(e);
  return e;
}
const f = v => +v.toFixed(3);

// ---------- math ----------
function texNode(tex, size) {
  let mml = MathJax.tex2mml(tex, { display: false });
  mml = mml.replace(/<math([^>]*)>/, `<math$1><mstyle scriptminsize="${(FS / size).toFixed(4)}em">`)
           .replace(/<\/math>\s*$/, '</mstyle></math>');
  const src = MathJax.mathml2svg(mml, { display: false }).querySelector('svg');
  const [minX, minY, vw, vh] = src.getAttribute('viewBox').split(/\s+/).map(Number);
  const k = size / 1000;
  const out = document.createElementNS(NS, 'svg');
  out.setAttribute('viewBox', `${minX} ${minY} ${vw} ${vh}`);
  out.setAttribute('width', f(vw * k));
  out.setAttribute('height', f(vh * k));
  out.setAttribute('overflow', 'visible');
  out.setAttribute('class', 'math');
  out.dataset.size = size;
  while (src.firstChild) out.appendChild(src.firstChild);
  return { node: out, w: vw * k, top: minY * k };
}

// ---------- rich text ($...$ = TeX) ----------
// '$...$' is TeX, '**...**' is semibold text.
function parseRich(s) {
  const out = []; let buf = '', math = false, bold = false;
  const flush = () => { if (buf) out.push({ t: math ? 'math' : 'text', v: buf, bold }); buf = ''; };
  for (let i = 0; i < s.length; i++) {
    if (s[i] === '$') { flush(); math = !math; continue; }
    if (!math && s.startsWith('**', i)) { flush(); bold = !bold; i++; continue; }
    buf += s[i];
  }
  flush();
  return out;
}

function line(parent, x, y, s, o = {}) {
  const size = o.size || FS, fill = o.fill || INK;
  const g = el('g', { class: 'tl' }, parent);
  let cx = 0;
  for (const seg of parseRich(s)) {
    if (seg.t === 'text') {
      const weight = seg.bold ? 600 : (o.weight || 400);
      const t = el('text', { x: f(cx), y: 0, 'font-size': size, 'font-weight': weight,
        'font-style': o.italic ? 'italic' : null, fill, 'xml:space': 'preserve' }, g);
      t.textContent = seg.v;
      cx += t.getComputedTextLength();
    } else {
      const m = texNode(seg.v, MFS * size / FS);
      m.node.setAttribute('x', f(cx));
      m.node.setAttribute('y', f(m.top));
      m.node.setAttribute('color', fill);
      g.appendChild(m.node);
      cx += m.w;
    }
  }
  let dx = x;
  if (o.anchor === 'middle') dx -= cx / 2; else if (o.anchor === 'end') dx -= cx;
  g.setAttribute('transform', `translate(${f(dx)},${f(y)})`);
  if (o.box) g.dataset.box = o.box;
  return { g, w: cx };
}

function measure(s, o = {}) {
  const r = line(L.scratch, 0, 0, s, o);
  r.g.remove();
  return r.w;
}

function wrap(s, maxW, o = {}) {
  const out = [];
  for (const para of s.split('\n')) {
    const words = []; let cur = '', inM = false;
    for (const ch of para) {
      if (ch === '$') inM = !inM;
      if (ch === ' ' && !inM) { words.push(cur); cur = ''; } else cur += ch;
    }
    words.push(cur);
    let ln = '';
    for (const w of words) {
      const cand = ln ? ln + ' ' + w : w;
      if (ln && measure(cand, o) > maxW) { out.push(ln); ln = w; } else ln = cand;
    }
    out.push(ln);
  }
  return out;
}

// ---------- glyphs ----------
function snowflake(cx, cy, r = 3, color = '#333') {
  const g = el('g', { class: 'glyph', stroke: color, 'stroke-width': 0.5, 'stroke-linecap': 'round', fill: 'none' }, L.text);
  for (let a = 0; a < 3; a++) {
    const t = a * Math.PI / 3;
    const dx = Math.cos(t) * r, dy = Math.sin(t) * r;
    el('line', { x1: f(cx - dx), y1: f(cy - dy), x2: f(cx + dx), y2: f(cy + dy) }, g);
    for (const s of [1, -1]) {
      const bx = cx + s * dx * 0.6, by = cy + s * dy * 0.6;
      for (const sg of [1, -1]) {
        const tt = t + (s > 0 ? 0 : Math.PI) + sg * Math.PI / 4;
        el('line', { x1: f(bx), y1: f(by), x2: f(bx - Math.cos(tt) * r * 0.35), y2: f(by - Math.sin(tt) * r * 0.35) }, g);
      }
    }
  }
}
function flame(cx, cy, s = 3.4, color = '#333') {
  const d = 'M0,-1 C0.42,-0.55 0.72,-0.2 0.68,0.3 C0.64,0.72 0.34,1 0,1 C-0.38,1 -0.68,0.74 -0.7,0.34 ' +
            'C-0.72,-0.04 -0.46,-0.28 -0.3,-0.52 C-0.24,-0.22 -0.1,-0.08 0.06,-0.02 C0.14,-0.38 0.06,-0.72 0,-1 Z';
  el('path', { class: 'glyph', d, fill: color, transform: `translate(${f(cx)},${f(cy)}) scale(${s})` }, L.text);
}
function badge(cx, cy, n, r = 4, fill = '#3A3A3A') {
  el('circle', { cx: f(cx), cy: f(cy), r, fill }, L.text);
  line(L.text, cx, cy + 2.55, String(n), { weight: 600, fill: '#FFFFFF', anchor: 'middle' });
}

// ---------- boxes ----------
let boxId = 0;
function rect(x, y, w, h, role, o = {}) {
  const R = ROLE[role];
  return el('rect', { x: f(x), y: f(y), width: f(w), height: f(h), rx: o.r ?? RAD,
    fill: o.fill ?? R.fill, stroke: o.stroke ?? R.stroke, 'stroke-width': o.sw ?? SW,
    'stroke-dasharray': o.dash || null, id: o.id }, o.layer || L.boxes);
}

// Text box. body: string or array of strings / {s, weight, fill, step}; '\n' forces breaks.
// {step: n} draws a numbered badge and hangs the wrapped lines.
const STEP_IND = 9.5;
function tbox(o) {
  const pad = o.pad ?? PAD;
  const gw = o.glyph ? 8 : 0;
  const items = [];
  const maxW = o.w === 'auto' ? Infinity : o.w - 2 * pad - gw;
  if (o.title) for (const s of wrap(o.title, maxW, { weight: 600 })) items.push({ s, weight: 600, ind: 0 });
  const body = o.body === undefined ? [] : Array.isArray(o.body) ? o.body : [o.body];
  for (const it of body) {
    const st = typeof it === 'string' ? { s: it } : it;
    const ind = st.step ? STEP_IND : 0;
    wrap(st.s, maxW - ind, st).forEach((s, i) => items.push({ ...st, s, ind, step: i === 0 ? st.step : undefined }));
  }
  const widths = items.map(it => measure(it.s, it) + it.ind);
  const w = o.w === 'auto' ? Math.max(...widths) + 2 * pad + gw : o.w;
  const lh = o.lh || LH;
  const natH = 2 * pad + ASC + (items.length - 1) * lh + DESC;
  const h = Math.max(natH, o.h || 0);
  const id = `b${boxId++}`;
  rect(o.x, o.y, w, h, o.role || 'plain', { id, dash: o.dash, fill: o.fill });
  const by = o.y + pad + ASC + (o.valign === 'top' ? 0 : (h - natH) / 2);
  const center = o.align === 'center';
  items.forEach((it, i) => {
    const x0 = center ? o.x + (w - gw) / 2 : o.x + pad + it.ind;
    line(L.text, x0, by + i * lh, it.s, { ...it, anchor: center ? 'middle' : 'start', box: id });
    if (it.step) badge(o.x + pad + 3.5, by + i * lh - 2.5, it.step, 3.5, '#6A6A6A');
  });
  if (o.glyph === 'frozen') snowflake(o.x + w - pad - 3, o.y + h / 2, 3);
  if (o.glyph === 'trainable') flame(o.x + w - pad - 3, o.y + h / 2, 3.2);
  return { x: o.x, y: o.y, w, h, r: o.x + w, b: o.y + h, cx: o.x + w / 2, cy: o.y + h / 2, id };
}

function panel(x, y, w, h, num, title, maxW = Infinity) {
  const id = `p${boxId++}`;
  rect(x, y, w, h, 'panel', { id, layer: L.panels, r: 3.5 });
  const bx = x + PPAD + 4.5, by = y + PPAD + 4.5;
  badge(bx, by, num, 4.5);
  const tx = bx + 8.5;
  const lines = wrap(title, Math.min(maxW, x + w - PPAD - tx), { size: FS_TITLE, weight: 600 });
  lines.forEach((s, i) => line(L.text, tx, by + 2.9 + i * 9.6, s, { size: FS_TITLE, weight: 600, box: id }));
  return { x, y, w, h, r: x + w, b: y + h, id };
}

// ---------- arrows ----------
function arrow(pts, o = {}) {
  const d = 'M' + pts.map(p => `${f(p[0])},${f(p[1])}`).join('L');
  return el('path', { d, fill: 'none', stroke: o.color || ARROW, 'stroke-width': o.sw || SW,
    'stroke-dasharray': o.dash ? '2.4 1.6' : null, 'stroke-linejoin': 'round',
    'marker-end': o.head === false ? null : 'url(#ah)', class: 'arrow' }, L.arrows);
}
function label(x, y, s, o = {}) { return line(L.text, x, y, s, { ...o, cls: 'lbl' }); }

