'use strict';
// Architecture figure for the Methods section (fig:model_architecture), drawn from
// docs/methods_revised.tex as evaluated. Layout only; drawing helpers are in lib.js.

function build() {
  svg.innerHTML = '';
  const defs = el('defs');
  const mk = el('marker', { id: 'ah', viewBox: '0 0 4.4 3.4', refX: 4.2, refY: 1.7, markerWidth: 4.4,
    markerHeight: 3.4, markerUnits: 'userSpaceOnUse', orient: 'auto' }, defs);
  el('path', { d: 'M0,0 L4.4,1.7 L0,3.4 Z', fill: ARROW }, mk);
  L.panels = el('g', { id: 'panels' });
  L.boxes = el('g', { id: 'boxes' });
  L.arrows = el('g', { id: 'arrows' });
  L.text = el('g', { id: 'text', 'font-family': 'Inter' });
  L.scratch = el('g', { id: 'scratch', 'font-family': 'Inter', visibility: 'hidden' });
  const lblY = (y0, y1) => (y0 + y1) / 2 + 2.5;   // baseline for a one-line label centred between y0 and y1

  // ======================= TOP ROW: training =======================
  // ---- Stage 1: Multimodal Data Grounding ----
  const s1x = colL(1), s1w = colR(4) - colL(1);
  const bx = s1x + PPAD, ax = bx + 8;
  const bw1 = s1w - 2 * PPAD;
  let y = TITLE_H;
  const chain = [];
  const steps1 = [
    { role: 'sensor', body: '**event stream** $\\mathcal{E}$' },
    { role: 'sensor', body: '**windows** $W_i$: 60 s, 50% overlap' },
    { role: 'det', body: '**context** $\\mathbf{z}_i=\\phi(W_i)$: time of day, day of week, activated rooms, special sensors' },
    { role: 'lang', lh: 9.6, body: '**captions** $\\mathcal{C}_i=\\{c_{i,1},\\dots,c_{i,4}\\}$ from rule-based templates' },
    { role: 'lang', body: '**caption** $c_i$ sampled each epoch' },
  ];
  for (const st of steps1) {
    const b = tbox({ x: bx, y, w: bw1, role: st.role, body: st.body, lh: st.lh });
    chain.push(b);
    y = b.b + 6.5;
  }
  for (let i = 1; i < chain.length; i++) arrow([[ax, chain[i - 1].b], [ax, chain[i].y]]);
  const bW = chain[1], bc = chain[4];
  const TOP_H = bc.b + PPAD + 1;
  panel(s1x, 0, s1w, TOP_H, 1, 'Multimodal Data Grounding');

  // ---- Stage 2: Cross-Modal Representation Learning ----
  const s2x = colL(5), s2w = colR(10) - colL(5), s2r = s2x + s2w - PPAD;
  panel(s2x, 0, s2w, TOP_H, 2, 'Cross-Modal Representation Learning');
  const tokX = s2x + 17;
  const tokH = 12, tokY = bW.cy - tokH / 2;
  let tx = tokX;
  for (const t of ['[CLS]', '[TOD]', '[DOW]']) {
    const b = tbox({ x: tx, y: tokY, w: 'auto', h: tokH, role: 'sensor', body: t, pad: 2, align: 'center' });
    tx = b.r + 1.5;
  }
  tx += 2.5;
  const sq = 7.2, sq0 = tx;
  for (let i = 0; i < 3; i++) {
    const masked = i === 1;
    rect(tx, tokY + (tokH - sq) / 2, sq, sq, masked ? 'plain' : 'sensor', { r: 1, dash: masked ? '1.2 0.9' : null });
    tx += sq + 1.5;
  }
  const stripR = tx - 1.5;
  // bracket over the sensor-value tokens
  const brY = tokY - 2.2;
  el('path', { d: `M${f(sq0)},${f(brY + 1.4)}V${f(brY)}H${f(stripR)}V${f(brY + 1.4)}`, fill: 'none',
    stroke: MUTED, 'stroke-width': 0.45 }, L.arrows);
  label(stripR, brY - 2, '$(s_j,v_j)$ embeddings, 15% masked', { anchor: 'end', fill: MUTED });
  arrow([[bW.r, bW.cy], [tokX, bW.cy]]);
  label((s2x + tokX) / 2 + 0.5, bW.cy + 8.2, '$W_i$', { anchor: 'middle' });

  const encY = tokY + tokH + 6;
  const enc = tbox({ x: tokX, y: encY, w: stripR - tokX, role: 'sensor', body: 'Transformer encoder',
    glyph: 'trainable', align: 'center' });
  arrow([[(tokX + stripR) / 2, tokY + tokH], [(tokX + stripR) / 2, encY]]);

  const txt = tbox({ x: tokX, y: bc.cy - enc.h / 2, w: 'auto', h: enc.h, role: 'lang', body: 'text encoder', glyph: 'frozen' });
  const proj = tbox({ x: txt.r + 7, y: txt.y, w: 'auto', h: enc.h, role: 'lang', body: 'Proj', glyph: 'trainable' });
  arrow([[txt.r, txt.cy], [proj.x, proj.cy]]);
  arrow([[bc.r, bc.cy], [txt.x, bc.cy]]);
  label((s2x + tokX) / 2 + 0.5, bc.cy - 2.6, '$c_i$', { anchor: 'middle' });

  const loss = tbox({ x: tokX, y: enc.b + 6, w: stripR - tokX, role: 'plain', lh: 9.6,
    body: ['$\\mathcal{L}_{\\text{final}}=\\mathcal{L}_{\\text{align}}+\\lambda\\,\\mathcal{L}_{\\text{MLM}}$',
           '$\\mathcal{L}_{\\text{MLM}}$: masked tokens; $\\lambda=1.0$'] });
  arrow([[(tokX + stripR) / 2, enc.b], [(tokX + stripR) / 2, loss.y]]);

  // shared embedding space
  const spX = stripR + 18, spY = tokY - 10, spB = txt.b, spW = s2r - spX;
  rect(spX, spY, spW, spB - spY, 'plain', { id: 'space' });
  const spT = wrap('shared space', spW - 2 * PAD, { weight: 600 });
  spT.forEach((s, i) => line(L.text, spX + PAD, spY + PAD + ASC + i * LH, s, { weight: 600, box: 'space' }));
  const spL = wrap('$\\mathcal{L}_{\\text{align}}$: symmetric InfoNCE', spW - 2 * PAD);
  spL.forEach((s, i) => line(L.text, spX + PAD, spB - PAD - DESC - (spL.length - 1 - i) * 9.4, s, { box: 'space' }));
  const scT = spY + PAD + ASC + (spT.length - 1) * LH + DESC + 3;
  const scB = spB - PAD - DESC - ASC - (spL.length - 1) * 9.4 - 3;
  for (const [u, v] of [[0.22, 0.25], [0.68, 0.18], [0.45, 0.72]]) {
    const px = spX + 7 + u * (spW - 14), py = scT + 2 + v * (scB - scT - 4);
    el('line', { x1: f(px - 2.2), y1: f(py), x2: f(px + 2.2), y2: f(py + 0.8), stroke: '#9A9A9A', 'stroke-width': 0.4 }, L.arrows);
    el('circle', { cx: f(px - 2.2), cy: f(py), r: 1.5, fill: ROLE.sensor.stroke }, L.text);
    el('circle', { cx: f(px + 2.2), cy: f(py + 0.8), r: 1.5, fill: ROLE.lang.stroke }, L.text);
  }
  arrow([[enc.r, enc.cy], [spX, enc.cy]]);
  label((enc.r + spX) / 2, enc.cy - 2.6, '$e_S$', { anchor: 'middle' });
  arrow([[proj.r, proj.cy], [spX, proj.cy]]);
  label((proj.r + spX) / 2, proj.cy - 2.6, '$e_T$', { anchor: 'middle' });

  // ---- gamma calibration (side box, the only place labels are used) ----
  const calX = colL(11), calW = W - colL(11);
  const cal = tbox({ x: calX, y: 0, w: calW, h: TOP_H, role: 'label', valign: 'top', title: '$\\gamma$ calibration',
    body: ['Once per home, on a held-out split with ground-truth labels.',
           'Prototype query per label; threshold sweep; $\\gamma$ maximizes macro $F_{0.5}$.',
           { s: 'Labels are used only here.', weight: 600 }] });

  // ======================= BOTTOM ROW: one user question =======================
  const Y = TOP_H + ROW_GAP;
  // ---- Stage 3: Instrumentation-Aware Analysis Planning ----
  const s3x = colL(1), s3w = colR(5) - colL(1);
  const s3in = s3x + PPAD, s3r = s3x + s3w - PPAD;
  const inv = tbox({ x: s3in, y: Y + TITLE_H, w: 80, role: 'det', title: 'sensor inventory $\\mathcal{M}$',
    body: 'rooms, sensor types, fixture descriptions, date span, eight time-of-day periods, example captions' });
  const qx = inv.r + 8;
  const q = tbox({ x: qx, y: inv.y, w: s3r - qx, role: 'lang', body: 'user question $Q_{user}$' });
  const plan = tbox({ x: qx, y: q.b + 8, w: s3r - qx, role: 'llm', title: 'planner $g$',
    body: ['LLM call 1 of 2', 'no\u00A0ground-truth labels'] });
  arrow([[q.cx, q.b], [q.cx, plan.y]]);
  arrow([[inv.r, plan.cy], [plan.x, plan.cy]]);
  arrow([[bc.x + 8, bc.b], [bc.x + 8, inv.y]], { dash: true });
  label(bc.x + 11, lblY(TOP_H, Y), 'example captions', { fill: MUTED });

  const pY = Math.max(inv.b, plan.b) + 13;
  const planCard = tbox({ x: s3in, y: pY, w: s3r - s3in, role: 'plain', body: [
    { s: 'plan $P=(\\mathcal{S},\\{C_1,\\dots,C_K\\},\\mathcal{A})$', weight: 600 },
    '$\\mathcal{S}$: date scope',
    '$\\mathcal{A}$: ordered analyses (trend, period comparison, timing, profiles, summary)',
  ] });
  const concept = tbox({ x: s3in + 4, y: planCard.b - 1, w: s3r - s3in - 8, role: 'plain', body: [
    { s: 'concept $C_k=(\\mathcal{Q}_k,F_k,\\delta_k)$', weight: 600 },
    'physical definition',
    '$\\delta_k\\in\\{0,1\\}$: detectability flag and note',
    '$\\mathcal{Q}_k$: 1–6 caption-style queries',
    '$F_k$: time-of-day, day-of-week, room-visit, required-sensor filters',
    'start-time interval (timing concepts)',
  ] });
  const planB = concept.b + 4;
  document.getElementById(planCard.id).setAttribute('height', f(planB - pY));
  arrow([[plan.cx, plan.b], [plan.cx, pY]]);
  label(plan.cx + 3, lblY(plan.b, pY), '$P$');

  // delta_k = 0 branch terminates here
  const nfX = s3in + 18;
  const nf = tbox({ x: nfX, y: planB + 9, w: s3r - nfX, role: 'plain', dash: '2 1.5',
    body: ['$\\delta_k=0$: not feasible; no retrieval or measurement; reporter explains why'] });
  arrow([[s3in + 9, planB], [s3in + 9, nf.cy], [nf.x, nf.cy]]);

  // ---- Stage 4: Filtered Retrieval and Quantitative Reporting ----
  const s4x = colL(6), s4w = W - colL(6);
  const s4in = s4x + PPAD, s4r = s4x + s4w - PPAD;
  const retX = s4in + 13;
  const ret = tbox({ x: retX, y: Y + TITLE_H, w: s4r - retX, role: 'det',
    title: 'retrieval for each $C_k$ with $\\delta_k=1$', body: [
      { step: 1, s: 'restrict $D$ to the date scope $\\mathcal{S}$' },
      { step: 2, s: 'time-of-day and day-of-week filters' },
      { step: 3, s: 'room-visit and required-sensor filters' },
      { step: 4, s: 'match if $\\max_{q\\in\\mathcal{Q}_k}\\mathrm{sim}(e_q,e_{S,i})\\geq\\gamma$' },
    ] });
  // plan -> retrieval (elbow through the gutter)
  const gx = (colR(5) + colL(6)) / 2;
  const pOut = planCard.y + 9;
  arrow([[s3r, pOut], [gx, pOut], [gx, ret.cy], [ret.x, ret.cy]]);
  label((gx + ret.x) / 2 + 0.5, ret.cy - 2.6, '$C_k$', { anchor: 'middle' });
  // trained encoders and gamma enter from the top row
  const dA = s4x + 126, dB = s2r - 6, dG = cal.cx;
  arrow([[dA, TOP_H], [dA, ret.y]], { dash: true });
  label(dA - 2.5, lblY(TOP_H, Y), 'pre-encoded windows $\\{e_{S,1},\\dots,e_{S,N}\\}$', { anchor: 'end' });
  arrow([[dB, TOP_H], [dB, ret.y]], { dash: true });
  label(dB + 2.5, lblY(TOP_H, Y), '$e_q$');
  arrow([[dG, TOP_H], [dG, ret.y]], { dash: true });
  label(dG + 2.5, lblY(TOP_H, Y), '$\\gamma$');

  // measurement h
  const mX = ret.x, mW = 106, mY = ret.b + 15;
  const meas = tbox({ x: mX, y: mY, w: mW, role: 'det', title: 'measurement $h$', body: [
    '**daily:** matched minutes as interval union of $[t_i,t_i+60\\,\\mathrm{s})$; observed active time; share; valid day if $\\geq 30$ min of activity; episodes with gaps $\\leq 120$ s',
    '**timing:** activity day from 04:00; first/last or all sessions',
    '**tests:** OLS + Kendall $\\tau$ trend; Mann–Whitney, first vs. last third; time-of-day and weekday profiles',
  ] });
  arrow([[meas.cx, ret.b], [meas.cx, mY]]);
  label(meas.cx + 2.5, lblY(ret.b, mY), '$R(C_k)$');
  const fig = tbox({ x: mX, y: meas.b + 8, w: mW, role: 'det', body: 'figure: one row per concept, drawn by code' });
  arrow([[meas.cx, meas.b], [meas.cx, fig.y]]);

  // reporting
  const rX = meas.r + 19, rW = s4r - rX;
  const rep = tbox({ x: rX, y: mY, w: rW, role: 'llm', title: 'reporter $r$',
    body: ['LLM call 2 of 2', 'inputs: $Q_{user}$, $P$, $\\{\\Sigma_k\\}$', 'no\u00A0ground-truth labels'] });
  arrow([[meas.r, rep.cy], [rX, rep.cy]]);
  label((meas.r + rX) / 2, rep.cy - 2.6, '$\\Sigma_k$', { anchor: 'middle' });
  const report = tbox({ x: rX, y: rep.b + 9, w: rW, role: 'lang', title: 'report',
    body: ['one-sentence headline', 'body: 3–6 sentences', '2–8 atomic findings', 'text for the figure'] });
  arrow([[rep.cx, rep.b], [rep.cx, report.y]]);
  const chk = tbox({ x: rX, y: report.b + 9, w: rW, role: 'det', title: 'numeric-grounding check',
    body: 'post-hoc; flags report numbers not in $\\{\\Sigma_k\\}$' });
  arrow([[report.cx, report.b], [report.cx, chk.y]]);

  // deterministic region: everything between the two LLM calls
  const m = 3;
  const detB = fig.b + m + LH + 1;
  el('path', { d: `M${f(ret.x - m)},${f(ret.y - m)}H${f(ret.r + m)}V${f(ret.b + 7)}H${f(meas.r + 1.5)}` +
    `V${f(detB)}H${f(ret.x - m)}Z`, fill: 'none', stroke: MUTED, 'stroke-width': 0.6,
    'stroke-dasharray': '3 1.8', id: 'detregion' }, L.boxes);
  line(L.text, ret.x, detB - 2.6, 'deterministic code (no LLM)', { weight: 600, fill: MUTED });

  const BOT_H = Math.max(nf.b, detB, chk.b) + PPAD - Y;
  panel(s3x, Y, s3w, BOT_H, 3, 'Instrumentation-Aware Analysis Planning');
  panel(s4x, Y, s4w, BOT_H, 4, 'Filtered Retrieval and Quantitative Reporting', 100);

  // ---- legend: two rows (roles, then glyphs and line styles) ----
  const rows = [
    [['sw', 'sensor', 'sensor modality'], ['sw', 'lang', 'language modality'], ['sw', 'llm', 'LLM call'],
     ['sw', 'det', 'deterministic code'], ['sw', 'label', 'ground-truth labels']],
    [['frozen', null, 'frozen weights'], ['trainable', null, 'trainable weights'],
     ['solid', null, 'data flow'], ['dash', null, 'reused from training or calibration']],
  ];
  let legY = Y + BOT_H + 10;
  for (const row of rows) {
    let lxp = 0;
    for (const [kind, role, text] of row) {
      const lead = kind === 'dash' || kind === 'solid' ? 14 : 10.5;
      if (kind === 'sw') rect(lxp, legY - 5.6, 8, 6, role, { r: 1.2 });
      if (kind === 'frozen') snowflake(lxp + 4, legY - 2.6, 3);
      if (kind === 'trainable') flame(lxp + 4, legY - 2.6, 3.2);
      if (kind === 'dash' || kind === 'solid') arrow([[lxp, legY - 2.6], [lxp + 11, legY - 2.6]], { dash: kind === 'dash' });
      const r = line(L.text, lxp + lead, legY, text, {});
      lxp += lead + r.w + 12;
    }
    legY += LH + 1.5;
  }
  const H = Math.ceil(legY - LH - 1.5 + DESC + 1.5);
  return { H };
}

async function main() {
  await MathJax.startup.promise;
  await Promise.all(['400', '600', 'italic 400'].map(w => document.fonts.load(`${w} 7px Inter`)));
  const { H } = build();
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('width', `${W}pt`);
  svg.setAttribute('height', `${H}pt`);
  document.getElementById('scratch').remove();
  window.FIGURE = { W, H };
  window.FIGURE_READY = true;
}
main();
