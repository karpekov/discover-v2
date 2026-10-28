'use strict';
const M = window.REVIEW_MANIFEST;
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const STEP_NAMES = {1: 'Answerable?', 2: 'Queries', 3: 'Retrieval', 4: 'Findings'};
const PROMPTS = {
  1: 'Can this question be answered (at least partly) with this home\'s sensors?',
  2: 'Is this data query relevant for finding this behavior, to answer the question?',
  3: 'Does the highlighted segment show this behavior?',
  4: 'Is this finding correct and supported by the figure / statistics?',
};
const KEYS = {y: 'yes', 1: 'yes', n: 'no', 2: 'no', u: 'unsure', 3: 'unsure'};

let D = null, home = '', step = 1, state = null, SEQ = {}, img = null;
let t = 0, playing = false, lastFrame = null;
const reviewer = () => $('reviewer').value.trim();

// ---------------------------------------------------------------- state
const storeKey = () => `review:${M.packet}:${home}`;
function loadState() {
  state = JSON.parse(localStorage.getItem(storeKey()) || '{}');
  state.ratings = state.ratings || {}; state.cursor = state.cursor || {1: 0, 2: 0, 3: 0, 4: 0};
}
function save() { localStorage.setItem(storeKey(), JSON.stringify(state)); }
const rating = id => state.ratings[id];

function buildSeq() {
  SEQ = {1: [], 2: [], 3: [], 4: []};
  for (const q of D.questions) {
    SEQ[1].push({id: q.step1.id, q, c: ''});
    q.step2.forEach((b, bi) => b.items.forEach((it, ii) => SEQ[2].push({id: it.id, q, b, it, bi, ii, c: b.concept_id})));
    q.step4.items.forEach((f, fi) => SEQ[4].push({id: f.id, q, f, fi, c: ''}));
  }
  SEQ[3] = D.samples.map(s => ({id: s.id, s, q: null, c: ''}));
}
const cur = () => SEQ[step][state.cursor[step]];

// ---------------------------------------------------------------- header
function renderSteps() {
  if (!D) { $('steps').innerHTML = ''; return; }
  $('steps').innerHTML = [1, 2, 3, 4].map(s => {
    const done = SEQ[s].filter(x => rating(x.id)?.rating).length;
    return `<button data-s="${s}" class="${s === step ? 'active' : ''}">${s} · ${STEP_NAMES[s]}<span class="cnt">${done}/${SEQ[s].length}</span></button>`;
  }).join('');
  $('steps').querySelectorAll('button').forEach(b => b.onclick = () => showStep(+b.dataset.s));
}

// ---------------------------------------------------------------- rendering
function stepHead() {
  const n = SEQ[step].length, i = state.cursor[step], done = SEQ[step].filter(x => rating(x.id)?.rating).length;
  return `<div class="stephead"><span class="title">Step ${step} · ${STEP_NAMES[step]}</span>
    <span class="muted">item ${Math.min(i + 1, n)} of ${n} · ${done} rated</span>
    <div class="bar"><div style="width:${n ? 100 * done / n : 0}%"></div></div>
    <button id="prevBtn">← Prev</button><button id="nextBtn">Next →</button><button id="nextUnrated">Next unrated</button></div>`;
}

const introOpen = {};
function intro() {
  const k = `${home}:${step}`, done = SEQ[step].some(x => rating(x.id)?.rating);
  const open = introOpen[k] ?? !done;
  return `<details class="card intro" id="intro" ${open ? 'open' : ''}><summary>What does "${STEP_NAMES[step].replace('?', '')}" mean?
    <span class="muted">${open ? '' : '(click to expand)'}</span></summary>${$('tpl-intro-' + step).innerHTML}</details>`;
}

function rateBar(x) {
  const r = rating(x.id) || {};
  return `<div class="ratebar" id="ratebar"><div class="prompt">${PROMPTS[step]}</div>
    ${[['yes', 'y', 1], ['no', 'n', 2], ['unsure', 'u', 3]].map(([v, k, d]) =>
      `<button class="r ${r.rating === v ? 'sel-' + v : ''}" data-r="${v}">${v[0].toUpperCase() + v.slice(1)} <kbd>${k}</kbd> <kbd>${d}</kbd></button>`).join('')}
    <input class="note" id="note" placeholder="note (optional) — Enter to save" value="${esc(r.note || '')}"></div>`;
}

const qLine = (q, small) => `<div class="qline"><span class="cat">${esc(q.category)} ${esc(q.id_sonia)}</span></div>
  <div class="qtext ${small ? 'small' : ''}">${esc(q.question)}</div>`;

function view1(x) {
  return `<div class="card house"><canvas id="houseCanvas"></canvas><div>
      <div class="muted" style="margin-bottom:6px">Sensors installed in this home (motion: red dots, doors: blue squares).
      Data ${D.data_range[0]} → ${D.data_range[1]}.</div><pre>${esc(D.house_context)}</pre></div></div>
    <div class="card">${qLine(x.q)}</div>`;
}

function view2(x) {
  const b = x.b;
  return `<div class="card">${qLine(x.q, true)}</div>
    <div class="card beh"><div class="pos">Behavior ${x.bi + 1} of ${x.q.step2.length} for this question · query ${x.ii + 1} of ${b.items.length}</div>
      <div class="bn">${esc(b.name)}</div><div>${esc(b.definition)}</div>
      ${b.filters.length ? `<div class="filters">Filters: ${b.filters.map(esc).join(' · ')}</div>` : ''}
      <div class="query">${esc(x.it.query)}</div></div>`;
}

const speed = () => +(localStorage.getItem('review:speed') || 10);

function view3(x) {
  const s = x.s;
  return `<div class="card"><div class="hint">Does the <b style="color:#b45309">highlighted segment</b> of this replay show the following behavior?</div>
      <div class="qtext small">${esc(s.behavior)}</div><div>${esc(s.definition)}</div></div>
    <div id="stage"><div id="canvasWrap"><canvas id="map"></canvas></div>
      <div id="side">
        <div class="card"><div id="clock">--:--:--</div><div class="hint" id="date"></div><div id="inTarget"></div></div>
        <div class="card legend"><div><span style="background:#dc2626"></span>motion (highlighted segment)</div>
          <div><span style="background:#9ca3af"></span>motion (context)</div>
          <div><span style="background:#2563eb;border-radius:2px"></span>door</div>
          <div class="hint" style="margin-top:6px">Lines show the most recent movement path.</div></div>
        <div class="hint"><kbd>space</kbd> play/pause · <kbd>t</kbd> jump to segment</div>
      </div></div>
    <div id="timeline"><canvas id="tl" style="width:100%;height:100%;display:block"></canvas></div>
    <div id="controls"><button id="playBtn">▶ Play</button><button id="targetBtn">Jump to segment</button>
      <label class="hint">Speed <select id="speed">${[1, 2, 5, 10, 20].map(v =>
        `<option ${v === speed() ? 'selected' : ''}>${v}</option>`).join('')}</select>×</label></div>`;
}

function view4(x) {
  const s = x.q.step4;
  return `<div class="card">${qLine(x.q, true)}</div><div class="cols4"><div>
      <div class="pos" style="margin-bottom:6px">Finding ${x.fi + 1} of ${s.items.length} for this question</div>
      ${s.items.map((f, i) => { const r = rating(f.id)?.rating;
        return `<div class="finding ${i === x.fi ? 'cur' : ''}" data-id="${f.id}">${r ? `<span class="st ${r}">${r}</span>` : ''}
          <div class="ft"><b>${i + 1}.</b> ${esc(f.statement)}</div><div class="muted" style="font-size:12px">${esc(f.behavior)}<span class="kind">${esc(f.kind)}</span></div></div>`; }).join('')}
    </div><div>${s.figure ? `<img class="fig" src="${s.figure}" alt="figure">` : '<p class="muted">no figure</p>'}
      ${s.hourly ? `<img class="fig" src="${s.hourly}" alt="time of day" style="margin-top:8px">` : ''}
      <div class="card stats">${s.stats.map(st => `<div class="sn">${esc(st.name)}</div><div class="muted">${esc(st.definition)}</div>
        <ul>${st.lines.map(l => `<li>${esc(l)}</li>`).join('')}</ul>`).join('')}</div></div></div>`;
}

function render() {
  renderSteps();
  const c = $('content'), seq = SEQ[step];
  if (!seq.length) { c.innerHTML = stepHead() + `<div class="card done">Nothing to review in this step for this house.</div>`; wireNav(); return; }
  if (state.cursor[step] >= seq.length) {
    c.innerHTML = stepHead() + `<div class="card done"><h2>Step ${step} done</h2>
      <p>${seq.filter(x => rating(x.id)?.rating).length} of ${seq.length} items rated.</p>
      ${step < 4 ? `<button class="primary" id="goNext">Continue to step ${step + 1} · ${STEP_NAMES[step + 1]}</button>` : '<p>All steps done — please <b>Export ratings CSV</b> and send the file back.</p>'}</div>`;
    wireNav(); if ($('goNext')) $('goNext').onclick = () => showStep(step + 1); return;
  }
  const x = cur();
  c.innerHTML = stepHead() + intro() + {1: view1, 2: view2, 3: view3, 4: view4}[step](x) + rateBar(x);
  wireNav();
  $('intro').querySelector('summary').onclick = () => { introOpen[`${home}:${step}`] = !$('intro').open; };
  $('intro').ontoggle = e => {
    e.target.querySelector('summary .muted').textContent = e.target.open ? '' : '(click to expand)';
    if (step === 1) drawHouse(); if (step === 3) draw();
  };
  c.querySelectorAll('.ratebar button.r').forEach(b => b.onclick = () => rate(b.dataset.r));
  $('note').onkeydown = e => { if (e.key === 'Enter') { saveNote(); e.target.blur(); } };
  $('note').onchange = saveNote;
  if (step === 1) drawHouse();
  if (step === 3) { t = Math.max(0, x.s.target[0] - 5); playing = false; wireReplay(); draw(); }
  if (step === 4) {
    c.querySelectorAll('.finding').forEach(el => el.onclick = () => go(SEQ[4].findIndex(y => y.id === el.dataset.id)));
    c.querySelectorAll('img.fig').forEach(im => im.onclick = () => { $('lightbox').querySelector('img').src = im.src; $('lightbox').style.display = 'flex'; });
  }
}

function wireNav() {
  $('prevBtn').onclick = () => go(state.cursor[step] - 1);
  $('nextBtn').onclick = () => go(state.cursor[step] + 1);
  $('nextUnrated').onclick = () => {
    const n = SEQ[step].length, i0 = state.cursor[step];
    for (let k = 1; k <= n; k++) { const j = (i0 + k) % n; if (!rating(SEQ[step][j].id)?.rating) return go(j); }
    go(n);
  };
}

function go(i) {
  state.cursor[step] = Math.max(0, Math.min(SEQ[step].length, i)); save();
  const keepScroll = step === 4 && cur() && SEQ[4][Math.max(0, i - 1)]?.q === cur().q;
  const y = window.scrollY; render(); window.scrollTo(0, keepScroll ? y : 0);
}

function rate(r) {
  if (!reviewer()) { alert('Please enter your reviewer name first (top left).'); $('reviewer').focus(); return; }
  const x = cur(); if (!x) return;
  state.ratings[x.id] = {rating: r, note: $('note') ? $('note').value : '', rated_at: new Date().toISOString()};
  delete introOpen[`${home}:${step}`];
  $('ratebar').classList.add('flash');
  setTimeout(() => go(state.cursor[step] + 1), 120);
}
function saveNote() { const x = cur(); if (x && rating(x.id)) { rating(x.id).note = $('note').value; save(); } }

// ---------------------------------------------------------------- house map (step 1)
function drawHouse() {
  const cv = $('houseCanvas'); if (!cv) return;
  const W = cv.clientWidth, H = W * D.image.h / D.image.w, dpr = window.devicePixelRatio || 1;
  cv.style.height = H + 'px'; cv.width = W * dpr; cv.height = H * dpr;
  const ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const s = W / D.image.w;
  if (img.complete && img.width) { ctx.globalAlpha = 0.7; ctx.drawImage(img, 0, 0, W, H); ctx.globalAlpha = 1; }
  ctx.font = '9px sans-serif';
  for (const [id, [x, y]] of Object.entries(D.sensors)) {
    const door = id.startsWith('D');
    ctx.fillStyle = door ? '#2563eb' : '#dc2626'; ctx.beginPath();
    door ? ctx.rect(x * s - 4, y * s - 4, 8, 8) : ctx.arc(x * s, y * s, 3.5, 0, 7); ctx.fill();
    ctx.fillStyle = '#111'; ctx.fillText(id, x * s + 5, y * s - 4);
  }
}

// ---------------------------------------------------------------- replay (step 3)
function wireReplay() {
  $('playBtn').onclick = togglePlay;
  $('targetBtn').onclick = jumpToSegment;
  $('speed').onchange = e => { localStorage.setItem('review:speed', e.target.value); e.target.blur(); };
  $('timeline').onclick = e => { const r = e.currentTarget.getBoundingClientRect(); t = (e.clientX - r.left) / r.width * cur().s.duration; draw(); };
}
function jumpToSegment() { t = Math.max(0, cur().s.target[0] - 3); playing = true; $('playBtn').textContent = '❚❚ Pause'; }
function togglePlay() { const it = cur()?.s; if (!it) return; if (t >= it.duration) t = 0; playing = !playing; $('playBtn').textContent = playing ? '❚❚ Pause' : '▶ Play'; }

function draw() {
  const map = $('map'); if (!map || !cur()) return;
  const it = cur().s, ctx = map.getContext('2d'), tl = $('tl'), tctx = tl.getContext('2d');
  const w = map.clientWidth, h = map.clientHeight, dpr = window.devicePixelRatio || 1;
  map.width = w * dpr; map.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  tl.width = tl.clientWidth * dpr; tl.height = tl.clientHeight * dpr; tctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const s = Math.min(w / D.image.w, h / D.image.h), ox = (w - D.image.w * s) / 2, oy = (h - D.image.h * s) / 2;
  const P = id => { const c = D.sensors[id]; return c ? [ox + c[0] * s, oy + c[1] * s] : null; };
  ctx.clearRect(0, 0, w, h);
  if (img && img.complete && img.width) { ctx.globalAlpha = 0.6; ctx.drawImage(img, ox, oy, D.image.w * s, D.image.h * s); ctx.globalAlpha = 1; }
  for (const id in D.sensors) { const p = P(id); ctx.fillStyle = '#d1d5db'; ctx.beginPath(); ctx.arc(p[0], p[1], 4, 0, 7); ctx.fill(); }
  const active = {}, path = [];
  for (const [et, sid, st] of it.events) {
    if (et > t) break;
    active[sid] = {on: st === 'ON' || st === 'OPEN', t: et, st};
    if (st === 'ON' && P(sid) && (!path.length || path[path.length - 1].sid !== sid)) path.push({sid, t: et});
  }
  const recent = path.filter(p => t - p.t < 30).slice(-6);
  for (let i = 1; i < recent.length; i++) {
    const a = P(recent[i - 1].sid), b = P(recent[i].sid);
    ctx.strokeStyle = `rgba(220,38,38,${0.15 + 0.6 * i / recent.length})`; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
  }
  const inT = t >= it.target[0] && t <= it.target[1];
  for (const sid in active) {
    const a = active[sid], p = P(sid); if (!p) continue;
    const age = t - a.t, door = a.st === 'OPEN' || a.st === 'CLOSE';
    if (!a.on && age > 2) continue;
    const alpha = a.on ? Math.max(0.35, 1 - age / 20) : 0.3, r = 7 + 10 * Math.max(0, 1 - age / 3);
    ctx.fillStyle = door ? `rgba(37,99,235,${alpha})` : (inT ? `rgba(220,38,38,${alpha})` : `rgba(156,163,175,${alpha})`);
    ctx.beginPath(); door ? ctx.rect(p[0] - r / 2, p[1] - r / 2, r, r) : ctx.arc(p[0], p[1], r, 0, 7); ctx.fill();
  }
  if (inT) { ctx.strokeStyle = '#f5c542'; ctx.lineWidth = 6; ctx.strokeRect(3, 3, w - 6, h - 6); }
  const clock = new Date(new Date(it.clock_start).getTime() + t * 1000);
  $('clock').textContent = clock.toTimeString().slice(0, 8);
  $('date').textContent = `${it.weekday}, ${clock.toLocaleDateString('en-CA')}`;
  $('inTarget').innerHTML = inT ? '<span style="color:#b45309">● in highlighted segment</span>' : '<span class="hint">context (before/after)</span>';
  const W = tl.clientWidth, H = tl.clientHeight, X = v => v / it.duration * W;
  tctx.clearRect(0, 0, W, H);
  tctx.fillStyle = 'rgba(245,197,66,0.45)'; tctx.fillRect(X(it.target[0]), 0, X(it.target[1]) - X(it.target[0]), H);
  tctx.fillStyle = '#6b7280';
  for (const [et, , st] of it.events) if (st === 'ON' || st === 'OPEN') tctx.fillRect(X(et), H * 0.3, 1, H * 0.4);
  tctx.fillStyle = '#3A6BC4'; tctx.fillRect(X(t) - 1, 0, 3, H);
}
function loop(ts) {
  if (playing && step === 3 && cur()?.s) {
    const it = cur().s;
    if (lastFrame !== null) t += (ts - lastFrame) / 1000 * speed();
    if (t >= it.duration) { t = it.duration; playing = false; if ($('playBtn')) $('playBtn').textContent = '▶ Play'; }
    draw();
  }
  lastFrame = ts; requestAnimationFrame(loop);
}

// ---------------------------------------------------------------- navigation
function showStep(s) {
  step = s; playing = false; render(); window.scrollTo(0, 0);
  localStorage.setItem('review:last', JSON.stringify({home, step}));
}

function landing() {
  D = null; renderSteps();
  $('content').innerHTML = `<div class="card">${$('tpl-help').innerHTML}</div><h2>Pick a house</h2><div class="homes">${M.homes.map(h =>
    `<div class="homecard" data-h="${h.home}"><b style="text-transform:capitalize;font-size:16px">${h.home}</b>
     <div class="muted">${h.n_questions} questions · ${Object.values(h.n_items).reduce((a, b) => a + b, 0)} items</div></div>`).join('')}</div>`;
  document.querySelectorAll('.homecard').forEach(d => d.onclick = () => { $('home').value = d.dataset.h; pickHome(d.dataset.h); });
}

function pickHome(h, s = 1) {
  if (!h) { landing(); return; }
  const go_ = () => {
    home = h; D = window.REVIEW_DATA[h]; loadState(); buildSeq();
    img = new Image(); img.onload = () => { if (step === 1) drawHouse(); if (step === 3) draw(); }; img.src = D.image.src;
    showStep(s);
  };
  if (window.REVIEW_DATA && window.REVIEW_DATA[h]) return go_();
  const sc = document.createElement('script'); sc.src = `data/${h}.js`; sc.onload = go_;
  sc.onerror = () => alert(`Could not load data/${h}.js`); document.body.appendChild(sc);
}

function exportCsv() {
  if (!D) { alert('Pick a house first.'); return; }
  if (!reviewer()) { alert('Please enter your reviewer name first.'); return; }
  const q = v => `"${String(v ?? '').replace(/"/g, '""')}"`;
  const rows = [['reviewer', 'packet', 'home', 'step', 'question_id', 'concept_id', 'item_id', 'rating', 'note', 'rated_at'].join(',')];
  for (const s of [1, 2, 3, 4]) for (const x of SEQ[s]) {
    const r = rating(x.id); if (!r?.rating) continue;
    rows.push([reviewer(), M.packet, home, s, x.q ? x.q.key : '', x.c, x.id, r.rating, r.note, r.rated_at].map(q).join(','));
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([rows.join('\n')], {type: 'text/csv'}));
  a.download = `review_${M.packet}_${home}_${reviewer().replace(/\W+/g, '_')}.csv`; a.click();
}

// ---------------------------------------------------------------- wiring
$('reviewer').value = localStorage.getItem('review:reviewer') || '';
$('reviewer').oninput = e => localStorage.setItem('review:reviewer', e.target.value.trim());
$('home').innerHTML += M.homes.map(h => `<option value="${h.home}">${h.home}</option>`).join('');
$('home').onchange = e => pickHome(e.target.value);
$('helpBtn').onclick = () => { $('modalContent').innerHTML = $('tpl-help').innerHTML; $('modal').style.display = 'flex'; };
$('modalClose').onclick = () => $('modal').style.display = 'none';
$('modal').onclick = e => { if (e.target === $('modal')) $('modal').style.display = 'none'; };
$('lightbox').onclick = () => $('lightbox').style.display = 'none';
$('exportBtn').onclick = exportCsv;
window.onresize = () => { if (step === 3) draw(); if (step === 1) drawHouse(); };
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') { $('modal').style.display = 'none'; $('lightbox').style.display = 'none'; return; }
  if (!D || ['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName) || e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (KEYS[k] && cur()) { e.preventDefault(); rate(KEYS[k]); }
  else if (e.key === 'ArrowRight') go(state.cursor[step] + 1);
  else if (e.key === 'ArrowLeft') go(state.cursor[step] - 1);
  else if (step === 3 && e.key === ' ') { e.preventDefault(); togglePlay(); }
  else if (step === 3 && k === 't') jumpToSegment();
});
const last = JSON.parse(localStorage.getItem('review:last') || 'null');
if (last && M.homes.some(h => h.home === last.home)) { $('home').value = last.home; pickHome(last.home, last.step); } else landing();
requestAnimationFrame(loop);
