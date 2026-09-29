'use strict';
const $ = id => document.getElementById(id);
const number = n => Number(n).toLocaleString(undefined, {maximumFractionDigits: 1});
const bytes = n => n >= 1024 ** 3 ? (n / 1024 ** 3).toFixed(2) + ' GiB' : (n / 1024 ** 2).toFixed(1) + ' MiB';
const percent = n => (100 * n).toFixed(2) + '%';
let data, nodes = new Map(), selected = null, focus = '1', paused = false, busy = false;
let detailToken = 0, lastDetailAt = 0, detailGeneration = null, splits = 0, merges = 0;
const history = [], labels = new Map();
function el(tag, text, cls) {
  const e = document.createElement(tag);
  if (text !== undefined) e.textContent = text;
  if (cls) e.className = cls;
  return e;
}
function svg(tag, attrs = {}) {
  const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
}
async function get(path) {
  const r = await fetch(path, {cache: 'no-store', signal: AbortSignal.timeout(15000)});
  const d = await r.json();
  if (!r.ok) { const e = new Error(d.error || r.statusText); e.status = r.status; throw e; }
  return d;
}
function showError(message) { $('error').hidden = !message; $('error').textContent = message || ''; }
function color(id) { return `hsl(${(Number(id) * 137.508) % 360} 42% 34%)`; }
function drawMap() {
  const map = $('treemap'); map.replaceChildren();
  const root = nodes.get(focus) || nodes.get('1'); if (!root) return;
  const path = []; let n = root;
  while (n) { path.unshift('#' + n.id); n = nodes.get(n.parent); }
  $('breadcrumb').textContent = path.join(' / ') + ' · ' + percent(root.probability) + ' of tree probability';
  function tile(n, x, y, w, h) {
    if (n.leaf) {
      const g = svg('g', {tabindex: 0, role: 'button', 'aria-label': `Leaf ${n.id}, ${n.count} posts`});
      const title = svg('title');
      title.textContent = `Leaf #${n.id} · ${number(n.count)} posts · ${percent(n.probability)} probability`;
      const rect = svg('rect', {x: x + 1, y: y + 1, width: Math.max(0,w - 2), height: Math.max(0,h - 2),
        rx: 3, fill: color(n.id), stroke: selected === n.id ? '#ffde87' : '#142234',
        'stroke-width': selected === n.id ? 4 : 1});
      g.append(title, rect);
      if (w > 65 && h > 38) {
        const text = svg('text', {x: x + 8, y: y + 21});
        const label = labels.get(n.id) || '#' + n.id;
        text.textContent = label.slice(0, Math.max(3, Math.floor((w - 18) / 8)));
        g.append(text);
        if (h > 60) {
          const count = svg('text', {x: x + 8, y: y + 41, opacity: .7});
          count.textContent = number(n.count); g.append(count);
        }
      }
      g.addEventListener('click', () => select(n.id));
      g.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(n.id); } });
      map.append(g); return;
    }
    const a = nodes.get(n.left), b = nodes.get(n.right);
    const fraction = n.probability > 0 ? a.probability / n.probability : .5;
    if (w >= h) { tile(a,x,y,w*fraction,h); tile(b,x+w*fraction,y,w*(1-fraction),h); }
    else { tile(a,x,y,w,h*fraction); tile(b,x,y+h*fraction,w,h*(1-fraction)); }
    map.append(svg('rect', {x,y,width:w,height:h,fill:'none',stroke:'#a8c8df','stroke-opacity':.15,'pointer-events':'none'}));
  }
  tile(root,0,0,900,450);
}
function drawTree() {
  const open = new Set([...$('tree').querySelectorAll('details[open]')].map(e => e.dataset.id));
  const initial = !$('tree').children.length;
  function branch(n) {
    const b = el('button', `${n.leaf ? 'Leaf' : 'Cut'} #${n.id} · ${number(n.count)} posts`, 'node' + (selected === n.id ? ' selected' : ''));
    b.dataset.id = n.id; b.addEventListener('click', e => { e.preventDefault(); select(n.id); });
    if (n.leaf) { const d = el('div'); d.append(b); return d; }
    const d = el('details'); d.dataset.id = n.id; d.open = open.has(n.id) || (initial && n.depth < 2);
    const s = el('summary'); s.append(b); d.append(s);
    const children = el('div', undefined, 'branch');
    children.append(branch(nodes.get(n.left)), branch(nodes.get(n.right))); d.append(children); return d;
  }
  $('tree').replaceChildren(branch(nodes.get('1')));
}
function stats(n, detail) {
  const row = el('div', undefined, 'detail-stats');
  const entries = [['Active posts',number(n.count)],['Tree probability',percent(n.probability)],['Decayed mass',number(n.mass)],
    ['Depth',n.depth],['Log density (natural)',n.log_density == null ? '—' : n.log_density.toFixed(2)]];
  if (detail?.ppca) entries.push(['PPCA rank',detail.ppca.rank],['Reservoir points',detail.ppca.reservoir],
    ['PPCA signal fraction',percent(detail.ppca.signal_fraction)]);
  for (const [k,v] of entries) { const d = el('div',String(v)); d.prepend(el('small',k)); row.append(d); }
  return row;
}
function select(id) {
  selected = id; detailToken++; detailGeneration = null;
  const n = nodes.get(id); if (!n) return;
  $('detail-title').textContent = (n.leaf ? 'Leaf #' : 'Cut #') + id;
  $('detail-generation').textContent = '';
  if (!n.leaf) {
    focus = id;
    const body = $('detail-body'); body.replaceChildren(stats(n));
    body.append(el('p',`Dimension ${n.axis} (1-based), threshold ${n.cut.toPrecision(6)}. Left child #${n.left}, right child #${n.right}.`));
    body.append(el('p','This is an actual partition cut. Rectangle orientation is only a hierarchy layout; it does not represent this embedding axis.','note'));
  } else {
    $('detail-body').replaceChildren(stats(n),el('p','Loading keywords and examples…'));
    loadDetail();
  }
  drawMap(); drawTree();
}
function renderDetail(d, n) {
  $('detail-title').textContent = 'Leaf #' + d.id;
  const body = $('detail-body'); body.replaceChildren(stats(n,d));
  const k = d.keywords;
  const hashtags = k.extractor === 'hashtags-only-v1';
  body.append(el('p',k.enabled
    ? `Processed coverage: ${number(k.covered)} / ${number(d.count)} active posts · ${k.extractor}${hashtags ? ' · up to 4 distinct hashtags per post; posts without tags have no labels' : ''}`
    : 'Keywords are disabled for this store.','note'));
  const columns = el('div',undefined,'keywords');
  for (const [key,title] of [['common','Common terms'],['distinctive','Distinctive terms']]) {
    const c = el('div'); c.append(el('h2',hashtags ? title.replace('terms','hashtags') : title));
    c.append(el('p',key === 'common' ? 'Ordered by decayed support.' : 'Support × positive log-lift; at least 3 supporting posts.','note'));
    for (const t of k[key]) {
      const line = el('div',undefined,'keyword'); line.append(el('span',hashtags ? '#'+t.term : t.term),el('small',`${t.count} posts · ${number(t.support)} support`)); c.append(line);
    }
    if (!k[key].length) c.append(el('div','No qualifying terms.','empty'));
    columns.append(c);
  }
  body.append(columns,el('h2','Example posts'),el('p',d.sample_method + '.', 'note'));
  const posts = el('div',undefined,'posts');
  for (const p of d.posts) {
    const card = el('article',undefined,'post');
    card.append(el('small',`${p.platform} · ${p.author || 'unknown author'} · post #${p.id}`),el('p',p.text));
    if (p.truncated) card.append(el('small','Display truncated to 8,000 characters.'));
    try {
      const u = new URL(p.permalink);
      if (['https:','http:'].includes(u.protocol)) { const a = el('a','Open post ↗'); a.href = u.href; a.target = '_blank'; a.rel = 'noopener noreferrer'; card.append(a); }
    } catch {}
    posts.append(card);
  }
  body.append(posts);
  if (!d.posts.length) body.append(el('p','No available example posts.'));
  if (d.unavailable) body.append(el('p',`${d.unavailable} payload(s) unavailable during concurrent reclamation or decoding. Refresh to retry.`,'note'));
  $('detail-generation').textContent = 'Generation ' + d.generation;
}
async function loadDetail(retry = true) {
  const id = selected, n = nodes.get(id); if (!n?.leaf) return;
  const token = ++detailToken, generation = data.generation;
  lastDetailAt = Date.now();
  try {
    const d = await get('/api/leaf?id=' + encodeURIComponent(id) + '&generation=' + encodeURIComponent(generation));
    if (token !== detailToken || selected !== id) return;
    detailGeneration = d.generation; renderDetail(d,n);
    if (data.generation !== d.generation) $('detail-generation').textContent = 'Generation ' + d.generation + ' · refresh pending';
    if (d.keywords.common.length) labels.set(id,(d.keywords.extractor==='hashtags-only-v1' ? '#' : '')+d.keywords.common[0].term);
    drawMap();
  } catch (e) {
    if (token !== detailToken) return;
    if (e.status === 409 && retry && !busy) {
      await refresh(false); if (selected === id && nodes.get(id)?.leaf) return loadDetail(false);
    }
    $('detail-generation').textContent = 'Summary unavailable';
    $('detail-body').replaceChildren(el('p',e.status === 409 ? 'The tree changed during this read. Select the leaf again or refresh.' : e.message));
  }
}
function drawHistory() {
  const chart = $('history'); chart.replaceChildren();
  if (history.length < 2) { $('history-label').textContent = 'Waiting for a second observation.'; return; }
  const min = Math.min(...history.map(x=>x.active)), max = Math.max(...history.map(x=>x.active));
  const first = history[0].time, span = Math.max(1,history.at(-1).time-first);
  const points = history.map(x=>`${10+880*(x.time-first)/span},${95-75*(x.active-min)/Math.max(1,max-min)}`).join(' ');
  chart.append(svg('polyline',{points,fill:'none',stroke:'#72e3bf','stroke-width':2}));
  $('history-label').textContent = `Active posts: ${number(history[0].active)} → ${number(history.at(-1).active)} · last ${Math.round(span/1000)}s`;
}
function render(d) {
  if (data && data.keyword_extractor !== d.keyword_extractor) {
    labels.clear(); detailGeneration=null;
  }
  if (data && (data.store_id !== d.store_id || data.model !== d.model || BigInt(d.generation) < BigInt(data.generation))) {
    detailToken++; detailGeneration=null; selected=null; focus='1'; splits=0; merges=0;
    history.length=0; labels.clear(); nodes.clear(); $('tree').replaceChildren();
    $('detail-title').textContent='Store changed'; $('detail-generation').textContent='';
    $('detail-body').replaceChildren(el('p','Connected to a different store or earlier generation. Select a partition to inspect it.'));
  }
  const next = new Map(d.nodes.map(n=>[n.id,n]));
  for (const [id,n] of next) { const old=nodes.get(id); if (old && old.leaf !== n.leaf) n.leaf ? merges++ : splits++; }
  data = d; nodes = next;
  for (const id of labels.keys()) if (!nodes.get(id)?.leaf) labels.delete(id);
  if (!nodes.has(focus)) focus = '1';
  if (selected && !nodes.has(selected)) {
    detailToken++; selected = null; $('detail-title').textContent = 'Partition removed';
    $('detail-body').replaceChildren(el('p','This partition was merged or removed. Select another partition.'));
  } else if (selected && !nodes.get(selected).leaf) select(selected);
  $('active').textContent = number(d.active); $('leaves').textContent = `${d.leaves} / ${d.nodes.length-d.leaves}`;
  $('mass').textContent = number(d.mass);
  const proc = kind => d.processes.find(p=>p.kind===kind);
  $('rss').textContent = [proc('julia'),proc('embedding')].map(p=>p ? bytes(p.rss_bytes) : '—').join(' / ');
  const lag = Math.max(0,Date.now()/1000-d.at);
  $('freshness').textContent = `${d.store || ''} · generation ${d.generation} · committed ${new Date(d.at*1000).toLocaleString()} (${Math.round(lag)}s ago) · ${d.dimension}D · ${d.model} · ${d.format} · half-life ${number(d.half_life/3600)}h`;
  $('storage').textContent = `Registered payload files: ${bytes(d.payload_bytes)} · SQLite + WAL + SHM: ${bytes(d.metadata_bytes)} · Keyword coverage: ${number(d.keyword_covered)} / ${number(d.active)} posts. Payload totals may include unreclaimed history.`;
  $('transitions').textContent = `Observed transitions: ${splits} splits, ${merges} merges. Changes between polls may be missed; this is not an event log.`;
  if (detailGeneration && detailGeneration !== d.generation) $('detail-generation').textContent = 'Generation ' + detailGeneration + ' · refresh pending';
  history.push({time:Date.now(),active:d.active}); if (history.length>240) history.shift();
  drawMap(); drawTree(); drawHistory();
}
async function refresh(details = true) {
  if (busy) return; busy = true;
  try { render(await get('/api/snapshot')); showError(''); $('status').textContent = paused ? 'Paused' : 'Observing'; }
  catch(e) { showError('Snapshot unavailable: '+e.message+'. Keeping the last successful view.'); $('status').textContent='Reconnecting'; }
  finally { busy=false; }
  if (details && selected && (Date.now()-lastDetailAt>15000 || detailGeneration == null)) loadDetail();
}
$('pause').onclick = () => { paused=!paused; $('pause').textContent=paused?'Resume':'Pause'; $('status').textContent=paused?'Paused':'Observing'; if(!paused) refresh(); };
$('refresh').onclick = async () => { await refresh(false); loadDetail(); };
$('root').onclick = () => { focus='1'; drawMap(); };
setInterval(()=>{if(!paused) refresh();},5000);
refresh();
