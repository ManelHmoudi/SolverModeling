"""HTML report generator for IRP FunctionMerge combined solve results."""

import json
import math
import os

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IRP &mdash; Function Merge</title>
<style>
:root {
  --bg:        #f4f4f2;
  --surface:   #ffffff;
  --border:    #e5e5e3;
  --text:      #1a1a18;
  --text-2:    #888;
  --tab-bg:    #ececea;
  --row-bg:    #f4f4f2;
  --mesh:      #e0e0dd;
  --node-surf: #ffffff;
  --f1-bg:#e6f1fb; --f1-fg:#185fa5;
  --f2-bg:#e8f5e2; --f2-fg:#3b6d11;
  --f3-bg:#fdecea; --f3-fg:#a32d2d;
  --f4-bg:#eeedfe; --f4-fg:#534ab7;
  --cx-bg:#fff8e1; --cx-fg:#7a5800;
}
[data-dark] {
  --bg:        #111113;
  --surface:   #1c1c20;
  --border:    #2c2c34;
  --text:      #ededed;
  --text-2:    #777;
  --tab-bg:    #232328;
  --row-bg:    #222226;
  --mesh:      #252528;
  --node-surf: #1c1c20;
  --f1-bg:#1a2a3d; --f1-fg:#6aaae8;
  --f2-bg:#1a2e1a; --f2-fg:#6abd52;
  --f3-bg:#2e1a1a; --f3-fg:#e06060;
  --f4-bg:#2e2b52; --f4-fg:#9f9cf5;
  --cx-bg:#2a2210; --cx-fg:#f0c040;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
  background:var(--bg);color:var(--text);
  padding:1.25rem 1.75rem;font-size:14px;line-height:1.5;
  transition:background .2s,color .2s;
}
.hdr{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:1.25rem;gap:1rem}
.hdr h1{font-size:20px;font-weight:700;letter-spacing:-.015em}
.hdr .meta{color:var(--text-2);font-size:12px;margin-top:3px}
#tbtn{background:var(--surface);border:1px solid var(--border);border-radius:8px;
  padding:5px 11px;cursor:pointer;font-size:15px;color:var(--text);transition:background .15s}
#tbtn:hover{background:var(--tab-bg)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;
  padding:1.1rem 1.25rem;margin-bottom:1.1rem}
.ct{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;
  color:var(--text-2);margin-bottom:.9rem}
/* KPI grid */
.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:1.1rem}
@media(max-width:900px){.kpis{grid-template-columns:1fr 1fr 1fr}}
@media(max-width:560px){.kpis{grid-template-columns:1fr 1fr}}
.kpi{border-radius:10px;padding:.9rem 1.1rem;border:1px solid var(--border)}
.kpi.f1{background:var(--f1-bg)}
.kpi.f2{background:var(--f2-bg)}
.kpi.f3{background:var(--f3-bg)}
.kpi.f4{background:var(--f4-bg)}
.kpi.cx{background:var(--cx-bg)}
.kl{font-size:11px;margin-bottom:5px}
.kpi.f1 .kl{color:var(--f1-fg)}
.kpi.f2 .kl{color:var(--f2-fg)}
.kpi.f3 .kl{color:var(--f3-fg)}
.kpi.f4 .kl{color:var(--f4-fg)}
.kpi.cx .kl{color:var(--cx-fg)}
.kv{font-size:21px;font-weight:700;line-height:1;margin-bottom:3px;color:var(--text)}
.ku{font-size:11px;color:var(--text-2)}
/* Split layout */
.split{display:grid;grid-template-columns:1fr 1fr;gap:1.1rem;margin-bottom:1.1rem}
@media(max-width:860px){.split{grid-template-columns:1fr}}
/* Period tabs */
.tabs{display:flex;gap:5px;margin-bottom:.85rem;flex-wrap:wrap}
.tab{padding:4px 12px;border-radius:7px;font-size:11px;cursor:pointer;
  border:1px solid var(--border);background:var(--tab-bg);color:var(--text-2);
  transition:background .15s,color .15s,border-color .15s;white-space:nowrap}
.tab.active{background:var(--text);color:var(--bg);border-color:var(--text)}
/* SVG container */
.svgwrap{position:relative;overflow:hidden;border-radius:8px;background:var(--bg);
  border:1px solid var(--border)}
.svgwrap svg{display:block;width:100%;height:auto}
/* Delivery table */
table{width:100%;border-collapse:collapse;font-size:12px}
th{text-align:left;padding:6px 8px;border-bottom:2px solid var(--border);
  font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--text-2)}
td{padding:5px 8px;border-bottom:1px solid var(--border)}
tr:last-child td{border-bottom:none}
tr:nth-child(even) td{background:var(--row-bg)}
.badge{display:inline-block;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:600;cursor:default}
.ok{background:var(--f2-bg);color:var(--f2-fg)}
.partial{background:var(--cx-bg);color:var(--cx-fg)}
.none{background:var(--f3-bg);color:var(--f3-fg)}
.advance{background:var(--f4-bg);color:var(--f4-fg)}
/* BFR detail */
.bfr-row{display:grid;grid-template-columns:1.6fr 2fr 1fr;
  padding:6px 0;border-bottom:1px solid var(--border);font-size:13px;align-items:center}
.bfr-header-row{font-size:11px;font-weight:600;text-transform:uppercase;
  letter-spacing:.04em;color:var(--text-2);border-bottom:2px solid var(--border)}
.bfr-total-row{font-weight:700;border-top:2px solid var(--border);border-bottom:none;margin-top:2px}
.bfr-col-label{color:var(--text)}
.bfr-col-formula{color:var(--text-2)}
.bfr-formula-text{font-style:italic;font-size:12px}
.bfr-col-value{text-align:right;font-variant-numeric:tabular-nums}
/* Formula note */
.formula{font-size:11px;color:var(--text-2);margin-top:.5rem;padding-top:.5rem;
  border-top:1px solid var(--border)}
/* Stock equation blocks (calibration style) */
.sb{margin-bottom:1rem}
.sp{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--text-2);margin-bottom:.6rem}
.seq{display:flex;align-items:stretch;flex-wrap:wrap;gap:2px}
.sc{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:10px 14px;border-radius:9px;min-width:70px;text-align:center;flex:1}
.sl{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;margin-bottom:3px;opacity:.75}
.sv{font-size:17px;font-weight:700;line-height:1}
.sop{display:flex;align-items:center;justify-content:center;font-size:20px;font-weight:300;color:var(--text-2);padding:0 4px;flex-shrink:0}
.si{background:var(--f1-bg);color:var(--f1-fg)}
.sr{background:var(--f2-bg);color:var(--f2-fg)}
.sd{background:var(--f3-bg);color:var(--f3-fg)}
.sf{background:var(--f4-bg);color:var(--f4-fg)}
</style>
</head>
<body>
<main>

<div class="hdr">
  <div>
    <h1>IRP &mdash; Function Merge</h1>
    <div class="meta" id="metaLine"></div>
  </div>
  <button id="tbtn" onclick="toggleDark()" title="Toggle theme">&#9680;</button>
</div>

<!-- KPI row -->
<div class="kpis" id="kpiRow"></div>

<!-- Network + Depot stock -->
<div class="split">
  <div class="card">
    <div class="ct">Route network</div>
    <div class="tabs" id="periodTabs"></div>
    <div class="svgwrap" id="svgWrap"></div>
  </div>
  <div class="card">
    <div class="ct">Depot stock evolution</div>
    <div id="depotTable"></div>
  </div>
</div>

<!-- Deliveries -->
<div class="card">
  <div class="ct">Deliveries per client &amp; period</div>
  <div id="delivTable"></div>
</div>

<!-- BFR breakdown -->
<div class="card">
  <div class="ct">Working capital breakdown (f4)</div>
  <div id="bfrDetail"></div>
</div>

</main>
<script>
const D = /*DATA_PLACEHOLDER*/null;

/* ── Theme ─────────────────────────────────────────────────────────────── */
function toggleDark(){
  const on = document.documentElement.toggleAttribute('data-dark');
  localStorage.setItem('fm-dark', on ? '1' : '');
  renderNetwork(currentPeriod);
}
(function(){
  if(localStorage.getItem('fm-dark')==='1')
    document.documentElement.setAttribute('data-dark','');
})();

/* ── Vehicle colours ────────────────────────────────────────────────────── */
const TRUCK_COLORS = ['#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6','#06b6d4'];
function truckColor(k){ return TRUCK_COLORS[(k-1) % TRUCK_COLORS.length]; }

/* ── Meta header ────────────────────────────────────────────────────────── */
function renderMeta(){
  const m = D.meta;
  document.getElementById('metaLine').textContent =
    `${m.model_name}  ·  ${m.n_nodes} nodes  ·  ${m.n_periods} periods  ·  ${m.n_vehicles} vehicles`;
}

/* ── KPI row ────────────────────────────────────────────────────────────── */
function renderKPIs(){
  const o  = D.objectives;
  const defs = [
    {cls:'f1', label:'f1 — Logistics cost',   val:o.f1, unit:'cost units'},
    {cls:'f2', label:'f2 — CO₂ emissions',    val:o.f2, unit:'kg CO₂'},
    {cls:'f3', label:'f3 — Travel time',      val:o.f3, unit:'hours'},
    {cls:'f4', label:'f4 — Working capital',  val:o.f4, unit:'currency'},
    {cls:'cx', label:'Composite  f1+f2+f3+f4',val:o.composite, unit:'scalarised objective'},
  ];
  const row = document.getElementById('kpiRow');
  row.innerHTML = defs.map(d => `
    <div class="kpi ${d.cls}">
      <div class="kl">${d.label}</div>
      <div class="kv">${fmt(d.val)}</div>
      <div class="ku">${d.unit}</div>
    </div>`).join('');
}

function fmt(v){
  if(v===null||v===undefined) return '—';
  const av = Math.abs(v);
  if(av>=1e6)  return (v/1e6).toFixed(2)+'M';
  if(av>=1e3)  return (v/1e3).toFixed(2)+'k';
  return v.toFixed ? v.toFixed(3) : v;
}

/* ── Period tabs ────────────────────────────────────────────────────────── */
let currentPeriod = null;

function renderPeriodTabs(){
  const periods = Object.keys(D.routes).map(Number).sort((a,b)=>a-b);
  currentPeriod = periods[0];
  const wrap = document.getElementById('periodTabs');
  wrap.innerHTML = periods.map(t =>
    `<div class="tab${t===currentPeriod?' active':''}" onclick="selectPeriod(${t})">Period ${t}</div>`
  ).join('');
}

function selectPeriod(t){
  currentPeriod = t;
  document.querySelectorAll('#periodTabs .tab').forEach(el=>{
    el.classList.toggle('active', el.textContent===`Period ${t}`);
  });
  renderNetwork(t);
}

/* ── Network SVG ────────────────────────────────────────────────────────── */
function renderNetwork(t){
  const pos   = D.meta.node_positions;
  const nodes = Object.keys(pos).map(Number);
  const period = D.routes[String(t)];
  const trucks = period ? period.trucks : [];

  // Infer SVG size from positions
  let maxX=0, maxY=0;
  nodes.forEach(n => {
    maxX = Math.max(maxX, pos[String(n)][0]);
    maxY = Math.max(maxY, pos[String(n)][1]);
  });
  const W = maxX + 60, H = maxY + 50;

  const isDark = document.documentElement.hasAttribute('data-dark');
  const nodeFill   = isDark ? '#1c1c20' : '#ffffff';
  const nodeStroke = isDark ? '#8494ab' : '#555';
  const textFill   = isDark ? '#ededed' : '#1a1a18';
  const meshFill   = isDark ? '#252528' : '#e0e0dd';
  const depotFill  = isDark ? '#4ab0cc' : '#1a6f8a';

  let arcs = '';
  const seen = new Set();

  trucks.forEach(tr => {
    const col = truckColor(tr.k);
    const path = tr.path;
    for(let i=0; i<path.length-1; i++){
      const a = path[i], b = path[i+1];
      const key = `${a}-${b}`;
      if(seen.has(key)) return;
      seen.add(key);
      const [x1,y1] = pos[String(a)];
      const [x2,y2] = pos[String(b)];
      const mx = (x1+x2)/2, my = (y1+y2)/2;
      const dx = x2-x1, dy = y2-y1;
      const nx = -dy, ny = dx;
      const len = Math.sqrt(nx*nx+ny*ny)||1;
      const bend = 22;
      const cx = mx + (nx/len)*bend, cy = my + (ny/len)*bend;

      const delivered = tr.qty && tr.qty[String(b)] ? tr.qty[String(b)] : null;
      const labelTxt  = delivered !== null ? String(delivered) : '';

      // arrow marker id per truck
      const mid = `arr${tr.k}`;
      arcs += `<path d="M${x1},${y1} Q${cx},${cy} ${x2},${y2}"
        fill="none" stroke="${col}" stroke-width="2.2" stroke-linecap="round"
        marker-end="url(#${mid})" opacity="0.82"/>`;
      if(labelTxt){
        arcs += `<text x="${cx}" y="${cy-5}" text-anchor="middle" font-size="9"
          fill="${col}" font-weight="600">${labelTxt}</text>`;
      }
    }
    // return to depot (dashed)
    if(path.length>0){
      const last = path[path.length-1];
      if(last !== 0){
        const [x1,y1] = pos[String(last)];
        const [x2,y2] = pos['0'];
        const mx=(x1+x2)/2, my=(y1+y2)/2;
        const dx=x2-x1, dy=y2-y1;
        const nx=-dy, ny=dx;
        const len=Math.sqrt(nx*nx+ny*ny)||1;
        const bend=18;
        const cx=mx+(nx/len)*bend, cy=my+(ny/len)*bend;
        arcs += `<path d="M${x1},${y1} Q${cx},${cy} ${x2},${y2}"
          fill="none" stroke="${col}" stroke-width="1.4" stroke-dasharray="5,3"
          opacity="0.5"/>`;
      }
    }
  });

  // Defs for arrowheads per truck colour
  const defs = trucks.map(tr => {
    const col = truckColor(tr.k);
    return `<marker id="arr${tr.k}" markerWidth="7" markerHeight="7"
      refX="5" refY="3.5" orient="auto">
      <polygon points="0 0,7 3.5,0 7" fill="${col}" opacity="0.85"/>
    </marker>`;
  }).join('');

  // Nodes
  const nodesSvg = nodes.map(n => {
    const [x,y] = pos[String(n)];
    const isDepot = (n===0);
    const r = isDepot ? 17 : 14;
    const fill = isDepot ? depotFill : nodeFill;
    const stroke = isDepot ? depotFill : nodeStroke;
    const tFill  = isDepot ? '#fff' : textFill;
    const label  = isDepot ? 'D' : String(n);
    return `<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="1.8"/>
      <text x="${x}" y="${y+4}" text-anchor="middle" font-size="${isDepot?11:10}"
        font-weight="700" fill="${tFill}">${label}</text>`;
  }).join('');

  const svgStr = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">
    <defs>${defs}</defs>
    <rect width="${W}" height="${H}" fill="${isDark?'#111113':'#f4f4f2'}"/>
    ${arcs}
    ${nodesSvg}
  </svg>`;

  // HTML legend below SVG
  let lgd = '<div style="display:flex;gap:5px;flex-wrap:wrap;margin-top:9px;align-items:center">';
  if(trucks.length > 0){
    const pr = D.routes[String(t)];
    if(pr && pr.tau_return != null){
      lgd += `<span style="font-size:11px;color:var(--text-2);padding:3px 10px;
        background:var(--row-bg);border:1px solid var(--border);border-radius:99px;white-space:nowrap">
        &#128339; Return: ${pr.tau_return.toFixed(2)}h</span>`;
    }
    trucks.forEach(tr => {
      const c = truckColor(tr.k);
      lgd += `<span style="display:inline-flex;align-items:center;gap:4px;padding:3px 9px 3px 6px;
        background:var(--row-bg);border:1px solid var(--border);border-radius:99px;font-size:11px;white-space:nowrap">
        <span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${c};flex-shrink:0"></span>
        <b style="color:var(--text)">k=${tr.k}</b></span>`;
    });
  }
  lgd += '</div>';

  document.getElementById('svgWrap').innerHTML = svgStr + lgd;
}

/* ── Depot stock (equation style) ────────────────────────────────────────── */
function renderDepotTable(){
  const stock = D.depot_stock;
  const periods = Object.keys(stock).map(Number).sort((a,b)=>a-b);
  let prevF = D.meta.I_O_init_frigo, prevNF = D.meta.I_O_init_nonfrigo;

  function stockRow(label, prev, R, fin){
    const livr = (typeof prev==='number' && typeof R==='number' && typeof fin==='number')
      ? Math.round((prev + R - fin)*100)/100 : '?';
    return `<div style="margin-bottom:6px">
      <div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--text-2);margin-bottom:3px">${label}</div>
      <div class="seq">
        <div class="sc si"><span class="sl">Init.</span><span class="sv">${prev}</span></div>
        <div class="sop">+</div>
        <div class="sc sr"><span class="sl">Replen.</span><span class="sv">${R}</span></div>
        <div class="sop">&#8722;</div>
        <div class="sc sd"><span class="sl">Delivered</span><span class="sv">${livr}</span></div>
        <div class="sop">=</div>
        <div class="sc sf"><span class="sl">End stock</span><span class="sv">${fin}</span></div>
      </div></div>`;
  }

  document.getElementById('depotTable').innerHTML = periods.map(t => {
    const s = stock[String(t)];
    const r = D.routes[String(t)] || {};
    const rf = r.R_frigo||0, rnf = r.R_nonfrigo||0;
    const finF = s ? s.frigo : '?', finNF = s ? s.nonfrigo : '?';
    const html = `<div class="sb">
      <div class="sp">Period ${t}</div>
      ${stockRow('&#10052;&#65039; Refrigerated', prevF, rf, finF)}
      ${stockRow('&#128230; Non-refrigerated', prevNF, rnf, finNF)}
    </div>`;
    prevF = finF; prevNF = finNF;
    return html;
  }).join('');
}

/* ── Deliveries table ───────────────────────────────────────────────────── */
function renderDelivTable(){
  const rows = D.deliveries.map(d => {
    const ratio = d.dem>0 ? d.recu/d.dem : 1;
    const badge = ratio > 1.001 ? `<span class="badge advance" title="Early delivery: also covers future demand periods">Early</span>`
                : ratio >= 0.999 ? '<span class="badge ok">Full</span>'
                : ratio > 0.001  ? `<span class="badge partial">${(ratio*100).toFixed(0)}%</span>`
                                 : '<span class="badge none">None</span>';
    return `<tr>
      <td>Client ${d.l}</td>
      <td>t=${d.t}</td>
      <td><span style="color:${truckColor(d.k)};font-weight:600">Truck ${d.k}</span></td>
      <td>${d.recu}</td>
      <td>${d.dem}</td>
      <td>${badge}</td>
    </tr>`;
  }).join('');

  document.getElementById('delivTable').innerHTML = `
    <table>
      <thead><tr>
        <th>Client</th><th>Period</th><th>Vehicle</th>
        <th>Delivered</th>
        <th title="Demand for this period only. An &quot;Early&quot; delivery also covers future demand periods.">Demand (t) ⓘ</th>
        <th>Status</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

/* ── BFR breakdown ──────────────────────────────────────────────────────── */
function renderBFR(){
  const b = D.bfr_sub;
  const net = (b.stock + b.receivables - b.payables).toFixed(4);
  document.getElementById('bfrDetail').innerHTML = `
    <div class="bfr-row bfr-header-row">
      <span class="bfr-col-label">Component</span>
      <span class="bfr-col-formula">Formula</span>
      <span class="bfr-col-value">Value</span>
    </div>
    <div class="bfr-row">
      <span class="bfr-col-label">Stock value</span>
      <span class="bfr-col-formula bfr-formula-text">I × P<sub>purchase</sub> × DIO / 365</span>
      <span class="bfr-col-value">${b.stock}</span>
    </div>
    <div class="bfr-row">
      <span class="bfr-col-label">+ Accounts receivable</span>
      <span class="bfr-col-formula bfr-formula-text">q × P<sub>sale</sub> × DSO / 365</span>
      <span class="bfr-col-value">${b.receivables}</span>
    </div>
    <div class="bfr-row">
      <span class="bfr-col-label">− Accounts payable</span>
      <span class="bfr-col-formula bfr-formula-text">q × P<sub>purchase</sub> × DPO / 365</span>
      <span class="bfr-col-value">−${b.payables}</span>
    </div>
    <div class="bfr-row bfr-total-row">
      <span class="bfr-col-label">f4 = BFR</span>
      <span class="bfr-col-formula bfr-formula-text">Stock + Receivables − Payables</span>
      <span class="bfr-col-value">${net}</span>
    </div>
    <p class="formula">Scalarization: minimize f1 + f2 + f3 + f4</p>`;
}

/* ── Boot ───────────────────────────────────────────────────────────────── */
renderMeta();
renderKPIs();
renderPeriodTabs();
renderNetwork(currentPeriod);
renderDepotTable();
renderDelivTable();
renderBFR();
</script>
</body>
</html>"""


def compute_node_positions(N, svg_w=480, svg_h=290):
    """Depot (node 0) on the left; clients spread in a fan on the right."""
    clients = sorted(n for n in N if n != 0)
    nc      = len(clients)
    pos     = {}

    if nc > 6 and svg_w == 480 and svg_h == 290:
        svg_w, svg_h = 1120, 680

    pos[0] = [int(svg_w * (0.11 if nc > 6 else 0.13)), svg_h // 2]

    if nc == 0:
        pass
    elif nc == 1:
        pos[clients[0]] = [int(svg_w * 0.82), svg_h // 2]
    elif nc == 2:
        pos[clients[0]] = [int(svg_w * 0.78), int(svg_h * 0.22)]
        pos[clients[1]] = [int(svg_w * 0.78), int(svg_h * 0.78)]
    elif nc == 3:
        pos[clients[0]] = [int(svg_w * 0.55), int(svg_h * 0.13)]
        pos[clients[1]] = [int(svg_w * 0.84), int(svg_h * 0.52)]
        pos[clients[2]] = [int(svg_w * 0.52), int(svg_h * 0.87)]
    elif nc == 5:
        xs = [0.40, 0.84, 0.91, 0.76, 0.38]
        ys = [0.12, 0.17, 0.53, 0.87, 0.84]
        for i, c in enumerate(clients):
            pos[c] = [int(svg_w * xs[i]), int(svg_h * ys[i])]
    else:
        cx  = svg_w * 0.58
        cy  = svg_h / 2
        rx  = svg_w * 0.34
        ry  = svg_h * 0.39
        for i, c in enumerate(clients):
            angle = -math.pi / 2 + 2 * math.pi * i / nc
            pos[c] = [int(cx + rx * math.cos(angle)),
                      int(cy + ry * math.sin(angle))]

    return {str(k): v for k, v in pos.items()}


def render_report_html(data):
    return _TEMPLATE.replace(
        "/*DATA_PLACEHOLDER*/null",
        json.dumps(data, ensure_ascii=False),
    )


def write_report(data, output_path):
    html = render_report_html(data)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return os.path.abspath(output_path)
