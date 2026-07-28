"""HTML report generator for NSGA-III IRP results."""

import json
import math
import os
import subprocess
import sys
import webbrowser

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IRP &mdash; NSGA-III</title>
<style>
:root {
  --bg:      #f4f4f2;
  --surface: #ffffff;
  --border:  #e5e5e3;
  --text:    #1a1a18;
  --text-2:  #888;
  --tab-bg:  #ececea;
  --row-bg:  #f4f4f2;
  --accent:  #534ab7;
  --accent2: #185fa5;
  --f1-bg:#e6f1fb; --f1-fg:#185fa5;
  --f2-bg:#e8f5e2; --f2-fg:#3b6d11;
  --f3-bg:#fdecea; --f3-fg:#a32d2d;
  --f4-bg:#eeedfe; --f4-fg:#534ab7;
  --si-bg:#e6f1fb; --si-fg:#185fa5;
  --sr-bg:#e8f5e2; --sr-fg:#3b6d11;
  --sd-bg:#fdecea; --sd-fg:#a32d2d;
  --sf-bg:#eeedfe; --sf-fg:#534ab7;
}
[data-dark] {
  --bg:      #111113;
  --surface: #1c1c20;
  --border:  #2c2c34;
  --text:    #ededed;
  --text-2:  #777;
  --tab-bg:  #232328;
  --row-bg:  #222226;
  --accent:  #9f9cf5;
  --accent2: #6aaae8;
  --f1-bg:#1a2a3d; --f1-fg:#6aaae8;
  --f2-bg:#1a2e1a; --f2-fg:#6abd52;
  --f3-bg:#2e1a1a; --f3-fg:#e06060;
  --f4-bg:#2e2b52; --f4-fg:#9f9cf5;
  --si-bg:#1a2a3d; --si-fg:#6aaae8;
  --sr-bg:#1a2e1a; --sr-fg:#6abd52;
  --sd-bg:#2e1a1a; --sd-fg:#e06060;
  --sf-bg:#2e2b52; --sf-fg:#9f9cf5;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
  background:var(--bg);color:var(--text);
  padding:1.25rem 1.75rem;font-size:14px;line-height:1.5;
  transition:background .2s,color .2s;
}
/* ── Header ── */
.hdr{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:1.1rem;gap:1rem}
.hdr h1{font-size:20px;font-weight:700;letter-spacing:-.015em}
.hdr .meta{color:var(--text-2);font-size:12px;margin-top:3px}
#tbtn{background:var(--surface);border:1px solid var(--border);border-radius:8px;
  padding:5px 11px;cursor:pointer;font-size:15px;color:var(--text);transition:background .15s}
#tbtn:hover{background:var(--tab-bg)}
/* ── Cards ── */
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;
  padding:1.1rem 1.25rem;margin-bottom:1.1rem}
.ct{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;
  color:var(--text-2);margin-bottom:.85rem}
/* ── Summary bar ── */
.sbar{display:flex;flex-wrap:wrap;gap:18px}
.sp{font-size:12px;color:var(--text-2)}
.sp b{color:var(--text)}
/* ── KPI row ── */
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:1.1rem}
@media(max-width:740px){.kpis{grid-template-columns:1fr 1fr}}
.kpi{border-radius:10px;padding:.85rem 1.1rem;border:1px solid var(--border)}
.kpi.f1{background:var(--f1-bg)} .kpi.f2{background:var(--f2-bg)}
.kpi.f3{background:var(--f3-bg)} .kpi.f4{background:var(--f4-bg)}
.kl{font-size:11px;margin-bottom:4px}
.kpi.f1 .kl{color:var(--f1-fg)} .kpi.f2 .kl{color:var(--f2-fg)}
.kpi.f3 .kl{color:var(--f3-fg)} .kpi.f4 .kl{color:var(--f4-fg)}
.kv{font-size:20px;font-weight:700;line-height:1;margin-bottom:2px;color:var(--text)}
.ku{font-size:11px;color:var(--text-2)}
/* ── Pareto chart ── */
.pareto-wrap{overflow-x:auto}
.pareto-wrap svg{display:block;min-width:480px;min-height:320px}
.hint{font-size:11px;color:var(--text-2);margin-top:.5rem}
/* ── Solution selector ── */
.sol-hdr{display:flex;align-items:center;gap:10px;margin-bottom:.85rem;flex-wrap:wrap}
.sol-hdr h2{font-size:14px;font-weight:700}
.sol-nav{display:flex;gap:5px}
.snav{background:var(--tab-bg);border:1px solid var(--border);border-radius:6px;
  padding:3px 10px;cursor:pointer;font-size:12px;color:var(--text);transition:background .15s}
.snav:hover{background:var(--border)}
/* ── Run selector ── */
.run-btn{background:var(--tab-bg);border:1px solid var(--border);border-radius:6px;
  padding:4px 12px;cursor:pointer;font-size:12px;color:var(--text);
  transition:background .15s,color .15s,border-color .15s}
.run-btn:hover{background:var(--border)}
.run-btn.active{background:var(--accent);color:#fff;border-color:var(--accent)}
/* ── Comparison table footer ── */
tfoot td{border-top:2px solid var(--border);font-size:11px;color:var(--text-2);
  font-style:italic;padding:5px 8px}
/* ── Split layout ── */
.split{display:grid;grid-template-columns:1fr 1fr;gap:1.1rem;margin-bottom:1.1rem}
@media(max-width:860px){.split{grid-template-columns:1fr}}
/* ── Period tabs ── */
.tabs{display:flex;gap:5px;margin-bottom:.8rem;flex-wrap:wrap}
.tab{padding:4px 12px;border-radius:7px;font-size:11px;cursor:pointer;
  border:1px solid var(--border);background:var(--tab-bg);color:var(--text-2);
  transition:background .15s,color .15s}
.tab.active{background:var(--text);color:var(--bg);border-color:var(--text)}
/* ── SVG network ── */
.svgwrap{overflow:hidden;border-radius:8px;background:var(--bg);border:1px solid var(--border)}
.svgwrap svg{display:block;width:100%;height:auto}
/* ── Tables ── */
table{width:100%;border-collapse:collapse;font-size:12px}
th{text-align:left;padding:6px 8px;border-bottom:2px solid var(--border);
  font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--text-2)}
td{padding:5px 8px;border-bottom:1px solid var(--border)}
tr:last-child td{border-bottom:none}
tr:nth-child(even) td{background:var(--row-bg)}
.badge{display:inline-block;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:600}
.ok{background:var(--f2-bg);color:var(--f2-fg)}
.advance{background:var(--f4-bg);color:var(--f4-fg)}
.partial{background:var(--f3-bg);color:var(--f3-fg)}
.none{background:var(--tab-bg);color:var(--text-2)}
/* ── Depot stock ── */
.sb{margin-bottom:1rem}
.sp2{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;
  color:var(--text-2);margin-bottom:.5rem}
.seq{display:flex;align-items:stretch;flex-wrap:wrap;gap:2px}
.sc{display:flex;flex-direction:column;align-items:center;justify-content:center;
  padding:9px 12px;border-radius:9px;min-width:64px;text-align:center;flex:1}
.sl{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;
  margin-bottom:2px;opacity:.75}
.sv{font-size:16px;font-weight:700;line-height:1}
.sop{display:flex;align-items:center;justify-content:center;
  font-size:18px;font-weight:300;color:var(--text-2);padding:0 3px;flex-shrink:0}
.si{background:var(--si-bg);color:var(--si-fg)}
.sr{background:var(--sr-bg);color:var(--sr-fg)}
.sd{background:var(--sd-bg);color:var(--sd-fg)}
.sf{background:var(--sf-bg);color:var(--sf-fg)}
/* ── BFR breakdown ── */
.bfr-row{display:grid;grid-template-columns:1.6fr 2fr 1fr;
  padding:6px 0;border-bottom:1px solid var(--border);font-size:13px;align-items:center}
.bfr-header{font-size:11px;font-weight:600;text-transform:uppercase;
  letter-spacing:.04em;color:var(--text-2);border-bottom:2px solid var(--border)}
.bfr-total{font-weight:700;border-top:2px solid var(--border);border-bottom:none;margin-top:2px}
.bfr-formula{color:var(--text-2);font-style:italic;font-size:12px}
.bfr-val{text-align:right;font-variant-numeric:tabular-nums}
</style>
</head>
<body>
<main>

<!-- Header -->
<div class="hdr">
  <div>
    <h1>IRP &mdash; NSGA-III</h1>
    <div class="meta" id="metaLine"></div>
  </div>
  <button id="tbtn" onclick="toggleDark()">&#9680;</button>
</div>

<!-- Algorithm summary -->
<div class="card">
  <div class="ct">Algorithm parameters</div>
  <div class="sbar" id="summaryBar"></div>
</div>

<!-- Quality metrics -->
<div class="card">
  <div class="ct">Pareto-front quality indicators</div>
  <div class="sbar" id="qualityBar"></div>
</div>

<!-- Runs comparison (shown only when n_runs > 1) -->
<div class="card" id="runsCompCard" style="display:none">
  <div class="ct">Runs comparison &amp; stability</div>
  <div id="runsCompTable"></div>
</div>

<!-- Run selector (shown only when n_runs > 1) -->
<div id="runSelectorBlock" style="display:none;margin-bottom:1.1rem">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--text-2);margin-bottom:.6rem">Explore run</div>
  <div id="runBtns" style="display:flex;gap:6px;flex-wrap:wrap"></div>
</div>

<!-- Pareto front -->
<div class="card">
  <div class="ct">Pareto front &mdash; parallel coordinates (click a line to explore that solution)</div>
  <div class="pareto-wrap"><div id="paretoChart"></div></div>
  <p class="hint" id="hintText"></p>
</div>

<!-- Selected solution header -->
<div class="sol-hdr">
  <h2 id="solTitle"></h2>
  <div class="sol-nav">
    <button class="snav" onclick="stepSolution(-1)">&#8592; Prev</button>
    <button class="snav" onclick="stepSolution(1)">Next &#8594;</button>
  </div>
</div>

<!-- KPI row -->
<div class="kpis" id="kpiRow"></div>

<!-- Route network + Depot stock -->
<div class="split">
  <div class="card">
    <div class="ct">Route network</div>
    <div class="tabs" id="periodTabs"></div>
    <div class="svgwrap" id="svgWrap"></div>
    <div id="tourBlock"></div>
  </div>
  <div class="card">
    <div class="ct">Depot stock evolution</div>
    <div id="depotBlock"></div>
  </div>
</div>

<!-- Deliveries -->
<div class="card">
  <div class="ct">Deliveries per client &amp; period</div>
  <div id="delivTable"></div>
</div>

<!-- BFR -->
<div class="card">
  <div class="ct">Working capital breakdown (f4)</div>
  <div id="bfrBlock"></div>
</div>

</main>
<script>
const D = /*DATA_PLACEHOLDER*/null;
const M    = D.meta;
const RUNS = D.runs;
const STAB = D.stability;

/* ── Theme ─────────────────────────────────────────────── */
function toggleDark(){
  const on = document.documentElement.toggleAttribute('data-dark');
  localStorage.setItem('nsga3-dark', on ? '1' : '');
  renderPareto();
  renderNetwork(currentPeriod);
}
(function(){ if(localStorage.getItem('nsga3-dark')==='1') document.documentElement.setAttribute('data-dark',''); })();

/* ── Truck colours ──────────────────────────────────────── */
const TC = ['#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6','#06b6d4'];
function tc(k){ return TC[(k-1) % TC.length]; }

/* ── Number formatter ───────────────────────────────────── */
function fmt(v){
  if(v==null) return '—';
  const a=Math.abs(v);
  if(a>=1e6) return (v/1e6).toFixed(2)+'M';
  if(a>=1e3) return (v/1e3).toFixed(2)+'k';
  return typeof v.toFixed==='function' ? v.toFixed(3) : String(v);
}

/* ── Per-run state ──────────────────────────────────────── */
let currentRunIdx = 0;
let selectedIdx   = 0;
let currentPeriod = null;
let SOLS     = RUNS[0].solutions;
let RUN_META = RUNS[0].meta;

function switchRun(i){
  currentRunIdx = i;
  SOLS     = RUNS[i].solutions;
  RUN_META = RUNS[i].meta;
  selectedIdx   = 0;
  currentPeriod = null;
  renderAll();
}

/* ── Meta header ────────────────────────────────────────── */
document.getElementById('metaLine').textContent =
  M.instance.replace('_',' ') + '  ·  ' + M.n_nodes + ' nodes  ·  ' +
  M.n_periods + ' periods  ·  ' + M.n_vehicles + ' vehicles';

/* ── Summary bar ────────────────────────────────────────── */
function renderSummaryBar(){
  const items = [
    ['Population',      M.pop_size],
    ['Generations',     M.n_gen],
    ['Crossover (SBX)', M.crossover_prob],
    ['Mutation (PM)',   M.mutation_prob],
    ['Pareto solutions',RUN_META.n_pareto],
    ['Runtime',         RUN_META.elapsed_s + 's'],
  ];
  document.getElementById('summaryBar').innerHTML = items.map(([k,v]) =>
    `<span class="sp"><b>${v}</b> ${k}</span>`).join('');
}

/* ── Quality metrics bar ────────────────────────────────── */
function renderQualityBar(){
  const Q = RUN_META.quality || {};
  const fmt6 = v => (v == null ? '—' : Number(v).toFixed(6));
  const items = [
    ['HV ↑',      fmt6(Q.HV)],
    ['GD ↓',      fmt6(Q.GD)],
    ['IGD ↓',     fmt6(Q.IGD)],
    ['Spacing ↓', fmt6(Q.Spacing)],
  ];
  document.getElementById('qualityBar').innerHTML = items.map(([k,v]) =>
    `<span class="sp"><b>${v}</b> ${k}</span>`).join('');
}

/* ── Runs comparison & selector ─────────────────────────── */
function renderRunsComparison(){
  const card  = document.getElementById('runsCompCard');
  const block = document.getElementById('runSelectorBlock');
  if(RUNS.length <= 1){ card.style.display='none'; block.style.display='none'; return; }
  card.style.display=''; block.style.display='';

  // Run selector buttons
  document.getElementById('runBtns').innerHTML = RUNS.map((r,i) => {
    const label = `Run ${r.meta.run_id||i+1}`;
    return `<button class="run-btn${i===currentRunIdx?' active':''}" onclick="switchRun(${i})">
      ${label}
      <span style="font-size:10px;opacity:.7;margin-left:5px">${r.meta.n_pareto} sol.</span>
    </button>`;
  }).join('');

  // Comparison table
  const fmt6 = v => (v==null?'—':Number(v).toFixed(6));
  const fmtS = s => (s&&s.mean!=null
    ? `${Number(s.mean).toFixed(6)}<br><span style="opacity:.7">± ${Number(s.std).toFixed(6)}</span>`
    : '—');
  const fmtN = s => (s&&s.mean!=null
    ? `${Number(s.mean).toFixed(1)}<br><span style="opacity:.7">± ${Number(s.std).toFixed(1)}</span>`
    : '—');

  const rows = RUNS.map((r,i) => {
    const q  = r.meta.quality || {};
    const hl = i === currentRunIdx ? 'background:var(--f4-bg)' : '';
    return `<tr${hl?' style="'+hl+'"':''}>
      <td><b>Run ${r.meta.run_id||i+1}</b></td>
      <td>${r.meta.seed||'—'}</td>
      <td>${r.meta.n_pareto}</td>
      <td>${fmt6(q.HV)}</td>
      <td>${fmt6(q.GD)}</td>
      <td>${fmt6(q.IGD)}</td>
      <td>${fmt6(q.Spacing)}</td>
      <td>${r.meta.elapsed_s}s</td>
    </tr>`;
  }).join('');

  const S = STAB||{};
  document.getElementById('runsCompTable').innerHTML = `
    <table>
      <thead><tr>
        <th>Run</th><th>Seed</th><th>Pareto</th>
        <th>HV ↑</th><th>GD ↓</th><th>IGD ↓</th><th>Spacing ↓</th><th>Time</th>
      </tr></thead>
      <tbody>${rows}</tbody>
      <tfoot><tr>
        <td colspan="2"><b>Mean ± Std</b></td>
        <td>${fmtN(S.n_pareto)}</td>
        <td>${fmtS(S.HV)}</td>
        <td>${fmtS(S.GD)}</td>
        <td>${fmtS(S.IGD)}</td>
        <td>${fmtS(S.Spacing)}</td>
        <td>—</td>
      </tr></tfoot>
    </table>`;
}

/* ── Pareto parallel coordinates ────────────────────────── */
function renderPareto(){
  const W=860, H=380, MT=58, MB=36, ML=28, MR=50;
  const CH = H - MT - MB;
  const AXES = ['f1  Logistics','f2  CO₂','f3  Travel time','f4  BFR (min)'];
  const AX = [0,1,2,3].map(i => ML + i*(W-ML-MR)/3);

  const keys=['f1','f2','f3','f4'];
  const mins=keys.map(k=>Math.min(...SOLS.map(s=>s.objectives[k])));
  const maxs=keys.map(k=>Math.max(...SOLS.map(s=>s.objectives[k])));

  function norm(val,j){
    const r=maxs[j]-mins[j];
    if(r<1e-9) return 0.5;
    return (val-mins[j])/r;
  }
  function yp(n){ return MT + n*CH; }

  const isDark=document.documentElement.hasAttribute('data-dark');
  const axColor  = isDark ? '#3a3a44' : '#ddd';
  const selColor = isDark ? '#9f9cf5' : '#534ab7';
  const lblColor = isDark ? '#999'    : '#666';
  const valColor = isDark ? '#ccc'    : '#444';
  const n = SOLS.length;

  function solColor(i){
    const h = Math.round(240 - (i / Math.max(n-1,1)) * 200);
    return isDark ? `hsl(${h},65%,65%)` : `hsl(${h},65%,42%)`;
  }

  let svg = '';

  AX.forEach((ax,j)=>{
    svg += `<line x1="${ax}" y1="${MT}" x2="${ax}" y2="${H-MB}" stroke="${axColor}" stroke-width="2"/>`;
    [0.25,0.5,0.75].forEach(t=>{
      svg += `<line x1="${ax-4}" y1="${yp(t)}" x2="${ax+4}" y2="${yp(t)}" stroke="${axColor}" stroke-width="1"/>`;
    });
    svg += `<text x="${ax}" y="${MT-18}" text-anchor="middle" font-size="11" font-weight="700" fill="${lblColor}">${AXES[j]}</text>`;
    const best  = j===3 ? fmt(maxs[j]) : fmt(mins[j]);
    const worst = j===3 ? fmt(mins[j]) : fmt(maxs[j]);
    svg += `<text x="${ax}" y="${MT-5}" text-anchor="middle" font-size="9.5" fill="${valColor}">${best}</text>`;
    svg += `<text x="${ax}" y="${H-MB+13}" text-anchor="middle" font-size="9.5" fill="${valColor}">${worst}</text>`;
  });

  SOLS.forEach((sol,i)=>{
    if(i===selectedIdx) return;
    const c   = solColor(i);
    const pts = AX.map((ax,j)=>`${ax},${yp(norm(sol.objectives[keys[j]],j))}`).join(' ');
    svg += `<polyline points="${pts}" fill="none" stroke="${c}" stroke-width="1.5"
      opacity="0.3" style="cursor:pointer;transition:opacity .12s,stroke-width .12s"
      onmouseenter="this.setAttribute('stroke-width','3');this.style.opacity='0.95'"
      onmouseleave="this.setAttribute('stroke-width','1.5');this.style.opacity='0.3'"
      onclick="selectSolution(${i})"/>`;
  });

  const sel=SOLS[selectedIdx];
  const selPts=AX.map((ax,j)=>`${ax},${yp(norm(sel.objectives[keys[j]],j))}`).join(' ');
  svg += `<polyline points="${selPts}" fill="none" stroke="${selColor}" stroke-width="3.5"/>`;
  AX.forEach((ax,j)=>{
    const cy=yp(norm(sel.objectives[keys[j]],j));
    svg += `<circle cx="${ax}" cy="${cy}" r="5.5" fill="${selColor}" stroke="var(--surface)" stroke-width="2"/>`;
  });

  document.getElementById('paretoChart').innerHTML =
    `<svg viewBox="0 0 ${W} ${H}" style="width:100%;display:block">${svg}</svg>`;
  document.getElementById('hintText').textContent = '';
}

/* ── KPI cards ──────────────────────────────────────────── */
function renderKPIs(){
  const o=SOLS[selectedIdx].objectives;
  const defs=[
    {cls:'f1',label:'f1 — Logistics cost',  val:o.f1, unit:'cost units'},
    {cls:'f2',label:'f2 — CO₂ emissions',   val:o.f2, unit:'kg CO₂'},
    {cls:'f3',label:'f3 — Travel time',     val:o.f3, unit:'hours'},
    {cls:'f4',label:'f4 — Working capital', val:o.f4, unit:'currency  (↓ min)'},
  ];
  document.getElementById('kpiRow').innerHTML=defs.map(d=>
    `<div class="kpi ${d.cls}">
       <div class="kl">${d.label}</div>
       <div class="kv">${fmt(d.val)}</div>
       <div class="ku">${d.unit}</div>
     </div>`).join('');
}

/* ── Solution title ─────────────────────────────────────── */
function renderSolTitle(){
  document.getElementById('solTitle').textContent =
    `Solution ${selectedIdx+1} of ${SOLS.length}`;
}

/* ── Period tabs ────────────────────────────────────────── */
function renderPeriodTabs(){
  const routes=SOLS[selectedIdx].routes;
  const periods=Object.keys(routes).map(Number).sort((a,b)=>a-b);
  if(currentPeriod===null || !periods.includes(currentPeriod)) currentPeriod=periods[0];
  const wrap=document.getElementById('periodTabs');
  wrap.innerHTML=periods.map(t=>
    `<div class="tab${t===currentPeriod?' active':''}" onclick="selectPeriod(${t})">Period ${t}</div>`
  ).join('');
}
function selectPeriod(t){
  currentPeriod=t;
  document.querySelectorAll('#periodTabs .tab').forEach(el=>{
    el.classList.toggle('active',el.textContent===`Period ${t}`);
  });
  renderNetwork(t);
}

/* ── Route network SVG ──────────────────────────────────── */
function renderNetwork(t){
  const pos=M.node_positions;
  const nodes=Object.keys(pos).map(Number);
  const period=SOLS[selectedIdx].routes[String(t)]||{};
  const trucks=period.trucks||[];

  let maxX=0,maxY=0;
  nodes.forEach(n=>{maxX=Math.max(maxX,pos[String(n)][0]);maxY=Math.max(maxY,pos[String(n)][1]);});
  const W=maxX+60,H=maxY+50;

  const isDark=document.documentElement.hasAttribute('data-dark');
  const nodeFill   =isDark?'#1c1c20':'#ffffff';
  const nodeStroke =isDark?'#8494ab':'#555';
  const textFill   =isDark?'#ededed':'#1a1a18';
  const bgFill     =isDark?'#111113':'#f4f4f2';
  const depotFill  =isDark?'#9f9cf5':'#534ab7';

  const markerDefs=trucks.map(tr=>{
    const col=tc(tr.k);
    return `<marker id="arrN3_${tr.k}" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto">
      <polygon points="0 0,7 3.5,0 7" fill="${col}" opacity="0.85"/></marker>`;
  }).join('');

  let arcs='';
  const seen={};
  trucks.forEach(tr=>{
    const col=tc(tr.k);
    const path=tr.path;
    const qty=tr.qty||{};
    for(let i=0;i<path.length-1;i++){
      const a=path[i],b=path[i+1];
      const key=a+'-'+b;
      if(seen[key]) continue;
      seen[key]=true;
      const [x1,y1]=pos[String(a)],[x2,y2]=pos[String(b)];
      const mx=(x1+x2)/2,my=(y1+y2)/2;
      const nx=-(y2-y1),ny=x2-x1;
      const len=Math.sqrt(nx*nx+ny*ny)||1;
      const cx=mx+nx/len*22,cy=my+ny/len*22;
      const delivered=qty[String(b)]||0;
      arcs+=`<path d="M${x1},${y1} Q${cx},${cy} ${x2},${y2}"
        fill="none" stroke="${col}" stroke-width="2.2" stroke-linecap="round"
        marker-end="url(#arrN3_${tr.k})" opacity="0.82"/>`;
      if(delivered>0)
        arcs+=`<text x="${cx}" y="${cy-5}" text-anchor="middle" font-size="9" fill="${col}" font-weight="600">${delivered}</text>`;
    }
    if(path.length>0){
      const last=path[path.length-1];
      if(last!==0){
        const [x1,y1]=pos[String(last)],[x2,y2]=pos['0'];
        const mx=(x1+x2)/2,my=(y1+y2)/2;
        const nx=-(y2-y1),ny=x2-x1;
        const len=Math.sqrt(nx*nx+ny*ny)||1;
        const cx=mx+nx/len*18,cy=my+ny/len*18;
        arcs+=`<path d="M${x1},${y1} Q${cx},${cy} ${x2},${y2}"
          fill="none" stroke="${col}" stroke-width="1.4" stroke-dasharray="5,3" opacity="0.5"/>`;
      }
    }
  });

  const nodesSvg=nodes.map(n=>{
    const [x,y]=pos[String(n)];
    const isD=(n===0);
    const fill=isD?depotFill:nodeFill;
    const stroke=isD?depotFill:nodeStroke;
    const tFill=isD?'#fff':textFill;
    const r=isD?17:14;
    return `<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="1.8"/>
      <text x="${x}" y="${y+4}" text-anchor="middle" font-size="${isD?11:10}" font-weight="700" fill="${tFill}">${isD?'D':n}</text>`;
  }).join('');

  let lgd=`<div style="display:flex;gap:5px;flex-wrap:wrap;margin-top:8px;align-items:center">`;
  if(period.tau_return!=null)
    lgd+=`<span style="font-size:11px;color:var(--text-2);padding:3px 10px;background:var(--row-bg);border:1px solid var(--border);border-radius:99px">&#128339; Return: ${period.tau_return}h</span>`;
  trucks.forEach(tr=>{
    const c=tc(tr.k);
    lgd+=`<span style="display:inline-flex;align-items:center;gap:4px;padding:3px 9px 3px 6px;background:var(--row-bg);border:1px solid var(--border);border-radius:99px;font-size:11px">
      <span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${c}"></span>
      <b style="color:var(--text)">k=${tr.k}</b></span>`;
  });
  lgd+=`</div>`;

  document.getElementById('svgWrap').innerHTML=
    `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">
       <defs>${markerDefs}</defs>
       <rect width="${W}" height="${H}" fill="${bgFill}"/>
       ${arcs}${nodesSvg}
     </svg>${lgd}`;

  let tourHtml='';
  if(trucks.length>0){
    tourHtml+=`<div style="margin-top:10px;padding-top:8px;border-top:1px solid var(--border)">`;
    tourHtml+=`<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--text-2);margin-bottom:6px">Delivery tours — period ${t}</div>`;
    trucks.forEach(tr=>{
      const c=tc(tr.k);
      const nodes=tr.path.slice();
      if(nodes.length>0 && nodes[nodes.length-1]!==0) nodes.push(0);
      const stops=nodes.map(n=>n===0?'D':String(n));
      const stopsHtml=stops.map((s,i)=>{
        const isD=(s==='D');
        const chip=`<span style="padding:2px 7px;border-radius:6px;font-size:11px;font-weight:${isD?700:500};background:${isD?c:'var(--row-bg)'};color:${isD?'#fff':'var(--text)'};border:1px solid ${isD?c:'var(--border)'}">${s}</span>`;
        return i<stops.length-1?chip+`<span style="color:var(--text-2);font-size:12px;margin:0 1px">&#8594;</span>`:chip;
      }).join('');
      tourHtml+=`<div style="display:flex;align-items:center;gap:4px;flex-wrap:wrap;margin-bottom:5px">
        <span style="display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:99px;background:var(--row-bg);border:1px solid var(--border);font-size:11px;flex-shrink:0">
          <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${c}"></span>
          <b style="color:${c}">k=${tr.k}</b>
        </span>
        ${stopsHtml}
      </div>`;
    });
    tourHtml+=`</div>`;
  }
  document.getElementById('tourBlock').innerHTML=tourHtml;
}

/* ── Depot stock ────────────────────────────────────────── */
function renderDepot(){
  const stock=SOLS[selectedIdx].depot_stock;
  const routes=SOLS[selectedIdx].routes;
  const periods=Object.keys(stock).map(Number).sort((a,b)=>a-b);
  let pf=M.I_O_init_frigo, pnf=M.I_O_init_nonfrigo;

  function row(label,prev,R,fin){
    const livr=(typeof prev==='number'&&typeof fin==='number') ? Math.round((prev+R-fin)*100)/100 : '?';
    return `<div style="margin-bottom:5px">
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

  document.getElementById('depotBlock').innerHTML=periods.map(t=>{
    const s=stock[String(t)];
    const r=routes[String(t)]||{};
    const html=`<div class="sb">
      <div class="sp2">Period ${t}</div>
      ${row('&#10052;&#65039; Refrigerated', pf,  r.R_frigo||0,  s?s.frigo:'?')}
      ${row('&#128230; Non-refrigerated',   pnf, r.R_nonfrigo||0, s?s.nonfrigo:'?')}
    </div>`;
    pf=s?s.frigo:pf; pnf=s?s.nonfrigo:pnf;
    return html;
  }).join('');
}

/* ── Deliveries table ───────────────────────────────────── */
function renderDeliveries(){
  const rows=(SOLS[selectedIdx].deliveries||[]).map(d=>{
    const cumRecu=d.cum_recu??d.recu;
    const cumDem=d.cum_dem??d.dem;
    const balance=d.balance??(cumRecu-cumDem);
    const ratio=cumDem>0?cumRecu/cumDem:1;
    const veh=d.recu>0 && d.k>0
      ?`<span style="color:${tc(d.k)};font-weight:600">Truck ${d.k}</span>`
      :'<span style="color:var(--text-2)">—</span>';
    const badge=balance>0
      ?`<span class="badge advance">Ahead +${balance}</span>`
      :ratio>=0.999
      ?'<span class="badge ok">Full</span>'
      :cumRecu>0
      ?`<span class="badge partial">${(ratio*100).toFixed(0)}%</span>`
      :'<span class="badge none">None</span>';
    return `<tr>
      <td>Client ${d.l}</td><td>t=${d.t}</td>
      <td>${veh}</td>
      <td>${d.recu}</td><td>${d.dem}</td>
      <td>${cumRecu} / ${cumDem}</td><td>${badge}</td>
    </tr>`;
  }).join('');
  document.getElementById('delivTable').innerHTML=`
    <table>
      <thead><tr>
        <th>Client</th><th>Period</th><th>Vehicle</th>
        <th>Delivered</th><th>Demand (t)</th><th>Cumulative</th><th>Status</th>
      </tr></thead>
      <tbody>${rows||'<tr><td colspan="7" style="color:var(--text-2);padding:8px">No deliveries.</td></tr>'}</tbody>
    </table>`;
}

/* ── BFR breakdown ──────────────────────────────────────── */
function renderBFR(){
  const b=SOLS[selectedIdx].bfr_sub;
  if(!b){ document.getElementById('bfrBlock').innerHTML=''; return; }
  const net=(b.stock+b.receivables-b.payables).toFixed(4);
  document.getElementById('bfrBlock').innerHTML=`
    <div class="bfr-row bfr-header">
      <span>Component</span><span>Formula</span><span style="text-align:right">Value</span>
    </div>
    <div class="bfr-row">
      <span>Stock value</span>
      <span class="bfr-formula">I &times; P<sub>purchase</sub> &times; DIO / 365</span>
      <span class="bfr-val">${b.stock}</span>
    </div>
    <div class="bfr-row">
      <span>+ Accounts receivable</span>
      <span class="bfr-formula">q &times; P<sub>sale</sub> &times; DSO / 365</span>
      <span class="bfr-val">${b.receivables}</span>
    </div>
    <div class="bfr-row">
      <span>&#8722; Accounts payable</span>
      <span class="bfr-formula">q &times; P<sub>purchase</sub> &times; DPO / 365</span>
      <span class="bfr-val">&#8722;${b.payables}</span>
    </div>
    <div class="bfr-row bfr-total">
      <span>f4 = BFR</span>
      <span class="bfr-formula">Stock + Receivables &#8722; Payables</span>
      <span class="bfr-val">${net}</span>
    </div>`;
}

/* ── Select a solution ──────────────────────────────────── */
function selectSolution(i){
  selectedIdx=i;
  currentPeriod=null;
  renderAll();
}
function stepSolution(dir){
  selectSolution((selectedIdx+dir+SOLS.length)%SOLS.length);
}

/* ── Full render ────────────────────────────────────────── */
function renderAll(){
  renderRunsComparison();
  renderSummaryBar();
  renderQualityBar();
  renderPareto();
  renderSolTitle();
  renderKPIs();
  renderPeriodTabs();
  renderNetwork(currentPeriod);
  renderDepot();
  renderDeliveries();
  renderBFR();
}

renderAll();
</script>
</body>
</html>"""


# ── Node layout (same logic as FunctionMerge) ────────────────────────────────
def compute_node_positions(N, svg_w=480, svg_h=290):
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
    else:
        cx = svg_w * 0.58
        cy = svg_h / 2
        rx = svg_w * 0.34
        ry = svg_h * 0.39
        for i, c in enumerate(clients):
            angle   = -math.pi / 2 + 2 * math.pi * i / nc
            pos[c]  = [int(cx + rx * math.cos(angle)),
                       int(cy + ry * math.sin(angle))]

    return {str(k): v for k, v in pos.items()}


# ── Chrome opener ─────────────────────────────────────────────────────────────
def _open_chrome(url):
    if sys.platform == "win32":
        candidates = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expanduser(r"~\AppData\Local\Google\Chrome\Application\chrome.exe"),
        ]
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe",
            )
            rp = winreg.QueryValue(key, None)
            if rp:
                candidates.insert(0, rp)
        except Exception:
            pass
        for path in candidates:
            if os.path.exists(path):
                subprocess.Popen([path, url])
                return
    elif sys.platform == "darwin":
        try:
            subprocess.Popen(["open", "-a", "Google Chrome", url])
            return
        except Exception:
            pass
    webbrowser.open(url)


# ── Public API ────────────────────────────────────────────────────────────────

def render_html(data, algo_label: str = "NSGA-III"):
    """Return the complete HTML string for the given data dict.

    algo_label overrides the 'NSGA-III' heading shown in the browser tab
    and the report title bar (e.g. 'QI-NSGA-III').
    """
    html = _TEMPLATE.replace(
        "/*DATA_PLACEHOLDER*/null",
        json.dumps(data, ensure_ascii=False),
    )
    if algo_label != "NSGA-III":
        html = html.replace("IRP &mdash; NSGA-III", f"IRP &mdash; {algo_label}", 2)
    return html


def write_report(data, output_path, algo_label: str = "NSGA-III"):
    """Write HTML report to disk (used by standalone CLI runs)."""
    html = render_html(data, algo_label)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return os.path.abspath(output_path)


def generate_and_open(output_path):
    abs_path = os.path.abspath(output_path).replace("\\", "/")
    _open_chrome(f"file:///{abs_path}")
    return output_path
