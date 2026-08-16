"""HTML report generator for IRP calibration results."""

import json
import math
import os

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IRP &mdash; Dashboard</title>
<style>
:root {
  --bg:         #f4f4f2;
  --surface:    #ffffff;
  --border:     #e5e5e3;
  --text:       #1a1a18;
  --text-2:     #888;
  --text-3:     #ccc;
  --tab-bg:     #ececea;
  --row-bg:     #f4f4f2;
  --mesh:       #e0e0dd;
  --node-surf:  #ffffff;
  --si-bg:#e6f1fb; --si-fg:#185fa5;
  --sr-bg:#e8f5e2; --sr-fg:#3b6d11;
  --sd-bg:#fdecea; --sd-fg:#a32d2d;
  --sf-bg:#eeedfe; --sf-fg:#534ab7;
}
[data-dark] {
  --bg:         #111113;
  --surface:    #1c1c20;
  --border:     #2c2c34;
  --text:       #ededed;
  --text-2:     #777;
  --text-3:     #3a3a44;
  --tab-bg:     #232328;
  --row-bg:     #222226;
  --mesh:       #252528;
  --node-surf:  #1c1c20;
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
.hdr{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:1.25rem;gap:1rem}
.hdr h1{font-size:20px;font-weight:700;letter-spacing:-.015em}
.hdr .meta{color:var(--text-2);font-size:12px;margin-top:3px}
#tbtn{background:var(--surface);border:1px solid var(--border);border-radius:8px;
  padding:5px 11px;cursor:pointer;font-size:15px;color:var(--text);transition:background .15s}
#tbtn:hover{background:var(--tab-bg)}
/* ── Cards ── */
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1.1rem 1.25rem;margin-bottom:1.1rem}
.ct{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--text-2);margin-bottom:.9rem}
/* ── KPI ── */
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:1.1rem}
@media(max-width:680px){.kpis{grid-template-columns:1fr 1fr}}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:.9rem 1.1rem}
.kl{font-size:11px;color:var(--text-2);margin-bottom:5px}
.kv{font-size:22px;font-weight:700;line-height:1;margin-bottom:3px}
.kb{font-size:11px;color:var(--text-2)}
/* ── Split row (network | depot) ── */
.split{display:grid;grid-template-columns:1fr 1fr;gap:1.1rem;margin-bottom:1.1rem}
.split.stacked{grid-template-columns:1fr}
@media(max-width:860px){.split{grid-template-columns:1fr}}
/* ── Tabs ── */
.tabs{display:flex;gap:5px;margin-bottom:.85rem;flex-wrap:wrap}
.tab{padding:4px 12px;border-radius:7px;font-size:11px;cursor:pointer;
  border:1px solid var(--border);background:var(--tab-bg);color:var(--text-2);
  user-select:none;transition:opacity .12s}
.tab:hover{opacity:.8}
.tab.active{background:#eeedfe;border-color:#c4c0f5;color:#534ab7;font-weight:600}
.tab.period.active{background:#e6f1fb;border-color:#90c4f5;color:#185fa5;font-weight:600}
[data-dark] .tab.active{background:#2e2b52;border-color:#534ab7;color:#9f9cf5}
[data-dark] .tab.period.active{background:#1a2a3d;border-color:#185fa5;color:#6aaae8}
/* ── SVG ── */
.nsvg{width:100%;display:block;min-height:360px;border-radius:8px}
/* ── Delivery table ── */
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;padding:7px 9px;color:var(--text-2);font-weight:700;font-size:11px;
  text-transform:uppercase;letter-spacing:.05em;border-bottom:2px solid var(--border)}
td{padding:7px 9px;border-bottom:1px solid var(--border)}
tr:last-child td{border-bottom:none}
.ok{color:#3b6d11;font-weight:600} .ante{color:#854f0b;font-weight:600} .part{color:#a32d2d;font-weight:600}
/* ── Stock equation ── */
.sb{margin-bottom:1rem}
.sp{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;
  color:var(--text-2);margin-bottom:.6rem}
.seq{display:flex;align-items:stretch;gap:0;flex-wrap:wrap;gap:2px}
.sc{display:flex;flex-direction:column;align-items:center;justify-content:center;
  padding:10px 14px;border-radius:9px;min-width:76px;text-align:center;flex:1}
.sl{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;
  margin-bottom:3px;opacity:.75}
.sv{font-size:19px;font-weight:700;line-height:1}
.sop{display:flex;align-items:center;justify-content:center;
  font-size:20px;font-weight:300;color:var(--text-2);padding:0 4px;flex-shrink:0}
.si{background:var(--si-bg);color:var(--si-fg)}
.sr{background:var(--sr-bg);color:var(--sr-fg)}
.sd{background:var(--sd-bg);color:var(--sd-fg)}
.sf{background:var(--sf-bg);color:var(--sf-fg)}
/* ── BFR ── */
.bfrg{display:grid;grid-template-columns:repeat(4,1fr);gap:9px}
@media(max-width:680px){.bfrg{grid-template-columns:1fr 1fr}}
.bi{background:var(--row-bg);border-radius:8px;padding:11px 14px}
.bl{font-size:11px;color:var(--text-2);margin-bottom:3px}
.bv{font-size:18px;font-weight:700}
.empty{color:var(--text-2);font-style:italic;font-size:13px;padding:6px 0}
</style>
</head>
<body>

<div class="hdr">
  <div>
    <h1>IRP &mdash; Calibration Dashboard</h1>
    <p class="meta" id="meta-line"></p>
  </div>
  <button id="tbtn" onclick="toggleTheme()">🌙</button>
</div>

<div class="kpis" id="kpi-grid"></div>

<!-- Network (left) + Depot stock (right) -->
<div class="split">
  <div class="card">
    <div class="ct">Network &amp; active arcs (i &#8594; j) &#8212; transported load</div>
    <div class="tabs" id="g-obj"></div>
    <div class="tabs" id="g-per"></div>
    <div id="graph"></div>
  </div>
  <div class="card">
    <div class="ct">Depot stock &#8212; evolution by period</div>
    <div id="depot"></div>
  </div>
</div>

<!-- Deliveries -->
<div class="card">
  <div class="ct">Delivery summary</div>
  <div class="tabs" id="d-obj"></div>
  <table>
    <thead><tr>
      <th>Client</th><th>Period</th><th>Truck</th>
      <th>Received</th><th>Demand</th><th>Status</th>
    </tr></thead>
    <tbody id="dtbody"></tbody>
  </table>
</div>

<!-- BFR -->
<div class="card" id="bfr-card" style="display:none">
  <div class="ct">BFR detail &#8212; Working capital (f4)</div>
  <div class="bfrg" id="bfrd"></div>
</div>

<script>
var DATA  = /*DATA_PLACEHOLDER*/null;
var OBJS  = DATA.objs, BOUNDS = DATA.bounds, META = DATA.meta;
var OC    = ["#534ab7","#3b6d11","#185fa5","#854f0b"];
var TC    = ["#2e9e5a","#d07c20","#3b7dd8","#a040c0"];

if(!OBJS||!OBJS.length){
  document.getElementById('meta-line').textContent=
    META.n_nodes+' nodes · '+META.n_periods+' periods · '+META.n_vehicles+' vehicles · '+META.n_obj+' objectives';
  document.getElementById('kpi-grid').innerHTML=
    '<div style="grid-column:1/-1;color:#a32d2d;font-weight:600;padding:.5rem 0">'+
    'No solution found — the model is infeasible or the solver failed.</div>';
  throw new Error('No solutions');
}

// ── Theme ────────────────────────────────────────────────────────────────────
function toggleTheme(){
  var d=document.documentElement;
  d.toggleAttribute('data-dark');
  document.getElementById('tbtn').textContent=d.hasAttribute('data-dark')?'☀️':'🌙';
  localStorage.setItem('irp-theme',d.hasAttribute('data-dark')?'dark':'light');
}
if(localStorage.getItem('irp-theme')==='dark'){
  document.documentElement.setAttribute('data-dark','');
  document.getElementById('tbtn').textContent='☀️';
}

// ── Meta ─────────────────────────────────────────────────────────────────────
var objCountText = (META.n_obj_attempted && META.n_obj_attempted!==META.n_obj)
  ? META.n_obj+'/'+META.n_obj_attempted+' objectives (partial — see warnings below)'
  : META.n_obj+' objectives';
document.getElementById('meta-line').textContent=
  META.n_nodes+' nodes · '+META.n_periods+' periods · '+META.n_vehicles+' vehicles · '+objCountText;

// ── KPI cards ─────────────────────────────────────────────────────────────────
OBJS.forEach(function(o,i){
  var bk=o.budget_key;
  var statusLower = (o.solve_status||'').toLowerCase();
  var isOptimal = statusLower.indexOf('optimal')!==-1;
  var statusBadge = isOptimal ? '' :
    '<div class="kb" style="color:#a32d2d;font-weight:600;margin-top:3px">'+
    '⚠ not proven optimal ('+(o.solve_status||'unknown status')+') — budget threshold may be looser than intended</div>';
  document.getElementById('kpi-grid').innerHTML+=
    '<div class="kpi"><div class="kl">'+o.label+'</div>'+
    '<div class="kv" style="color:'+OC[i]+'">'+o.value.toFixed(4)+'</div>'+
    '<div class="kb">Budget '+bk+' = '+BOUNDS[bk].toFixed(4)+'</div>'+statusBadge+'</div>';
});

// ── State ─────────────────────────────────────────────────────────────────────
var periods=[];
OBJS.forEach(function(o){Object.keys(o.routes).forEach(function(p){var n=Number(p);if(periods.indexOf(n)<0)periods.push(n);});});
periods.sort(function(a,b){return a-b;});
var curObj=0, curPer=periods[0];

function buildTabs(id,labels,ai,ec,fn){
  var el=document.getElementById(id); el.innerHTML='';
  labels.forEach(function(l,i){
    var d=document.createElement('div');
    d.className='tab '+ec+(i===ai?' active':'');
    d.textContent=l;
    d.onclick=(function(x){return function(){el.querySelectorAll('.tab').forEach(function(t,j){t.classList.toggle('active',j===x)});fn(x);}})(i);
    el.appendChild(d);
  });
}
function syncObj(idx){
  curObj=idx;
  ['g-obj','d-obj'].forEach(function(id){
    var el=document.getElementById(id);
    if(el)el.querySelectorAll('.tab').forEach(function(t,j){t.classList.toggle('active',j===idx)});
  });
  renderAll();
}

var OL=OBJS.map(function(o){return o.label;});
var PL=periods.map(function(p){
  var rd=OBJS[0].routes[p];
  var Rf=(rd)?rd.R_frigo:0, Rnf=(rd)?rd.R_nonfrigo:0;
  return 'Period '+p+' (❄️'+Rf+' 📦'+Rnf+')';
});
['g-obj','d-obj'].forEach(function(id){buildTabs(id,OL,0,'',function(i){syncObj(i);});});
buildTabs('g-per',PL,0,'period',function(i){curPer=periods[i];document.getElementById('g-per').querySelectorAll('.tab').forEach(function(t,j){t.classList.toggle('active',j===i)});renderGraph();renderDepot();});

// ── SVG ──────────────────────────────────────────────────────────────────────
var NP=META.node_positions;
var isLarge = META.n_nodes > 6;
var SW = isLarge ? 1120 : 480;
var SH = isLarge ? 680 : 290;

// Stacked layout for large instances
if (isLarge) {
  var splitEl = document.querySelector('.split');
  if (splitEl) splitEl.classList.add('stacked');
}

function renderGraph(){
  var pr=OBJS[curObj].routes[curPer];
  var nodes=Object.keys(NP).map(Number);
  var trucks=(pr&&pr.trucks)?pr.trucks:[];

  var isDark=document.documentElement.hasAttribute('data-dark');
  var nodeFill=isDark?'#1c1c20':'#ffffff';
  var nodeStroke=isDark?'#8494ab':'#555';
  var textFill=isDark?'#ededed':'#1a1a18';
  var bgFill=isDark?'#111113':'#f4f4f2';
  var depotFill=isDark?'#9f9cf5':'#534ab7';

  var seen={};
  var arcs='';

  var markerDefs=trucks.map(function(tr){
    var col=TC[(tr.k-1)%TC.length];
    return '<marker id="arr'+tr.k+'" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto">'+
      '<polygon points="0 0,7 3.5,0 7" fill="'+col+'" opacity="0.85"/></marker>';
  }).join('');

  trucks.forEach(function(tr){
    var col=TC[(tr.k-1)%TC.length];
    var path=tr.path, qty=tr.qty||{};
    for(var i=0;i<path.length-1;i++){
      var a=path[i],b=path[i+1];
      var key=a+'-'+b;
      if(seen[key]) continue;
      seen[key]=true;
      var x1=NP[a][0],y1=NP[a][1],x2=NP[b][0],y2=NP[b][1];
      var mx=(x1+x2)/2,my=(y1+y2)/2;
      var nx=-(y2-y1),ny=x2-x1;
      var len=Math.sqrt(nx*nx+ny*ny)||1;
      var cx=mx+nx/len*22,cy=my+ny/len*22;
      var delivered=qty[String(b)]||0;
      arcs+='<path d="M'+x1+','+y1+' Q'+cx+','+cy+' '+x2+','+y2+'"'+
        ' fill="none" stroke="'+col+'" stroke-width="2.2" stroke-linecap="round"'+
        ' marker-end="url(#arr'+tr.k+')" opacity="0.82"/>';
      if(delivered>0){
        arcs+='<text x="'+cx+'" y="'+(cy-5)+'" text-anchor="middle" font-size="9"'+
          ' fill="'+col+'" font-weight="600">'+delivered+'u</text>';
      }
    }
    // dashed return to depot
    if(path.length>0){
      var last=path[path.length-1];
      if(last!==0){
        var rx1=NP[last][0],ry1=NP[last][1],rx2=NP[0][0],ry2=NP[0][1];
        var rmx=(rx1+rx2)/2,rmy=(ry1+ry2)/2;
        var rnx=-(ry2-ry1),rny=rx2-rx1;
        var rlen=Math.sqrt(rnx*rnx+rny*rny)||1;
        var rcx=rmx+rnx/rlen*18,rcy=rmy+rny/rlen*18;
        arcs+='<path d="M'+rx1+','+ry1+' Q'+rcx+','+rcy+' '+rx2+','+ry2+'"'+
          ' fill="none" stroke="'+col+'" stroke-width="1.4" stroke-dasharray="5,3" opacity="0.5"/>';
      }
    }
  });

  var nodesSvg=nodes.map(function(n){
    var x=NP[n][0],y=NP[n][1];
    var isD=(n===0),r=isD?17:14;
    var fill=isD?depotFill:nodeFill;
    var stroke=isD?depotFill:nodeStroke;
    var tFill=isD?'#fff':textFill;
    return '<circle cx="'+x+'" cy="'+y+'" r="'+r+'" fill="'+fill+'" stroke="'+stroke+'" stroke-width="1.8"/>'+
      '<text x="'+x+'" y="'+(y+4)+'" text-anchor="middle" font-size="'+(isD?11:10)+'"'+
      ' font-weight="700" fill="'+tFill+'">'+(isD?'D':String(n))+'</text>';
  }).join('');

  var svgStr='<svg viewBox="0 0 '+SW+' '+SH+'" class="nsvg">'+
    '<defs>'+markerDefs+'</defs>'+
    '<rect width="'+SW+'" height="'+SH+'" fill="'+bgFill+'"/>'+
    arcs+nodesSvg+'</svg>';

  var lgd='<div style="display:flex;gap:5px;flex-wrap:wrap;margin-top:9px;align-items:center">';
  if(trucks.length>0){
    if(pr&&pr.tau_return!=null){
      lgd+='<span style="font-size:11px;color:var(--text-2);padding:3px 10px;'+
           'background:var(--row-bg);border:1px solid var(--border);border-radius:99px;white-space:nowrap">'+
           '&#128339; Return: '+pr.tau_return.toFixed(2)+'h</span>';
    }
    trucks.forEach(function(tr){
      var c=TC[(tr.k-1)%TC.length];
      lgd+='<span style="display:inline-flex;align-items:center;gap:4px;padding:3px 9px 3px 6px;'+
           'background:var(--row-bg);border:1px solid var(--border);border-radius:99px;font-size:11px;white-space:nowrap">'+
           '<span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:'+c+';flex-shrink:0"></span>'+
           '<b style="color:var(--text)">k='+tr.k+'</b></span>';
    });
    lgd+='<span style="flex:1;min-width:8px"></span>';
    lgd+='<span style="display:inline-flex;align-items:center;gap:5px;font-size:10px;color:var(--text-2);white-space:nowrap">'+
         '<svg width="24" height="10" style="flex-shrink:0;overflow:visible">'+
         '<line x1="1" y1="5" x2="16" y2="5" stroke="#888" stroke-width="2.2" stroke-linecap="round"/>'+
         '<polygon points="14,2 23,5 14,8" fill="#888"/></svg>delivery</span>';
    lgd+='<span style="display:inline-flex;align-items:center;gap:5px;font-size:10px;color:var(--text-2);white-space:nowrap">'+
         '<svg width="24" height="10" style="flex-shrink:0;overflow:visible">'+
         '<line x1="1" y1="5" x2="16" y2="5" stroke="#aaa" stroke-width="1.5" stroke-dasharray="4,3"/>'+
         '<polygon points="14,2 23,5 14,8" fill="#aaa" opacity="0.5"/></svg>return to depot</span>';
  }
  lgd+='</div>';
  document.getElementById('graph').innerHTML=svgStr+lgd;
}

// ── Depot stock ───────────────────────────────────────────────────────────────
function renderDepot(){
  var stocks=OBJS[curObj].depot_stock;
  var prev_f=META.I_O_init_frigo, prev_nf=META.I_O_init_nonfrigo;

  function stockRow(label, color, prev, R, fin){
    var livr=(typeof prev==='number'&&typeof R==='number'&&typeof fin==='number')
      ? Math.round((prev+R-fin)*100)/100 : '?';
    return '<div style="margin-bottom:6px">'+
      '<div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--text-2);margin-bottom:3px">'+label+'</div>'+
      '<div class="seq">'+
        '<div class="sc si"><span class="sl">Init.</span><span class="sv" style="font-size:16px">'+prev+'</span></div>'+
        '<div class="sop">+</div>'+
        '<div class="sc sr"><span class="sl">R&#233;appro</span><span class="sv" style="font-size:16px">'+R+'</span></div>'+
        '<div class="sop">&#8722;</div>'+
        '<div class="sc sd"><span class="sl">Livr&#233;</span><span class="sv" style="font-size:16px">'+livr+'</span></div>'+
        '<div class="sop">=</div>'+
        '<div class="sc sf" style="background:'+color+'"><span class="sl">End stock</span><span class="sv" style="font-size:16px">'+fin+'</span></div>'+
      '</div></div>';
  }

  document.getElementById('depot').innerHTML=periods.map(function(t){
    var rdata=OBJS[curObj].routes[t];
    var R_f =(rdata)?rdata.R_frigo:0;
    var R_nf=(rdata)?rdata.R_nonfrigo:0;
    var fin_f =(stocks[t])?stocks[t].frigo:'?';
    var fin_nf=(stocks[t])?stocks[t].nonfrigo:'?';

    var html='<div class="sb"><div class="sp">P&#233;riode '+t+'</div>'+
      stockRow('&#10052;&#65039; Refrigerated',   'var(--si-bg)', prev_f,  R_f,  fin_f)+
      stockRow('&#128230; Non-refrigerated', '#f0f0ea',      prev_nf, R_nf, fin_nf)+
      '</div>';

    prev_f=fin_f; prev_nf=fin_nf; return html;
  }).join('');
}

// ── Delivery table ─────────────────────────────────────────────────────────────
function renderDel(){
  var dl=OBJS[curObj].deliveries;
  var el=document.getElementById('dtbody');
  if(!dl||!dl.length){el.innerHTML='<tr><td colspan="6" class="empty">No deliveries.</td></tr>';return;}
  el.innerHTML=dl.map(function(d){
    var ex=Math.abs(d.recu-d.dem)<0.01, an=d.recu>d.dem+0.01;
    var st=ex?'<span class="ok">&#10003; exact</span>':an?'<span class="ante">anticipated</span>':'<span class="part">partial</span>';
    return '<tr><td>Client '+d.l+'</td><td>t='+d.t+'</td><td>k='+d.k+'</td><td><b>'+d.recu+'</b></td><td>'+d.dem+'</td><td>'+st+'</td></tr>';
  }).join('');
}

// ── BFR ───────────────────────────────────────────────────────────────────────
function renderBFR(){
  var b=OBJS[curObj].bfr_sub, card=document.getElementById('bfr-card');
  if(!b){card.style.display='none';return;}
  card.style.display='block';
  var net=b.stock+b.receivables-b.payables;
  document.getElementById('bfrd').innerHTML=
    '<div class="bi"><div class="bl">Stock (value)</div><div class="bv" style="color:#534ab7">'+b.stock.toFixed(2)+'</div></div>'+
    '<div class="bi"><div class="bl">Customer receivables</div><div class="bv" style="color:#185fa5">'+b.receivables.toFixed(2)+'</div></div>'+
    '<div class="bi"><div class="bl">Supplier payables</div><div class="bv" style="color:#a32d2d">'+b.payables.toFixed(2)+'</div></div>'+
    '<div class="bi" style="background:var(--sf-bg)"><div class="bl">Net WCR</div><div class="bv" style="color:var(--sf-fg)">'+net.toFixed(2)+'</div></div>';
}

function renderAll(){renderGraph();renderDepot();renderDel();renderBFR();}
renderAll();
</script>
</body>
</html>"""


# ── Node layout — fan/triangle spread ─────────────────────────────────────────
def compute_node_positions(N, svg_w=480, svg_h=290):
    """
    Depot on the far left; clients spread in a triangular fan.
    No two clients share the same column, so inter-client arcs are visible.
    """
    clients = sorted(n for n in N if n != 0)
    nc      = len(clients)
    pos     = {}

    if nc > 6 and svg_w == 480 and svg_h == 290:
        svg_w, svg_h = 1120, 680

    # Depot: vertically centered, pushed left
    pos[0] = [int(svg_w * (0.11 if nc > 6 else 0.13)), svg_h // 2]

    if nc == 0:
        pass
    elif nc == 1:
        pos[clients[0]] = [int(svg_w * 0.82), svg_h // 2]
    elif nc == 2:
        pos[clients[0]] = [int(svg_w * 0.78), int(svg_h * 0.22)]
        pos[clients[1]] = [int(svg_w * 0.78), int(svg_h * 0.78)]
    elif nc == 3:
        # Triangle: top-centre-right, far-right-middle, bottom-centre-right
        pos[clients[0]] = [int(svg_w * 0.55), int(svg_h * 0.13)]   # top
        pos[clients[1]] = [int(svg_w * 0.84), int(svg_h * 0.52)]   # right
        pos[clients[2]] = [int(svg_w * 0.52), int(svg_h * 0.87)]   # bottom
    elif nc == 5:
        # Pentagon layout: depot left, 5 clients well-spread on the right half
        xs = [0.40, 0.84, 0.91, 0.76, 0.38]
        ys = [0.12, 0.17, 0.53, 0.87, 0.84]
        for i, c in enumerate(clients):
            pos[c] = [int(svg_w * xs[i]), int(svg_h * ys[i])]
    else:
        # Large instances: wide ellipse, leaving a clear left lane for depot arcs.
        cx  = svg_w * 0.58
        cy  = svg_h / 2
        rx  = svg_w * 0.34
        ry  = svg_h * 0.39
        for i, c in enumerate(clients):
            # Start from top, go clockwise, avoid the leftmost point (depot side)
            angle = -math.pi / 2 + 2 * math.pi * i / nc
            pos[c] = [int(cx + rx * math.cos(angle)),
                      int(cy + ry * math.sin(angle))]

    return {str(k): v for k, v in pos.items()}


# ── Public API ─────────────────────────────────────────────────────────────────
def render_report_html(data):
    html = _TEMPLATE.replace(
        "/*DATA_PLACEHOLDER*/null",
        json.dumps(data, ensure_ascii=False),
    )
    return html


def write_report(data, output_path):
    html = render_report_html(data)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return os.path.abspath(output_path)
