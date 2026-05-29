"""HTML report generator for IRP calibration results."""

import json
import math
import os
import subprocess
import sys
import webbrowser

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IRP &mdash; Tableau de Bord</title>
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
.nsvg{width:100%;display:block}
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
    <h1>IRP &mdash; Tableau de Bord de Calibration</h1>
    <p class="meta" id="meta-line"></p>
  </div>
  <button id="tbtn" onclick="toggleTheme()">🌙</button>
</div>

<div class="kpis" id="kpi-grid"></div>

<!-- Network (left) + Stock dépôt (right) -->
<div class="split">
  <div class="card">
    <div class="ct">R&#233;seau &amp; arcs actifs (i &#8594; j) &#8212; charge transport&#233;e</div>
    <div class="tabs" id="g-obj"></div>
    <div class="tabs" id="g-per"></div>
    <div id="graph"></div>
  </div>
  <div class="card">
    <div class="ct">Stock d&#233;p&#244;t &#8212; &#233;volution par p&#233;riode</div>
    <div id="depot"></div>
  </div>
</div>

<!-- Deliveries -->
<div class="card">
  <div class="ct">R&#233;sum&#233; des livraisons</div>
  <div class="tabs" id="d-obj"></div>
  <table>
    <thead><tr>
      <th>Client</th><th>P&#233;riode</th><th>Camion</th>
      <th>Re&#231;u</th><th>Demande</th><th>Statut</th>
    </tr></thead>
    <tbody id="dtbody"></tbody>
  </table>
</div>

<!-- BFR -->
<div class="card" id="bfr-card" style="display:none">
  <div class="ct">D&#233;tail BFR &#8212; Capital de travail (f4)</div>
  <div class="bfrg" id="bfrd"></div>
</div>

<script>
var DATA  = /*DATA_PLACEHOLDER*/null;
var OBJS  = DATA.objs, BOUNDS = DATA.bounds, META = DATA.meta;
var OC    = ["#534ab7","#3b6d11","#185fa5","#854f0b"];
var TC    = ["#2e9e5a","#d07c20","#3b7dd8","#a040c0"];
var BK    = ["C_max","E_max","T_max","B"];

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
document.getElementById('meta-line').textContent=
  META.n_nodes+' nœuds · '+META.n_periods+' périodes · '+META.n_vehicles+' véhicules · '+META.n_obj+' objectifs';

// ── KPI cards ─────────────────────────────────────────────────────────────────
OBJS.forEach(function(o,i){
  var bk=BK[i];
  document.getElementById('kpi-grid').innerHTML+=
    '<div class="kpi"><div class="kl">'+o.label+'</div>'+
    '<div class="kv" style="color:'+OC[i]+'">'+o.value.toFixed(4)+'</div>'+
    '<div class="kb">Budget '+bk+' = '+BOUNDS[bk].toFixed(4)+'</div></div>';
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
var PL=periods.map(function(p){var R=(OBJS[0].routes[p])?OBJS[0].routes[p].R:0;return 'Période '+p+' (R='+R+')';});
['g-obj','d-obj'].forEach(function(id){buildTabs(id,OL,0,'',function(i){syncObj(i);});});
buildTabs('g-per',PL,0,'period',function(i){curPer=periods[i];document.getElementById('g-per').querySelectorAll('.tab').forEach(function(t,j){t.classList.toggle('active',j===i)});renderGraph();renderDepot();});

// ── SVG ──────────────────────────────────────────────────────────────────────
var NP=META.node_positions;
var SW=400, SH=330;

function getR(n){return n===0?24:20;}

// Compute outward direction from depot for node label placement
function labelPos(n){
  var cx=NP[n][0], cy=NP[n][1];
  var r=getR(n);
  var d0=NP[0];
  var dx=cx-d0[0], dy=cy-d0[1];
  var len=Math.sqrt(dx*dx+dy*dy);
  if(len<1)return [cx,cy-r-12];
  return [cx+dx/len*(r+13), cy+dy/len*(r+13)];
}

// Arc loads: load on arc path[i]→path[i+1] = sum qty from i+1..end
function arcLoads(path,qty){
  var out=[];
  for(var i=0;i<path.length-1;i++){
    var s=0;for(var j=i+1;j<path.length;j++)s+=(qty[String(path[j])]||0);
    out.push(Math.round(s*10)/10);
  }
  return out;
}

// Draw one bezier arc
function drawArc(nA,nB,off,curve,color,mid,dashed,load){
  var p1=NP[nA],p2=NP[nB];
  var dx=p2[0]-p1[0],dy=p2[1]-p1[1];
  var len=Math.sqrt(dx*dx+dy*dy); if(len<1)return '';
  var ux=dx/len,uy=dy/len,px=-uy,py=ux;
  var rA=getR(nA),rB=getR(nB);
  var x0=p1[0]+ux*(rA+3)+px*off, y0=p1[1]+uy*(rA+3)+py*off;
  var x2=p2[0]-ux*(rB+14)+px*off, y2=p2[1]-uy*(rB+14)+py*off;
  var cx=(x0+x2)/2+px*curve, cy=(y0+y2)/2+py*curve;
  var sw=dashed?1.5:Math.max(2,Math.min(5,1.5+load/22));
  var d='M'+x0.toFixed(1)+','+y0.toFixed(1)+' Q'+cx.toFixed(1)+','+cy.toFixed(1)+' '+x2.toFixed(1)+','+y2.toFixed(1);
  var r='<path d="'+d+'" fill="none" stroke="'+color+'" stroke-width="'+sw.toFixed(1)+'"'+
        (dashed?' stroke-dasharray="6,4" opacity="0.45"':'')+
        ' marker-end="url(#'+mid+')" stroke-linecap="round"/>';
  // Load label at bezier midpoint
  if(!dashed&&load>0){
    var mx=x0/4+cx/2+x2/4, my=y0/4+cy/2+y2/4;
    var txt=load+'u', lw=txt.length*6+12;
    r+='<rect x="'+(mx-lw/2).toFixed(1)+'" y="'+(my-8).toFixed(1)+
       '" width="'+lw+'" height="15" rx="5" fill="var(--surface)" opacity="0.82"/>'+
       '<text x="'+mx.toFixed(1)+'" y="'+(my+3).toFixed(1)+
       '" text-anchor="middle" font-size="9" font-weight="700" fill="'+color+'">'+txt+'</text>';
  }
  return r;
}

function renderGraph(){
  var pr=OBJS[curObj].routes[curPer];
  var nodes=Object.keys(NP).map(Number);
  var nT=(pr&&pr.trucks)?pr.trucks.length:1;

  var defs=TC.map(function(c,i){
    return '<marker id="ah'+i+'" markerWidth="11" markerHeight="8" refX="10" refY="4" orient="auto" markerUnits="userSpaceOnUse">'+
           '<path d="M0,0 L11,4 L0,8 Z" fill="'+c+'"/></marker>'+
           '<marker id="ahd'+i+'" markerWidth="11" markerHeight="8" refX="10" refY="4" orient="auto" markerUnits="userSpaceOnUse">'+
           '<path d="M0,0 L11,4 L0,8 Z" fill="'+c+'" opacity="0.38"/></marker>';
  }).join('');

  var svg='<svg viewBox="0 0 '+SW+' '+SH+'" class="nsvg"><defs>'+defs+'</defs>';

  // Background mesh
  nodes.forEach(function(i){nodes.forEach(function(j){
    if(i>=j)return;
    var a=NP[i],b=NP[j];
    svg+='<line x1="'+a[0]+'" y1="'+a[1]+'" x2="'+b[0]+'" y2="'+b[1]+'" stroke="var(--mesh)" stroke-width="1"/>';
  });});

  // Active arcs
  if(pr&&pr.trucks){
    pr.trucks.forEach(function(tr){
      var ki=(tr.k-1)%TC.length, color=TC[ki];
      var off=(ki-(nT-1)/2)*14;
      var curve=30+off*0.4;
      var loads=arcLoads(tr.path,tr.qty);
      for(var s=0;s<tr.path.length-1;s++)
        svg+=drawArc(tr.path[s],tr.path[s+1],off,curve,color,'ah'+ki,false,loads[s]);
      var last=tr.path[tr.path.length-1];
      if(last!==0) svg+=drawArc(last,0,off,-curve,color,'ahd'+ki,true,0);
    });
  }

  // Node circles
  nodes.forEach(function(n){
    var cx=NP[n][0],cy=NP[n][1],r=getR(n);
    var isD=(n===0);
    var fill=isD?'#534ab7':'#185fa5';
    var lbl=isD?'D':String(n);

    // Quantity delivered
    var qt='';
    if(!isD&&pr&&pr.trucks){
      var q=pr.trucks.reduce(function(s,tr){return s+(tr.qty[String(n)]||0);},0);
      if(q>0){
        var lp=labelPos(n);
        qt='<text x="'+lp[0].toFixed(1)+'" y="'+(lp[1]+4).toFixed(1)+
           '" text-anchor="middle" font-size="10" font-weight="700" fill="'+fill+'">'+q+'u</text>';
      }
    }
    // Node label (client name, on the other side from qty)
    var nl='';
    if(!isD){
      var lp2=labelPos(n);
      // Use same outward direction but smaller offset for a "C1" tag
      nl='<text x="'+NP[n][0]+'" y="'+NP[n][1]+
         '" text-anchor="middle" dominant-baseline="middle" fill="#fff"'+
         ' font-size="12" font-weight="700">'+n+'</text>';
    }

    svg+='<circle cx="'+cx+'" cy="'+cy+'" r="'+r+
         '" fill="'+fill+'" stroke="var(--node-surf)" stroke-width="3"/>'+
         (isD?'<text x="'+cx+'" y="'+cy+'" text-anchor="middle" dominant-baseline="middle" fill="#fff" font-size="13" font-weight="700">D</text>':nl)+
         qt;
  });

  // Legend
  if(pr&&pr.trucks&&pr.trucks.length>0){
    var lx=8;
    pr.trucks.forEach(function(tr){
      var c=TC[(tr.k-1)%TC.length];
      svg+='<rect x="'+lx+'" y="'+(SH-22)+'" width="12" height="9" rx="2" fill="'+c+'"/>'+
           '<text x="'+(lx+16)+'" y="'+(SH-14)+'" fill="var(--text-2)" font-size="10">k='+tr.k+' – retour '+tr.ret.toFixed(2)+'h</text>';
      lx+=130;
    });
    svg+='<line x1="'+lx+'" y1="'+(SH-18)+'" x2="'+(lx+16)+'" y2="'+(SH-18)+'" stroke="#999" stroke-width="2.2"/>'+
         '<text x="'+(lx+20)+'" y="'+(SH-14)+'" fill="var(--text-2)" font-size="10">livraison</text>'+
         '<line x1="'+lx+'" y1="'+(SH-9)+'" x2="'+(lx+16)+'" y2="'+(SH-9)+'" stroke="#999" stroke-width="1.4" stroke-dasharray="4,3"/>'+
         '<text x="'+(lx+20)+'" y="'+(SH-5)+'" fill="var(--text-2)" font-size="10">retour dépôt</text>';
  }
  svg+='</svg>';
  document.getElementById('graph').innerHTML=svg;
}

// ── Depot stock ───────────────────────────────────────────────────────────────
function renderDepot(){
  var stocks=OBJS[curObj].depot_stock;
  var delivs=OBJS[curObj].deliveries;
  var prev=META.I_O_init;
  document.getElementById('depot').innerHTML=periods.map(function(t){
    var R=(OBJS[curObj].routes[t])?OBJS[curObj].routes[t].R:0;
    var fin=(stocks[t]!==undefined)?stocks[t]:'?';
    var tot=delivs.filter(function(d){return d.t===t;}).reduce(function(s,d){return s+d.recu;},0);
    tot=Math.round(tot*100)/100;
    var html='<div class="sb"><div class="sp">Période '+t+'</div>'+
      '<div class="seq">'+
        '<div class="sc si"><span class="sl">Stock initial</span><span class="sv">'+prev+'</span></div>'+
        '<div class="sop">+</div>'+
        '<div class="sc sr"><span class="sl">R&#233;appro (R)</span><span class="sv">'+R+'</span></div>'+
        '<div class="sop">&#8722;</div>'+
        '<div class="sc sd"><span class="sl">Livraisons</span><span class="sv">'+tot+'</span></div>'+
        '<div class="sop">=</div>'+
        '<div class="sc sf"><span class="sl">Stock final</span><span class="sv">'+fin+'</span></div>'+
      '</div></div>';
    prev=fin; return html;
  }).join('');
}

// ── Delivery table ─────────────────────────────────────────────────────────────
function renderDel(){
  var dl=OBJS[curObj].deliveries;
  var el=document.getElementById('dtbody');
  if(!dl||!dl.length){el.innerHTML='<tr><td colspan="6" class="empty">Aucune livraison.</td></tr>';return;}
  el.innerHTML=dl.map(function(d){
    var ex=Math.abs(d.recu-d.dem)<0.01, an=d.recu>d.dem+0.01;
    var st=ex?'<span class="ok">&#10003; exact</span>':an?'<span class="ante">anticipé</span>':'<span class="part">partiel</span>';
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
    '<div class="bi"><div class="bl">Stock (valeur)</div><div class="bv" style="color:#534ab7">'+b.stock.toFixed(2)+'</div></div>'+
    '<div class="bi"><div class="bl">Créances clients</div><div class="bv" style="color:#185fa5">'+b.receivables.toFixed(2)+'</div></div>'+
    '<div class="bi"><div class="bl">Dettes fournisseurs</div><div class="bv" style="color:#a32d2d">'+b.payables.toFixed(2)+'</div></div>'+
    '<div class="bi" style="background:var(--sf-bg)"><div class="bl">BFR Net</div><div class="bv" style="color:var(--sf-fg)">'+net.toFixed(2)+'</div></div>';
}

function renderAll(){renderGraph();renderDepot();renderDel();renderBFR();}
renderAll();
</script>
</body>
</html>"""


# ── Node layout — fan/triangle spread ─────────────────────────────────────────
def compute_node_positions(N, svg_w=400, svg_h=330):
    """
    Depot on the far left; clients spread in a triangular fan.
    No two clients share the same column, so inter-client arcs are visible.
    """
    clients = sorted(n for n in N if n != 0)
    nc      = len(clients)
    pos     = {}

    # Depot: vertically centered, pushed left
    pos[0] = [int(svg_w * 0.14), svg_h // 2]

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
    else:
        # General: semicircle on the right half
        cx  = svg_w * 0.62
        cy  = svg_h / 2
        rx  = svg_w * 0.30
        ry  = svg_h * 0.40
        for i, c in enumerate(clients):
            angle = math.pi * (0.85 - 1.7 * i / max(nc - 1, 1))
            pos[c] = [int(cx + rx * math.cos(angle)),
                      int(cy - ry * math.sin(angle))]

    return {str(k): v for k, v in pos.items()}


# ── Chrome-first opener ────────────────────────────────────────────────────────
def _open_chrome(url):
    if sys.platform == "win32":
        candidates = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expanduser(
                r"~\AppData\Local\Google\Chrome\Application\chrome.exe"),
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


# ── Public API ─────────────────────────────────────────────────────────────────
def generate_and_open(data, output_path):
    html = _TEMPLATE.replace(
        "/*DATA_PLACEHOLDER*/null",
        json.dumps(data, ensure_ascii=False),
    )
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    abs_path = os.path.abspath(output_path).replace("\\", "/")
    _open_chrome(f"file:///{abs_path}")
