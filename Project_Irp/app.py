"""Local web menu for the IRP project modules."""

import csv
import json
import os
import socket
import statistics as _statistics
import threading
import traceback
import uuid as _uuid
import webbrowser
import time
from html import escape

from flask import Flask, jsonify, render_template_string, request, send_file, url_for

from Solvers.ObjectiveCalibration.main import DEFAULT_REPORT_PATH, run_objective_calibration
from Solvers.FunctionMerge.main import DEFAULT_REPORT_PATH as FM_REPORT_PATH, run_function_merge
from Solvers.NSGA3.main    import DEFAULT_REPORT_PATH as NSGA3_REPORT_PATH, run_nsga3_report, render_from_instance as nsga3_render_from_instance
from Solvers.NSGA3.report  import render_html as nsga3_render_html
from Solvers.QINSGA3.main  import DEFAULT_REPORT_PATH as QINSGA3_REPORT_PATH, run_qinsga3_report, render_from_instance as qinsga3_render_from_instance
from Solvers.MOEAD.main    import DEFAULT_REPORT_PATH as MOEAD_REPORT_PATH, run_moead_report, render_from_instance as moead_render_from_instance


# ── Async job tracker ────────────────────────────────────────────────────────
# Each job: {status: "running"|"done"|"error", redirect: url|None, error: str|None, algo: str}
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

# ── Per-(algorithm, instance) run locks ─────────────────────────────────────
# Prevents two concurrent runs of the SAME algorithm on the SAME instance
# (double-click, two browser tabs, ...) from racing to write the same
# instance-specific output/cache files.
_run_locks: dict[tuple[str, str], threading.Lock] = {}
_run_locks_guard = threading.Lock()


def _get_run_lock(algo: str, inst_key: str) -> threading.Lock:
    key = (algo, inst_key)
    with _run_locks_guard:
        lock = _run_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _run_locks[key] = lock
        return lock


def _new_job(algo: str) -> str:
    job_id = _uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "algo": algo, "redirect": None, "error": None}
    return job_id


def _job_done(job_id: str, redirect_url: str) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["redirect"] = redirect_url


def _job_error(job_id: str, error: str) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = error


_JOB_PAGE = """<!DOCTYPE html>
<html lang="fr" data-theme="light">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>En cours — {algo}</title>
<style>
:root{{--bg:#f2f5f9;--surface:#fff;--border:#dce3ec;--text:#0f1923;--text-2:#4e6070;
  --accent:#0f6a87;--accent-dim:#e6f3f8;--green:#0d7a55;--green-bg:#e6f5ee;}}
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{min-height:100vh;display:flex;flex-direction:column;align-items:center;
  justify-content:center;background:var(--bg);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;
  color:var(--text);gap:24px;padding:24px;}}
.card{{background:var(--surface);border:1px solid var(--border);border-radius:16px;
  padding:40px 48px;text-align:center;max-width:480px;width:100%;
  box-shadow:0 4px 24px rgba(0,0,0,.07);}}
.spinner{{width:52px;height:52px;border:4px solid var(--border);
  border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite;margin:0 auto 24px;}}
@keyframes spin{{to{{transform:rotate(360deg)}}}}
h1{{font-size:20px;font-weight:700;margin-bottom:8px;}}
p{{font-size:13.5px;color:var(--text-2);line-height:1.6;}}
.algo{{display:inline-block;margin-top:16px;padding:4px 14px;border-radius:99px;
  background:var(--accent-dim);color:var(--accent);font-size:12px;font-weight:700;letter-spacing:.04em;}}
.back{{margin-top:20px;font-size:12.5px;color:var(--text-2);}}
.back a{{color:var(--accent);font-weight:600;text-decoration:none;}}
</style>
</head>
<body>
<div class="card">
  <div class="spinner"></div>
  <h1>Solveur en cours…</h1>
  <p>L'optimisation s'exécute en arrière-plan.<br>
     Cette page se rafraîchit automatiquement — vous pouvez ouvrir un autre module dans un nouvel onglet.</p>
  <span class="algo">{algo}</span>
  <p class="back"><a href="/">← Retour au menu</a> &nbsp;|&nbsp; les résultats s'ouvrent ici une fois prêts</p>
</div>
<script>
(function poll(){{
  fetch('/job/api?job={job_id}')
    .then(r=>r.json())
    .then(d=>{{
      if(d.status==='done')   {{ window.location.href = d.redirect; return; }}
      if(d.status==='error')  {{ window.location.href = '/job/error?job={job_id}'; return; }}
      setTimeout(poll, 2000);
    }})
    .catch(()=>setTimeout(poll, 3000));
}})();
</script>
</body>
</html>"""


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# One suite = one Validation/Benchmarking/<suite>/results/ folder. Both follow
# the same Cui et al. (2025) protocol (same p/H/N/Tmax, Table 2 gives an
# identical row for "DTLZ 1-7" and "MaF 1-7"), so they share the M values and
# the whole card/chart/table rendering — only the problem list and results
# dir differ.
_BENCHMARK_SUITES = {
    "dtlz": {
        "results_dir": os.path.join(BASE_DIR, "Validation", "Benchmarking", "dtlz", "results"),
        "problems": ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7"],
    },
    "maf": {
        "results_dir": os.path.join(BASE_DIR, "Validation", "Benchmarking", "maf", "results"),
        "problems": ["MaF1", "MaF2", "MaF3", "MaF4", "MaF5", "MaF6", "MaF7"],
    },
}
_BENCHMARK_M_VALUES = [3, 4]  # Cui et al. (2025), Table 2, only studies M=3 and M=4


_BENCHMARK_ALGOS = ["nsga3", "qinsga3", "moead"]


def _read_igd_runs(results_dir: str, algo: str, problem: str, n_obj: int):
    path = os.path.join(results_dir, algo, f"igd_{problem}_M{n_obj}.csv")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "run":  int(row["run"]),
                "seed": int(row["seed"]),
                "igd":  float(row["igd"]),
            })
    return rows


def _build_benchmark_data():
    data = {}
    for suite, cfg in _BENCHMARK_SUITES.items():
        data[suite] = {}
        for p in cfg["problems"]:
            data[suite][p] = {}
            for m in _BENCHMARK_M_VALUES:
                m_data = {}
                for algo in _BENCHMARK_ALGOS:
                    details = _read_igd_runs(cfg["results_dir"], algo, p, m)
                    if not details:
                        continue
                    igds = [r["igd"] for r in details]
                    m_data[algo] = {
                        "best":   min(igds),
                        "median": float(_statistics.median(igds)),
                        "worst":  max(igds),
                        "mean":   float(_statistics.mean(igds)),
                        "std":    float(_statistics.pstdev(igds)),
                        "runs":   igds,
                        "run_details": details,
                        "n_runs": len(igds),
                    }
                if m_data:
                    data[suite][p][f"M{m}"] = m_data
    return data

def _discover_instances():
    data_dir = os.path.join(BASE_DIR, "data")
    found = {}
    if os.path.isdir(data_dir):
        for fname in os.listdir(data_dir):
            if fname.startswith("instance_") and fname.endswith("_clients.json"):
                key = fname[len("instance_"):-len("_clients.json")]
                if key.isdigit():
                    found[key] = os.path.join(data_dir, fname)
    return dict(sorted(found.items(), key=lambda x: int(x[0])))

INSTANCES = _discover_instances()
DEFAULT_INSTANCE = (
    "15" if "15" in INSTANCES
    else (sorted(INSTANCES.keys(), key=int)[0] if INSTANCES else None)
)


def _resolve_instance():
    key = request.args.get("instance", DEFAULT_INSTANCE or "")
    if key in INSTANCES:
        return INSTANCES[key], key
    if INSTANCES:
        fallback = sorted(INSTANCES.keys(), key=int)[0]
        return INSTANCES[fallback], fallback
    raise ValueError("No instance JSON files found in the data/ directory.")


def _instance_report_path(base_path: str, inst_key: str) -> str:
    """base_path (e.g. DEFAULT_REPORT_PATH) with the instance key folded into
    the filename, so a cached report for one instance is never silently
    served for another (inst_key comes only from INSTANCES' own dict keys
    via _resolve_instance(), never raw request text, so it's filename-safe)."""
    root, ext = os.path.splitext(base_path)
    return f"{root}_instance{inst_key}{ext}"

app = Flask(__name__)


BENCHMARK_TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DTLZ / MaF Benchmarking &mdash; NSGA-III</title>
<style>
:root {
  --bg:#f2f5f9;--surface:#fff;--surface-2:#f8fafc;--border:#dce3ec;
  --border-focus:#93b4ca;--text:#0f1923;--text-2:#4e6070;--text-3:#8fa0b0;
  --accent:#0f6a87;--accent-dim:#e6f3f8;--accent-fg:#fff;
  --green:#0d7a55;--green-bg:#e6f5ee;--red:#b91c1c;--red-bg:#fef2f2;
  --shadow-sm:0 1px 2px rgba(0,0,0,.06),0 3px 8px rgba(0,0,0,.04);
  --shadow-md:0 2px 6px rgba(0,0,0,.07),0 8px 24px rgba(0,0,0,.07);
  --radius:14px;--radius-sm:8px;
}
[data-theme="dark"] {
  --bg:#0b0f16;--surface:#131922;--surface-2:#1a2130;--border:#1e2a3a;
  --border-focus:#2d4a62;--text:#e0e8f2;--text-2:#8097b0;--text-3:#3d5068;
  --accent:#38b2d4;--accent-dim:#0d2535;--accent-fg:#060f18;
  --green:#34c98a;--green-bg:#0a2419;--red:#f87171;--red-bg:#2a0a0a;
  --shadow-sm:0 1px 2px rgba(0,0,0,.3),0 3px 8px rgba(0,0,0,.25);
}
*,*::before,*::after { box-sizing:border-box;margin:0;padding:0; }
body { min-height:100vh;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;background:var(--bg);color:var(--text);-webkit-font-smoothing:antialiased; }
.page { max-width:1020px;margin:0 auto;padding:36px 24px 60px; }
.header { display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:28px;padding-bottom:20px;border-bottom:1px solid var(--border); }
.brand { display:flex;align-items:center;gap:14px; }
.brand-icon { width:44px;height:44px;border-radius:12px;background:var(--accent);display:flex;align-items:center;justify-content:center;flex-shrink:0;box-shadow:0 2px 8px rgba(15,106,135,.35); }
.brand-icon svg { width:22px;height:22px;fill:#fff; }
.brand-name { font-size:18px;font-weight:700;color:var(--text); }
.brand-sub { font-size:12px;color:var(--text-2);margin-top:1px; }
.hdr-actions { display:flex;align-items:center;gap:8px; }
.btn-back { display:inline-flex;align-items:center;gap:6px;height:32px;padding:0 14px;border-radius:var(--radius-sm);border:1px solid var(--border);background:var(--surface);color:var(--text-2);font-size:12.5px;font-weight:600;cursor:pointer;text-decoration:none;transition:all .15s; }
.btn-back:hover { border-color:var(--border-focus);color:var(--text); }
.btn-theme { display:inline-flex;align-items:center;gap:5px;height:32px;padding:0 13px;border:1px solid var(--border);border-radius:var(--radius-sm);background:var(--surface);color:var(--text-2);font-size:12.5px;font-weight:500;cursor:pointer;transition:all .15s;outline:none; }
.btn-theme:hover { border-color:var(--border-focus);color:var(--text); }
.btn-theme svg { width:14px;height:14px;flex-shrink:0; }
.suite-bar { display:flex;gap:8px;margin-bottom:16px; }
.suite-tab { padding:8px 20px;border:1.5px solid var(--border);border-radius:99px;background:var(--surface);color:var(--text-2);font-size:13px;font-weight:700;cursor:pointer;transition:all .15s; }
.suite-tab:hover { border-color:var(--border-focus);color:var(--text); }
.suite-tab.active { background:var(--accent);border-color:var(--accent);color:var(--accent-fg); }
.m-bar { display:flex;gap:8px;margin-bottom:20px; }
.m-tab { padding:7px 20px;border:1.5px solid var(--border);border-radius:99px;background:var(--surface);color:var(--text-2);font-size:12.5px;font-weight:700;cursor:pointer;transition:all .15s; }
.m-tab:hover:not(:disabled) { border-color:var(--border-focus);color:var(--text); }
.m-tab.active { background:var(--text);border-color:var(--text);color:var(--bg); }
.m-tab:disabled { opacity:.35;cursor:not-allowed; }
.tab-bar { display:flex;flex-wrap:wrap;gap:4px;background:var(--surface-2);border:1px solid var(--border);border-radius:10px;padding:4px;margin-bottom:20px; }
.tab { flex:1;min-width:64px;padding:9px 12px;border:none;border-radius:7px;background:transparent;color:var(--text-2);font-size:13px;font-weight:700;cursor:pointer;transition:all .15s; }
.tab:hover { background:var(--surface);color:var(--text); }
.tab.active { background:var(--surface);color:var(--accent);box-shadow:var(--shadow-sm); }
.prob-desc { background:var(--accent-dim);border:1px solid color-mix(in srgb,var(--accent) 20%,transparent);border-radius:var(--radius-sm);padding:12px 16px;font-size:13px;color:var(--text-2);line-height:1.6;margin-bottom:20px; }
.prob-desc strong { color:var(--text); }
.grid2 { display:flex;flex-direction:column;gap:20px; }
.algo-compare { display:flex;flex-wrap:wrap;gap:16px; }
.algo-compare .rcard { flex:1 1 380px;min-width:320px; }
.rcard { background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow-sm);overflow:hidden; }
.rcard-hdr { padding:14px 18px 12px;border-bottom:1px solid var(--border); }
.rcard-hdr h3 { font-size:13.5px;font-weight:700;color:var(--text); }
.rcard-sub { font-size:11px;color:var(--text-3);margin-top:3px;line-height:1.7; }
.rcard-sub em { font-style:normal;color:var(--accent);font-weight:600; }
.param-lbl { color:var(--text-3);font-weight:400;font-style:italic; }
.param-note { color:var(--text-3);opacity:.8; }
h3 .param-lbl { font-size:10px;text-transform:none;letter-spacing:0; }
.rcard-body { padding:16px 18px; }
.stat-row { display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:14px; }
.stat-box { border-radius:8px;padding:10px 12px;text-align:center; }
.stat-box.best { background:var(--green-bg); }
.stat-box.med  { background:var(--accent-dim); }
.stat-box.mean { background:var(--surface-2); }
.stat-note { font-size:10.5px;color:var(--text-3);margin:-6px 0 14px; }
.stat-box.worst { background:var(--red-bg); }
.stat-lbl { font-size:9px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--text-3);margin-bottom:4px; }
.stat-val { font-size:13px;font-weight:700;font-family:monospace; }
.stat-val-dec { display:block;font-size:9.5px;font-weight:600;font-family:monospace;color:var(--text-3);margin-top:2px; }
.stat-box.best .stat-val { color:var(--green); }
.stat-box.worst .stat-val { color:var(--red); }
.chart-lbl { font-size:10px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:var(--text-3);margin-bottom:6px; }
.chart-legend { display:flex;gap:12px;font-size:9.5px;color:var(--text-3);margin-top:6px;flex-wrap:wrap; }
.chart-legend span { display:inline-flex;align-items:center;gap:4px; }
.chart-legend i { width:8px;height:8px;border-radius:50%;flex-shrink:0; }
.dot-svg { width:100%;overflow:visible; }
.table-wrap { overflow-x:auto;margin-top:8px;border:1px solid var(--border);border-radius:var(--radius-sm); }
.runs-table { width:100%;border-collapse:collapse;font-size:12.5px; }
.runs-table th { text-align:left;padding:9px 12px;font-size:9.5px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--text-3);background:var(--surface-2);border-bottom:1px solid var(--border);white-space:nowrap; }
.runs-table td { padding:9px 12px;border-bottom:1px solid var(--border);font-family:monospace;white-space:nowrap; }
.runs-table tbody tr:last-child td { border-bottom:none; }
.runs-table tbody tr:hover td { background:var(--surface-2); }
.run-badge { display:inline-flex;align-items:center;gap:7px;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;font-size:12px;color:var(--text-2); }
.run-dot { width:8px;height:8px;border-radius:50%;flex-shrink:0; }
.nodata { padding:28px;text-align:center;color:var(--text-3);font-size:13px; }

/* ── Article-style parameter table (Cui et al. 2025, Table 2) ── */
.lit-table-block { margin-bottom:28px; }
.lit-table-caption { font-size:12.5px;font-weight:700;color:var(--text);margin-bottom:2px; }
.lit-table-cite { font-size:11px;color:var(--text-3);margin-bottom:10px; }
.lit-table-scroll { overflow-x:auto; }
.lit-table { width:100%;min-width:520px;border-collapse:collapse;font-size:13px; }
.lit-table thead tr { border-top:2px solid var(--text);border-bottom:1.5px solid var(--text); }
.lit-table th { padding:8px 14px;text-align:center;font-weight:700;color:var(--text);white-space:nowrap; }
.lit-table th:first-child { text-align:left; }
.lit-table td { padding:7px 14px;text-align:center;color:var(--text);font-variant-numeric:tabular-nums;font-family:monospace; }
.lit-table td:first-child { text-align:left;font-weight:700;font-family:inherit; }
.lit-table tbody tr.grp-end td { border-bottom:1px solid var(--border); }
.lit-table tbody tr:last-child td { border-bottom:2px solid var(--text); }
.lit-table-note { font-size:11px;color:var(--text-3);margin-top:10px;line-height:1.65; }
.lit-table-note b { color:var(--text-2); }

/* ── Mean-rank bar charts (Cui et al. 2025, Fig. 1/2 style) ── */
.rank-block { margin-bottom:28px; }
.rank-charts { display:flex;flex-wrap:wrap;gap:16px;margin-top:12px; }
.rank-card { flex:1 1 260px;min-width:240px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow-sm);padding:14px 18px 10px; }
.rank-card-title { font-size:11.5px;font-weight:700;color:var(--text);margin-bottom:2px; }
.rank-card-sub { font-size:10.5px;color:var(--text-3);margin-bottom:8px; }
.rank-svg { width:100%;overflow:visible; }
.rank-bar-lbl { font-size:9.5px;font-weight:700;fill:var(--text);text-anchor:middle; }
.rank-axis-lbl { font-size:9px;fill:var(--text-2);text-anchor:middle;font-weight:600; }
.rank-grid-lbl { font-size:8px;fill:var(--text-3);text-anchor:end; }
.rank-y-title { font-size:9px;font-weight:700;fill:var(--text-2);text-anchor:middle; }

@media (max-width:680px) { .grid2 { grid-template-columns:1fr; } .page { padding:16px 12px 40px; } .header { flex-direction:column;align-items:flex-start; } }
</style>
</head>
<body>
<main class="page">

  <header class="header">
    <div class="brand">
      <div class="brand-icon">
        <svg viewBox="0 0 24 24"><path d="M3 3h7v7H3V3zm11 0h7v7h-7V3zm0 11h7v7h-7v-7zM3 14h7v7H3v-7z"/></svg>
      </div>
      <div>
        <div class="brand-name">Many-Objective Benchmarking</div>
        <div class="brand-sub">NSGA-III &mdash; Cui et al. (2025) protocol &mdash; IGD metric &mdash; 30 independent runs</div>
      </div>
    </div>
    <div class="hdr-actions">
      <a class="btn-back" href="/">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" style="width:13px;height:13px"><polyline points="15 18 9 12 15 6"/></svg>
        Menu
      </a>
      <button class="btn-theme" id="themeToggle" onclick="toggleTheme()">
        <svg id="themeIcon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="5"/>
          <line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/>
          <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
          <line x1="1" y1="12" x2="3" y2="12"/>
          <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
        </svg>
        <span id="themeLabel">Light</span>
      </button>
    </div>
  </header>

  <div class="lit-table-block">
    <div class="lit-table-caption">Tableau 2 &mdash; Param&egrave;tres exp&eacute;rimentaux reproduits</div>
    <div class="lit-table-scroll">
      <table class="lit-table">
        <thead>
          <tr><th>Suite</th><th>M</th><th>p</th><th>H</th><th>N</th><th>G</th><th>T<sub>max</sub></th></tr>
        </thead>
        <tbody>
          <tr><td>DTLZ 1&ndash;7</td><td>3</td><td>12</td><td>91</td><td>92</td><td>326</td><td>30 000</td></tr>
          <tr class="grp-end"><td></td><td>4</td><td>7</td><td>120</td><td>120</td><td>250</td><td></td></tr>
          <tr><td>MaF 1&ndash;7</td><td>3</td><td>12</td><td>91</td><td>92</td><td>326</td><td>30 000</td></tr>
          <tr><td></td><td>4</td><td>7</td><td>120</td><td>120</td><td>250</td><td></td></tr>
        </tbody>
      </table>
    </div>
    <div class="lit-table-note">
      <b>p</b> = divisions Das-Dennis &middot; <b>H</b> = points de r&eacute;f&eacute;rence &middot; <b>N</b> = taille de population (plus petit multiple de 4 &ge; H) &middot; <b>G</b> = g&eacute;n&eacute;rations (T<sub>max</sub>&nbsp;&divide;&nbsp;N) &middot; SBX &eta;=20, PM &eta;=20, p<sub>c</sub>=1.0, p<sub>m</sub>=1/D &middot; 30 runs ind&eacute;pendants, IGD sur P*&asymp;10&thinsp;000 points de r&eacute;f&eacute;rence.<br>
      <b>D</b> (variables de d&eacute;cision) suit ici la convention standard de la litt&eacute;rature, D = M + k &minus; 1 (k=5/10/20 selon le probl&egrave;me DTLZ, k=10 pour MaF1&ndash;6, k=20 pour MaF7), et non la valeur fixe D=54 de l'article
    </div>
  </div>

  <div class="rank-block">
    <div class="lit-table-caption">Classement moyen (rang IGD) &mdash; DTLZ1&ndash;7 + MaF1&ndash;7 combin&eacute;s</div>
    <div class="lit-table-cite">M&eacute;thode de Cui et al. (2025), Fig. 1/2 &mdash; pour chaque probl&egrave;me disponible, les algorithmes sont class&eacute;s par IGD moyenne (rang 1 = meilleur ; ex aequo &agrave; rang moyen), puis le rang est moyenn&eacute; sur tous les probl&egrave;mes. Plus bas = meilleur.</div>
    <div class="rank-charts" id="rankCharts"></div>
  </div>

  <div class="suite-bar" role="tablist">
    <button class="suite-tab active" data-suite="dtlz" onclick="selectSuite('dtlz')">DTLZ1&ndash;7</button>
    <button class="suite-tab"        data-suite="maf"  onclick="selectSuite('maf')">MaF1&ndash;7</button>
  </div>

  <div class="tab-bar" role="tablist" id="probTabs"></div>

  <div id="prob-desc" class="prob-desc"></div>
  <div class="m-bar" id="mBar" role="tablist"></div>
  <div class="grid2" id="results"></div>

</main>
<script>
const DATA = {{ data_json | safe }};

const DESCS = {
  DTLZ1: '<strong>DTLZ1</strong> &mdash; Front de Pareto lin&eacute;aire (un plan inclin&eacute; dans l&rsquo;espace des objectifs). Le probl&egrave;me cache 3<sup>k-1</sup> optima locaux qui pi&egrave;gent facilement l&rsquo;algorithme : c&rsquo;est un terrain multimodal difficile. <em>&Agrave; retenir :</em> un IGD faible signifie que l&rsquo;algorithme a bien converg&eacute; vers le vrai front ; un run isol&eacute; avec un IGD nettement plus &eacute;lev&eacute; s&rsquo;est probablement arr&ecirc;t&eacute; sur un optimum local au lieu du vrai front.',
  DTLZ2: '<strong>DTLZ2</strong> &mdash; Front de Pareto concave (une sph&egrave;re), sans aucun optimum local : c&rsquo;est le probl&egrave;me de r&eacute;f&eacute;rence, le plus simple des quatre. <em>&Agrave; retenir :</em> un NSGA-III bien r&eacute;gl&eacute; doit converger vers le vrai front sur presque tous les runs, donc les valeurs Best/Median/Worst doivent rester proches. Un grand &eacute;cart entre elles indique un probl&egrave;me de r&eacute;glage.',
  DTLZ3: '<strong>DTLZ3</strong> &mdash; Combine le front sph&eacute;rique de DTLZ2 avec les 3<sup>k-1</sup> optima locaux de DTLZ1 : c&rsquo;est le plus difficile des quatre probl&egrave;mes. <em>&Agrave; retenir :</em> un IGD m&eacute;dian &eacute;lev&eacute; est normal ici, car la plupart des runs restent bloqu&eacute;s sur un front local avant d&rsquo;atteindre la vraie sph&egrave;re ; seule une minorit&eacute; de runs converge compl&egrave;tement.',
  DTLZ4: '<strong>DTLZ4</strong> &mdash; M&ecirc;me front sph&eacute;rique que DTLZ2, mais la population de d&eacute;part est pouss&eacute;e artificiellement vers un p&ocirc;le du front (biais &alpha;=100). <em>&Agrave; retenir :</em> la difficult&eacute; n&rsquo;est pas de converger mais de r&eacute;partir les solutions uniform&eacute;ment sur tout le front ; un IGD &eacute;lev&eacute; signale un probl&egrave;me de diversit&eacute; (points group&eacute;s), pas un probl&egrave;me de convergence.',
  DTLZ5: '<strong>DTLZ5</strong> &mdash; Front de Pareto d&eacute;g&eacute;n&eacute;r&eacute; : la vraie solution optimale n&rsquo;est pas une surface (M-1)-dimensionnelle mais une simple courbe repli&eacute;e dans l&rsquo;espace des objectifs. <em>&Agrave; retenir :</em> un IGD &eacute;lev&eacute; signale ici que la population reste dispers&eacute;e sur toute la surface au lieu de se concentrer sur la courbe r&eacute;elle &mdash; un probl&egrave;me de forme du front, pas seulement de distance.',
  DTLZ6: '<strong>DTLZ6</strong> &mdash; Le m&ecirc;me front d&eacute;g&eacute;n&eacute;r&eacute; (une courbe) que DTLZ5, mais avec une fonction de distance beaucoup plus punitive qui ralentit fortement la convergence. <em>&Agrave; retenir :</em> c&rsquo;est le test de robustesse le plus dur pour les fronts d&eacute;g&eacute;n&eacute;r&eacute;s &mdash; un IGD &eacute;lev&eacute; reste fr&eacute;quent m&ecirc;me pour un bon algorithme ; on compare surtout la vitesse relative de convergence entre m&eacute;thodes.',
  DTLZ7: '<strong>DTLZ7</strong> &mdash; Front de Pareto disjoint : la vraie solution optimale se d&eacute;compose en 2<sup>M-1</sup> r&eacute;gions s&eacute;par&eacute;es et d&eacute;connect&eacute;es les unes des autres. <em>&Agrave; retenir :</em> ce probl&egrave;me teste la capacit&eacute; &agrave; maintenir de la diversit&eacute; sur plusieurs r&eacute;gions &agrave; la fois ; un IGD &eacute;lev&eacute; indique souvent qu&rsquo;une partie des runs ne couvre qu&rsquo;une partie des r&eacute;gions du front, pas toutes.',
  MaF1: '<strong>MaF1</strong> &mdash; DTLZ1 invers&eacute; : m&ecirc;me front lin&eacute;aire (un hyperplan) que DTLZ1, orient&eacute; vers le coin oppos&eacute; plut&ocirc;t que vers l&rsquo;origine, mais avec la fonction de distance simple de DTLZ2 (pas de pi&egrave;ge multimodal). <em>&Agrave; retenir :</em> nettement plus facile que DTLZ1 malgr&eacute; la forme similaire &mdash; un IGD &eacute;lev&eacute; signale ici un probl&egrave;me de r&eacute;partition sur l&rsquo;hyperplan, pas un blocage sur un optimum local.',
  MaF2: '<strong>MaF2</strong> &mdash; Front sph&eacute;rique comme DTLZ2, mais restreint &agrave; une petite r&eacute;gion angulaire de la sph&egrave;re, et chaque objectif poss&egrave;de sa propre fonction de distance (bas&eacute;e sur un groupe distinct de variables). <em>&Agrave; retenir :</em> teste la capacit&eacute; &agrave; localiser une zone &eacute;troite plut&ocirc;t que tout un octant &mdash; un IGD &eacute;lev&eacute; signale que l&rsquo;algorithme peine &agrave; trouver cette r&eacute;gion restreinte.',
  MaF3: '<strong>MaF3</strong> &mdash; M&ecirc;me pi&egrave;ge multimodal que DTLZ3 (3<sup>k-1</sup> optima locaux), mais le front est rendu convexe au lieu de concave. <em>&Agrave; retenir :</em> comme DTLZ3, un IGD m&eacute;dian &eacute;lev&eacute; est normal ici &mdash; la forme convexe change la g&eacute;om&eacute;trie du front, pas la difficult&eacute; de convergence sous-jacente.',
  MaF4: '<strong>MaF4</strong> &mdash; DTLZ3 invers&eacute; et &agrave; l&rsquo;&eacute;chelle : m&ecirc;me pi&egrave;ge multimodal que DTLZ3/MaF3, mais chaque objectif est en plus multipli&eacute; par 2<sup>i</sup> (de 2 &agrave; 2<sup>M</sup>). <em>&Agrave; retenir :</em> les objectifs vivent sur des &eacute;chelles tr&egrave;s diff&eacute;rentes &mdash; comparez l&rsquo;IGD relativement &agrave; 2<sup>M</sup>, pas en valeur brute, sous peine de surestimer la difficult&eacute; par rapport &agrave; DTLZ3.',
  MaF5: '<strong>MaF5</strong> &mdash; DTLZ4 &agrave; l&rsquo;&eacute;chelle : m&ecirc;me biais de diversit&eacute; que DTLZ4 (population pouss&eacute;e vers un p&ocirc;le, &alpha;=100), avec en plus des objectifs multipli&eacute;s par 2<sup>M</sup>&hellip;2. <em>&Agrave; retenir :</em> un IGD &eacute;lev&eacute; refl&egrave;te surtout un manque de diversit&eacute; (comme DTLZ4), amplifi&eacute; par l&rsquo;&eacute;chelle &mdash; &agrave; lire relativement &agrave; 2<sup>M</sup>.',
  MaF6: '<strong>MaF6</strong> &mdash; Front d&eacute;g&eacute;n&eacute;r&eacute; comme DTLZ5 (une courbe repli&eacute;e, pas une surface), mais d&eacute;fini analytiquement pour n&rsquo;importe quel M &mdash; contrairement &agrave; DTLZ5/6 ici limit&eacute;s &agrave; M=3, MaF6 est donc aussi &eacute;valu&eacute; &agrave; M=4. <em>&Agrave; retenir :</em> un IGD &eacute;lev&eacute; signale que la population reste &eacute;tal&eacute;e sur la surface au lieu de se concentrer sur la courbe r&eacute;elle.',
  MaF7: '<strong>MaF7</strong> &mdash; Strictement identique &agrave; DTLZ7 (front disjoint en 2<sup>M-1</sup> r&eacute;gions), mais &eacute;valu&eacute; ici &agrave; M=4 en plus de M=3 &mdash; l&agrave; o&ugrave; DTLZ7 s&rsquo;arr&ecirc;te &agrave; M=3 faute de front de r&eacute;f&eacute;rence pymoo au-del&agrave;. <em>&Agrave; retenir :</em> un IGD &eacute;lev&eacute; indique souvent qu&rsquo;une partie des runs ne couvre qu&rsquo;une partie des r&eacute;gions du front.',
};

const SUITE_PROBLEMS = {
  dtlz: ['DTLZ1', 'DTLZ2', 'DTLZ3', 'DTLZ4', 'DTLZ5', 'DTLZ6', 'DTLZ7'],
  maf:  ['MaF1', 'MaF2', 'MaF3', 'MaF4', 'MaF5', 'MaF6', 'MaF7'],
};
let currentSuite = 'dtlz';

const ALGO_LABEL = {
  nsga3:   'NSGA-III (classique)',
  qinsga3: 'QI-NSGA-III (quantum-inspired)',
  moead:   'MOEA/D (décomposition)',
};
const ALGO_ORDER = ['nsga3', 'qinsga3', 'moead'];

// Paramètres complets (p/H/N/G/η/...) affichés une seule fois dans le Tableau 2
// en haut de page — ici on ne rappelle que N et G pour situer la carte sans dupliquer.
const M_META = { M3: { N: 92, G: 326 }, M4: { N: 120, G: 250 } };

// ── Mean-rank bar charts (Cui et al. 2025, Fig. 1/2 style) ──────────────
// For a given M, rank the available algorithms by mean IGD on EACH problem
// (DTLZ1-7 + MaF1-7 combined, whichever have data for that M — DTLZ5/6/7
// are M=3-only, see DEGENERATE_PROBLEMS below), then average each
// algorithm's rank across every problem it was ranked on. Ties (equal mean
// IGD) share the average of the tied rank positions, the standard
// competition-ranking convention. Driven entirely by ALGO_ORDER.filter(a
// => mData[a]), so this extends to a 3rd algorithm automatically once its
// benchmark results exist -- no changes needed here when that happens.
function computeMeanRanks(mKey) {
  const problems = [
    ...SUITE_PROBLEMS.dtlz.map(name => ({ suite: 'dtlz', name })),
    ...SUITE_PROBLEMS.maf.map(name => ({ suite: 'maf', name })),
  ];
  const rankSums = {}, rankCounts = {};
  problems.forEach(({ suite, name }) => {
    const mData = ((DATA[suite] || {})[name] || {})[mKey];
    if (!mData) return;
    const present = ALGO_ORDER.filter(a => mData[a]);
    if (present.length < 2) return;
    const sorted = [...present].sort((a, b) => mData[a].mean - mData[b].mean);
    let i = 0;
    while (i < sorted.length) {
      let j = i;
      while (j + 1 < sorted.length && mData[sorted[j + 1]].mean === mData[sorted[i]].mean) j++;
      const avgRank = (i + 1 + j + 1) / 2;
      for (let k = i; k <= j; k++) {
        const a = sorted[k];
        rankSums[a] = (rankSums[a] || 0) + avgRank;
        rankCounts[a] = (rankCounts[a] || 0) + 1;
      }
      i = j + 1;
    }
  });
  const meanRanks = {};
  ALGO_ORDER.forEach(a => { if (rankCounts[a]) meanRanks[a] = { rank: rankSums[a] / rankCounts[a], n: rankCounts[a] }; });
  return meanRanks;
}

function renderRankChart(mKey) {
  const n = mKey === 'M3' ? 3 : 4;
  const meanRanks = computeMeanRanks(mKey);
  const algos = ALGO_ORDER.filter(a => meanRanks[a]);
  if (!algos.length) {
    return `<div class="rank-card"><div class="rank-card-title">M = ${n}</div><div class="nodata" style="padding:14px">Pas assez de r&eacute;sultats pour classer (au moins 2 algorithmes requis sur un m&ecirc;me probl&egrave;me).</div></div>`;
  }
  const maxRank = algos.length + 1; // headroom above the worst possible rank
  const OX = 16, PLOT_W = 280, PLOT_H = 110, PX = 10, PY_TOP = 10, BAR_GAP = 14;
  const W = OX + PLOT_W;
  const barW = (PLOT_W - 2 * PX - BAR_GAP * (algos.length - 1)) / algos.length;
  const yOf = v => PY_TOP + (1 - v / maxRank) * PLOT_H;
  const gridLines = [];
  for (let g = 0; g <= maxRank; g++) {
    const y = yOf(g);
    gridLines.push(`<line x1="${OX + PX}" y1="${y.toFixed(1)}" x2="${OX + PLOT_W - PX}" y2="${y.toFixed(1)}" stroke="var(--border)" stroke-width="1"/>`);
    gridLines.push(`<text class="rank-grid-lbl" x="${OX + PX - 3}" y="${(y + 3).toFixed(1)}">${g}</text>`);
  }
  const bars = algos.map((a, i) => {
    const r = meanRanks[a].rank;
    const x = OX + PX + i * (barW + BAR_GAP);
    const yTop = yOf(r);
    const h = yOf(0) - yTop;
    const cx = x + barW / 2;
    return `
      <rect x="${x.toFixed(1)}" y="${yTop.toFixed(1)}" width="${barW.toFixed(1)}" height="${Math.max(h, 0).toFixed(1)}" rx="4" fill="var(--accent)" opacity=".92">
        <title>${ALGO_LABEL[a]} — rang moyen ${r.toFixed(2)} (sur ${meanRanks[a].n} problèmes, M=${n})</title>
      </rect>
      <text class="rank-bar-lbl" x="${cx.toFixed(1)}" y="${(yTop - 5).toFixed(1)}">${r.toFixed(2)}</text>
      <text class="rank-axis-lbl" x="${cx.toFixed(1)}" y="${(yOf(0) + 14).toFixed(1)}">${ALGO_LABEL[a].split(' (')[0]}</text>`;
  }).join('');
  const titleX = (OX / 2 - 2).toFixed(1), titleY = (PY_TOP + PLOT_H / 2).toFixed(1);
  const axisTitle = `<text class="rank-y-title" x="${titleX}" y="${titleY}" transform="rotate(-90 ${titleX} ${titleY})">Mean rank</text>`;
  return `<div class="rank-card">
    <div class="rank-card-title">M = ${n}</div>
    <div class="rank-card-sub">rang moyen IGD, ${algos.length} algorithme${algos.length > 1 ? 's' : ''} &mdash; plus bas = meilleur</div>
    <svg class="rank-svg" viewBox="0 0 ${W} ${PLOT_H + 24}" style="height:${PLOT_H + 24}px">${axisTitle}${gridLines.join('')}${bars}</svg>
  </div>`;
}

function renderRankCharts() {
  document.getElementById('rankCharts').innerHTML = ['M3', 'M4'].map(renderRankChart).join('');
}

// Format décimal court, en complément de la notation scientifique.
function fmtDec(v) {
  if (v === 0) return '0';
  const abs = Math.abs(v);
  const dec = abs < 0.001 ? 6 : abs < 0.01 ? 5 : abs < 1 ? 4 : 3;
  return v.toFixed(dec);
}

// Seuils absolus (indépendants du run) : vert = succès, orange = optimum local, rouge = échec de convergence.
const IGD_GREEN = [13, 122, 85], IGD_ORANGE = [217, 119, 6], IGD_RED = [185, 28, 28];
function _lerpRgb(c0, c1, t) {
  t = Math.min(1, Math.max(0, t));
  return `rgb(${Math.round(c0[0] + (c1[0] - c0[0]) * t)},${Math.round(c0[1] + (c1[1] - c0[1]) * t)},${Math.round(c0[2] + (c1[2] - c0[2]) * t)})`;
}
function igdColor(v) {
  if (v <= 0.06) return `rgb(${IGD_GREEN.join(',')})`;
  if (v <= 0.25) return _lerpRgb(IGD_GREEN, IGD_ORANGE, (v - 0.06) / (0.25 - 0.06));
  if (v <= 0.55) return `rgb(${IGD_ORANGE.join(',')})`;
  if (v <= 0.75) return _lerpRgb(IGD_ORANGE, IGD_RED, (v - 0.55) / (0.75 - 0.55));
  return `rgb(${IGD_RED.join(',')})`;
}
function igdStatus(v) {
  if (v <= 0.06) return 'Succ&egrave;s';
  if (v <= 0.25) return 'Convergence partielle';
  if (v <= 0.55) return 'Optimum local';
  if (v <= 0.75) return 'Convergence faible';
  return '&Eacute;chec';
}

// Nuage de points en "marches d'escalier" : rang (x, trié) x valeur (y, échelle log)
// pour rendre visibles les paliers (optima locaux) et les décrochages, avec un code
// couleur à seuils fixes plutôt qu'un dégradé relatif au run affiché.
function drawDots(runs, svgEl) {
  if (!runs || !runs.length) return;
  const s = [...runs].sort((a, b) => a - b), n = s.length;
  const mn = s[0], mx = s[n - 1];
  const W = 360, PLOT_H = 56, LABEL_H = 18, H = PLOT_H + LABEL_H, PX = 16, PY = 8, R = 4.5;
  const logMn = Math.log10(Math.max(mn, 1e-12));
  const logMx = Math.log10(Math.max(mx, 1e-12));
  const rng = logMx - logMn || 1;
  const pts = s.map((v, i) => {
    const t = (Math.log10(Math.max(v, 1e-12)) - logMn) / rng;
    const x = PX + (n > 1 ? i / (n - 1) : 0.5) * (W - 2 * PX);
    const y = PY + (1 - t) * (PLOT_H - 2 * PY);
    return { v, x, y };
  });
  const line = pts.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
  const dots = pts.map(p =>
    `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="${R}" fill="${igdColor(p.v)}" opacity=".92" stroke="rgba(0,0,0,.18)" stroke-width=".5"><title>IGD = ${p.v.toExponential(4)} (${fmtDec(p.v)})</title></circle>`
  ).join('');
  svgEl.innerHTML = `
    <polyline points="${line}" fill="none" stroke="var(--border)" stroke-width="1.25"/>
    ${dots}
    <text x="${PX}" y="${PLOT_H + 13}" font-size="8.5" fill="var(--text-3)">${mn.toExponential(2)} <tspan font-size="7">(${fmtDec(mn)})</tspan></text>
    <text x="${W - PX}" y="${PLOT_H + 13}" font-size="8.5" fill="var(--text-3)" text-anchor="end">${mx.toExponential(2)} <tspan font-size="7">(${fmtDec(mx)})</tspan></text>`;
}

function renderRunsTable(details) {
  if (!details || !details.length) return '';
  const rows = [...details].sort((a, b) => a.run - b.run).map(r => `
        <tr>
          <td>#${r.run}</td>
          <td>${r.seed}</td>
          <td>${r.igd.toExponential(3)}</td>
          <td>${fmtDec(r.igd)}</td>
          <td><span class="run-badge"><i class="run-dot" style="background:${igdColor(r.igd)}"></i>${igdStatus(r.igd)}</span></td>
        </tr>`).join('');
  return `
      <div class="chart-lbl" style="margin-top:20px">D&eacute;tail des ${details.length} runs (num&eacute;ro, seed, IGD)</div>
      <div class="table-wrap">
        <table class="runs-table">
          <thead><tr><th>Run</th><th>Seed</th><th>IGD (sci.)</th><th>IGD (d&eacute;c.)</th><th>Statut</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>`;
}

function renderCard(mKey, algo, d) {
  const dataKey = `${mKey}:${algo}`;
  const nObj = mKey === 'M3' ? 3 : 4;
  return `<div class="rcard">
    <div class="rcard-hdr">
      <h3>${ALGO_LABEL[algo]} &mdash; M <span class="param-lbl">(objectifs)</span> = ${nObj}</h3>
      <div class="rcard-sub">N=${M_META[mKey].N}, G=${M_META[mKey].G} &nbsp;|&nbsp; ${d.n_runs} runs</div>
    </div>
    <div class="rcard-body">
      <div class="stat-row">
        <div class="stat-box best"><div class="stat-lbl">Best</div><div class="stat-val">${d.best.toExponential(3)}<span class="stat-val-dec">${fmtDec(d.best)}</span></div></div>
        <div class="stat-box med"><div class="stat-lbl">Median</div><div class="stat-val">${d.median.toExponential(3)}<span class="stat-val-dec">${fmtDec(d.median)}</span></div></div>
        <div class="stat-box mean"><div class="stat-lbl">Mean <span class="param-lbl">(&plusmn;&sigma;)</span></div><div class="stat-val">${d.mean.toExponential(3)}<span class="stat-val-dec">${fmtDec(d.mean)} &plusmn; ${fmtDec(d.std)}</span></div></div>
        <div class="stat-box worst"><div class="stat-lbl">Worst</div><div class="stat-val">${d.worst.toExponential(3)}<span class="stat-val-dec">${fmtDec(d.worst)}</span></div></div>
      </div>
      <div class="stat-note">Mean = statistique compar&eacute;e dans Cui et al. 2025 (PlatEMO) &mdash; Median = statistique la plus robuste aux runs rat&eacute;s</div>
      <div class="chart-lbl">Run distribution &mdash; ${d.n_runs} runs, tri&eacute;s par IGD (rang &times; valeur, &eacute;chelle log)</div>
      <svg class="dot-svg" viewBox="0 0 360 74" style="height:74px" data-key="${dataKey}"></svg>
      <div class="chart-legend">
        <span><i style="background:rgb(${IGD_GREEN.join(',')})"></i>Succ&egrave;s (IGD &le; 0.06)</span>
        <span><i style="background:rgb(${IGD_ORANGE.join(',')})"></i>Optimum local (0.25&ndash;0.55)</span>
        <span><i style="background:rgb(${IGD_RED.join(',')})"></i>&Eacute;chec de convergence (&gt; 0.75)</span>
      </div>
      ${renderRunsTable(d.run_details)}
    </div>
  </div>`;
}

// DTLZ5/6 have a degenerate (curve) true front and DTLZ7 a disconnected one;
// pymoo only ships a precomputed reference front for these three at M=3
// (pareto_front() raises "Not implemented yet." otherwise) — so M=4 is
// structurally unavailable here, not just "not run yet".
const DEGENERATE_PROBLEMS = new Set(['DTLZ5', 'DTLZ6', 'DTLZ7']);

function renderProbTabs() {
  const probs = SUITE_PROBLEMS[currentSuite];
  document.getElementById('probTabs').innerHTML = probs
    .map((p, i) => `<button class="tab${i === 0 ? ' active' : ''}" data-prob="${p}" onclick="selectProb('${p}')">${p}</button>`)
    .join('');
}

let currentProb = null;
let currentM = 'M3';
let currentPd = {};

function selectSuite(suite) {
  currentSuite = suite;
  document.querySelectorAll('.suite-tab').forEach(b => b.classList.toggle('active', b.dataset.suite === suite));
  renderProbTabs();
  selectProb(SUITE_PROBLEMS[suite][0]);
}

function selectProb(name) {
  currentProb = name;
  document.querySelectorAll('#probTabs .tab').forEach(b => b.classList.toggle('active', b.dataset.prob === name));
  document.getElementById('prob-desc').innerHTML = DESCS[name] || '';
  currentPd = (DATA[currentSuite] || {})[name] || {};
  const keys = ['M3', 'M4'].filter(k => currentPd[k]);
  if (!keys.includes(currentM)) currentM = keys[0] || 'M3';
  renderMBar(keys);
  renderResultsForM();
}

// M3/M4 sont des onglets exclusifs : un seul jeu de résultats (graphe + tableau) affiché à la fois.
function renderMBar(keys) {
  document.getElementById('mBar').innerHTML = ['M3', 'M4'].map(k => {
    const n = k === 'M3' ? 3 : 4;
    const disabled = !keys.includes(k);
    return `<button class="m-tab${k === currentM ? ' active' : ''}" ${disabled ? 'disabled' : ''} onclick="selectM('${k}')">M&nbsp;=&nbsp;${n}</button>`;
  }).join('');
}

function selectM(k) {
  currentM = k;
  document.querySelectorAll('#mBar .m-tab').forEach((b, i) => b.classList.toggle('active', (i === 0 ? 'M3' : 'M4') === k));
  renderResultsForM();
}

function renderResultsForM() {
  const results = document.getElementById('results');
  const mData = currentPd[currentM] || {};
  const isDegenerateM4 = DEGENERATE_PROBLEMS.has(currentProb) && currentM === 'M4';
  const availableAlgos = ALGO_ORDER.filter(a => mData[a]);
  let html;
  if (availableAlgos.length) {
    html = `<div class="algo-compare">${availableAlgos.map(a => renderCard(currentM, a, mData[a])).join('')}</div>`;
  } else if (isDegenerateM4) {
    html = `<div class="rcard"><div class="nodata">M=4 is not shown for ${currentProb}: it has a degenerate/disconnected true Pareto front, and pymoo only ships a reference front for it at M=3 &mdash; IGD can&rsquo;t be scored otherwise.</div></div>`;
  } else {
    html = `<div class="rcard"><div class="nodata">No benchmark results yet for this problem &mdash; run <code>python -m Validation.Benchmarking.${currentSuite}.main_validation</code> to generate them.</div></div>`;
  }
  results.innerHTML = html;
  document.querySelectorAll('[data-key]').forEach(svg => {
    const [mKey, algo] = svg.dataset.key.split(':');
    const dd = (currentPd[mKey] || {})[algo];
    if (dd) drawDots(dd.runs, svg);
  });
}

const MOON = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>';
const SUN  = '<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>';
const ti = document.getElementById('themeIcon');
const tl = document.getElementById('themeLabel');
function applyTheme(t) {
  document.documentElement.setAttribute('data-theme', t);
  if (t === 'dark') { ti.innerHTML = MOON; ti.setAttribute('fill', 'currentColor'); ti.removeAttribute('stroke'); tl.textContent = 'Dark'; }
  else { ti.innerHTML = SUN; ti.setAttribute('stroke', 'currentColor'); ti.setAttribute('fill', 'none'); tl.textContent = 'Light'; }
  localStorage.setItem('irp-theme', t);
}
function toggleTheme() { applyTheme(document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark'); }
(function () { applyTheme(localStorage.getItem('irp-theme') || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')); })();

renderRankCharts();
renderProbTabs();
selectProb(SUITE_PROBLEMS[currentSuite][0]);
</script>
</body>
</html>"""


MENU_TEMPLATE = """<!DOCTYPE html>
<html lang="fr" data-theme="light">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IRP — Suite de résolution</title>
<style>
:root {
  --bg:            #f2f5f9;
  --surface:       #ffffff;
  --surface-2:     #f8fafc;
  --border:        #dce3ec;
  --border-focus:  #93b4ca;
  --text:          #0f1923;
  --text-2:        #4e6070;
  --text-3:        #8fa0b0;
  --accent:        #0f6a87;
  --accent-dim:    #e6f3f8;
  --accent-hover:  #0a5570;
  --accent-fg:     #ffffff;
  --green:         #0d7a55;
  --green-bg:      #e6f5ee;
  --shadow-sm:     0 1px 2px rgba(0,0,0,.06), 0 3px 8px rgba(0,0,0,.04);
  --shadow-md:     0 2px 6px rgba(0,0,0,.07), 0 8px 24px rgba(0,0,0,.07);
  --shadow-lg:     0 4px 12px rgba(0,0,0,.09), 0 16px 40px rgba(0,0,0,.08);
  --radius:        14px;
  --radius-sm:     8px;
  --c1: #0f6a87; --c1-bg: #e6f3f8;
  --c2: #0d7a55; --c2-bg: #e6f5ee;
  --c3: #6741d9; --c3-bg: #ede9fb;
  --c4: #b45309; --c4-bg: #fef3e2;
}
[data-theme="dark"] {
  --bg:            #0b0f16;
  --surface:       #131922;
  --surface-2:     #1a2130;
  --border:        #1e2a3a;
  --border-focus:  #2d4a62;
  --text:          #e0e8f2;
  --text-2:        #8097b0;
  --text-3:        #3d5068;
  --accent:        #38b2d4;
  --accent-dim:    #0d2535;
  --accent-hover:  #2aa3c5;
  --accent-fg:     #060f18;
  --green:         #34c98a;
  --green-bg:      #0a2419;
  --shadow-sm:     0 1px 2px rgba(0,0,0,.3), 0 3px 8px rgba(0,0,0,.25);
  --shadow-md:     0 2px 6px rgba(0,0,0,.35), 0 8px 24px rgba(0,0,0,.3);
  --shadow-lg:     0 4px 12px rgba(0,0,0,.4), 0 16px 40px rgba(0,0,0,.35);
  --c1: #38b2d4; --c1-bg: #0d2535;
  --c2: #34c98a; --c2-bg: #0a2419;
  --c3: #a78bfa; --c3-bg: #1e1545;
  --c4: #fbbf24; --c4-bg: #2a1d06;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  min-height: 100vh;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  background: var(--bg);
  color: var(--text);
  -webkit-font-smoothing: antialiased;
  transition: background .2s, color .2s;
}

/* ── Layout ── */
.page { max-width: 1020px; margin: 0 auto; padding: 36px 24px 60px; }

/* ── Header ── */
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 32px;
  padding-bottom: 24px;
  border-bottom: 1px solid var(--border);
}
.brand { display: flex; align-items: center; gap: 16px; }
.brand-icon {
  width: 48px; height: 48px;
  border-radius: 12px;
  background: var(--accent);
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
  box-shadow: 0 2px 8px rgba(15,106,135,.35);
}
.brand-icon svg { width: 24px; height: 24px; fill: #fff; }
.brand-name {
  font-size: 20px;
  font-weight: 700;
  letter-spacing: -.3px;
  color: var(--text);
}
.brand-sub {
  font-size: 12.5px;
  color: var(--text-2);
  margin-top: 1px;
}
.header-right { display: flex; align-items: center; gap: 10px; }
.badge-env {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: .04em;
  padding: 4px 10px;
  border-radius: 99px;
  background: var(--accent-dim);
  color: var(--accent);
  border: 1px solid color-mix(in srgb, var(--accent) 20%, transparent);
}
.btn-theme {
  display: flex; align-items: center; gap: 5px;
  height: 34px;
  padding: 0 13px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text-2);
  font-size: 12.5px;
  font-weight: 500;
  cursor: pointer;
  transition: border-color .15s, background .15s, color .15s;
  outline: none;
}
.btn-theme:hover { border-color: var(--border-focus); color: var(--text); }
.btn-theme svg { width: 14px; height: 14px; flex-shrink: 0; }

/* ── Instance panel ── */
.panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 20px;
  margin-bottom: 24px;
  box-shadow: var(--shadow-sm);
}
.panel-hdr {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
}
.panel-icon {
  width: 28px; height: 28px;
  border-radius: 7px;
  background: var(--accent-dim);
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}
.panel-icon svg { width: 14px; height: 14px; stroke: var(--accent); fill: none; stroke-width: 2; stroke-linecap: round; }
.panel-title {
  font-size: 12px;
  font-weight: 700;
  letter-spacing: .07em;
  text-transform: uppercase;
  color: var(--text-2);
}
.inst-chips {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}
.inst-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 14px;
  border-radius: 99px;
  border: 1px solid var(--border);
  background: var(--surface-2);
  color: var(--text-2);
  font-size: 12.5px;
  font-weight: 600;
  cursor: pointer;
  transition: all .15s;
  white-space: nowrap;
  user-select: none;
}
.inst-chip:hover {
  border-color: var(--border-focus);
  background: var(--surface);
  color: var(--text);
}
.inst-chip.selected {
  background: var(--accent);
  border-color: var(--accent);
  color: var(--accent-fg);
  box-shadow: 0 2px 6px rgba(15,106,135,.3);
}
.chip-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: currentColor;
  opacity: .65;
  flex-shrink: 0;
}

/* ── Section header ── */
.section-hdr {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 14px;
}
.section-hdr-line {
  flex: 1;
  height: 1px;
  background: var(--border);
}
.section-hdr-label {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--text-3);
  white-space: nowrap;
}

/* ── Part (academic section) header ── */
.part { margin-top: 44px; }
.part:first-of-type { margin-top: 0; }
.part-hdr {
  display: flex;
  align-items: flex-start;
  gap: 16px;
  margin-bottom: 22px;
  padding-bottom: 18px;
  border-bottom: 2px solid var(--text);
}
.part-num {
  font-size: 13px;
  font-weight: 800;
  letter-spacing: .04em;
  color: var(--accent-fg);
  background: var(--text);
  width: 30px; height: 30px;
  border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
  margin-top: 2px;
}
.part-eyebrow {
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 3px;
}
.part-title {
  font-size: 19px;
  font-weight: 800;
  letter-spacing: -.3px;
  color: var(--text);
}
.part-sub {
  font-size: 12.5px;
  color: var(--text-2);
  margin-top: 4px;
  line-height: 1.6;
  max-width: 640px;
}

/* ── Grid ── */
.modules {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

/* ── Card ── */
.card {
  display: flex;
  flex-direction: column;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow-sm);
  overflow: hidden;
  transition: box-shadow .2s, border-color .2s, transform .15s;
  position: relative;
}
.card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: var(--card-color, var(--accent));
  border-radius: var(--radius) var(--radius) 0 0;
}
.card:hover {
  box-shadow: var(--shadow-lg);
  border-color: var(--border-focus);
  transform: translateY(-1px);
}
.card-body {
  flex: 1;
  padding: 14px 14px 10px;
  display: flex;
  flex-direction: column;
  gap: 7px;
}
.card-top {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 10px;
}
.card-icon {
  width: 38px; height: 38px;
  border-radius: 10px;
  background: var(--card-icon-bg, var(--accent-dim));
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}
.card-icon svg { width: 18px; height: 18px; }
.card-badges { display: flex; align-items: center; gap: 6px; }
.badge-num {
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: .05em;
  color: var(--text-3);
}
.badge-status {
  font-size: 10.5px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 99px;
}
.badge-status.active {
  background: var(--green-bg);
  color: var(--green);
}
.card-title {
  font-size: 15px;
  font-weight: 700;
  letter-spacing: -.15px;
  color: var(--text);
  line-height: 1.3;
}
.card-desc {
  font-size: 12.5px;
  color: var(--text-2);
  line-height: 1.6;
  flex: 1;
}
.runs-row {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  padding-top: 2px;
}
.runs-label {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: .05em;
  text-transform: uppercase;
  color: var(--text-3);
  white-space: nowrap;
}
.runs-group { display: flex; gap: 4px; flex-wrap: wrap; }
.card-tags { display: flex; gap: 5px; flex-wrap: wrap; margin-top: 2px; }
.card-tag {
  font-size: 10.5px;
  font-weight: 600;
  letter-spacing: .03em;
  padding: 2px 8px;
  border-radius: 99px;
  border: 1px solid var(--border);
  color: var(--text-3);
  background: var(--surface-2);
  white-space: nowrap;
}
.runs-btn {
  padding: 4px 10px;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: var(--surface-2);
  color: var(--text-2);
  font-size: 11.5px;
  font-weight: 600;
  cursor: pointer;
  transition: all .13s;
  user-select: none;
}
.runs-btn:hover { border-color: var(--border-focus); color: var(--text); background: var(--surface); }
.runs-btn.selected {
  background: var(--card-color, var(--accent));
  border-color: var(--card-color, var(--accent));
  color: #fff;
}

/* ── Card footer ── */
.card-footer {
  padding: 12px 20px 16px;
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  align-items: center;
}
.btn {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  height: 34px;
  padding: 0 16px;
  border-radius: var(--radius-sm);
  border: 1px solid transparent;
  font-size: 12.5px;
  font-weight: 600;
  text-decoration: none;
  cursor: pointer;
  white-space: nowrap;
  transition: all .15s;
  letter-spacing: .01em;
}
.btn svg { width: 13px; height: 13px; flex-shrink: 0; }
.btn-primary {
  background: var(--card-color, var(--accent));
  color: #fff;
  box-shadow: 0 1px 3px rgba(0,0,0,.2);
}
.btn-primary:hover { filter: brightness(1.08); box-shadow: 0 2px 8px rgba(0,0,0,.25); }
.btn-secondary {
  background: var(--surface-2);
  border-color: var(--border);
  color: var(--text-2);
}
.btn-secondary:hover { border-color: var(--border-focus); color: var(--text); background: var(--surface); }

/* ── Footer ── */
.page-footer {
  margin-top: 40px;
  padding-top: 20px;
  border-top: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  flex-wrap: wrap;
}
.footer-text { font-size: 12px; color: var(--text-3); }
.footer-pills { display: flex; gap: 6px; flex-wrap: wrap; }
.footer-pill {
  font-size: 11px;
  font-weight: 600;
  padding: 3px 9px;
  border-radius: 99px;
  border: 1px solid var(--border);
  color: var(--text-3);
  background: var(--surface);
}

/* ── Benchmarking link card ── */
.blink {
  display: flex;
  align-items: center;
  gap: 14px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 20px;
  text-decoration: none;
  color: var(--text);
  box-shadow: var(--shadow-sm);
  transition: box-shadow .2s, border-color .2s, transform .15s;
  position: relative;
  margin-top: 16px;
}
.blink::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: #059669;
  border-radius: var(--radius) var(--radius) 0 0;
}
.blink:hover {
  box-shadow: var(--shadow-lg);
  border-color: var(--border-focus);
  transform: translateY(-1px);
}
.blink-icon {
  width: 38px; height: 38px;
  border-radius: 10px;
  background: #ecfdf5;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}
[data-theme="dark"] .blink-icon { background: #0a2419; }
.blink-icon svg { width: 18px; height: 18px; stroke: #059669; fill: none; stroke-width: 1.8; stroke-linecap: round; }
.blink-title { font-size: 15px; font-weight: 700; color: var(--text); }
.blink-sub { font-size: 12.5px; color: var(--text-2); margin-top: 3px; line-height: 1.5; }
.blink-arrow { width: 18px; height: 18px; margin-left: auto; color: var(--text-3); flex-shrink: 0; }

@media (max-width: 680px) {
  .modules { grid-template-columns: 1fr; }
  .page { padding: 20px 16px 40px; }
  .header { flex-direction: column; align-items: flex-start; }
  .badge-env { display: none; }
}
</style>
</head>
<body>
<main class="page">

  <header class="header">
    <div class="brand">
      <div class="brand-icon">
        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path d="M3 3h7v7H3V3zm11 0h7v7h-7V3zm0 11h7v7h-7v-7zM3 14h7v7H3v-7z"/>
        </svg>
      </div>
      <div>
        <div class="brand-name">Suite de résolution IRP</div>
        <div class="brand-sub">Problème de tournées avec gestion des stocks — optimisation multi-objectif</div>
      </div>
    </div>
    <div class="header-right">
      <button class="btn-theme" id="themeToggle" onclick="toggleTheme()" aria-label="Changer le thème">
        <svg id="themeIcon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="5"/>
          <line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/>
          <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
          <line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/>
          <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
        </svg>
        <span id="themeLabel">Clair</span>
      </button>
    </div>
  </header>

  <div class="part">
  <div class="part-hdr">
    <div class="part-num">1</div>
    <div>
      <div class="part-eyebrow">Partie 1</div>
      <div class="part-title">Résolution &amp; validation par instance</div>
      <div class="part-sub">Quatre méthodes appliquées aux instances du problème de tournées avec gestion des stocks (IRP) — un solveur exact par objectif, un solveur scalarisé, et deux métaheuristiques many-objective (NSGA-III et sa variante quantique-inspirée QI-NSGA-III).</div>
    </div>
  </div>

  <div class="panel">
    <div class="panel-hdr">
      <div class="panel-icon">
        <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M12 2v3m0 14v3M2 12h3m14 0h3M4.93 4.93l2.12 2.12m9.9 9.9 2.12 2.12M4.93 19.07l2.12-2.12m9.9-9.9 2.12-2.12"/></svg>
      </div>
      <span class="panel-title">Instance</span>
    </div>
    <div class="inst-chips" id="instanceBar">
      {% for key in inst_keys %}
      <button class="inst-chip{% if key == default_inst %} selected{% endif %}" data-instance="{{ key }}" onclick="selectInstance('{{ key }}')"><span class="chip-dot"></span>{{ key }} clients</button>
      {% endfor %}
    </div>
  </div>

  <div class="section-hdr">
    <span class="section-hdr-label">Modules disponibles</span>
    <div class="section-hdr-line"></div>
    <span class="section-hdr-label">4 actifs</span>
  </div>

  <section class="modules" aria-label="Modules du solveur">

    <!-- ── 01 Objective Calibration ── -->
    <article class="card" style="--card-color:var(--c1);--card-icon-bg:var(--c1-bg)">
      <div class="card-body">
        <div class="card-top">
          <div class="card-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="var(--c1)" stroke-width="1.8" stroke-linecap="round">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
            </svg>
          </div>
          <div class="card-badges">
            <span class="badge-num">01</span>
            <span class="badge-status active">Actif</span>
          </div>
        </div>
        <h2 class="card-title">Calibration des objectifs</h2>
        <p class="card-desc">Calibre et résout le MIP many-objectifs individuellement — dérive les bornes budgétaires C_max, E_max, T_max, B pour les quatre objectifs.</p>
      </div>
      <div class="card-footer">
        <a class="btn btn-primary" id="oc-run" href="{{ url_for('run_objective_calibration_route') }}?instance=15" target="_blank" rel="noopener noreferrer">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
          Lancer
        </a>
        <a class="btn btn-secondary" id="oc-report" href="{{ url_for('objective_calibration_report') }}?instance=15" target="_blank" rel="noopener noreferrer">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
          Dernier rapport
        </a>
      </div>
    </article>

    <!-- ── 02 Function Merge ── -->
    <article class="card" style="--card-color:var(--c2);--card-icon-bg:var(--c2-bg)">
      <div class="card-body">
        <div class="card-top">
          <div class="card-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="var(--c2)" stroke-width="1.8" stroke-linecap="round">
              <circle cx="12" cy="12" r="10"/><path d="M8 12h8M12 8v8"/>
            </svg>
          </div>
          <div class="card-badges">
            <span class="badge-num">02</span>
            <span class="badge-status active">Actif</span>
          </div>
        </div>
        <h2 class="card-title">Fusion des fonctions</h2>
        <p class="card-desc">Résolution CPLEX scalarisée en une seule exécution — minimise simultanément f1 + f2 + f3 + f4 (coût, CO₂, temps et fonds de roulement).</p>
      </div>
      <div class="card-footer">
        <a class="btn btn-primary" id="fm-run" href="{{ url_for('run_function_merge_route') }}?instance=15" target="_blank" rel="noopener noreferrer">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
          Lancer
        </a>
        <a class="btn btn-secondary" id="fm-report" href="{{ url_for('function_merge_report') }}?instance=15" target="_blank" rel="noopener noreferrer">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
          Dernier rapport
        </a>
      </div>
    </article>

    <!-- ── 03 NSGA-III ── -->
    <article class="card" style="--card-color:var(--c3);--card-icon-bg:var(--c3-bg)">
      <div class="card-body">
        <div class="card-top">
          <div class="card-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="var(--c3)" stroke-width="1.8" stroke-linecap="round">
              <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
            </svg>
          </div>
          <div class="card-badges">
            <span class="badge-num">03</span>
            <span class="badge-status active">Actif</span>
          </div>
        </div>
        <h2 class="card-title">NSGA-III</h2>
        <p class="card-desc">Algorithme génétique many-objectifs. Construit un front de Pareto dense sur les quatre objectifs de l'IRP.</p>
        <div class="runs-row">
          <span class="runs-label">Exécutions</span>
          <div class="runs-group" id="n3RunsBtns">
            <button class="runs-btn selected" data-runs="1"  onclick="setN3Runs(1)">1×</button>
            <button class="runs-btn"          data-runs="3"  onclick="setN3Runs(3)">3×</button>
            <button class="runs-btn"          data-runs="5"  onclick="setN3Runs(5)">5×</button>
            <button class="runs-btn"          data-runs="10" onclick="setN3Runs(10)">10×</button>
            <button class="runs-btn"          data-runs="20" onclick="setN3Runs(20)">20×</button>
            <button class="runs-btn"          data-runs="30" onclick="setN3Runs(30)">30×</button>
          </div>
        </div>
      </div>
      <div class="card-footer">
        <a class="btn btn-primary" id="n3-run" href="{{ url_for('run_nsga3_route') }}?instance=25&runs=1" target="_blank" rel="noopener noreferrer">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
          Lancer
        </a>
        <a class="btn btn-secondary" id="n3-report" href="{{ url_for('nsga3_report') }}?instance=25" target="_blank" rel="noopener noreferrer">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
          Dernier rapport
        </a>
      </div>
    </article>

    <!-- ── 04 QI-NSGA-III ── -->
    <article class="card" style="--card-color:var(--c4);--card-icon-bg:var(--c4-bg)">
      <div class="card-body">
        <div class="card-top">
          <div class="card-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="var(--c4)" stroke-width="1.8" stroke-linecap="round">
              <path d="M12 2a10 10 0 1 0 10 10"/><path d="M12 6a6 6 0 1 0 6 6"/><circle cx="12" cy="12" r="2"/>
            </svg>
          </div>
          <div class="card-badges">
            <span class="badge-num">04</span>
            <span class="badge-status active">Actif</span>
          </div>
        </div>
        <h2 class="card-title">QI-NSGA-III</h2>
        <p class="card-desc">Variante quantique-inspirée de NSGA-III, dont la convergence est pilotée par des portes de rotation adaptatives.</p>
        <div class="runs-row">
          <span class="runs-label">Exécutions</span>
          <div class="runs-group" id="qi3RunsBtns">
            <button class="runs-btn selected" data-runs="1"  onclick="setQi3Runs(1)">1×</button>
            <button class="runs-btn"          data-runs="3"  onclick="setQi3Runs(3)">3×</button>
            <button class="runs-btn"          data-runs="5"  onclick="setQi3Runs(5)">5×</button>
            <button class="runs-btn"          data-runs="10" onclick="setQi3Runs(10)">10×</button>
            <button class="runs-btn"          data-runs="20" onclick="setQi3Runs(20)">20×</button>
            <button class="runs-btn"          data-runs="30" onclick="setQi3Runs(30)">30×</button>
          </div>
        </div>
      </div>
      <div class="card-footer">
        <a class="btn btn-primary" id="qi3-run" href="{{ url_for('run_qinsga3_route') }}?instance=25&runs=1" target="_blank" rel="noopener noreferrer">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
          Lancer
        </a>
        <a class="btn btn-secondary" id="qi3-report" href="{{ url_for('qinsga3_report') }}?instance=25" target="_blank" rel="noopener noreferrer">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
          Dernier rapport
        </a>
      </div>
    </article>

  </section>
  </div>

  <div class="part">
  <div class="part-hdr">
    <div class="part-num">2</div>
    <div>
      <div class="part-eyebrow">Partie 2</div>
      <div class="part-title">Benchmarking académique</div>
      <div class="part-sub">Validation de NSGA-III, QI-NSGA-III et MOEA/D sur les suites de test de référence de la littérature (DTLZ1&ndash;7, MaF1&ndash;7), en reproduisant le protocole expérimental de Cui et al. (2025) &mdash; population, opérateurs, budget d'évaluations et métrique IGD.</div>
    </div>
  </div>

  <a class="blink" href="{{ url_for('benchmarking') }}" target="_blank" rel="noopener noreferrer">
    <div class="blink-icon">
      <svg viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
    </div>
    <div>
      <div class="blink-title">Benchmarking DTLZ / MaF</div>
      <div class="blink-sub">NSGA-III, QI-NSGA-III et MOEA/D sur DTLZ1&ndash;7 / MaF1&ndash;7 &mdash; résultats IGD &mdash; 30 exécutions &mdash; M=3/4 objectifs</div>
    </div>
    <svg class="blink-arrow" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="9 18 15 12 9 6"/></svg>
  </a>
  </div>

  <footer class="page-footer">
    <div class="footer-pills">
      <span class="footer-pill">f1 Coût</span>
      <span class="footer-pill">f2 CO₂</span>
      <span class="footer-pill">f3 Temps</span>
      <span class="footer-pill">f4 BFR</span>
    </div>
  </footer>

</main>

<script>
let n3Runs = 1;
function setN3Runs(n) {
  n3Runs = n;
  document.querySelectorAll('#n3RunsBtns .runs-btn').forEach(b => {
    b.classList.toggle('selected', parseInt(b.dataset.runs) === n);
  });
  const el = document.getElementById('n3-run');
  if (el) el.href = '{{ url_for("run_nsga3_route") }}?instance=' + selectedInstance + '&runs=' + n;
}

let qi3Runs = 1;
function setQi3Runs(n) {
  qi3Runs = n;
  document.querySelectorAll('#qi3RunsBtns .runs-btn').forEach(b => {
    b.classList.toggle('selected', parseInt(b.dataset.runs) === n);
  });
  const el = document.getElementById('qi3-run');
  if (el) el.href = '{{ url_for("run_qinsga3_route") }}?instance=' + selectedInstance + '&runs=' + n;
}

let moeadRuns = 1;
function setMoeadRuns(n) {
  moeadRuns = n;
  document.querySelectorAll('#moeadRunsBtns .runs-btn').forEach(b => {
    b.classList.toggle('selected', parseInt(b.dataset.runs) === n);
  });
  const el = document.getElementById('moead-run');
  if (el) el.href = '{{ url_for("run_moead_route") }}?instance=' + selectedInstance + '&runs=' + n;
}

const MOON_SVG = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>';
const SUN_SVG  = '<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>';
const icon  = document.getElementById('themeIcon');
const label = document.getElementById('themeLabel');

let selectedInstance = localStorage.getItem('irp-instance') || '{{ default_inst }}';

function selectInstance(key) {
  selectedInstance = key;
  localStorage.setItem('irp-instance', key);
  document.querySelectorAll('.inst-chip').forEach(b => {
    b.classList.toggle('selected', b.dataset.instance === key);
  });
  const pairs = [
    ['oc-run',    '{{ url_for("run_objective_calibration_route") }}'],
    ['oc-report', '{{ url_for("objective_calibration_report") }}'],
    ['fm-run',    '{{ url_for("run_function_merge_route") }}'],
    ['fm-report', '{{ url_for("function_merge_report") }}'],
    ['n3-report', '{{ url_for("nsga3_report") }}'],
    ['qi3-report','{{ url_for("qinsga3_report") }}'],
    ['moead-report', '{{ url_for("moead_report") }}'],
  ];
  pairs.forEach(([id, base]) => {
    const el = document.getElementById(id);
    if (el) el.href = base + '?instance=' + key;
  });
  const n3run = document.getElementById('n3-run');
  if (n3run) n3run.href = '{{ url_for("run_nsga3_route") }}?instance=' + key + '&runs=' + n3Runs;
  const qi3run = document.getElementById('qi3-run');
  if (qi3run) qi3run.href = '{{ url_for("run_qinsga3_route") }}?instance=' + key + '&runs=' + qi3Runs;
  const moeadrun = document.getElementById('moead-run');
  if (moeadrun) moeadrun.href = '{{ url_for("run_moead_route") }}?instance=' + key + '&runs=' + moeadRuns;
}

(function() {
  const valid = {{ inst_keys | tojson }};
  const saved = localStorage.getItem('irp-instance') || '{{ default_inst }}';
  selectInstance(valid.includes(saved) ? saved : '{{ default_inst }}');
})();

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  if (theme === 'dark') {
    icon.innerHTML = MOON_SVG;
    icon.setAttribute('fill', 'currentColor');
    icon.removeAttribute('stroke');
    label.textContent = 'Sombre';
  } else {
    icon.innerHTML = SUN_SVG;
    icon.setAttribute('stroke', 'currentColor');
    icon.setAttribute('fill', 'none');
    label.textContent = 'Clair';
  }
  localStorage.setItem('irp-theme', theme);
}
function toggleTheme() {
  applyTheme(document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
}
(function() {
  const saved = localStorage.getItem('irp-theme');
  applyTheme(saved || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));
})();
</script>
</body>
</html>"""


@app.route("/")
def menu():
    inst_keys   = sorted(INSTANCES.keys(), key=int)
    default_inst = DEFAULT_INSTANCE or (inst_keys[0] if inst_keys else "15")
    return render_template_string(MENU_TEMPLATE, inst_keys=inst_keys, default_inst=default_inst)


@app.route("/benchmarking")
def benchmarking():
    data = _build_benchmark_data()
    data_json = json.dumps(data, separators=(",", ":"))
    return render_template_string(BENCHMARK_TEMPLATE, data_json=data_json), 200, {"Content-Type": "text/html; charset=utf-8"}


# ── Job status API ─────────────────────────────────────────────────────────────

@app.route("/job/api")
def job_api():
    """JSON polling endpoint — returns {status, redirect?, error?}."""
    job_id = request.args.get("job", "")
    with _jobs_lock:
        job = _jobs.get(job_id, {}).copy()
    if not job:
        return jsonify({"status": "not_found"})
    return jsonify({
        "status":   job["status"],
        "redirect": job.get("redirect"),
    })


@app.route("/job/error")
def job_error():
    job_id = request.args.get("job", "")
    with _jobs_lock:
        job = _jobs.get(job_id, {}).copy()
    err = job.get("error") or "Erreur inconnue."
    return render_error(err), 500


@app.route("/objective-calibration/run")
def run_objective_calibration_route():
    data_path, inst_key = _resolve_instance()
    report_path = _instance_report_path(DEFAULT_REPORT_PATH, inst_key)
    job_id = _new_job("Calibration des objectifs")

    def _run():
        lock = _get_run_lock("objective-calibration", inst_key)
        if not lock.acquire(blocking=False):
            _job_error(job_id, f"Une exécution de Calibration des objectifs pour l'instance {inst_key} est déjà en cours -- attendez qu'elle se termine.")
            return
        try:
            run_objective_calibration(output_path=report_path, data_path=data_path)
            _job_done(job_id, f"/objective-calibration/report?instance={inst_key}")
        except Exception:
            tb = traceback.format_exc()
            print(tb, flush=True)
            _job_error(job_id, tb)
        finally:
            lock.release()

    threading.Thread(target=_run, daemon=True).start()
    return _JOB_PAGE.format(algo="Calibration des objectifs", job_id=job_id), 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/objective-calibration/report")
def objective_calibration_report():
    data_path, inst_key = _resolve_instance()
    report_path = _instance_report_path(DEFAULT_REPORT_PATH, inst_key)
    try:
        if not os.path.exists(report_path):
            run_objective_calibration(output_path=report_path, data_path=data_path)
        return send_file(report_path)
    except Exception:
        return render_error(traceback.format_exc()), 500


@app.route("/function-merge/run")
def run_function_merge_route():
    data_path, inst_key = _resolve_instance()
    report_path = _instance_report_path(FM_REPORT_PATH, inst_key)
    job_id = _new_job("Fusion des fonctions")

    def _run():
        lock = _get_run_lock("function-merge", inst_key)
        if not lock.acquire(blocking=False):
            _job_error(job_id, f"Une exécution de Fusion des fonctions pour l'instance {inst_key} est déjà en cours -- attendez qu'elle se termine.")
            return
        try:
            result = run_function_merge(output_path=report_path, data_path=data_path)
            if result is None:
                _job_error(job_id,
                    f"No feasible solution found for instance {inst_key} within the time limit.\n"
                    "Try increasing timelimit or mipgap in Solvers/FunctionMerge/main.py.")
            else:
                _job_done(job_id, f"/function-merge/report?instance={inst_key}")
        except Exception:
            tb = traceback.format_exc()
            print(tb, flush=True)
            _job_error(job_id, tb)
        finally:
            lock.release()

    threading.Thread(target=_run, daemon=True).start()
    return _JOB_PAGE.format(algo="Fusion des fonctions", job_id=job_id), 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/function-merge/report")
def function_merge_report():
    data_path, inst_key = _resolve_instance()
    report_path = _instance_report_path(FM_REPORT_PATH, inst_key)
    try:
        if not os.path.exists(report_path):
            run_function_merge(output_path=report_path, data_path=data_path)
        return send_file(report_path)
    except Exception:
        return render_error(traceback.format_exc()), 500


@app.route("/nsga3/run")
def run_nsga3_route():
    data_path, inst_key = _resolve_instance()
    n_runs = max(1, min(20, int(request.args.get("runs", 1))))
    job_id = _new_job("NSGA-III")

    def _run():
        lock = _get_run_lock("nsga3", inst_key)
        if not lock.acquire(blocking=False):
            _job_error(job_id, f"Une exécution de NSGA-III pour l'instance {inst_key} est déjà en cours -- attendez qu'elle se termine.")
            return
        try:
            run_nsga3_report(output_path=NSGA3_REPORT_PATH, data_path=data_path, n_runs=n_runs)
            _job_done(job_id, f"/nsga3/report?instance={inst_key}")
        except Exception:
            tb = traceback.format_exc()
            print(tb, flush=True)
            _job_error(job_id, tb)
        finally:
            lock.release()

    threading.Thread(target=_run, daemon=True).start()
    return _JOB_PAGE.format(algo="NSGA-III", job_id=job_id), 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/nsga3/report")
def nsga3_report():
    data_path, _ = _resolve_instance()
    try:
        try:
            data = nsga3_render_from_instance(data_path)
        except FileNotFoundError:
            run_nsga3_report(output_path=NSGA3_REPORT_PATH, data_path=data_path)
            data = nsga3_render_from_instance(data_path)
        return nsga3_render_html(data), 200, {"Content-Type": "text/html; charset=utf-8"}
    except Exception:
        return render_error(traceback.format_exc()), 500


@app.route("/qinsga3/run")
def run_qinsga3_route():
    data_path, inst_key = _resolve_instance()
    n_runs = max(1, min(20, int(request.args.get("runs", 1))))
    job_id = _new_job("QI-NSGA-III")

    def _run():
        lock = _get_run_lock("qinsga3", inst_key)
        if not lock.acquire(blocking=False):
            _job_error(job_id, f"Une exécution de QI-NSGA-III pour l'instance {inst_key} est déjà en cours -- attendez qu'elle se termine.")
            return
        try:
            run_qinsga3_report(output_path=QINSGA3_REPORT_PATH, data_path=data_path, n_runs=n_runs)
            _job_done(job_id, f"/qinsga3/report?instance={inst_key}")
        except Exception:
            tb = traceback.format_exc()
            print(tb, flush=True)
            _job_error(job_id, tb)
        finally:
            lock.release()

    threading.Thread(target=_run, daemon=True).start()
    return _JOB_PAGE.format(algo="QI-NSGA-III", job_id=job_id), 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/qinsga3/report")
def qinsga3_report():
    data_path, _ = _resolve_instance()
    try:
        try:
            data = qinsga3_render_from_instance(data_path)
        except FileNotFoundError:
            run_qinsga3_report(output_path=QINSGA3_REPORT_PATH, data_path=data_path)
            data = qinsga3_render_from_instance(data_path)
        return nsga3_render_html(data, algo_label="QI-NSGA-III"), 200, {"Content-Type": "text/html; charset=utf-8"}
    except Exception:
        return render_error(traceback.format_exc()), 500


@app.route("/moead/run")
def run_moead_route():
    data_path, inst_key = _resolve_instance()
    n_runs = max(1, min(20, int(request.args.get("runs", 1))))
    job_id = _new_job("MOEA/D")

    def _run():
        lock = _get_run_lock("moead", inst_key)
        if not lock.acquire(blocking=False):
            _job_error(job_id, f"Une exécution de MOEA/D pour l'instance {inst_key} est déjà en cours -- attendez qu'elle se termine.")
            return
        try:
            run_moead_report(output_path=MOEAD_REPORT_PATH, data_path=data_path, n_runs=n_runs)
            _job_done(job_id, f"/moead/report?instance={inst_key}")
        except Exception:
            tb = traceback.format_exc()
            print(tb, flush=True)
            _job_error(job_id, tb)
        finally:
            lock.release()

    threading.Thread(target=_run, daemon=True).start()
    return _JOB_PAGE.format(algo="MOEA/D", job_id=job_id), 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/moead/report")
def moead_report():
    data_path, _ = _resolve_instance()
    try:
        try:
            data = moead_render_from_instance(data_path)
        except FileNotFoundError:
            run_moead_report(output_path=MOEAD_REPORT_PATH, data_path=data_path)
            data = moead_render_from_instance(data_path)
        return nsga3_render_html(data, algo_label="MOEA/D"), 200, {"Content-Type": "text/html; charset=utf-8"}
    except Exception:
        return render_error(traceback.format_exc()), 500


def render_error(details):
    details = escape(details)
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Erreur du module</title>
<style>
body {{
  margin: 0;
  padding: 28px;
  font-family: Arial, Helvetica, sans-serif;
  background: #f6f7f9;
  color: #1d2733;
}}
.box {{
  max-width: 1100px;
  margin: 0 auto;
  border: 1px solid #d9e0e8;
  border-radius: 8px;
  background: #fff;
  padding: 20px;
}}
h1 {{ margin: 0 0 8px; font-size: 22px; }}
p {{ color: #667485; }}
pre {{
  overflow: auto;
  white-space: pre-wrap;
  border-radius: 6px;
  background: #111318;
  color: #f2f4f8;
  padding: 14px;
}}
a {{ color: #256f83; font-weight: 700; }}
</style>
</head>
<body>
  <main class="box">
    <h1>Erreur du module</h1>
    <p>La route du solveur a levé cette erreur Python. Copiez la trace ci-dessous si vous avez besoin d'aide pour la déboguer.</p>
    <pre>{details}</pre>
    <p><a href="{url_for('menu')}">Retour au menu</a></p>
  </main>
</body>
</html>"""


def _find_port(start_port=5000):
    for port in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("No free local port found.")


def _open_browser_when_ready(url, port):
    browser_url = f"{url}?session={int(time.time())}"
    for _ in range(50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            try:
                sock.connect(("127.0.0.1", port))
            except OSError:
                time.sleep(0.1)
                continue
        webbrowser.open_new_tab(browser_url)
        return
    print(f"Open this URL in your browser: {url}", flush=True)


def main():
    port = _find_port()
    url = f"http://127.0.0.1:{port}/"
    print(f"IRP menu: {url}", flush=True)
    threading.Thread(
        target=_open_browser_when_ready,
        args=(url, port),
        daemon=True,
    ).start()
    # threaded=True: report routes can fall back to a synchronous solve
    # (objective_calibration_report, function_merge_report, nsga3_report,
    # qinsga3_report) when no cached report exists yet -- without this, the
    # dev server's default single-threaded request handling means that solve
    # blocks every other request, including /job/api polling for an
    # unrelated already-running background job.
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
