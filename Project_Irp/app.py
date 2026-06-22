"""Local web menu for the IRP project modules."""

import os
import socket
import threading
import traceback
import webbrowser
import time
from html import escape

from flask import Flask, redirect, render_template_string, request, send_file, url_for

from ObjectiveCalibration.main import DEFAULT_REPORT_PATH, run_objective_calibration
from FunctionMerge.main import DEFAULT_REPORT_PATH as FM_REPORT_PATH, run_function_merge
from NSGA3.main    import DEFAULT_REPORT_PATH as NSGA3_REPORT_PATH, run_nsga3_report, render_from_instance as nsga3_render_from_instance
from NSGA3.report  import render_html as nsga3_render_html
from QINSGA3.main  import DEFAULT_REPORT_PATH as QINSGA3_REPORT_PATH, run_qinsga3_report, render_from_instance as qinsga3_render_from_instance


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INSTANCES = {
    "3":  os.path.join(BASE_DIR, "data", "instance_3_clients.json"),
    "5":  os.path.join(BASE_DIR, "data", "instance_5_clients.json"),
    "15": os.path.join(BASE_DIR, "data", "instance_15_clients.json"),
    "25": os.path.join(BASE_DIR, "data", "instance_25_clients.json"),
    "30": os.path.join(BASE_DIR, "data", "instance_30_clients.json"),
    "40": os.path.join(BASE_DIR, "data", "instance_40_clients.json"),
}
DEFAULT_INSTANCE = "15"


def _resolve_instance():
    key = request.args.get("instance", DEFAULT_INSTANCE)
    return INSTANCES.get(key, INSTANCES[DEFAULT_INSTANCE]), key

app = Flask(__name__)


MENU_TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IRP — Solver Suite</title>
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
  padding: 20px 20px 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
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
  justify-content: space-between;
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
        <div class="brand-name">IRP Solver Suite</div>
        <div class="brand-sub">Inventory Routing Problem — multi-objective optimisation</div>
      </div>
    </div>
    <div class="header-right">
      <button class="btn-theme" id="themeToggle" onclick="toggleTheme()" aria-label="Toggle theme">
        <svg id="themeIcon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="5"/>
          <line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/>
          <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
          <line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/>
          <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
        </svg>
        <span id="themeLabel">Light</span>
      </button>
    </div>
  </header>

  <div class="panel">
    <div class="panel-hdr">
      <div class="panel-icon">
        <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M12 2v3m0 14v3M2 12h3m14 0h3M4.93 4.93l2.12 2.12m9.9 9.9 2.12 2.12M4.93 19.07l2.12-2.12m9.9-9.9 2.12-2.12"/></svg>
      </div>
      <span class="panel-title">Instance</span>
    </div>
    <div class="inst-chips" id="instanceBar">
      <button class="inst-chip" data-instance="3"  onclick="selectInstance('3')"><span class="chip-dot"></span>3 clients</button>
      <button class="inst-chip" data-instance="5"  onclick="selectInstance('5')"><span class="chip-dot"></span>5 clients</button>
      <button class="inst-chip selected" data-instance="15" onclick="selectInstance('15')"><span class="chip-dot"></span>15 clients</button>
      <button class="inst-chip" data-instance="25" onclick="selectInstance('25')"><span class="chip-dot"></span>25 clients</button>
      <button class="inst-chip" data-instance="30" onclick="selectInstance('30')"><span class="chip-dot"></span>30 clients</button>
      <button class="inst-chip" data-instance="40" onclick="selectInstance('40')"><span class="chip-dot"></span>40 clients</button>
    </div>
  </div>

  <div class="section-hdr">
    <span class="section-hdr-label">Available modules</span>
    <div class="section-hdr-line"></div>
    <span class="section-hdr-label">4 active</span>
  </div>

  <section class="modules" aria-label="Solver modules">

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
            <span class="badge-status active">Active</span>
          </div>
        </div>
        <h2 class="card-title">Objective Calibration</h2>
        <p class="card-desc">Calibrate and solve the many-objective MIP individually — derives budget bounds C_max, E_max, T_max, B for all four objectives.</p>
      </div>
      <div class="card-footer">
        <a class="btn btn-primary" id="oc-run" href="{{ url_for('run_objective_calibration_route') }}?instance=15" target="_blank" rel="noopener noreferrer">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
          Run
        </a>
        <a class="btn btn-secondary" id="oc-report" href="{{ url_for('objective_calibration_report') }}?instance=15" target="_blank" rel="noopener noreferrer">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
          Last report
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
            <span class="badge-status active">Active</span>
          </div>
        </div>
        <h2 class="card-title">Function Merge</h2>
        <p class="card-desc">Scalarised single-run CPLEX solve — minimises f1 + f2 + f3 + f4 simultaneously across cost, CO₂, time and working capital.</p>
      </div>
      <div class="card-footer">
        <a class="btn btn-primary" id="fm-run" href="{{ url_for('run_function_merge_route') }}?instance=15" target="_blank" rel="noopener noreferrer">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
          Run
        </a>
        <a class="btn btn-secondary" id="fm-report" href="{{ url_for('function_merge_report') }}?instance=15" target="_blank" rel="noopener noreferrer">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
          Last report
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
            <span class="badge-status active">Active</span>
          </div>
        </div>
        <h2 class="card-title">NSGA-III</h2>
        <p class="card-desc">Many-objective genetic algorithm. Builds a dense Pareto front across all four IRP objectives.</p>
        <div class="card-tags">
          <span class="card-tag">Pareto front</span>
          <span class="card-tag">Ref. directions</span>
          <span class="card-tag">4 objectives</span>

        </div>
        <div class="runs-row">
          <span class="runs-label">Runs</span>
          <div class="runs-group" id="n3RunsBtns">
            <button class="runs-btn selected" data-runs="1"  onclick="setN3Runs(1)">1×</button>
            <button class="runs-btn"          data-runs="3"  onclick="setN3Runs(3)">3×</button>
            <button class="runs-btn"          data-runs="5"  onclick="setN3Runs(5)">5×</button>
            <button class="runs-btn"          data-runs="10" onclick="setN3Runs(10)">10×</button>
            <button class="runs-btn"          data-runs="20" onclick="setN3Runs(20)">20×</button>
          </div>
        </div>
      </div>
      <div class="card-footer">
        <a class="btn btn-primary" id="n3-run" href="{{ url_for('run_nsga3_route') }}?instance=25&runs=1" target="_blank" rel="noopener noreferrer">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
          Run
        </a>
        <a class="btn btn-secondary" id="n3-report" href="{{ url_for('nsga3_report') }}?instance=25" target="_blank" rel="noopener noreferrer">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
          Last report
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
            <span class="badge-status active">Active</span>
          </div>
        </div>
        <h2 class="card-title">QI-NSGA-III</h2>
        <p class="card-desc">Quantum-inspired variant of NSGA-III. Chromosomes are encoded as rotation angles θ; convergence is driven by adaptive rotation gates instead of crossover </p>
        <div class="card-tags">
          <span class="card-tag">Quantum encoding</span>
          <span class="card-tag">Rotation gate θ</span>
        </div>
        <div class="runs-row">
          <span class="runs-label">Runs</span>
          <div class="runs-group" id="qi3RunsBtns">
            <button class="runs-btn selected" data-runs="1"  onclick="setQi3Runs(1)">1×</button>
            <button class="runs-btn"          data-runs="3"  onclick="setQi3Runs(3)">3×</button>
            <button class="runs-btn"          data-runs="5"  onclick="setQi3Runs(5)">5×</button>
            <button class="runs-btn"          data-runs="10" onclick="setQi3Runs(10)">10×</button>
            <button class="runs-btn"          data-runs="20" onclick="setQi3Runs(20)">20×</button>
          </div>
        </div>
      </div>
      <div class="card-footer">
        <a class="btn btn-primary" id="qi3-run" href="{{ url_for('run_qinsga3_route') }}?instance=25&runs=1" target="_blank" rel="noopener noreferrer">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
          Run
        </a>
        <a class="btn btn-secondary" id="qi3-report" href="{{ url_for('qinsga3_report') }}?instance=25" target="_blank" rel="noopener noreferrer">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
          Last report
        </a>
      </div>
    </article>

  </section>

  <footer class="page-footer">
    <span class="footer-text">IRP Solver Suite &mdash; Many-objective Inventory Routing</span>
    <div class="footer-pills">
      <span class="footer-pill">f1 Cost</span>
      <span class="footer-pill">f2 CO₂</span>
      <span class="footer-pill">f3 Time</span>
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

const MOON_SVG = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>';
const SUN_SVG  = '<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>';
const icon  = document.getElementById('themeIcon');
const label = document.getElementById('themeLabel');

let selectedInstance = localStorage.getItem('irp-instance') || '15';

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
  ];
  pairs.forEach(([id, base]) => {
    const el = document.getElementById(id);
    if (el) el.href = base + '?instance=' + key;
  });
  const n3run = document.getElementById('n3-run');
  if (n3run) n3run.href = '{{ url_for("run_nsga3_route") }}?instance=' + key + '&runs=' + n3Runs;
  const qi3run = document.getElementById('qi3-run');
  if (qi3run) qi3run.href = '{{ url_for("run_qinsga3_route") }}?instance=' + key + '&runs=' + qi3Runs;
}

(function() {
  const valid = ['3','5','15','25','30','40'];
  const saved = localStorage.getItem('irp-instance') || '15';
  selectInstance(valid.includes(saved) ? saved : '15');
})();

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  if (theme === 'dark') {
    icon.innerHTML = MOON_SVG;
    icon.setAttribute('fill', 'currentColor');
    icon.removeAttribute('stroke');
    label.textContent = 'Dark';
  } else {
    icon.innerHTML = SUN_SVG;
    icon.setAttribute('stroke', 'currentColor');
    icon.setAttribute('fill', 'none');
    label.textContent = 'Light';
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
    return render_template_string(MENU_TEMPLATE)


@app.route("/objective-calibration/run")
def run_objective_calibration_route():
    data_path, inst_key = _resolve_instance()
    try:
        run_objective_calibration(data_path=data_path)
        return redirect(url_for("objective_calibration_report") + f"?instance={inst_key}")
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        return render_error(tb), 500


@app.route("/objective-calibration/report")
def objective_calibration_report():
    data_path, inst_key = _resolve_instance()
    try:
        if not os.path.exists(DEFAULT_REPORT_PATH):
            run_objective_calibration(data_path=data_path)
        return send_file(DEFAULT_REPORT_PATH)
    except Exception:
        return render_error(traceback.format_exc()), 500


@app.route("/function-merge/run")
def run_function_merge_route():
    data_path, inst_key = _resolve_instance()
    try:
        result = run_function_merge(data_path=data_path)
        if result is None:
            return render_error(
                f"No feasible solution found for instance {inst_key} within the time limit.\n"
                "Try increasing timelimit or mipgap in FunctionMerge/main.py."
            ), 500
        return redirect(url_for("function_merge_report") + f"?instance={inst_key}")
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        return render_error(tb), 500


@app.route("/function-merge/report")
def function_merge_report():
    data_path, inst_key = _resolve_instance()
    try:
        if not os.path.exists(FM_REPORT_PATH):
            run_function_merge(data_path=data_path)
        return send_file(FM_REPORT_PATH)
    except Exception:
        return render_error(traceback.format_exc()), 500


@app.route("/nsga3/run")
def run_nsga3_route():
    data_path, inst_key = _resolve_instance()
    try:
        n_runs = max(1, min(20, int(request.args.get("runs", 1))))
        run_nsga3_report(output_path=NSGA3_REPORT_PATH, data_path=data_path, n_runs=n_runs)
        return redirect(url_for("nsga3_report") + f"?instance={inst_key}")
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        return render_error(tb), 500


@app.route("/nsga3/report")
def nsga3_report():
    data_path, inst_key = _resolve_instance()
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
    try:
        n_runs = max(1, min(20, int(request.args.get("runs", 1))))
        run_qinsga3_report(output_path=QINSGA3_REPORT_PATH, data_path=data_path, n_runs=n_runs)
        return redirect(url_for("qinsga3_report") + f"?instance={inst_key}")
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        return render_error(tb), 500


@app.route("/qinsga3/report")
def qinsga3_report():
    data_path, inst_key = _resolve_instance()
    try:
        try:
            data = qinsga3_render_from_instance(data_path)
        except FileNotFoundError:
            run_qinsga3_report(output_path=QINSGA3_REPORT_PATH, data_path=data_path)
            data = qinsga3_render_from_instance(data_path)
        return nsga3_render_html(data), 200, {"Content-Type": "text/html; charset=utf-8"}
    except Exception:
        return render_error(traceback.format_exc()), 500


def render_error(details):
    details = escape(details)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Module Error</title>
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
    <h1>Module error</h1>
    <p>The solver route raised this Python error. Copy the traceback below if you need help debugging it.</p>
    <pre>{details}</pre>
    <p><a href="{url_for('menu')}">Back to menu</a></p>
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
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
