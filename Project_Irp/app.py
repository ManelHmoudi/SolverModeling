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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INSTANCES = {
    "3":  os.path.join(BASE_DIR, "data", "instance_3_clients.json"),
    "5":  os.path.join(BASE_DIR, "data", "instance_5_clients.json"),
    "15": os.path.join(BASE_DIR, "data", "instance_15_clients.json"),
    "25": os.path.join(BASE_DIR, "data", "instance_25_clients.json"),
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
<title>IRP — Module Launcher</title>
<style>
:root {
  --bg: #f0f4f8;
  --surface: #ffffff;
  --surface-hover: #f5f8fb;
  --border: #cdd7e3;
  --border-active: #a8bdd1;
  --text: #0d1b2a;
  --muted: #546475;
  --accent: #1a6f8a;
  --accent-hover: #145c74;
  --accent-fg: #ffffff;
  --tag-active-bg: #dff3f0;
  --tag-active-fg: #0e6b5e;
  --tag-soon-bg: #f0f1f3;
  --tag-soon-fg: #6b7585;
  --shadow: 0 1px 3px rgba(0,0,0,.07), 0 4px 12px rgba(0,0,0,.05);
  --shadow-hover: 0 2px 8px rgba(0,0,0,.09), 0 8px 24px rgba(0,0,0,.08);
  --num-bg: #eaf3f7;
  --num-fg: #1a6f8a;
  --btn-sec-bg: #ffffff;
  --btn-sec-border: #cdd7e3;
  --btn-sec-fg: #0d1b2a;
  --btn-sec-hover: #eef3f7;
  --btn-dis-bg: #eceff2;
  --btn-dis-fg: #9daab6;
  --toggle-bg: #dde5ee;
  --toggle-fg: #546475;
}
[data-theme="dark"] {
  --bg: #0e1118;
  --surface: #161c27;
  --surface-hover: #1c2436;
  --border: #252e42;
  --border-active: #334360;
  --text: #dde4ef;
  --muted: #8493ab;
  --accent: #4ab0cc;
  --accent-hover: #38a0be;
  --accent-fg: #0a1520;
  --tag-active-bg: #0d2e2a;
  --tag-active-fg: #4ecbb8;
  --tag-soon-bg: #1c2236;
  --tag-soon-fg: #6b7c9a;
  --shadow: 0 1px 3px rgba(0,0,0,.3), 0 4px 12px rgba(0,0,0,.25);
  --shadow-hover: 0 2px 8px rgba(0,0,0,.35), 0 8px 24px rgba(0,0,0,.3);
  --num-bg: #0e2535;
  --num-fg: #4ab0cc;
  --btn-sec-bg: #1c2436;
  --btn-sec-border: #2d3a55;
  --btn-sec-fg: #dde4ef;
  --btn-sec-hover: #232d44;
  --btn-dis-bg: #181e2e;
  --btn-dis-fg: #3e4e6a;
  --toggle-bg: #252e42;
  --toggle-fg: #8493ab;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  min-height: 100vh;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  background: var(--bg);
  color: var(--text);
  transition: background .25s, color .25s;
  -webkit-font-smoothing: antialiased;
}
.page {
  max-width: 1080px;
  margin: 0 auto;
  padding: 48px 24px 64px;
}

/* ── Header ─────────────────────────────────────────────── */
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  margin-bottom: 48px;
}
.header-brand { display: flex; align-items: center; gap: 14px; }
.brand-mark {
  width: 44px; height: 44px;
  border-radius: 10px;
  background: var(--accent);
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}
.brand-mark svg { width: 22px; height: 22px; fill: var(--accent-fg); }
.brand-text h1 {
  font-size: 22px;
  font-weight: 700;
  letter-spacing: -.4px;
  line-height: 1.2;
  color: var(--text);
}
.brand-text p {
  font-size: 13px;
  color: var(--muted);
  margin-top: 2px;
}
.theme-toggle {
  display: flex; align-items: center; gap: 6px;
  padding: 7px 12px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--toggle-bg);
  color: var(--toggle-fg);
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: background .2s, border-color .2s, color .2s;
  outline: none;
  white-space: nowrap;
}
.theme-toggle:hover { border-color: var(--border-active); }
.theme-toggle svg { width: 15px; height: 15px; flex-shrink: 0; }

/* ── Instance bar ────────────────────────────────────────── */
.instance-bar {
  display: flex;
  align-items: center;
  gap: 10px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 10px 16px;
  margin-bottom: 28px;
  flex-wrap: wrap;
}
.instance-bar-label {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: .07em;
  text-transform: uppercase;
  color: var(--muted);
  white-space: nowrap;
  margin-right: 4px;
}
.inst-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 14px;
  border-radius: 7px;
  border: 1px solid var(--border);
  background: transparent;
  color: var(--muted);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: background .15s, border-color .15s, color .15s;
  white-space: nowrap;
}
.inst-btn:hover {
  border-color: var(--border-active);
  background: var(--surface-hover);
  color: var(--text);
}
.inst-btn.selected {
  background: var(--accent);
  border-color: var(--accent);
  color: var(--accent-fg);
}
.inst-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: currentColor;
  opacity: .7;
  flex-shrink: 0;
}

/* ── Divider ─────────────────────────────────────────────── */
.section-label {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 14px;
}

/* ── Grid ─────────────────────────────────────────────────── */
.modules {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
}

/* ── Card ─────────────────────────────────────────────────── */
.module {
  display: flex;
  flex-direction: column;
  gap: 0;
  border: 1px solid var(--border);
  border-radius: 12px;
  background: var(--surface);
  box-shadow: var(--shadow);
  overflow: hidden;
  transition: box-shadow .2s, border-color .2s, background .2s;
}
.module.available:hover {
  box-shadow: var(--shadow-hover);
  border-color: var(--border-active);
}
.card-body {
  flex: 1;
  padding: 20px 20px 18px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.card-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.module-num {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: .06em;
  color: var(--num-fg);
  background: var(--num-bg);
  border-radius: 5px;
  padding: 3px 7px;
}
.status-tag {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: .04em;
  border-radius: 20px;
  padding: 3px 9px;
}
.status-tag.active {
  background: var(--tag-active-bg);
  color: var(--tag-active-fg);
}
.status-tag.soon {
  background: var(--tag-soon-bg);
  color: var(--tag-soon-fg);
}
.card-title {
  font-size: 16px;
  font-weight: 700;
  letter-spacing: -.2px;
  color: var(--text);
  line-height: 1.3;
}
.card-desc {
  font-size: 13px;
  color: var(--muted);
  line-height: 1.55;
}
.card-footer {
  padding: 12px 20px;
  border-top: 1px solid var(--border);
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  background: transparent;
}
.button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  height: 36px;
  padding: 0 14px;
  border-radius: 7px;
  border: 1px solid transparent;
  font-size: 13px;
  font-weight: 600;
  text-decoration: none;
  cursor: pointer;
  white-space: nowrap;
  transition: background .15s, border-color .15s, color .15s;
}
.button.primary {
  background: var(--accent);
  color: var(--accent-fg);
}
.button.primary:hover { background: var(--accent-hover); }
.button.secondary {
  background: var(--btn-sec-bg);
  border-color: var(--btn-sec-border);
  color: var(--btn-sec-fg);
}
.button.secondary:hover { background: var(--btn-sec-hover); }
.button.disabled {
  pointer-events: none;
  background: var(--btn-dis-bg);
  color: var(--btn-dis-fg);
  border-color: transparent;
}

@media (max-width: 780px) {
  .modules { grid-template-columns: 1fr; }
  .header { flex-direction: row; }
  .page { padding: 32px 16px 48px; }
}
</style>
</head>
<body>
<main class="page">

  <header class="header">
    <div class="header-brand">
      <div class="brand-mark">
        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path d="M3 3h7v7H3V3zm11 0h7v7h-7V3zm0 11h7v7h-7v-7zM3 14h7v7H3v-7z"/>
        </svg>
      </div>
      <div class="brand-text">
        <h1>IRP Modules</h1>
        <p>Select a module to launch as a standalone app</p>
      </div>
    </div>
    <button class="theme-toggle" id="themeToggle" onclick="toggleTheme()" aria-label="Toggle theme">
      <svg id="themeIcon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="5"/>
        <line x1="12" y1="1" x2="12" y2="3"/>
        <line x1="12" y1="21" x2="12" y2="23"/>
        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
        <line x1="1" y1="12" x2="3" y2="12"/>
        <line x1="21" y1="12" x2="23" y2="12"/>
        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
      </svg>
      <span id="themeLabel">Light</span>
    </button>
  </header>

  <!-- Instance selector -->
  <div class="instance-bar" id="instanceBar">
    <span class="instance-bar-label">Instance</span>
    <button class="inst-btn" data-instance="3"  onclick="selectInstance('3')">
      <span class="inst-dot"></span>3 clients
    </button>
    <button class="inst-btn" data-instance="5"  onclick="selectInstance('5')">
      <span class="inst-dot"></span>5 clients
    </button>
    <button class="inst-btn selected" data-instance="15" onclick="selectInstance('15')">
      <span class="inst-dot"></span>15 clients
    </button>
    <button class="inst-btn" data-instance="25" onclick="selectInstance('25')">
      <span class="inst-dot"></span>25 clients
    </button>
  </div>

  <p class="section-label">Available modules</p>

  <section class="modules" aria-label="Project modules">

    <article class="module available">
      <div class="card-body">
        <div class="card-meta">
          <span class="module-num">01</span>
          <span class="status-tag active">Active</span>
        </div>
        <h2 class="card-title">Objective Calibration</h2>
        <p class="card-desc">Run the many-objective calibration model and open the generated dashboard report.</p>
      </div>
      <div class="card-footer">
        <a class="button primary"    id="oc-run"    href="{{ url_for('run_objective_calibration_route') }}?instance=15">Run module</a>
        <a class="button secondary"  id="oc-report" href="{{ url_for('objective_calibration_report') }}?instance=15">Last report</a>
      </div>
    </article>

    <article class="module available">
      <div class="card-body">
        <div class="card-meta">
          <span class="module-num">02</span>
          <span class="status-tag active">Active</span>
        </div>
        <h2 class="card-title">Function Merge</h2>
        <p class="card-desc">Solve all four objectives together in one run: minimise f1 (cost), f2 (CO₂), f3 (time) and maximise f4 (working capital) via a single scalarised CPLEX solve.</p>
      </div>
      <div class="card-footer">
        <a class="button primary"   id="fm-run"    href="{{ url_for('run_function_merge_route') }}?instance=15">Run module</a>
        <a class="button secondary" id="fm-report" href="{{ url_for('function_merge_report') }}?instance=15">Last report</a>
      </div>
    </article>

    <article class="module">
      <div class="card-body">
        <div class="card-meta">
          <span class="module-num">03</span>
          <span class="status-tag soon">Coming soon</span>
        </div>
        <h2 class="card-title">Module 3</h2>
        <p class="card-desc">Reserved for another workflow using the shared project model.</p>
      </div>
      <div class="card-footer">
        <span class="button disabled">Unavailable</span>
      </div>
    </article>

  </section>
</main>

<script>
const MOON_SVG = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>';
const SUN_SVG = '<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>';
const icon = document.getElementById('themeIcon');
const label = document.getElementById('themeLabel');

/* ── Instance selector ───────────────────────────────────── */
let selectedInstance = localStorage.getItem('irp-instance') || '15';

function selectInstance(key) {
  selectedInstance = key;
  localStorage.setItem('irp-instance', key);

  // Update button styles
  document.querySelectorAll('.inst-btn').forEach(btn => {
    btn.classList.toggle('selected', btn.dataset.instance === key);
  });

  // Update all module button hrefs
  const pairs = [
    ['oc-run',    '{{ url_for("run_objective_calibration_route") }}'],
    ['oc-report', '{{ url_for("objective_calibration_report") }}'],
    ['fm-run',    '{{ url_for("run_function_merge_route") }}'],
    ['fm-report', '{{ url_for("function_merge_report") }}'],
  ];
  pairs.forEach(([id, base]) => {
    const el = document.getElementById(id);
    if (el) el.href = base + '?instance=' + key;
  });
}

// Restore saved selection on load — default to 15 if saved value no longer valid
(function() {
  const valid = ['3','5','15','25'];
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
  const current = document.documentElement.getAttribute('data-theme');
  applyTheme(current === 'dark' ? 'light' : 'dark');
}

(function() {
  const saved = localStorage.getItem('irp-theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  applyTheme(saved || (prefersDark ? 'dark' : 'light'));
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
    <h1>ObjectiveCalibration failed</h1>
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
                threading.Event().wait(0.1)
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
