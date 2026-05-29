"""Local web menu for the IRP project modules."""

import os
import socket
import threading
import traceback
import webbrowser
from html import escape

from flask import Flask, redirect, render_template_string, send_file, url_for

from ObjectiveCalibration.main import DEFAULT_REPORT_PATH, run_objective_calibration


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)


MENU_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IRP Modules</title>
<style>
:root {
  --bg: #f6f7f9;
  --surface: #ffffff;
  --surface-soft: #eef3f8;
  --border: #d9e0e8;
  --text: #1d2733;
  --muted: #667485;
  --accent: #256f83;
  --accent-strong: #164f5f;
  --disabled: #9aa6b3;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  font-family: Arial, Helvetica, sans-serif;
  background: var(--bg);
  color: var(--text);
}
.page {
  max-width: 1040px;
  margin: 0 auto;
  padding: 36px 22px;
}
.topbar {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 20px;
  margin-bottom: 28px;
}
h1 {
  margin: 0 0 6px;
  font-size: 30px;
  line-height: 1.15;
  font-weight: 700;
}
.subtitle {
  margin: 0;
  color: var(--muted);
  font-size: 14px;
}
.badge {
  border: 1px solid var(--border);
  background: var(--surface);
  border-radius: 8px;
  padding: 8px 11px;
  color: var(--muted);
  font-size: 12px;
  white-space: nowrap;
}
.modules {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
}
.module {
  min-height: 205px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  gap: 18px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--surface);
  padding: 18px;
}
.module.available {
  background: linear-gradient(180deg, #ffffff 0%, var(--surface-soft) 100%);
}
.label {
  margin: 0 0 8px;
  font-size: 18px;
  line-height: 1.25;
  font-weight: 700;
}
.description {
  margin: 0;
  color: var(--muted);
  font-size: 13px;
  line-height: 1.45;
}
.actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
.button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 38px;
  padding: 9px 13px;
  border-radius: 7px;
  border: 1px solid transparent;
  background: var(--accent);
  color: #fff;
  text-decoration: none;
  font-weight: 700;
  font-size: 13px;
}
.button:hover { background: var(--accent-strong); }
.button.secondary {
  background: var(--surface);
  border-color: var(--border);
  color: var(--text);
}
.button.secondary:hover { background: var(--surface-soft); }
.button.disabled {
  pointer-events: none;
  background: #edf0f3;
  color: var(--disabled);
  border-color: var(--border);
}
@media (max-width: 820px) {
  .modules { grid-template-columns: 1fr; }
  .topbar { flex-direction: column; }
  .badge { white-space: normal; }
}
</style>
</head>
<body>
  <main class="page">
    <header class="topbar">
      <div>
        <h1>IRP Project Modules</h1>
        <p class="subtitle">Choose a module to run it as a separate app.</p>
      </div>
      <div class="badge">Shared: models, helpers, data</div>
    </header>

    <section class="modules" aria-label="Project modules">
      <article class="module available">
        <div>
          <h2 class="label">ObjectiveCalibration</h2>
          <p class="description">
            Run the many-objective calibration model and open the generated dashboard report.
          </p>
        </div>
        <div class="actions">
          <a class="button" href="{{ url_for('run_objective_calibration_route') }}">Run module</a>
          <a class="button secondary" href="{{ url_for('objective_calibration_report') }}">Last report</a>
        </div>
      </article>

      <article class="module">
        <div>
          <h2 class="label">Module 2</h2>
          <p class="description">Reserved for the next IRP workflow.</p>
        </div>
        <div class="actions">
          <span class="button disabled">Coming soon</span>
        </div>
      </article>

      <article class="module">
        <div>
          <h2 class="label">Module 3</h2>
          <p class="description">Reserved for another workflow using the shared project model.</p>
        </div>
        <div class="actions">
          <span class="button disabled">Coming soon</span>
        </div>
      </article>
    </section>
  </main>
</body>
</html>"""


@app.route("/")
def menu():
    return render_template_string(MENU_TEMPLATE)


@app.route("/objective-calibration/run")
def run_objective_calibration_route():
    try:
        run_objective_calibration()
        return redirect(url_for("objective_calibration_report"))
    except Exception:
        return render_error(traceback.format_exc()), 500


@app.route("/objective-calibration/report")
def objective_calibration_report():
    try:
        if not os.path.exists(DEFAULT_REPORT_PATH):
            run_objective_calibration()
        return send_file(DEFAULT_REPORT_PATH)
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


def main():
    port = _find_port()
    url = f"http://127.0.0.1:{port}/"
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"IRP menu: {url}")
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
