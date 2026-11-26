"""Export a representative operational plan (answers "aggregate solution-set
indicators, without representative operational plans").

Takes the last cached run of an algorithm, picks the knee solution of its
Pareto front (closest to the ideal point after ideal/nadir normalisation) and
writes a Markdown plan: objectives, supplier orders, depot stock, working
capital per period, and every route with its stops, quantities and arrival times.

Usage:
    python -m scripts.export_plan --algo nsga3 --instance data/instance_100_clients.json
    (algo: nsga3 | qinsga3 | moead; the algorithm must have been run once)
Writes results/plan_<algo>_<instance>.md
"""
import argparse
import importlib
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from models.parameters import load_instance                      # noqa: E402
from solvers.nsga3.evaluator import route_arrivals                # noqa: E402


def knee(solutions):
    F = np.array([[s["objectives"][k] for k in ("f1", "f2", "f3", "f4")] for s in solutions])
    lo, hi = F.min(axis=0), F.max(axis=0)
    norm = (F - lo) / np.where(hi - lo > 1e-12, hi - lo, 1.0)
    return solutions[int(np.argmin(np.linalg.norm(norm, axis=1)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=["nsga3", "qinsga3", "moead"])
    ap.add_argument("--instance", required=True)
    args = ap.parse_args()

    render = importlib.import_module(f"solvers.{args.algo}.main").render_from_instance
    data = render(args.instance)
    run = data["runs"][-1] if "runs" in data else data
    sol = knee(run["solutions"])
    sets_, params_ = load_instance(args.instance)
    T = sets_["T"]

    o = sol["objectives"]
    lines = [f"# Representative plan — {args.algo.upper()} — {os.path.basename(args.instance)}", "",
             "Knee solution of the Pareto front (closest to the ideal point after normalisation).", "",
             "| f1 cost (€) | f2 CO₂ (kg) | f3 longest routes (h) | f4 peak working capital (€) |",
             "|---|---|---|---|", f"| {o['f1']} | {o['f2']} | {o['f3']} | {o['f4']} |", ""]
    if sol.get("f1_sub"):
        lines += ["f1 breakdown: " + ", ".join(f"{k} {v}" for k, v in sol["f1_sub"].items()), ""]
    wc = sol.get("bfr_sub", {}).get("series", {})
    lines += ["## Supplier orders, depot stock, working capital", "",
              "| period | order cold | order ambient | stock cold | stock ambient | shipped | WC (€) |",
              "|---|---|---|---|---|---|---|"]
    for t in T:
        r = sol["routes"][str(t)]
        st = sol["depot_stock"][str(t)]
        lines.append(f"| {t} | {r['R_frigo']:.0f} | {r['R_nonfrigo']:.0f} | {st['frigo']:.0f} | "
                     f"{st['nonfrigo']:.0f} | {r['shipped']} | {wc.get(str(t), '')} |")
    lines += ["", "## Routes", ""]
    for t in T:
        lines += [f"### Period {t}", "", "| vehicle | class | stops (client: units @ arrival h) | load | return (h) |",
                  "|---|---|---|---|---|"]
        for truck in sol["routes"][str(t)]["trucks"]:
            k, path, qty = truck["k"], truck["path"], truck["qty"]
            arr = dict(route_arrivals(path, k, params_))
            stops = ", ".join(f"{l}: {qty[str(l)]} @ {arr[l]:.2f}" for l in path[1:-1])
            cls = "cold" if k in params_["frigo_trucks"] else "ambient"
            dur = sum(params_["s"][path[i]] + params_["d"][path[i], path[i + 1]] / params_["v"][k]
                      for i in range(len(path) - 1))
            lines.append(f"| {k} | {cls} | {stops} | {sum(qty.values())} | {dur:.2f} |")
        lines.append("")

    out = os.path.join(BASE_DIR, "results")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"plan_{args.algo}_{os.path.splitext(os.path.basename(args.instance))[0]}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"written {os.path.relpath(path, BASE_DIR)}")


if __name__ == "__main__":
    main()
