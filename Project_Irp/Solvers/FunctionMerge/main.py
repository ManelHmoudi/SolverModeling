"""FunctionMerge — Combined multi-objective IRP solve.

Scalarization: minimize a normalized weighted sum of f1..f4,
    F = sum_m w_m * (f_m - f_m_ideal) / (f_m_bound - f_m_ideal),
ideal/bound taken from ObjectiveCalibration's single-objective solves
(bound = ideal + 20% margin, same convention as the C_max/E_max/T_max/B
budgets). This is a calibrated upper bound / tolerance threshold, not a
true nadir point (which would require a payoff table across all
single-objective optima, not just a margin around each ideal).
Raw f1+f2+f3+f4 would add DT + kg CO2 + hours + DT with no dimensional
meaning and would favor whichever objective has the largest raw magnitude.
Called from app.py via run_function_merge().
"""

import json
import os
import sys
import time

from docplex.mp.model import Model
from docplex.mp.progress import SolutionListener

MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(MODULE_DIR))

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from models.parametres import load_instance
from models.constraints import add_all_constraints, add_budget_constraints
from models.objectives  import build_all_objectives
from models.variables   import build_variables

try:
    from .report import compute_node_positions, write_report
except ImportError:
    from report import compute_node_positions, write_report

try:
    from ..ObjectiveCalibration.main import build_report_data as _build_calibration_data
except ImportError:
    from Solvers.ObjectiveCalibration.main import build_report_data as _build_calibration_data


DEFAULT_REPORT_PATH = os.path.join(MODULE_DIR, "irp_function_merge_report.html")
CHECKPOINT_DIR      = os.path.join(MODULE_DIR, "checkpoints")
CHECKPOINT_HISTORY  = os.path.join(CHECKPOINT_DIR, "history.json")
SOLVE_LOG_PATH      = os.path.join(CHECKPOINT_DIR, "solve.log")


class _CheckpointListener(SolutionListener):
    """Saves every new incumbent to its own file in CHECKPOINT_DIR."""

    def __init__(self):
        super().__init__()
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        self._count    = self._load_count()
        self._best_obj = float("inf")

    @staticmethod
    def _load_count():
        if not os.path.exists(CHECKPOINT_HISTORY):
            return 0
        try:
            with open(CHECKPOINT_HISTORY) as fh:
                return len(json.load(fh))
        except Exception:
            return 0

    def notify_solution(self, s):
        obj = s.objective_value
        if obj >= self._best_obj - 1e-8:
            return
        self._best_obj = obj
        self._count   += 1
        ts    = time.strftime("%Y%m%d_%H%M%S")
        fname = f"ckpt_{self._count:04d}_{ts}_obj{obj:.4f}.json"
        fpath = os.path.join(CHECKPOINT_DIR, fname)
        data  = {v.name: val for v, val in s.iter_var_values()
                 if v.is_binary() or v.is_integer()}
        record = {"n": self._count, "timestamp": ts, "objective": obj, "file": fname}
        try:
            with open(fpath, "w") as fh:
                json.dump({"meta": record, "vars": data}, fh)
            history = []
            if os.path.exists(CHECKPOINT_HISTORY):
                try:
                    with open(CHECKPOINT_HISTORY) as fh:
                        history = json.load(fh)
                except Exception:
                    pass
            history.append(record)
            with open(CHECKPOINT_HISTORY, "w") as fh:
                json.dump(history, fh, indent=2)
            print(f"[checkpoint #{self._count}] obj={obj:.4f}  → {fname}", flush=True)
        except OSError as exc:
            print(f"[checkpoint] Save failed: {exc}", flush=True)


def _best_checkpoint_path():
    if not os.path.exists(CHECKPOINT_HISTORY):
        return None
    try:
        with open(CHECKPOINT_HISTORY) as fh:
            history = json.load(fh)
        if not history:
            return None
        best = min(history, key=lambda r: r["objective"])
        return os.path.join(CHECKPOINT_DIR, best["file"])
    except Exception:
        return None


def _get_ordered_path(arcs):
    if not arcs:
        return []
    next_node    = {i: j for (i, j) in arcs}
    destinations = {j for (_, j) in arcs}
    starts       = [i for i in next_node if i not in destinations]
    current      = starts[0] if starts else 0
    path         = [current]
    visited      = {current}
    while current in next_node and next_node[current] not in visited:
        current = next_node[current]
        path.append(current)
        visited.add(current)
    return path


def _build_model(sets_, params_):
    mdl   = Model(name="IRP_FunctionMerge")
    vars_ = build_variables(mdl, sets_["N"], sets_["A"], sets_["T"], sets_["M"], sets_["clients"])
    objectives = build_all_objectives(mdl, vars_, sets_, params_)
    add_all_constraints(mdl, vars_, sets_, params_)
    add_budget_constraints(
        mdl,
        objectives["f1"], objectives["f2"],
        objectives["f3"], objectives["f4"],
        params_["C_max"], params_["E_max"],
        params_["T_max"], params_["B"],
    )
    return mdl, vars_, objectives


def run_combined_solve(data_path=None, weights=None):
    """weights: optional dict {"f1":.., "f2":.., "f3":.., "f4":..} for the
    normalized weighted sum below (defaults to equal weights 0.25 each)."""
    sets_, params_ = load_instance(data_path)

    # Ideal / calibrated-upper-bound values for normalization -- reuses the
    # single-objective solves already performed by ObjectiveCalibration for
    # the C_max/E_max/T_max/B budget thresholds (bound = ideal + 20% margin,
    # same convention as those budgets), instead of re-solving f1..f4
    # individually here. Not a true nadir point (see module docstring).
    calib = _build_calibration_data(data_path)
    ideal = {row["budget_key"]: row["value"] for row in calib["objs"]}
    bound = calib["bounds"]

    mdl, vars_, objectives = _build_model(sets_, params_)

    f1 = objectives["f1"]
    f2 = objectives["f2"]
    f3 = objectives["f3"]
    f4 = objectives["f4"]

    if weights is None:
        weights = {"f1": 0.25, "f2": 0.25, "f3": 0.25, "f4": 0.25}

    _terms = [("f1", f1, "C_max"), ("f2", f2, "E_max"),
              ("f3", f3, "T_max"), ("f4", f4, "B")]

    def _denom(key):
        d = bound[key] - ideal[key]
        return d if abs(d) > 1e-6 else 1.0

    # Normalized weighted sum (Deb-style min-max normalization), replacing
    # the raw f1+f2+f3+f4 sum: additioning DT + kg CO2 + hours + DT with no
    # normalization favored whichever objective had the largest raw
    # magnitude and had no dimensional meaning.
    composite = mdl.sum(
        weights[name] * (expr - ideal[key]) / _denom(key)
        for name, expr, key in _terms
    )
    mdl.minimize(composite)

    mdl.parameters.emphasis.mip              = 1   # feasibility first
    mdl.parameters.mip.strategy.heuristicfreq = 5
    mdl.parameters.mip.strategy.fpheur        = 1
    mdl.parameters.mip.tolerances.mipgap      = 0.05
    mdl.parameters.mip.tolerances.absmipgap   = 0
    mdl.parameters.timelimit                   = 39600
    mdl.parameters.workmem                     = 8192
    mdl.parameters.mip.strategy.file          = 2

    best_ckpt = _best_checkpoint_path()
    if best_ckpt and os.path.exists(best_ckpt):
        try:
            with open(best_ckpt) as fh:
                ckpt = json.load(fh)
            ms = mdl.new_solution()
            for var_name, val in ckpt.get("vars", {}).items():
                v = mdl.get_var_by_name(var_name)
                if v is not None:
                    ms.add_var_value(v, val)
            mdl.add_mip_start(ms)
            print(f"[checkpoint] Warmstart loaded from {os.path.basename(best_ckpt)} "
                  f"(obj={ckpt['meta']['objective']:.4f})", flush=True)
        except Exception as exc:
            print(f"[checkpoint] Could not load warmstart: {exc}", flush=True)

    mdl.add_progress_listener(_CheckpointListener())

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(SOLVE_LOG_PATH, "a") as log_fh:
        log_fh.write(f"\n{'='*60}\n  Run started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}\n")
        log_fh.flush()

        class _TeeLog:
            def write(self, text):
                log_fh.write(text)
                sys.stdout.write(text)
            def flush(self):
                log_fh.flush()
                sys.stdout.flush()

        solution = mdl.solve(log_output=_TeeLog())

    if not solution:
        print("[solve] No feasible solution found within time limit.", flush=True)
        return None

    print(f"[solve] Status: {mdl.solve_details.status} | "
          f"MIP gap: {mdl.solve_details.mip_relative_gap:.4%}", flush=True)

    n_nodes  = sets_["N"]
    arcs     = sets_["A"]
    periods  = sets_["T"]
    vehicles = sets_["M"]
    clients  = sets_["clients"]
    q_lt     = params_["q_lt"]

    routes = {}
    for t in periods:
        trucks = []
        for k in vehicles:
            arcs_k = [(i, j) for (i, j) in arcs if vars_["x"][i, j, t, k].solution_value > 0.5]
            if not arcs_k:
                continue
            path = _get_ordered_path(arcs_k)
            qty  = {}
            for node in path:
                if node == 0:
                    continue
                inbound   = sum(vars_["f"][i, node, t, k].solution_value for i in n_nodes if i != node)
                outbound  = sum(vars_["f"][node, j, t, k].solution_value for j in n_nodes if j != node)
                delivered = round(inbound - outbound, 4)
                if delivered > 1e-4:
                    qty[str(node)] = delivered
            trucks.append({"k": k, "path": path, "qty": qty})

        shipped_t = round(
            sum(vars_["f"][0, j, t, k].solution_value for j in clients for k in vehicles), 4
        )
        routes[str(t)] = {
            "R":          params_["R"].get(t, 0.0),
            "R_frigo":    params_["R_frigo"].get(t, 0.0),
            "R_nonfrigo": params_["R_nonfrigo"].get(t, 0.0),
            "tau_return": round(vars_["tau_return"][t].solution_value, 4),
            "trucks":     trucks,
            "shipped":    shipped_t,
        }

    deliveries = []
    for l in clients:
        for t in periods:
            for k in vehicles:
                inbound   = sum(vars_["f"][i, l, t, k].solution_value for i in n_nodes if i != l)
                outbound  = sum(vars_["f"][l, j, t, k].solution_value for j in n_nodes if j != l)
                delivered = round(inbound - outbound, 4)
                if delivered > 1e-4:
                    deliveries.append({"l": l, "t": t, "k": k, "recu": delivered,
                                       "dem": q_lt.get((l, t), 0)})

    depot_stock = {
        str(t): {
            "frigo":    round(vars_["I_O_frigo"][t].solution_value, 4),
            "nonfrigo": round(vars_["I_O_nonfrigo"][t].solution_value, 4),
        }
        for t in periods
    }

    f4_sub  = objectives["f4_sub"]
    bfr_sub = {
        "stock":       round(f4_sub["stock_value"].solution_value, 4),
        "receivables": round(f4_sub["receivables"].solution_value, 4),
        "payables":    round(f4_sub["payables"].solution_value, 4),
    }

    return {
        "meta": {
            "model_name":        mdl.name,
            "n_nodes":           len(sets_["N"]),
            "n_periods":         len(sets_["T"]),
            "n_vehicles":        len(sets_["M"]),
            "I_O_init_frigo":    params_["I_O_init_frigo"],
            "I_O_init_nonfrigo": params_["I_O_init_nonfrigo"],
            "node_positions":    compute_node_positions(sets_["N"]),
        },
        "objectives": {
            "f1":        round(f1.solution_value, 4),
            "f2":        round(f2.solution_value, 4),
            "f3":        round(f3.solution_value, 4),
            "f4":        round(f4.solution_value, 4),
            "composite": round(composite.solution_value, 4),
        },
        "routes":      routes,
        "deliveries":  deliveries,
        "depot_stock": depot_stock,
        "bfr_sub":     bfr_sub,
    }


def build_report_data(data_path=None):
    return run_combined_solve(data_path)


def run_function_merge(output_path=DEFAULT_REPORT_PATH, data_path=None):
    data = build_report_data(data_path)
    if data is None:
        return None
    return write_report(data, output_path)


if __name__ == "__main__":
    print(run_function_merge())
