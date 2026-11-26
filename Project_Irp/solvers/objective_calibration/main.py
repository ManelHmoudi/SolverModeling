"""
Objective calibration module for the many-objective IRP model.
"""

import os
import sys

from docplex.mp.model import Model

MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(MODULE_DIR))

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from models.parameters import load_instance
from models.constraints import add_all_constraints
from models.objectives import build_all_objectives, peak_breakdown, exact_objective_values
from models.variables import build_variables

try:
    from .report import compute_node_positions, write_report
except ImportError:
    from report import compute_node_positions, write_report


DEFAULT_REPORT_PATH = os.path.join(MODULE_DIR, "objective_calibration_report.html")


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


def _build_model(sets_, params_, timelimit=600):
    mdl  = Model(name="IRP_ManyObjective")
    mdl.parameters.timelimit = timelimit
    vars_ = build_variables(mdl, sets_, params_)
    objectives = build_all_objectives(mdl, vars_, sets_, params_)
    add_all_constraints(mdl, vars_, sets_, params_)
    return mdl, vars_, objectives


def _solve_single(label, expr, param_name, mdl, vars_, objectives, sets_, params_):
    mdl.minimize(expr)
    solution = mdl.solve(log_output=False)
    if not solution:
        return None, None
    solve_status = mdl.solve_details.status

    n_nodes  = sets_["N"]
    arcs     = sets_["A"]
    periods  = sets_["T"]
    vehicles = sets_["M"]
    clients  = sets_["clients"]
    q_lt    = params_["q_lt"]

    routes = {}
    for t in periods:
        trucks = []
        for k in vehicles:
            arcs_k = [
                (i, j)
                for (i, j) in arcs
                if vars_["x"][i, j, t, k].solution_value > 0.5
            ]
            if not arcs_k:
                continue

            path = _get_ordered_path(arcs_k)
            qty  = {}
            for node in path:
                if node == 0:
                    continue
                inbound  = sum(vars_["f"][i, node, t, k].solution_value for i in n_nodes if i != node)
                outbound = sum(vars_["f"][node, j, t, k].solution_value for j in n_nodes if j != node)
                delivered = round(inbound - outbound, 4)
                if delivered > 1e-4:
                    qty[str(node)] = delivered

            trucks.append({
                "k":    k,
                "path": path,
                "qty":  qty,
            })

        shipped_t = round(
            sum(vars_["f"][0, j, t, k].solution_value for j in clients for k in vehicles),
            4,
        )
        routes[str(t)] = {
            # supplier orders received in t (v2: decision variables)
            "R":          round(vars_["r"]["frigo", t].solution_value + vars_["r"]["nonfrigo", t].solution_value, 4),
            "R_frigo":    round(vars_["r"]["frigo", t].solution_value, 4),
            "R_nonfrigo": round(vars_["r"]["nonfrigo", t].solution_value, 4),
            # latest return among the vehicles used in t (return times are per vehicle in v2)
            "tau_return": round(max(
                [vars_["e"][t, k].solution_value for k in sets_["M"]
                 if sum(vars_["x"][sets_["O"], j, t, k].solution_value for j in sets_["clients"]) > 0.5]
                or [0.0]), 4),
            "trucks":     trucks,
            "shipped":    shipped_t,
        }

    deliveries = []
    for l in clients:
        for t in periods:
            for k in vehicles:
                inbound  = sum(vars_["f"][i, l, t, k].solution_value for i in n_nodes if i != l)
                outbound = sum(vars_["f"][l, j, t, k].solution_value for j in n_nodes if j != l)
                delivered = round(inbound - outbound, 4)
                if delivered > 1e-4:
                    deliveries.append({
                        "l":    l,
                        "t":    t,
                        "k":    k,
                        "recu": delivered,
                        "dem":  q_lt.get((l, t), 0),
                    })

    depot_stock = {
        str(t): {
            "frigo":    round(vars_["I_O_frigo"][t].solution_value, 4),
            "nonfrigo": round(vars_["I_O_nonfrigo"][t].solution_value, 4),
        }
        for t in periods
    }

    bfr_sub = None
    if "f4" in label:
        bfr_sub = peak_breakdown(objectives["f4_sub"])   # WC components at the peak period

    exact = exact_objective_values(vars_, objectives, sets_, params_)
    value = exact[param_name]
    return value, {
        "label":        label,
        "value":        round(value, 4),
        "solve_status": solve_status,
        "budget_key":   param_name,
        "payoff":       {k: round(v, 4) for k, v in exact.items()},
        "routes":       routes,
        "deliveries":   deliveries,
        "depot_stock":  depot_stock,
        "bfr_sub":      bfr_sub,
    }


def build_report_data(data_path=None, timelimit=600):
    """timelimit (seconds, default 600): CPLEX time limit PER OBJECTIVE (4
    single-objective solves total). Without a limit an infeasible-to-prove-
    optimal instance can hang indefinitely; docplex still returns the best
    incumbent found so far when the limit is hit (status reported in
    obj_data["solve_status"]), so a timed-out solve degrades gracefully to a
    feasible-but-not-proven-optimal point instead of failing."""
    sets_, params_ = load_instance(data_path)
    mdl, vars_, objectives = _build_model(sets_, params_, timelimit=timelimit)
    calibration = [
        ("f1  Logistics cost",        objectives["f1"], "f1"),
        ("f2  CO2 emissions",         objectives["f2"], "f2"),
        ("f3  Longest routes",        objectives["f3"], "f3"),
        ("f4  Peak working capital",  objectives["f4"], "f4"),
    ]

    # Payoff table: minimise each objective alone, record all four (exact
    # values). Ideal = diagonal, nadir estimate = worst value in each column.
    # Phase 1: ideal value of each objective (each minimised alone).
    ideal = {}
    for label, expr, param_name in calibration:
        mdl.minimize(expr)
        if mdl.solve(log_output=False):
            ideal[param_name] = exact_objective_values(vars_, objectives, sets_, params_)[param_name]
    # Phase 2: keep objective i at its ideal and minimise the SUM of the other
    # objectives, each divided by its own ideal (a normalised weighted sum, not
    # a lexicographic order), so every payoff row is Pareto-optimal and the
    # nadir estimate is not an arbitrary tie-break of the solver.
    objs_data = []
    for label, expr, param_name in calibration:
        if param_name not in ideal:
            continue
        tol = 1e-6 * max(1.0, abs(ideal[param_name]))
        ct = mdl.add_constraint(expr <= ideal[param_name] + tol, ctname=f"lex_{param_name}")
        others = mdl.sum(objectives[k] / max(1.0, abs(ideal[k])) for k in ideal if k != param_name)
        value, obj_data = _solve_single(label, others, param_name, mdl, vars_, objectives, sets_, params_)
        mdl.remove_constraint(ct)
        if value is not None:
            obj_data["mip_gap"] = round(mdl.solve_details.mip_relative_gap, 6)
            objs_data.append(obj_data)
    nadir = {k: max(o["payoff"][k] for o in objs_data) for k in ideal} if objs_data else {}

    return {
        "meta": {
            "model_name":        mdl.name,
            "n_nodes":           len(sets_["N"]),
            "n_periods":         len(sets_["T"]),
            "n_vehicles":        len(sets_["M"]),
            "n_obj":             len(objs_data),
            "n_obj_attempted":   len(calibration),
            "I_O_init_frigo":    params_["I_O_init_frigo"],
            "I_O_init_nonfrigo": params_["I_O_init_nonfrigo"],
            "node_positions":    compute_node_positions(sets_["N"]),
        },
        "objs":   objs_data,
        "ideal":  ideal,
        "nadir":  nadir,
        "bounds": nadir,          # displayed in the report next to each ideal
    }


def run_objective_calibration(output_path=DEFAULT_REPORT_PATH, data_path=None, timelimit=600):
    return write_report(build_report_data(data_path, timelimit=timelimit), output_path)


if __name__ == "__main__":
    print(run_objective_calibration())