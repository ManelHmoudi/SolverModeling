"""
Objective calibration module for the many-objective IRP model.
"""

import os
import sys

from docplex.mp.model import Model

MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(MODULE_DIR)

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from models.parametres import sets_, params_
from models.constraints import add_all_constraints, add_budget_constraints
from models.objectives import build_all_objectives
from models.variables import build_variables

try:
    from .report import compute_node_positions, write_report
except ImportError:
    from report import compute_node_positions, write_report


DEFAULT_REPORT_PATH = os.path.join(MODULE_DIR, "irp_calibration_report.html")


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


def _build_model():
    mdl  = Model(name="IRP_ManyObjective")
    vars_ = build_variables(
        mdl,
        sets_["N"],
        sets_["A"],
        sets_["T"],
        sets_["M"],
        sets_["clients"],
    )
    objectives = build_all_objectives(mdl, vars_, sets_, params_)
    add_all_constraints(mdl, vars_, sets_, params_)
    return mdl, vars_, objectives


def _solve_single(label, expr, mdl, vars_, objectives):
    mdl.minimize(expr)
    solution = mdl.solve(log_output=False)
    if not solution:
        return None, None

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
                "ret":  round(vars_["tau_return"][t].solution_value, 4),
            })

        shipped_t = round(
            sum(vars_["f"][0, j, t, k].solution_value for j in clients for k in vehicles),
            4,
        )
        routes[str(t)] = {
            "R":         params_["R"].get(t, 0.0),
            "R_frigo":   params_["R_frigo"].get(t, 0.0),
            "R_nonfrigo": params_["R_nonfrigo"].get(t, 0.0),
            "trucks":    trucks,
            "shipped":   shipped_t,
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
        sub = objectives["f4_sub"]
        bfr_sub = {
            "stock":       round(sub["stock_value"].solution_value, 4),
            "receivables": round(sub["receivables"].solution_value, 4),
            "payables":    round(sub["payables"].solution_value, 4),
        }

    value = expr.solution_value
    return value, {
        "label":       label,
        "value":       round(value, 4),
        "routes":      routes,
        "deliveries":  deliveries,
        "depot_stock": depot_stock,
        "bfr_sub":     bfr_sub,
    }


def build_report_data():
    mdl, vars_, objectives = _build_model()
    calibration = [
        ("f1  Logistics cost",       objectives["f1"], "C_max"),
        ("f2  CO2 emissions",        objectives["f2"], "E_max"),
        ("f3  Total travel time",    objectives["f3"], "T_max"),
        ("f4  Working capital (BFR)", objectives["f4"], "B"),
    ]

    calib_results = {}
    objs_data     = []

    for label, expr, param_name in calibration:
        value, obj_data = _solve_single(label, expr, mdl, vars_, objectives)
        if value is not None:
            calib_results[param_name] = round(value * 1.2, 4)
            objs_data.append(obj_data)

    if len(calib_results) == 4:
        add_budget_constraints(
            mdl,
            objectives["f1"],
            objectives["f2"],
            objectives["f3"],
            objectives["f4"],
            calib_results["C_max"],
            calib_results["E_max"],
            calib_results["T_max"],
            calib_results["B"],
        )

    return {
        "meta": {
            "model_name":        mdl.name,
            "n_nodes":           len(sets_["N"]),
            "n_periods":         len(sets_["T"]),
            "n_vehicles":        len(sets_["M"]),
            "n_obj":             len(calibration),
            "I_O_init_frigo":    params_["I_O_init_frigo"],
            "I_O_init_nonfrigo": params_["I_O_init_nonfrigo"],
            "node_positions":    compute_node_positions(sets_["N"]),
        },
        "objs":   objs_data,
        "bounds": calib_results,
    }


def run_objective_calibration(output_path=DEFAULT_REPORT_PATH):
    return write_report(build_report_data(), output_path)


if __name__ == "__main__":
    print(run_objective_calibration())
