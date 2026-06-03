"""
FunctionMerge — Combined multi-objective IRP solve.

Scalarization:  minimize  f1 + f2 + f3 − f4
    f1  Logistics cost       (minimised)
    f2  CO2 emissions        (minimised)
    f3  Total travel time    (minimised)
    f4  Working capital BFR  (maximised → negated in composite)

All four objective expressions are active in a single CPLEX solve, which
verifies that every model component (variables, constraints, objectives)
works correctly together end-to-end.
"""

import os
import sys

from docplex.mp.model import Model

MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(MODULE_DIR)

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


DEFAULT_REPORT_PATH = os.path.join(MODULE_DIR, "irp_function_merge_report.html")


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
    add_budget_constraints(
        mdl,
        objectives["f1"], objectives["f2"],
        objectives["f3"], objectives["f4"],
        params_["C_max"], params_["E_max"],
        params_["T_max"], params_["B"],
    )
    return mdl, vars_, objectives


def run_combined_solve(data_path=None):
    sets_, params_ = load_instance(data_path)
    mdl, vars_, objectives = _build_model(sets_, params_)

    f1 = objectives["f1"]
    f2 = objectives["f2"]
    f3 = objectives["f3"]
    f4 = objectives["f4"]

    # Linear scalarization: minimise f1 + f2 + f3 − f4
    # f4 is negated so the solver maximises working-capital requirement.
    composite = f1 + f2 + f3 - f4
    mdl.minimize(composite)

    n_clients = len(sets_["clients"])
    if n_clients <= 5:
        mdl.parameters.timelimit = 60
    elif n_clients <= 15:
        mdl.parameters.timelimit = 300
        mdl.parameters.mip.tolerances.mipgap = 0.01
    else:
        mdl.parameters.timelimit = 300
        mdl.parameters.mip.tolerances.mipgap = 0.05

    mdl.parameters.mip.strategy.heuristicfreq = 10

    solution = mdl.solve(log_output=True)
    if not solution:
        return None

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
                (i, j) for (i, j) in arcs
                if vars_["x"][i, j, t, k].solution_value > 0.5
            ]
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
            sum(vars_["f"][0, j, t, k].solution_value for j in clients for k in vehicles),
            4,
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
