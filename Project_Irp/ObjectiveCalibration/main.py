"""
Objective calibration module for the many-objective IRP model.
"""

import json
import os
import sys

from docplex.mp.model import Model

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(MODULE_DIR)

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from models.constraints import add_all_constraints, add_budget_constraints
from models.objectives import build_all_objectives
from models.variables import build_variables

try:
    from .report import compute_node_positions, write_report
except ImportError:
    from report import compute_node_positions, write_report


DEFAULT_REPORT_PATH = os.path.join(MODULE_DIR, "irp_calibration_report.html")

DATA_PATH = os.path.join(PROJECT_DIR, "data", "instance_5_clients.json")


def _imap(params_raw, key):
    return {int(k): v for k, v in params_raw[key].items()}


def _tmap(params_raw, key):
    return {
        (int(k.split(",")[0]), int(k.split(",")[1])): v
        for k, v in params_raw[key].items()
    }


def _load_problem_data(data_path=DATA_PATH):
    with open(data_path, encoding="utf-8") as fh:
        data = json.load(fh)

    sets_raw = data["sets"]
    params_raw = data["parameters"]

    n_nodes = sets_raw["N"]
    clients = sets_raw["clients"]
    depot = sets_raw["O"]
    periods = sets_raw["T"]
    vehicles = sets_raw["M"]
    arcs = [(i, j) for i in n_nodes for j in n_nodes if i != j]

    sets_ = {
        "N": n_nodes,
        "A": arcs,
        "T": periods,
        "M": vehicles,
        "O": depot,
        "clients": clients,
    }

    q_lt = _tmap(params_raw, "q_lt")
    k_lt = _tmap(params_raw, "K_lt")
    requires_cold = _tmap(params_raw, "requires_cold")
    speed = _imap(params_raw, "v")
    speed_squared = {k: (speed[k] / 3.6) ** 2 for k in vehicles}
    capacity = _imap(params_raw, "Q")

    distance = {(i, j): abs(i - j) * 10 for (i, j) in arcs}
    distance_meters = {(i, j): distance[i, j] * 1000 for (i, j) in arcs}

    route_cost = params_raw["c_route_value"]
    refrigeration_cost = params_raw["p5"]
    holding_cost = params_raw["h_O_space"] + params_raw["alpha_r"] * params_raw["e_stock"]

    c_ijk = {
        (i, j, k): (route_cost + refrigeration_cost / speed[k] if k in [1, 2] else route_cost)
        for (i, j) in arcs
        for k in vehicles
    }

    replenishment = {t: sum(q_lt[l, t] for l in clients) for t in periods}

    alpha_co2 = {(i, j): params_raw["g"] * params_raw["Cr"] for (i, j) in arcs}
    beta_co2 = 0.5 * params_raw["Cd"] * params_raw["A_f"] * params_raw["rho"]

    params_ = {
        "q_lt": q_lt,
        "K_lt": k_lt,
        "v": speed,
        "v2": speed_squared,
        "Q": capacity,
        "d": distance,
        "d_m": distance_meters,
        "c_ijk": c_ijk,
        "ET": {(l, t): params_raw["ET_value"] for l in clients for t in periods},
        "LT": {(l, t): params_raw["LT_value"] for l in clients for t in periods},
        "s": {i: params_raw["s_value"] for i in n_nodes},
        "tau_min": params_raw["tau_min"],
        "tau_max": params_raw["tau_max"],
        "I_O_init": params_raw["I_O_init"],
        "I_O_max": params_raw["I_O_max"],
        "I_O_min": params_raw["I_O_min"],
        "h_O": holding_cost,
        "R": replenishment,
        "alpha_co2": alpha_co2,
        "beta_co2": beta_co2,
        "w": params_raw["w"],
        "kg_per_unit": params_raw["kg_per_unit"],
        "e_co2": params_raw["e_co2"],
        "fuel_to_joules": params_raw["fuel_to_joules"],
        "P_sale": _imap(params_raw, "P_sale"),
        "P_purchase": _imap(params_raw, "P_purchase"),
        "DIO": params_raw["DIO"],
        "DSO": params_raw["DSO"],
        "DPO": params_raw["DPO"],
        "c1": params_raw["c1"],
        "c2": params_raw["c2"],
        "BIG_M": params_raw["BIG_M"],
        "requires_cold": requires_cold,
    }

    return sets_, params_


def _get_ordered_path(arcs):
    if not arcs:
        return []

    next_node = {i: j for (i, j) in arcs}
    destinations = {j for (_, j) in arcs}
    starts = [i for i in next_node if i not in destinations]
    current = starts[0] if starts else 0
    path = [current]
    visited = {current}

    while current in next_node and next_node[current] not in visited:
        current = next_node[current]
        path.append(current)
        visited.add(current)

    return path


def _build_model(data_path=DATA_PATH):
    sets_, params_ = _load_problem_data(data_path)
    mdl = Model(name="IRP_ManyObjective")
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
    return mdl, vars_, sets_, params_, objectives


def _solve_single(label, expr, mdl, vars_, sets_, params_, objectives):
    mdl.minimize(expr)
    solution = mdl.solve(log_output=False)
    if not solution:
        return None, None

    n_nodes = sets_["N"]
    arcs = sets_["A"]
    periods = sets_["T"]
    vehicles = sets_["M"]
    clients = sets_["clients"]
    q_lt = params_["q_lt"]
    replenishment = params_["R"]

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
            qty = {}
            for node in path:
                if node == 0:
                    continue
                inbound = sum(
                    vars_["f"][i, node, t, k].solution_value
                    for i in n_nodes
                    if i != node
                )
                outbound = sum(
                    vars_["f"][node, j, t, k].solution_value
                    for j in n_nodes
                    if j != node
                )
                delivered = round(inbound - outbound, 4)
                if delivered > 1e-4:
                    qty[str(node)] = delivered

            trucks.append(
                {
                    "k": k,
                    "path": path,
                    "qty": qty,
                    "ret": round(vars_["tau_return"][t].solution_value, 4),
                }
            )

        shipped_t = round(
            sum(vars_["f"][0, j, t, k].solution_value for j in clients for k in vehicles),
            4,
        )
        routes[str(t)] = {
            "R": replenishment.get(t, 0.0),
            "trucks": trucks,
            "shipped": shipped_t,
        }

    deliveries = []
    for l in clients:
        for t in periods:
            for k in vehicles:
                inbound = sum(
                    vars_["f"][i, l, t, k].solution_value
                    for i in n_nodes
                    if i != l
                )
                outbound = sum(
                    vars_["f"][l, j, t, k].solution_value
                    for j in n_nodes
                    if j != l
                )
                delivered = round(inbound - outbound, 4)
                if delivered > 1e-4:
                    deliveries.append(
                        {
                            "l": l,
                            "t": t,
                            "k": k,
                            "recu": delivered,
                            "dem": q_lt.get((l, t), 0),
                        }
                    )

    depot_stock = {
        str(t): round(vars_["I_O"][t].solution_value, 4)
        for t in periods
    }

    bfr_sub = None
    if "f4" in label:
        sub = objectives["f4_sub"]
        bfr_sub = {
            "stock": round(sub["stock_value"].solution_value, 4),
            "receivables": round(sub["receivables"].solution_value, 4),
            "payables": round(sub["payables"].solution_value, 4),
        }

    value = expr.solution_value
    return value, {
        "label": label,
        "value": round(value, 4),
        "routes": routes,
        "deliveries": deliveries,
        "depot_stock": depot_stock,
        "bfr_sub": bfr_sub,
    }


def build_report_data(data_path=DATA_PATH):
    mdl, vars_, sets_, params_, objectives = _build_model(data_path)
    calibration = [
        ("f1  Logistics cost", objectives["f1"], "C_max"),
        ("f2  CO2 emissions", objectives["f2"], "E_max"),
        ("f3  Total travel time", objectives["f3"], "T_max"),
        ("f4  Working capital (BFR)", objectives["f4"], "B"),
    ]

    calib_results = {}
    objs_data = []

    for label, expr, param_name in calibration:
        value, obj_data = _solve_single(label, expr, mdl, vars_, sets_, params_, objectives)
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
            "model_name": mdl.name,
            "n_nodes": len(sets_["N"]),
            "n_periods": len(sets_["T"]),
            "n_vehicles": len(sets_["M"]),
            "n_obj": len(calibration),
            "I_O_init": params_["I_O_init"],
            "node_positions": compute_node_positions(sets_["N"]),
        },
        "objs": objs_data,
        "bounds": calib_results,
    }


def run_objective_calibration(output_path=DEFAULT_REPORT_PATH, data_path=DATA_PATH):
    report_data = build_report_data(data_path)
    return write_report(report_data, output_path)


if __name__ == "__main__":
    print(run_objective_calibration())