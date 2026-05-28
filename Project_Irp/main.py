"""
Many-Objective Inventory Routing Problem (IRP)
===============================================
Objectives:
    f1 - Logistics cost         (transport + storage + time-window penalties)
    f2 - CO2 emissions          (CMEM model, Bektas & Laporte 2011)
    f3 - Total travel time      (sum of arc travel times)
    f4 - Working capital (BFR) (stock value + receivables - payables)

Network:
    G = (N, A)
    Depot   = node 0
    Clients = {1, 2, 3}

Workflow:
    1. Build model with structural constraints only (C1–C13)
    2. Calibration — solve each objective independently (no budget constraints)
       to find unconstrained optima f1*, f2*, f3*, f4*
    3. Set budget constraints C14–C17 at 1.2 × each optimum
    4. The model is now ready for Pareto-front exploration
"""

import sys
import os
import json
from docplex.mp.model import Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from models.variables   import build_variables
from models.objectives  import build_all_objectives
from models.constraints import add_all_constraints, add_budget_constraints

DATA_PATH = os.path.join(BASE_DIR, "data", "instance_3_clients.json")
with open(DATA_PATH) as fh:
    data = json.load(fh)

sets_raw, params_raw = data["sets"], data["parameters"]

def _imap(key):      return {int(k): v for k, v in params_raw[key].items()}
def _tmap(key):      return {(int(k.split(",")[0]), int(k.split(",")[1])): v
                             for k, v in params_raw[key].items()}
def _tmap_list(key): return {(int(k.split(",")[0]), int(k.split(",")[1])): v
                             for k, v in params_raw[key].items()}


# ── Sets ─────────────────────────────────────────────────────────────────────
N       = sets_raw["N"]
clients = sets_raw["clients"]
O       = sets_raw["O"]
T, M    = sets_raw["T"], sets_raw["M"]
A       = [(i, j) for i in N for j in N if i != j]

sets_ = {"N": N, "A": A, "T": T, "M": M, "O": O, "clients": clients}


# ── Parameters ───────────────────────────────────────────────────────────────
q_lt = _tmap("q_lt")
K_lt = _tmap_list("K_lt")
requires_cold = _tmap_list("requires_cold")
v  = _imap("v")
v2 = {k: (v[k] / 3.6) ** 2 for k in M}
Q  = _imap("Q")

d   = {(i, j): abs(i - j) * 10  for (i, j) in A}
d_m = {(i, j): d[i, j] * 1000   for (i, j) in A}

p5, e_stock, alpha_r = params_raw["p5"], params_raw["e_stock"], params_raw["alpha_r"]
c_route_val          = params_raw["c_route_value"]

c_ijk = {
    (i, j, k): (c_route_val + p5 / v[k] if k == 1 else c_route_val)
    for (i, j) in A for k in M
}

ET = {(l, t): params_raw["ET_value"] for l in clients for t in T}
LT = {(l, t): params_raw["LT_value"] for l in clients for t in T}
s  = {i: params_raw["s_value"] for i in N}

I_O_init = params_raw["I_O_init"]
I_O_max  = params_raw["I_O_max"]
I_O_min  = params_raw["I_O_min"]
h_O      = params_raw["h_O_space"] + alpha_r * e_stock

R = {t: sum(q_lt[l, t] for l in clients) for t in T}

g, Cr        = params_raw["g"],  params_raw["Cr"]
Cd, A_f, rho = params_raw["Cd"], params_raw["A_f"], params_raw["rho"]
w            = params_raw["w"]

alpha_co2 = {(i, j): g * Cr for (i, j) in A}
beta_co2  = 0.5 * Cd * A_f * rho

P_sale, P_purchase = _imap("P_sale"), _imap("P_purchase")

params_ = {
    "q_lt": q_lt,           "K_lt": K_lt,
    "v": v,                 "v2": v2,               "Q": Q,
    "d": d,                 "d_m": d_m,             "c_ijk": c_ijk,
    "ET": ET,               "LT": LT,               "s": s,
    "tau_min":        params_raw["tau_min"],
    "tau_max":        params_raw["tau_max"],
    "I_O_init": I_O_init,   "I_O_max": I_O_max,     "I_O_min": I_O_min,
    "h_O": h_O,             "R": R,
    "alpha_co2": alpha_co2, "beta_co2": beta_co2,   "w": w,
    "kg_per_unit":    params_raw["kg_per_unit"],
    "e_co2":          params_raw["e_co2"],
    "fuel_to_joules": params_raw["fuel_to_joules"],
    "P_sale": P_sale,       "P_purchase": P_purchase,
    "DIO": params_raw["DIO"], "DSO": params_raw["DSO"], "DPO": params_raw["DPO"],
    "c1": params_raw["c1"], "c2": params_raw["c2"],
    "BIG_M": params_raw["BIG_M"],
    "requires_cold": requires_cold,
}


# ── Model — structural constraints only (C1–C13) ─────────────────────────────
mdl   = Model(name="IRP_ManyObjective")
vars_ = build_variables(mdl, N, A, T, M, clients)

objectives = build_all_objectives(mdl, vars_, sets_, params_)
add_all_constraints(mdl, vars_, sets_, params_)

SEP = "─" * 60
print(f"\n{SEP}")
print(f"  {mdl.name}")
print(f"{SEP}")
print(f"  Nodes {len(N)} (depot 0 + {len(clients)} clients) | Arcs {len(A)} | Periods {len(T)} | Vehicles {len(M)}")
print(f"  Variables {mdl.number_of_variables} | Constraints {mdl.number_of_constraints} (structural)")
print(f"{SEP}")


# ── Calibration ──────────────────────────────────────────────────────────────

def _get_ordered_path(arcs):
    """Reconstructs the precise sequence of nodes from a list of active arcs."""
    if not arcs:
        return []
    next_node = {i: j for (i, j) in arcs}
    destinations = {j for (_, j) in arcs}
    starts = [i for i in next_node if i not in destinations]
    current = starts[0] if starts else 0  # Default to depot
    
    path = [current]
    visited = {current}
    while current in next_node and next_node[current] not in visited:
        current = next_node[current]
        path.append(current)
        visited.add(current)
    return path


def _solve_single(label, expr):
    mdl.minimize(expr)
    sol = mdl.solve(log_output=False)
    if not sol:
        print(f"  {label}: INFEASIBLE")
        return None

    val = expr.solution_value
    print(f"\n  >> {label.upper()} = {val:.4f}")

    # ── Affichage routes ─────────────────────────────────
    for t in T:
        print(f"    [Période t={t}]")
        has_activity = False
        for k in M:
            arcs_k = [(i, j) for (i, j) in A
                      if vars_["x"][i, j, t, k].solution_value > 0.5]
            if not arcs_k:
                continue
            has_activity = True
            path = _get_ordered_path(arcs_k)
            steps = []
            for node in path:
                if node == 0:
                    steps.append("0")
                else:
                    inflow  = sum(vars_["f"][i, node, t, k].solution_value
                                  for i in N if i != node)
                    outflow = sum(vars_["f"][node, j, t, k].solution_value
                                  for j in N if j != node)
                    qty = inflow - outflow
                    steps.append(f"{node}({'✓' if qty > 1e-4 else 'transit'}:{qty:.1f})")
            ret_time = vars_["tau_return"][t].solution_value
            print(f"      k={k} : {' → '.join(steps)} | retour {ret_time:.2f}h")
        if not has_activity:
            print("      Aucun camion.")

    # ── Résumé par client ────────────────────────────────
    print(f"\n    {'Client':<8} {'Période':<10} {'Camion':<8} {'Reçu':>8} {'Demande':>9} ")
    print(f"    {'─'*6:<8} {'─'*7:<10} {'─'*6:<8} {'─'*6:>8} {'─'*7:>9} ")

    for l in clients:
        for t in T:
            for k in M:
                inflow  = sum(vars_["f"][i, l, t, k].solution_value
                              for i in N if i != l)
                outflow = sum(vars_["f"][l, j, t, k].solution_value
                              for j in N if j != l)
                qty = inflow - outflow
                if qty > 1e-4:
                    demande = q_lt.get((l, t), 0)
                    print(f"    {l:<8} {t:<10} {k:<8} {qty:>8.1f} {demande:>9.1f} ")

    return val


print("\n── Calibration (without budget constraints) ────────")

calibration = [
    ("f1  Logistics cost",        objectives["f1"], "C_max"),
    ("f2  CO2 emissions",         objectives["f2"], "E_max"),
    ("f3  Total travel time",     objectives["f3"], "T_max"),
    ("f4  Working capital (BFR)", objectives["f4"], "B"),
]

calib_results = {}
for label, expr, param_name in calibration:
    val = _solve_single(label, expr)
    if val is not None:
        budget = val * 1.2
        calib_results[param_name] = budget
        if "f4" in label:
            sub = objectives["f4_sub"]
            print(f"    → Financement : Stock={sub['stock_value'].solution_value:.2f} | Créances={sub['receivables'].solution_value:.2f} | Dettes={sub['payables'].solution_value:.2f}")
        print(f"    → Borne Budget ({param_name}) = {budget:.4f}")


# ── Adding budget constraints C14–C17 ────────────────────────────────────────
print(f"\n{SEP}")
if len(calib_results) == 4:
    print("── Adding budget constraints (C14–C17) ─────────────")
    add_budget_constraints(
        mdl,
        objectives["f1"], objectives["f2"],
        objectives["f3"], objectives["f4"],
        calib_results["C_max"], calib_results["E_max"],
        calib_results["T_max"], calib_results["B"],
    )
    print(f"  Total constraints   : {mdl.number_of_constraints}")
    print(f"  C_max = {calib_results['C_max']:.4f} | E_max = {calib_results['E_max']:.4f}")
    print(f"  T_max = {calib_results['T_max']:.4f} | B     = {calib_results['B']:.4f}")
    print("  Model ready for Pareto-front exploration ✓")
else:
    print(f"    Calibration incomplete — budget constraints not added.")

print(f"{SEP}\n")
