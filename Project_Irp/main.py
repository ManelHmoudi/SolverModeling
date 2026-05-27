"""
Many-Objective Inventory Routing Problem (IRP)
===============================================
Objectives:
    f1 - Logistics cost        (transport + storage + time-window penalties)
    f2 - CO2 emissions         (CMEM model, Bektas & Laporte 2011)
    f3 - Total travel time     (sum of arc travel times)
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
}


# ── Model — structural constraints only (C1–C13) ─────────────────────────────
mdl   = Model(name="IRP_ManyObjective")
vars_ = build_variables(mdl, N, A, T, M, clients)

objectives = build_all_objectives(mdl, vars_, sets_, params_)
add_all_constraints(mdl, vars_, sets_, params_)

SEP = "─" * 52
print(f"\n{SEP}")
print(f"  {mdl.name}")
print(f"{SEP}")
print(f"  Nodes {len(N)} (depot + {len(clients)} clients) | "
      f"Arcs {len(A)} | Periods {len(T)} | Vehicles {len(M)}")
print(f"  Variables {mdl.number_of_variables} | "
      f"Constraints {mdl.number_of_constraints}  (before budget constraints)")
print(f"{SEP}")


# ── Calibration — solves each objective independently, without budget constraints ─

def _reconstruct_path(arcs):
    """Reconstruct the ordered path from a list of arcs (i, j)."""
    if not arcs:
        return "(no arc)"
    next_node    = {i: j for (i, j) in arcs}
    destinations = {j for (_, j) in arcs}
    starts       = [i for (i, _) in arcs if i not in destinations]
    current      = starts[0] if starts else arcs[0][0]
    path, visited = [current], {current}
    while current in next_node and next_node[current] not in visited:
        current = next_node[current]
        path.append(current)
        visited.add(current)
    return "->".join(str(n) for n in path)


def _solve_single(label, expr):
    """Solve the model for a single objective and print the solution."""
    mdl.minimize(expr)
    sol = mdl.solve(log_output=False)
    if not sol:
        print(f"  {label}: INFEASIBLE  ({mdl.solve_details.status})")
        return None

    val = expr.solution_value
    print(f"  {label} = {val:.4f}")

    # Active routes
    arcs_used = {
        (t, k): [(i, j) for (i, j) in A
                 if vars_["x"][i, j, t, k].solution_value > 0.5]
        for t in T for k in M
    }
    route_str = "  ".join(
        f"t{t}·k{k}: {_reconstruct_path(arcs_used[t, k])}"
        for (t, k) in sorted(arcs_used)
        if arcs_used[t, k]
    )
    print(f"    Routes        : {route_str or '(none)'}")

    # Depot return time
    for t in T:
        ret = vars_["tau_return"][t].solution_value
        print(f"    Depot return t={t} : {ret:.3f} h")

    # Effective deliveries
    deliveries = {
        (l, t, td): round(vars_["q_prime"][l, t, td].solution_value, 2)
        for (l, t, td) in vars_["q_prime"]
        if vars_["q_prime"][l, t, td].solution_value > 1e-6
    }
    if deliveries:
        print("    Deliveries : " + "  ".join(
            f"l{l} t{t} td{td}={q}" for (l, t, td), q in deliveries.items()
        ))

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
    print(f"\n  {label}")
    val = _solve_single(label, expr)
    if val is not None:
        budget = val * 1.2
        calib_results[param_name] = budget
        extra = ""
        if "f4" in label:
            sub = objectives["f4_sub"]
            extra = (f"  (stock {sub['stock_value'].solution_value:.2f}  "
                     f"recv {sub['receivables'].solution_value:.2f}  "
                     f"pay -{sub['payables'].solution_value:.2f})")
        print(f"    → {param_name:<5} = {budget:.4f}{extra}")


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
    print(f"  C_max = {calib_results['C_max']:.4f}")
    print(f"  E_max = {calib_results['E_max']:.4f}")
    print(f"  T_max = {calib_results['T_max']:.4f}")
    print(f"  B     = {calib_results['B']:.4f}")
    print("  Model ready for Pareto-front exploration ✓")
else:
    print(f"    Calibration incomplete ({len(calib_results)}/4) — budget constraints not added.")

print(f"{SEP}\n")