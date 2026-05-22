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
    Depot       = node 0
    Clients     = {1, 2, 3}
    Destination = node 4
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from docplex.mp.model import Model

from models.variables   import build_variables
from models.objectives  import build_all_objectives
from models.constraints import add_all_constraints

# =============================================================================
# LOAD INSTANCE DATA
# =============================================================================

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "instance_3_clients.json")

with open(DATA_PATH, "r") as f:
    data = json.load(f)

sets_raw   = data["sets"]
params_raw = data["parameters"]


# =============================================================================
# SETS
# =============================================================================

N           = sets_raw["N"]            # all nodes: depot + clients + destination
clients     = sets_raw["clients"]      # customer nodes
stock_nodes = sets_raw["stock_nodes"]  # nodes that hold inventory
O           = sets_raw["O"]            # origin (depot)
D           = sets_raw["D"]            # destination node
T           = sets_raw["T"]            # planning periods
M           = sets_raw["M"]            # vehicle types (1 = refrigerated, 2 = standard)
A           = [(i, j) for i in N for j in N if i != j]  # all directed arcs

cold_nodes_by_period = {
    1: [0, 1],
    2: [0, 2, 3]
}

sets_ = {
    "N":           N,
    "A":           A,
    "T":           T,
    "M":           M,
    "O":           O,
    "D":           D,
    "clients":     clients,
    "stock_nodes": stock_nodes,
}


# =============================================================================
# PARAMETERS
# =============================================================================

# --- Demand ------------------------------------------------------------------
# q_lt[l, t]: demand of customer l in period t (units)
q_lt = {
    (int(k.split(",")[0]), int(k.split(",")[1])): v
    for k, v in params_raw["q_lt"].items()
}

# --- Vehicle type compatibility ----------------------------------------------
requires_cold = {
    (int(k.split(",")[0]), int(k.split(",")[1])): v
    for k, v in params_raw["requires_cold"].items()
}

# --- Vehicles ----------------------------------------------------------------
v    = {int(k): val for k, val in params_raw["v"].items()}           # speed (km/h)
v_ms = {k: v[k] / 3.6 for k in M}                                   # speed (m/s)
v2   = {k: v_ms[k] ** 2 for k in M}                                 # speed squared (m/s)²
Q    = {int(k): val for k, val in params_raw["Q"].items()}           # vehicle capacity (units)

# --- Network -----------------------------------------------------------------
d       = {(i, j): abs(i - j) * 10 for (i, j) in A}                # distance (km)
d_m     = {(i, j): d[i, j] * 1000  for (i, j) in A}                # distance (m), used in CMEM
c_route = {(i, j): params_raw["c_route_value"] for (i, j) in A}    # base routing cost (currency/km)

# --- Refrigeration cost (vehicle type k=1 only) ------------------------------
p5      = params_raw["p5"]       # refrigeration operating cost per unit time (currency/h)
e_stock = params_raw["e_stock"]  # energy consumed per stored unit
alpha_r = params_raw["alpha_r"]  # coefficient converting inventory to energy consumption

# --- Per-arc transport unit cost c_ijk (currency/km) -------------------------
# Refrigerated vehicles (k=1) add a time-based refrigeration surcharge.
c_ijk = {
    (i, j, k): (
        c_route[i, j] + p5 / v[k]
        if k == 1
        else c_route[i, j]
    )
    for (i, j) in A for k in M
}

# --- Time windows and service times -----------------------------------------
ET      = {(l, t): params_raw["ET_value"] for l in clients for t in T}  # earliest arrival time (h)
LT      = {(l, t): params_raw["LT_value"] for l in clients for t in T}  # latest arrival time (h)
tau_min = params_raw["tau_min"]   # minimum arrival time at destination (h)
tau_max = params_raw["tau_max"]   # maximum arrival time at destination (h)
s       = {i: params_raw["s_value"] for i in N}                          # service time at each node (h)

# --- Inventory ---------------------------------------------------------------
I_init  = {int(k): val for k, val in params_raw["I_init"].items()}  # initial inventory level
I_max   = {int(k): val for k, val in params_raw["I_max"].items()}   # maximum inventory capacity
I_min   = {int(k): val for k, val in params_raw["I_min"].items()}   # minimum inventory level (safety stock)
h_space = {int(k): val for k, val in params_raw["h_space"].items()}

# Holding cost per unit per period:
#   cold nodes add an energy-based refrigeration term on top of the space cost.
h = {}
for i in stock_nodes:
    for t in T:
        if i in cold_nodes_by_period[t]:
            h[i, t] = h_space[i] + alpha_r * e_stock
        else:
            h[i, t] = h_space[i]

# --- CO2 emissions — Comprehensive Modal Emission Model (CMEM) ---------------
# Reference: Bektas & Laporte (2011)
# Fuel consumption depends on vehicle weight, payload, aerodynamic drag, and speed.
g    = params_raw["g"]    # gravitational acceleration (m/s²)
Cr   = params_raw["Cr"]   # rolling resistance coefficient
Cd   = params_raw["Cd"]   # aerodynamic drag coefficient
A_f  = params_raw["A_f"]  # vehicle frontal area (m²)
rho  = params_raw["rho"]  # air density (kg/m³)
a    = params_raw["a"]    # vehicle acceleration (m/s²), assumed zero
w    = params_raw["w"]    # curb weight of vehicle (kg)

# alpha_co2[i,j] = g * Cr  (zero gradient and zero acceleration assumed)
alpha_co2 = {(i, j): g * Cr for (i, j) in A}

# beta_co2 = 0.5 * Cd * A_f * rho  (aerodynamic drag term)
beta_co2 = 0.5 * Cd * A_f * rho

e_co2          = params_raw["e_co2"]          # CO2 emission factor (kg CO2 / litre of fuel)
kg_per_unit    = params_raw["kg_per_unit"]    # weight of one inventory unit (kg)
fuel_to_joules = params_raw["fuel_to_joules"] # energy conversion factor (J/litre)

# --- Financial parameters (BFR) ----------------------------------------------
DSO        = params_raw["DSO"]   # days sales outstanding  (customer payment delay, days)
DPO        = params_raw["DPO"]   # days payable outstanding (supplier payment delay, days)
DIO        = params_raw["DIO"]   # days inventory outstanding (average inventory holding time, days)
P_sale     = {int(k): val for k, val in params_raw["P_sale"].items()}     # selling price per unit at client l
P_purchase = {int(k): val for k, val in params_raw["P_purchase"].items()} # purchase price per unit at depot

# --- Time-window penalty coefficients ----------------------------------------
c1 = params_raw["c1"]  # penalty per hour of early arrival
c2 = params_raw["c2"]  # penalty per hour of late arrival

# --- Scalar bounds -----------------------------------------------------------
C_max = params_raw["C_max"]   # logistics cost budget (f1 upper bound)
E_max = params_raw["E_max"]   # carbon emission budget (f2 upper bound)
T_max = params_raw["T_max"]   # travel time budget (f3 upper bound)
B     = params_raw["B"]       # BFR budget (f4 upper bound)
BIG_M = params_raw["BIG_M"]   # big-M constant for linearisation

params_ = {
    "q_lt":          q_lt,
    "requires_cold": requires_cold,
    "v":             v,
    "v2":            v2,
    "Q":             Q,
    "d":             d,
    "d_m":           d_m,
    "c_ijk":         c_ijk,
    "ET":            ET,
    "LT":            LT,
    "tau_min":       tau_min,
    "tau_max":       tau_max,
    "s":             s,
    "I_init":        I_init,
    "I_max":         I_max,
    "I_min":         I_min,
    "h":             h,
    "alpha_co2":     alpha_co2,
    "beta_co2":      beta_co2,
    "w":             w,
    "kg_per_unit":   kg_per_unit,
    "e_co2":         e_co2,
    "fuel_to_joules":fuel_to_joules,
    "P_sale":        P_sale,
    "P_purchase":    P_purchase,
    "DIO":           DIO,
    "DSO":           DSO,
    "DPO":           DPO,
    "c1":            c1,
    "c2":            c2,
    "C_max":         C_max,
    "E_max":         E_max,
    "T_max":         T_max,
    "B":             B,
    "BIG_M":         BIG_M,
}


# =============================================================================
# MODEL AND DECISION VARIABLES
# =============================================================================

mdl   = Model(name="IRP_ManyObjective")
vars_ = build_variables(mdl, N, A, T, M, clients, stock_nodes)


# =============================================================================
# OBJECTIVE FUNCTIONS
# =============================================================================

objectives = build_all_objectives(mdl, vars_, sets_, params_)

# Inject objective expressions into params_ so budget constraints can reference them
params_["objectives"] = objectives


# =============================================================================
# CONSTRAINTS
# =============================================================================

add_all_constraints(mdl, vars_, sets_, params_)


# =============================================================================
# MODEL SUMMARY
# =============================================================================

print(f"\n{'─'*52}")
print(f"  {mdl.name}")
print(f"{'─'*52}")
print(f"  Nodes {len(N)} (depot+{len(clients)} clients+dest) | Arcs {len(A)} | Periods {len(T)} | Vehicles {len(M)}")
print(f"  Variables {mdl.number_of_variables} | Constraints {mdl.number_of_constraints}")
print(f"{'─'*52}")


# =====================================================================================================
# CALIBRATION — solve each objective independently to suggest bound values _ mono-objectif calibration
# =====================================================================================================

def _reconstruct_path(arcs):
    """Build an ordered path string like '0->1->3->4' from a list of (i, j) arc tuples."""
    if not arcs:
        return "(no arcs)"
    next_node = {i: j for (i, j) in arcs}
    destinations = {j for (_, j) in arcs}
    starts = [i for (i, _) in arcs if i not in destinations]
    current = starts[0] if starts else arcs[0][0]
    path = [current]
    while current in next_node:
        current = next_node[current]
        path.append(current)
    return "->".join(str(n) for n in path)


def test_objective(name, expr):
    mdl.minimize(expr)
    sol = mdl.solve(log_output=False)
    if not sol:
        print(f"  {name}: INFEASIBLE  ({mdl.solve_details})")
        return None

    val = expr.solution_value
    print(f"  {name} = {val:.4f}")

    arcs_by_tk = {}
    for (i, j, t, k) in vars_["x"]:
        if vars_["x"][i, j, t, k].solution_value > 0.5:
            arcs_by_tk.setdefault((t, k), []).append((i, j))

    if arcs_by_tk:
        routes = "  ".join(
            f"t{t}·k{k}: {_reconstruct_path(arcs_by_tk[t, k])}"
            for (t, k) in sorted(arcs_by_tk)
        )
        print(f"    Routes  : {routes}")
    else:
        print(f"    Routes  : (none)")

    nonzero = {
        (l, t, td): round(vars_["q_prime"][l, t, td].solution_value, 2)
        for (l, t, td) in vars_["q_prime"]
        if vars_["q_prime"][l, t, td].solution_value > 1e-6
    }
    if nonzero:
        deliv_str = "  ".join(f"l{l} t{t} td{td} = {q}" for (l, t, td), q in nonzero.items())
        print(f"    Deliver : {deliv_str}")

    return val


print("\n── Calibration ─────────────────────────────────────")

print("\n  f1  Logistics cost")
f1_val = test_objective("f1", objectives["f1"])
if f1_val: print(f"    → C_max = {f1_val * 1.2:.4f}")

print("\n  f2  CO2 emissions")
f2_val = test_objective("f2", objectives["f2"])
if f2_val: print(f"    → E_max = {f2_val * 1.2:.4f}")

print("\n  f3  Travel time")
f3_val = test_objective("f3", objectives["f3"])
if f3_val: print(f"    → T_max = {f3_val * 1.2:.4f}")

print("\n  f4  BFR (working capital)")
f4_val = test_objective("f4", objectives["f4"])
if f4_val:
    sub = objectives["f4_sub"]
    print(f"    → B     = {f4_val * 1.2:.4f}  "
          f"(stock {sub['stock_value'].solution_value:.2f}  "
          f"recv {sub['receivables'].solution_value:.2f}  "
          f"pay -{sub['payables'].solution_value:.2f})")

print(f"\n{'─'*52}")