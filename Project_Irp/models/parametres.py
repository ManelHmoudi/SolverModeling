"""
Load instance data from JSON and build sets_ / params_ dicts for the IRP model.
"""

import json
import os

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "instance_3_clients.json")

with open(DATA_PATH, "r") as f:
    data = json.load(f)

sets_raw   = data["sets"]
params_raw = data["parameters"]


# =============================================================================
# SETS
# =============================================================================

N           = sets_raw["N"]
clients     = sets_raw["clients"]
stock_nodes = sets_raw["stock_nodes"]
O           = sets_raw["O"]
D           = sets_raw["D"]
T           = sets_raw["T"]
M           = sets_raw["M"]
A           = [(i, j) for i in N for j in N if i != j]

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

q_lt = {
    (int(k.split(",")[0]), int(k.split(",")[1])): v
    for k, v in params_raw["q_lt"].items()
}

requires_cold = {
    (int(k.split(",")[0]), int(k.split(",")[1])): v
    for k, v in params_raw["requires_cold"].items()
}

v    = {int(k): val for k, val in params_raw["v"].items()}
v_ms = {k: v[k] / 3.6 for k in M}
v2   = {k: v_ms[k] ** 2 for k in M}
Q    = {int(k): val for k, val in params_raw["Q"].items()}

d       = {(i, j): abs(i - j) * 10 for (i, j) in A}
d_m     = {(i, j): d[i, j] * 1000  for (i, j) in A}
c_route = {(i, j): params_raw["c_route_value"] for (i, j) in A}

p5      = params_raw["p5"]
e_stock = params_raw["e_stock"]
alpha_r = params_raw["alpha_r"]

c_ijk = {
    (i, j, k): (
        c_route[i, j] + p5 / v[k]
        if k == 1
        else c_route[i, j]
    )
    for (i, j) in A for k in M
}

ET      = {(l, t): params_raw["ET_value"] for l in clients for t in T}
LT      = {(l, t): params_raw["LT_value"] for l in clients for t in T}
tau_min = params_raw["tau_min"]
tau_max = params_raw["tau_max"]
s       = {i: params_raw["s_value"] for i in N}

I_init  = {int(k): val for k, val in params_raw["I_init"].items()}
I_max   = {int(k): val for k, val in params_raw["I_max"].items()}
I_min   = {int(k): val for k, val in params_raw["I_min"].items()}
h_space = {int(k): val for k, val in params_raw["h_space"].items()}

h = {}
for i in stock_nodes:
    for t in T:
        if i in cold_nodes_by_period[t]:
            h[i, t] = h_space[i] + alpha_r * e_stock
        else:
            h[i, t] = h_space[i]

g    = params_raw["g"]
Cr   = params_raw["Cr"]
Cd   = params_raw["Cd"]
A_f  = params_raw["A_f"]
rho  = params_raw["rho"]
a    = params_raw["a"]
w    = params_raw["w"]

alpha_co2      = {(i, j): g * Cr for (i, j) in A}
beta_co2       = 0.5 * Cd * A_f * rho
e_co2          = params_raw["e_co2"]
kg_per_unit    = params_raw["kg_per_unit"]
fuel_to_joules = params_raw["fuel_to_joules"]

DSO        = params_raw["DSO"]
DPO        = params_raw["DPO"]
DIO        = params_raw["DIO"]
P_sale     = {int(k): val for k, val in params_raw["P_sale"].items()}
P_purchase = {int(k): val for k, val in params_raw["P_purchase"].items()}

c1    = params_raw["c1"]
c2    = params_raw["c2"]
C_max = params_raw["C_max"]
E_max = params_raw["E_max"]
T_max = params_raw["T_max"]
B     = params_raw["B"]
BIG_M = params_raw["BIG_M"]

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
