"""
Many-Objective Inventory Routing Problem (IRP)
===============================================
Objectives : f1 - Logistics cost
             f2 - CO2 emissions (CMEM)
             f3 - Total travel time
             f4 - Working capital requirement (BFR)

Network    : G = (N, A)
             Depot   = 0
             Clients = {1, 2, 3}
             Destination = 4
"""

import json
import os
from docplex.mp.model import Model


# =============================================================================
# LOAD INSTANCE
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "instance_3_clients.json"), "r") as f:
    data = json.load(f)

sets   = data["sets"]
params = data["parameters"]


# =============================================================================
# SETS
# =============================================================================

N           = sets["N"]
clients     = sets["clients"]
stock_nodes = sets["stock_nodes"]
cold_nodes  = sets["cold_nodes"]
O, D        = sets["O"], sets["D"]
T           = sets["T"]
M           = sets["M"]
A           = [(i, j) for i in N for j in N if i != j]


# =============================================================================
# PARAMETERS
# =============================================================================

# -- Demand -------------------------------------------------------------------
q_lt = {
    (int(k.split(",")[0]), int(k.split(",")[1])): v
    for k, v in params["q_lt"].items()
}

# -- Vehicles -----------------------------------------------------------------
v    = {int(k): val for k, val in params["v"].items()}
v_ms = {k: v[k] / 3.6 for k in M}             # speed in m/s
v2   = {k: v_ms[k]**2 for k in M}             # squared speed (m/s)²
Q    = {int(k): val for k, val in params["Q"].items()}

# -- Network ------------------------------------------------------------------
d       = {(i, j): abs(i - j) * 10  for (i, j) in A}   # distance in km
d_m     = {(i, j): d[i, j] * 1000   for (i, j) in A}   # distance in metres (CMEM only)
c_route = {(i, j): params["c_route_value"] for (i, j) in A}   # base routing cost (currency/km)

# -- Refrigeration (vehicle type k=1) ----------------------------------------
p5      = params["p5"]       # refrigeration cost per unit time (currency/h)
e_stock = params["e_stock"]  # energy consumed per stored unit
alpha_r = params["alpha_r"]  # energy coefficient (inventory → consumption)

# -- Transport unit cost c_ijk (currency/km) ----------------------------------
# Refrigerated vehicles (k=1) incur an additional cost proportional to travel time.
c_ijk = {
    (i, j, k): (
        c_route[i, j] + p5 * d[i, j] / v[k]
        if k == 1
        else c_route[i, j]
    )
    for (i, j) in A for k in M
}

# -- Time windows & service times ---------------------------------------------
ET      = {(l, t): params["ET_value"] for l in clients for t in T}   # earliest arrival time
LT      = {(l, t): params["LT_value"] for l in clients for t in T}   # latest arrival time
tau_min = params["tau_min"]
tau_max = params["tau_max"]
s       = {i: params["s_value"] for i in N}                           # service time at node i (hours)

# -- Inventory ----------------------------------------------------------------
I_init  = {int(k): val for k, val in params["I_init"].items()}
I_max   = {int(k): val for k, val in params["I_max"].items()}
I_min   = {int(k): val for k, val in params["I_min"].items()}

h_space = {int(k): val for k, val in params["h_space"].items()}
h = {
    i: h_space[i] + alpha_r * e_stock if i in cold_nodes else h_space[i]
    for i in stock_nodes
}

# -- CO2 emissions — CMEM (Bektas & Laporte 2011) ----------------------------
g    = params["g"]      # gravitational acceleration (m/s²)
Cr   = params["Cr"]     # rolling resistance coefficient
Cd   = params["Cd"]     # aerodynamic drag coefficient
A_f  = params["A_f"]    # frontal area (m²)
rho  = params["rho"]    # air density (kg/m³)
a    = params["a"]      # vehicle acceleration (assumed zero)
w    = params["w"]      # curb weight (kg)

# alpha_co2 = g * Cr  (derived assuming zero gradient and zero acceleration)
alpha_co2 = {(i, j): g * Cr for (i, j) in A}

# beta_co2 = 0.5 * Cd * A_f * rho
beta_co2  = 0.5 * Cd * A_f * rho

e_co2 = params["e_co2"]    # CO2 conversion factor (kg CO2 / litre)

# -- Financial (BFR) ----------------------------------------------------------
BFR    = params["BFR"]
DSO    = params["DSO"]     # days sales outstanding (customer payment delay)
DPO    = params["DPO"]     # days payable outstanding (supplier payment delay)
V_val  = {int(k): val for k, val in params["V_val"].items()}

P_sale = {int(k): val for k, val in params["P_sale"].items()}

P_purchase = {int(k): val for k, val in params["P_purchase"].items()}

# -- Penalty coefficients -----------------------------------------------------
c1 = params["c1"]    # early-arrival penalty
c2 = params["c2"]    # late-arrival penalty

# -- Budget upper bounds ------------------------------------------------------
C_max  = params["C_max"]
E_max  = params["E_max"]
T_max  = params["T_max"]
B      = params["B"]
BIG_M  = params["BIG_M"]


# =============================================================================
# MODEL & DECISION VARIABLES
# =============================================================================

mdl = Model(name="IRP_ManyObjective")

# x[i,j,t,k]    : 1 if vehicle k traverses arc (i,j) in period t
x = {
    (i, j, t, k): mdl.binary_var(name=f"x_{i}_{j}_{t}_{k}")
    for (i, j) in A for t in T for k in M
}

# f[i,j,t,k]    : load carried by vehicle k on arc (i,j) in period t (units)
f = {
    (i, j, t, k): mdl.continuous_var(lb=0, name=f"f_{i}_{j}_{t}_{k}")
    for (i, j) in A for t in T for k in M
}

# u[k,t]        : 1 if vehicle k is active in period t
u = {
    (k, t): mdl.binary_var(name=f"u_{k}_{t}")
    for k in M for t in T
}

# q_prime[l,t]  : quantity delivered to customer l in period t (units)
q_prime = {
    (l, t): mdl.integer_var(lb=0, name=f"qprime_{l}_{t}")
    for l in clients for t in T
}

# tau[i,t]      : arrival time at node i in period t (hours)
tau = {
    (i, t): mdl.continuous_var(lb=0, name=f"tau_{i}_{t}")
    for i in N for t in T
}

# I_var[i,t]    : inventory level at node i at the end of period t
I_var = {
    (i, t): mdl.continuous_var(lb=0, name=f"I_{i}_{t}")
    for i in stock_nodes for t in T
}


# =============================================================================
# CONSTRAINTS
# =============================================================================

# -- Network flow -------------------------------------------------------------

# Each active vehicle departs exactly once from the depot
for k in M:
    for t in T:
        mdl.add_constraint(
            mdl.sum(x[O, j, t, k] for j in N if j != O) == u[k, t],
            ctname=f"c1_k{k}_t{t}"
        )

# Each active vehicle arrives exactly once at the destination
for k in M:
    for t in T:
        mdl.add_constraint(
            mdl.sum(x[i, D, t, k] for i in N if i != D) == u[k, t],
            ctname=f"c2_k{k}_t{t}"
        )

# Direct arc O → D is forbidden
for k in M:
    for t in T:
        mdl.add_constraint(x[O, D, t, k] == 0, ctname=f"c2b_k{k}_t{t}")

# Flow conservation at every node, conditioned on vehicle activation u[k,t]
for k in M:
    for t in T:
        for j in N:
            inflow  = mdl.sum(x[i, j, t, k] for i in N if i != j)
            outflow = mdl.sum(x[j, i, t, k] for i in N if i != j)
            if j == O:
                mdl.add_constraint(inflow - outflow == -u[k, t], ctname=f"c3_O_k{k}_t{t}")
            elif j == D:
                mdl.add_constraint(inflow - outflow ==  u[k, t], ctname=f"c3_D_k{k}_t{t}")
            else:
                mdl.add_constraint(inflow - outflow ==  0,       ctname=f"c3_{j}_k{k}_t{t}")

# At least one vehicle must be active per period
for t in T:
    mdl.add_constraint(
        mdl.sum(u[k, t] for k in M) >= 1,
        ctname=f"min_vehicle_t{t}"
    )

# -- Capacity & inventory -----------------------------------------------------

# Load on each arc cannot exceed vehicle capacity
for k in M:
    for (i, j) in A:
        for t in T:
            mdl.add_constraint(
                f[i, j, t, k] <= Q[k] * x[i, j, t, k],
                ctname=f"c4_{i}{j}_k{k}_t{t}"
            )

# Inventory levels must remain within bounds
for i in stock_nodes:
    for t in T:
        mdl.add_constraint(I_var[i, t] >= I_min[i], ctname=f"c5a_min_{i}_t{t}")
        mdl.add_constraint(I_var[i, t] <= I_max[i], ctname=f"c5a_max_{i}_t{t}")

# Inventory balance: depot decreases by outbound shipments; clients increase by deliveries
for t in T:
    prev_O  = I_var[O, t - 1] if t > 1 else I_init[O]
    shipped = mdl.sum(f[O, j, t, k] for j in clients for k in M)
    mdl.add_constraint(I_var[O, t] == prev_O - shipped, ctname=f"c5b_O_t{t}")

    for l in clients:
        prev_l = I_var[l, t - 1] if t > 1 else I_init[l]
        mdl.add_constraint(
            I_var[l, t] == prev_l + q_prime[l, t] - q_lt[l, t],
            ctname=f"c5b_{l}_t{t}"
        )

# Total delivery over the horizon equals total demand per customer
for l in clients:
    mdl.add_constraint(
        mdl.sum(q_prime[l, t] for t in T) == mdl.sum(q_lt[l, t] for t in T),
        ctname=f"c6_l{l}"
    )

# Freight flow balance at customer nodes
for l in clients:
    for t in T:
        inbound  = mdl.sum(f[i, l, t, k] for i in N if i != l for k in M)
        outbound = mdl.sum(f[l, j, t, k] for j in N if j != l for k in M)
        mdl.add_constraint(
            inbound - outbound == q_prime[l, t],
            ctname=f"c7_{l}_t{t}"
        )

# -- Time & time windows ------------------------------------------------------

# Depot departure time is fixed at zero
for t in T:
    mdl.add_constraint(tau[O, t] == 0, ctname=f"tau_origin_t{t}")

# Arrival time propagation using big-M linearisation
for k in M:
    for (i, j) in A:
        for t in T:
            mdl.add_constraint(
                tau[j, t] >= tau[i, t] + s[i] + d[i, j] / v[k]
                             - BIG_M * (1 - x[i, j, t, k]),
                ctname=f"c8_{i}{j}_k{k}_t{t}"
            )

# Destination arrival time bounds
for t in T:
    mdl.add_constraint(tau[D, t] >= tau_min, ctname=f"c9_min_t{t}")
    mdl.add_constraint(tau[D, t] <= tau_max, ctname=f"c9_max_t{t}")


# =============================================================================
# OBJECTIVE FUNCTIONS
# =============================================================================

# -- f1 : Logistics cost (transport + storage + time-window penalties) --------

# Transport cost: load-weighted, with refrigeration surcharge for k=1
y1 = mdl.sum(
    c_ijk[i, j, k] * d[i, j] * f[i, j, t, k]
    for (i, j) in A for t in T for k in M
)

# Storage cost: holding cost per unit per period
y2 = mdl.sum(h[i] * I_var[i, t] for i in stock_nodes for t in T)

# Time-window penalty: linearised via non-negative slack variables
w1 = {(l, t): mdl.continuous_var(lb=0, name=f"w1_{l}_{t}") for l in clients for t in T}
w2 = {(l, t): mdl.continuous_var(lb=0, name=f"w2_{l}_{t}") for l in clients for t in T}

for l in clients:
    for t in T:
        mdl.add_constraint(w1[l, t] >= ET[l, t] - tau[l, t], ctname=f"w1_{l}_t{t}")
        mdl.add_constraint(w2[l, t] >= tau[l, t] - LT[l, t], ctname=f"w2_{l}_t{t}")

y3 = mdl.sum(c1 * w1[l, t] + c2 * w2[l, t] for l in clients for t in T)

f1_expr = y1 + y2 + y3

# -- f2 : Carbon emissions — linearised CMEM formulation ----------------------
# Based on Bektas & Laporte (2011): fuel consumption depends on vehicle weight,
# payload, aerodynamic drag, and speed. Units: km / (km/h) = hours.
f2_expr = e_co2 * mdl.sum(
    (
        alpha_co2[i, j] * w * d_m[i, j] * x[i, j, t, k]
      + alpha_co2[i, j] * d_m[i, j]     * f[i, j, t, k]
      + beta_co2  * v2[k] * d_m[i, j]   * x[i, j, t, k]
    )
    for (i, j) in A for t in T for k in M
)

# -- f3 : Total travel time ---------------------------------------------------
# Sum of travel times over all used arcs. Units: km / (km/h) = hours.
f3_expr = mdl.sum(
    x[i, j, t, k] * d[i, j] / v[k]
    for (i, j) in A for t in T for k in M
)

# -- f4 : Working capital requirement (BFR) -----------------------------------

# Inventory value: capital tied up in stock
stock_value = mdl.sum(I_var[i, t] * P_purchase[i] for i in stock_nodes for t in T)

# Accounts receivable: outstanding revenue from customers
receivables = mdl.sum(q_prime[l, t] * P_sale[l] * (DSO / 365) for l in clients for t in T)

f4_expr = stock_value + receivables

# -- Budget constraints -------------------------------------------------------
mdl.add_constraint(f1_expr <= C_max, ctname="c10_logistics_budget")
mdl.add_constraint(f2_expr <= E_max, ctname="c11_carbon_budget")
mdl.add_constraint(f4_expr <= B,     ctname="c12_bfr_budget")
mdl.add_constraint(f3_expr <= T_max, ctname="c13_time_budget")


# =============================================================================
# MODEL SUMMARY
# =============================================================================

print(f"Model        : {mdl.name}")
print(f"Nodes        : {len(N)}  (depot + {len(clients)} customers + destination)")
print(f"Stock nodes  : {stock_nodes}")
print(f"Arcs         : {len(A)}")
print(f"Periods      : {len(T)}")
print(f"Vehicles     : {len(M)}")
print(f"Variables    : {mdl.number_of_variables}")
print(f"Constraints  : {mdl.number_of_constraints}")


# =============================================================================
# CALIBRATION — solve each objective independently
# =============================================================================

def test_objective(name, expr):
    mdl.minimize(expr) #fix the objective function
    sol = mdl.solve(log_output=False)
    if sol:
        print(f"✅ {name} = {expr.solution_value:.4f}")
        used = [
            (i, j, t, k)
            for (i, j, t, k) in x
            if x[i, j, t, k].solution_value > 0.5
        ]
        print(f"   Arcs used  : {used}")
        deliveries = {
            (l, t): round(q_prime[l, t].solution_value, 6)
            for (l, t) in q_prime
        }
        print(f"   Deliveries : {deliveries}")
        return expr.solution_value
    else:
        print(f"❌ {name} : infeasible — {mdl.solve_details}")
        return None

print("\n── Calibration f1 (logistics cost) ──")
f1_val = test_objective("f1", f1_expr)
if f1_val:
    print(f"   → Suggested C_max : {f1_val * 1.2:.4f}")

print("\n── Calibration f2 (CO2 emissions) ──")
f2_val = test_objective("f2", f2_expr)
if f2_val:
    print(f"   → Suggested E_max : {f2_val * 1.2:.4f}")

print("\n── Calibration f3 (travel time) ──")
f3_val = test_objective("f3", f3_expr)
if f3_val:
    print(f"   → Suggested T_max : {f3_val * 1.2:.4f}")

print("\n── Calibration f4 (BFR) ──")
f4_val = test_objective("f4", f4_expr)
if f4_val:
    print(f"   → Suggested B     : {f4_val * 1.2:.4f}")
    print(f"   Breakdown :")
    print(f"     Stock value  = {stock_value.solution_value:.4f}")
    print(f"     Receivables  = {receivables.solution_value:.4f}")