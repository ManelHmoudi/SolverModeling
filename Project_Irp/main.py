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

import json
import os
from docplex.mp.model import Model


# =============================================================================
# LOAD INSTANCE DATA
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(
    os.path.join(
        BASE_DIR,
        "C:\\Users\\Mariem\\OneDrive\\Bureau\\SolverModeling\\Project_Irp\\data\\instance_3_clients.json"
    ),
    "r"
) as f:
    data = json.load(f)

sets   = data["sets"]
params = data["parameters"]


# =============================================================================
# SETS
# =============================================================================

N           = sets["N"]            # all nodes: depot + clients + destination
clients     = sets["clients"]      # customer nodes
stock_nodes = sets["stock_nodes"]  # nodes that hold inventory
cold_nodes_by_period = {
    1: [0, 1],     
    2: [0, 2, 3]    
}
O, D        = sets["O"], sets["D"] # origin (depot) and destination nodes
T           = sets["T"]            # planning periods
M           = sets["M"]            # vehicle types (1 = refrigerated, 2 = standard)
A           = [(i, j) for i in N for j in N if i != j]  # all directed arcs


# =============================================================================
# PARAMETERS
# =============================================================================

# --- Demand ------------------------------------------------------------------
# q_lt[l, t]: demand of customer l in period t (units)
q_lt = {
    (int(k.split(",")[0]), int(k.split(",")[1])): v
    for k, v in params["q_lt"].items()
}

# --- Vehicles ----------------------------------------------------------------
v    = {int(k): val for k, val in params["v"].items()}           # speed (km/h)
v_ms = {k: v[k] / 3.6 for k in M}                               # speed (m/s)
v2   = {k: v_ms[k] ** 2 for k in M}                             # speed squared (m/s)²
Q    = {int(k): val for k, val in params["Q"].items()}           # vehicle capacity (units)

# --- Network -----------------------------------------------------------------
d       = {(i, j): abs(i - j) * 10 for (i, j) in A}            # distance (km)
d_m     = {(i, j): d[i, j] * 1000  for (i, j) in A}            # distance (m), used in CMEM
c_route = {(i, j): params["c_route_value"] for (i, j) in A}    # base routing cost (currency/km)

# --- Refrigeration cost (vehicle type k=1 only) ------------------------------
p5      = params["p5"]       # refrigeration operating cost per unit time (currency/h)
e_stock = params["e_stock"]  # energy consumed per stored unit
alpha_r = params["alpha_r"]  # coefficient converting inventory to energy consumption

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
ET      = {(l, t): params["ET_value"] for l in clients for t in T}  # earliest arrival time (h)
LT      = {(l, t): params["LT_value"] for l in clients for t in T}  # latest arrival time (h)
tau_min = params["tau_min"]   # minimum arrival time at destination (h)
tau_max = params["tau_max"]   # maximum arrival time at destination (h)
s       = {i: params["s_value"] for i in N}                          # service time at each node (h)

# --- Inventory ---------------------------------------------------------------
I_init = {int(k): val for k, val in params["I_init"].items()}  # initial inventory level
I_max  = {int(k): val for k, val in params["I_max"].items()}   # maximum inventory capacity
I_min  = {int(k): val for k, val in params["I_min"].items()}   # minimum inventory level (safety stock)

# Holding cost per unit per period:
#   cold nodes add an energy-based refrigeration term on top of the space cost.
h_space = {int(k): val for k, val in params["h_space"].items()}
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
g    = params["g"]    # gravitational acceleration (m/s²)
Cr   = params["Cr"]   # rolling resistance coefficient
Cd   = params["Cd"]   # aerodynamic drag coefficient
A_f  = params["A_f"]  # vehicle frontal area (m²)
rho  = params["rho"]  # air density (kg/m³)
a    = params["a"]    # vehicle acceleration (m/s²), assumed zero
w    = params["w"]    # curb weight of vehicle (kg)

# alpha_co2[i,j] = g * Cr  (zero gradient and zero acceleration assumed)
alpha_co2 = {(i, j): g * Cr for (i, j) in A}

# beta_co2 = 0.5 * Cd * A_f * rho  (aerodynamic drag term)
beta_co2 = 0.5 * Cd * A_f * rho

e_co2       = params["e_co2"]        # CO2 emission factor (kg CO2 / litre of fuel)
kg_per_unit = params["kg_per_unit"]  # weight of one inventory unit (kg)

# --- Financial parameters (BFR) ----------------------------------------------
DSO   = params["DSO"]   # days sales outstanding  (customer payment delay, days)
DPO   = params["DPO"]   # days payable outstanding (supplier payment delay, days)
DIO = params["DIO"]   = params["DIO"]   # days inventory outstanding (average inventory holding time, days)
P_sale     = {int(k): val for k, val in params["P_sale"].items()}     # selling price per unit at client l
P_purchase = {int(k): val for k, val in params["P_purchase"].items()} # purchase price per unit at depot

# --- Time-window penalty coefficients ----------------------------------------
c1 = params["c1"]  # penalty per hour of early arrival
c2 = params["c2"]  # penalty per hour of late arrival

requires_cold = {
    (int(k.split(",")[0]), int(k.split(",")[1])): v  # v est maintenant une liste
    for k, v in params["requires_cold"].items()
}

# --- Scalar bounds -----------------------------------------------------------
C_max  = params["C_max"]   # logistics cost budget (f1 upper bound)
E_max  = params["E_max"]   # carbon emission budget (f2 upper bound)
T_max  = params["T_max"]   # travel time budget (f3 upper bound)
B      = params["B"]       # BFR budget (f4 upper bound)
BIG_M  = params["BIG_M"]   # big-M constant for linearisation


# =============================================================================
# MODEL AND DECISION VARIABLES
# =============================================================================

mdl = Model(name="IRP_ManyObjective")

# x[i,j,t,k] : binary — 1 if vehicle k travels arc (i,j) in period t
x = {
    (i, j, t, k): mdl.binary_var(name=f"x_{i}_{j}_{t}_{k}")
    for (i, j) in A for t in T for k in M
}

# f[i,j,t,k] : continuous — load (units) carried by vehicle k on arc (i,j) in period t
f = {
    (i, j, t, k): mdl.continuous_var(lb=0, name=f"f_{i}_{j}_{t}_{k}")
    for (i, j) in A for t in T for k in M
}

# q_prime[l,t] : integer — quantity delivered to customer l in period t (units)
q_prime = {
    (l, t): mdl.integer_var(lb=0, name=f"qprime_{l}_{t}")
    for l in clients for t in T
}

# tau[i,t] : continuous — arrival time at node i in period t (hours)
tau = {
    (i, t): mdl.continuous_var(lb=0, name=f"tau_{i}_{t}")
    for i in N for t in T
}

# I_var[i,t] : continuous — inventory level at node i at the end of period t (units)
I_var = {
    (i, t): mdl.continuous_var(lb=0, name=f"I_{i}_{t}")
    for i in stock_nodes for t in T
}


# =============================================================================
# CONSTRAINTS
# =============================================================================

# --- Vehicle routing: departure from depot -----------------------------------
# Each vehicle departs at most once from the depot per period.
for k in M:
    for t in T:
        mdl.add_constraint(
            mdl.sum(x[O, j, t, k] for j in N if j != O) <= 1,
            ctname=f"c1_k{k}_t{t}"
        )

# --- Vehicle routing: arrival at destination ---------------------------------
# Each vehicle that departs the depot must arrive at the destination exactly once.
for k in M:
    for t in T:
        mdl.add_constraint(
            mdl.sum(x[i, D, t, k] for i in N if i != D and i != O)
            == mdl.sum(x[O, j, t, k] for j in N if j != O),
            ctname=f"c2_k{k}_t{t}"
        )

# --- Flow conservation at intermediate nodes ---------------------------------
# For every non-depot, non-destination node: inflow equals outflow.
for k in M:
    for t in T:
        for j in N:
            if j != O and j != D:
                inflow  = mdl.sum(x[i, j, t, k] for i in N if i != j)
                outflow = mdl.sum(x[j, i, t, k] for i in N if i != j)
                mdl.add_constraint(
                    inflow == outflow,
                    ctname=f"c3_{j}_k{k}_t{t}"
                )

# --- Vehicle capacity --------------------------------------------------------
# Load on any arc cannot exceed the vehicle's capacity.
for k in M:
    for (i, j) in A:
        for t in T:
            mdl.add_constraint(
                f[i, j, t, k] <= Q[k] * x[i, j, t, k],
                ctname=f"c4_{i}{j}_k{k}_t{t}"
            )

# --- Inventory bounds --------------------------------------------------------
# Inventory at each stock node must stay within [I_min, I_max].
for i in stock_nodes:
    for t in T:
        mdl.add_constraint(I_var[i, t] >= I_min[i], ctname=f"c5_min_{i}_t{t}")
        mdl.add_constraint(I_var[i, t] <= I_max[i], ctname=f"c5_max_{i}_t{t}")

# --- Inventory balance -------------------------------------------------------
# Depot: inventory decreases by the total quantity shipped to clients.
# Clients: inventory increases by delivered quantity and decreases by demand.
for t in T:
    prev_O  = I_var[O, t - 1] if t > 1 else I_init[O]
    shipped = mdl.sum(f[O, j, t, k] for j in clients for k in M)
    mdl.add_constraint(I_var[O, t] == prev_O - shipped, ctname=f"c6_a_{t}")

    for l in clients:
        prev_l = I_var[l, t - 1] if t > 1 else I_init[l]
        mdl.add_constraint(
            I_var[l, t] == prev_l + q_prime[l, t] - q_lt[l, t],
            ctname=f"c6_b_{l}_t{t}"
        )

# --- Total delivery equals total demand over the horizon ---------------------
for l in clients:
    mdl.add_constraint(
        mdl.sum(q_prime[l, t] for t in T) == mdl.sum(q_lt[l, t] for t in T),
        ctname=f"c7_{l}"
    )

# --- Freight flow balance at customer nodes ----------------------------------
# Net inbound flow at each client equals the quantity delivered to that client.
for l in clients:
    for t in T:
        K_lt = requires_cold[l, t]   # véhicules compatibles avec client l à période t
        inbound  = mdl.sum(f[i, l, t, k] for i in N if i != l for k in K_lt)
        outbound = mdl.sum(f[l, j, t, k] for j in N if j != l for k in K_lt)
        mdl.add_constraint(
            inbound - outbound == q_prime[l, t],
            ctname=f"c8_{l}_t{t}"
        )

# --- Arrival time propagation (big-M linearisation) -------------------------
# Departure from the depot is fixed at time zero.
for t in T:
    mdl.add_constraint(tau[O, t] == 0, ctname=f"c9_{t}")

# If vehicle k uses arc (i,j) in period t, arrival at j is at least
# (arrival at i) + (service time at i) + (travel time on arc).
for k in M:
    for (i, j) in A:
        for t in T:
            mdl.add_constraint(
                tau[j, t] >= tau[i, t] + s[i] + d[i, j] / v[k]
                             - BIG_M * (1 - x[i, j, t, k]),
                ctname=f"c10_{i}{j}_k{k}_t{t}"
            )

# --- Destination arrival window ----------------------------------------------
for t in T:
    mdl.add_constraint(tau[D, t] >= tau_min, ctname=f"c11_min_t{t}")
    mdl.add_constraint(tau[D, t] <= tau_max, ctname=f"c11_max_t{t}")
# --- Vehicle type compatibility ----------------------------------------------
for l in clients:
    for t in T:
        types_requis = requires_cold[l, t]
        if 1 not in types_requis:
            mdl.add_constraint(
                mdl.sum(x[i, l, t, 1] for i in N if i != l) == 0,
                ctname=f"c12_no_frigo_{l}_t{t}"
            )
        if 2 not in types_requis:
            mdl.add_constraint(
                mdl.sum(x[i, l, t, 2] for i in N if i != l) == 0,
                ctname=f"c12_no_standard_{l}_t{t}"
            )


# =============================================================================
# OBJECTIVE FUNCTIONS
# =============================================================================

# --- f1: Logistics cost ------------------------------------------------------
# Three components:
#   y1 — transport cost: load-weighted arc cost (with refrigeration surcharge for k=1)
#   y2 — storage cost: holding cost per unit per period
#   y3 — time-window penalty: linearised via non-negative slack variables

y1 = mdl.sum(
    c_ijk[i, j, k] * d[i, j] * f[i, j, t, k]
    for (i, j) in A for t in T for k in M
)

y2 = mdl.sum(h[i, t] * I_var[i, t] for i in stock_nodes for t in T)

# Slack variables for early (w1) and late (w2) arrival penalties
w1 = {(l, t): mdl.continuous_var(lb=0, name=f"w1_{l}_{t}") for l in clients for t in T}
w2 = {(l, t): mdl.continuous_var(lb=0, name=f"w2_{l}_{t}") for l in clients for t in T}

for l in clients:
    for t in T:
        mdl.add_constraint(w1[l, t] >= ET[l, t] - tau[l, t], ctname=f"w1_{l}_t{t}")
        mdl.add_constraint(w2[l, t] >= tau[l, t] - LT[l, t], ctname=f"w2_{l}_t{t}")

y3 = mdl.sum(c1 * w1[l, t] + c2 * w2[l, t] for l in clients for t in T)

f1_expr = y1 + y2 + y3

# --- f2: CO2 emissions (linearised CMEM) -------------------------------------
# Fuel consumption on each arc depends on:
#   - vehicle curb weight (w) and payload (f[i,j,t,k]), scaled by alpha_co2
#   - aerodynamic drag, scaled by beta_co2 and speed squared
# Units: energy in Joules (kg·m²/s²); converted to litres via fuel_to_joules,
#        then to kg CO2 via e_co2 (kg CO2/litre).
fuel_to_joules = params["fuel_to_joules"]

f2_expr = (e_co2 / fuel_to_joules) * mdl.sum(
    d_m[i, j] * (
        alpha_co2[i, j] * (w * x[i, j, t, k] + kg_per_unit * f[i, j, t, k])
      + beta_co2         *  v2[k]              * x[i, j, t, k]
    )
    for (i, j) in A for t in T for k in M
)

# --- f3: Total travel time ---------------------------------------------------
# Sum of travel times (hours) over all arcs used across all periods and vehicles.
f3_expr = mdl.sum(
    x[i, j, t, k] * d[i, j] / v[k]
    for (i, j) in A for t in T for k in M
)

# --- f4: Working capital requirement (BFR) -----------------------------------
# BFR = inventory value + accounts receivable - accounts payable
stock_value = mdl.sum(I_var[O, t] * P_purchase[O] * (DIO / 365) for t in T)

receivables = mdl.sum(
    q_prime[l, t] * P_sale[l] * (DSO / 365)
    for l in clients for t in T
)

payables = mdl.sum(
    q_prime[l, t] * P_purchase[O] * (DPO / 365)
    for l in clients for t in T
)

f4_expr = stock_value + receivables - payables

# --- Budget constraints (feasibility bounds on each objective) ---------------
mdl.add_constraint(f1_expr <= C_max, ctname="c13_logistics_budget")
mdl.add_constraint(f2_expr <= E_max, ctname="c14_carbon_budget")
mdl.add_constraint(f4_expr <= B,     ctname="c15_bfr_budget")
mdl.add_constraint(f3_expr <= T_max, ctname="c16_time_budget")


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
# CALIBRATION — solve each objective independently to suggest bound values
# =============================================================================

def test_objective(name, expr):
    """Minimise a single objective and report the optimal value with routing details."""
    mdl.minimize(expr)
    sol = mdl.solve(log_output=False)
    if sol:
        print(f" {name} = {expr.solution_value:.4f}")
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
        print(f" {name} : infeasible — {mdl.solve_details}")
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
    print(f"     Stock value   = {stock_value.solution_value:.4f}")
    print(f"     Receivables   = {receivables.solution_value:.4f}")
    print(f"     Payables (-)  = {payables.solution_value:.4f}")