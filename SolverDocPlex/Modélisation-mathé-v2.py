"""
Many-Objective Inventory Routing Problem (IRP)
===============================================
Objectives : f1 (logistics cost)  |  f2 (CO2 emissions)
             f3 (time)             |  f4 (financial cost)
Network    : G = (N, A)
             O = 0  (depot)
             clients = {1, 2, 3}
             D = 4  (destination)
"""

from docplex.mp.model import Model


# =============================================================================
# 1. SETS
# =============================================================================

N           = [0, 1, 2, 3, 4]          # all nodes
clients     = [1, 2, 3]                 # customer nodes
stock_nodes = [0, 1, 2, 3]             # nodes that carry inventory  (N \ {D})
cold_nodes  = [0, 1]                    # nodes with refrigerated storage
O, D        = 0, 4                      # origin depot, destination
T           = [1, 2]                    # planning periods
M           = [1, 2]                    # vehicle types: 1=refrigerated, 2=standard
A           = [(i, j) for i in N for j in N if i != j]   # valid arcs


# =============================================================================
# 2. PARAMETERS
# =============================================================================

# ── 2.1 Demand ────────────────────────────────────────────────────────────────
q_lt = {
    (1, 1): 20, (1, 2): 25,
    (2, 1): 30, (2, 2): 35,
    (3, 1): 15, (3, 2): 20,
}

# ── 2.2 Vehicles ──────────────────────────────────────────────────────────────
v = {1: 80, 2: 60}          # speed (km/h)
Q = {1: 100, 2: 80}         # capacity (units)

# ── 2.3 Network ───────────────────────────────────────────────────────────────
d = {(i, j): abs(i - j) * 10 for (i, j) in A}      # arc distances (km) — placeholder
c_route = {(i, j): 1.0      for (i, j) in A}        # base route cost (currency/km)

# ── 2.4 Refrigeration (vehicle type k=1) ─────────────────────────────────────
p5        = 3.0     # refrigeration cost per unit time (currency/h)
e_stock   = 0.1     # energy consumed per stored unit
alpha_r   = 0.05    # energy coefficient (stock → consumption)

# ── 2.5 Transport unit cost  c_ijk ───────────────────────────────────────────
# ── 2.5 Transport unit cost c_ijk (currency/km) ──────────────────────────────
c_ijk = {
    (i, j, k): (
        c_route[i, j] + p5 / v[1]   # k=1 : route + réfrigération/km
        if k == 1
        else c_route[i, j]           # k=2 : route uniquement
    )
    for (i, j) in A for k in M
}

# ── 2.6 Time windows & service times ─────────────────────────────────────────
ET      = {(l, t):  0.0 for l in clients for t in T}
LT      = {(l, t): 99.0 for l in clients for t in T}
tau_min = 0.0
tau_max = 9_999.0
s       = {i: 0.5 for i in N}      # service time at node i (hours)

# ── 2.7 Inventory ────────────────────────────────────────────────────────────
I_init = {i:  50 for i in stock_nodes}
I_max  = {i: 200 for i in stock_nodes}
I_min  = {i:  10 for i in stock_nodes}

h_space = {i: 0.3 for i in stock_nodes}
h = {
    i: h_space[i] + alpha_r * e_stock if i in cold_nodes else h_space[i]
    for i in stock_nodes
}

# ── 2.8 CO2 (Bektas & Laporte 2011 CMEM) ────────────────────────────────────
alpha_co2 = {(i, j): 0.01 for (i, j) in A}     # load-related energy coefficient
beta_co2  = 0.002                                # speed-related energy coefficient
e_co2     = 2.68                                 # CO2 conversion factor (kg/energy unit)

# ── 2.9 Financial ─────────────────────────────────────────────────────────────
BFR        = 3_000
DSO        = 30                                 # customer payment delay (days)
DPO        = 45                                 # supplier payment delay (days)
V_val      = {i: 100 for i in N}
P_sale     = 8.0
P_purchase = 5.0

# ── 2.10 Penalty coefficients ────────────────────────────────────────────────
c1 = 2.0    # early-arrival penalty coefficient
c2 = 5.0    # late-arrival  penalty coefficient

# ── 2.11 Upper bounds ────────────────────────────────────────────────────────
C_max  = 8_000   # logistics cost budget   (c12)
E_max  = 5_000   # CO2 emissions budget    (c13)
B      = 10_000  # financial cost budget   (c14)
BIG_M  = 99_999


# =============================================================================
# 3. MODEL & DECISION VARIABLES
# =============================================================================

mdl = Model(name="IRP_ManyObjective")

# x[i,j,t,k]      : 1 if vehicle k uses arc (i,j) in period t
x = {
    (i, j, t, k): mdl.binary_var(name=f"x_{i}_{j}_{t}_{k}")
    for (i, j) in A for t in T for k in M
}

# f[i,j,t,k]      : load carried by vehicle k on arc (i,j) in period t (units)
f = {
    (i, j, t, k): mdl.continuous_var(lb=0, name=f"f_{i}_{j}_{t}_{k}")
    for (i, j) in A for t in T for k in M
}

# q_prime[l,t]    : quantity delivered to customer l in period t (units)
q_prime = {
    (l, t): mdl.continuous_var(lb=0, name=f"qprime_{l}_{t}")
    for l in clients for t in T
}

# tau[i,t]        : arrival time at node i in period t (hours)
tau = {
    (i, t): mdl.continuous_var(lb=0, name=f"tau_{i}_{t}")
    for i in N for t in T
}

# I_var[i,t]      : stock level at node i at end of period t  (i ∈ N\{D})
I_var = {
    (i, t): mdl.continuous_var(lb=0, name=f"I_{i}_{t}")
    for i in stock_nodes for t in T
}


# =============================================================================
# 4. CONSTRAINTS
# =============================================================================

# ── 4.1 Network flow ──────────────────────────────────────────────────────────

# (c1) Each vehicle departs from O exactly once per period
for k in M:
    for t in T:
        mdl.add_constraint(
            mdl.sum(x[O, j, t, k] for j in N if j != O) == 1,
            ctname=f"c1_k{k}_t{t}"
        )

# (c2) Each vehicle arrives at D exactly once per period
for k in M:
    for t in T:
        mdl.add_constraint(
            mdl.sum(x[i, D, t, k] for i in N if i != D) == 1,
            ctname=f"c2_k{k}_t{t}"
        )

# (c2b) No direct arc O→D: vehicle must visit at least one customer
for k in M:
    for t in T:
        mdl.add_constraint(x[O, D, t, k] == 0, ctname=f"c2b_k{k}_t{t}")

# (c3) Flow conservation at each node
#      net flow = -1 (O) | 0 (transit) | +1 (D)
for k in M:
    for t in T:
        for j in N:
            inflow  = mdl.sum(x[i, j, t, k] for i in N if i != j)
            outflow = mdl.sum(x[j, i, t, k] for i in N if i != j)
            rhs     = -1 if j == O else (1 if j == D else 0)
            mdl.add_constraint(
                inflow - outflow == rhs,
                ctname=f"c3_{j}_k{k}_t{t}"
            )

# ── 4.2 Capacity & inventory ──────────────────────────────────────────────────

# (c4) Load on arc (i,j) ≤ vehicle capacity  (active only when arc is used)
for k in M:
    for (i, j) in A:
        for t in T:
            mdl.add_constraint(
                f[i, j, t, k] <= Q[k] * x[i, j, t, k],
                ctname=f"c4_{i}{j}_k{k}_t{t}"
            )

# (c5a) Stock bounds: I_min ≤ I_it ≤ I_max  for all i ∈ N\{D}
for i in stock_nodes:
    for t in T:
        mdl.add_constraint(I_var[i, t] >= I_min[i], ctname=f"c5a_min_{i}_t{t}")
        mdl.add_constraint(I_var[i, t] <= I_max[i], ctname=f"c5a_max_{i}_t{t}")

# (c5b) Inventory balance
#   depot  O : I_O,t  = I_O,t-1 − shipped_t
#   client l : I_l,t  = I_l,t-1 + q'_lt − q_lt
for t in T:
    prev_O  = I_var[O, t - 1] if t > 1 else I_init[O]
    shipped = mdl.sum(f[O, j, t, k] for j in N if j != O for k in M)
    mdl.add_constraint(I_var[O, t] == prev_O - shipped, ctname=f"c5b_O_t{t}")

    for l in clients:
        prev_l = I_var[l, t - 1] if t > 1 else I_init[l]
        mdl.add_constraint(
            I_var[l, t] == prev_l + q_prime[l, t] - q_lt[l, t],
            ctname=f"c5b_{l}_t{t}"
        )

# (c6) Total delivery = total demand over the horizon
for l in clients:
    mdl.add_constraint(
        mdl.sum(q_prime[l, t] for t in T) == mdl.sum(q_lt[l, t] for t in T),
        ctname=f"c6_l{l}"
    )

# (c7) Inbound flow at customer l = quantity delivered q'_lt
for l in clients:
    for t in T:
        inbound = mdl.sum(f[i, l, t, k] for i in N if i != l for k in M)
        mdl.add_constraint(inbound == q_prime[l, t], ctname=f"c7_{l}_t{t}")

# ── 4.3 Time & time windows ───────────────────────────────────────────────────

# (c8) Travel time lower bound (Big-M inactive when arc not used)
for k in M:
    for (i, j) in A:
        for t in T:
            mdl.add_constraint(
                tau[j, t] >= d[i, j] / v[k] - BIG_M * (1 - x[i, j, t, k]),
                ctname=f"c8_{i}{j}_k{k}_t{t}"
            )

# (c9) Arrival time propagation: τ_j ≥ τ_i + s_i + d_ij / v^k
for k in M:
    for (i, j) in A:
        for t in T:
            mdl.add_constraint(
                tau[j, t] >= tau[i, t] + s[i] + d[i, j] / v[k]
                             - BIG_M * (1 - x[i, j, t, k]),
                ctname=f"c9_{i}{j}_k{k}_t{t}"
            )

# (c10) Global route time window: τ_min ≤ τ_D,t ≤ τ_max
for t in T:
    mdl.add_constraint(tau[D, t] >= tau_min, ctname=f"c10_min_t{t}")
    mdl.add_constraint(tau[D, t] <= tau_max, ctname=f"c10_max_t{t}")

# (c11) Customer time windows: ET_lt ≤ τ_lt ≤ LT_lt
for l in clients:
    for t in T:
        mdl.add_constraint(tau[l, t] >= ET[l, t], ctname=f"c11_ET_l{l}_t{t}")
        mdl.add_constraint(tau[l, t] <= LT[l, t], ctname=f"c11_LT_l{l}_t{t}")


# =============================================================================
# 5. OBJECTIVE FUNCTIONS
# =============================================================================

# ── f1 : Logistics cost = transport + storage + time-window penalties ─────────

# y1 — transport cost (load-weighted, refrigeration surcharge for k=1)
y1 = mdl.sum(
    c_ijk[i, j, k] * d[i, j] * f[i, j, t, k]
    for (i, j) in A for t in T for k in M
)

# y2 — storage cost
y2 = mdl.sum(h[i] * I_var[i, t] for i in stock_nodes for t in T)

# y3 — time-window penalty (linearised via slack variables w1/w2)
w1 = {(l, t): mdl.continuous_var(lb=0, name=f"w1_{l}_{t}") for l in clients for t in T}
w2 = {(l, t): mdl.continuous_var(lb=0, name=f"w2_{l}_{t}") for l in clients for t in T}

for l in clients:
    for t in T:
        mdl.add_constraint(w1[l, t] >= ET[l, t] - tau[l, t], ctname=f"w1_{l}_t{t}")
        mdl.add_constraint(w2[l, t] >= tau[l, t] - LT[l, t], ctname=f"w2_{l}_t{t}")

y3 = mdl.sum(
    q_lt[l, t] * (c1 * w1[l, t] + c2 * w2[l, t])
    for l in clients for t in T
)

f1_expr = y1 + y2 + y3

# (c12) Logistics cost budget
mdl.add_constraint(f1_expr <= C_max, ctname="c12_logistics_budget")


# =============================================================================
# 6. SUMMARY
# =============================================================================

print(f"Model        : {mdl.name}")
print(f"Nodes        : {len(N)}  (depot + {len(clients)} customers + destination)")
print(f"Stock nodes  : {stock_nodes}")
print(f"Arcs         : {len(A)}")
print(f"Periods      : {len(T)}")
print(f"Vehicles     : {len(M)}")
print(f"Variables    : {mdl.number_of_variables}")
print(f"Constraints  : {mdl.number_of_constraints}")