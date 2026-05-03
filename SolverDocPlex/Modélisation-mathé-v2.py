from docplex.mp.model import Model

# ─────────────────────────────────────────────────────────────────────────────
# Many-Objective Inventory Routing Problem (IRP)
# Objectives : f1 (logistics cost), f2 (CO2), f3 (time), f4 (financial)
# Network    : G = (N, A)  —  O=depot, clients={1,2,3}, D=destination
# ─────────────────────────────────────────────────────────────────────────────

mdl = Model(name="IRP_ManyObjective")


# ═════════════════════════════════════════════════════════════════════════════
# SETS
# ═════════════════════════════════════════════════════════════════════════════

N           = [0, 1, 2, 3, 4]  # all nodes : 0=depot, 1-3=customers, 4=destination
clients     = [1, 2, 3]         # customer nodes  (l in clients subset N)
stock_nodes = [0, 1, 2, 3]      # nodes with stock : N \ {D}
O           = 0                  # origin depot
D           = 4                  # destination node
T           = [1, 2]             # planning periods
M           = [1, 2]             # vehicle types : 1=refrigerated, 2=standard

A = [(i, j) for i in N for j in N if i != j]   # all valid arcs


# ═════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ═════════════════════════════════════════════════════════════════════════════

# --- Customer demand q_lt (units) ---
q_lt = {
    (1, 1): 20, (1, 2): 25,
    (2, 1): 30, (2, 2): 35,
    (3, 1): 15, (3, 2): 20
}

# --- Vehicle speed v^k (km/h) ---
v = {1: 80, 2: 60}

# --- Arc distances d_ij (km) : placeholder, replace with real data ---
d = {(i, j): abs(i - j) * 10 for (i, j) in A}

# --- Unit transport cost c_ijk (currency/t·km) ---
# k=1 (refrigerated) : higher unit cost
# k=2 (standard)     : base unit cost
c_ijk = {
    (i, j, k): 1.2 if k == 1 else 1.0
    for (i, j) in A for k in M
}

# --- Customer time windows [ET_lt, LT_lt] (hours) ---
ET = {(l, t): 0.0  for l in clients for t in T}
LT = {(l, t): 99.0 for l in clients for t in T}

# --- Global route time bounds (hours) ---
tau_min = 0.0
tau_max = 9999.0

# --- Service time s_i at each node (hours) ---
s = {i: 0.5 for i in N}

# --- Inventory parameters ---
I_init = {i: 50  for i in N}   # initial stock at t=0
I_max  = {i: 200 for i in N}   # maximum storage capacity
I_min  = {i: 10  for i in N}   # minimum safety stock
h      = {i: 0.5 for i in N}   # unit holding cost (currency/unit/period)

# --- Vehicle capacity Q_k (units) ---
Q = {1: 100, 2: 80}

# --- Cost bounds ---
C_max = 8000    # f1 <= C_max  (c12)
E_max = 5000    # f2 <= E_max  (c13)
B     = 10000   # f4 <= B      (c14)

# --- Financial parameters ---
BFR        = 3000
DSO        = 30                 # customer payment delay (days)
DPO        = 45                 # supplier payment delay (days)
V_val      = {i: 100 for i in N}
P_sale     = 8.0
P_purchase = 5.0

# --- Time window penalty coefficients ---
c1 = 2.0    # early arrival cost
c2 = 5.0    # late arrival cost

# --- CO2 parameters — Bektas & Laporte (2011) CMEM model ---
alpha_co2 = {(i, j): 0.01 for (i, j) in A}   # energy coefficient related to load
beta_co2  = 0.002                              # energy coefficient related to speed
e_co2     = 2.68                               # CO2 conversion factor (kg CO2/energy unit)

# --- Refrigeration parameters (k=1 only) ---
p5      = 3.0    # refrigeration cost per unit time (currency/h)
e_stock = 0.1    # energy consumed per stored unit
alpha_r = 0.05   # energy coefficient (stock -> consumption)


# ═════════════════════════════════════════════════════════════════════════════
# DECISION VARIABLES
# ═════════════════════════════════════════════════════════════════════════════

# x[i,j,t,k] : 1 if vehicle k uses arc (i,j) at period t, else 0
x = {
    (i, j, t, k): mdl.binary_var(name=f"x_{i}_{j}_{t}_{k}")
    for (i, j) in A for t in T for k in M
}

# f[i,j,t,k] : load carried by vehicle k on arc (i,j) at period t (units)
f = {
    (i, j, t, k): mdl.continuous_var(lb=0, name=f"f_{i}_{j}_{t}_{k}")
    for (i, j) in A for t in T for k in M
}

# q_prime[l,t] : quantity actually delivered to customer l at period t (units)
q_prime = {
    (l, t): mdl.continuous_var(lb=0, name=f"qprime_{l}_{t}")
    for l in clients for t in T
}

# tau[i,t] : arrival time at node i at period t (hours)
tau = {
    (i, t): mdl.continuous_var(lb=0, name=f"tau_{i}_{t}")
    for i in N for t in T
}

# I_var[i,t] : stock level at node i at end of period t — N\{D}
I_var = {
    (i, t): mdl.continuous_var(lb=0, name=f"I_{i}_{t}")
    for i in stock_nodes for t in T
}


# ═════════════════════════════════════════════════════════════════════════════
# CONSTRAINTS
# ═════════════════════════════════════════════════════════════════════════════

BIG_M = 99999


# ── 6.1 Network flow ─────────────────────────────────────────────────────────

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

# (c2b) No direct arc O->D : vehicle must visit at least one customer
for k in M:
    for t in T:
        mdl.add_constraint(x[O, D, t, k] == 0, ctname=f"c2b_k{k}_t{t}")

# (c3) Flow conservation : inflow - outflow = -1 (O), 0 (transit), +1 (D)
for k in M:
    for t in T:
        for j in N:
            inflow  = mdl.sum(x[i, j, t, k] for i in N if i != j)
            outflow = mdl.sum(x[j, i, t, k] for i in N if i != j)
            if j == O:
                mdl.add_constraint(inflow - outflow == -1, ctname=f"c3_O_k{k}_t{t}")
            elif j == D:
                mdl.add_constraint(inflow - outflow ==  1, ctname=f"c3_D_k{k}_t{t}")
            else:
                mdl.add_constraint(inflow - outflow ==  0, ctname=f"c3_{j}_k{k}_t{t}")


# ── 6.2 Capacity and inventory ───────────────────────────────────────────────

# (c4) Load on arc (i,j) cannot exceed vehicle capacity Q_k
for k in M:
    for (i, j) in A:
        for t in T:
            mdl.add_constraint(
                f[i, j, t, k] <= Q[k] * x[i, j, t, k],
                ctname=f"c4_{i}{j}_k{k}_t{t}"
            )

# (c5a) Stock bounds : I_min <= I_it <= I_max  for all i in N\{D}, t in T
for i in stock_nodes:
    for t in T:
        mdl.add_constraint(I_var[i, t] >= I_min[i], ctname=f"c5a_min_{i}_t{t}")
        mdl.add_constraint(I_var[i, t] <= I_max[i], ctname=f"c5a_max_{i}_t{t}")

# (c5b) Inventory balance for all t in T
#   i = O      : I_Ot = I_O,t-1 - shipped
#   i in clients: I_lt = I_l,t-1 + q'_lt - q_lt
for t in T:
    shipped    = mdl.sum(f[O, j, t, k] for j in N if j != O for k in M)
    stock_prev = I_var[O, t-1] if t > 1 else I_init[O]
    mdl.add_constraint(
        I_var[O, t] == stock_prev - shipped,
        ctname=f"c5b_O_t{t}"
    )
    for l in clients:
        stock_prev = I_var[l, t-1] if t > 1 else I_init[l]
        mdl.add_constraint(
            I_var[l, t] == stock_prev + q_prime[l, t] - q_lt[l, t],
            ctname=f"c5b_{l}_t{t}"
        )

# (c6) Total delivered quantity equals total demand over horizon
for l in clients:
    mdl.add_constraint(
        mdl.sum(q_prime[l, t] for t in T) == mdl.sum(q_lt[l, t] for t in T),
        ctname=f"c6_l{l}"
    )

# (c7) Inbound flow at customer l = delivered quantity q'_lt
for l in clients:
    for t in T:
        inbound = mdl.sum(f[i, l, t, k] for i in N if i != l for k in M)
        mdl.add_constraint(
            inbound == q_prime[l, t],
            ctname=f"c7_{l}_t{t}"
        )


# ── 6.3 Time and time windows ────────────────────────────────────────────────

# (c8) Arc travel time lower bound — inactive if arc not used (Big-M)
for k in M:
    for (i, j) in A:
        for t in T:
            mdl.add_constraint(
                tau[j, t] >= d[i, j] / v[k] - BIG_M * (1 - x[i, j, t, k]),
                ctname=f"c8_{i}{j}_k{k}_t{t}"
            )

# (c9) Arrival time propagation : tau_jt >= tau_it + s_i + d_ij/v^k
for k in M:
    for (i, j) in A:
        for t in T:
            mdl.add_constraint(
                tau[j, t] >= tau[i, t] + s[i] + d[i, j] / v[k] - BIG_M * (1 - x[i, j, t, k]),
                ctname=f"c9_{i}{j}_k{k}_t{t}"
            )

# (c10) Global route time window : tau_min <= tau_Dt <= tau_max
for t in T:
    mdl.add_constraint(tau[D, t] >= tau_min, ctname=f"c10_min_t{t}")
    mdl.add_constraint(tau[D, t] <= tau_max, ctname=f"c10_max_t{t}")

# (c11) Customer time windows : ET_lt <= tau_lt <= LT_lt
for l in clients:
    for t in T:
        mdl.add_constraint(tau[l, t] >= ET[l, t], ctname=f"c11_ET_l{l}_t{t}")
        mdl.add_constraint(tau[l, t] <= LT[l, t], ctname=f"c11_LT_l{l}_t{t}")


# ── 6.4 Economic / environmental bounds ──────────────────────────────────────
# (c12) f1 <= C_max  — added after f1_expr is defined below
# (c13) f2 <= E_max  — added after f2_expr is defined below
# (c14) f4 <= B      — added after f4_expr is defined below


# ═════════════════════════════════════════════════════════════════════════════
# OBJECTIVE FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

# ── f1 : Logistics cost = y1 (transport) + y2 (storage) + y3 (penalty) ──────

# y1 : transport cost — linear via arc loads f
y1 = mdl.sum(
    c_ijk[i, j, k] * d[i, j] * f[i, j, t, k]
    for (i, j) in A for t in T for k in M
)

# y2 : storage cost — h_i * I_it  for all i in N\{D}
y2 = mdl.sum(
    h[i] * I_var[i, t]
    for i in stock_nodes for t in T
)

# y3 : time window penalty — linearized via w1 (early) and w2 (late)
# w1[l,t] = max(ET_lt - tau_lt, 0),  w2[l,t] = max(tau_lt - LT_lt, 0)
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

# (c12)
mdl.add_constraint(f1_expr <= C_max, ctname="c12_logistics_budget")


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

print(f"Model          : {mdl.name}")
print(f"Nodes          : {len(N)}  (depot + {len(clients)} customers + destination)")
print(f"Stock nodes    : {stock_nodes}")
print(f"Arcs           : {len(A)}")
print(f"Periods        : {len(T)}")
print(f"Vehicle types  : {len(M)}")
print(f"Variables      : {mdl.number_of_variables}")
print(f"Constraints    : {mdl.number_of_constraints}")