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
clients     = [1, 2, 3]         # customer nodes  (l ∈ clients ⊂ N)
stock_nodes = [0, 1, 2, 3]      # nodes with stock : N \ {D}
O           = 0                  # origin depot
D           = 4                  # destination node
T           = [1, 2]             # planning periods
M           = [1, 2]             # vehicle types : 1=refrigerated, 2=standard

A = [(i, j) for i in N for j in N if i != j]   # all valid arcs


# ═════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ═════════════════════════════════════════════════════════════════════════════

# --- Customer demand q_lt : demand of customer l at period t (units) ---
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
# k=1 (refrigerated) : higher unit cost due to cooling equipment
# k=2 (standard)     : base unit cost
c_ijk = {
    (i, j, k): 1.2 if k == 1 else 1.0
    for (i, j) in A for k in M
}

# --- Customer time windows [ET_lt, LT_lt] (hours) ---
ET = {(l, t): 8  for l in clients for t in T}   # earliest accepted arrival
LT = {(l, t): 12 for l in clients for t in T}   # latest accepted arrival

# --- Global route time bounds (hours) ---
tau_min = 6.0
tau_max = 24.0

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
C_max = 8000    # maximum allowed logistics cost  (c12 : f1 <= C_max)
E_max = 5000    # maximum allowed carbon cost     (c13 : f2 <= E_max)

# --- Financial parameters ---
B          = 10000              # total financial budget  (c14 : f4 <= B)
BFR        = 3000               # working capital requirement
DSO        = 30                 # customer payment delay (days)
DPO        = 45                 # supplier payment delay (days)
V_val      = {i: 100 for i in N}
P_sale     = 8.0
P_purchase = 5.0

# --- Time window penalty coefficients ---
c1 = 2.0    # unit cost : early arrival  (cargo waits → storage cost)
c2 = 5.0    # unit cost : late arrival   (penalty cost)

# --- CO2 parameters — Bektas & Laporte (2011) CMEM model ---
alpha_co2 = {(i, j): 0.01 for (i, j) in A}   # energy coefficient related to load
beta_co2  = 0.002                              # energy coefficient related to speed
e_co2     = 2.68                               # CO2 conversion factor (kg CO2/energy unit)

# --- Refrigeration parameters (k=1 only) ---
p5      = 3.0    # refrigeration cost per unit time during transport (currency/h)
e_stock = 0.1    # energy consumed per stored unit
alpha_r = 0.05   # energy coefficient (stock level → energy consumption)


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

# I_var[i,t] : stock level at node i at end of period t — defined for N\{D}
I_var = {
    (i, t): mdl.continuous_var(lb=0, name=f"I_{i}_{t}")
    for i in stock_nodes for t in T
}


# ═════════════════════════════════════════════════════════════════════════════
# CONSTRAINTS
# ═════════════════════════════════════════════════════════════════════════════

BIG_M = 500   # Big-M constant for time propagation linearization


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

# (c5a) Stock bounds : I_min <= I_it <= I_max  ∀i ∈ N\{D}, ∀t ∈ T
for i in stock_nodes:
    for t in T:
        mdl.add_constraint(I_var[i, t] >= I_min[i], ctname=f"c5a_min_{i}_t{t}")
        mdl.add_constraint(I_var[i, t] <= I_max[i], ctname=f"c5a_max_{i}_t{t}")

# (c5b) Inventory balance  ∀t ∈ T
#   i = O      : I_Ot = I_O,t-1 - shipped
#   i ∈ clients: I_lt = I_l,t-1 + q'_lt - q_lt
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

# (c7) Load balance at customer l : inbound - outbound = q'_lt
for l in clients:
    for t in T:
        inbound  = mdl.sum(f[i, l, t, k] for i in N if i != l for k in M)
        outbound = mdl.sum(f[l, j, t, k] for j in N if j != l for k in M)
        mdl.add_constraint(
            inbound - outbound == q_prime[l, t],
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
# Activated after objective expressions are defined in the objectives section.
# (c12)  f1 <= C_max
# (c13)  f2 <= E_max
# (c14)  f4 <= B


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