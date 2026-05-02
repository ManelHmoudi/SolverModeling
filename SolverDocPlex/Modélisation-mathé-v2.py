from docplex.mp.model import Model

# ─────────────────────────────────────────────────────────────────────────────
# Many-Objective Inventory Routing Problem (IRP)
# ─────────────────────────────────────────────────────────────────────────────

mdl = Model(name="IRP_ManyObjective")

# ═════════════════════════════════════════════════════════════════════════════
# SETS
# ═════════════════════════════════════════════════════════════════════════════

N       = [0, 1, 2, 3, 4]   # nodes : 0 = origin depot, 1-3 = customers, 4 = destination
clients = [1, 2, 3]          # customer nodes
O       = 0                  # origin
D       = 4                  # destination
T       = [1, 2]             # planning periods
M       = [1, 2]             # vehicle types : 1 = refrigerated, 2 = non-refrigerated

A = [(i, j) for i in N for j in N if i != j]   # valid arcs

# ═════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ═════════════════════════════════════════════════════════════════════════════

# --- Customer demand (units) ---
q_lt = {
    (1, 1): 20, (1, 2): 25,
    (2, 1): 30, (2, 2): 35,
    (3, 1): 15, (3, 2): 20
}

# --- Vehicle speed (km/h) ---
v = {1: 80, 2: 60}

# --- Arc distances (km) : placeholder formula, replace with real data ---
d = {(i, j): abs(i - j) * 10 for (i, j) in A}

# --- Unit transport cost per vehicle type on each arc ---
c_ijk = {
    (i, j, k): d[(i, j)] * (1.2 if k == 1 else 1.0)
    for (i, j) in A for k in M
}

# --- Time window parameters ---
ET = {(l, t): 8  for l in clients for t in T}   # earliest accepted arrival (h)
LT = {(l, t): 12 for l in clients for t in T}   # latest accepted arrival (h)

tau_min = 6.0    # minimum total travel time O -> D (h)
tau_max = 24.0   # maximum total travel time O -> D (h)

s = {i: 0.5 for i in N}   # service/unloading time at each node (h)

# --- Inventory parameters ---
I_init = {i: 50  for i in N}    # initial stock at t=0
I_max  = {i: 200 for i in N}    # maximum storage capacity
I_min  = {i: 10  for i in N}    # minimum safety stock
h      = {i: 0.5 for i in N}    # unit holding cost (per unit per period)

# --- Vehicle capacity (units) ---
Q = {1: 100, 2: 80}

# --- Cost bounds ---
C_max = 8000    # maximum allowed logistics cost (budget constraint on f1)
E_max = 5000    # maximum allowed total carbon cost (constraint on f2)

# --- Financial parameters ---
B            = 10000   # total financial budget
BFR          = 3000    # working capital requirement
DSO          = 30      # customer payment delay (days)
DPO          = 45      # supplier payment delay (days)
V_val        = {i: 100 for i in N}
P_sale       = 8.0
P_purchase   = 5.0

# --- Penalty / inventory cost coefficients ---
c1 = 2.0    # unit cost : cargo arrives earlier than requested (waiting)
c2 = 5.0    # unit cost : cargo arrives later than requested (penalty)

# --- CO2 / environmental parameters (Bektas & Laporte, 2011) ---
alpha_co2 = {(i, j): 0.01 for (i, j) in A}   # energy coefficient related to load
beta_co2  = 0.002                              # energy coefficient related to speed
e_co2     = 2.68                               # CO2 conversion factor (kg CO2 / energy unit)

# --- Refrigeration parameters (vehicle k=1 only) ---
p5      = 3.0    # refrigeration cost per unit time during transport (currency/h)
e_stock = 0.1    # energy consumed per stored unit
alpha_r = 0.05   # energy coefficient (stock level -> consumption)


# ═════════════════════════════════════════════════════════════════════════════
# DECISION VARIABLES
# ═════════════════════════════════════════════════════════════════════════════

# x[i,j,t,k] : 1 if vehicle k traverses arc (i,j) at period t, else 0
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

# tau[i,t] : arrival time at node i during period t (hours)
tau = {
    (i, t): mdl.continuous_var(lb=0, name=f"tau_{i}_{t}")
    for i in N for t in T
}

# I_var[i,t] : stock level at customer i at the end of period t (units)
I_var = {
    (i, t): mdl.continuous_var(lb=0, name=f"I_{i}_{t}")
    for i in clients for t in T
}


# ═════════════════════════════════════════════════════════════════════════════
# CONSTRAINTS
# ═════════════════════════════════════════════════════════════════════════════

BIG_M = 99999   # linearization constant for time propagation constraints


# ── 5.1 Network flow ─────────────────────────────────────────────────────────

# (c1) Each vehicle departs from origin exactly once per period
for k in M:
    for t in T:
        mdl.add_constraint(
            mdl.sum(x[O, j, t, k] for j in N if j != O) == 1,
            ctname=f"c1_k{k}_t{t}"
        )

# (c2) Each vehicle arrives at destination exactly once per period
for k in M:
    for t in T:
        mdl.add_constraint(
            mdl.sum(x[i, D, t, k] for i in N if i != D) == 1,
            ctname=f"c2_k{k}_t{t}"
        )

# (c3) Flow conservation at each node (inflow = outflow for transit nodes)
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


# ── 5.2 Capacity and inventory ───────────────────────────────────────────────

# (c4) Load on any arc cannot exceed vehicle capacity
for k in M:
    for (i, j) in A:
        for t in T:
            mdl.add_constraint(
                f[i, j, t, k] <= Q[k] * x[i, j, t, k],
                ctname=f"c4_{i}{j}_k{k}_t{t}"
            )

# (c5a) Stock bounds : safety stock <= I_var <= max capacity
for l in clients:
    for t in T:
        mdl.add_constraint(I_var[l, t] >= I_min[l], ctname=f"c5a_min_{l}_t{t}")
        mdl.add_constraint(I_var[l, t] <= I_max[l], ctname=f"c5a_max_{l}_t{t}")

# (c5b) Inventory balance : I(t) = I(t-1) + delivered(t) - demand(t)
for l in clients:
    for t in T:
        stock_prev = I_var[l, t - 1] if t > 1 else I_init[l]
        mdl.add_constraint(
            I_var[l, t] == stock_prev + q_prime[l, t] - q_lt[l, t],
            ctname=f"c5b_{l}_t{t}"
        )

# (c6) Total delivered quantity over horizon equals total demand
for l in clients:
    mdl.add_constraint(
        mdl.sum(q_prime[l, t] for t in T) == mdl.sum(q_lt[l, t] for t in T),
        ctname=f"c6_l{l}"
    )

# (c7) Load balance at each customer node : inbound - outbound = delivered quantity
for j in clients:
    for t in T:
        inbound  = mdl.sum(f[i, j, t, k] for i in N if i != j for k in M)
        outbound = mdl.sum(f[j, l, t, k] for l in N if l != j for k in M)
        mdl.add_constraint(
            inbound - outbound == q_prime[j, t],
            ctname=f"c7_{j}_t{t}"
        )


# ── 5.3 Time and time windows ────────────────────────────────────────────────

# (c8) Arc travel time lower bound (active only if arc is used)
for k in M:
    for (i, j) in A:
        for t in T:
            mdl.add_constraint(
                tau[j, t] >= d[i, j] / v[k] - BIG_M * (1 - x[i, j, t, k]),
                ctname=f"c8_{i}{j}_k{k}_t{t}"
            )

# (c9) Arrival time propagation along the route (includes service time at i)
for k in M:
    for (i, j) in A:
        for t in T:
            mdl.add_constraint(
                tau[j, t] >= tau[i, t] + s[i] + d[i, j] / v[k] - BIG_M * (1 - x[i, j, t, k]),
                ctname=f"c9_{i}{j}_k{k}_t{t}"
            )

# (c10) Total route duration must respect contract time window
for t in T:
    mdl.add_constraint(tau[D, t] >= tau_min, ctname=f"c10_min_t{t}")
    mdl.add_constraint(tau[D, t] <= tau_max, ctname=f"c10_max_t{t}")

# (c11) Customer time windows : arrival must fall within [ET, LT]
for l in clients:
    for t in T:
        mdl.add_constraint(tau[l, t] >= ET[l, t], ctname=f"c11_ET_l{l}_t{t}")
        mdl.add_constraint(tau[l, t] <= LT[l, t], ctname=f"c11_LT_l{l}_t{t}")


# ── 5.4 Economic / environmental bounds (activated once objectives are defined) ──

# (c12)  f1 <= C_max   (logistics cost budget)
# (c13)  f2 <= E_max   (carbon emissions cap)
# (c14)  f4 <= B       (financial budget)
# These are commented out here and must be added after the objective expressions
# f1_expr, f2_expr, f4_expr are built in the objectives section below.


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

print(f"Model          : {mdl.name}")
print(f"Nodes          : {len(N)}  (origin + {len(clients)} customers + destination)")
print(f"Arcs           : {len(A)}")
print(f"Periods        : {len(T)}")
print(f"Vehicle types  : {len(M)}")
print(f"Variables      : {mdl.number_of_variables}")
print(f"Constraints    : {mdl.number_of_constraints}")