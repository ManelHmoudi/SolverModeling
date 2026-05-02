from docplex.mp.model import Model

# Initialize the many-objective Inventory Routing Problem model
mdl = Model(name="IRP_ManyObjective")

# =============================================================================
# SETS
# =============================================================================
# N : all network nodes (0=Origin depot, 1-3=customers, 4=Destination)
N       = [0, 1, 2, 3, 4]
clients = [1, 2, 3]          # one customer per node
O       = 0                  # origin node
D       = 4                  # destination node
T       = [1, 2]             # planning periods
M       = [1, 2]             # vehicle types: 1=refrigerated, 2=non-refrigerated

# A : set of all valid arcs (i,j) where i != j
A = [(i, j) for i in N for j in N if i != j]

# =============================================================================
# 3.1 CUSTOMER PARAMETERS
# =============================================================================
# phi_l : customer identifier
phi = {1: "customer_1", 2: "customer_2", 3: "customer_3"}

# q_lt : demand of customer l at period t
q_lt = {
    (1, 1): 20, (1, 2): 25,
    (2, 1): 30, (2, 2): 35,
    (3, 1): 15, (3, 2): 20
}

# q'_lt : actual quantity transported to customer l at period t
# planned based on the "small first, large later" service principle
q_prime_lt = {(l, t): q_lt[(l, t)] for l in clients for t in T}

# =============================================================================
# 3.2 TRANSPORT PARAMETERS
# =============================================================================
# v_k : constant speed of vehicle type k (km/h)
v = {1: 80, 2: 60}

# d_ij : distance between node i and node j (km)
d = {(i, j): abs(i - j) * 10 for (i, j) in A}

# c_ijk : unit transport cost of vehicle k on arc (i,j)
c_ijk = {
    (i, j, k): d[(i, j)] * (1.2 if k == 1 else 1.0)
    for (i, j) in A for k in M
}

# c1 : unit inventory cost when cargo arrives earlier than requested (waiting cost)
c1 = 2.0

# c2 : unit penalty cost when cargo arrives later than requested
c2 = 5.0

# =============================================================================
# 3.3 TIME PARAMETERS
# =============================================================================
# tau_lt : arrival time of cargo at customer l during period t (given/estimated)
tau_lt = {
    (1, 1): 9.0,  (1, 2): 10.0,
    (2, 1): 11.0, (2, 2): 13.0,
    (3, 1): 8.5,  (3, 2): 9.5
}

# [ET_lt, LT_lt] : time window accepted by customer l at period t
ET = {(l, t): 8  for l in clients for t in T}   # earliest acceptable arrival
LT = {(l, t): 12 for l in clients for t in T}   # latest acceptable arrival

# tau_min, tau_max : minimum and maximum allowed total travel time (O -> D)
tau_min = 6.0
tau_max = 24.0

# =============================================================================
# 3.4 INVENTORY PARAMETERS
# =============================================================================
# I_init : initial stock level at node i at period t (given data)
I_init = {(i, t): 50 for i in N for t in T}

# I_max_i : maximum storage capacity at node i
I_max = {i: 200 for i in N}

# I_min_i : minimum safety stock level at node i
I_min = {i: 10 for i in N}

# h_i : unit holding cost at node i (per unit per period)
h = {i: 0.5 for i in N}

# =============================================================================
# 3.5 CAPACITY PARAMETERS
# =============================================================================
# Q_k : maximum load capacity of vehicle type k
Q = {1: 100, 2: 80}

# C_max : maximum allowed total economic cost (budget constraint)
C_max = 8000

# =============================================================================
# 3.6 FINANCIAL PARAMETERS
# =============================================================================
# B : total financial budget
B = 10000

# BFR : working capital requirement (Besoin en Fonds de Roulement)
BFR = 3000

# DSO : customer payment delay (days)
DSO = 30

# DPO : supplier payment delay (days)
DPO = 45

# V_i : product value at node i
V_val = {i: 100 for i in N}

# P_sale, P_purchase : unit selling and purchasing prices
P_sale     = 8.0
P_purchase = 5.0

# =============================================================================
# 3.7 ENVIRONMENTAL (CO2) PARAMETERS
# =============================================================================
# alpha_ij : energy parameter related to cargo weight — Bektas & Laporte (2011)
alpha_co2 = {(i, j): 0.01 for (i, j) in A}

# beta : energy parameter related to vehicle speed
beta_co2 = 0.002

# e : CO2 conversion factor (kg CO2 per energy unit)
e_co2 = 2.68

# E_max : maximum allowed total carbon emission cost
E_max = 5000

# =============================================================================
# 3.8 REFRIGERATION ENERGY PARAMETERS (vehicle k=1 only)
# =============================================================================
# p5 : refrigeration cost per unit time during transport (yuan/h)
p5 = 3.0

# e_stock : energy consumption per stored unit
e_stock = 0.1

# alpha_r : energy coefficient (stock -> consumption)
alpha_r = 0.05

# =============================================================================
# DECISION VARIABLES
# =============================================================================
# x[i,j,t,k] : binary — 1 if vehicle k travels arc (i,j) at period t, 0 otherwise
# x_ijt^k ∈ {0,1}
x = {
    (i, j, t, k): mdl.binary_var(name=f"x_{i}_{j}_{t}_{k}")
    for (i, j) in A for t in T for k in M
}

# f[i,j,t,k] : continuous — load carried by vehicle k on arc (i,j) at period t
# f_ijt^k >= 0
f = {
    (i, j, t, k): mdl.continuous_var(lb=0, name=f"f_{i}_{j}_{t}_{k}")
    for (i, j) in A for t in T for k in M
}

# =============================================================================
# AUXILIARY VARIABLES
# =============================================================================
# tau[i,t] : arrival time at node i during period t
# required for time window constraints (c8, c9, c10, c11)
tau = {
    (i, t): mdl.continuous_var(lb=0, name=f"tau_{i}_{t}")
    for i in N for t in T
}

# I_var[i,t] : stock level at customer node i at end of period t
# computed by CPLEX via inventory balance constraint (c5b)
# appears in objective f1 as holding cost: h_i * I_var[i,t]
I_var = {
    (i, t): mdl.continuous_var(lb=0, name=f"I_{i}_{t}")
    for i in clients for t in T
}

# Summary
print("=" * 45)
print(f"  Model        : {mdl.name}")
print(f"  Nodes        : {len(N)} (1 origin, {len(clients)} customers, 1 destination)")
print(f"  Arcs         : {len(A)}")
print(f"  Periods      : {len(T)}")
print(f"  Vehicle types: {len(M)} (refrigerated / non-refrigerated)")
print("-" * 45)
print(f"  Decision variables x  : {len(x)}")
print(f"  Decision variables f  : {len(f)}")
print(f"  Auxiliary var  tau    : {len(tau)}")
print(f"  Auxiliary var  I_var  : {len(I_var)}")
print(f"  Total variables       : {mdl.number_of_variables}")
print("=" * 45)

# =============================================================================
# CONSTRAINTS
# =============================================================================

BIG_M = 99999  # large constant for Big-M linearization

# =============================================================================
# 5.1 NETWORK STRUCTURE AND FLOW CONSTRAINTS
# =============================================================================

# (c1) Each vehicle k departs from origin O exactly once per period t
# sum_{j in N} x_Ojt^k = 1   ∀k ∈ M, ∀t ∈ T
for k in M:
    for t in T:
        mdl.add_constraint(
            mdl.sum(x[O, j, t, k] for j in N if j != O) == 1,
            ctname=f"c1_k{k}_t{t}"
        )

# (c2) Each vehicle k arrives at destination D exactly once per period t
# sum_{i in N} x_iDt^k = 1   ∀k ∈ M, ∀t ∈ T
for k in M:
    for t in T:
        mdl.add_constraint(
            mdl.sum(x[i, D, t, k] for i in N if i != D) == 1,
            ctname=f"c2_k{k}_t{t}"
        )

# (c3) Flow conservation at each node j
# sum_i x_ijt^k - sum_i x_jit^k = -1 if j=O, 0 if j intermediate, +1 if j=D
for k in M:
    for t in T:
        for j in N:
            inflow  = mdl.sum(x[i, j, t, k] for i in N if i != j)
            outflow = mdl.sum(x[j, i, t, k] for i in N if i != j)
            if j == O:
                mdl.add_constraint(inflow - outflow == -1, ctname=f"c3_O_k{k}_t{t}")
            elif j == D:
                mdl.add_constraint(inflow - outflow == 1,  ctname=f"c3_D_k{k}_t{t}")
            else:
                mdl.add_constraint(inflow - outflow == 0,  ctname=f"c3_{j}_k{k}_t{t}")

# =============================================================================
# 5.2 CAPACITY AND LOGISTICS CONSTRAINTS
# =============================================================================

# (c4) Vehicle capacity : load on arc cannot exceed vehicle capacity
# f_ijt^k <= Q_k * x_ijt^k   ∀k, ∀(i,j) ∈ A, ∀t
# if vehicle k does not use arc (i,j), load is automatically forced to 0
for k in M:
    for (i, j) in A:
        for t in T:
            mdl.add_constraint(
                f[i, j, t, k] <= Q[k] * x[i, j, t, k],
                ctname=f"c4_{i}{j}_k{k}_t{t}"
            )

# (c5a) Stock bounds : safety stock <= I_var <= maximum capacity
# I_min_i <= I_it <= I_max_i   ∀i ∈ clients, ∀t ∈ T
for i in clients:
    for t in T:
        mdl.add_constraint(I_var[i, t] >= I_min[i], ctname=f"c5a_min_{i}_t{t}")
        mdl.add_constraint(I_var[i, t] <= I_max[i], ctname=f"c5a_max_{i}_t{t}")

# (c5b) Inventory balance : I_it = I_i,t-1 + inbound flow - demand
# I_it = I_i,t-1 + sum_{k,j} f_jit^k - q_lt
for i in clients:
    for t in T:
        inbound    = mdl.sum(f[j, i, t, k] for j in N if j != i for k in M)
        stock_prev = I_var[i, t - 1] if t > 1 else I_init[i, 1]
        mdl.add_constraint(
            I_var[i, t] == stock_prev + inbound - q_lt[i, t],
            ctname=f"c5b_{i}_t{t}"
        )

# (c6) Demand fulfillment : total transported quantity equals total demand
# sum_t q'_lt = sum_t q_lt   ∀l ∈ clients
for l in clients:
    mdl.add_constraint(
        mdl.sum(q_prime_lt[l, t] for t in T) == mdl.sum(q_lt[l, t] for t in T),
        ctname=f"c6_l{l}"
    )

# (c7) Load balance at each customer node j
# inbound load - outbound load = quantity delivered to j
# sum_{k,i} f_ijt^k - sum_{k,l} f_jlt^k = q'_jt
for j in clients:
    for t in T:
        inbound  = mdl.sum(f[i, j, t, k] for i in N if i != j for k in M)
        outbound = mdl.sum(f[j, l, t, k] for l in N if l != j for k in M)
        mdl.add_constraint(
            inbound - outbound == q_prime_lt[j, t],
            ctname=f"c7_{j}_t{t}"
        )

# =============================================================================
# 5.3 TIME AND TIME WINDOW CONSTRAINTS
# =============================================================================

# s_i : service time (unloading) at node i (hours)
s = {i: 0.5 for i in N}

# (c8) Arc travel time — active only if vehicle k uses arc (i,j)
# tau_jt >= d_ij / v_k - M*(1 - x_ijt^k)
for k in M:
    for (i, j) in A:
        for t in T:
            mdl.add_constraint(
                tau[j, t] >= d[i, j] / v[k] - BIG_M * (1 - x[i, j, t, k]),
                ctname=f"c8_{i}{j}_k{k}_t{t}"
            )

# (c9) Arrival time propagation along the route
# tau_jt >= tau_it + s_i + d_ij / v_k - M*(1 - x_ijt^k)
for k in M:
    for (i, j) in A:
        for t in T:
            mdl.add_constraint(
                tau[j, t] >= tau[i, t] + s[i] + d[i, j] / v[k] - BIG_M * (1 - x[i, j, t, k]),
                ctname=f"c9_{i}{j}_k{k}_t{t}"
            )

# (c10) Global time window : total travel time O -> D must stay within contract bounds
# tau_min <= tau_Dt <= tau_max
for t in T:
    mdl.add_constraint(tau[D, t] >= tau_min, ctname=f"c10_min_t{t}")
    mdl.add_constraint(tau[D, t] <= tau_max, ctname=f"c10_max_t{t}")

# (c11) Customer time windows : arrival must fall within accepted time window
# ET_lt <= tau_lt <= LT_lt   ∀l ∈ clients, ∀t ∈ T
for l in clients:
    for t in T:
        mdl.add_constraint(tau[l, t] >= ET[l, t], ctname=f"c11_ET_l{l}_t{t}")
        mdl.add_constraint(tau[l, t] <= LT[l, t], ctname=f"c11_LT_l{l}_t{t}")

# =============================================================================
# 5.4 ECONOMIC, FINANCIAL AND ENVIRONMENTAL CONSTRAINTS
# =============================================================================
# (c12) Economic budget  : f1 <= C_max  — activated after objective f1 is defined
# (c13) Carbon limit     : f2 <= E_max  — activated after objective f2 is defined
# (c14) Financial budget : f4 <= B      — activated after objective f4 is defined
# mdl.add_constraint(f1_expr <= C_max, ctname="c12_economic_budget")
# mdl.add_constraint(f2_expr <= E_max, ctname="c13_carbon_limit")
# mdl.add_constraint(f4_expr <= B,     ctname="c14_financial_budget")

print(f"  Constraints added     : {mdl.number_of_constraints}")
print("=" * 45)