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
v = {1: 80, 2: 60}  
v_ms = {k: v[k] / 3.6 for k in M}        # conversion km/h → m/s
v2   = {k: v_ms[k]**2 for k in M}         # speed m²/s
Q = {1: 100, 2: 80}         # capacity (units)

# ── 2.3 Network ───────────────────────────────────────────────────────────────
d   = {(i,j): abs(i-j) * 10   for (i,j) in A}  # km  — use for travel time and cost
d_m = {(i,j): d[i,j]  * 1000  for (i,j) in A}  # metres — use only in CMEM (f2)
c_route = {(i, j): 1.0      for (i, j) in A}        # base route cost (currency/km)
# ── 2.4 Refrigeration (vehicle type k=1) ─────────────────────────────────────
p5        = 3.0     # refrigeration cost per unit time (currency/h)
e_stock   = 0.1     # energy consumed per stored unit
alpha_r   = 0.05    # energy coefficient (stock → consumption)

# ── 2.5 Transport unit cost c_ijk (currency/km) ──────────────────────────────
c_ijk = {
    (i, j, k): (
        c_route[i, j] + p5 * d[i, j] / v[k]  # 
        if k == 1
        else c_route[i, j]
    )
    for (i, j) in A for k in M
}

# ── 2.6 Time windows & service times ─────────────────────────────────────────
ET      = {(l, t):  1.0 for l in clients for t in T}
LT      = {(l, t): 3.0 for l in clients for t in T}
tau_min = 0.0
tau_max = 9_999.0
s       = {i: 0.5 for i in N}      # service time at node i (hours)

# ── 2.7 Inventory ────────────────────────────────────────────────────────────
I_init = {i: (200 if i == O else 50) for i in stock_nodes}
I_max  = {i: 200 for i in stock_nodes}
I_min  = {i:  10 for i in stock_nodes}

h_space = {i: 0.3 for i in stock_nodes}
h = {
    i: h_space[i] + alpha_r * e_stock if i in cold_nodes else h_space[i]
    for i in stock_nodes
}

# ── 2.8 CO2 (Bektas & Laporte 2011 CMEM) ────────────────────────────────────

# Paramètres physiques
g    = 9.81     # gravité (m/s²)
Cr   = 0.01     # résistance au roulement
Cd   = 0.7      # coefficient de traînée
A_f  = 5.0      # surface frontale (m²)
rho  = 1.2041   # densité de l'air (kg/m³)
a    = 0.0      # accélération (supposée nulle)
w    = 2500     # poids camion vide (kg)

# Coefficients dérivés
# alpha_ij = g*Cr*cos(0) + g*sin(0) + a = g*Cr  (pente=0, a=0)
alpha_co2 = {(i, j): g * Cr for (i, j) in A}

# beta = 0.5 * Cd * A * rho
beta_co2  = 0.5 * Cd * A_f * rho

# Facteur de conversion CO2 (kg CO2 / litre)
e_co2 = 2.32


# ── 2.9 Financial ─────────────────────────────────────────────────────────────
BFR        = 3_000
DSO        = 30                                 # customer payment delay (days)
DPO        = 45                                 # supplier payment delay (days)
V_val      = {i: 100 for i in N}
P_sale = {
    1: 8.0,   # client 1
    2: 12.5,   # client 2
    3: 7.8,   # client 3
}

P_purchase = {
    0: 5.0,   # dépôt
    1: 5.5,   # client 1
    2: 6.0,   # client 2
    3: 5.8,   # client 3
}

# ── 2.10 Penalty coefficients ────────────────────────────────────────────────
c1 = 2.0    # early-arrival penalty coefficient
c2 = 5.0    # late-arrival  penalty coefficient

# ── 2.11 Upper bounds ────────────────────────────────────────────────────────
C_max  = 4_000   # logistics cost budget   (c12)
E_max  = 300_000_000   # CO2 emissions budget    (c13)
T_max  = 1.2
B      = 3_500  # financial cost budget   (c14)
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

# Variable d'activation
u = {
    (k, t): mdl.binary_var(name=f"u_{k}_{t}")
    for k in M for t in T
}  

# q_prime[l,t]    : quantity delivered to customer l in period t (units)
q_prime = {
    (l, t): mdl.integer_var(lb=0, name=f"qprime_{l}_{t}")
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

# (c1) départ de O
for k in M:
    for t in T:
        mdl.add_constraint(
            mdl.sum(x[O, j, t, k] for j in N if j != O) == u[k, t],
            ctname=f"c1_k{k}_t{t}"
        )

# (c2) arrivée à D
for k in M:
    for t in T:
        mdl.add_constraint(
            mdl.sum(x[i, D, t, k] for i in N if i != D) == u[k, t],
            ctname=f"c2_k{k}_t{t}"
        )

# (c2b) pas d'arc direct O→D
for k in M:
    for t in T:
        mdl.add_constraint(x[O, D, t, k] == 0, ctname=f"c2b_k{k}_t{t}")

# (c3) Flow conservation — CONDITIONNÉ à u[k,t]
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
                mdl.add_constraint(inflow - outflow ==  0, ctname=f"c3_{j}_k{k}_t{t}")

# (c_min) au moins 1 véhicule actif par période
for t in T:
    mdl.add_constraint(
        mdl.sum(u[k, t] for k in M) >= 1,
        ctname=f"min_vehicle_t{t}"
    )

# ── 4.2 Capacity & inventory ──────────────────────────────────────────────────

# (c4)
for k in M:
    for (i, j) in A:
        for t in T:
            mdl.add_constraint(
                f[i, j, t, k] <= Q[k] * x[i, j, t, k],
                ctname=f"c4_{i}{j}_k{k}_t{t}"
            )

# (c5a)
for i in stock_nodes:
    for t in T:
        mdl.add_constraint(I_var[i, t] >= I_min[i], ctname=f"c5a_min_{i}_t{t}")
        mdl.add_constraint(I_var[i, t] <= I_max[i], ctname=f"c5a_max_{i}_t{t}")

# (c5b) — FIX: shipped limité aux clients
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

# (c6)
for l in clients:
    mdl.add_constraint(
        mdl.sum(q_prime[l, t] for t in T) == mdl.sum(q_lt[l, t] for t in T),
        ctname=f"c6_l{l}"
    )

# (c7)
for l in clients:
    for t in T:
        inbound  = mdl.sum(f[i, l, t, k] for i in N if i != l for k in M)
        outbound = mdl.sum(f[l, j, t, k] for j in N if j != l for k in M)
        mdl.add_constraint(
            inbound - outbound == q_prime[l, t],
            ctname=f"c7_{l}_t{t}"
        )
# ── 4.3 Time & time windows ───────────────────────────────────────────────────

for t in T:
    mdl.add_constraint(tau[O, t] == 0, ctname=f"tau_origin_t{t}")

# (c8)
for k in M:
    for (i, j) in A:
        for t in T:
            mdl.add_constraint(
                tau[j, t] >= tau[i, t] + s[i] + d[i, j] / v[k]
                             - BIG_M * (1 - x[i, j, t, k]),
                ctname=f"c8_{i}{j}_k{k}_t{t}"
            )

# (c9)
for t in T:
    mdl.add_constraint(tau[D, t] >= tau_min, ctname=f"c9_min_t{t}")
    mdl.add_constraint(tau[D, t] <= tau_max, ctname=f"c9_max_t{t}")


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
w1 = {(l,t): mdl.continuous_var(lb=0, name=f"w1_{l}_{t}") for l in clients for t in T}
w2 = {(l,t): mdl.continuous_var(lb=0, name=f"w2_{l}_{t}") for l in clients for t in T}

for l in clients:
    for t in T:
        mdl.add_constraint(w1[l,t] >= ET[l,t] - tau[l,t], ctname=f"w1_{l}_t{t}")
        mdl.add_constraint(w2[l,t] >= tau[l,t] - LT[l,t], ctname=f"w2_{l}_t{t}")

y3 = mdl.sum(
     c1 * w1[l,t] + c2 * w2[l,t]
    for l in clients for t in T
)

f1_expr = y1 + y2 + y3

# ── f2 : Carbon cost (CMEM) — version linéaire ───────────────────────────────
f2_expr = e_co2 * mdl.sum(
    (
        alpha_co2[i, j] * w * d_m[i, j] * x[i, j, t, k]
      + alpha_co2[i, j] * d_m[i, j] * f[i, j, t, k]
      + beta_co2 * v2[k] * d_m[i,j] * x[i, j, t, k]
    )
    for (i, j) in A for t in T for k in M
)

# ── f3 : Total travel time — sum of travel time on all used arcs
#         x[i,j,t,k] * d[i,j]/v[k] gives travel time on arc (i,j)
#         only when the arc is actually used (x=1), zero otherwise.
#         Units: km / (km/h) = hours
f3_expr = mdl.sum(
    x[i, j, t, k] * d[i, j] / v[k]
    for (i, j) in A for t in T for k in M
)

# ── f4 : BFR (Besoin en Fonds de Roulement) ──────────────────────────────────

# Composante 1 : valeur des stocks (argent immobilisé dans vos étagères)
stock_value = mdl.sum(I_var[i, t] * P_purchase[i] for i in stock_nodes for t in T)

# Composante 2 : créances clients (argent que les pharmacies/hôpitaux vous doivent)
receivables = mdl.sum(q_prime[l, t] * P_sale[l] * (DSO / 365) for l in clients for t in T)

# Le BFR total (sans dettes puisque vous gérez l'entrepôt en propre)
f4_expr = stock_value + receivables


# (c10) Logistics cost budget
mdl.add_constraint(f1_expr <= C_max, ctname="c10_logistics_budget")

# (c11) CO2 budget
mdl.add_constraint(f2_expr <= E_max, ctname="c11_carbon_budget")

# (c12) Budget BFR maximal
mdl.add_constraint(f4_expr <= B, ctname="c12_bfr_budget")

#(c13) time minimization
T_max = 1.2
mdl.add_constraint(f3_expr <= T_max, ctname="c13_time_budget")

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


# =============================================================================
# 7. CALIBRATION TESTS — solve each objective independently
# =============================================================================

def test_objective(name, expr):
    mdl.minimize(expr)
    sol = mdl.solve(log_output=False)
    if sol:
        print(f"✅ {name} = {expr.solution_value:.4f}")
        # Print which arcs are used — confirms routing is sensible
        used = [
            (i, j, t, k)
            for (i, j, t, k) in x
            if x[i, j, t, k].solution_value > 0.5
        ]
        print(f"   Arcs used : {used}")
        # Print deliveries
        deliveries = {
    (l,t): round(q_prime[l,t].solution_value, 6)
    for (l,t) in q_prime
         }
        print(f"   Deliveries: {deliveries}")
        return expr.solution_value
    else:
        print(f"❌ {name} : infeasible — {mdl.solve_details}")
        return None

print("\n── Test f1 (logistics cost) ──")
f1_val = test_objective("f1", f1_expr)
if f1_val:
    print(f"   → C_max set to {f1_val * 1.2:.4f}")

print("\n── Test f2 (CO2) ──")
f2_val = test_objective("f2", f2_expr)
if f2_val:
    print(f"   → E_max set to {f2_val * 1.2:.4f}")

print("\n── Test f3 (travel time) ──")
f3_val = test_objective("f3", f3_expr)
if f3_val:
    print(f"   → T_max set to {f3_val * 1.2:.4f}")

print("\n── Test f4 (BFR) ──")
f4_val = test_objective("f4", f4_expr)
if f4_val:
    print(f"   → B set to {f4_val * 1.2:.4f}")
    print(f"   Décomposition :")
    print(f"     Stock value  = {stock_value.solution_value:.4f}")
    print(f"     Receivables  = {receivables.solution_value:.4f}")