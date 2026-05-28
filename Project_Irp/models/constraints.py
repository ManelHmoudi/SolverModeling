"""
Many-Objective Inventory Routing Problem (IRP)
===============================================
Constraints C1 – C17  (as numbered in the model specification)

  6.1 Network structure & flow
      C1  : each vehicle departs the depot at most once per period
      C2  : each departing vehicle must return to the depot
      C3  : flow conservation at every non-depot node

  6.2 Capacity & logistics
      C4  : load on an arc cannot exceed the vehicle's capacity
      C5  : depot inventory stays within [I_O_min, I_O_max]
      C6  : depot inventory balance (previous + replenishment - shipped)
      C7  : total delivery over all periods equals the client's demand
      C8  : net inbound flow at a client equals the quantity delivered there

  6.3 Time & time windows
      C9  : vehicles depart the depot at time 0
      C10 : arrival time propagation via Big-M linearisation
      C11 : soft time-window — w1/w2 slacks penalised in f1
            (kept soft intentionally; a hard bound would make c1/c2 in f1 always zero)
      C12 : global tour window  T_min <= tau[O,t] <= T_max

  6.4 Vehicle compatibility
      C13 : vehicle types not in K_lt[l,t] cannot serve client l in period t

  6.5 Economic, financial & environmental budgets
      C14 : f1 <= C      (logistics cost budget)
      C15 : f2 <= E_max  (carbon emissions limit)
      C16 : f4 <= B      (working-capital / BFR budget)
      C17 : f3 <= T_max  (total travel-time limit)
"""


# ── 6.1  Network structure & flow ────────────────────────────────────────────

def add_routing_constraints(mdl, x, f, N, T, M, O):
    for k in M:
        for t in T:
            # C1 — each vehicle k departs depot O at most once per period t
            mdl.add_constraint(
                mdl.sum(x[O, j, t, k] for j in N if j != O) <= 1,
                ctname=f"c1_k{k}_t{t}"
            )

            # C2 — if vehicle k leaves the depot it must return to it
            mdl.add_constraint(
                mdl.sum(x[i, O, t, k] for i in N if i != O) ==
                mdl.sum(x[O, j, t, k] for j in N if j != O),
                ctname=f"c2_k{k}_t{t}"
            )

            # C3 — flow conservation: inflow == outflow at every non-depot node
            for j in N:
                if j != O:
                    inflow  = mdl.sum(x[i, j, t, k] for i in N if i != j)
                    outflow = mdl.sum(x[j, i, t, k] for i in N if i != j)
                    mdl.add_constraint(inflow == outflow, ctname=f"c3_j{j}_k{k}_t{t}")

            # C4 — a vehicle leaves the depot only if it carries at least one unit
            #prevents empty tours: x[O,j,t,k]=1 implies f[O,j,t,k] >= 1
            depart     = mdl.sum(x[O, j, t, k] for j in N if j != O)
            load_depot = mdl.sum(f[O, j, t, k] for j in N if j != O)
            mdl.add_constraint(
                load_depot >= depart,
                ctname=f"c4_k{k}_t{t}"
            )
           
# ── 6.2  Capacity & logistics ────────────────────────────────────────────────

def add_capacity_constraints(mdl, x, f, A, T, M, Q):
    # C5 — load on arc (i,j) by vehicle k in period t <= vehicle capacity Q[k]
    #      (automatically forces f=0 when arc is not used, since x=0 => f<=0)
    for k in M:
        for (i, j) in A:
            for t in T:
                mdl.add_constraint(
                    f[i, j, t, k] <= Q[k] * x[i, j, t, k],
                    ctname=f"c5_{i}{j}_k{k}_t{t}"
                )


def add_inventory_constraints(mdl, I_O, q_prime, f, clients, T, M,
                               I_O_min, I_O_max, I_O_init, q_lt, R, O):
    # C6 — depot inventory must stay within safety stock and storage capacity
    for t in T:
        mdl.add_constraint(I_O[t] >= I_O_min, ctname=f"c6_min_t{t}")
        mdl.add_constraint(I_O[t] <= I_O_max, ctname=f"c6_max_t{t}")

    # C7 — depot inventory balance: I_O[t] = I_O[t-1] + R[t] - shipped
    #      R[t] = total replenishment received at depot in period t
    #      shipped = sum of all loads departing from depot O in period t
    for t in T:
        prev_stock = I_O[t - 1] if t > 1 else I_O_init
        shipped    = mdl.sum(f[O, j, t, k] for j in clients for k in M)
        mdl.add_constraint(
            I_O[t] == prev_stock + R[t] - shipped,
            ctname=f"c7_t{t}"
        )

    # C8 — total quantity delivered to client l for demand period td = demand q_lt[l,td]
    #      deliveries can be split across periods t <= td (advance or on-time delivery)
    for l in clients:
        for td in T:
            mdl.add_constraint(
                mdl.sum(q_prime[l, t, td] for t in T if t <= td) == q_lt[l, td],
                ctname=f"c8_l{l}_td{td}"
            )


def add_flow_balance_constraints(mdl, f, q_prime, N, T, clients, requires_cold):
    # C9 — net inbound flow at client l in period t equals total quantity delivered
    for l in clients:
        for t in T:
            for td in T:
                if t <= td:
                    # On récupère la liste des camions compatibles pour la marchandise de la période td
                    # Exemple: [1] pour le frigo, [2] pour le non-frigo
                    K_ltd = requires_cold[l, td]
                    
                    inbound  = mdl.sum(f[i, l, t, k] for i in N if i != l for k in K_ltd)
                    outbound = mdl.sum(f[l, j, t, k] for j in N if j != l for k in K_ltd)
                    
                    mdl.add_constraint(
                        inbound - outbound == q_prime[l, t, td], 
                        ctname=f"c9_l{l}_t{t}_td{td}"
                    )


# ── 6.3  Time & time windows ─────────────────────────────────────────────────

def add_time_constraints(mdl, x, tau, tau_return,w1, w2, N, A, T, M, O,
                         s, d, v,ET, LT, tau_min, tau_max, BIG_M):
    # C10 — vehicles depart the depot at time 0 in every period
    for t in T:
        mdl.add_constraint(tau[O, t] == 0, ctname=f"c10_t{t}")

    # C11 — arrival time propagation (Big-M linearisation):
    #       if vehicle k uses arc (i,j) in period t then
    #       tau[j,t] >= tau[i,t] + s[i] + d[i,j]/v[k]
    #       Big-M term deactivates the constraint when x[i,j,t,k] = 0
    for k in M:
        for (i, j) in A:
            if j == O:          
                continue
            for t in T:
                mdl.add_constraint(
                    tau[j, t] >= tau[i, t] + s[i] + d[i, j] / v[k]
                                 - BIG_M * (1 - x[i, j, t, k]),
                    ctname=f"c11_{i}{j}_k{k}_t{t}"
                )
    # C11b — arrival time propagation back to the depot → tau_return[t]
    for k in M:
        for i in N:
            if i == O:
                continue
            for t in T:
                mdl.add_constraint(
                    tau_return[t] >= tau[i, t] + s[i] + d[i, O] / v[k]
                                     - BIG_M * (1 - x[i, O, t, k]),
                    ctname=f"c11b_{i}0_k{k}_t{t}"
                )
    # C12 — soft time window: ET[l,t] <= tau[l,t] <= LT[l,t]
    #       implemented via non-negative slack variables w1 (early) and w2 (late)
    #       penalised in objective f1 by c1*w1 + c2*w2
    #       kept soft so the solver can trade time-window violations against other objectives
    for l in [n for n in N if n != O]:
        for t in T:
          # V_lt
          visited = mdl.sum(
            x[i, l, t, k]
            for i in N if i != l
            for k in M
          )

          # tau[l,t] <= M * V_lt
          mdl.add_constraint(
            tau[l, t] <= BIG_M * visited,
            ctname=f"c12_activate_tau_l{l}_t{t}"
          )

          # advanced penality
          mdl.add_constraint(
            w1[l, t] >= ET[l, t] - tau[l, t] - BIG_M * (1 - visited),
            ctname=f"c12a_early_l{l}_t{t}"
          )

          # late penalty
          mdl.add_constraint(
            w2[l, t] >= tau[l, t] - LT[l, t] - BIG_M * (1 - visited),
            ctname=f"c12b_late_l{l}_t{t}"
          )
    # C13 — global tour window: depot return time must be within [T_min, T_max]
    for t in T:
        mdl.add_constraint(tau[O, t] >= tau_min, ctname=f"c13_min_t{t}")
        mdl.add_constraint(tau[O, t] <= tau_max, ctname=f"c13_max_t{t}")
    

# ── 6.4  Vehicle compatibility ───────────────────────────────────────────────

def add_vehicle_compatibility_constraints(mdl, x, N, T, clients, K_lt, M):
    # C14 — vehicle types not in K_lt[l,t] cannot serve client l in period t
    #       K_lt[l,t] is the subset of compatible vehicle types (e.g. [1] for cold-chain clients)
    for l in clients:
        for t in T:
            incompatible = [k for k in M if k not in K_lt[l, t]]
            if incompatible:
                mdl.add_constraint(
                    mdl.sum(x[i, l, t, k]
                            for i in N if i != l
                            for k in incompatible) == 0,
                    ctname=f"c14_l{l}_t{t}"
                )
    

# ── 6.5  Economic, financial & environmental budgets ────────────────────────

def add_budget_constraints(mdl, f1_expr, f2_expr, f3_expr, f4_expr,
                            C_max, E_max, T_max, B):
    # C15 — logistics cost budget: f1 <= C
    mdl.add_constraint(f1_expr <= C_max, ctname="c15_logistics_budget")

    # C16 — carbon emissions limit: f2 <= E_max
    mdl.add_constraint(f2_expr <= E_max, ctname="c16_carbon_budget")

    # C17 — working-capital (BFR) budget: f4 <= B
    mdl.add_constraint(f4_expr <= B,     ctname="c17_bfr_budget")

    # C18 — total travel-time limit: f3 <= T_max
    mdl.add_constraint(f3_expr <= T_max, ctname="c18_time_budget")


# ── Entry point ───────────────────────────────────────────────────────────────

def add_all_constraints(mdl, vars_, sets_, params_):
    x       = vars_["x"]
    f       = vars_["f"]
    q_prime = vars_["q_prime"]
    tau     = vars_["tau"]
    w1      = vars_["w1"]
    w2      = vars_["w2"]
    tau_return = vars_["tau_return"]
    I_O     = vars_["I_O"]

    N       = sets_["N"]
    A       = sets_["A"]
    T       = sets_["T"]
    M       = sets_["M"]
    O       = sets_["O"]
    clients = sets_["clients"]

    # C1 – C4
    add_routing_constraints(mdl, x, f, N, T, M, O)

    # C5
    add_capacity_constraints(mdl, x, f, A, T, M, params_["Q"])

    # C6 – C8
    add_inventory_constraints(
        mdl, I_O, q_prime, f, clients, T, M,
        params_["I_O_min"], params_["I_O_max"], params_["I_O_init"],
        params_["q_lt"], params_["R"], O
    )

    # C9
    add_flow_balance_constraints(mdl, f, q_prime, N, T, clients, params_["requires_cold"])

    # C10 – C13
    add_time_constraints(
        mdl, x, tau, vars_["tau_return"], w1, w2, N, A, T, M, O,
        params_["s"], params_["d"], params_["v"],
        params_["ET"], params_["LT"],
        params_["tau_min"], params_["tau_max"], params_["BIG_M"]
    )

    # C14
    add_vehicle_compatibility_constraints(mdl, x, N, T, clients, params_["K_lt"], M)

    