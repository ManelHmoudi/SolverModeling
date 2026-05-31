"""
Many-Objective Inventory Routing Problem (IRP)
===============================================
Constraints C1 – C18
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
            depart     = mdl.sum(x[O, j, t, k] for j in N if j != O)
            load_depot = mdl.sum(f[O, j, t, k] for j in N if j != O)
            mdl.add_constraint(
                load_depot >= depart,
                ctname=f"c4_k{k}_t{t}"
            )
           
# ── 6.2  Capacity & logistics ────────────────────────────────────────────────

def add_capacity_constraints(mdl, x, f, A, T, M, Q):
    # C5 — load on arc (i,j) by vehicle k in period t <= vehicle capacity Q[k]
    for k in M:
        for (i, j) in A:
            for t in T:
                mdl.add_constraint(
                    f[i, j, t, k] <= Q[k] * x[i, j, t, k],
                    ctname=f"c5_{i}{j}_k{k}_t{t}"
                )


def add_inventory_constraints(mdl, I_O_frigo, I_O_nonfrigo, q_prime, clients, T,
                               I_O_min_frigo, I_O_max_frigo, I_O_init_frigo,
                               I_O_min_nonfrigo, I_O_max_nonfrigo, I_O_init_nonfrigo,
                               q_lt, R_frigo, R_nonfrigo, requires_cold, frigo_trucks):
    # C6 — inventory bounds by product type (refrigerated / non-refrigerated)
    for t in T:
        mdl.add_constraint(I_O_frigo[t]    >= I_O_min_frigo,    ctname=f"c6f_min_t{t}")
        mdl.add_constraint(I_O_frigo[t]    <= I_O_max_frigo,    ctname=f"c6f_max_t{t}")
        mdl.add_constraint(I_O_nonfrigo[t] >= I_O_min_nonfrigo, ctname=f"c6nf_min_t{t}")
        mdl.add_constraint(I_O_nonfrigo[t] <= I_O_max_nonfrigo, ctname=f"c6nf_max_t{t}")

    # C7 — depot stock balance by product type (refrigerated / non-refrigerated)
    # Product type is determined by requires_cold[l, td]:
    # if the assigned vehicle is in frigo_trucks → refrigerated, otherwise → non-refrigerated
    for t in T:
        prev_frigo    = I_O_frigo[t - 1]    if t > 1 else I_O_init_frigo
        prev_nonfrigo = I_O_nonfrigo[t - 1] if t > 1 else I_O_init_nonfrigo

        shipped_frigo = mdl.sum(
            q_prime[l, t, td]
            for l in clients for td in T if t <= td
            and requires_cold[l, td][0] in frigo_trucks
        )
        shipped_nonfrigo = mdl.sum(
            q_prime[l, t, td]
            for l in clients for td in T if t <= td
            and requires_cold[l, td][0] not in frigo_trucks
        )

        mdl.add_constraint(
            I_O_frigo[t] == prev_frigo + R_frigo[t] - shipped_frigo,
            ctname=f"c7f_t{t}"
        )
        mdl.add_constraint(
            I_O_nonfrigo[t] == prev_nonfrigo + R_nonfrigo[t] - shipped_nonfrigo,
            ctname=f"c7nf_t{t}"
        )

    # C8 — total quantity delivered to client l for demand period td must equal q_lt[l,td]
    for l in clients:
        for td in T:
            mdl.add_constraint(
                mdl.sum(q_prime[l, t, td] for t in T if t <= td) == q_lt[l, td],
                ctname=f"c8_l{l}_td{td}"
            )


def add_no_empty_visits_constraints(mdl, x, f, N, T, M, clients):
    """
    STRICT RULE: A vehicle cannot visit a client (x=1) without delivering physical units (Net flow >= 1).
    Radically eliminates empty transit loops.
    """
    for l in clients:
        for t in T:
            for k in M:
                visited_by_k = mdl.sum(x[i, l, t, k] for i in N if i != l)
                inbound_f  = mdl.sum(f[i, l, t, k] for i in N if i != l)
                outbound_f = mdl.sum(f[l, j, t, k] for j in N if j != l)
                net_delivery = inbound_f - outbound_f
                
                mdl.add_constraint(
                    net_delivery >= visited_by_k,
                    ctname=f"force_real_delivery_l{l}_t{t}_k{k}"
                )
                

def add_flow_balance_constraints(mdl, f, q_prime, N, T, M, clients, requires_cold):
    for l in clients:
        for t in T:
            for k in M:
                inbound  = mdl.sum(f[i, l, t, k] for i in N if i != l)
                outbound = mdl.sum(f[l, j, t, k] for j in N if j != l)

                assigned = mdl.sum(
                    q_prime[l, t, td]
                    for td in T if t <= td
                    and k in requires_cold[l, td]
                )

                mdl.add_constraint(
                    inbound - outbound == assigned,
                    ctname=f"c9_l{l}_t{t}_k{k}"
                )


# ── 6.3  Time & time windows ─────────────────────────────────────────────────

def add_time_constraints(mdl, x, tau, tau_return, w1, w2, N, A, T, M, O,
                         s, d, v, ET, LT, tau_min, tau_max, BIG_M):
    # C10 — vehicles depart the depot at time 0 in every period
    for t in T:
        mdl.add_constraint(tau[O, t] == 0, ctname=f"c10_t{t}")

    # C11 — arrival time propagation (Big-M linearisation)
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

    # C11b — arrival time propagation back to the depot -> tau_return[t]
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

    # C12 — soft time window constraints
    for l in [n for n in N if n != O]:
        for t in T:
            visited = mdl.sum(
                x[i, l, t, k]
                for i in N if i != l
                for k in M
            )

            mdl.add_constraint(
                tau[l, t] <= BIG_M * visited,
                ctname=f"c12_activate_tau_l{l}_t{t}"
            )

            mdl.add_constraint(
                w1[l, t] >= ET[l, t] - tau[l, t] - BIG_M * (1 - visited),
                ctname=f"c12a_early_l{l}_t{t}"
            )

            mdl.add_constraint(
                w2[l, t] >= tau[l, t] - LT[l, t] - BIG_M * (1 - visited),
                ctname=f"c12b_late_l{l}_t{t}"
            )

    # C13 — global tour window with tight upper bound linearization
    for t in T:
        mdl.add_constraint(tau_return[t] >= tau_min, ctname=f"c13_ret_min_t{t}")
        mdl.add_constraint(tau_return[t] <= tau_max, ctname=f"c13_ret_max_t{t}")
        
        # Prevents tau_return from floating up to Big-M or tau_max when unconstrained
        for k in M:
            for i in N:
                if i == O:
                    continue
                mdl.add_constraint(
                    tau_return[t] <= tau[i, t] + s[i] + d[i, O] / v[k] + BIG_M * (1 - x[i, O, t, k]),
                    ctname=f"c13_upper_bound_tight_i{i}_k{k}_t{t}"
                )
    

# ── 6.4  Vehicle compatibility ───────────────────────────────────────────────

def add_vehicle_compatibility_constraints(mdl, x, N, T, clients, K_lt, M):
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
    mdl.add_constraint(f1_expr <= C_max, ctname="c15_logistics_budget")
    mdl.add_constraint(f2_expr <= E_max, ctname="c16_carbon_budget")
    mdl.add_constraint(f4_expr <= B,     ctname="c17_bfr_budget")
    mdl.add_constraint(f3_expr <= T_max, ctname="c18_time_budget")


# ── Entry point ───────────────────────────────────────────────────────────────

def add_all_constraints(mdl, vars_, sets_, params_):
    x       = vars_["x"]
    f       = vars_["f"]
    q_prime = vars_["q_prime"]
    tau     = vars_["tau"]
    w1      = vars_["w1"]
    w2      = vars_["w2"]
    I_O_frigo    = vars_["I_O_frigo"]
    I_O_nonfrigo = vars_["I_O_nonfrigo"]

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
        mdl, I_O_frigo, I_O_nonfrigo, q_prime, clients, T,
        params_["I_O_min_frigo"],    params_["I_O_max_frigo"],    params_["I_O_init_frigo"],
        params_["I_O_min_nonfrigo"], params_["I_O_max_nonfrigo"], params_["I_O_init_nonfrigo"],
        params_["q_lt"], params_["R_frigo"], params_["R_nonfrigo"],
        params_["requires_cold"], params_["frigo_trucks"],
    )

    # Empty visits restriction
    add_no_empty_visits_constraints(mdl, x, f, N, T, M, clients)

    # C9
    add_flow_balance_constraints(
        mdl, f, q_prime, N, T, M, clients, params_["requires_cold"]
    )

    # C10 – C13
    add_time_constraints(
        mdl, x, tau, vars_["tau_return"], w1, w2, N, A, T, M, O,
        params_["s"], params_["d"], params_["v"],
        params_["ET"], params_["LT"],
        params_["tau_min"], params_["tau_max"], params_["BIG_M"]
    )

    # C14
    add_vehicle_compatibility_constraints(mdl, x, N, T, clients, params_["K_lt"], M)