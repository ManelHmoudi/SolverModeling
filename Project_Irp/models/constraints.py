"""
Many-Objective Inventory Routing Problem (IRP)
===============================================
Constraints

"""


def add_routing_constraints(mdl, x, N, A, T, M, O, D):
    """
    C1 : each vehicle departs at most once from the depot per period.
    C2 : each vehicle that departs the depot must arrive at the destination exactly once.
    C3 : for every non-depot, non-destination node: inflow equals outflow.
    """
    for k in M:
        for t in T:
            # C1 -- departure from depot
            mdl.add_constraint(
                mdl.sum(x[O, j, t, k] for j in N if j != O) <= 1,
                ctname=f"c1_k{k}_t{t}"
            )
            # C2 -- arrival at destination equals departure from depot
            mdl.add_constraint(
                mdl.sum(x[i, D, t, k] for i in N if i != D and i != O)
                == mdl.sum(x[O, j, t, k] for j in N if j != O),
                ctname=f"c2_k{k}_t{t}"
            )
            # C3 -- flow conservation at intermediate nodes
            for j in N:
                if j != O and j != D:
                    inflow  = mdl.sum(x[i, j, t, k] for i in N if i != j)
                    outflow = mdl.sum(x[j, i, t, k] for i in N if i != j)
                    mdl.add_constraint(inflow == outflow, ctname=f"c3_{j}_k{k}_t{t}")


def add_capacity_constraints(mdl, x, f, A, T, M, Q):
    """
    C4 : load on any arc cannot exceed the vehicle's capacity.
    """
    for k in M:
        for (i, j) in A:
            for t in T:
                mdl.add_constraint(
                    f[i, j, t, k] <= Q[k] * x[i, j, t, k],
                    ctname=f"c4_{i}{j}_k{k}_t{t}"
                )


def add_inventory_constraints(mdl, I_var, q_prime, f, stock_nodes, clients, T, M,
                               I_min, I_max, I_init, q_lt, O):
    """
    C5 : inventory at each stock node must stay within [I_min, I_max].
    C6 : depot inventory decreases by the total quantity shipped to clients;
         client inventory increases by delivered quantity and decreases by demand.
    C7 : total delivery equals total demand over the horizon.
    """
    # C5 -- inventory bounds
    for i in stock_nodes:
        for t in T:
            mdl.add_constraint(I_var[i, t] >= I_min[i], ctname=f"c5_min_{i}_t{t}")
            mdl.add_constraint(I_var[i, t] <= I_max[i], ctname=f"c5_max_{i}_t{t}")

    # C6 -- inventory balance
    for t in T:
        prev_O  = I_var[O, t - 1] if t > 1 else I_init[O]
        shipped = mdl.sum(f[O, j, t, k] for j in clients for k in M)
        mdl.add_constraint(I_var[O, t] == prev_O - shipped, ctname=f"c6_a_{t}")

        for t in T:
          for l in clients:
            prev_l = I_var[l, t - 1] if t > 1 else I_init[l]
            livré_en_t = mdl.sum(
              q_prime[l, t, td]
              for td in T if td >= t
            )
            mdl.add_constraint(
              I_var[l, t] == prev_l + livré_en_t - q_lt[l, t],
              ctname=f"c6_b_{l}_t{t}"
           )

    # C7 -- total delivery equals total demand
    for l in clients:
        for td in T:
            mdl.add_constraint(
                mdl.sum(q_prime[l, t, td] for t in T if t <= td) == q_lt[l, td],
                ctname=f"c7_{l}_td{td}"
            )


def add_flow_balance_constraints(mdl, f, q_prime, N, T, clients, requires_cold):
    """
    C8 : net inbound flow at each client equals the quantity delivered to that client.
    """
    for l in clients:
        for t in T:
            for td in T:
              if t <= td:
                    K_ltd = requires_cold[l, td]
                    inbound  = mdl.sum(
                        f[i, l, t, k] for i in N if i != l for k in K_ltd
                    )
                    outbound = mdl.sum(
                        f[l, j, t, k] for j in N if j != l for k in K_ltd
                    )
                    mdl.add_constraint(
                        inbound - outbound == q_prime[l, t, td],
                        ctname=f"c8_{l}_tl{t}_td{td}"
                    )


def add_time_constraints(mdl, x, tau, N, A, T, M, O, D,
                         s, d, v, tau_min, tau_max, BIG_M):
    """
    C9  : departure from the depot is fixed at time zero.
    C10 : if vehicle k uses arc (i,j) in period t, arrival at j is at least
          (arrival at i) + (service time at i) + (travel time on arc).
    C11 : destination arrival must lie within [tau_min, tau_max].
    """
    # C9 -- departure from depot fixed at time zero
    for t in T:
        mdl.add_constraint(tau[O, t] == 0, ctname=f"c9_{t}")

    # C10 -- arrival time propagation (big-M linearisation)
    for k in M:
        for (i, j) in A:
            for t in T:
                mdl.add_constraint(
                    tau[j, t] >= tau[i, t] + s[i] + d[i, j] / v[k]
                                 - BIG_M * (1 - x[i, j, t, k]),
                    ctname=f"c10_{i}{j}_k{k}_t{t}"
                )

    # C11 -- destination arrival window
    for t in T:
        mdl.add_constraint(tau[D, t] >= tau_min, ctname=f"c11_min_t{t}")
        mdl.add_constraint(tau[D, t] <= tau_max, ctname=f"c11_max_t{t}")


def add_vehicle_compatibility_constraints(mdl, x, N, T, clients, requires_cold):
    """
    C12 : a vehicle type incompatible with client l at period t
          cannot serve that client.
    """
    for l in clients:
      for t in T:
        types_requis_tl = set()
        for td in T:
            if td >= t:
                for k in requires_cold[l, td]:
                    types_requis_tl.add(k)

        if 1 not in types_requis_tl:
            mdl.add_constraint(
                mdl.sum(x[i, l, t, 1] for i in N if i != l) == 0,
                ctname=f"c12_no_frigo_{l}_tl{t}"
            )
        if 2 not in types_requis_tl:
            mdl.add_constraint(
                mdl.sum(x[i, l, t, 2] for i in N if i != l) == 0,
                ctname=f"c12_no_standard_{l}_tl{t}"
            )


def add_time_window_slacks(mdl, tau, w1, w2, clients, T, ET, LT):
    """
    Auxiliary constraints for time-window penalty slack variables.
        w1[l,t] >= ET[l,t] - tau[l,t]   (early arrival)
        w2[l,t] >= tau[l,t] - LT[l,t]   (late arrival)
    """
    for l in clients:
        for t in T:
            mdl.add_constraint(w1[l, t] >= ET[l, t] - tau[l, t], ctname=f"w1_{l}_t{t}")
            mdl.add_constraint(w2[l, t] >= tau[l, t] - LT[l, t], ctname=f"w2_{l}_t{t}")


def add_budget_constraints(mdl, f1_expr, f2_expr, f3_expr, f4_expr,
                            C_max, E_max, T_max, B):
    """
    C13-C16 : budget constraints (feasibility bounds on each objective).
    """
    mdl.add_constraint(f1_expr <= C_max, ctname="c13_logistics_budget")
    mdl.add_constraint(f2_expr <= E_max, ctname="c14_carbon_budget")
    mdl.add_constraint(f4_expr <= B,     ctname="c15_bfr_budget")
    mdl.add_constraint(f3_expr <= T_max, ctname="c16_time_budget")


def add_all_constraints(mdl, vars_, sets_, params_):
    """
    Single entry point: call all constraint-building functions.

    Parameters
    ----------
    mdl      : docplex Model
    vars_    : dict returned by build_variables()
    sets_    : dict with N, A, T, M, O, D, clients, stock_nodes
    params_  : dict with all model parameters;
               must include params_['objectives'] with pre-built f1..f4 expressions.
    """
    x       = vars_["x"]
    f       = vars_["f"]
    q_prime = vars_["q_prime"]
    tau     = vars_["tau"]
    I_var   = vars_["I_var"]
    w1      = vars_["w1"]
    w2      = vars_["w2"]

    N           = sets_["N"]
    A           = sets_["A"]
    T           = sets_["T"]
    M           = sets_["M"]
    O           = sets_["O"]
    D           = sets_["D"]
    clients     = sets_["clients"]
    stock_nodes = sets_["stock_nodes"]

    add_routing_constraints(mdl, x, N, A, T, M, O, D)
    add_capacity_constraints(mdl, x, f, A, T, M, params_["Q"])
    add_inventory_constraints(
        mdl, I_var, q_prime, f, stock_nodes, clients, T, M,
        params_["I_min"], params_["I_max"], params_["I_init"],
        params_["q_lt"], O
    )
    add_flow_balance_constraints(mdl, f, q_prime, N, T, clients, params_["requires_cold"])
    add_time_constraints(
        mdl, x, tau, N, A, T, M, O, D,
        params_["s"], params_["d"], params_["v"],
        params_["tau_min"], params_["tau_max"], params_["BIG_M"]
    )
    add_vehicle_compatibility_constraints(mdl, x, N, T, clients, params_["requires_cold"])
    add_time_window_slacks(mdl, tau, w1, w2, clients, T, params_["ET"], params_["LT"])

    # Budget constraints require pre-built objective expressions
    obj = params_["objectives"]
    add_budget_constraints(
        mdl,
        obj["f1"], obj["f2"], obj["f3"], obj["f4"],
        params_["C_max"], params_["E_max"], params_["T_max"], params_["B"]
    )