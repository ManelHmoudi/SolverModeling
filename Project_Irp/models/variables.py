"""
Many-Objective Inventory Routing Problem (IRP)
===============================================
Decision Variables
------------------
"""

def build_variables(mdl, N, A, T, M, clients):

    # ── Transport variables ───────────────────────────────────────────────────
    # x[i,j,t,k] : binary -- 1 if vehicle k travels arc (i,j) in period t
    x = {
        (i, j, t, k): mdl.binary_var(name=f"x_{i}_{j}_{t}_{k}")
        for (i, j) in A for t in T for k in M
    }

    # ── Load / delivery variables ─────────────────────────────────────────────
    # f[i,j,t,k] : continuous -- load (units) carried on arc (i,j) by vehicle k in period t
    f = {
        (i, j, t, k): mdl.continuous_var(lb=0, name=f"f_{i}_{j}_{t}_{k}")
        for (i, j) in A for t in T for k in M
    }

    # q_prime[l,t,td] : quantity delivered to customer l dispatched in period t for demand period td
    q_prime = {
        (l, t, td): mdl.continuous_var(lb=0, name=f"qprime_{l}_{t}_{td}")
        for l in clients
        for t in T
        for td in T
        if t <= td
    }

    # ── Auxiliary variables ───────────────────────────────────────────────────
    # tau[i,t] : continuous -- arrival time at node i in period t (hours)
    tau = {
        (i, t): mdl.continuous_var(lb=0, name=f"tau_{i}_{t}")
        for i in N for t in T
    }
    # tau_return[t] : effective depot return time in period t (hours)
    tau_return = {
    t: mdl.continuous_var(lb=0, name=f"tau_ret_{t}")
    for t in T
    }

    # I_O_frigo[t] : depot refrigerated product inventory at end of period t
    I_O_frigo = {
        t: mdl.continuous_var(lb=0, name=f"I_O_frigo_{t}")
        for t in T
    }

    # I_O_nonfrigo[t] : depot non-refrigerated product inventory at end of period t
    I_O_nonfrigo = {
        t: mdl.continuous_var(lb=0, name=f"I_O_nonfrigo_{t}")
        for t in T
    }

    # w1[l,t] / w2[l,t] : slack variables for early (w1) and late (w2) arrival penalties
    w1 = {
        (l, t): mdl.continuous_var(lb=0, name=f"w1_{l}_{t}")
        for l in clients for t in T
    }
    w2 = {
        (l, t): mdl.continuous_var(lb=0, name=f"w2_{l}_{t}")
        for l in clients for t in T
    }

    return {
        "x":       x,
        "f":       f,
        "q_prime": q_prime,
        "tau":     tau,
        "tau_return":    tau_return,
        "I_O_frigo":     I_O_frigo,
        "I_O_nonfrigo":  I_O_nonfrigo,
        "w1":      w1,
        "w2":      w2,
    }
