"""
Many-Objective Inventory Routing Problem (IRP)
===============================================
Decision variables -- docs/model_formulation.md §5
"""

PRODUCT_CLASSES = ("frigo", "nonfrigo")


def build_variables(mdl, sets_, params_):
    N, A, T, M, clients = sets_["N"], sets_["A"], sets_["T"], sets_["M"], sets_["clients"]
    tau_max = params_["tau_max"]   # every arrival/return of a used vehicle is <= tau_max

    return {
        # x[i,j,t,k] = 1 if vehicle k travels arc (i,j) in period t
        "x": {(i, j, t, k): mdl.binary_var(name=f"x_{i}_{j}_{t}_{k}")
              for (i, j) in A for t in T for k in M},
        # Quantities are whole units of product (model hypothesis): f, q', r are integers.
        # f[i,j,t,k] = load (units) on arc (i,j)
        "f": {(i, j, t, k): mdl.integer_var(lb=0, name=f"f_{i}_{j}_{t}_{k}")
              for (i, j) in A for t in T for k in M},
        # q'[l,t,td] = quantity delivered to l in t for the demand due in td (t <= td)
        "q_prime": {(l, t, td): mdl.integer_var(lb=0, name=f"qp_{l}_{t}_{td}")
                    for l in clients for t in T for td in T if t <= td},
        # r[p,t] = supplier order of class p received at the start of t; z = order placed
        "r": {(p, t): mdl.integer_var(lb=0, name=f"r_{p}_{t}") for p in PRODUCT_CLASSES for t in T},
        "z": {(p, t): mdl.binary_var(name=f"z_{p}_{t}") for p in PRODUCT_CLASSES for t in T},
        # depot stock per class at the end of t
        "I_O_frigo":    {t: mdl.continuous_var(lb=0, name=f"I_frigo_{t}") for t in T},
        "I_O_nonfrigo": {t: mdl.continuous_var(lb=0, name=f"I_nonfrigo_{t}") for t in T},
        # a[i,t,k] = arrival time of k at i (h); e[t,k] = return time of k (h)
        "a": {(i, t, k): mdl.continuous_var(lb=0, ub=tau_max, name=f"a_{i}_{t}_{k}")
              for i in N for t in T for k in M},
        "e": {(t, k): mdl.continuous_var(lb=0, ub=tau_max, name=f"e_{t}_{k}") for t in T for k in M},
        # earliness / lateness of k's visit to l (h)
        "w1": {(l, t, k): mdl.continuous_var(lb=0, name=f"w1_{l}_{t}_{k}") for l in clients for t in T for k in M},
        "w2": {(l, t, k): mdl.continuous_var(lb=0, name=f"w2_{l}_{t}_{k}") for l in clients for t in T for k in M},
        # E[t] = completion time of period t = return time of its longest route (h)
        "E": {t: mdl.continuous_var(lb=0, ub=tau_max, name=f"E_{t}") for t in T},
        # W = peak net working capital (€)
        "W": mdl.continuous_var(lb=-mdl.infinity, name="W_peak"),
    }
