"""
Many-Objective Inventory Routing Problem (IRP)
===============================================
Objective functions -- docs/model_formulation.md §6
-------------------
f1 -- Logistics cost (€)  dispatch + distance (incl. empty return) + load
                          + depot holding + time windows + ordering
f2 -- CO2 emissions (kg)  linearised CMEM (Bektas & Laporte 2011)
f3 -- Longest route per period, summed over the periods (h)
f4 -- Peak net working capital (€), cyclic (steady-state) accounting
"""

from .variables import PRODUCT_CLASSES


def cyclic_window(t, L, T):
    """Periods t, t-1, ..., t-L+1 of a repeating horizon (indices wrap around).
    A period appears several times when L > |T|, as in steady state."""
    n = len(T)
    pos = T.index(t)
    return [T[(pos - j) % n] for j in range(L)]


def delivered(mdl, vars_, l, t, T):
    """D_l,t: units delivered to client l in period t."""
    return mdl.sum(vars_["q_prime"][l, t, td] for td in T if td >= t)


def build_f1_logistics_cost(mdl, vars_, sets_, params_):
    """f1 = transport + depot holding + time windows + ordering (€).

    Transport has three terms with explicit units (editor remark E2):
        F_veh  [€/route]  ·  dispatched routes
        c_km_k [€/km]     ·  distance of every arc, the empty return leg included
        c_load [€/(unit·km)] · load × distance
    """
    x, f = vars_["x"], vars_["f"]
    A, T, M, O, N, clients = (sets_[k] for k in ("A", "T", "M", "O", "N", "clients"))
    d = params_["d"]

    transport = (
        params_["F_veh"] * mdl.sum(x[O, j, t, k] for j in N if j != O for t in T for k in M)
        + mdl.sum(params_["c_km_k"][k] * d[i, j] * x[i, j, t, k] for (i, j) in A for t in T for k in M)
        + mdl.sum(params_["c_load"] * d[i, j] * f[i, j, t, k] for (i, j) in A for t in T for k in M)
    )
    depot_holding = mdl.sum(params_["h_frigo"] * vars_["I_O_frigo"][t]
                            + params_["h_nonfrigo"] * vars_["I_O_nonfrigo"][t] for t in T)
    time_windows = mdl.sum(params_["c1"] * vars_["w1"][l, t, k] + params_["c2"] * vars_["w2"][l, t, k]
                           for l in clients for t in T for k in M)
    ordering = params_["F_order"] * mdl.sum(vars_["z"][p, t] for p in PRODUCT_CLASSES for t in T)
    return transport + depot_holding + time_windows + ordering


def build_f2_co2_emissions(mdl, vars_, sets_, params_):
    """f2 -- CO2 (kg), linearised CMEM (Bektas & Laporte 2011).

    Energy on arc (i,j) [J] = d_m · [ alpha · (w·x + kg_per_unit·f) + beta · v² · x ],
    alpha = a + g·Cr, beta = ½·Cd·A_f·ρ, v in m/s, d_m in m.
    Fuel [L] = energy / fuel_to_joules ;  CO2 [kg] = fuel · e_co2.
    """
    x, f = vars_["x"], vars_["f"]
    d_m, v_ms = params_["d_m"], params_["v_ms"]
    alpha, beta = params_["alpha_co2"], params_["beta_co2"]
    w, kg = params_["w"], params_["kg_per_unit"]
    return (params_["e_co2"] / params_["fuel_to_joules"]) * mdl.sum(
        d_m[i, j] * (alpha * (w * x[i, j, t, k] + kg * f[i, j, t, k]) + beta * v_ms[k] ** 2 * x[i, j, t, k])
        for (i, j) in sets_["A"] for t in sets_["T"] for k in sets_["M"]
    )


def build_f3_longest_route(mdl, vars_, sets_, params_):
    """f3 = Σ_t E_t with (C19) E_t >= e_t,k for every vehicle k (h).

    E_t is the time at which the last vehicle of period t is back at the
    depot: the working day of the most loaded driver. Unlike total driving
    time it is not proportional to the kilometres (so not redundant with
    f2): it measures how evenly the work is spread over the fleet. Unused
    vehicles have e_t,k free in [0, tau_max]; minimising f3 drives them to 0,
    and exact_objective_values() reads only used vehicles.
    """
    E, e = vars_["E"], vars_["e"]
    for t in sets_["T"]:
        for k in sets_["M"]:
            mdl.add_constraint(E[t] >= e[t, k], ctname=f"C19_longest_t{t}_k{k}")
    return mdl.sum(E[t] for t in sets_["T"])


def working_capital_terms(mdl, vars_, sets_, params_):
    """Per-period stock value, outstanding receivables and payables (€), with
    cyclic (steady-state) accounting: the plan repeats, so invoices of the
    last periods are still outstanding at the start of the next cycle.

    AR_t: invoices issued at delivery in the last L_S periods.
    AP_t: supplier invoices for orders received in the last L_P periods.
    """
    T, clients = sets_["T"], sets_["clients"]
    P_pur, P_sale = params_["P_purchase"], params_["P_sale"]
    stock, recv, pay = {}, {}, {}
    for t in T:
        stock[t] = P_pur * (vars_["I_O_frigo"][t] + vars_["I_O_nonfrigo"][t])
        recv[t] = mdl.sum(P_sale[l] * delivered(mdl, vars_, l, tp, T)
                          for l in clients for tp in cyclic_window(t, params_["L_S"], T))
        pay[t] = P_pur * mdl.sum(vars_["r"][p, tp]
                                 for p in PRODUCT_CLASSES for tp in cyclic_window(t, params_["L_P"], T))
    return stock, recv, pay


def build_f4_working_capital(mdl, vars_, sets_, params_):
    """f4 = W with (C18) W >= stock_t + AR_t - AP_t for every t (editor remark E3).
    The epigraph is added here so W is always tied to WC when f4 exists."""
    stock, recv, pay = working_capital_terms(mdl, vars_, sets_, params_)
    W = vars_["W"]
    for t in sets_["T"]:
        mdl.add_constraint(W >= stock[t] + recv[t] - pay[t], ctname=f"C18_peak_wc_t{t}")
    return W, {"stock_value": stock, "receivables": recv, "payables": pay}


def peak_breakdown(f4_sub):
    """After a solve: WC components at the peak period, as floats."""
    best = None
    for t in f4_sub["stock_value"]:
        s = f4_sub["stock_value"][t].solution_value
        r = f4_sub["receivables"][t].solution_value
        p = f4_sub["payables"][t].solution_value
        if best is None or s + r - p > best["stock"] + best["receivables"] - best["payables"]:
            best = {"stock": s, "receivables": r, "payables": p, "peak_period": t}
    return {k: (round(v, 4) if k != "peak_period" else v) for k, v in best.items()}


def exact_objective_values(vars_, objectives, sets_, params_):
    """Objective values recomputed from the solution itself (not from the
    epigraph variables W and w1/w2, which are only lower-bounded and are exact
    only when their objective is being minimised)."""
    O, T, M, clients = sets_["O"], sets_["T"], sets_["M"], sets_["clients"]
    x, a = vars_["x"], vars_["a"]
    tw_epigraph = sum(params_["c1"] * vars_["w1"][l, t, k].solution_value
                      + params_["c2"] * vars_["w2"][l, t, k].solution_value
                      for l in clients for t in T for k in M)
    tw_exact = 0.0
    for t in T:
        for k in M:
            for l in clients:
                visited = sum(x[i, l, t, k].solution_value for i in sets_["N"] if i != l) > 0.5
                if visited:
                    arr = a[l, t, k].solution_value
                    tw_exact += (params_["c1"] * max(0.0, params_["ET"][l, t] - arr)
                                 + params_["c2"] * max(0.0, arr - params_["LT"][l, t]))
    pk = peak_breakdown(objectives["f4_sub"])
    longest = 0.0
    for t in T:
        used = [vars_["e"][t, k].solution_value for k in M
                if sum(x[O, j, t, k].solution_value for j in sets_["N"] if j != O) > 0.5]
        longest += max(used, default=0.0)
    return {
        "f1": objectives["f1"].solution_value - tw_epigraph + tw_exact,
        "f2": objectives["f2"].solution_value,
        "f3": longest,
        "f4": pk["stock"] + pk["receivables"] - pk["payables"],
    }


def build_all_objectives(mdl, vars_, sets_, params_):
    f4, f4_sub = build_f4_working_capital(mdl, vars_, sets_, params_)
    return {
        "f1":     build_f1_logistics_cost(mdl, vars_, sets_, params_),
        "f2":     build_f2_co2_emissions(mdl, vars_, sets_, params_),
        "f3":     build_f3_longest_route(mdl, vars_, sets_, params_),
        "f4":     f4,
        "f4_sub": f4_sub,
    }
