"""
Many-Objective Inventory Routing Problem (IRP)
===============================================
Objective Functions
-------------------
f1 -- Logistics cost        (transport + storage + time-window penalties)
f2 -- CO2 emissions         (CMEM model, Bektas & Laporte 2011)
f3 -- Total travel time     (sum of arc travel times)
f4 -- Working capital (BFR) (stock value + receivables - payables)
"""


def build_f1_logistics_cost(mdl, vars_, sets_, params_):
    """
    f1 = y1 (transport cost) + y2 (storage cost) + y3 (time-window penalties)

    Three components:
        y1 -- transport cost: load-weighted arc cost (with refrigeration surcharge for k=1)
        y2 -- storage cost: holding cost per unit per period
        y3 -- time-window penalty: linearised via non-negative slack variables

    """
    x       = vars_["x"]
    f_var   = vars_["f"]
    I_var   = vars_["I_var"]
    w1      = vars_["w1"]
    w2      = vars_["w2"]

    A           = sets_["A"]
    T           = sets_["T"]
    M           = sets_["M"]
    clients     = sets_["clients"]
    stock_nodes = sets_["stock_nodes"]

    c_ijk = params_["c_ijk"]
    d     = params_["d"]
    h     = params_["h"]
    c1    = params_["c1"]
    c2    = params_["c2"]

    # y1 -- transport cost: load-weighted arc cost
    y1 = mdl.sum(
        c_ijk[i, j, k] * d[i, j] * f_var[i, j, t, k]
        for (i, j) in A for t in T for k in M
    )

    # y2 -- storage cost: holding cost per unit per period
    y2 = mdl.sum(h[i, t] * I_var[i, t] for i in stock_nodes for t in T)

    # y3 -- time-window penalty: linearised via non-negative slack variables
    y3 = mdl.sum(c1 * w1[l, t] + c2 * w2[l, t] for l in clients for t in T)

    return y1 + y2 + y3


def build_f2_co2_emissions(mdl, vars_, sets_, params_):
    """
    f2 -- CO2 emissions (linearised CMEM)

    Fuel consumption on each arc depends on:
        - vehicle curb weight (w) and payload (f[i,j,t,k]), scaled by alpha_co2
        - aerodynamic drag, scaled by beta_co2 and speed squared
    Units: energy in Joules (kg.m²/s²); converted to litres via fuel_to_joules,
           then to kg CO2 via e_co2 (kg CO2/litre).

    Reference: Bektas & Laporte (2011) -- Pollution-Routing Problem / CMEM.

    """
    x     = vars_["x"]
    f_var = vars_["f"]

    A = sets_["A"]
    T = sets_["T"]
    M = sets_["M"]

    alpha_co2      = params_["alpha_co2"]
    beta_co2       = params_["beta_co2"]
    d_m            = params_["d_m"]
    v2             = params_["v2"]
    w              = params_["w"]
    kg_per_unit    = params_["kg_per_unit"]
    e_co2          = params_["e_co2"]
    fuel_to_joules = params_["fuel_to_joules"]

    f2_expr = (e_co2 / fuel_to_joules) * mdl.sum(
        d_m[i, j] * (
            alpha_co2[i, j] * (w * x[i, j, t, k] + kg_per_unit * f_var[i, j, t, k])
          + beta_co2         *  v2[k]              * x[i, j, t, k]
        )
        for (i, j) in A for t in T for k in M
    )

    return f2_expr


def build_f3_travel_time(mdl, vars_, sets_, params_):
    """
    f3 -- Total travel time
    Sum of travel times (hours) over all arcs used across all periods and vehicles.

    """
    x = vars_["x"]

    A = sets_["A"]
    T = sets_["T"]
    M = sets_["M"]

    d = params_["d"]
    v = params_["v"]

    return mdl.sum(
        x[i, j, t, k] * d[i, j] / v[k]
        for (i, j) in A for t in T for k in M
    )


def build_f4_working_capital(mdl, vars_, sets_, params_):
    """
    f4 -- Working capital requirement (BFR)

    BFR = inventory value + accounts receivable - accounts payable
        = Σ I_var[O,t] * P_purchase[O] * (DIO/365)
        + Σ q_prime[l,t] * P_sale[l] * (DSO/365)
        - Σ q_prime[l,t] * P_purchase[O] * (DPO/365)

    Returns
    -------
    f4_expr : docplex linear expression for f4
    sub     : dict with the three sub-expressions for calibration reporting
    """
    I_var   = vars_["I_var"]
    q_prime = vars_["q_prime"]

    T       = sets_["T"]
    clients = sets_["clients"]
    O       = sets_["O"]

    P_sale     = params_["P_sale"]
    P_purchase = params_["P_purchase"]
    DIO        = params_["DIO"]
    DSO        = params_["DSO"]
    DPO        = params_["DPO"]

    stock_value = mdl.sum(I_var[O, t] * P_purchase[O] * (DIO / 365) for t in T)

    receivables = mdl.sum(
        q_prime[l, t, td] * P_sale[l] * (DSO / 365)
        for l in clients for t in T for td in T if t <= td
    )

    payables = mdl.sum(
        q_prime[l, t, td] * P_purchase[O] * (DPO / 365)
        for l in clients for t in T for td in T if t <= td
    )

    f4_expr = stock_value + receivables - payables

    # Sub-expressions kept for calibration reporting
    sub = {
        "stock_value": stock_value,
        "receivables": receivables,
        "payables":    payables,
    }

    return f4_expr, sub


def build_all_objectives(mdl, vars_, sets_, params_):
    """
    Single entry point: build all four objective functions.

    Returns
    -------
    dict :
        'f1'     : logistics cost expression
        'f2'     : CO2 emissions expression
        'f3'     : travel time expression
        'f4'     : working capital expression
        'f4_sub' : dict of BFR sub-expressions (stock_value, receivables, payables)
    """
    f1          = build_f1_logistics_cost(mdl, vars_, sets_, params_)
    f2          = build_f2_co2_emissions(mdl, vars_, sets_, params_)
    f3          = build_f3_travel_time(mdl, vars_, sets_, params_)
    f4, f4_sub  = build_f4_working_capital(mdl, vars_, sets_, params_)

    return {
        "f1":     f1,
        "f2":     f2,
        "f3":     f3,
        "f4":     f4,
        "f4_sub": f4_sub,
    }