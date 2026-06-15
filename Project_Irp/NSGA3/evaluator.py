"""Objective functions for the many-objective IRP (no docplex dependency)."""


def compute_f1(route_result, sets_, params_):
    """Logistics cost = transport (y1) + depot holding (y2) + time-window penalties (y3)."""
    f_vars        = route_result["f"]
    depot_stock   = route_result["depot_stock"]
    arrival_times = route_result["arrival_times"]
    c_ijk   = params_["c_ijk"]
    d       = params_["d"]
    h_O     = params_["h_O"]
    c1      = params_["c1"]
    c2      = params_["c2"]
    ET      = params_["ET"]
    LT      = params_["LT"]
    clients = sets_["clients"]
    T       = sets_["T"]

    # y1: load-weighted arc transport cost
    y1 = sum(c_ijk[i, j, k] * d[i, j] * flow for (i, j, t, k), flow in f_vars.items())

    # y2: depot holding cost over all periods
    y2 = sum(h_O * (depot_stock[t]["frigo"] + depot_stock[t]["nonfrigo"]) for t in T)

    # y3: soft time-window penalty (C12) — only for visited clients
    y3 = 0.0
    for l in clients:
        for t in T:
            arr = arrival_times.get((l, t), 0.0)
            if arr > 0.0:
                y3 += c1 * max(0.0, ET[l, t] - arr)
                y3 += c2 * max(0.0, arr - LT[l, t])

    return y1 + y2 + y3


def compute_f2(route_result, sets_, params_):
    """CO2 emissions — linearised CMEM model (Bektas & Laporte 2011)."""
    x_vars         = route_result["x"]
    f_vars         = route_result["f"]
    e_co2          = params_["e_co2"]
    fuel_to_joules = params_["fuel_to_joules"]
    d_m            = params_["d_m"]
    alpha_co2      = params_["alpha_co2"]
    beta_co2       = params_["beta_co2"]
    v2             = params_["v2"]
    w              = params_["w"]
    kg_per_unit    = params_["kg_per_unit"]

    total = sum(
        d_m[i, j] * (alpha_co2[i, j] * (w + kg_per_unit * f_vars.get((i, j, t, k), 0.0))
                     + beta_co2 * v2[k])
        for (i, j, t, k) in x_vars
    )
    return (e_co2 / fuel_to_joules) * total


def compute_f3(route_result, sets_, params_):
    """Total arc travel time across all trucks and periods."""
    x_vars = route_result["x"]
    d      = params_["d"]
    v      = params_["v"]
    return sum(d[i, j] / v[k] for (i, j, t, k) in x_vars)


def _f4_components(route_result, sets_, params_):
    """Shared computation for f4: returns (stock_value, receivables, payables)."""
    actual_qty  = route_result["actual_qty"]
    depot_stock = route_result["depot_stock"]
    T           = sets_["T"]
    clients     = sets_["clients"]
    O           = sets_["O"]
    P_sale      = params_["P_sale"]
    P_purchase  = params_["P_purchase"]
    DIO         = params_["DIO"]
    DSO         = params_["DSO"]
    DPO         = params_["DPO"]

    stock = sum(
        (depot_stock[t]["frigo"] + depot_stock[t]["nonfrigo"]) * P_purchase[O] * (DIO / 365)
        for t in T
    )
    recv = sum(actual_qty[l, t] * P_sale[l]    * (DSO / 365) for l in clients for t in T)
    pay  = sum(actual_qty[l, t] * P_purchase[O] * (DPO / 365) for l in clients for t in T)
    return stock, recv, pay


def compute_f4(route_result, sets_, params_):
    """Working capital (BFR) scalar — fast path used during optimization."""
    stock, recv, pay = _f4_components(route_result, sets_, params_)
    return stock + recv - pay


def compute_f4_detail(route_result, sets_, params_):
    """Working capital with sub-component breakdown — used only in reporting."""
    stock, recv, pay = _f4_components(route_result, sets_, params_)
    return stock + recv - pay, {
        "stock":       round(stock, 4),
        "receivables": round(recv,  4),
        "payables":    round(pay,   4),
    }