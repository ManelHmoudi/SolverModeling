"""Objective functions computed from a decoded solution (no docplex).

Implements exactly docs/model_formulation.md §6 -- the same formulas as
models/objectives.py.
"""

PRODUCT_CLASSES = ("frigo", "nonfrigo")


# ── Route-level helpers (shared with solvers/qinsga3/repair.py) ──────────────

def route_arrivals(path, k, params_):
    """[(client, arrival time)] in visit order (no idle waiting)."""
    d, v, s = params_["d"], params_["v"], params_["s"]
    out, now = [], 0.0
    for i in range(1, len(path) - 1):
        now += s[path[i - 1]] + d[path[i - 1], path[i]] / v[k]
        out.append((path[i], now))
    return out


def route_duration(path, k, params_):
    """Depot-to-depot duration of one route (h)."""
    d, v, s = params_["d"], params_["v"], params_["s"]
    return sum(s[path[i]] + d[path[i], path[i + 1]] / v[k] for i in range(len(path) - 1))


def route_loads(path, f_vars, t, k):
    return [f_vars.get((path[i], path[i + 1], t, k), 0.0) for i in range(len(path) - 1)]


def route_transport_cost(path, loads, k, params_):
    """F_veh + c_km_k·distance (every arc, empty return leg included) + c_load·Σ d·load."""
    if len(path) <= 2:
        return 0.0
    d = params_["d"]
    dist = sum(d[path[i], path[i + 1]] for i in range(len(path) - 1))
    load_km = sum(d[path[i], path[i + 1]] * loads[i] for i in range(len(path) - 1))
    return params_["F_veh"] + params_["c_km_k"][k] * dist + params_["c_load"] * load_km


def route_co2(path, loads, k, params_):
    """CO2 (kg) of one route, linearised CMEM (Bektas & Laporte 2011):
    e_co2 / fuel_to_joules · Σ d_m [alpha (w + kg·load) + beta v²]."""
    d_m, v_ms = params_["d_m"], params_["v_ms"][k]
    alpha, beta = params_["alpha_co2"], params_["beta_co2"]
    energy = sum(
        d_m[path[i], path[i + 1]] * (alpha * (params_["w"] + params_["kg_per_unit"] * loads[i])
                                     + beta * v_ms ** 2)
        for i in range(len(path) - 1))
    return params_["e_co2"] * energy / params_["fuel_to_joules"]


def time_window_penalty(l, t, arrival, params_):
    return (params_["c1"] * max(0.0, params_["ET"][l, t] - arrival)
            + params_["c2"] * max(0.0, arrival - params_["LT"][l, t]))


def cyclic_window(t, L, T):
    n, pos = len(T), T.index(t)
    return [T[(pos - j) % n] for j in range(L)]


# ── Objectives ────────────────────────────────────────────────────────────────

def compute_f1_detail(route_result, sets_, params_):
    T = sets_["T"]
    f_vars = route_result["f"]
    transport = tw = 0.0
    for t, routes in route_result["routes_data"].items():
        for k, info in routes.items():
            path = info["path"]
            transport += route_transport_cost(path, route_loads(path, f_vars, t, k), k, params_)
            for node, arr in route_arrivals(path, k, params_):
                tw += time_window_penalty(node, t, arr, params_)
    stock, orders = route_result["depot_stock"], route_result["orders"]
    depot_holding = sum(params_["h_frigo"] * stock[t]["frigo"]
                        + params_["h_nonfrigo"] * stock[t]["nonfrigo"] for t in T)
    ordering = params_["F_order"] * sum(1 for t in T for p in PRODUCT_CLASSES if orders[t][p] > 0)
    total = transport + depot_holding + tw + ordering
    return total, {"transport": round(transport, 4), "depot_holding": round(depot_holding, 4),
                   "time_windows": round(tw, 4), "ordering": round(ordering, 4)}


def compute_f1(route_result, sets_, params_):
    return compute_f1_detail(route_result, sets_, params_)[0]


def compute_f2(route_result, sets_, params_):
    f_vars = route_result["f"]
    return sum(route_co2(info["path"], route_loads(info["path"], f_vars, t, k), k, params_)
               for t, routes in route_result["routes_data"].items()
               for k, info in routes.items() if len(info["path"]) > 2)


def compute_f3(route_result, sets_, params_):
    """Σ over periods of the longest route duration (depot to depot, h)."""
    return sum(max((route_duration(info["path"], k, params_) for k, info in routes.items()
                    if len(info["path"]) > 2), default=0.0)
               for routes in route_result["routes_data"].values())


def working_capital_series(route_result, sets_, params_):
    """{t: (stock value, receivables, payables)} with cyclic accounting (€)."""
    T, clients = sets_["T"], sets_["clients"]
    P_pur, P_sale = params_["P_purchase"], params_["P_sale"]
    actual, stock, orders = route_result["actual_qty"], route_result["depot_stock"], route_result["orders"]
    invoiced = {t: sum(P_sale[l] * actual.get((l, t), 0) for l in clients) for t in T}
    purchased = {t: P_pur * sum(orders[t][p] for p in PRODUCT_CLASSES) for t in T}
    return {t: (P_pur * (stock[t]["frigo"] + stock[t]["nonfrigo"]),
                sum(invoiced[tp] for tp in cyclic_window(t, params_["L_S"], T)),
                sum(purchased[tp] for tp in cyclic_window(t, params_["L_P"], T)))
            for t in T}


def compute_f4(route_result, sets_, params_):
    """Peak net working capital (€)."""
    return max(s + r - p for s, r, p in working_capital_series(route_result, sets_, params_).values())


def compute_f4_detail(route_result, sets_, params_):
    series = working_capital_series(route_result, sets_, params_)
    t_peak = max(series, key=lambda t: series[t][0] + series[t][1] - series[t][2])
    s, r, p = series[t_peak]
    return s + r - p, {
        "stock": round(s, 4), "receivables": round(r, 4), "payables": round(p, 4),
        "peak_period": t_peak,
        "series": {str(t): round(a + b - c, 4) for t, (a, b, c) in series.items()},
    }
