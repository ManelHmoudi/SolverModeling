"""Independent solution checker for the IRP model.

Re-derives every constraint and every objective of docs/model_formulation.md
§6-§7 from the raw instance JSON and a solution's routes and orders ONLY.
It imports nothing from models/ or solvers/, so it is a genuine second
implementation: a bug in the decoder, the evaluator or the parameter loader
cannot hide itself here.

A solution is:
    routes : {t: {k: {"path": [0, ..., 0], "qty": {client: units}}}}
    orders : {t: {"frigo": units, "nonfrigo": units}}
"""

import json
import math

CLASSES = ("frigo", "nonfrigo")
TOL = 1e-6


class Instance:
    def __init__(self, path):
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        s, p = raw["sets"], raw["parameters"]
        self.clients, self.T, self.K, self.O = list(s["clients"]), list(s["T"]), list(s["M"]), s["O"]
        coords = {int(i): tuple(c) for i, c in s["coordinates"].items()}
        self.dist = lambda i, j: math.dist(coords[i], coords[j])
        key = lambda k: tuple(map(int, k.split(",")))          # noqa: E731
        self.q = {key(k): v for k, v in p["q_lt"].items()}
        self.cold = {key(k): bool(v) for k, v in p["cold"].items()}
        self.frigo_trucks = set(p["frigo_trucks"])
        self.Q = {int(k): v for k, v in p["Q"].items()}
        self.v = {int(k): float(v) for k, v in p["v"].items()}
        self.p = p

    def truck_class(self, k):
        return "frigo" if k in self.frigo_trucks else "nonfrigo"

    def demand_class(self, l, t):
        return "frigo" if self.cold[l, t] else "nonfrigo"


def _window(t, L, T):
    i = T.index(t)
    return [T[(i - j) % len(T)] for j in range(L)]


def check(inst: Instance, routes, orders):
    """Return (violations: list[str], objectives: dict)."""
    p, O, T = inst.p, inst.O, inst.T
    bad = []
    delivered = {}                                   # (l, t, class) -> units
    shipped = {(c, t): 0.0 for c in CLASSES for t in T}
    transport = tw = co2_energy = 0.0
    longest = {t: 0.0 for t in T}
    alpha = p["a"] + p["g"] * p["Cr"]
    beta = 0.5 * p["Cd"] * p["A_f"] * p["rho"]

    for t in T:
        for k, info in routes.get(t, {}).items():
            path = list(info["path"])
            qty = {int(l): q for l, q in info["qty"].items()}
            if len(path) <= 2:
                continue
            tag = f"t{t} k{k}"
            if path[0] != O or path[-1] != O:
                bad.append(f"{tag}: route does not start/end at the depot")
            inner = path[1:-1]
            if O in inner or len(set(inner)) != len(inner):
                bad.append(f"{tag}: depot or a client visited twice inside a route")
            if set(qty) != set(inner):
                bad.append(f"{tag}: delivered clients differ from visited clients")
            for l in inner:
                if qty.get(l, 0) < 1 - TOL:
                    bad.append(f"{tag}: empty visit to client {l}")
            load = sum(qty.values())
            if load > inst.Q[k] + TOL:
                bad.append(f"{tag}: load {load} > capacity {inst.Q[k]}")
            cls = inst.truck_class(k)
            for l, q in qty.items():
                delivered[l, t, cls] = delivered.get((l, t, cls), 0.0) + q
                shipped[cls, t] += q

            now, on_board, v_ms = 0.0, load, inst.v[k] / 3.6
            c_km = p["c_km"] + (p["p5"] / inst.v[k] if cls == "frigo" else 0.0)
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                dab = inst.dist(a, b)
                now += p["s_value"] + dab / inst.v[k]
                transport += c_km * dab + p["c_load"] * dab * on_board
                co2_energy += dab * 1000 * (alpha * (p["w"] + p["kg_per_unit"] * on_board) + beta * v_ms ** 2)
                if b != O:
                    tw += p["c1"] * max(0.0, p["ET"][str(b)] - now) + p["c2"] * max(0.0, now - p["LT"][str(b)])
                    on_board -= qty.get(b, 0)
            transport += p["F_veh"]
            if abs(on_board) > TOL:
                bad.append(f"{tag}: vehicle does not return empty")
            longest[t] = max(longest[t], now)
            if now > p["tau_max"] + TOL:
                bad.append(f"{tag}: return time {now:.3f} > tau_max")

    # (c11) at most one truck per product class serves a client in a period
    for t in T:
        seen = {}
        for k, info in routes.get(t, {}).items():
            if len(info["path"]) <= 2:
                continue
            cls = inst.truck_class(k)
            for l in info["qty"]:
                key = (int(l), cls)
                if key in seen:
                    bad.append(f"t{t}: client {l} served by two {cls} trucks ({seen[key]}, {k})")
                seen[key] = k

    # demand: per class, never late, never above the total
    for l in inst.clients:
        for cls in CLASSES:
            cum_del = cum_dem = 0.0
            total = sum(inst.q[l, t] for t in T if inst.demand_class(l, t) == cls)
            for t in T:
                cum_del += delivered.get((l, t, cls), 0.0)
                cum_dem += inst.q[l, t] if inst.demand_class(l, t) == cls else 0
                if cum_del < cum_dem - TOL:
                    bad.append(f"client {l} {cls}: late at t{t}")
                if cum_del > total + TOL:
                    bad.append(f"client {l} {cls}: over-delivered at t{t}")
            if abs(cum_del - total) > TOL:
                bad.append(f"client {l} {cls}: total delivered {cum_del} != demand {total}")

    # depot stock and supplier orders
    stock, depot_hold, n_orders = {}, 0.0, 0
    h = {"frigo": p["h_cold"], "nonfrigo": p["h_ambient"]}
    for cls in CLASSES:
        I = p[f"I_O_init_{cls}"]
        for t in T:
            r = float(orders[t][cls])
            if r < -TOL:
                bad.append(f"{cls} t{t}: negative order")
            if r > TOL:
                n_orders += 1
            I = I + r - shipped[cls, t]
            stock[cls, t] = I
            depot_hold += h[cls] * I
            if I < p[f"I_O_min_{cls}"] - TOL or I > p[f"I_O_max_{cls}"] + TOL:
                bad.append(f"{cls} t{t}: depot stock {I} out of bounds")
        if abs(stock[cls, T[-1]] - p[f"I_O_init_{cls}"]) > TOL:
            bad.append(f"{cls}: end stock != initial stock (cycle not closed)")

    # peak net working capital, cyclic accounting
    P_pur = p["P_purchase"]
    invoiced = {t: sum(p["P_sale"][str(l)] * delivered.get((l, t, c), 0.0)
                       for l in inst.clients for c in CLASSES) for t in T}
    bought = {t: P_pur * sum(float(orders[t][c]) for c in CLASSES) for t in T}
    wc = {t: P_pur * sum(stock[c, t] for c in CLASSES)
          + sum(invoiced[x] for x in _window(t, p["L_S"], T))
          - sum(bought[x] for x in _window(t, p["L_P"], T)) for t in T}

    objectives = {
        "f1": transport + depot_hold + tw + p["F_order"] * n_orders,
        "f2": p["e_co2"] * co2_energy / p["fuel_to_joules"],
        "f3": sum(longest.values()),
        "f4": max(wc.values()),
    }
    return bad, objectives


def from_route_result(route_result):
    routes = {t: {k: {"path": info["path"], "qty": info["qty"]} for k, info in r.items()}
              for t, r in route_result["routes_data"].items()}
    return routes, route_result["orders"]
