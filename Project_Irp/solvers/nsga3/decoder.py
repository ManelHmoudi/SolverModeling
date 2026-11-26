"""Chromosome → IRP solution (routes + supplier orders), greedy decoder.

Implements docs/model_formulation.md §8.

Chromosome layout:
  [0 … nC*nT - 1]               quantity genes — q(l,t) for each (client, period)
  [nC*nT … nC*nT + nC - 1]      priority genes — p(l) ∈ [0,1] per client
  [nC*nT + nC … end]            order genes    — g(p,t) ∈ [0,1] per (class, period),
                                                 ordered frigo t1..tn, nonfrigo t1..tn

Search-space restrictions of the decoder (subsets of the model's feasible set,
stated in §8 of the formulation; one truck per client, period and class is
a constraint of the model itself, C11):
  * a client receives goods early only in periods whose own demand has the same class;
  * optional (anticipated) deliveries below MIN_OPTIONAL_DELIVERY units are dropped;
  * routes are built truck after truck by nearest neighbour, ordered by the priority genes.
"""

PRODUCT_CLASSES = ("frigo", "nonfrigo")

# Heuristic of the decoder, not a model parameter: an anticipated delivery
# smaller than this is not worth a stop. Mandatory deliveries are never dropped.
MIN_OPTIONAL_DELIVERY = 5


def n_genes(sets_):
    """Chromosome length for an instance."""
    nC, nT = len(sets_["clients"]), len(sets_["T"])
    return nC * nT + nC + len(PRODUCT_CLASSES) * nT


def decode_chromosome(chromosome, sets_):
    """Flat array → ({(l,t): quantity}, {l: priority}, {(p,t): order gene})."""
    clients, T = sets_["clients"], sets_["T"]
    if len(chromosome) != n_genes(sets_):
        raise ValueError(f"chromosome has {len(chromosome)} genes, the instance needs {n_genes(sets_)}")
    nT, n_qty = len(T), len(clients) * len(T)
    quantities = {(l, t): float(chromosome[li * nT + ti])
                  for li, l in enumerate(clients) for ti, t in enumerate(T)}
    priorities = {l: float(chromosome[n_qty + li]) for li, l in enumerate(clients)}
    base = n_qty + len(clients)
    order_genes = {(p, t): float(chromosome[base + pi * nT + ti])
                   for pi, p in enumerate(PRODUCT_CLASSES) for ti, t in enumerate(T)}
    return quantities, priorities, order_genes


def decide_order(gene, I_prev, shipped, I_min, I_max, end_target=None, cap_after=None):
    """Supplier order for one class in one period (§8 step 3), integer units.

      r_min: stock at the end of the period >= I_min (C5)
      r_max: stock at the end of the period <= I_max (C5) and, if given,
             <= cap_after (keeps the closed cycle reachable)
    The gene interpolates between r_min and r_max. In the last period the
    order is forced to close the cycle exactly.
    """
    after = I_prev - shipped
    if end_target is not None:                       # last period: close the cycle (C7)
        return float(max(0, int(round(end_target - after))))
    r_min = max(0, int(-(-(I_min - after) // 1)))     # ceil
    r_max = int((I_max - after) // 1)
    if cap_after is not None:
        r_max = min(r_max, int((cap_after - after) // 1))
    if r_max < r_min:
        return float(r_min)
    g = min(max(float(gene), 0.0), 1.0)
    return float(int(round(r_min + g * (r_max - r_min))))


def build_routes(quantities, sets_, params_, priorities=None, order_genes=None):
    """Decode quantities into vehicle routes and supplier orders.

    Returns dict: x, f, depot_stock, orders, arrival_times (keyed (l,t,k)),
                  truck_assign, actual_qty, routes_data, tau_return (max per period).
    """
    clients, T, M, O = sets_["clients"], sets_["T"], sets_["M"], sets_["O"]
    Q, d, v, s = params_["Q"], params_["d"], params_["v"], params_["s"]
    cold, q_lt = params_["requires_cold"], params_["q_lt"]
    frigo_list = sorted(k for k in M if k in params_["frigo_trucks"])
    ambient_list = sorted(k for k in M if k not in params_["frigo_trucks"])
    order_genes = order_genes or {}
    tau_max = params_["tau_max"]

    stock = {p: float(params_[f"I_O_init_{p}"]) for p in PRODUCT_CLASSES}
    total = {(l, p): sum(q_lt[l, t] for t in T if cold[l, t] == (p == "frigo"))
             for l in clients for p in PRODUCT_CLASSES}
    cum_dem = {(l, t, p): sum(q_lt[l, td] for td in T if td <= t and cold[l, td] == (p == "frigo"))
               for l in clients for t in T for p in PRODUCT_CLASSES}
    dlv = {(l, p): 0 for l in clients for p in PRODUCT_CLASSES}

    x_vars, f_vars, arrival_times, truck_assign = {}, {}, {}, {}
    actual_qty, depot_stock, orders, routes_data = {}, {}, {}, {}
    T_last = T[-1]

    for ti, t in enumerate(T):
        desired = {p: {} for p in PRODUCT_CLASSES}
        floor = {p: {} for p in PRODUCT_CLASSES}
        for l in clients:
            if t == T_last:
                classes = PRODUCT_CLASSES                         # everything still due
            else:
                classes = ("frigo",) if cold[l, t] else ("nonfrigo",)
            for p in classes:
                remaining = max(0, total[l, p] - dlv[l, p])
                needed = max(0, cum_dem[l, t, p] - dlv[l, p])
                if t == T_last:
                    qty = remaining
                else:
                    qty = max(needed, min(max(0, int(round(quantities[l, t]))), remaining))
                if qty > 0 and (needed > 0 or qty >= MIN_OPTIONAL_DELIVERY):
                    desired[p][l] = qty
                    floor[p][l] = needed if t != T_last else qty

        routes_data[t] = {}
        for p, trucks in (("frigo", frigo_list), ("nonfrigo", ambient_list)):
            if not desired[p] or not trucks:
                continue
            # shipments limited by the fleet (the supplier order covers the rest)
            capped = _clamp_to_integer_budget(desired[p], sum(Q[k] for k in trucks), floor[p])
            r, tx, tf, ta, tassign = _nearest_neighbour(capped, trucks, t, d, v, s, Q, O, tau_max,
                                                        floor[p], priorities=priorities)
            routes_data[t].update(r)
            x_vars.update(tx)
            f_vars.update(tf)
            arrival_times.update(ta)
            truck_assign.update(tassign)

        served = {(l, p): 0 for l in clients for p in PRODUCT_CLASSES}
        for k, info in routes_data[t].items():
            p = "frigo" if k in params_["frigo_trucks"] else "nonfrigo"
            for l_str, q in info["qty"].items():
                served[int(l_str), p] += q
        shipped = {p: 0 for p in PRODUCT_CLASSES}
        for l in clients:
            actual_qty[l, t] = served[l, "frigo"] + served[l, "nonfrigo"]
            for p in PRODUCT_CLASSES:
                dlv[l, p] += served[l, p]
                shipped[p] += served[l, p]

        # Supplier orders. At every period the stock must not exceed
        # I_init + everything still to ship (all remaining demand must be
        # delivered later), so that the stock can always come back to I_init
        # at the end of the horizon (closed cycle, C7).
        orders[t] = {}
        for p in PRODUCT_CLASSES:
            cap_after = None
            if t != T_last:
                cap_after = params_[f"I_O_init_{p}"] + sum(max(0, total[l, p] - dlv[l, p]) for l in clients)
            r = decide_order(order_genes.get((p, t), 0.0), stock[p], shipped[p],
                             params_[f"I_O_min_{p}"], params_[f"I_O_max_{p}"],
                             params_[f"I_O_init_{p}"] if t == T_last else None, cap_after)
            orders[t][p] = r
            stock[p] = stock[p] + r - shipped[p]
        depot_stock[t] = {p: round(stock[p], 4) for p in PRODUCT_CLASSES}

    tau_return = {}
    for t in T:
        tau_return[t] = max(
            [sum(s[path[i]] + d[path[i], path[i + 1]] / v[k] for i in range(len(path) - 1))
             for k, info in routes_data[t].items() for path in [info["path"]]] or [0.0])

    return {
        "x":             x_vars,
        "f":             f_vars,
        "depot_stock":   depot_stock,
        "orders":        orders,
        "arrival_times": arrival_times,
        "truck_assign":  truck_assign,
        "actual_qty":    actual_qty,
        "routes_data":   routes_data,
        "tau_return":    tau_return,
    }


def _clamp_to_integer_budget(qty_dict, max_total, min_required=None):
    """Trim allocations until total fits within max_total, never cutting below min_required."""
    if not qty_dict:
        return {}
    result = dict(qty_dict)
    floors = min_required or {}
    excess = sum(result.values()) - max_total
    if excess <= 0:
        return result
    for l in sorted(result, key=lambda c: result[c] - floors.get(c, 0), reverse=True):
        if excess <= 0:
            break
        cut = min(result[l] - floors.get(l, 0), excess)
        if cut > 0:
            result[l] -= cut
            excess -= cut
    return {l: q for l, q in result.items() if q > 0}


def _nearest_neighbour(qty_dict, trucks, t, d, v, s, Q, O, tau_max, floors, priorities=None):
    """Build routes for one truck group, truck after truck, by nearest neighbour.

    Mandatory clients (floor > 0) are served first. Every client must fit in
    the shift (tau_max) of the current truck; when none fits, the next truck
    is opened. Only on the LAST truck of the group may a mandatory client
    exceed tau_max (otherwise it would be late), and G then reports it.
    Priority genes bias the visit order: score = distance / (0.5 + priority).
    """
    x_vars, f_vars, arrivals, assign, routes = {}, {}, {}, {}, {}
    prios = priorities or {}
    pending = dict(qty_dict)

    for truck_idx, k in enumerate(trucks):
        if not pending:
            break
        last_truck = truck_idx == len(trucks) - 1
        cap, speed = Q[k], v[k]
        path, qty_on_route, load, now, current = [O], {}, 0, 0.0, O

        while True:
            mandatory_left = any(floors.get(l, 0) > 0 for l in pending)
            best, best_score, best_qty = None, float("inf"), 0
            for l, q in pending.items():
                floor_l = floors.get(l, 0)
                is_mandatory = floor_l > 0
                if mandatory_left and not is_mandatory:
                    continue
                if load + q <= cap:
                    q_eff = q
                elif is_mandatory and load + floor_l <= cap:
                    q_eff = floor_l
                else:
                    continue
                projected = now + s[current] + d[current, l] / speed + s[l] + d[l, O] / speed
                if projected > tau_max and not (is_mandatory and last_truck):
                    continue
                score = d[current, l] / (0.5 + prios.get(l, 0.5))
                if score < best_score:
                    best, best_score, best_qty = l, score, q_eff
            if best is None:
                break
            now += s[current] + d[current, best] / speed
            path.append(best)
            pending.pop(best)
            qty_on_route[best] = best_qty
            load += best_qty
            arrivals[best, t, k] = now
            assign[best, t] = k
            current = best

        path.append(O)
        if len(path) > 2:
            suffix = 0
            loads = []
            for node in reversed(path[1:]):
                suffix += qty_on_route.get(node, 0) if node != O else 0
                loads.append(suffix)
            loads = list(reversed(loads))            # loads[i] = load on arc path[i] -> path[i+1]
            for i in range(len(path) - 1):
                x_vars[path[i], path[i + 1], t, k] = 1
                f_vars[path[i], path[i + 1], t, k] = loads[i]
            routes[k] = {"path": path, "qty": {str(l): q for l, q in qty_on_route.items()}}

    return routes, x_vars, f_vars, arrivals, assign
