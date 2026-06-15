"""Chromosome → feasible IRP routes (greedy nearest-neighbour decoder)."""


def decode_chromosome(chromosome, sets_):
    """Flat numpy array → {(l, t): quantity} dict."""
    clients = sets_["clients"]
    T       = sets_["T"]
    return {
        (l, t): float(chromosome[l_idx * len(T) + t_idx])
        for l_idx, l in enumerate(clients)
        for t_idx, t in enumerate(T)
    }


def build_routes(quantities, sets_, params_):
    """
    Decode quantities into vehicle routes, enforcing all hard constraints
    (capacity C5, depot inventory C6-C7, demand satisfaction C8).

    Returns dict: x, f, depot_stock, arrival_times, truck_assign,
                  actual_qty, routes_data, tau_return.
    """
    clients          = sets_["clients"]
    T                = sets_["T"]
    M                = sets_["M"]
    O                = sets_["O"]
    Q                = params_["Q"]
    d                = params_["d"]
    v                = params_["v"]
    s                = params_["s"]
    requires_cold    = params_["requires_cold"]
    frigo_trucks     = params_["frigo_trucks"]
    non_frigo_trucks = [k for k in M if k not in frigo_trucks]
    frigo_list       = sorted(frigo_trucks)
    q_lt             = params_["q_lt"]

    I_frigo    = float(params_["I_O_init_frigo"])
    I_nonfrigo = float(params_["I_O_init_nonfrigo"])
    I_min_f    = params_["I_O_min_frigo"]
    I_max_f    = params_["I_O_max_frigo"]
    I_min_nf   = params_["I_O_min_nonfrigo"]
    I_max_nf   = params_["I_O_max_nonfrigo"]
    R_frigo    = params_["R_frigo"]
    R_nonfrigo = params_["R_nonfrigo"]

    total_demand     = {l: sum(q_lt[l, t] for t in T) for l in clients}
    delivered_so_far = {l: 0 for l in clients}

    x_vars        = {}
    f_vars        = {}
    arrival_times = {}
    truck_assign  = {}
    actual_qty    = {}
    depot_stock   = {}
    routes_data   = {}
    T_last        = T[-1]

    for t in T:
        # C8: last period delivers exactly remaining demand; earlier periods round & cap
        if t == T_last:
            desired = {l: max(0, total_demand[l] - delivered_so_far[l]) for l in clients}
        else:
            desired = {
                l: min(
                    max(0, int(round(quantities[l, t]))),
                    max(0, total_demand[l] - delivered_so_far[l]),
                )
                for l in clients
            }

        # C14: split by temperature requirement
        frigo_desired    = {l: q for l, q in desired.items() if     requires_cold[l, t] and q > 0}
        nonfrigo_desired = {l: q for l, q in desired.items() if not requires_cold[l, t] and q > 0}

        # C6: maximum releasable = available stock above safety level
        max_rel_f  = max(0, int(I_frigo    + R_frigo[t]    - I_min_f))
        max_rel_nf = max(0, int(I_nonfrigo + R_nonfrigo[t] - I_min_nf))

        frigo_qty    = _clamp_to_integer_budget(frigo_desired,    max_rel_f)
        nonfrigo_qty = _clamp_to_integer_budget(nonfrigo_desired, max_rel_nf)

        for l in clients:
            qty = frigo_qty.get(l, nonfrigo_qty.get(l, 0))
            actual_qty[l, t]     = qty
            delivered_so_far[l] += qty

        shipped_f  = sum(frigo_qty.values())
        shipped_nf = sum(nonfrigo_qty.values())

        # C7: depot stock balance, clamped within [I_min, I_max]
        I_frigo    = max(I_min_f,  min(I_max_f,  I_frigo    + R_frigo[t]    - shipped_f))
        I_nonfrigo = max(I_min_nf, min(I_max_nf, I_nonfrigo + R_nonfrigo[t] - shipped_nf))

        depot_stock[t] = {
            "frigo":    round(I_frigo,    4),
            "nonfrigo": round(I_nonfrigo, 4),
        }

        routes_data[t] = {}
        for qty_group, trucks in [(frigo_qty, frigo_list), (nonfrigo_qty, non_frigo_trucks)]:
            if not qty_group or not trucks:
                continue
            r, tx, tf, ta, tassign = _nearest_neighbour(qty_group, trucks, t, d, v, s, Q, O)
            routes_data[t].update(r)
            x_vars.update(tx)
            f_vars.update(tf)
            arrival_times.update(ta)
            truck_assign.update(tassign)

    # C13: latest truck return time per period (used as hard constraint in problem.py)
    tau_return = {}
    for t in T:
        max_ret = 0.0
        for k, info in routes_data.get(t, {}).items():
            path  = info["path"]
            speed = v[k]
            total = sum(
                s.get(path[idx], 0.0) + d[path[idx], path[idx + 1]] / speed
                for idx in range(len(path) - 1)
            )
            max_ret = max(max_ret, total)
        tau_return[t] = max_ret

    return {
        "x":             x_vars,
        "f":             f_vars,
        "depot_stock":   depot_stock,
        "arrival_times": arrival_times,
        "truck_assign":  truck_assign,
        "actual_qty":    actual_qty,
        "routes_data":   routes_data,
        "tau_return":    tau_return,
    }


def _clamp_to_integer_budget(qty_dict, max_total):
    """Trim allocations (largest first) until total fits within max_total."""
    if not qty_dict:
        return {}

    result = dict(qty_dict)
    total  = sum(result.values())

    if total <= max_total:
        return result

    # Sorted descending: subtract excess greedily, O(n log n) instead of O(n×excess)
    excess = total - max_total
    for l in sorted(result, key=result.__getitem__, reverse=True):
        if excess <= 0:
            break
        cut        = min(result[l], excess)
        result[l] -= cut
        excess     -= cut

    return {l: q for l, q in result.items() if q > 0}


def _nearest_neighbour(qty_dict, trucks, t, d, v, s, Q, O):
    """Build routes for one truck group using greedy nearest-neighbour."""
    x_vars   = {}
    f_vars   = {}
    arrivals = {}
    assign   = {}
    routes   = {}

    pending   = dict(qty_dict)
    truck_idx = 0

    while pending and truck_idx < len(trucks):
        k     = trucks[truck_idx]
        cap   = Q[k]
        speed = v[k]

        path         = [O]
        qty_on_route = {}
        load         = 0
        current_time = 0.0
        current      = O

        while True:
            best, best_dist = None, float("inf")
            for l, q in pending.items():
                if load + q <= cap:
                    dist = d[current, l]
                    if dist < best_dist:
                        best_dist, best = dist, l
            if best is None:
                break

            current_time      += s[current] + d[current, best] / speed
            path.append(best)
            qty_on_route[best] = pending.pop(best)
            load              += qty_on_route[best]
            arrivals[best, t]  = current_time
            assign[best, t]    = k
            current            = best

        path.append(O)

        if len(path) > 2:
            # Suffix sum: flow on arc (i→j) = total qty still on truck from j onwards
            n   = len(path)
            suf = [0] * (n + 1)
            for idx in range(n - 2, -1, -1):
                node    = path[idx + 1]
                suf[idx] = suf[idx + 1] + (qty_on_route.get(node, 0) if node != O else 0)

            for idx in range(n - 1):
                i, j               = path[idx], path[idx + 1]
                x_vars[i, j, t, k] = 1
                f_vars[i, j, t, k] = suf[idx]

            routes[k] = {
                "path": path,
                "qty":  {str(l): q for l, q in qty_on_route.items()},
            }

        truck_idx += 1

    return routes, x_vars, f_vars, arrivals, assign