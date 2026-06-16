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
    (capacity C5, depot inventory C6-C7, demand satisfaction C8, C9, C14).

    One truck type per client per period (except T_last which delivers all remaining).
    C9 mirror: frigo truck ↔ requires_cold[l,td]=True ; nonfrigo ↔ False.

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

    # Total demand split by product type per client (C9 mirror)
    frigo_total    = {l: sum(q_lt[l, t] for t in T if     requires_cold[l, t]) for l in clients}
    nonfrigo_total = {l: sum(q_lt[l, t] for t in T if not requires_cold[l, t]) for l in clients}

    # Cumulative demand by type up to each period t (used to enforce delivery deadlines)
    cum_frigo_demand    = {
        (l, t): sum(q_lt[l, td] for td in T if td <= t and     requires_cold[l, td])
        for l in clients for t in T
    }
    cum_nonfrigo_demand = {
        (l, t): sum(q_lt[l, td] for td in T if td <= t and not requires_cold[l, td])
        for l in clients for t in T
    }

    # Separate delivery counters per type (needed to cap pre-delivery correctly)
    frigo_dlv    = {l: 0 for l in clients}
    nonfrigo_dlv = {l: 0 for l in clients}

    x_vars        = {}
    f_vars        = {}
    arrival_times = {}
    truck_assign  = {}
    actual_qty    = {}
    depot_stock   = {}
    routes_data   = {}
    T_last        = T[-1]

    for t in T:
        if t == T_last:
            # C8: last period must satisfy ALL remaining demand of both types.
            # The entire desired amount is mandatory (min_required = desired).
            frigo_desired = {
                l: max(0, frigo_total[l] - frigo_dlv[l])
                for l in clients
                if frigo_total[l] - frigo_dlv[l] > 0
            }
            nonfrigo_desired = {
                l: max(0, nonfrigo_total[l] - nonfrigo_dlv[l])
                for l in clients
                if nonfrigo_total[l] - nonfrigo_dlv[l] > 0
            }
            frigo_floor    = dict(frigo_desired)
            nonfrigo_floor = dict(nonfrigo_desired)
        else:
            # C9 + C14: one truck type per client per period = requires_cold[l, t].
            # Only pre-deliver demand of the SAME type as the current dispatch period.
            # DEADLINE RULE: cumulative delivered >= cumulative demand up to t.
            # A client may receive less than their period demand ONLY if an advance
            # was made earlier; otherwise the full deficit must be covered now.
            frigo_desired    = {}
            nonfrigo_desired = {}
            frigo_floor      = {}   # mandatory minimum per client (Bug-1 fix)
            nonfrigo_floor   = {}
            for l in clients:
                qty = max(0, int(round(quantities[l, t])))
                if requires_cold[l, t]:
                    remaining  = max(0, frigo_total[l] - frigo_dlv[l])
                    min_needed = max(0, cum_frigo_demand[l, t] - frigo_dlv[l])
                    actual     = max(min_needed, min(qty, remaining))
                    if actual > 0:
                        frigo_desired[l] = actual
                        frigo_floor[l]   = min_needed
                else:
                    remaining  = max(0, nonfrigo_total[l] - nonfrigo_dlv[l])
                    min_needed = max(0, cum_nonfrigo_demand[l, t] - nonfrigo_dlv[l])
                    actual     = max(min_needed, min(qty, remaining))
                    if actual > 0:
                        nonfrigo_desired[l] = actual
                        nonfrigo_floor[l]   = min_needed

        # C6: maximum releasable = available stock above safety level
        max_rel_f  = max(0, int(I_frigo    + R_frigo[t]    - I_min_f))
        max_rel_nf = max(0, int(I_nonfrigo + R_nonfrigo[t] - I_min_nf))

        frigo_qty    = _clamp_to_integer_budget(frigo_desired,    max_rel_f,  frigo_floor)
        nonfrigo_qty = _clamp_to_integer_budget(nonfrigo_desired, max_rel_nf, nonfrigo_floor)

        for l in clients:
            f  = frigo_qty.get(l, 0)
            nf = nonfrigo_qty.get(l, 0)
            actual_qty[l, t]  = f + nf
            frigo_dlv[l]    += f
            nonfrigo_dlv[l] += nf

        shipped_f  = sum(frigo_qty.values())
        shipped_nf = sum(nonfrigo_qty.values())

        # C7: depot stock balance — strict equality, same as MIP: I_t = I_{t-1} + R_t - shipped_t
        I_frigo    = I_frigo    + R_frigo[t]    - shipped_f
        I_nonfrigo = I_nonfrigo + R_nonfrigo[t] - shipped_nf

        depot_stock[t] = {
            "frigo":    round(I_frigo,    4),
            "nonfrigo": round(I_nonfrigo, 4),
        }

        routes_data[t] = {}
        tau_max = params_.get("tau_max")
        for qty_group, trucks in [(frigo_qty, frigo_list), (nonfrigo_qty, non_frigo_trucks)]:
            if not qty_group or not trucks:
                continue
            r, tx, tf, ta, tassign = _nearest_neighbour(qty_group, trucks, t, d, v, s, Q, O, tau_max)
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


def _clamp_to_integer_budget(qty_dict, max_total, min_required=None):
    """
    Trim allocations until total fits within max_total.
    min_required: mandatory floor per client — never cut below it.
    Only the discretionary surplus (qty - floor) is eligible for cutting.
    """
    if not qty_dict:
        return {}

    result   = dict(qty_dict)
    floors   = min_required or {}
    total    = sum(result.values())

    if total <= max_total:
        return result

    # Sort by discretionary surplus descending: cut optional portion first
    excess = total - max_total
    for l in sorted(result, key=lambda l: result[l] - floors.get(l, 0), reverse=True):
        if excess <= 0:
            break
        floor     = floors.get(l, 0)
        surplus   = result[l] - floor
        if surplus <= 0:
            continue
        cut        = min(surplus, excess)
        result[l] -= cut
        excess     -= cut

    return {l: q for l, q in result.items() if q > 0}


def _nearest_neighbour(qty_dict, trucks, t, d, v, s, Q, O, tau_max=None):
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
                    if tau_max is not None:
                        # Ensure visiting l and returning to depot stays within tau_max
                        projected = (current_time
                                     + s.get(current, 0.0) + dist / speed
                                     + s.get(l, 0.0) + d[l, O] / speed)
                        if projected > tau_max:
                            continue
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
