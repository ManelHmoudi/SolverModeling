"""Chromosome → feasible IRP routes (greedy nearest-neighbour decoder).

Chromosome layout:
  [0 … n_clients*n_periods - 1]  quantity genes  — q(l,t) for each (client, period)
  [n_clients*n_periods … end]     priority genes  — p(l) ∈ [0,1] per client
"""


def decode_chromosome(chromosome, sets_):
    """Flat numpy array → ({(l,t): quantity}, {l: priority}) tuple."""
    clients = sets_["clients"]
    T       = sets_["T"]
    n_qty   = len(clients) * len(T)

    quantities = {
        (l, t): float(chromosome[l_idx * len(T) + t_idx])
        for l_idx, l in enumerate(clients)
        for t_idx, t in enumerate(T)
    }
    priorities = {
        l: float(chromosome[n_qty + l_idx])
        for l_idx, l in enumerate(clients)
    }
    return quantities, priorities


def build_routes(quantities, sets_, params_, priorities=None):
    """Decode quantities into vehicle routes, enforcing all hard constraints.

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

    frigo_total    = {l: sum(q_lt[l, t] for t in T if     requires_cold[l, t]) for l in clients}
    nonfrigo_total = {l: sum(q_lt[l, t] for t in T if not requires_cold[l, t]) for l in clients}

    cum_frigo_demand = {
        (l, t): sum(q_lt[l, td] for td in T if td <= t and     requires_cold[l, td])
        for l in clients for t in T
    }
    cum_nonfrigo_demand = {
        (l, t): sum(q_lt[l, td] for td in T if td <= t and not requires_cold[l, td])
        for l in clients for t in T
    }

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
            frigo_desired = {
                l: max(0, frigo_total[l] - frigo_dlv[l])
                for l in clients if frigo_total[l] - frigo_dlv[l] > 0
            }
            nonfrigo_desired = {
                l: max(0, nonfrigo_total[l] - nonfrigo_dlv[l])
                for l in clients if nonfrigo_total[l] - nonfrigo_dlv[l] > 0
            }
            frigo_floor    = dict(frigo_desired)
            nonfrigo_floor = dict(nonfrigo_desired)
        else:
            frigo_desired    = {}
            nonfrigo_desired = {}
            frigo_floor      = {}
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

        max_rel_f  = max(0, int(I_frigo    + R_frigo[t]    - I_min_f))
        max_rel_nf = max(0, int(I_nonfrigo + R_nonfrigo[t] - I_min_nf))

        frigo_cap_total = sum(Q[k] for k in frigo_list)
        nf_cap_total    = sum(Q[k] for k in non_frigo_trucks) if non_frigo_trucks else 0

        frigo_qty    = _clamp_to_integer_budget(
            frigo_desired,    min(max_rel_f,  frigo_cap_total), frigo_floor)
        nonfrigo_qty = _clamp_to_integer_budget(
            nonfrigo_desired, min(max_rel_nf, nf_cap_total),    nonfrigo_floor)

        routes_data[t] = {}
        tau_max = params_.get("tau_max")
        for qty_group, trucks, floor_group in [
            (frigo_qty,    frigo_list,       frigo_floor),
            (nonfrigo_qty, non_frigo_trucks, nonfrigo_floor),
        ]:
            if not qty_group or not trucks:
                continue
            r, tx, tf, ta, tassign = _nearest_neighbour(
                qty_group, trucks, t, d, v, s, Q, O, tau_max, floor_group,
                priorities=priorities,
            )
            routes_data[t].update(r)
            x_vars.update(tx)
            f_vars.update(tf)
            arrival_times.update(ta)
            truck_assign.update(tassign)

        actually_served_frigo    = {}
        actually_served_nonfrigo = {}
        for k, info in routes_data[t].items():
            is_frigo = k in frigo_trucks
            for l_str, q in info["qty"].items():
                l_int = int(l_str)
                if is_frigo:
                    actually_served_frigo[l_int]    = actually_served_frigo.get(l_int, 0)    + q
                else:
                    actually_served_nonfrigo[l_int] = actually_served_nonfrigo.get(l_int, 0) + q

        shipped_f = shipped_nf = 0
        for l in clients:
            f  = actually_served_frigo.get(l, 0)
            nf = actually_served_nonfrigo.get(l, 0)
            actual_qty[l, t]  = f + nf
            frigo_dlv[l]    += f
            nonfrigo_dlv[l] += nf
            shipped_f += f
            shipped_nf += nf

        I_frigo    = I_frigo    + R_frigo[t]    - shipped_f
        I_nonfrigo = I_nonfrigo + R_nonfrigo[t] - shipped_nf

        depot_stock[t] = {
            "frigo":    round(I_frigo,    4),
            "nonfrigo": round(I_nonfrigo, 4),
        }

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
    """Trim allocations until total fits within max_total, never cutting below min_required."""
    if not qty_dict:
        return {}

    result = dict(qty_dict)
    floors = min_required or {}
    total  = sum(result.values())

    if total <= max_total:
        return result

    excess = total - max_total
    for l in sorted(result, key=lambda l: result[l] - floors.get(l, 0), reverse=True):
        if excess <= 0:
            break
        floor   = floors.get(l, 0)
        surplus = result[l] - floor
        if surplus <= 0:
            continue
        cut        = min(surplus, excess)
        result[l] -= cut
        excess     -= cut

    return {l: q for l, q in result.items() if q > 0}


def _nearest_neighbour(qty_dict, trucks, t, d, v, s, Q, O, tau_max=None, floors=None,
                        priorities=None):
    """Build routes for one truck group using greedy nearest-neighbour.

    Mandatory clients (floor > 0) bypass the tau_max check to preserve delivery deadlines.
    Priority genes produce diverse visit orders, widening the Pareto front.
    """
    x_vars   = {}
    f_vars   = {}
    arrivals = {}
    assign   = {}
    routes   = {}

    prios     = priorities or {}
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
            best, best_score, best_qty = None, float("inf"), 0
            has_mandatory = False

            any_mandatory_pending = any((floors or {}).get(l2, 0) > 0 for l2 in pending)

            for l, q in pending.items():
                floor_l      = (floors or {}).get(l, 0)
                is_mandatory = floor_l > 0

                if not is_mandatory and any_mandatory_pending:
                    continue

                if load + q <= cap:
                    q_effective = q
                elif is_mandatory and load + floor_l <= cap:
                    q_effective = floor_l
                else:
                    continue

                dist = d[current, l]

                if tau_max is not None and not is_mandatory:
                    projected = (current_time
                                 + s.get(current, 0.0) + dist / speed
                                 + s.get(l, 0.0) + d[l, O] / speed)
                    if projected > tau_max:
                        continue

                score = dist / (0.5 + prios.get(l, 0.5))
                if is_mandatory:
                    if not has_mandatory or score < best_score:
                        best_score, best, best_qty = score, l, q_effective
                        has_mandatory = True
                elif not has_mandatory and score < best_score:
                    best_score, best, best_qty = score, l, q_effective

            if best is None:
                break

            current_time      += s.get(current, 0.0) + d[current, best] / speed
            path.append(best)
            pending.pop(best)
            qty_on_route[best] = best_qty
            load              += qty_on_route[best]
            arrivals[best, t]  = current_time
            assign[best, t]    = k
            current            = best

        path.append(O)

        if len(path) > 2:
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
