"""Prins (2004) DP-optimal split of a fixed giant-tour order into vehicle
routes, adapted to production's heterogeneous ordered fleet via a layered
DP: the k-th contiguous segment in the fixed order is always assigned to
`trucks[k-1]` -- mirrors `Solvers/NSGA3/decoder.py::_nearest_neighbour`'s
sequential truck consumption (truck k+1 is only ever used once truck k has
been tried, including the case where truck k ends up serving nobody).

Unlike a single left-to-right greedy fill (`diagnose_giant_tour_continuity.
py::_giant_tour_split`, which always extends the current truck's route as
far as capacity/tau_max allow before moving on), this DP evaluates ALL
contiguous partitions of the fixed order and picks the one minimising total
distance -- a "maximal fill" is not always distance-optimal (e.g. it can
pair a far-away client with a near one on the same truck just because
capacity allows it, when leaving that far client for the next truck --
paired with clients actually close to it -- would cost much less overall).

See docs/superpowers/specs/2026-08-10-qinsga3-prins-split-decoder-design.md.
"""
from __future__ import annotations

from sensitivity.diagnose_giant_tour_continuity import _giant_tour_order


def _segment_metrics(order, i, j, qty_dict, floors, d, v_k, s, O, tau_max):
    """Metrics for assigning order[i:j] (contiguous client ids) to one truck
    of speed v_k, visited in the given fixed order. Returns None if a
    non-mandatory client's return-trip projection exceeds tau_max --
    mandatory clients (floor > 0) bypass the check, same as
    `_nearest_neighbour`. Otherwise returns (distance, load_full, qty_full,
    load_floor, qty_floor): *_full uses each client's full requested
    quantity; *_floor substitutes the mandatory floor for mandatory clients
    only (non-mandatory quantities are identical in both) -- the caller
    picks whichever fits the truck's capacity.
    """
    floors = floors or {}
    distance = 0.0
    current_time = 0.0
    current = O
    qty_full, qty_floor = {}, {}
    load_full = load_floor = 0

    for idx in range(i, j):
        l = order[idx]
        floor_l = floors.get(l, 0)
        is_mandatory = floor_l > 0
        dist = d[current, l]

        if tau_max is not None and not is_mandatory:
            projected = (current_time + s.get(current, 0.0) + dist / v_k
                         + s.get(l, 0.0) + d[l, O] / v_k)
            if projected > tau_max:
                return None

        current_time += s.get(current, 0.0) + dist / v_k
        distance += dist

        q_full  = qty_dict[l]
        q_floor = floor_l if is_mandatory else q_full
        qty_full[l]  = q_full
        qty_floor[l] = q_floor
        load_full  += q_full
        load_floor += q_floor
        current = l

    distance += d[current, O]
    return distance, load_full, qty_full, load_floor, qty_floor


def _prins_split_from_order(order, qty_dict, trucks, t, d, v, s, Q, O,
                             tau_max=None, floors=None):
    """DP-optimal partition of `order` into contiguous per-truck segments,
    `trucks[k-1]` serving the k-th segment. Returns the same 5-tuple shape
    as `_nearest_neighbour`: (routes, x_vars, f_vars, arrivals, assign).
    """
    floors = floors or {}
    n = len(order)
    K = len(trucks)
    INF = float("inf")

    dp     = [[INF] * (n + 1) for _ in range(K + 1)]
    origin = [[None] * (n + 1) for _ in range(K + 1)]
    dp[0][0] = 0.0

    for k in range(1, K + 1):
        truck = trucks[k - 1]
        v_k, cap = v[truck], Q[truck]
        dp[k]     = list(dp[k - 1])       # truck k serves nobody (carry forward)
        origin[k] = list(origin[k - 1])

        for j in range(0, n + 1):
            if dp[k - 1][j] == INF:
                continue
            for j2 in range(j + 1, n + 1):
                metrics = _segment_metrics(order, j, j2, qty_dict, floors, d, v_k, s, O, tau_max)
                if metrics is None:
                    break
                distance, load_full, qty_full, load_floor, qty_floor = metrics
                if load_full <= cap:
                    qty_on_route = qty_full
                elif load_floor <= cap:
                    qty_on_route = qty_floor
                else:
                    break
                candidate = dp[k - 1][j] + distance
                if candidate < dp[k][j2]:
                    dp[k][j2]     = candidate
                    origin[k][j2] = (j, qty_on_route)

    served_upto = n if dp[K][n] < INF else max(
        (j for j in range(n + 1) if dp[K][j] < INF), default=0)

    segments = []
    k, j = K, served_upto
    while j > 0:
        if origin[k][j] is None:
            k -= 1
            continue
        prev_j, qty_on_route = origin[k][j]
        segments.append((trucks[k - 1], order[prev_j:j], qty_on_route))
        j = prev_j
        k -= 1
    segments.reverse()

    routes, x_vars, f_vars, arrivals, assign = {}, {}, {}, {}, {}
    for truck, clients_in_segment, qty_on_route in segments:
        speed = v[truck]
        path  = [O] + list(clients_in_segment) + [O]
        m     = len(path)
        suf   = [0] * (m + 1)
        for pos in range(m - 2, -1, -1):
            node = path[pos + 1]
            suf[pos] = suf[pos + 1] + (qty_on_route.get(node, 0) if node != O else 0)

        current_time = 0.0
        current = O
        for pos in range(m - 1):
            i_node, j_node = path[pos], path[pos + 1]
            x_vars[i_node, j_node, t, truck] = 1
            f_vars[i_node, j_node, t, truck] = suf[pos]
            if j_node != O:
                current_time += s.get(current, 0.0) + d[current, j_node] / speed
                arrivals[j_node, t] = current_time
                assign[j_node, t]   = truck
            current = j_node

        routes[truck] = {
            "path": path,
            "qty":  {str(l): q for l, q in qty_on_route.items()},
        }

    return routes, x_vars, f_vars, arrivals, assign


def _prins_split(qty_dict, trucks, t, d, v, s, Q, O, tau_max=None, floors=None,
                  priorities=None):
    """Same call signature as `_nearest_neighbour` -- drop-in monkey-patch
    target. Computes the fixed giant-tour order internally, then delegates
    to `_prins_split_from_order`."""
    order = _giant_tour_order(qty_dict, floors, priorities)
    return _prins_split_from_order(order, qty_dict, trucks, t, d, v, s, Q, O,
                                    tau_max=tau_max, floors=floors)
