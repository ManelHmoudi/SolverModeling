"""Sweep-partition (Gillett & Miller 1974) zone-locked decoder for QINSGA3
continuity screening: clients are assigned once to a fixed geographic zone
(one per truck, by polar angle around the depot), independent of theta.
Construction reuses production's exact greedy scoring
(`Solvers/NSGA3/decoder.py::_nearest_neighbour`), with one added
restriction: the truck at position `truck_idx` only ever considers pending
clients in its own zone -- a theta perturbation can still reorder or drop
clients within a zone, but can never move one into a different truck's
route. See
docs/superpowers/specs/2026-08-10-qinsga3-zone-locked-decoder-design.md.
"""
from __future__ import annotations

import math

_unserved_count = [0]   # mutable counter: clients whose zone's truck couldn't
                          # fit them this call (no cross-zone fallback), for
                          # the smoke test's own reporting -- not used by the
                          # decoder itself.


def _zone_assignment(coords, O, client_ids, trucks, Q, total_demand):
    """Partition `client_ids` into `len(trucks)` contiguous angular sectors
    around `coords[O]`, one sector per truck in the given fixed order. Cut
    thresholds are proportional to each truck's capacity `Q[truck]` against
    cumulative aggregate demand (`total_demand`, a static per-client
    weight -- NOT the smaller per-period quantity seen at any single decode
    call), keeping zones roughly capacity-balanced despite a heterogeneous
    fleet. Returns {client_id: zone_index}, zone_index in 0..len(trucks)-1.
    """
    ox, oy = coords[O]
    order = sorted(
        client_ids,
        key=lambda l: (math.atan2(coords[l][1] - oy, coords[l][0] - ox), l),
    )
    total_Q = sum(Q[truck] for truck in trucks) or 1.0
    total_d = sum(total_demand.get(l, 0.0) for l in client_ids)

    zone, cum, k_pos = {}, 0.0, 0
    cum_target = total_d * (Q[trucks[0]] / total_Q)
    for l in order:
        while k_pos < len(trucks) - 1 and cum >= cum_target:
            k_pos += 1
            cum_target += total_d * (Q[trucks[k_pos]] / total_Q)
        zone[l] = k_pos
        cum += total_demand.get(l, 0.0)
    return zone


def _make_zone_locked_decoder(zone_by_trucks):
    """Returns a `_nearest_neighbour`-compatible decoder closed over
    `zone_by_trucks` (`{tuple(sorted(trucks)): zone_dict}`, one entry per
    truck group -- looked up by truck-id set, so the same returned function
    works for both the frigo and non-frigo groups)."""

    def _zone_locked_nearest_neighbour(qty_dict, trucks, t, d, v, s, Q, O,
                                        tau_max=None, floors=None, priorities=None):
        zone = zone_by_trucks[tuple(sorted(trucks))]
        prios = priorities or {}
        floors = floors or {}
        pending = dict(qty_dict)
        truck_idx = 0

        x_vars, f_vars, arrivals, assign, routes = {}, {}, {}, {}, {}

        while pending and truck_idx < len(trucks):
            k     = trucks[truck_idx]
            cap   = Q[k]
            speed = v[k]

            path, qty_on_route, load, current_time, current = [O], {}, 0, 0.0, O

            while True:
                best, best_score, best_qty = None, float("inf"), 0
                has_mandatory = False

                zone_pending = {l: q for l, q in pending.items() if zone.get(l) == truck_idx}
                any_mandatory_pending = any(floors.get(l2, 0) > 0 for l2 in zone_pending)

                for l, q in zone_pending.items():
                    floor_l      = floors.get(l, 0)
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
                        projected = (current_time + s.get(current, 0.0) + dist / speed
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

        _unserved_count[0] += len(pending)
        return routes, x_vars, f_vars, arrivals, assign

    return _zone_locked_nearest_neighbour
