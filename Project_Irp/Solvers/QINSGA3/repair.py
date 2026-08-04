"""Post-decode 2-opt route repair for QI-NSGA-III (remedy G) -- see
docs/superpowers/specs/2026-08-04-qinsga3-route-repair-design.md.

Baldwinian repair: improves the FITNESS assigned to a decoded route_result,
never re-encoded back into the chromosome (theta/X) -- the visit order a
2-opt swap produces has no defined inverse into priority genes. Scoped
entirely to QINSGA3: Solvers/NSGA3/decoder.py, evaluator.py, and problem.py
(shared with NSGA3) are never imported for modification here, only their
pure functions (decode_chromosome, build_routes, compute_f1) are reused
unchanged, from Solvers/QINSGA3/algorithm.py's _evaluate_with_repair.
"""


def _two_opt_candidates(path: list):
    """Yield (i, j, candidate_path) for every 2-opt segment reversal of the
    interior of path (positions 1..len(path)-2 -- the depot at both ends,
    index 0 and index len(path)-1, is never moved). Pure: never inspects
    distances, costs, or any domain state.
    """
    n = len(path)
    for i in range(1, n - 2):
        for j in range(i + 1, n - 1):
            candidate = path[:i] + path[i:j + 1][::-1] + path[j + 1:]
            yield i, j, candidate


def _route_traversal_time(path: list, k, params_: dict) -> float:
    """Total round-trip time for one truck's path (depot to depot), summing
    service time + travel time over every consecutive pair -- the exact
    formula Solvers/NSGA3/decoder.py's build_routes uses for its own
    per-route tau_return computation (decoder.py, the block right after
    the main per-period loop), factored out here as a standalone helper
    since build_routes doesn't expose it.
    """
    d = params_["d"]
    v = params_["v"]
    s = params_["s"]
    speed = v[k]
    return sum(
        s.get(path[idx], 0.0) + d[path[idx], path[idx + 1]] / speed
        for idx in range(len(path) - 1)
    )


def _rebuild_route_arcs(path: list, qty_on_route: dict, t, k, params_: dict):
    """Recompute one route's x/f arc entries and per-client arrival times
    for a (possibly reordered) path, given the SAME qty_on_route mapping
    the original decode produced -- a 2-opt swap never changes which
    clients are visited or their delivered quantities, only the order.
    Mirrors decoder.py's own arrival-time accumulation (line 342) and
    suffix-sum arc-flow construction (lines 353-368) exactly, factored out
    here since build_routes doesn't expose either as a standalone function.

    Returns (x_vars, f_vars, arrivals) for THIS route only -- the caller
    merges these into the full route_result's dicts via _replace_route_arcs.
    """
    d = params_["d"]
    v = params_["v"]
    s = params_["s"]
    speed = v[k]
    depot = path[0]

    x_vars = {}
    f_vars = {}
    arrivals = {}

    current_time = 0.0
    current = depot
    for node in path[1:]:
        current_time += s.get(current, 0.0) + d[current, node] / speed
        if node != depot:
            arrivals[node, t] = current_time
        current = node

    n = len(path)
    suf = [0] * (n + 1)
    for idx in range(n - 2, -1, -1):
        node = path[idx + 1]
        suf[idx] = suf[idx + 1] + (qty_on_route.get(node, 0) if node != depot else 0)

    for idx in range(n - 1):
        i, j = path[idx], path[idx + 1]
        x_vars[i, j, t, k] = 1
        f_vars[i, j, t, k] = suf[idx]

    return x_vars, f_vars, arrivals


def _replace_route_arcs(arc_dict: dict, old_path: list, t, k, new_entries: dict) -> dict:
    """Remove old_path's (i, j, t, k) arc keys from arc_dict, then merge in
    new_entries -- used to update route_result["x"]/["f"] for one repaired
    route without disturbing other routes' entries in the same shared dict.
    """
    result = dict(arc_dict)
    for idx in range(len(old_path) - 1):
        i, j = old_path[idx], old_path[idx + 1]
        result.pop((i, j, t, k), None)
    result.update(new_entries)
    return result


_MAX_REPAIR_ITER = 5   # internal constant, not exposed -- see design doc's "New parameters".
# Lowered from 20 after the timing check (sensitivity/compare_route_repair.py,
# instance 100, gen=10, pop=50) measured a ~29x slowdown vs baseline, growing
# with generation count (15.4x at gen=3 -> 28.9x at gen=10) -- well past the
# design doc's Risk-section 10x threshold. Each whole-individual compute_f1
# recomputation per candidate swap is the dominant cost; capping the search
# depth trades repair thoroughness for tractability.


def _repair_route_result(route_result: dict, sets_: dict, params_: dict) -> dict:
    """First-improvement 2-opt local search per route: for every truck's
    path longer than 3 nodes (more than 1 client), repeatedly applies the
    first candidate swap that (a) does not push this period's worst-case
    travel time above its pre-repair value, and (b) strictly reduces f1
    for the whole individual -- until no such candidate exists or
    _MAX_REPAIR_ITER is reached. Baldwinian: returns a NEW route_result
    with updated x/f/arrival_times/routes_data/tau_return; the chromosome
    that produced the original route_result is never touched by the caller.
    """
    from Solvers.NSGA3.evaluator import compute_f1

    working = dict(route_result)
    working["x"] = dict(route_result["x"])
    working["f"] = dict(route_result["f"])
    working["arrival_times"] = dict(route_result["arrival_times"])
    working["tau_return"] = dict(route_result["tau_return"])
    working["routes_data"] = {
        t: dict(routes) for t, routes in route_result["routes_data"].items()
    }

    for t, routes in route_result["routes_data"].items():
        tau_return_before = route_result["tau_return"].get(t, 0.0)

        for k, info in routes.items():
            path = list(info["path"])
            if len(path) <= 3:
                continue
            qty_on_route = {int(l): q for l, q in info["qty"].items()}

            current_f1 = compute_f1(working, sets_, params_)

            for _ in range(_MAX_REPAIR_ITER):
                improved = False
                for i, j, candidate in _two_opt_candidates(path):
                    candidate_time = _route_traversal_time(candidate, k, params_)
                    if candidate_time > tau_return_before:
                        continue

                    trial_x, trial_f, trial_arrivals = _rebuild_route_arcs(
                        candidate, qty_on_route, t, k, params_
                    )
                    scratch = dict(working)
                    scratch["x"] = _replace_route_arcs(working["x"], path, t, k, trial_x)
                    scratch["f"] = _replace_route_arcs(working["f"], path, t, k, trial_f)
                    scratch["arrival_times"] = {**working["arrival_times"], **trial_arrivals}

                    trial_f1 = compute_f1(scratch, sets_, params_)
                    if trial_f1 < current_f1:
                        working["x"] = scratch["x"]
                        working["f"] = scratch["f"]
                        working["arrival_times"] = scratch["arrival_times"]
                        working["routes_data"][t][k] = {"path": candidate, "qty": info["qty"]}
                        working["tau_return"][t] = max(
                            _route_traversal_time(r["path"], k2, params_)
                            for k2, r in working["routes_data"][t].items()
                        )
                        path = candidate
                        current_f1 = trial_f1
                        improved = True
                        break
                if not improved:
                    break

    return working
