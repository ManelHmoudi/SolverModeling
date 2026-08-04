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
