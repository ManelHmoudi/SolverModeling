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


def _two_opt_candidates(path: list, max_window: int | None = None):
    """Yield (i, j, candidate_path) for every 2-opt segment reversal of the
    interior of path (positions 1..len(path)-2 -- the depot at both ends,
    index 0 and index len(path)-1, is never moved). Pure: never inspects
    distances, costs, or any domain state.

    max_window (optional, default None): when None, yields every (i, j)
    pair -- the original, exhaustive behaviour this function has always
    had. When set to an int, only yields pairs with j - i <= max_window,
    bounding the search to nearby positions -- real IRP routes average
    ~18 nodes and range up to 29 (measured on instance_100_clients.json),
    so full O(n^2) candidate generation is worth bounding on the longer
    ones. UNLIKE every other optimisation in this module, this is NOT
    mathematically guaranteed equivalent to the unbounded search: an
    improving swap between two distant positions could exist and never be
    tried. This is a deliberate, documented trade-off of search
    completeness for speed (see docs/superpowers/specs/
    2026-08-04-qinsga3-route-repair-design.md), not a bug.
    """
    n = len(path)
    for i in range(1, n - 2):
        j_upper = n - 1 if max_window is None else min(n - 1, i + 1 + max_window)
        for j in range(i + 1, j_upper):
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


def _route_f1_contribution(path: list, f_vars: dict, arrival_times: dict, t, k, params_: dict) -> float:
    """The portion of compute_f1's y1 (transport cost) + y3 (time-window
    penalty) attributable to ONE route (t, k) only -- O(route length), not
    O(whole network). A 2-opt swap only ever changes ONE route's arcs and
    arrival times; every other term in f1 (y2 holding cost, every other
    route's y1/y3 contribution) is identical before and after the swap and
    cancels exactly in the difference. So
        compute_f1(new_route_result) - compute_f1(old_route_result)
        == _route_f1_contribution(new_path, ...) - _route_f1_contribution(old_path, ...)
    exactly -- see test_route_f1_contribution_delta_matches_compute_f1_delta,
    which verifies this equivalence against the real compute_f1 formula.
    Replaces an earlier version of this search that called compute_f1 on a
    full scratch route_result per candidate -- correct, but O(whole network)
    per candidate; measured ~43x slower than baseline on the real IRP
    (sensitivity/route_repair_timing_check_iter5.txt, instance 100, gen=10),
    which this fixes.
    """
    c_ijk = params_["c_ijk"]
    d     = params_["d"]
    c1    = params_["c1"]
    c2    = params_["c2"]
    ET    = params_["ET"]
    LT    = params_["LT"]

    y1 = 0.0
    for idx in range(len(path) - 1):
        i, j = path[idx], path[idx + 1]
        y1 += c_ijk[i, j, k] * d[i, j] * f_vars[i, j, t, k]

    y3 = 0.0
    for l in path[1:-1]:
        arr = arrival_times.get((l, t), 0.0)
        if arr > 0.0:
            y3 += c1 * max(0.0, ET[l, t] - arr)
            y3 += c2 * max(0.0, arr - LT[l, t])

    return y1 + y3


def _evaluate_candidate(path: list, qty_on_route: dict, t, k, tau_return_before: float, params_: dict):
    """Single-pass, fast-path replacement for calling _route_traversal_time,
    _rebuild_route_arcs, and _route_f1_contribution separately on the same
    candidate -- the three together walk the path five times in total
    (_route_traversal_time once, _rebuild_route_arcs three times internally,
    _route_f1_contribution once more); this walks it at most three times
    (matching _rebuild_route_arcs's own pass count, with zero extra passes)
    by accumulating y3 during the arrival-time forward pass and y1 during
    the arc-construction forward pass, instead of two separate follow-up
    passes over already-computed data.

    Returns None if the tau_return guard fails -- checked incrementally, so
    a failing candidate exits as soon as cumulative time exceeds
    tau_return_before, without walking the rest of the path. This is safe
    because cumulative time is monotonically non-decreasing along any path
    (service times and travel times are never negative): if the running
    total already exceeds tau_return_before partway through, the final
    total (>= the running total, since only non-negative terms remain)
    would exceed it too -- so an early exit can never accept a candidate
    the full computation would have rejected, or vice versa.

    Otherwise returns (x_vars, f_vars, arrivals, contribution) -- exactly
    what calling _rebuild_route_arcs(...) then
    _route_f1_contribution(candidate, f_vars, arrivals, t, k, params_)
    would have produced. See
    test_evaluate_candidate_matches_separate_calls_on_accepted_and_rejected_candidates
    for the equivalence proof against those two reference functions (which
    are themselves proven equivalent to the real compute_f1 by
    test_route_f1_contribution_delta_matches_compute_f1_delta) -- kept
    unchanged as the "obviously correct" reference this fast path is
    checked against, not removed.
    """
    d     = params_["d"]
    v     = params_["v"]
    s     = params_["s"]
    c_ijk = params_["c_ijk"]
    c1    = params_["c1"]
    c2    = params_["c2"]
    ET    = params_["ET"]
    LT    = params_["LT"]
    speed = v[k]
    depot = path[0]

    arrivals = {}
    current_time = 0.0
    current = depot
    y3 = 0.0
    for node in path[1:]:
        current_time += s.get(current, 0.0) + d[current, node] / speed
        if current_time > tau_return_before:
            return None
        if node != depot:
            arrivals[node, t] = current_time
            if current_time > 0.0:
                y3 += c1 * max(0.0, ET[node, t] - current_time)
                y3 += c2 * max(0.0, current_time - LT[node, t])
        current = node

    n = len(path)
    suf = [0] * (n + 1)
    for idx in range(n - 2, -1, -1):
        node = path[idx + 1]
        suf[idx] = suf[idx + 1] + (qty_on_route.get(node, 0) if node != depot else 0)

    x_vars = {}
    f_vars = {}
    y1 = 0.0
    for idx in range(n - 1):
        i, j = path[idx], path[idx + 1]
        x_vars[i, j, t, k] = 1
        f_vars[i, j, t, k] = suf[idx]
        y1 += c_ijk[i, j, k] * d[i, j] * suf[idx]

    return x_vars, f_vars, arrivals, y1 + y3


_MAX_REPAIR_ITER = 20   # internal constant, not exposed -- see design doc's "New parameters".
# _route_f1_contribution (above) replaced the earlier full-compute_f1-per-
# candidate acceptance check, so the dominant per-candidate cost is now
# O(route length) instead of O(whole network) -- restored to the design
# doc's original value now that the real bottleneck is fixed, rather than
# trading search thoroughness for speed.

_TWO_OPT_WINDOW = 8   # internal constant, not exposed -- bounds each 2-opt
# candidate scan to (i, j) pairs with j - i <= 8. Real IRP routes average
# ~18 nodes and range up to 29 (measured on instance_100_clients.json), so
# full O(n^2) candidate generation is worth bounding on the longer ones.
# Trades search completeness for speed (see _two_opt_candidates's
# docstring) -- unlike _MAX_REPAIR_ITER and the delta-cost/merged-pass
# optimisations above, this one is NOT mathematically guaranteed
# equivalent to the unbounded search.


def _repair_route_result(route_result: dict, sets_: dict, params_: dict) -> dict:
    """Best-improvement 2-opt local search per route: for every truck's
    path longer than 3 nodes (more than 1 client), repeatedly scans every
    candidate swap in the window and applies the STRICTLY BEST one -- the
    one minimising the route's f1 contribution the most, among candidates
    that (a) do not push this period's worst-case travel time above its
    pre-repair value, and (b) strictly reduce f1 for the whole individual --
    until no improving candidate exists or _MAX_REPAIR_ITER is reached.
    Baldwinian: returns a NEW route_result with updated x/f/arrival_times/
    routes_data/tau_return; the chromosome that produced the original
    route_result is never touched by the caller.

    Best-improvement (not first-improvement, the original remedy-G choice):
    validated in `sensitivity/compare_repair_best_improvement.py` against
    first-improvement at the same window (_TWO_OPT_WINDOW=8) -- paired
    Wilcoxon signed-rank at 20 seeds (the project's gold-standard scale),
    HV p=0.0005, GD p=0.000002, IGD p=0.00003, all significant, at zero
    added cost (317.7s vs 318.3s mean per run). Mathematically sound, not
    just empirical: at every iteration, best-improvement evaluates the same
    candidate first-improvement would have picked, plus every other
    candidate in the window, and applies strictly the best -- it cannot do
    worse than first-improvement at that step. (A prior attempt at widening
    _TWO_OPT_WINDOW to unbounded, keeping first-improvement, did NOT help --
    `sensitivity/compare_repair_thoroughness.py` -- because first-improvement
    is path-dependent: a wider neighbourhood just changes which improving
    swap is found first, not whether the eventual local optimum is better.
    Only changing the acceptance rule itself, as done here, gives the
    monotonic-non-worse guarantee.) See `Solvers/IRP_results_summary.md`.
    """
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

            current_contrib = _route_f1_contribution(
                path, working["f"], working["arrival_times"], t, k, params_
            )

            for _ in range(_MAX_REPAIR_ITER):
                best = None   # (candidate, trial_x, trial_f, trial_arrivals, trial_contrib)
                best_contrib = current_contrib
                for i, j, candidate in _two_opt_candidates(path, max_window=_TWO_OPT_WINDOW):
                    evaluated = _evaluate_candidate(
                        candidate, qty_on_route, t, k, tau_return_before, params_
                    )
                    if evaluated is None:
                        continue
                    trial_x, trial_f, trial_arrivals, trial_contrib = evaluated
                    if trial_contrib < best_contrib:
                        best_contrib = trial_contrib
                        best = (candidate, trial_x, trial_f, trial_arrivals, trial_contrib)

                if best is None:
                    break
                candidate, trial_x, trial_f, trial_arrivals, trial_contrib = best
                working["x"] = _replace_route_arcs(working["x"], path, t, k, trial_x)
                working["f"] = _replace_route_arcs(working["f"], path, t, k, trial_f)
                working["arrival_times"] = {**working["arrival_times"], **trial_arrivals}
                working["routes_data"][t][k] = {"path": candidate, "qty": info["qty"]}
                working["tau_return"][t] = max(
                    _route_traversal_time(r["path"], k2, params_)
                    for k2, r in working["routes_data"][t].items()
                )
                path = candidate
                current_contrib = trial_contrib

    return working
