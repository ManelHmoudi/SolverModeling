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


_OR_OPT_SEGMENT_LENGTHS = (2, 3)   # single-client relocation (length 1) is a
# separate, not-yet-tested neighbourhood -- kept out of this generator on
# purpose so it can be A/B tested independently later, matching how
# best-improvement vs first-improvement (below) and the 2-opt window were
# each validated as their own isolated change.

_SINGLE_RELOCATION_LENGTH = (1,)   # the "deplacement d'un client dans la
# meme tournee" lever from the advisor's Niveau-1 plan (2-opt; Or-opt;
# single-client relocation) -- same _or_opt_candidates generator, segment
# length 1 only. Tested as its OWN independent ablation (use_single_relocation),
# not stacked onto use_or_opt, matching how every other lever this session
# (dedup, archive, compensate_dx_dtheta, use_or_opt itself) was first tested
# in isolation against the plain 2-opt baseline.


def _or_opt_candidates(path: list, segment_lengths=_OR_OPT_SEGMENT_LENGTHS, max_window: int | None = None):
    """Yield (start, q, candidate_path) for every Or-opt move: remove a
    segment of `segment_lengths` consecutive INTERIOR clients (positions
    1..len(path)-2 -- the depot at both ends is never part of a moved
    segment) and reinsert it, same internal order (no reversal), at a
    different interior position in the same path. Pure: never inspects
    distances, costs, or any domain state -- same contract as
    _two_opt_candidates.

    max_window (optional, default None): when set, bounds how far the
    reinsertion point can be from the segment's original position -- same
    completeness-for-speed trade-off as _two_opt_candidates' own
    max_window (see that function's docstring): NOT mathematically
    guaranteed equivalent to the unbounded search.
    """
    n = len(path)
    for seg_len in segment_lengths:
        if n - 2 < seg_len:
            continue   # not enough interior clients for a segment this long
        for start in range(1, n - seg_len):
            end = start + seg_len   # exclusive
            segment = path[start:end]
            rest = path[:start] + path[end:]
            m = len(rest)

            q_lo, q_hi = 0, m - 2   # insert after rest[q], q in [0, m-2]
            if max_window is not None:
                q_lo = max(q_lo, (start - 1) - max_window)
                q_hi = min(q_hi, (start - 1) + max_window)

            for q in range(q_lo, q_hi + 1):
                if q == start - 1:
                    continue   # reinserts exactly where it was -- not a move
                candidate = rest[:q + 1] + segment + rest[q + 1:]
                yield start, q, candidate


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

_OR_OPT_WINDOW = 8   # mirrors _TWO_OPT_WINDOW's own bound and rationale,
# applied to how far an Or-opt segment may be reinserted from its original
# position -- same completeness-for-speed trade-off, not exposed.


def _repair_route_result(route_result: dict, sets_: dict, params_: dict,
                          use_or_opt: bool = False,
                          use_single_relocation: bool = False,
                          use_two_opt: bool = True,
                          use_inter_route_relocate: bool = False,
                          use_route_swap: bool = False,
                          use_delivery_shift: bool = False) -> dict:
    """Best-improvement local search per route: for every truck's path
    longer than 3 nodes (more than 1 client), repeatedly scans every 2-opt
    candidate swap in the window (plus every Or-opt segment-relocation
    candidate too, when use_or_opt=True) and applies the STRICTLY BEST one
    across the combined neighbourhood -- the one minimising the route's f1
    contribution the most, among candidates that (a) do not push this
    period's worst-case travel time above its pre-repair value, and (b)
    strictly reduce f1 for the whole individual -- until no improving
    candidate exists or _MAX_REPAIR_ITER is reached. Baldwinian: returns a
    NEW route_result with updated x/f/arrival_times/routes_data/tau_return;
    the chromosome that produced the original route_result is never touched
    by the caller.

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

    use_or_opt (default False -- ablation-only, not adopted): widens the
    neighbourhood beyond pure sequencing (2-opt) to also relocating a
    2- or 3-client segment elsewhere in the same route (see
    _or_opt_candidates) -- the MPIRP's decision space includes vehicle
    assignment, delivered quantities, delivery periods, frigo compatibility
    and stock levels, none of which pure 2-opt (a sequencing-only move)
    ever touches; Or-opt is the first, smallest step toward a richer
    neighbourhood. Both move types are evaluated together each iteration
    (same best-improvement rule, extended to the combined candidate set)
    via the same delta-cost _evaluate_candidate every candidate already
    goes through, so this cannot do worse than 2-opt-only at any iteration
    for the same reason best-improvement cannot do worse than
    first-improvement (see above): it always sees every 2-opt-only
    candidate too, plus the Or-opt ones. Single-client relocation (segment
    length 1) is deliberately left out of _or_opt_candidates for now, to
    A/B test the 2-/3-client-segment version in isolation first.

    use_single_relocation (default False -- ablation-only, independent of
    use_or_opt): widens the neighbourhood to also relocating a SINGLE
    client elsewhere in the same route (_or_opt_candidates with
    segment_lengths=_SINGLE_RELOCATION_LENGTH) -- the third lever from the
    advisor's Niveau-1 plan (2-opt; Or-opt; single-client relocation),
    tested in isolation against the plain 2-opt baseline rather than
    stacked onto use_or_opt, matching how use_or_opt itself was first
    tested alone. Evaluated alongside 2-opt in the same best-improvement
    comparison, so it cannot do worse than 2-opt-only AT ANY SINGLE STEP
    for the same reason use_or_opt cannot -- but unlike the 2-/3-client
    case, a hand-built example showed plain 2-opt, given enough
    iterations, can reach the SAME final local optimum via a sequence of
    smaller reversals; no small-scale proof of irreplaceability is claimed
    for this lever, its value is assessed empirically via the fairness
    campaign.

    use_two_opt (default True -- production behaviour unchanged): when
    False, the 2-opt candidate scan is skipped entirely, isolating
    whichever of use_or_opt/use_single_relocation is enabled as the ONLY
    neighbourhood searched -- answers "does Or-opt/relocation help on its
    own, without 2-opt riding along," a different question from "does
    adding it to 2-opt help" (which use_or_opt/use_single_relocation with
    use_two_opt left at its default already answers). If use_two_opt,
    use_or_opt and use_single_relocation are all False, no candidate is
    ever generated and the route is returned unchanged (same as calling
    with every flag at its default).

    use_inter_route_relocate (default False -- ablation-only): runs
    _repair_inter_route_relocate FIRST (relocating single clients ACROSS
    routes, the advisor's Niveau-2 "relocate d'un client vers un autre
    vehicule" move), before the intra-route 2-opt/Or-opt/single-relocation
    loop below -- a fundamentally different neighbourhood (touches which
    TRUCK serves a client, not just visit order within one truck's route),
    so it is a separate pass rather than folded into the per-route loop
    here. See _repair_inter_route_relocate's own docstring.

    use_route_swap (default False -- ablation-only): runs
    _repair_route_swap next (client-for-client exchange between two
    DIFFERENT routes, the advisor's Niveau-2 "swap entre deux tournees"
    move), after use_inter_route_relocate and before the intra-route loop
    -- see _repair_route_swap's own docstring.

    use_delivery_shift (default False -- ablation-only): runs
    _repair_delivery_shift last (partial quantity shift between two
    ADJACENT periods a client is already served in, the advisor's Niveau-3
    "deplacement partiel d'une livraison vers une periode voisine" move) --
    a fundamentally different neighbourhood from every other lever here
    (inventory timing, not route topology) -- see _repair_delivery_shift's
    own docstring.
    """
    if use_inter_route_relocate:
        route_result = _repair_inter_route_relocate(route_result, sets_, params_)
    if use_route_swap:
        route_result = _repair_route_swap(route_result, sets_, params_)
    if use_delivery_shift:
        route_result = _repair_delivery_shift(route_result, sets_, params_)

    working = dict(route_result)
    working["x"] = dict(route_result["x"])
    working["f"] = dict(route_result["f"])
    working["arrival_times"] = dict(route_result["arrival_times"])
    working["tau_return"] = dict(route_result["tau_return"])
    working["truck_assign"] = dict(route_result["truck_assign"])
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
                if use_two_opt:
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

                if use_or_opt:
                    for start, q, candidate in _or_opt_candidates(path, max_window=_OR_OPT_WINDOW):
                        evaluated = _evaluate_candidate(
                            candidate, qty_on_route, t, k, tau_return_before, params_
                        )
                        if evaluated is None:
                            continue
                        trial_x, trial_f, trial_arrivals, trial_contrib = evaluated
                        if trial_contrib < best_contrib:
                            best_contrib = trial_contrib
                            best = (candidate, trial_x, trial_f, trial_arrivals, trial_contrib)

                if use_single_relocation:
                    for start, q, candidate in _or_opt_candidates(
                        path, segment_lengths=_SINGLE_RELOCATION_LENGTH, max_window=_OR_OPT_WINDOW
                    ):
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


def _inter_route_relocate_candidates(routes: dict, requires_cold: dict, frigo_trucks: set,
                                      Q: dict, t):
    """Yield every valid single-client relocation from one truck's route to a
    DIFFERENT truck's route within the same period t -- the advisor's Niveau-2
    "relocate d'un client vers un autre vehicule" move.

    Unlike _two_opt_candidates/_or_opt_candidates (always structurally valid
    regardless of domain state), a candidate here is filtered against TWO
    domain constraints that make it a legal pair of routes in the first
    place, not just a matter of cost:
      - frigo/non-frigo compatibility: a client whose period-t demand
        requires cold storage (requires_cold[client, t]) can only move to a
        frigo truck, and vice versa -- moving it across the divide would
        violate the model's vehicle-compatibility constraint (c14).
      - target vehicle capacity: the target route's total load, plus the
        relocated client's own quantity, must not exceed Q[k_to] (c5).

    Yields (k_from, client, k_to, new_path_from, new_path_to, qty) for every
    (source truck, client, target truck, insertion position) combination
    that respects both. new_path_from has the client removed (order of the
    remaining clients preserved); new_path_to has it inserted at every
    possible interior position (same insertion sweep as _or_opt_candidates).
    Pure with respect to cost/distance -- never inspects d/c_ijk/ET/LT, only
    the two structural-validity constraints above.
    """
    truck_is_frigo = {k: (k in frigo_trucks) for k in routes}

    for k_from, info_from in routes.items():
        path_from = info_from["path"]
        qty_from  = {int(l): q for l, q in info_from["qty"].items()}

        for client, q in qty_from.items():
            client_needs_frigo = requires_cold[client, t]
            new_path_from = [n for n in path_from if n != client]

            for k_to, info_to in routes.items():
                if k_to == k_from:
                    continue
                if truck_is_frigo[k_to] != client_needs_frigo:
                    continue

                qty_to = {int(l): qq for l, qq in info_to["qty"].items()}
                if sum(qty_to.values()) + q > Q[k_to]:
                    continue

                path_to = info_to["path"]
                for pos in range(len(path_to) - 1):
                    new_path_to = path_to[:pos + 1] + [client] + path_to[pos + 1:]
                    yield k_from, client, k_to, new_path_from, new_path_to, q


_INTER_ROUTE_MAX_ITER = 20   # mirrors _MAX_REPAIR_ITER's own rationale.


def _repair_inter_route_relocate(route_result: dict, sets_: dict, params_: dict) -> dict:
    """Niveau-2 local search: best-improvement relocation of a single client
    from one truck's route to a DIFFERENT (frigo/capacity-compatible) truck's
    route, within the same period -- see _inter_route_relocate_candidates.

    A fundamentally different loop shape from _repair_route_result's
    intra-route moves: those improve ONE route independently of every other
    (per-(t,k) loop), so 2-opt/Or-opt/single-relocation candidates are always
    structurally valid without looking at any other route. An inter-route
    relocation necessarily touches TWO routes' costs and TWO routes' arcs at
    once, so this operates per-PERIOD (considering every truck active that
    period jointly), not per-route -- kept as its own separate pass/function
    rather than folded into _repair_route_result's loop.

    Each iteration: evaluate every candidate's COMBINED delta (new source
    route contribution + new target route contribution) - (old source + old
    target), using the same per-candidate tau_return guard as the intra-route
    moves (this period's own pre-pass worst-case travel time), and apply the
    single most-negative delta if one exists (best-improvement, same
    acceptance rule as _repair_route_result). Repeats per period until no
    improving relocation exists or _INTER_ROUTE_MAX_ITER is reached, so total
    cost within a period strictly decreases every accepted step and the loop
    cannot cycle.

    Baldwinian, like _repair_route_result: returns a NEW route_result: x, f,
    arrival_times, tau_return, routes_data, and truck_assign are updated (the
    last is never read by compute_f1..f4 -- verified against evaluator.py --
    but IS read by report_builder.py's delivery-row display, so it is kept
    accurate here rather than left stale). actual_qty and depot_stock are
    untouched: relocating a client to a different truck changes neither which
    clients are served, their quantities, nor the period they're served in,
    so both remain valid as-is.
    """
    requires_cold = params_["requires_cold"]
    frigo_trucks  = params_["frigo_trucks"]
    Q             = params_["Q"]

    working = dict(route_result)
    working["x"]            = dict(route_result["x"])
    working["f"]            = dict(route_result["f"])
    working["arrival_times"] = dict(route_result["arrival_times"])
    working["tau_return"]   = dict(route_result["tau_return"])
    working["truck_assign"] = dict(route_result["truck_assign"])
    working["routes_data"]  = {
        t: dict(routes) for t, routes in route_result["routes_data"].items()
    }

    for t in route_result["routes_data"]:
        for _ in range(_INTER_ROUTE_MAX_ITER):
            routes = working["routes_data"][t]
            if len(routes) < 2:
                break

            tau_return_before = working["tau_return"].get(t, 0.0)
            best = None   # (k_from, client, k_to, new_path_from, new_path_to,
                          #  new_qty_from, new_qty_to, eval_from, eval_to, delta)

            for k_from, client, k_to, new_path_from, new_path_to, q in \
                    _inter_route_relocate_candidates(routes, requires_cold, frigo_trucks, Q, t):
                info_from = routes[k_from]
                info_to   = routes[k_to]
                qty_from  = {int(l): qq for l, qq in info_from["qty"].items()}
                qty_to    = {int(l): qq for l, qq in info_to["qty"].items()}

                old_contrib = (
                    _route_f1_contribution(info_from["path"], working["f"],
                                            working["arrival_times"], t, k_from, params_)
                    + _route_f1_contribution(info_to["path"], working["f"],
                                              working["arrival_times"], t, k_to, params_)
                )

                new_qty_from = {l: qq for l, qq in qty_from.items() if l != client}
                new_qty_to   = dict(qty_to)
                new_qty_to[client] = q

                if len(new_path_from) <= 2:
                    eval_from = ({}, {}, {}, 0.0)   # route emptied out -- no arcs, zero cost
                else:
                    eval_from = _evaluate_candidate(
                        new_path_from, new_qty_from, t, k_from, tau_return_before, params_
                    )
                    if eval_from is None:
                        continue

                eval_to = _evaluate_candidate(
                    new_path_to, new_qty_to, t, k_to, tau_return_before, params_
                )
                if eval_to is None:
                    continue

                delta = (eval_from[3] + eval_to[3]) - old_contrib
                if delta < 0 and (best is None or delta < best[-1]):
                    best = (k_from, client, k_to, new_path_from, new_path_to,
                            new_qty_from, new_qty_to, eval_from, eval_to, delta)

            if best is None:
                break

            (k_from, client, k_to, new_path_from, new_path_to,
             new_qty_from, new_qty_to, eval_from, eval_to, _delta) = best
            x_from, f_from, arr_from, _ = eval_from
            x_to,   f_to,   arr_to,   _ = eval_to

            old_path_from = routes[k_from]["path"]
            old_path_to   = routes[k_to]["path"]

            working["x"] = _replace_route_arcs(working["x"], old_path_from, t, k_from, x_from)
            working["x"] = _replace_route_arcs(working["x"], old_path_to,   t, k_to,   x_to)
            working["f"] = _replace_route_arcs(working["f"], old_path_from, t, k_from, f_from)
            working["f"] = _replace_route_arcs(working["f"], old_path_to,   t, k_to,   f_to)
            working["arrival_times"] = {**working["arrival_times"], **arr_from, **arr_to}
            working["truck_assign"][client, t] = k_to

            if len(new_path_from) > 2:
                working["routes_data"][t][k_from] = {
                    "path": new_path_from,
                    "qty":  {str(l): qq for l, qq in new_qty_from.items()},
                }
            else:
                del working["routes_data"][t][k_from]

            working["routes_data"][t][k_to] = {
                "path": new_path_to,
                "qty":  {str(l): qq for l, qq in new_qty_to.items()},
            }

            working["tau_return"][t] = max(
                (_route_traversal_time(r["path"], k2, params_)
                 for k2, r in working["routes_data"][t].items()),
                default=0.0,
            )

    return working


def _route_swap_candidates(routes: dict, requires_cold: dict, frigo_trucks: set, Q: dict, t):
    """Yield every valid client-for-client swap between two DIFFERENT routes
    within the same period t -- the advisor's Niveau-2 "swap entre deux
    tournees" move.

    Unlike _inter_route_relocate_candidates (which searches every possible
    insertion position for the relocated client, O(route length) per
    candidate), a swap keeps each client at the exact SLOT (path index) the
    other one occupied -- e.g. path1=[0,a,x,0], path2=[0,b,y,z,0] swapping
    a<->b gives new_path1=[0,b,x,0], new_path2=[0,a,y,z,0]. This makes each
    candidate O(1) to construct instead of O(n), a deliberately cheaper move
    than relocate's insertion sweep.

    Same two structural-validity filters as relocate: frigo/non-frigo
    compatibility (a must fit k2's type, b must fit k1's type -- both
    directions, since both clients change trucks) and capacity on BOTH
    routes after the exchange (removing one client's quantity, adding the
    other's). Each unordered route pair (k1, k2) is visited once (k2 > k1
    by truck id) since a swap is symmetric -- visiting both orderings would
    yield the same candidate pairs twice.

    Yields (k1, client_a, k2, client_b, new_path1, new_path2, qty_a, qty_b).
    """
    truck_is_frigo = {k: (k in frigo_trucks) for k in routes}
    truck_ids = list(routes)

    for idx1, k1 in enumerate(truck_ids):
        info1 = routes[k1]
        path1 = info1["path"]
        qty1  = {int(l): q for l, q in info1["qty"].items()}
        load1 = sum(qty1.values())

        for k2 in truck_ids[idx1 + 1:]:
            info2 = routes[k2]
            path2 = info2["path"]
            qty2  = {int(l): q for l, q in info2["qty"].items()}
            load2 = sum(qty2.values())

            for a, qa in qty1.items():
                if requires_cold[a, t] != truck_is_frigo[k2]:
                    continue
                for b, qb in qty2.items():
                    if requires_cold[b, t] != truck_is_frigo[k1]:
                        continue
                    if load1 - qa + qb > Q[k1] or load2 - qb + qa > Q[k2]:
                        continue

                    new_path1 = [b if n == a else n for n in path1]
                    new_path2 = [a if n == b else n for n in path2]
                    yield k1, a, k2, b, new_path1, new_path2, qa, qb


def _repair_route_swap(route_result: dict, sets_: dict, params_: dict) -> dict:
    """Niveau-2 local search: best-improvement client-for-client swap between
    two DIFFERENT trucks' routes, within the same period -- see
    _route_swap_candidates. Same per-period loop shape and acceptance rule
    as _repair_inter_route_relocate (evaluate every candidate's COMBINED
    delta across both touched routes, apply the single most-negative delta
    if one exists, repeat until no improving swap exists or
    _INTER_ROUTE_MAX_ITER is reached -- total cost within a period strictly
    decreases every accepted step, so the loop cannot cycle).

    Baldwinian, like _repair_inter_route_relocate: updates x, f,
    arrival_times, tau_return, routes_data, and truck_assign (both clients'
    entries, since both change trucks). actual_qty and depot_stock are
    untouched -- a swap changes neither which clients are served, their
    quantities, nor their period, only which truck serves each.
    """
    requires_cold = params_["requires_cold"]
    frigo_trucks  = params_["frigo_trucks"]
    Q             = params_["Q"]

    working = dict(route_result)
    working["x"]             = dict(route_result["x"])
    working["f"]             = dict(route_result["f"])
    working["arrival_times"] = dict(route_result["arrival_times"])
    working["tau_return"]    = dict(route_result["tau_return"])
    working["truck_assign"]  = dict(route_result["truck_assign"])
    working["routes_data"]   = {
        t: dict(routes) for t, routes in route_result["routes_data"].items()
    }

    for t in route_result["routes_data"]:
        for _ in range(_INTER_ROUTE_MAX_ITER):
            routes = working["routes_data"][t]
            if len(routes) < 2:
                break

            tau_return_before = working["tau_return"].get(t, 0.0)
            best = None   # (k1, a, k2, b, new_path1, new_path2,
                          #  new_qty1, new_qty2, eval1, eval2, delta)

            for k1, a, k2, b, new_path1, new_path2, qa, qb in \
                    _route_swap_candidates(routes, requires_cold, frigo_trucks, Q, t):
                info1 = routes[k1]
                info2 = routes[k2]
                qty1  = {int(l): qq for l, qq in info1["qty"].items()}
                qty2  = {int(l): qq for l, qq in info2["qty"].items()}

                old_contrib = (
                    _route_f1_contribution(info1["path"], working["f"],
                                            working["arrival_times"], t, k1, params_)
                    + _route_f1_contribution(info2["path"], working["f"],
                                              working["arrival_times"], t, k2, params_)
                )

                new_qty1 = {l: qq for l, qq in qty1.items() if l != a}
                new_qty1[b] = qb
                new_qty2 = {l: qq for l, qq in qty2.items() if l != b}
                new_qty2[a] = qa

                eval1 = _evaluate_candidate(new_path1, new_qty1, t, k1, tau_return_before, params_)
                if eval1 is None:
                    continue
                eval2 = _evaluate_candidate(new_path2, new_qty2, t, k2, tau_return_before, params_)
                if eval2 is None:
                    continue

                delta = (eval1[3] + eval2[3]) - old_contrib
                if delta < 0 and (best is None or delta < best[-1]):
                    best = (k1, a, k2, b, new_path1, new_path2,
                            new_qty1, new_qty2, eval1, eval2, delta)

            if best is None:
                break

            (k1, a, k2, b, new_path1, new_path2,
             new_qty1, new_qty2, eval1, eval2, _delta) = best
            x1, f1, arr1, _ = eval1
            x2, f2, arr2, _ = eval2

            old_path1 = routes[k1]["path"]
            old_path2 = routes[k2]["path"]

            working["x"] = _replace_route_arcs(working["x"], old_path1, t, k1, x1)
            working["x"] = _replace_route_arcs(working["x"], old_path2, t, k2, x2)
            working["f"] = _replace_route_arcs(working["f"], old_path1, t, k1, f1)
            working["f"] = _replace_route_arcs(working["f"], old_path2, t, k2, f2)
            working["arrival_times"] = {**working["arrival_times"], **arr1, **arr2}
            working["truck_assign"][a, t] = k2
            working["truck_assign"][b, t] = k1

            working["routes_data"][t][k1] = {
                "path": new_path1,
                "qty":  {str(l): qq for l, qq in new_qty1.items()},
            }
            working["routes_data"][t][k2] = {
                "path": new_path2,
                "qty":  {str(l): qq for l, qq in new_qty2.items()},
            }

            working["tau_return"][t] = max(
                (_route_traversal_time(r["path"], k2b, params_)
                 for k2b, r in working["routes_data"][t].items()),
                default=0.0,
            )

    return working


def _truck_serving(routes_at_period: dict, client) -> object | None:
    """Return the truck id serving `client` at this period's routes_data
    entry, or None if the client isn't served that period at all."""
    client_str = str(client)
    for k, info in routes_at_period.items():
        if client_str in info["qty"]:
            return k
    return None


_DELIVERY_SHIFT_MAX_ITER = 20   # mirrors _MAX_REPAIR_ITER's own rationale.


def _delivery_shift_candidates(route_result: dict, sets_: dict, params_: dict):
    """Yield every valid partial-delivery shift between two ADJACENT periods
    -- the advisor's Niveau-3 "deplacement partiel d'une livraison vers une
    periode voisine" move -- for a client already served (actual_qty > 0)
    in BOTH periods with the SAME frigo/non-frigo label (a different label
    would need a different truck TYPE, not just a different truck, out of
    scope: this move never adds/removes a stop or changes which truck TYPE
    serves a client, only the quantity split between two already-existing
    stops).

    For adjacent periods t1 < t2, moving delta units of client l's delivery
    changes depot_stock at EXACTLY t1 (the earlier of the pair) by +-delta
    and leaves every other period's stock unchanged (proof: shipped[t1] and
    shipped[t2] each move by -+delta while their SUM is invariant, so the
    cumulative stock trajectory from t2 onward, which only depends on that
    sum, is unaffected -- see _repair_delivery_shift's own docstring for the
    full argument). Two directions per pair:
      - "delay"   (case A): donor=t1, receiver=t2 -- I[t1] increases by delta
        (less shipped at t1, so more stays in stock).
      - "advance" (case B): donor=t2, receiver=t1 -- I[t1] decreases by delta
        (more shipped at t1 to cover what receiver now gets early).

    Each direction's max feasible delta is capped by three independent,
    provably-safe bounds (see _repair_delivery_shift's docstring for why the
    donor-floor bound is safe): the receiving truck's capacity headroom,
    the donor's own nominal per-period demand floor (q_lt[l, donor period]:
    reducing actual_qty down to but not below this can never violate the
    cumulative-catch-up requirement, since periods before the donor are
    untouched by this move and periods from the donor onward can only gain
    slack, not lose it), and the depot's stock bound (I_max for case A,
    I_min for case B) at the one period that actually moves. Yields
    (client, t_donor, t_receiver, k_donor, k_receiver, t_early, sign,
    max_delta) -- sign is +1 for case A, -1 for case B, applied to
    depot_stock[t_early] as sign * delta.
    """
    T             = sets_["T"]
    clients       = sets_["clients"]
    requires_cold = params_["requires_cold"]
    q_lt          = params_["q_lt"]
    Q             = params_["Q"]
    I_min_f       = params_["I_O_min_frigo"]
    I_max_f       = params_["I_O_max_frigo"]
    I_min_nf      = params_["I_O_min_nonfrigo"]
    I_max_nf      = params_["I_O_max_nonfrigo"]
    actual_qty    = route_result["actual_qty"]
    depot_stock   = route_result["depot_stock"]
    routes_data   = route_result["routes_data"]

    for idx in range(len(T) - 1):
        t1, t2 = T[idx], T[idx + 1]
        routes_t1 = routes_data.get(t1, {})
        routes_t2 = routes_data.get(t2, {})

        for l in clients:
            if requires_cold[l, t1] != requires_cold[l, t2]:
                continue
            q1 = actual_qty.get((l, t1), 0)
            q2 = actual_qty.get((l, t2), 0)
            if q1 <= 0 or q2 <= 0:
                continue

            k1 = _truck_serving(routes_t1, l)
            k2 = _truck_serving(routes_t2, l)
            if k1 is None or k2 is None:
                continue

            is_frigo  = requires_cold[l, t1]
            I_min     = I_min_f if is_frigo else I_min_nf
            I_max     = I_max_f if is_frigo else I_max_nf
            stock_key = "frigo" if is_frigo else "nonfrigo"
            I_t1      = depot_stock.get(t1, {}).get(stock_key, 0.0)

            load1 = sum(int(qq) for qq in routes_t1[k1]["qty"].values())
            load2 = sum(int(qq) for qq in routes_t2[k2]["qty"].values())

            # Case A ("delay"): donor=t1/k1, receiver=t2/k2, I[t1] += delta.
            # q1 - 1 (not q1): the donor stop must keep at least 1 unit, or
            # it becomes a real visit delivering nothing -- routes_data's
            # qty dict would then disagree with the path (which still
            # contains the stop), a bookkeeping mismatch this move must
            # never introduce since it never restructures paths.
            max_delta_a = min(
                Q[k2] - load2,
                max(0, q1 - q_lt.get((l, t1), 0)),
                max(0, I_max - I_t1),
                q1 - 1,
            )
            if max_delta_a > 0:
                yield (l, t1, t2, k1, k2, t1, +1, int(max_delta_a))

            # Case B ("advance"): donor=t2/k2, receiver=t1/k1, I[t1] -= delta.
            max_delta_b = min(
                Q[k1] - load1,
                max(0, q2 - q_lt.get((l, t2), 0)),
                max(0, I_t1 - I_min),
                q2 - 1,
            )
            if max_delta_b > 0:
                yield (l, t2, t1, k2, k1, t1, -1, int(max_delta_b))


def _repair_delivery_shift(route_result: dict, sets_: dict, params_: dict) -> dict:
    """Niveau-3 local search: best-improvement partial shift of a client's
    delivered quantity between two ADJACENT periods it is already served in
    -- the advisor's "deplacement partiel d'une livraison vers une periode
    voisine" move. Distinct in kind from every Niveau-1/2 route-topology
    move already tried and rejected this session (2-opt aside): those move
    WHICH truck visits a client and in WHAT order; this one never touches
    either -- both periods' paths stay structurally identical, only the
    SPLIT of quantity between two already-existing stops moves.

    Why no multi-period stock-trajectory replay is needed: depot_stock
    follows I[t] = I[t-1] + R[t] - shipped[t]. Shifting delta units of one
    client's delivery from period t1 to the very next period t2 changes
    shipped[t1] by -delta and shipped[t2] by +delta -- opposite signs, equal
    magnitude. I[t1] = I[t1-1] + R[t1] - shipped[t1] therefore moves by
    +delta (t1-1 and R[t1] untouched). I[t2] = I[t1] + R[t2] - shipped[t2] =
    (I[t1]_old + delta) + R[t2] - (shipped[t2]_old + delta) = I[t1]_old +
    R[t2] - shipped[t2]_old = I[t2]_old exactly -- unchanged, and so is
    every period after it (the recurrence from t2 onward only ever sees
    I[t2], not the intermediate detour through t1). So this move touches
    depot_stock at exactly ONE period, a direct O(1) adjustment rather than
    a decoder re-run.

    f3 (pure distance/speed) is untouched -- x_vars/paths don't change. f2
    (CO2) and f1's y1 (transport) DO move slightly: the arc flow f_vars
    along both (unchanged) paths is a suffix-sum of quantities that now
    includes a different amount for this client, computed here via
    _evaluate_candidate on the SAME path (paths structurally identical, so
    tau_return/arrival_times cannot change -- verified neither
    _evaluate_candidate's timing pass nor _route_traversal_time reads
    qty_on_route). f1's y2 (holding cost) moves by h_O * delta at the one
    affected period. f4's stock component moves with depot_stock too; f4's
    receivables/payables are invariant to WHEN a fixed total quantity is
    delivered (both depend only on each client's TOTAL delivered quantity
    across the horizon, which this move conserves by construction -- it is
    zero-sum per client) -- not tracked here since the acceptance rule,
    matching every other repair in this module, is f1 only.

    Every cost term here (transport, holding) is LINEAR in the shifted
    delta, so an improving candidate's best delta is always the MAXIMUM
    feasible one (capacity/floor/stock-bound-limited, see
    _delivery_shift_candidates) -- no diminishing returns to search over,
    unlike topology moves where a fixed-size neighbourhood is re-scanned.

    Baldwinian: updates x, f, actual_qty, depot_stock, and routes_data (both
    touched periods' qty dicts). tau_return and arrival_times are untouched,
    per the timing argument above.
    """
    working = dict(route_result)
    working["x"]           = dict(route_result["x"])
    working["f"]           = dict(route_result["f"])
    working["actual_qty"]  = dict(route_result["actual_qty"])
    working["depot_stock"] = {t: dict(v) for t, v in route_result["depot_stock"].items()}
    working["routes_data"] = {
        t: dict(routes) for t, routes in route_result["routes_data"].items()
    }
    requires_cold = params_["requires_cold"]

    for _ in range(_DELIVERY_SHIFT_MAX_ITER):
        best = None   # (delta_f1, l, t_donor, t_receiver, k_donor, k_receiver,
                      #  t_early, sign, delta, eval_donor, eval_receiver,
                      #  new_qty_donor, new_qty_receiver)

        for l, t_donor, t_receiver, k_donor, k_receiver, t_early, sign, delta in \
                _delivery_shift_candidates(working, sets_, params_):
            info_donor    = working["routes_data"][t_donor][k_donor]
            info_receiver = working["routes_data"][t_receiver][k_receiver]
            qty_donor    = {int(x): qq for x, qq in info_donor["qty"].items()}
            qty_receiver = {int(x): qq for x, qq in info_receiver["qty"].items()}

            tau_before_donor    = working["tau_return"].get(t_donor, 0.0)
            tau_before_receiver = working["tau_return"].get(t_receiver, 0.0)

            old_contrib = (
                _route_f1_contribution(info_donor["path"], working["f"],
                                        working["arrival_times"], t_donor, k_donor, params_)
                + _route_f1_contribution(info_receiver["path"], working["f"],
                                          working["arrival_times"], t_receiver, k_receiver, params_)
            )

            new_qty_donor    = dict(qty_donor)
            new_qty_donor[l] = qty_donor[l] - delta
            new_qty_receiver = dict(qty_receiver)
            new_qty_receiver[l] = qty_receiver.get(l, 0) + delta

            eval_donor = _evaluate_candidate(
                info_donor["path"], new_qty_donor, t_donor, k_donor, tau_before_donor, params_
            )
            if eval_donor is None:
                continue
            eval_receiver = _evaluate_candidate(
                info_receiver["path"], new_qty_receiver, t_receiver, k_receiver, tau_before_receiver, params_
            )
            if eval_receiver is None:
                continue

            delta_holding = params_["h_O"] * sign * delta
            delta_f1 = (eval_donor[3] + eval_receiver[3] - old_contrib) + delta_holding

            if delta_f1 < 0 and (best is None or delta_f1 < best[0]):
                best = (delta_f1, l, t_donor, t_receiver, k_donor, k_receiver,
                        t_early, sign, delta, eval_donor, eval_receiver,
                        new_qty_donor, new_qty_receiver)

        if best is None:
            break

        (_delta_f1, l, t_donor, t_receiver, k_donor, k_receiver,
         t_early, sign, delta, eval_donor, eval_receiver,
         new_qty_donor, new_qty_receiver) = best

        info_donor    = working["routes_data"][t_donor][k_donor]
        info_receiver = working["routes_data"][t_receiver][k_receiver]
        x_d, f_d, _arr_d, _ = eval_donor
        x_r, f_r, _arr_r, _ = eval_receiver

        working["x"] = _replace_route_arcs(working["x"], info_donor["path"], t_donor, k_donor, x_d)
        working["x"] = _replace_route_arcs(working["x"], info_receiver["path"], t_receiver, k_receiver, x_r)
        working["f"] = _replace_route_arcs(working["f"], info_donor["path"], t_donor, k_donor, f_d)
        working["f"] = _replace_route_arcs(working["f"], info_receiver["path"], t_receiver, k_receiver, f_r)

        working["routes_data"][t_donor][k_donor] = {
            "path": info_donor["path"],
            "qty":  {str(x): qq for x, qq in new_qty_donor.items() if qq > 0},
        }
        working["routes_data"][t_receiver][k_receiver] = {
            "path": info_receiver["path"],
            "qty":  {str(x): qq for x, qq in new_qty_receiver.items()},
        }

        working["actual_qty"][l, t_donor]    = new_qty_donor[l]
        working["actual_qty"][l, t_receiver] = new_qty_receiver[l]

        stock_key = "frigo" if requires_cold[l, t_early] else "nonfrigo"
        working["depot_stock"][t_early][stock_key] = round(
            working["depot_stock"][t_early][stock_key] + sign * delta, 4
        )

    return working
