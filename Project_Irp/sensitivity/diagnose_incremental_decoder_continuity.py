"""Smoke test (not a performance comparison): does an INCREMENTAL decoder --
edit the parent's already-built route by a bounded number of local moves,
instead of reconstructing from scratch -- give theta a smooth, continuous
route landscape to search, unlike every from-scratch decoder tried so far?

Context: `diagnose_giant_tour_continuity.py` and `compare_damped_priority.py`
both kept the "decode the whole chromosome from scratch every generation"
structure -- only WHAT drives the choice changed (distance-only vs
priority-sorted, undamped vs damped priority). Both still let a single
priority change cascade through an unbounded number of downstream
decisions, because nothing caps how much of the route can change. Giant-tour
failed outright (worse continuity AND ~4x worse cost); damped-priority
looked promising at 3 seeds but dissolved at 7 (likely a small-sample false
positive, same pattern as the rejected archive-repair remedy).

This tests a structurally different idea: instead of decoding fresh, START
from the PARENT's already-decoded route and apply a BOUNDED number of local
"relocate" moves (remove one client, reinsert it at its cheapest feasible
position in its own existing (period, truck) route -- reuses
`Solvers/QINSGA3/repair.py`'s already-validated, decoder-agnostic
`_evaluate_candidate`/`_replace_route_arcs`/`_route_traversal_time`
UNCHANGED, no new route-cost machinery invented). The number of clients
relocated is capped and scales with the rotation magnitude alpha (small
alpha -> 1 move, large alpha -> up to K_MAX moves) -- which clients get
relocated is chosen by which ones the perturbation actually changed the most
(largest |priority delta|), so the edit targets exactly what the rotation
was trying to move, nothing else. This makes continuity a GUARANTEE of the
construction (blast radius capped by construction) rather than an emergent,
fragile property of a from-scratch decoder -- the structural gap identified
in both prior attempts.

Isolation: `Solvers/NSGA3/decoder.py` and `Solvers/QINSGA3/repair.py` are
NEVER modified -- this only calls repair.py's existing private helpers
(already used, tested, and validated in production for a different purpose
-- post-decode 2-opt repair) on a new set of candidate paths (single-client
relocation instead of 2-opt segment reversal).

What this script measures, using REAL cached NSGA-III chromosomes (100-client
instance, same source as the other two decoder smoke tests): at several
alpha values along the real schedule, compares (a) Jaccard route-distance and
(b) f1 cost, between the baseline route and (1) a FULL from-scratch
reconstruction of the perturbed chromosome (what production does today) vs
(2) this incremental, capped-relocation edit of the SAME baseline route.

Usage:
    python -m sensitivity.diagnose_incremental_decoder_continuity
"""
from __future__ import annotations

import json
import os
import sys

MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(MODULE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np

from models.parametres          import load_instance
from Solvers.NSGA3.decoder       import decode_chromosome, build_routes
from Solvers.NSGA3.evaluator     import compute_f1
from Solvers.NSGA3.problem       import IRPProblem
from Solvers.QINSGA3.algorithm   import _encode_theta
from Solvers.QINSGA3.repair      import (
    _evaluate_candidate, _replace_route_arcs, _route_traversal_time,
)
from sensitivity.test_route_diversity import _route_arcset, _jaccard_distance

INSTANCE   = "100"
SEEDS      = [42, 137, 271, 491, 613]
MAX_GEN    = 300
ALPHA_MAX  = 0.10 * np.pi
ALPHA_MIN  = 0.001 * np.pi
GENS_TO_TEST = [0, 150, 300]
K_MAX      = 20      # clients relocated at the largest alpha tested
RNG_SEED   = 12345

_NSGA3_CACHE = os.path.join(PROJECT_DIR, "Solvers", "NSGA3", "nsga3_chromosomes.json")


def _relocate_client(path, qty_on_route, client, t, k, tau_return_before, params_):
    """Best-insertion relocate of a single client within its OWN existing
    (t, k) route -- remove it, try every reinsertion position, keep the one
    minimising the route's f1 contribution (reuses repair.py's
    _evaluate_candidate exactly as the 2-opt repair loop does, just with
    relocate candidates instead of segment-reversal candidates). Returns
    None if the client isn't on this path, the route is too short to have
    an interior, or no feasible position exists (tau_max guard)."""
    if client not in path or len(path) <= 3:
        return None
    stripped = [n for n in path if n != client]
    best = None
    for pos in range(1, len(stripped)):
        candidate = stripped[:pos] + [client] + stripped[pos:]
        evaluated = _evaluate_candidate(candidate, qty_on_route, t, k, tau_return_before, params_)
        if evaluated is None:
            continue
        trial_x, trial_f, trial_arrivals, trial_contrib = evaluated
        if best is None or trial_contrib < best[-1]:
            best = (candidate, trial_x, trial_f, trial_arrivals, trial_contrib)
    return best


def _incremental_decode(route_result, priorities_before, priorities_after, k_moves, params_):
    """Bounded incremental edit of an already-decoded route_result: relocate
    the k_moves clients whose priority changed the MOST (largest
    |priority_after - priority_before|) to their cheapest feasible position
    within their own existing route -- never rebuilds anything from scratch,
    every other client's position is untouched. Baldwinian in spirit like
    repair.py: only x/f/arrival_times/routes_data/tau_return are updated."""
    all_clients = set(priorities_before) | set(priorities_after)
    deltas = sorted(
        all_clients,
        key=lambda l: -abs(priorities_after.get(l, 0.5) - priorities_before.get(l, 0.5)),
    )
    top_k = deltas[:k_moves]

    working = dict(route_result)
    working["x"]             = dict(route_result["x"])
    working["f"]              = dict(route_result["f"])
    working["arrival_times"]  = dict(route_result["arrival_times"])
    working["tau_return"]     = dict(route_result["tau_return"])
    working["routes_data"]    = {t: dict(routes) for t, routes in route_result["routes_data"].items()}

    location = {}
    for t, routes in route_result["routes_data"].items():
        for k, info in routes.items():
            for l_str in info["qty"]:
                location[int(l_str)] = (t, k)

    for client in top_k:
        if client not in location:
            continue
        t, k = location[client]
        info = working["routes_data"][t][k]
        path = list(info["path"])
        qty_on_route = {int(l): q for l, q in info["qty"].items()}
        tau_before = working["tau_return"].get(t, 0.0)

        result = _relocate_client(path, qty_on_route, client, t, k, tau_before, params_)
        if result is None:
            continue
        candidate, trial_x, trial_f, trial_arrivals, _ = result
        working["x"]            = _replace_route_arcs(working["x"], path, t, k, trial_x)
        working["f"]            = _replace_route_arcs(working["f"], path, t, k, trial_f)
        working["arrival_times"] = {**working["arrival_times"], **trial_arrivals}
        working["routes_data"][t][k] = {"path": candidate, "qty": info["qty"]}
        working["tau_return"][t] = max(
            _route_traversal_time(r["path"], k2, params_)
            for k2, r in working["routes_data"][t].items()
        )

    return working


def main() -> None:
    data_path = os.path.join(PROJECT_DIR, "data", f"instance_{INSTANCE}_clients.json")
    sets_, params_ = load_instance(data_path)
    problem = IRPProblem(sets_, params_)
    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)

    with open(_NSGA3_CACHE, encoding="utf-8") as f:
        cache = json.load(f)
    by_seed = {r["seed"]: r for r in cache["runs"]}
    baseline_X = []
    for sd in SEEDS:
        baseline_X.extend(by_seed[sd]["chromosomes"])
    baseline_X = np.array(baseline_X, dtype=float)
    N = len(baseline_X)
    theta = _encode_theta(baseline_X, xl, xu)
    print(f"N={N} (5 seeds, cache NSGA-III, instance {INSTANCE})")

    print("\nDecodage des baselines (une seule fois, route de reference)...")
    base_routes, base_arcs, base_f1, base_prios = [], [], [], []
    for i in range(N):
        q, p = decode_chromosome(baseline_X[i], sets_)
        rr = build_routes(q, sets_, params_, p)
        base_routes.append(rr)
        base_arcs.append(_route_arcset(rr))
        base_f1.append(compute_f1(rr, sets_, params_))
        base_prios.append(p)
    print(f"  f1 moyen baseline (sans perturbation) : {np.mean(base_f1):.1f}")

    print("\n" + "=" * 100)
    print("  CONTINUITE ET COUT : reconstruction complete vs edition incrementale bornee")
    print("=" * 100)
    print(f"  {'gen':<6} {'alpha':<9} {'k_moves':<9} "
          f"{'Jaccard complet':<18} {'Jaccard increment.':<20} "
          f"{'f1 complet':<14} {'f1 increment.':<14}")
    print("-" * 100)

    rng = np.random.default_rng(RNG_SEED)
    for gen in GENS_TO_TEST:
        alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * (1.0 - gen / MAX_GEN)
        k_moves = max(1, round(alpha / ALPHA_MAX * K_MAX))

        guide_idx = rng.integers(0, N, size=N)
        same = guide_idx == np.arange(N)
        guide_idx[same] = (guide_idx[same] + 1) % N
        theta_guide = theta[guide_idx]
        diff = theta_guide - theta
        theta_pert = np.clip(theta + alpha * np.tanh(diff / (np.pi / 8.0)), 0.0, np.pi / 2.0)
        p_pert = np.cos(theta_pert) ** 2
        X_pert = np.clip(xl + p_pert * (xu - xl), xl, xu)

        d_full, d_incr, f1_full, f1_incr = [], [], [], []
        for i in range(N):
            q_pert, prio_pert = decode_chromosome(X_pert[i], sets_)

            rr_full = build_routes(q_pert, sets_, params_, prio_pert)
            d_full.append(_jaccard_distance(base_arcs[i], _route_arcset(rr_full)))
            f1_full.append(compute_f1(rr_full, sets_, params_))

            rr_incr = _incremental_decode(base_routes[i], base_prios[i], prio_pert, k_moves, params_)
            d_incr.append(_jaccard_distance(base_arcs[i], _route_arcset(rr_incr)))
            f1_incr.append(compute_f1(rr_incr, sets_, params_))

        print(f"  {gen:<6} {alpha:<9.5f} {k_moves:<9} "
              f"{np.mean(d_full):<18.4f} {np.mean(d_incr):<20.4f} "
              f"{np.mean(f1_full):<14.1f} {np.mean(f1_incr):<14.1f}")

    print("-" * 100)
    print(f"  f1 baseline (sans perturbation, reference) : {np.mean(base_f1):.1f}")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
