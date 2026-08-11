"""Smoke test: does damping priority's leverage in the decoder's score
formula restore continuity, without sacrificing route quality the way
giant-tour+split did?

Mechanism: `Solvers/NSGA3/decoder.py:331` scores candidates with
`dist / (0.5 + priority)`. A priority perturbation can flip which candidate
wins a close-distance race, changing `current`, cascading forward -- the
root cause already established. Giant-tour+split removed distance from the
choice entirely and made it WORSE (0.19 vs 0.08 Jaccard at the smallest
alpha, plus ~4x worse f1) because the split step is itself just as
sequential/cascading.

This tests the opposite move: keep the exact same greedy, distance-driven
construction (so route quality stays close to production), but DAMP how
much a priority perturbation can move the score -- interpolate priority
toward a neutral constant so it can only break already-close distance ties,
never flip a clearly-better geographic choice.
  score = dist / (0.5 + w*priority + (1-w)*0.5),  w in [0, 1]
  w=1.0 -> identical to production. w=0.0 -> priority has zero effect
  (pure nearest-neighbour, no diversity source left).

Isolation: `Solvers/NSGA3/decoder.py` untouched on disk -- same monkey-patch
technique already verified byte-identical elsewhere in this project.

Usage:
    python -m sensitivity.diagnose_damped_priority_continuity
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
import Solvers.NSGA3.decoder as decoder_mod
from Solvers.NSGA3.decoder      import decode_chromosome
from Solvers.NSGA3.evaluator    import compute_f1
from Solvers.NSGA3.problem      import IRPProblem
from Solvers.QINSGA3.algorithm  import _encode_theta
from sensitivity.test_route_diversity import _route_arcset, _jaccard_distance

INSTANCE   = "100"
SEEDS      = [42, 137, 271, 491, 613]
MAX_GEN    = 300
ALPHA_MAX  = 0.10 * np.pi
ALPHA_MIN  = 0.001 * np.pi
GENS_TO_TEST = [0, 150, 300]
WEIGHTS_TO_TEST = [1.0, 0.5, 0.2, 0.05]
RNG_SEED   = 12345

_NSGA3_CACHE = os.path.join(PROJECT_DIR, "Solvers", "NSGA3", "nsga3_chromosomes.json")


def _make_damped(w: float):
    def _nn_damped(qty_dict, trucks, t, d, v, s, Q, O, tau_max=None, floors=None, priorities=None):
        prios     = priorities or {}
        pending   = dict(qty_dict)
        truck_idx = 0
        x_vars, f_vars, arrivals, assign, routes = {}, {}, {}, {}, {}

        while pending and truck_idx < len(trucks):
            k, cap, speed = trucks[truck_idx], Q[trucks[truck_idx]], v[trucks[truck_idx]]
            path, qty_on_route, load, current_time, current = [O], {}, 0, 0.0, O

            while True:
                best, best_score, best_qty = None, float("inf"), 0
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
                        projected = (current_time + s.get(current, 0.0) + dist / speed
                                     + s.get(l, 0.0) + d[l, O] / speed)
                        if projected > tau_max:
                            continue
                    p_raw   = prios.get(l, 0.5)
                    p_damped = w * p_raw + (1 - w) * 0.5
                    score = dist / (0.5 + p_damped)
                    if score < best_score:
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
                m   = len(path)
                suf = [0] * (m + 1)
                for idx in range(m - 2, -1, -1):
                    node    = path[idx + 1]
                    suf[idx] = suf[idx + 1] + (qty_on_route.get(node, 0) if node != O else 0)
                for idx in range(m - 1):
                    i, j               = path[idx], path[idx + 1]
                    x_vars[i, j, t, k] = 1
                    f_vars[i, j, t, k] = suf[idx]
                routes[k] = {"path": path, "qty": {str(l): q for l, q in qty_on_route.items()}}
            truck_idx += 1

        return routes, x_vars, f_vars, arrivals, assign
    return _nn_damped


def _build_routes_damped(quantities, sets_, params_, priorities, w):
    original = decoder_mod._nearest_neighbour
    decoder_mod._nearest_neighbour = _make_damped(w)
    try:
        return decoder_mod.build_routes(quantities, sets_, params_, priorities)
    finally:
        decoder_mod._nearest_neighbour = original


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

    # baseline quality per weight (f1 on the unperturbed baselines)
    print("\nQualite de base (f1 moyen, sans perturbation) :")
    base_routes = {}
    for w in WEIGHTS_TO_TEST:
        f1s, arcs = [], []
        for i in range(N):
            q, p = decode_chromosome(baseline_X[i], sets_)
            rr = _build_routes_damped(q, sets_, params_, p, w)
            f1s.append(compute_f1(rr, sets_, params_))
            arcs.append(_route_arcset(rr))
        base_routes[w] = arcs
        print(f"  w={w:<5} f1 moyen={np.mean(f1s):10.1f}")

    print("\nContinuite (Jaccard moyen, base vs perturbe) :")
    print(f"  {'gen':<6} {'alpha':<10}" + "".join(f"w={w:<10}" for w in WEIGHTS_TO_TEST))

    for gen in GENS_TO_TEST:
        alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * (1.0 - gen / MAX_GEN)
        rng = np.random.default_rng(RNG_SEED)   # same perturbation draw at each gen, across weights
        guide_idx = rng.integers(0, N, size=N)
        same = guide_idx == np.arange(N)
        guide_idx[same] = (guide_idx[same] + 1) % N
        theta_guide = theta[guide_idx]
        diff = theta_guide - theta
        theta_pert = np.clip(theta + alpha * np.tanh(diff / (np.pi / 8.0)), 0.0, np.pi / 2.0)
        p_pert = np.cos(theta_pert) ** 2
        X_pert = np.clip(xl + p_pert * (xu - xl), xl, xu)

        row = f"  {gen:<6} {alpha:<10.5f}"
        for w in WEIGHTS_TO_TEST:
            dists = []
            for i in range(N):
                q, p = decode_chromosome(X_pert[i], sets_)
                rr = _build_routes_damped(q, sets_, params_, p, w)
                dists.append(_jaccard_distance(base_routes[w][i], _route_arcset(rr)))
            row += f"{np.mean(dists):<12.4f}"
        print(row)


if __name__ == "__main__":
    main()
