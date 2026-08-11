"""Diagnostic (not a performance comparison): does the REAL per-step decoder
decision margin -- not a static geographic proxy -- predict decoder chaos?

Context: `sensitivity/diagnose_distance_separation_vs_chaos.py` tested whether
geographic client separation (ratio of 2nd/1st nearest distance among ALL
other nodes) predicts route chaos under a rotation-gate-style perturbation.
Result: r=+0.24 across only 7 instances -- wrong sign relative to the
predicted mechanism (near-ties should make chaos WORSE, i.e. separation
should correlate NEGATIVELY with chaos), weak, and statistically powerless
at n=7 regardless of sign.

Three flaws in that proxy, identified on review:
  1. It compares raw DISTANCE, but the real decoder score
     (`Solvers/NSGA3/decoder.py:331`) is `dist / (0.5 + priority)` -- a
     priority difference can offset or amplify a distance gap.
  2. It compares a client against ALL other nodes in the instance, but the
     decoder only ever chooses among clients still PENDING and FEASIBLE at
     that exact construction step (remaining truck capacity, `tau_max` time
     window, mandatory-floor priority) -- a set that shrinks as the route is
     built, not a static property of the whole instance.
  3. (Statistical, not conceptual) it correlated only 7 points (one per
     instance size) -- essentially no power regardless of the true effect.

This script fixes all three: a private instrumented fork of
`_nearest_neighbour` (production `Solvers/NSGA3/decoder.py` is NEVER
modified on disk -- monkey-patched onto the imported module only for the
duration of this script's own decode calls, restored in a `finally` block
immediately after) records, at every construction step, the margin between
the best and second-best FEASIBLE candidate's real score. Verified
behavior-equivalent to the original's two-branch mandatory/non-mandatory
logic -- see the note above `_nearest_neighbour_instrumented`.

For each of N_SAMPLES random chromosomes per instance, the mean margin along
its OWN decode trajectory is correlated, per-chromosome (not per-instance),
against that same chromosome's chaos (mean Jaccard route-distance after one
production-typical rotation-gate perturbation -- identical perturbation
methodology to `diagnose_distance_separation_vs_chaos.py`, reusing
`test_route_diversity.py`'s `_route_arcset`/`_jaccard_distance` unmodified).
Pooling all instances gives N_SAMPLES*7 data points instead of 7.

Usage:
    python -m sensitivity.diagnose_decision_margin_vs_chaos
"""
from __future__ import annotations

import os
import sys

MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(MODULE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np
from scipy.stats import pearsonr

from models.parametres          import load_instance
import Solvers.NSGA3.decoder as decoder_mod
from Solvers.NSGA3.decoder      import decode_chromosome
from Solvers.NSGA3.problem      import IRPProblem
from sensitivity.test_route_diversity import _route_arcset, _jaccard_distance

INSTANCES  = ["3", "5", "15", "25", "30", "40", "100"]
N_SAMPLES  = 15
ALPHA_MID  = 0.0505 * np.pi   # same production-typical alpha as diagnose_distance_separation_vs_chaos.py
RNG_SEED   = 20260806


def _nearest_neighbour_instrumented(qty_dict, trucks, t, d, v, s, Q, O, tau_max=None,
                                     floors=None, priorities=None, margins_out=None):
    """Behavior-identical fork of decoder.py::_nearest_neighbour.

    Verified equivalent: within a single construction step the evaluated
    candidate pool is always homogeneous -- all-mandatory or
    all-non-mandatory, never mixed, because non-mandatory candidates are
    filtered out (`continue`) whenever any mandatory one is pending. So a
    plain running argmin/second-argmin over whichever pool survives the
    filters is exactly equivalent to the original's two-branch
    mandatory/non-mandatory selection (both start comparing against
    best_score=inf, so "first candidate always becomes best" holds in both).
    Only addition: records (second_best_score - best_score) into
    margins_out at every step that has a runner-up.
    """
    x_vars, f_vars, arrivals, assign, routes = {}, {}, {}, {}, {}
    prios     = priorities or {}
    pending   = dict(qty_dict)
    truck_idx = 0

    while pending and truck_idx < len(trucks):
        k     = trucks[truck_idx]
        cap   = Q[k]
        speed = v[k]

        path, qty_on_route, load, current_time, current = [O], {}, 0, 0.0, O

        while True:
            best, best_score, best_qty = None, float("inf"), 0
            second_score = float("inf")
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
                if score < best_score:
                    second_score = best_score
                    best_score, best, best_qty = score, l, q_effective
                elif score < second_score:
                    second_score = score

            if best is None:
                break

            if second_score < float("inf") and margins_out is not None:
                margins_out.append(second_score - best_score)

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


def _build_routes_with_margins(quantities, sets_, params_, priorities, margins_out):
    """Runs the REAL build_routes (production code, unmodified) with
    `_nearest_neighbour` monkey-patched at the module level to the
    instrumented fork above, for the duration of this single call only --
    always restored in `finally`, even on exception. Everything else in
    build_routes (frigo/nonfrigo splitting, stock ceilings, C6/C8
    constraints) runs byte-identical to production; only the routing
    construction step is observed, never altered.
    """
    original = decoder_mod._nearest_neighbour

    def _wrapped(qty_dict, trucks, t, d, v, s, Q, O, tau_max=None, floors=None, priorities=None):
        return _nearest_neighbour_instrumented(
            qty_dict, trucks, t, d, v, s, Q, O, tau_max, floors, priorities, margins_out)

    decoder_mod._nearest_neighbour = _wrapped
    try:
        return decoder_mod.build_routes(quantities, sets_, params_, priorities)
    finally:
        decoder_mod._nearest_neighbour = original


def _encode_theta(X, xl, xu):
    p = np.clip((X - xl) / np.where(xu - xl > 1e-12, xu - xl, 1.0), 0.0, 1.0)
    return np.clip(np.arccos(np.sqrt(p)), 0.0, np.pi / 2.0)


def _sample_margin_and_chaos(sets_: dict, params_: dict, rng: np.random.Generator):
    """For N_SAMPLES random chromosomes: mean REAL decision margin along its
    own decode trajectory, and chaos (Jaccard route-distance) after one
    production-typical rotation-gate perturbation."""
    problem = IRPProblem(sets_, params_)
    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)
    n_var = problem.n_var

    X = xl + rng.random((N_SAMPLES, n_var)) * (xu - xl)
    theta = _encode_theta(X, xl, xu)

    guide_idx = rng.integers(0, N_SAMPLES, size=N_SAMPLES)
    same = guide_idx == np.arange(N_SAMPLES)
    guide_idx[same] = (guide_idx[same] + 1) % N_SAMPLES
    theta_guide = theta[guide_idx]

    diff = theta_guide - theta
    theta_pert = np.clip(theta + ALPHA_MID * np.tanh(diff / (np.pi / 8.0)), 0.0, np.pi / 2.0)
    p = np.cos(theta_pert) ** 2
    X_pert = np.clip(xl + p * (xu - xl), xl, xu)

    margins, chaoses = [], []
    for i in range(N_SAMPLES):
        q0, p0 = decode_chromosome(X[i], sets_)
        step_margins = []
        rr0 = _build_routes_with_margins(q0, sets_, params_, p0, step_margins)

        q1, p1 = decode_chromosome(X_pert[i], sets_)
        rr1 = decoder_mod.build_routes(q1, sets_, params_, p1)

        margins.append(float(np.mean(step_margins)) if step_margins else float("nan"))
        chaoses.append(_jaccard_distance(_route_arcset(rr0), _route_arcset(rr1)))

    return margins, chaoses


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    print("=" * 96)
    print("  DIAGNOSTIC -- marge de decision REELLE du decodeur vs chaos (par chromosome, pas par instance)")
    print("=" * 96)
    print(f"  {'Instance':<10} {'N valides':<12} {'Marge moyenne':<18} {'Chaos moyen':<14} {'r (par instance)':<12}")
    print("-" * 96)

    all_margins, all_chaoses = [], []
    for inst in INSTANCES:
        data_path = os.path.join(PROJECT_DIR, "data", f"instance_{inst}_clients.json")
        sets_, params_ = load_instance(data_path)
        margins, chaoses = _sample_margin_and_chaos(sets_, params_, rng)

        pairs = [(m, c) for m, c in zip(margins, chaoses) if not np.isnan(m)]
        m_arr = np.array([p[0] for p in pairs])
        c_arr = np.array([p[1] for p in pairs])
        r_inst = float(np.corrcoef(m_arr, c_arr)[0, 1]) if len(pairs) >= 3 else float("nan")

        all_margins.extend(m_arr.tolist())
        all_chaoses.extend(c_arr.tolist())

        print(f"  {inst:<10} {len(pairs):<12} {m_arr.mean():<18.4f} {c_arr.mean():<14.4f} {r_inst:<12.4f}")

    print("-" * 96)
    all_margins = np.array(all_margins)
    all_chaoses = np.array(all_chaoses)
    r, pval = pearsonr(all_margins, all_chaoses)

    print(f"\nCorrelation POOLEE (marge de decision, chaos), niveau CHROMOSOME, n={len(all_margins)} : "
          f"r = {r:.4f}, p = {pval:.6f}")
    print("\nINTERPRETATION")
    if r < -0.2 and pval < 0.05:
        print("  Correlation negative et significative : plus la marge de decision moyenne est faible")
        print("  (choix quasi a egalite), plus le decodage est chaotique -- confirme le mecanisme")
        print("  propose (quasi-egalites de score = point de bascule sous rotation).")
    elif pval >= 0.05:
        print("  Pas de correlation significative (p >= 0.05) -- la marge de decision moyenne par")
        print("  chromosome ne predit pas le chaos mesure, meme a cette echelle d'echantillonnage.")
    else:
        print("  Correlation significative mais dans le sens INATTENDU (positive) -- la marge de")
        print("  decision n'explique pas le chaos dans le sens propose par le mecanisme.")
    print(f"{'='*96}\n")


if __name__ == "__main__":
    main()
