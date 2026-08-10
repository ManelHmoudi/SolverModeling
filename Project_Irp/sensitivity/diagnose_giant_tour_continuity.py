"""Smoke test (not a performance comparison): does replacing the greedy
nearest-neighbour decoder with a giant-tour + split decoder restore
continuity between theta and the decoded route?

Context: `Solvers/IRP_results_summary.md`'s diagnostic chapter and eleven
independent remedies (A-J, plus the two `diagnose_*` scripts) all confirm the
same root cause -- `_nearest_neighbour` (`Solvers/NSGA3/decoder.py`) chooses
the next client with `score = dist(current, candidate) / (0.5 + priority)`,
where `current` is wherever the PREVIOUS greedy choice left the route. A
small theta perturbation can flip one early choice, which changes `current`,
which changes every downstream distance, cascading into a disproportionately
different route (measured directly: `test_theta_route_sensitivity_weighted.py`
finds only ~1.12% of routes unchanged on average across a full run's alpha
schedule). Every remedy that kept this decoder and instead tuned the
rotation (A-H), the guide-selection criterion (F, I), or even the decoder's
SELECTION RULE while keeping the same step-by-step cascading construction
(J, lookahead) failed or was neutral. The common thread: theta->x is smooth
(cos^2), but x->solution is not, because the construction is irrevocable and
context-dependent at every step.

This script tests a decoder of a genuinely different KIND: a giant-tour +
split construction (Prins 2004; the same principle BRKGA uses to give
continuous/evolutionary search a smooth handle on routing problems).
Priority becomes a pure SORT KEY (mandatory clients first, by floor>0 --
mirrors production's global mandatory-before-optional invariant -- then by
priority descending, tie-broken by client id for determinism) instead of
feeding a live, context-dependent score. The tour order is fixed ONCE before
any truck is assigned; a single greedy left-to-right pass then splits that
fixed sequence into truck routes by capacity/tau_max feasibility only --
DISTANCE never influences WHICH client is visited next, only whether the
already-fixed next client still fits time-wise. A small priority change can
only shift a client's rank by a few positions in the sort -- it cannot,
structurally, un-do or re-order the rest of the sequence the way a live
"current position" feedback loop can.

Isolation: `Solvers/NSGA3/decoder.py` is NEVER modified on disk. Reuses the
exact monkey-patch technique already verified byte-identical for
`diagnose_decision_margin_vs_chaos.py` -- `_nearest_neighbour` is swapped at
the module level only for the duration of a `build_routes` call, restored in
`finally`. `build_routes`'s surrounding logic (frigo/nonfrigo splitting,
stock ceilings, C6/C8 quantity constraints) is untouched and identical to
production; only the per-truck-group construction step changes.

Known approximation (documented, not hidden): if a client cannot fit even
alone on a freshly started truck (capacity or `tau_max` infeasible from the
depot), production's dynamic re-scan would still consider it against every
future truck; this split decoder marks it permanently unserved this period
once a fresh truck already failed it (skips forward in the fixed order)
rather than re-trying it against every remaining truck. Aggregate quantity
budgets are already capped to total truck capacity upstream in
`build_routes` (`_clamp_to_integer_budget`), so this should be rare -- not
verified not-degenerate before this smoke test, see printed diagnostics
below for how often it triggers.

What this script measures: using REAL cached NSGA-III chromosomes (100-client
instance, same source as `test_theta_route_sensitivity_weighted.py`),
perturb theta at several alpha values spanning the real production schedule,
and compare -- for BOTH decoders -- the mean Jaccard route-distance
(`test_route_diversity.py`'s `_route_arcset`/`_jaccard_distance`, unmodified)
between the unperturbed and perturbed decode. A smooth/continuous map should
show Jaccard distance scaling down toward 0 as alpha shrinks; the current
decoder instead saturates near its ceiling even at the smallest production
alpha. This is a cheap, fast check to run BEFORE investing in a full
2-opt-repair-equivalent campaign for this decoder.

Usage:
    python -m sensitivity.diagnose_giant_tour_continuity
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
from Solvers.NSGA3.problem      import IRPProblem
from Solvers.QINSGA3.algorithm  import _encode_theta
from sensitivity.test_route_diversity import _route_arcset, _jaccard_distance

INSTANCE   = "100"
SEEDS      = [42, 137, 271, 491, 613]
MAX_GEN    = 300
ALPHA_MAX  = 0.10 * np.pi
ALPHA_MIN  = 0.001 * np.pi
GENS_TO_TEST = [0, 50, 100, 150, 200, 250, 300]   # sparse smoke test, not the full 31-point sweep
RNG_SEED   = 12345

_NSGA3_CACHE = os.path.join(PROJECT_DIR, "Solvers", "NSGA3", "nsga3_chromosomes.json")

_skip_count = [0]   # mutable counter for the "permanently unserved this period" approximation


# ---------------------------------------------------------------------------
# Giant-tour + split decoder (local fork -- decoder.py never modified)
# ---------------------------------------------------------------------------

def _giant_tour_order(qty_dict: dict, floors: dict, priorities: dict) -> list:
    floors = floors or {}
    mandatory = sorted(
        (l for l in qty_dict if floors.get(l, 0) > 0),
        key=lambda l: (-priorities.get(l, 0.5), l),
    )
    rest = sorted(
        (l for l in qty_dict if floors.get(l, 0) <= 0),
        key=lambda l: (-priorities.get(l, 0.5), l),
    )
    return mandatory + rest


def _giant_tour_split(qty_dict, trucks, t, d, v, s, Q, O, tau_max=None, floors=None,
                       priorities=None):
    """Same call signature as decoder.py::_nearest_neighbour, so it can be
    monkey-patched in transparently. Fixed visit order from `_giant_tour_order`
    (priority-only, no distance feedback), then a single left-to-right greedy
    split into truck routes by capacity/tau_max feasibility."""
    prios  = priorities or {}
    floors = floors or {}
    order  = _giant_tour_order(qty_dict, floors, prios)

    x_vars, f_vars, arrivals, assign, routes = {}, {}, {}, {}, {}
    idx, n, truck_idx = 0, len(order), 0

    while idx < n and truck_idx < len(trucks):
        k     = trucks[truck_idx]
        cap   = Q[k]
        speed = v[k]

        path, qty_on_route, load, current_time, current = [O], {}, 0, 0.0, O

        while idx < n:
            l       = order[idx]
            q       = qty_dict[l]
            floor_l = floors.get(l, 0)
            is_mandatory = floor_l > 0

            if load + q <= cap:
                q_effective = q
            elif is_mandatory and load + floor_l <= cap:
                q_effective = floor_l
            else:
                if load == 0:
                    _skip_count[0] += 1
                    idx += 1
                    continue
                break

            dist = d[current, l]
            if tau_max is not None and not is_mandatory:
                projected = (current_time + s.get(current, 0.0) + dist / speed
                             + s.get(l, 0.0) + d[l, O] / speed)
                if projected > tau_max:
                    if load == 0:
                        _skip_count[0] += 1
                        idx += 1
                        continue
                    break

            current_time      += s.get(current, 0.0) + dist / speed
            path.append(l)
            qty_on_route[l]    = q_effective
            load              += q_effective
            arrivals[l, t]     = current_time
            assign[l, t]       = k
            current            = l
            idx += 1

        path.append(O)

        if len(path) > 2:
            m   = len(path)
            suf = [0] * (m + 1)
            for pos in range(m - 2, -1, -1):
                node    = path[pos + 1]
                suf[pos] = suf[pos + 1] + (qty_on_route.get(node, 0) if node != O else 0)

            for pos in range(m - 1):
                i, j               = path[pos], path[pos + 1]
                x_vars[i, j, t, k] = 1
                f_vars[i, j, t, k] = suf[pos]

            routes[k] = {
                "path": path,
                "qty":  {str(l): q for l, q in qty_on_route.items()},
            }

        truck_idx += 1

    return routes, x_vars, f_vars, arrivals, assign


def _build_routes_giant_tour(quantities, sets_, params_, priorities):
    original = decoder_mod._nearest_neighbour
    decoder_mod._nearest_neighbour = _giant_tour_split
    try:
        return decoder_mod.build_routes(quantities, sets_, params_, priorities)
    finally:
        decoder_mod._nearest_neighbour = original


# ---------------------------------------------------------------------------
# Continuity smoke test
# ---------------------------------------------------------------------------

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
    print(f"Baseline : {N} solutions reelles (NSGA-III cache, {len(SEEDS)} seeds)")

    theta = _encode_theta(baseline_X, xl, xu)
    rng = np.random.default_rng(RNG_SEED)

    print("\nDecodage des baselines (deux decodeurs, routes de reference)...")
    base_arcs_original, base_arcs_giant_tour = [], []
    for i in range(N):
        q, p = decode_chromosome(baseline_X[i], sets_)
        rr_o = decoder_mod.build_routes(q, sets_, params_, p)
        rr_g = _build_routes_giant_tour(q, sets_, params_, p)
        base_arcs_original.append(_route_arcset(rr_o))
        base_arcs_giant_tour.append(_route_arcset(rr_g))
    print(f"  Clients non-servis (approximation split, cumul sur {N} decodages) : {_skip_count[0]}")

    print("\n" + "=" * 100)
    print("  CONTINUITE theta -> route : decodeur original (NN glouton) vs giant-tour+split")
    print("=" * 100)
    print(f"  {'gen':<6} {'alpha':<10} {'Jaccard original (NN)':<26} {'Jaccard giant-tour+split':<26}")
    print("-" * 100)

    results = []
    for gen in GENS_TO_TEST:
        alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * (1.0 - gen / MAX_GEN)

        guide_idx = rng.integers(0, N, size=N)
        same = guide_idx == np.arange(N)
        guide_idx[same] = (guide_idx[same] + 1) % N
        theta_guide = theta[guide_idx]

        diff = theta_guide - theta
        theta_pert = np.clip(theta + alpha * np.tanh(diff / (np.pi / 8.0)), 0.0, np.pi / 2.0)
        p_pert = np.cos(theta_pert) ** 2
        X_pert = np.clip(xl + p_pert * (xu - xl), xl, xu)

        d_original, d_giant_tour = [], []
        for i in range(N):
            q, p = decode_chromosome(X_pert[i], sets_)
            rr_o = decoder_mod.build_routes(q, sets_, params_, p)
            rr_g = _build_routes_giant_tour(q, sets_, params_, p)
            d_original.append(_jaccard_distance(base_arcs_original[i],   _route_arcset(rr_o)))
            d_giant_tour.append(_jaccard_distance(base_arcs_giant_tour[i], _route_arcset(rr_g)))

        m_o, m_g = float(np.mean(d_original)), float(np.mean(d_giant_tour))
        results.append((gen, alpha, m_o, m_g))
        print(f"  {gen:<6} {alpha:<10.5f} {m_o:<26.4f} {m_g:<26.4f}")

    print("-" * 100)
    alphas    = np.array([r[1] for r in results])
    m_orig    = np.array([r[2] for r in results])
    m_giant   = np.array([r[3] for r in results])

    r_orig,  _ = (float(np.corrcoef(alphas, m_orig)[0, 1]),  None) if len(alphas) >= 3 else (float("nan"), None)
    r_giant, _ = (float(np.corrcoef(alphas, m_giant)[0, 1]), None) if len(alphas) >= 3 else (float("nan"), None)

    i_min, i_max = int(np.argmin(alphas)), int(np.argmax(alphas))
    print(f"\n  Jaccard au plus petit alpha teste ({alphas[i_min]:.5f}) : "
          f"original={m_orig[i_min]:.4f}  giant-tour+split={m_giant[i_min]:.4f}")
    print(f"  Jaccard au plus grand alpha teste  ({alphas[i_max]:.5f}) : "
          f"original={m_orig[i_max]:.4f}  giant-tour+split={m_giant[i_max]:.4f}")
    print(f"  Correlation (alpha, Jaccard) -- original          : r={r_orig:.4f}")
    print(f"  Correlation (alpha, Jaccard) -- giant-tour+split  : r={r_giant:.4f}")

    print("\nINTERPRETATION")
    print("  Continuite restauree si : (1) Jaccard giant-tour+split au plus petit alpha est")
    print("  nettement plus proche de 0 que le decodeur original au meme alpha, ET (2) la")
    print("  correlation (alpha, Jaccard) est nettement positive pour giant-tour+split (le Jaccard")
    print("  grandit avec alpha, un vrai gradient) alors que le decodeur original sature deja au")
    print("  plus petit alpha teste (peu ou pas de correlation, car deja proche de son plafond).")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
