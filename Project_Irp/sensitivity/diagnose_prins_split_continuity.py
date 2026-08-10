"""Smoke test (not a performance comparison): does the Prins DP-optimal
split (`sensitivity/prins_split.py`) restore theta -> route continuity
where the greedy-fill giant-tour attempt
(`diagnose_giant_tour_continuity.py`) failed?

Context: `docs/superpowers/specs/2026-08-10-qinsga3-prins-split-decoder-
design.md`. Same real cached NSGA-III chromosomes, same sparse alpha sweep,
and same Jaccard route-distance metric as `diagnose_giant_tour_continuity.
py`, extended to a three-way comparison (original greedy / giant-tour
greedy-fill / Prins DP-split).

Isolation: `Solvers/NSGA3/decoder.py` is NEVER modified on disk -- each
alternative decoder is monkey-patched onto `_nearest_neighbour` only for
the duration of its own `build_routes` call, restored in `finally`.

Usage:
    python -m sensitivity.diagnose_prins_split_continuity
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
from sensitivity.test_route_diversity          import _route_arcset, _jaccard_distance
from sensitivity.diagnose_giant_tour_continuity import _build_routes_giant_tour
from sensitivity.prins_split                    import _prins_split

INSTANCE   = "100"
SEEDS      = [42, 137, 271, 491, 613]
MAX_GEN    = 300
ALPHA_MAX  = 0.10 * np.pi
ALPHA_MIN  = 0.001 * np.pi
GENS_TO_TEST = [0, 50, 100, 150, 200, 250, 300]
RNG_SEED   = 12345

_NSGA3_CACHE = os.path.join(PROJECT_DIR, "Solvers", "NSGA3", "nsga3_chromosomes.json")


def _build_routes_prins_split(quantities, sets_, params_, priorities):
    original = decoder_mod._nearest_neighbour
    decoder_mod._nearest_neighbour = _prins_split
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
    print(f"Baseline : {N} solutions reelles (NSGA-III cache, {len(SEEDS)} seeds)")

    theta = _encode_theta(baseline_X, xl, xu)
    rng = np.random.default_rng(RNG_SEED)

    print("\nDecodage des baselines (trois decodeurs, routes de reference)...")
    base_arcs_original, base_arcs_giant_tour, base_arcs_prins = [], [], []
    for i in range(N):
        q, p = decode_chromosome(baseline_X[i], sets_)
        rr_o = decoder_mod.build_routes(q, sets_, params_, p)
        rr_g = _build_routes_giant_tour(q, sets_, params_, p)
        rr_p = _build_routes_prins_split(q, sets_, params_, p)
        base_arcs_original.append(_route_arcset(rr_o))
        base_arcs_giant_tour.append(_route_arcset(rr_g))
        base_arcs_prins.append(_route_arcset(rr_p))

    print("\n" + "=" * 120)
    print("  CONTINUITE theta -> route : original (NN glouton) vs giant-tour+greedy-fill vs Prins DP-split")
    print("=" * 120)
    print(f"  {'gen':<6} {'alpha':<10} {'Jaccard original':<20} {'Jaccard giant-tour':<20} {'Jaccard Prins-split':<20}")
    print("-" * 120)

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

        d_original, d_giant_tour, d_prins = [], [], []
        for i in range(N):
            q, p = decode_chromosome(X_pert[i], sets_)
            rr_o = decoder_mod.build_routes(q, sets_, params_, p)
            rr_g = _build_routes_giant_tour(q, sets_, params_, p)
            rr_p = _build_routes_prins_split(q, sets_, params_, p)
            d_original.append(_jaccard_distance(base_arcs_original[i],   _route_arcset(rr_o)))
            d_giant_tour.append(_jaccard_distance(base_arcs_giant_tour[i], _route_arcset(rr_g)))
            d_prins.append(_jaccard_distance(base_arcs_prins[i],          _route_arcset(rr_p)))

        m_o, m_g, m_p = float(np.mean(d_original)), float(np.mean(d_giant_tour)), float(np.mean(d_prins))
        results.append((gen, alpha, m_o, m_g, m_p))
        print(f"  {gen:<6} {alpha:<10.5f} {m_o:<20.4f} {m_g:<20.4f} {m_p:<20.4f}")

    print("-" * 120)
    alphas  = np.array([r[1] for r in results])
    m_orig  = np.array([r[2] for r in results])
    m_giant = np.array([r[3] for r in results])
    m_prins = np.array([r[4] for r in results])

    def _corr(y):
        return float(np.corrcoef(alphas, y)[0, 1]) if len(alphas) >= 3 else float("nan")

    i_min, i_max = int(np.argmin(alphas)), int(np.argmax(alphas))
    print(f"\n  Jaccard au plus petit alpha teste ({alphas[i_min]:.5f}) : "
          f"original={m_orig[i_min]:.4f}  giant-tour={m_giant[i_min]:.4f}  prins-split={m_prins[i_min]:.4f}")
    print(f"  Jaccard au plus grand alpha teste  ({alphas[i_max]:.5f}) : "
          f"original={m_orig[i_max]:.4f}  giant-tour={m_giant[i_max]:.4f}  prins-split={m_prins[i_max]:.4f}")
    print(f"  Correlation (alpha, Jaccard) -- original     : r={_corr(m_orig):.4f}")
    print(f"  Correlation (alpha, Jaccard) -- giant-tour   : r={_corr(m_giant):.4f}")
    print(f"  Correlation (alpha, Jaccard) -- prins-split  : r={_corr(m_prins):.4f}")

    print("\nINTERPRETATION")
    print("  Continuite restauree si : (1) Jaccard prins-split au plus petit alpha est")
    print("  nettement plus proche de 0 que original/giant-tour au meme alpha, ET (2) la")
    print("  correlation (alpha, Jaccard) est nettement positive pour prins-split (le Jaccard")
    print("  grandit avec alpha, un vrai gradient) alors que original/giant-tour saturent deja au")
    print("  plus petit alpha teste.")
    print(f"{'='*120}\n")


if __name__ == "__main__":
    main()
