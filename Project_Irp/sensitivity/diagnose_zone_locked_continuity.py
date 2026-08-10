"""Smoke test (not a performance comparison): does locking clients to fixed
geographic zones (`sensitivity/zone_locked.py`) restore theta -> route
continuity where greedy-fill giant-tour and Prins DP-split both failed?

Context: `docs/superpowers/specs/2026-08-10-qinsga3-zone-locked-decoder-
design.md`. Same real cached NSGA-III chromosomes, same sparse alpha
sweep, and same Jaccard route-distance metric as
`diagnose_giant_tour_continuity.py`/`diagnose_prins_split_continuity.py`,
extended to a four-way comparison (original greedy / giant-tour
greedy-fill / Prins DP-split / zone-locked).

Isolation: `Solvers/NSGA3/decoder.py` is NEVER modified on disk -- each
alternative decoder is monkey-patched onto `_nearest_neighbour` only for
the duration of its own `build_routes` call, restored in `finally`.

Usage:
    python -m sensitivity.diagnose_zone_locked_continuity
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
from sensitivity.test_route_diversity           import _route_arcset, _jaccard_distance
from sensitivity.diagnose_giant_tour_continuity import _build_routes_giant_tour
from sensitivity.prins_split                    import _prins_split
from sensitivity.zone_locked                    import _zone_assignment, _make_zone_locked_decoder

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


def _load_coordinates(data_path):
    with open(data_path, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): tuple(v) for k, v in raw["sets"]["coordinates"].items()}


def _zone_locked_builder(sets_, params_, coords):
    O                = sets_["O"]
    clients          = sets_["clients"]
    T                = sets_["T"]
    Q                = params_["Q"]
    frigo_trucks     = params_["frigo_trucks"]
    frigo_list       = sorted(frigo_trucks)
    non_frigo_trucks = [k for k in sets_["M"] if k not in frigo_trucks]
    q_lt             = params_["q_lt"]

    total_demand = {l: sum(q_lt.get((l, t), 0) for t in T) for l in clients}

    zone_by_trucks = {}
    for trucks in (frigo_list, non_frigo_trucks):
        if trucks:
            zone_by_trucks[tuple(sorted(trucks))] = _zone_assignment(
                coords, O, clients, trucks, Q, total_demand)

    zone_locked_decoder = _make_zone_locked_decoder(zone_by_trucks)

    def _build_routes_zone_locked(quantities, sets_inner, params_inner, priorities):
        original = decoder_mod._nearest_neighbour
        decoder_mod._nearest_neighbour = zone_locked_decoder
        try:
            return decoder_mod.build_routes(quantities, sets_inner, params_inner, priorities)
        finally:
            decoder_mod._nearest_neighbour = original

    return _build_routes_zone_locked


def main() -> None:
    data_path = os.path.join(PROJECT_DIR, "data", f"instance_{INSTANCE}_clients.json")
    sets_, params_ = load_instance(data_path)
    coords = _load_coordinates(data_path)
    build_routes_zone_locked = _zone_locked_builder(sets_, params_, coords)

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

    print("\nDecodage des baselines (quatre decodeurs, routes de reference)...")
    base_o, base_g, base_p, base_z = [], [], [], []
    for i in range(N):
        q, p = decode_chromosome(baseline_X[i], sets_)
        base_o.append(_route_arcset(decoder_mod.build_routes(q, sets_, params_, p)))
        base_g.append(_route_arcset(_build_routes_giant_tour(q, sets_, params_, p)))
        base_p.append(_route_arcset(_build_routes_prins_split(q, sets_, params_, p)))
        base_z.append(_route_arcset(build_routes_zone_locked(q, sets_, params_, p)))

    print("\n" + "=" * 140)
    print("  CONTINUITE theta -> route : original vs giant-tour vs Prins-split vs zone-locked")
    print("=" * 140)
    print(f"  {'gen':<6} {'alpha':<10} {'original':<14} {'giant-tour':<14} {'prins-split':<14} {'zone-locked':<14}")
    print("-" * 140)

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

        d_o, d_g, d_p, d_z = [], [], [], []
        for i in range(N):
            q, p = decode_chromosome(X_pert[i], sets_)
            d_o.append(_jaccard_distance(base_o[i], _route_arcset(decoder_mod.build_routes(q, sets_, params_, p))))
            d_g.append(_jaccard_distance(base_g[i], _route_arcset(_build_routes_giant_tour(q, sets_, params_, p))))
            d_p.append(_jaccard_distance(base_p[i], _route_arcset(_build_routes_prins_split(q, sets_, params_, p))))
            d_z.append(_jaccard_distance(base_z[i], _route_arcset(build_routes_zone_locked(q, sets_, params_, p))))

        m_o, m_g, m_p, m_z = (float(np.mean(x)) for x in (d_o, d_g, d_p, d_z))
        results.append((gen, alpha, m_o, m_g, m_p, m_z))
        print(f"  {gen:<6} {alpha:<10.5f} {m_o:<14.4f} {m_g:<14.4f} {m_p:<14.4f} {m_z:<14.4f}")

    print("-" * 140)
    alphas = np.array([r[1] for r in results])
    cols   = {"original": np.array([r[2] for r in results]),
              "giant-tour": np.array([r[3] for r in results]),
              "prins-split": np.array([r[4] for r in results]),
              "zone-locked": np.array([r[5] for r in results])}

    def _corr(y):
        return float(np.corrcoef(alphas, y)[0, 1]) if len(alphas) >= 3 else float("nan")

    i_min, i_max = int(np.argmin(alphas)), int(np.argmax(alphas))
    print(f"\n  Jaccard au plus petit alpha teste ({alphas[i_min]:.5f}):")
    for name, y in cols.items():
        print(f"    {name:<14}: {y[i_min]:.4f}")
    print(f"  Jaccard au plus grand alpha teste  ({alphas[i_max]:.5f}):")
    for name, y in cols.items():
        print(f"    {name:<14}: {y[i_max]:.4f}")
    print("  Correlation (alpha, Jaccard):")
    for name, y in cols.items():
        print(f"    {name:<14}: r={_corr(y):.4f}")

    print("\nINTERPRETATION")
    print("  Continuite restauree si : (1) Jaccard zone-locked au plus petit alpha est")
    print("  nettement plus proche de 0 que les trois autres au meme alpha, ET (2) la")
    print("  correlation (alpha, Jaccard) est nettement positive pour zone-locked (le Jaccard")
    print("  grandit avec alpha, un vrai gradient) alors que les autres saturent deja au")
    print("  plus petit alpha teste.")
    print(f"{'='*140}\n")


if __name__ == "__main__":
    main()
