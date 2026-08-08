"""Diagnostic (not a performance comparison): does an instance's geographic
CLIENT SEPARATION predict how chaotic `_nearest_neighbour`'s decoding is?

Context: `Solvers/NSGA3/decoder.py::_nearest_neighbour` picks the next client
by lowest score = distance(current, candidate) / (0.5 + priority(candidate))
(decoder.py:331). The project's diagnostic chapter already established the
decoder is chaotic (small theta change -> disproportionately different
route, `sensitivity/test_theta_route_sensitivity_weighted.py`) -- this
script tests WHY, mechanistically: if two candidates' scores are close (a
near-tie), a small priority perturbation (from the rotation gate) can flip
which one wins, cascading into a different route (greedy, irrevocable).
Well-separated candidates (large score gap) should be robust to the same
perturbation.

This does NOT construct or tune any instance -- it measures a property
(distance separation) already present in every EXISTING instance
(3/5/15/25/30/40/100 clients, `data/instance_*_clients.json`, unmodified)
and correlates it against directly measured decoder chaos on that same
instance. If separation predicts chaos, that is evidence for the mechanism,
using only data the project already has -- not a constructed favourable
case, and not a QI-NSGA-III vs NSGA-III performance run.

Two measurements per instance:
  1. SEPARATION -- for every client, the ratio (2nd-nearest distance /
     1st-nearest distance) among all other nodes (clients + depot). Mean
     across all clients. Close to 1.0 => frequent near-ties (many equally-
     close alternatives); >> 1.0 => the nearest choice is unambiguous.
     A pure property of the instance's distance matrix, independent of any
     chromosome/decoding.
  2. CHAOS -- mean Jaccard route-distance (reusing `sensitivity/
     test_route_diversity.py`'s `_route_arcset`/`_jaccard_distance`,
     unmodified) between each of N_SAMPLES random chromosomes' decoded
     route and the same chromosome's route after one production-typical
     rotation-gate perturbation (alpha at the schedule's midpoint). Higher
     = more chaotic (perturbation changes the route more).

Usage:
    python -m sensitivity.diagnose_distance_separation_vs_chaos
"""
from __future__ import annotations

import os
import sys

MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(MODULE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np

from models.parametres          import load_instance
from Solvers.NSGA3.decoder       import decode_chromosome, build_routes
from Solvers.NSGA3.problem       import IRPProblem
from sensitivity.test_route_diversity import _route_arcset, _jaccard_distance

INSTANCES  = ["3", "5", "15", "25", "30", "40", "100"]
N_SAMPLES  = 15          # random chromosomes decoded per instance
ALPHA_MID  = 0.0505 * np.pi   # midpoint of production's alpha_min..alpha_max schedule
RNG_SEED   = 20260806


def _separation_score(sets_: dict, params_: dict) -> float:
    """Mean, over all clients, of (2nd-nearest distance / 1st-nearest
    distance) among all other nodes (clients + depot O). Pure instance
    property -- no chromosome/decoding involved."""
    clients = sets_["clients"]
    O       = sets_["O"]
    d       = params_["d"]
    nodes   = list(clients) + [O]

    ratios = []
    for i in clients:
        dists = sorted(d[i, j] for j in nodes if j != i)
        if len(dists) < 2 or dists[0] <= 1e-9:
            continue
        ratios.append(dists[1] / dists[0])
    return float(np.mean(ratios)) if ratios else float("nan")


def _encode_theta(X, xl, xu):
    p = np.clip((X - xl) / np.where(xu - xl > 1e-12, xu - xl, 1.0), 0.0, 1.0)
    return np.clip(np.arccos(np.sqrt(p)), 0.0, np.pi / 2.0)


def _measure_chaos(sets_: dict, params_: dict, rng: np.random.Generator) -> float:
    """Mean Jaccard route-distance between N_SAMPLES random chromosomes'
    decoded routes and the same chromosomes after one rotation-gate-style
    perturbation toward a random guide, at a production-typical alpha."""
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

    dists = []
    for i in range(N_SAMPLES):
        q0, p0 = decode_chromosome(X[i], sets_)
        rr0 = build_routes(q0, sets_, params_, p0)
        q1, p1 = decode_chromosome(X_pert[i], sets_)
        rr1 = build_routes(q1, sets_, params_, p1)
        dists.append(_jaccard_distance(_route_arcset(rr0), _route_arcset(rr1)))
    return float(np.mean(dists))


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    print("=" * 92)
    print("  DIAGNOSTIC -- separation geographique vs chaos du decodeur (instances EXISTANTES, non modifiees)")
    print("=" * 92)
    print(f"  {'Instance':<10} {'N clients':<10} {'Separation (2e/1er plus proche)':<32} {'Chaos (Jaccard, alpha={:.4f})'.format(ALPHA_MID):<30}")
    print("-" * 92)

    rows = []
    for inst in INSTANCES:
        data_path = os.path.join(PROJECT_DIR, "data", f"instance_{inst}_clients.json")
        sets_, params_ = load_instance(data_path)
        sep = _separation_score(sets_, params_)
        chaos = _measure_chaos(sets_, params_, rng)
        rows.append((inst, len(sets_["clients"]), sep, chaos))
        print(f"  {inst:<10} {len(sets_['clients']):<10} {sep:<32.4f} {chaos:<30.4f}")

    seps   = np.array([r[2] for r in rows])
    chaoss = np.array([r[3] for r in rows])
    valid  = ~(np.isnan(seps) | np.isnan(chaoss))
    if valid.sum() >= 3:
        corr = float(np.corrcoef(seps[valid], chaoss[valid])[0, 1])
    else:
        corr = float("nan")

    print("-" * 92)
    print(f"\nCorrelation de Pearson (separation, chaos) sur les {int(valid.sum())} instances valides : r = {corr:.4f}")
    print("\nINTERPRETATION")
    print("  r fortement negatif (proche de -1) : plus les clients sont geographiquement separes,")
    print("  moins le decodeur est chaotique -- confirme le mecanisme (marge de score au choix du")
    print("  plus proche voisin).")
    print("  r proche de 0 : la separation geographique seule n'explique pas le chaos mesure --")
    print("  d'autres facteurs (fenetres de temps, contraintes de capacite) dominent probablement.")
    print(f"{'='*92}\n")


if __name__ == "__main__":
    main()
