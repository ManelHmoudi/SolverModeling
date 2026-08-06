"""H3 properly tested: does decoding lose MORE relative diversity than what
the chromosome-level deficit alone would predict?

Context: Solvers/IRP_results_summary.md's diagnostic chapter measured
chromosome diversity (Var(X_norm), pooled Pareto fronts, 6 seeds) at a ratio
QI/NSGA-III = 0.078 (QI-NSGA-III 13x lower). H3 ("perte d'information dans la
chaine theta -> x -> decodeur -> solution") was previously tested only with a
WEAK proxy: ratio of EXACTLY-unique decoded route structures, which was 100%
for both algorithms -- meaning no evidence of collisions, but also no real
measurement of HOW diverse the routes actually are (two routes can be almost
identical -- one swapped stop -- and still count as "unique").

This script replaces that weak proxy with a real diversity metric on the
DECODED ROUTES: mean pairwise Jaccard distance between the sets of
(period, vehicle, directed arc) triples each solution's routing uses. This is
the route-space analogue of the chromosome Var(X_norm) measurement -- same
pooled data (NSGA-III + QI-NSGA-III caches, 100-client instance, all cached
seeds), same question, different space.

If route-diversity ratio (QI/NSGA-III) is close to the chromosome-diversity
ratio (0.078): decoding is roughly diversity-PRESERVING -- whatever diversity
deficit exists is entirely explained by the upstream rotation gate (already
established root cause), no independent evidence for H3.
If the route-diversity ratio is MUCH LOWER than 0.078: decoding actively
compounds the loss -- real evidence for H3, and a new, actionable lead
(the decoder itself would deserve attention, beyond the already-tried
soft-decoder/2-opt remedies).
If HIGHER: decoding acts as a partial diversity amplifier (small chromosome
differences can cascade into disproportionately different routes -- consistent
with the "decoder is chaotic" finding from the theta-route sensitivity test).

Usage: python -m sensitivity.test_route_diversity
"""
import json
import os
import sys

PROJECT_DIR = r"c:\Users\Mariem\OneDrive\Bureau\SolverModeling\Project_Irp"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np

from models.parametres     import load_instance
from Solvers.NSGA3.decoder  import decode_chromosome, build_routes

INSTANCE   = "100"
N_PAIRS    = 3000   # sampled pairs per algorithm (all-pairs would be ~500k for QI-NSGA-III)
RNG_SEED   = 2026

_CACHE_PATHS = {
    "NSGA-III":    os.path.join(PROJECT_DIR, "Solvers", "NSGA3", "nsga3_chromosomes.json"),
    "QI-NSGA-III": os.path.join(PROJECT_DIR, "Solvers", "QINSGA3", "qinsga3_chromosomes.json"),
}


def _route_arcset(route_result) -> frozenset:
    """(period, vehicle, directed arc) triples -- the routing decisions a
    solution actually makes, independent of delivered quantities (those are
    a separate part of the chromosome, already captured by the Var(X_norm)
    chromosome-diversity measurement -- this isolates the ROUTING structure)."""
    arcs = set()
    for t in route_result["routes_data"]:
        for k, info in route_result["routes_data"][t].items():
            path = info["path"]
            for i in range(len(path) - 1):
                arcs.add((t, k, path[i], path[i + 1]))
    return frozenset(arcs)


def _decode_arcsets(chromosomes: list, sets_: dict, params_: dict) -> list:
    out = []
    for chrom in chromosomes:
        quantities, priorities = decode_chromosome(np.array(chrom), sets_)
        rr = build_routes(quantities, sets_, params_, priorities)
        out.append(_route_arcset(rr))
    return out


def _jaccard_distance(a: frozenset, b: frozenset) -> float:
    union = len(a | b)
    if union == 0:
        return 0.0
    return 1.0 - len(a & b) / union


def _mean_pairwise_jaccard(arcsets: list, rng: np.random.Generator, n_pairs: int) -> tuple[float, float]:
    n = len(arcsets)
    dists = []
    for _ in range(n_pairs):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        dists.append(_jaccard_distance(arcsets[i], arcsets[j]))
    dists = np.array(dists)
    return float(dists.mean()), float(dists.std())


def main() -> None:
    data_path = os.path.join(PROJECT_DIR, "data", f"instance_{INSTANCE}_clients.json")
    sets_, params_ = load_instance(data_path)

    rng = np.random.default_rng(RNG_SEED)

    print("=" * 92)
    print("  H3 -- DIVERSITE DES TOURNEES DECODEES (distance de Jaccard sur les arcs)")
    print("=" * 92)

    results = {}
    for algo, path in _CACHE_PATHS.items():
        with open(path, encoding="utf-8") as f:
            cache = json.load(f)
        seeds = [r["seed"] for r in cache["runs"]]
        chromosomes = [c for r in cache["runs"] for c in r["chromosomes"]]
        print(f"\n{algo}: {len(cache['runs'])} seeds {seeds}, {len(chromosomes)} chromosomes poolees")
        print("  Decodage en jeux d'arcs (period, vehicule, arc)...")
        arcsets = _decode_arcsets(chromosomes, sets_, params_)
        mean_arcs = np.mean([len(a) for a in arcsets])
        print(f"  Taille moyenne d'un jeu d'arcs par solution : {mean_arcs:.1f}")

        mean_d, std_d = _mean_pairwise_jaccard(arcsets, rng, N_PAIRS)
        results[algo] = mean_d
        print(f"  Distance de Jaccard moyenne (diversite tournees, {N_PAIRS} paires echantillonnees) : "
              f"{mean_d:.6f} (std={std_d:.6f})")

    ratio_routes = results["QI-NSGA-III"] / results["NSGA-III"] if results["NSGA-III"] > 0 else float("nan")
    ratio_chromosomes = 0.078  # measured previously: Var(X_norm) QI/NSGA-III, see IRP_results_summary.md

    print(f"\n{'='*92}")
    print("  COMPARAISON AU RATIO DE DIVERSITE CHROMOSOME (deja mesure : 0.078, QI 13x plus bas)")
    print(f"{'='*92}")
    print(f"  Ratio diversite TOURNEES  (QI-NSGA-III / NSGA-III) : {ratio_routes:.4f}")
    print(f"  Ratio diversite CHROMOSOME (deja mesure)            : {ratio_chromosomes:.4f}")
    print(f"{'='*92}")
    print("\nINTERPRETATION")
    if ratio_routes > 1.3 * ratio_chromosomes:
        print("  Ratio tournees NETTEMENT SUPERIEUR au ratio chromosome : le decodeur AMPLIFIE")
        print("  la diversite relative (coherent avec le decodeur chaotique deja identifie) --")
        print("  pas de preuve de perte d'information supplementaire (H3 non soutenue).")
    elif ratio_routes < 0.7 * ratio_chromosomes:
        print("  Ratio tournees NETTEMENT INFERIEUR au ratio chromosome : le decodeur PERD de la")
        print("  diversite en plus du deficit deja present en amont -- preuve directe de H3,")
        print("  piste actionnable independante du mecanisme de rotation.")
    else:
        print("  Ratio tournees proche du ratio chromosome : le decodeur est globalement")
        print("  diversite-preservant -- le deficit observe reste explique par la rotation guidee")
        print("  en amont (H1), pas par une perte d'information additionnelle au decodage (H3 non")
        print("  soutenue).")
    print(f"{'='*92}\n")


if __name__ == "__main__":
    main()
