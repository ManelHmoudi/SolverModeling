"""Cheap diagnostic: would the (reverted, un-adopted) DPQiEA diversity-preserving
operator [Tayarani-N & Akbarzadeh-T 2014, Evolutionary Intelligence 7:219-239,
Section 5] actually TRIGGER on the real IRP, before spending time restoring it
from git history (commits d70b396..1b7e369)?

Context: that operator was ported and validated on DTLZ1/DTLZ3 in July 2026 and
rejected -- not because it hurt anything, but because its own trigger condition
(eq. 11: individuals "converged" to theta near 0/pi/2) almost never fired, so it
had nothing to act on. QINSGA3's continuous crossover/mutation on DTLZ kept
preventing that level of collapse.

On the real IRP, Solvers/IRP_results_summary.md's diagnostic chapter found the
opposite: QI-NSGA-III's pooled Pareto-front chromosome diversity is ~13x lower
than NSGA-III's (Var(X norm.) = 0.0047 vs 0.060, 6 seeds). That is exactly the
symptom this operator targets. This script checks, on the SAME cached data
already used for that diversity measurement (now all 20 available seeds), how
often the operator's two structural preconditions actually hold:

  1. eq. 11 (convergence): for individual i, mean_k |cos(2*theta_ik)| > gamma
  2. eq. 12 (similarity, adapted): for a converged pair i,j in the SAME niche,
     mean_k |theta_ik - theta_jk| / (pi/2) < delta

If most niches never have >=2 simultaneously converged (let alone mutually
similar) individuals -- as happened on DTLZ -- restoring the operator would
likely repeat that dead end. If a large share of niches DO have such clusters,
that is strong evidence the operator's reset mechanism has real work to do here
and restoring it from git history is worth the effort.

Limitation (stated explicitly, matching this project's diagnostic style): this
uses the cached FINAL Pareto/archive front per run, not a raw mid-run working
population snapshot. The final front is already elitist-filtered, so it likely
UNDER-estimates how converged the raw generation-by-generation population is --
a positive trigger rate here is a conservative (lower-bound) signal in favour of
restoring the operator; a near-zero rate is not fully conclusive against it, but
matches the same data source already used for this diagnostic chapter's other
numbers.

Usage: python -m sensitivity.diagnose_diversity_preserving_trigger
"""
import json
import os
import sys

PROJECT_DIR = r"c:\Users\Mariem\OneDrive\Bureau\SolverModeling\Project_Irp"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np
from pymoo.util.ref_dirs import get_reference_directions

from models.parametres         import load_instance
from Solvers.NSGA3.decoder      import decode_chromosome, build_routes
from Solvers.NSGA3.evaluator    import compute_f1, compute_f2, compute_f3, compute_f4
from Solvers.NSGA3.problem      import IRPProblem
from Solvers.QINSGA3.algorithm  import _encode_theta, _normalise_F, _assign_ref_dirs

INSTANCE     = "100"
N_OBJ        = 4
N_PARTITIONS = 8  # matches production QINSGA3/main.py -- 165 ref dirs for M=4
GAMMAS       = [0.99, 0.95, 0.90, 0.70, 0.50]   # 0.99 = paper default; 0.5 = DTLZ's own recalibration attempt
DELTAS       = [0.01, 0.1, 0.3, 1.0]            # paper's own reported problem-sensitive range (Table 4)

_CACHE = os.path.join(PROJECT_DIR, "Solvers", "QINSGA3", "qinsga3_chromosomes.json")


def _decode_to_F(chromosomes: list, sets_: dict, params_: dict) -> np.ndarray:
    F = []
    for chrom in chromosomes:
        quantities, priorities = decode_chromosome(np.array(chrom), sets_)
        rr = build_routes(quantities, sets_, params_, priorities)
        F.append([
            compute_f1(rr, sets_, params_), compute_f2(rr, sets_, params_),
            compute_f3(rr, sets_, params_), compute_f4(rr, sets_, params_),
        ])
    return np.array(F)


def main() -> None:
    data_path = os.path.join(PROJECT_DIR, "data", f"instance_{INSTANCE}_clients.json")
    sets_, params_ = load_instance(data_path)
    problem = IRPProblem(sets_, params_)
    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)

    with open(_CACHE, encoding="utf-8") as f:
        cache = json.load(f)
    seeds = [r["seed"] for r in cache["runs"]]
    X = np.array([c for r in cache["runs"] for c in r["chromosomes"]], dtype=float)
    print(f"QI-NSGA-III cache: {len(cache['runs'])} seeds {seeds}, {len(X)} chromosomes poolees")

    print("Decodage des chromosomes en objectifs...")
    F = _decode_to_F(X.tolist(), sets_, params_)

    theta = _encode_theta(X, xl, xu)
    n_genes = theta.shape[1]

    ref_dirs = get_reference_directions("das-dennis", N_OBJ, n_partitions=N_PARTITIONS)
    F_norm = _normalise_F(F)
    assoc = _assign_ref_dirs(F_norm, ref_dirs)
    n_niches_occupied = len(np.unique(assoc))
    print(f"Niches occupees : {n_niches_occupied} / {len(ref_dirs)} directions de reference")

    # eq. 11 -- per-individual convergence score
    conv_score = np.abs(np.cos(2.0 * theta)).mean(axis=1)
    print(f"\nDistribution du score de convergence (eq. 11, mean|cos(2*theta)|) :")
    print(f"  min={conv_score.min():.4f}  median={np.median(conv_score):.4f}  "
          f"mean={conv_score.mean():.4f}  max={conv_score.max():.4f}")

    print("\n" + "=" * 96)
    print("  DECLENCHEMENT DE L'OPERATEUR DIVERSITY-PRESERVING (structurel, hors stagnation temporelle)")
    print("=" * 96)
    header = f"  {'gamma':>6} | {'delta':>6} | niches avec >=2 convergees | dont >=1 paire similaire (declenchement reel)"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for gamma in GAMMAS:
        converged_mask = conv_score > gamma
        n_niches_with_2plus_converged = 0
        for niche in np.unique(assoc):
            members = np.where((assoc == niche) & converged_mask)[0]
            if len(members) >= 2:
                n_niches_with_2plus_converged += 1

        for delta in DELTAS:
            n_niches_triggering = 0
            for niche in np.unique(assoc):
                members = np.where((assoc == niche) & converged_mask)[0]
                if len(members) < 2:
                    continue
                th = theta[members]
                # pairwise mean |theta_i - theta_j| / (pi/2), eq. 12 adapted
                diff = np.abs(th[:, None, :] - th[None, :, :]).mean(axis=2) / (np.pi / 2.0)
                np.fill_diagonal(diff, np.inf)
                if (diff < delta).any():
                    n_niches_triggering += 1

            print(f"  {gamma:>6.2f} | {delta:>6.2f} | "
                  f"{n_niches_with_2plus_converged:>3d}/{n_niches_occupied} "
                  f"({100*n_niches_with_2plus_converged/max(n_niches_occupied,1):>5.1f}%) | "
                  f"{n_niches_triggering:>3d}/{n_niches_occupied} "
                  f"({100*n_niches_triggering/max(n_niches_occupied,1):>5.1f}%)")

    print("=" * 96)
    print("\nINTERPRETATION")
    print("  'niches avec >=2 convergees' = precondition eq. 11 seule (le goulot qui a")
    print("  bloque l'operateur sur DTLZ -- y avait presque jamais 2+ individus convergees")
    print("  dans la meme niche en meme temps).")
    print("  'declenchement reel' = eq. 11 ET eq. 12 -- ce que l'operateur ferait vraiment.")
    print("  Rappel : mesure sur le front de Pareto final (deja filtre), donc borne")
    print("  inferieure conservatrice du taux reel sur la population de travail brute.")
    print("  Si le taux de declenchement reel est significatif (>> 0%) pour gamma=0.99")
    print("  (valeur du papier), la restauration de l'operateur vaut la peine d'etre testee.")
    print()


if __name__ == "__main__":
    main()
