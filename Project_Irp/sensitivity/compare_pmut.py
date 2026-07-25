"""Mutation probability comparison: 1/n (literature) vs 2/n (proposed) vs 3/n.

Runs QINSGA-III on the IRP instance with three p_mut values over N_RUNS
independent seeds. Hypervolume is computed with a FIXED global reference point
so values are directly comparable across configurations.

The standard rule from the literature (Deb 2001) is p_mut = 1/n_genes,
meaning one gene mutated per individual on average.
This script tests whether 2/n_genes (two mutations per individual) yields
a better exploration-exploitation trade-off on the IRP.

Metrics:
  - Hypervolume (HV, fixed ref)  — higher is better
  - Pareto front size            — higher is better
  - Runtime (seconds)

Usage:
    python -m sensitivity.compare_pmut --instance 15 --runs 10 --gen 200
    python -m sensitivity.compare_pmut --instance 5  --runs 5  --gen 150
"""

from __future__ import annotations

import argparse
import os
import sys
import time

MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(MODULE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.util.ref_dirs import get_reference_directions

from models.parametres import load_instance
from QINSGA3.algorithm import run_qinsga3

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEEDS = [42, 137, 271, 491, 613, 733, 857, 977, 1009, 1123]
N_PARTITIONS = 8

# p_mut multipliers tested — actual value = multiplier / n_genes
PMUT_MULTIPLIERS = {
    "1/n (littérature)": 1,
    "2/n (proposé)":     2,
    "3/n (au-dessus)":   3,
}


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def _run_one(
    sets_, params_, ref_dirs, p_mut, pop_size, max_gen, seed
) -> tuple[np.ndarray | None, float]:
    t0 = time.time()
    try:
        _, pareto_F, _ = run_qinsga3(
            sets_     = sets_,
            params_   = params_,
            ref_dirs  = ref_dirs,
            pop_size  = pop_size,
            max_gen   = max_gen,
            p_mut     = p_mut,
            seed      = seed,
        )
    except RuntimeError:
        pareto_F = None
    elapsed = time.time() - t0
    return pareto_F, round(elapsed, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _stats(values: list[float]) -> dict:
    arr = np.array([v for v in values if not np.isnan(v)])
    if len(arr) == 0:
        return {"mean": float("nan"), "best": float("nan"),
                "worst": float("nan"), "std": float("nan")}
    return {
        "mean":  float(np.mean(arr)),
        "best":  float(np.max(arr)),
        "worst": float(np.min(arr)),
        "std":   float(np.std(arr)),
    }


def _print_table(results: list[dict], metric: str, label: str) -> None:
    print(f"\n{'─'*74}")
    print(f"  {label}")
    print(f"{'─'*74}")
    print(f"  {'p_mut':<20} {'Moyenne':>14} {'Meilleur':>14} {'Pire':>10} {'Écart-type':>12}")
    print(f"  {'─'*20} {'─'*14} {'─'*14} {'─'*10} {'─'*12}")
    row_means = {}
    for label_key in PMUT_MULTIPLIERS:
        vals  = [r[metric] for r in results if r["label"] == label_key]
        stats = _stats(vals)
        row_means[label_key] = stats["mean"]
        print(
            f"  {label_key:<20} {stats['mean']:>14.2f} {stats['best']:>14.2f} "
            f"{stats['worst']:>10.2f} {stats['std']:>12.2f}"
        )
    best_key = max(row_means, key=lambda k: row_means[k])
    print(f"\n  ★ Meilleur : {best_key}")
    print(f"{'─'*74}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_comparison(
    instance: str = "5",
    n_runs:   int = 5,
    max_gen:  int = 150,
    pop_size: int = 200,
) -> None:
    data_path = os.path.join(PROJECT_DIR, "data", f"instance_{instance}_clients.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Instance non trouvée : {data_path}")

    sets_, params_ = load_instance(data_path)
    ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=N_PARTITIONS)
    effective_pop = max(pop_size, len(ref_dirs))

    n_clients = len(sets_["clients"])
    n_genes   = n_clients * len(sets_["T"]) + n_clients

    # Compute actual p_mut values
    pmut_values = {lbl: mult / n_genes for lbl, mult in PMUT_MULTIPLIERS.items()}

    print("=" * 74)
    print("  COMPARAISON PROBABILITÉ DE MUTATION — QINSGA-III")
    print("=" * 74)
    print(f"  Instance   : {instance} clients  |  Variables (n) : {n_genes}")
    print(f"  pop_size   : {effective_pop}  |  max_gen : {max_gen}  |  runs : {n_runs}")
    print(f"  Seeds      : {SEEDS[:n_runs]}")
    print()
    print(f"  {'Variante':<20} {'p_mut (valeur exacte)':>22}  {'Gènes mutés/individu (moy.)':>28}")
    print(f"  {'─'*20} {'─'*22}  {'─'*28}")
    for lbl, mult in PMUT_MULTIPLIERS.items():
        p = mult / n_genes
        print(f"  {lbl:<20} {p:>22.6f}  {mult:>28.1f}")
    print()
    print(f"  Référence littérature : Deb, K. (2001). Multi-Objective Optimization")
    print(f"  Using Evolutionary Algorithms. Wiley. (p_mut = 1/n)")
    print("=" * 74)

    # ── Phase 1 : collecte des fronts ───────────────────────────────────────
    raw: dict[str, list] = {lbl: [] for lbl in PMUT_MULTIPLIERS}

    for lbl, p_mut in pmut_values.items():
        print(f"\n>>> p_mut = {lbl}  ({p_mut:.6f})")
        for run_idx in range(n_runs):
            seed = SEEDS[run_idx % len(SEEDS)]
            print(f"  Run {run_idx+1}/{n_runs}  seed={seed} ... ", end="", flush=True)
            pareto_F, elapsed = _run_one(
                sets_, params_, ref_dirs, p_mut, effective_pop, max_gen, seed
            )
            raw[lbl].append((pareto_F, elapsed))
            size = len(pareto_F) if pareto_F is not None else 0
            print(f"front={size}  time={elapsed}s")

    # ── Phase 2 : point de référence HV global fixe ─────────────────────────
    all_F = [F for runs in raw.values() for (F, _) in runs if F is not None]
    if not all_F:
        print("\nAucune solution faisable. Arrêt.")
        return

    global_nadir = np.max(np.vstack([F.max(axis=0) for F in all_F]), axis=0) * 1.1 + 1e-6
    hv_indicator = HV(ref_point=global_nadir)
    print(f"\n  Point de référence HV global (nadir×1.1) : {np.round(global_nadir, 2)}")

    # ── Phase 3 : métriques ─────────────────────────────────────────────────
    all_results: list[dict] = []
    for lbl in PMUT_MULTIPLIERS:
        for run_idx, (pareto_F, elapsed) in enumerate(raw[lbl]):
            seed = SEEDS[run_idx % len(SEEDS)]
            hv   = float(hv_indicator(pareto_F)) if pareto_F is not None else float("nan")
            size = len(pareto_F) if pareto_F is not None else 0
            all_results.append({
                "label":      lbl,
                "seed":       seed,
                "hv":         hv,
                "front_size": size,
                "elapsed_s":  elapsed,
            })

    # ── Tableaux ─────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 74)
    print("  RÉSULTATS AGRÉGÉS  (HV avec point de référence FIXE global)")
    print("=" * 74)

    _print_table(all_results, "hv",         "Hypervolume (↑ meilleur)")
    _print_table(all_results, "front_size", "Taille front Pareto (↑ meilleur)")

    # Runtime
    print(f"\n{'─'*74}")
    print("  Temps moyen par run")
    print(f"{'─'*74}")
    for lbl in PMUT_MULTIPLIERS:
        times = [r["elapsed_s"] for r in all_results if r["label"] == lbl]
        print(f"  {lbl:<20}  moy={np.mean(times):.1f}s  "
              f"min={min(times):.1f}s  max={max(times):.1f}s")

    # Détail par run
    print(f"\n{'─'*74}")
    print("  DÉTAIL PAR RUN")
    print(f"{'─'*74}")
    print(f"  {'p_mut':<20} {'Seed':>6} {'HV':>16} {'Front':>8} {'Time':>8}")
    print(f"  {'─'*20} {'─'*6} {'─'*16} {'─'*8} {'─'*8}")
    for r in all_results:
        hv_str = f"{r['hv']:.2f}" if not np.isnan(r["hv"]) else "N/A"
        print(
            f"  {r['label']:<20} {r['seed']:>6} {hv_str:>16} "
            f"{r['front_size']:>8} {r['elapsed_s']:>7.1f}s"
        )

    # Classement final
    print(f"\n{'─'*74}")
    print("  CLASSEMENT FINAL (HV moyen)")
    print(f"{'─'*74}")
    ranking = sorted(
        PMUT_MULTIPLIERS.keys(),
        key=lambda lbl: np.nanmean([r["hv"] for r in all_results if r["label"] == lbl]),
        reverse=True,
    )
    hv_best = np.nanmean([r["hv"] for r in all_results if r["label"] == ranking[0]])
    for pos, lbl in enumerate(ranking, 1):
        hv_mean = np.nanmean([r["hv"] for r in all_results if r["label"] == lbl])
        diff    = (hv_best - hv_mean) / hv_best * 100 if hv_best > 0 else 0
        marker  = "★ MEILLEUR" if pos == 1 else f"  −{diff:.1f}% vs meilleur"
        print(f"  {pos}. {lbl:<20}  HV moy = {hv_mean:.2f}   {marker}")

    # Interprétation automatique
    best_lbl  = ranking[0]
    best_mult = PMUT_MULTIPLIERS[best_lbl]
    print(f"\n{'─'*74}")
    if best_mult == 2:
        print("  → La règle proposée 2/n EST empiriquement supérieure à 1/n et 3/n.")
        print("    Justification pour le rapport : exploration optimale sur l'IRP.")
    elif best_mult == 1:
        print("  → La règle standard 1/n (Deb 2001) est meilleure.")
        print("    Recommandation : utiliser p_mut = 1/n et citer Deb 2001 directement.")
    else:
        print(f"  → La règle {best_lbl} est meilleure — ajuster la valeur par défaut.")
    print(f"{'─'*74}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comparaison probabilité de mutation QINSGA-III"
    )
    parser.add_argument("--instance", default="5",
                        choices=["3", "5", "15", "25", "30", "40", "100"])
    parser.add_argument("--runs",    type=int, default=5)
    parser.add_argument("--gen",     type=int, default=150)
    parser.add_argument("--pop",     type=int, default=200)
    args = parser.parse_args()
    run_comparison(args.instance, args.runs, args.gen, args.pop)
