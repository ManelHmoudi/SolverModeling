"""p_mut_strong comparison: 0.15 vs 0.20 vs 0.30 (current default).

Context (see conversation): on the 100-client instance, 10 seeds, current
defaults (p_mut_strong=0.3), Run 3 (seed=271) was a clear outlier — GD=0.439
and HV=0.546, far worse than the other 9 runs (GD mean ~0.21). [HV=0.546 predates
the 2026-07 fix normalising HV by ref_point.prod() (~1.1**n_obj); on the current
[0,1] HV scale this run would read ~0.546/1.4641 ≈ 0.373 — re-run to get an
exact figure before citing it.] Hypothesis:
p_mut_strong=0.3 (30% of mutated genes get a FULL reset to Uniform(0, pi/2))
occasionally destroys converged structure late in the run, hurting GD
(convergence) on some seeds without benefiting Spacing enough to compensate.

This script tests p_mut_strong in {0.15, 0.20, 0.30} with everything else
held at production defaults (pop=200, gen=300, p_mut=2/n_genes,
mut_sigma=0.05*pi), on the SAME 10 seeds the user already ran, on the
100-client instance. Quality indicators (HV, GD, IGD, Spacing) are computed
with NSGA3.metrics.compute_pareto_metrics using a GLOBAL ideal/nadir shared
across all 30 runs, so values are directly comparable — same methodology
report_builder.py uses for the dashboard's "Runs comparison" table.

Usage:
    python -m validation.compare_pstrong
    python -m validation.compare_pstrong --seeds 42 137 271 --gen 100
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
from pymoo.util.ref_dirs import get_reference_directions

from models.parametres   import load_instance
from QINSGA3.algorithm   import run_qinsga3
from NSGA3.metrics       import compute_pareto_metrics

N_PARTITIONS = 8
N_OBJ        = 4

DEFAULT_SEEDS = [42, 137, 271, 491, 613, 733, 857, 977, 1009, 1123]

PSTRONG_VALUES = {
    "0.15 (reduit)":    0.15,
    "0.20 (reduit)":    0.20,
    "0.30 (actuel)":    0.30,
}


def _run_one(sets_, params_, ref_dirs, p_strong, pop_size, max_gen, seed):
    t0 = time.time()
    _, pareto_F, _ = run_qinsga3(
        sets_        = sets_,
        params_      = params_,
        ref_dirs     = ref_dirs,
        pop_size     = pop_size,
        max_gen      = max_gen,
        p_mut_strong = p_strong,
        seed         = seed,
    )
    elapsed = time.time() - t0
    return pareto_F, round(elapsed, 1)


def _stats(vals):
    arr = np.array([v for v in vals if v is not None], dtype=float)
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan"), "cv": float("nan")}
    mean = float(arr.mean())
    std  = float(arr.std())
    return {
        "mean": mean, "std": std,
        "min":  float(arr.min()), "max": float(arr.max()),
        "cv":   (std / mean * 100.0) if abs(mean) > 1e-9 else float("nan"),
    }


def _print_metric_table(results, key, label, higher_is_better):
    arrow = "^" if higher_is_better else "v"
    print(f"\n{'-'*92}")
    print(f"  {label} ({arrow})")
    print(f"{'-'*92}")
    print(f"  {'p_mut_strong':<18} {'Moyenne':>10} {'Std':>10} {'CV%':>8} "
          f"{'Min':>10} {'Max':>10}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*10}")
    for lbl in PSTRONG_VALUES:
        vals  = [r[key] for r in results if r["label"] == lbl]
        stats = _stats(vals)
        print(f"  {lbl:<18} {stats['mean']:>10.4f} {stats['std']:>10.4f} "
              f"{stats['cv']:>7.1f}% {stats['min']:>10.4f} {stats['max']:>10.4f}")


def run_comparison(instance: str, seeds: list[int], max_gen: int, pop_size: int) -> None:
    data_path = os.path.join(PROJECT_DIR, "data", f"instance_{instance}_clients.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Instance non trouvee : {data_path}")

    sets_, params_ = load_instance(data_path)
    ref_dirs = get_reference_directions("das-dennis", N_OBJ, n_partitions=N_PARTITIONS)
    effective_pop = max(pop_size, len(ref_dirs))

    n_clients = len(sets_["clients"])
    n_genes   = n_clients * len(sets_["T"]) + n_clients

    print("=" * 92)
    print("  COMPARAISON p_mut_strong — QINSGA-III")
    print("=" * 92)
    print(f"  Instance : {instance} clients | Variables : {n_genes} | "
          f"pop={effective_pop} | gen={max_gen}")
    print(f"  Seeds    : {seeds}")
    print(f"  Fixe     : p_mut=2/n_genes | mut_sigma=0.05*pi | p_cross=0.9")
    print("=" * 92)

    # Phase 1 : collecte des fronts
    raw: dict[str, list] = {lbl: [] for lbl in PSTRONG_VALUES}
    for lbl, p_strong in PSTRONG_VALUES.items():
        print(f"\n>>> p_mut_strong = {lbl}")
        for seed in seeds:
            print(f"  seed={seed} ... ", end="", flush=True)
            pareto_F, elapsed = _run_one(
                sets_, params_, ref_dirs, p_strong, effective_pop, max_gen, seed
            )
            raw[lbl].append((seed, pareto_F, elapsed))
            size = len(pareto_F) if pareto_F is not None else 0
            print(f"front={size}  time={elapsed}s", flush=True)

    # Phase 2 : ideal/nadir global partage entre toutes les configs x seeds
    all_F = [F for runs in raw.values() for (_, F, _) in runs if F is not None and len(F) > 0]
    if not all_F:
        print("\nAucune solution faisable. Arret.")
        return
    all_F_stack  = np.vstack(all_F)
    global_ideal = all_F_stack.min(axis=0)
    global_nadir = all_F_stack.max(axis=0)

    # Phase 3 : indicateurs de qualite (HV, GD, IGD, Spacing) via NSGA3.metrics
    all_results: list[dict] = []
    for lbl in PSTRONG_VALUES:
        for seed, pareto_F, elapsed in raw[lbl]:
            if pareto_F is None or len(pareto_F) == 0:
                continue
            q = compute_pareto_metrics(pareto_F, global_ideal, global_nadir)
            all_results.append({
                "label":      lbl,
                "seed":       seed,
                "front_size": len(pareto_F),
                "HV":         q["HV"],
                "GD":         q["GD"],
                "IGD":        q["IGD"],
                "Spacing":    q["Spacing"],
                "elapsed_s":  elapsed,
            })

    print("\n\n" + "=" * 92)
    print("  RESULTATS AGREGES (ideal/nadir global partage entre les 3 configs)")
    print("=" * 92)
    _print_metric_table(all_results, "front_size", "Taille front Pareto", True)
    _print_metric_table(all_results, "HV",         "Hypervolume",         True)
    _print_metric_table(all_results, "GD",         "GD (convergence)",    False)
    _print_metric_table(all_results, "IGD",        "IGD (couverture)",    False)
    _print_metric_table(all_results, "Spacing",    "Spacing (uniformite)", False)

    print(f"\n{'-'*92}")
    print("  DETAIL PAR RUN")
    print(f"{'-'*92}")
    print(f"  {'p_mut_strong':<18} {'Seed':>6} {'Front':>6} {'HV':>8} {'GD':>8} "
          f"{'IGD':>8} {'Spacing':>9} {'Time':>8}")
    for r in all_results:
        print(f"  {r['label']:<18} {r['seed']:>6} {r['front_size']:>6} "
              f"{r['HV']:>8.4f} {r['GD']:>8.4f} {r['IGD']:>8.4f} "
              f"{r['Spacing']:>9.4f} {r['elapsed_s']:>7.1f}s")

    print(f"\n{'='*92}")
    print("  Lecture : on cherche la config qui reduit le CV% de GD et de la taille")
    print("  de front SANS degrader le CV% de Spacing gagne par la mutation faible.")
    print(f"{'='*92}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparaison p_mut_strong QINSGA-III")
    parser.add_argument("--instance", default="100",
                        choices=["3", "5", "15", "25", "30", "40", "100"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--gen",   type=int, default=300)
    parser.add_argument("--pop",   type=int, default=200)
    args = parser.parse_args()
    run_comparison(args.instance, args.seeds, args.gen, args.pop)
