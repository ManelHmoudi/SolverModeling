"""Statistical comparison: QI-NSGA-III vs classic NSGA-III on the real IRP.

Mann-Whitney U (= Wilcoxon rank-sum) test on HV/GD/IGD/Spacing across the
cached runs of each algorithm on the same instance, testing whether the
observed difference in each indicator is statistically significant
(H0: the two algorithms' indicator distributions are identical).

Both algorithms' metrics are recomputed here from their cached chromosomes
using ONE shared global ideal/nadir across all runs of BOTH algorithms
combined -- NOT the per-run ideal/nadir already stored in the cache JSONs
(Solvers/NSGA3/nsga3_chromosomes.json, Solvers/QINSGA3/qinsga3_chromosomes.json),
which are each normalised to their own run's scale and are NOT directly
comparable across algorithms or even across runs of the same algorithm. This
mirrors Solvers/NSGA3/report_builder.py::_build_report_data's within-algorithm
global-ideal/nadir approach, extended across both algorithms so HV in
particular (which depends on the reference point) is measured on the same
scale for both.

CAVEAT: _decode_to_F decodes both algorithms' cached chromosomes with the
plain, unmodified decoder regardless of what each cache's
meta_base["repair_final_front"] says -- this script deliberately does NOT
apply the 2-opt final-front repair to either side, even if the cache was
produced with repair_final_front=True. That is intentional here: this
script wants the raw evolutionary-engine gap between the two algorithms,
not a gap that also depends on whether 2-opt repair was applied. See
sensitivity/compare_2opt_fairness.py for a 4-configuration comparison that
does account for repair_final_front.

Usage:
    python -m sensitivity.compare_qinsga3_vs_nsga3
    python -m sensitivity.compare_qinsga3_vs_nsga3 --instance 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys

MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(MODULE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np
from scipy.stats import mannwhitneyu

from models.parametres         import load_instance
from Solvers.NSGA3.decoder      import decode_chromosome, build_routes
from Solvers.NSGA3.evaluator    import compute_f1, compute_f2, compute_f3, compute_f4
from Solvers.NSGA3.metrics      import compute_pareto_metrics

_CACHE_PATHS = {
    "NSGA-III":    os.path.join(PROJECT_DIR, "Solvers", "NSGA3", "nsga3_chromosomes.json"),
    "QI-NSGA-III": os.path.join(PROJECT_DIR, "Solvers", "QINSGA3", "qinsga3_chromosomes.json"),
}

_INDICATORS = ["HV", "GD", "IGD", "Spacing"]
_HIGHER_IS_BETTER = {"HV": True, "GD": False, "IGD": False, "Spacing": False}


def _load_cache(algo: str) -> dict:
    path = _CACHE_PATHS[algo]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No cached chromosomes for {algo} at {path}. "
            "Run the algorithm at least once first (same instance for both)."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _decode_to_F(chromosomes: list, sets_: dict, params_: dict) -> np.ndarray:
    """Decode one run's cached Pareto chromosomes into their (f1,f2,f3,f4) objective matrix."""
    F = []
    for chrom in chromosomes:
        quantities, priorities = decode_chromosome(np.array(chrom), sets_)
        route_result = build_routes(quantities, sets_, params_, priorities)
        F.append([
            compute_f1(route_result, sets_, params_),
            compute_f2(route_result, sets_, params_),
            compute_f3(route_result, sets_, params_),
            compute_f4(route_result, sets_, params_),
        ])
    return np.array(F)


def _stats(vals: list) -> dict:
    a = np.array(vals, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std())}


def run_comparison(instance: str) -> None:
    data_path = os.path.join(PROJECT_DIR, "data", f"instance_{instance}_clients.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Instance non trouvee : {data_path}")
    sets_, params_ = load_instance(data_path)

    caches = {algo: _load_cache(algo) for algo in _CACHE_PATHS}

    expected_instance = f"{instance}_clients"
    for algo, cache in caches.items():
        cached_instance = cache["runs"][0]["meta_base"]["instance"]
        if cached_instance != expected_instance:
            raise ValueError(
                f"{algo}'s cache is for instance '{cached_instance}', not "
                f"'{expected_instance}' -- re-run both algorithms on the same "
                "instance before comparing, or pass the matching --instance."
            )

    print("=" * 88)
    print("  QI-NSGA-III vs NSGA-III -- Mann-Whitney U test (HV, GD, IGD, Spacing)")
    print("=" * 88)
    print(f"  Instance : {instance} clients")
    for algo, cache in caches.items():
        seeds = [r["seed"] for r in cache["runs"]]
        print(f"  {algo:<12} : {len(seeds)} runs, seeds={seeds}")
    print("=" * 88)

    print("\nDecoding cached chromosomes into objectives (re-evaluates every "
          "Pareto solution against the current instance data)...")
    per_run_F: dict = {}
    for algo, cache in caches.items():
        per_run_F[algo] = []
        for run in cache["runs"]:
            F_run = _decode_to_F(run["chromosomes"], sets_, params_)
            per_run_F[algo].append(F_run)
        n_sol = sum(len(f) for f in per_run_F[algo])
        print(f"  {algo}: {len(per_run_F[algo])} runs decoded ({n_sol} Pareto solutions total)")

    all_F = np.vstack([F for runs in per_run_F.values() for F in runs])
    g_ideal = all_F.min(axis=0)
    g_nadir = all_F.max(axis=0)
    print(f"\nShared global ideal : {g_ideal}")
    print(f"Shared global nadir  : {g_nadir}")

    values: dict = {ind: {algo: [] for algo in caches} for ind in _INDICATORS}
    for algo, runs in per_run_F.items():
        for F_run in runs:
            q = compute_pareto_metrics(F_run, g_ideal, g_nadir)
            for ind in _INDICATORS:
                values[ind][algo].append(q[ind])

    print(f"\n{'-'*88}")
    print("  RESULTATS (ideal/nadir global partage entre les 2 algorithmes)")
    print(f"{'-'*88}")
    for ind in _INDICATORS:
        arrow = "^" if _HIGHER_IS_BETTER[ind] else "v"
        print(f"\n  {ind} ({arrow})")
        for algo in caches:
            s = _stats(values[ind][algo])
            print(f"    {algo:<12} mean={s['mean']:.6f}  std={s['std']:.6f}")
        u_stat, p_value = mannwhitneyu(
            values[ind]["QI-NSGA-III"], values[ind]["NSGA-III"], alternative="two-sided"
        )
        sig = "significatif (p < 0.05)" if p_value < 0.05 else "non significatif (p >= 0.05)"
        print(f"    Mann-Whitney U = {u_stat:.1f}, p = {p_value:.6f}  -> {sig}")

    print(f"\n{'='*88}")
    print("  H0 : les deux algorithmes ont la meme distribution pour cet indicateur.")
    print("  p < 0.05 -> on rejette H0, l'ecart observe est statistiquement significatif.")
    print(f"{'='*88}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare QI-NSGA-III vs NSGA-III via Mann-Whitney U on HV/GD/IGD/Spacing"
    )
    parser.add_argument("--instance", default="100",
                        choices=["3", "5", "15", "25", "30", "40", "100"])
    args = parser.parse_args()
    run_comparison(args.instance)
