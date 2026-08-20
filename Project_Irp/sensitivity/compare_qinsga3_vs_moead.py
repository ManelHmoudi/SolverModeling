"""QI-NSGA-III vs MOEA/D -- statistical comparison on the real IRP.

Context: docs/superpowers/specs/2026-08-19-moead-integration-design.md.
Runs both algorithms' search fresh, once per seed, through the SAME
decode/repair/report pipeline (Solvers/NSGA3/report_builder.py::
_evaluate_pareto) every other algorithm comparison in this project uses.
Current production repair defaults (use_two_opt=False,
use_delivery_shift=True) are passed explicitly below -- _config_F's own
defaults (imported from compare_2opt_fairness.py) predate the production
default flip and are deliberately left unchanged there for that script's
own historical-campaign reproducibility, so they must be overridden here.

Statistics: this is an "engine effect" comparison (two independent search
engines, no shared chromosome array between them, unlike
sensitivity/compare_2opt_fairness.py's 2-opt-on/off pairing) -- Mann-Whitney
U (independent samples) is PRIMARY, matching
sensitivity/compare_qinsga3_vs_nsga3.py's own precedent for this same
comparison category. A same-seed-index paired Wilcoxon is also reported as
a SECONDARY, exploratory view -- do not read it as the headline result.
Brown-Forsythe (dispersion) and Holm-Bonferroni (multiple-comparison
correction across HV/GD/IGD/Spacing) apply to the primary Mann-Whitney
test. GD/IGD reference front: empirical leave-one-run-out (LORO), same as
every other comparison here (Solvers/NSGA3/metrics.py::
build_empirical_reference_front) -- each run scored against a front built
from every OTHER run only.

NOTE on statistical power at the default seed count: DEFAULT_SEEDS below is
a 3-seed smoke count, the same convention sensitivity/compare_2opt_
fairness.py uses for its own default (an established, deliberate
convention here -- a smoke-test default meant to be overridden via
--seeds for a real campaign, not something to change). The exact two-sided
Mann-Whitney U test used for this script's PRIMARY comparison has a floor
of p=0.10 at n=3 per group (see Solvers/QINSGA3/README.md's "small-sample
Mann-Whitney floor" discussion) -- it literally cannot reach p<0.05 no
matter how large the real effect is. Holm-Bonferroni's own most lenient
threshold across the 4 indicators tested together (HV/GD/IGD/Spacing) is
0.05/1=0.05, for whichever indicator has the largest (least significant)
raw p-value -- still below the n=3 Mann-Whitney floor of 0.10. So at the
DEFAULT seed count, no indicator can ever be reported significant after
correction, regardless of true effect size, and a "non significatif"
verdict from the Holm-Bonferroni table means nothing and should not be
read as "no difference between the algorithms". Use at least 5 seeds via
--seeds before trusting a non-significant result (minimum two-sided
Mann-Whitney p=0.0079 at n=5 per group); this project's own real
campaigns typically use 7-30 seeds.

MOEA/D's population is fixed at len(ref_dirs) = 165 (Das-Dennis
N_PARTITIONS=8, 4 objectives) -- NOT independently settable, unlike
QI-NSGA-III's pop_size=200. This is a structural property of MOEA/D
(pymoo's own MOEAD._setup(): one individual per decomposed subproblem),
not a tuning choice made here.

Usage:
    python -m sensitivity.compare_qinsga3_vs_moead --gen 5 --seeds 42   # smoke
    python -m sensitivity.compare_qinsga3_vs_moead --seeds 42 137 271 --gen 300
"""
from __future__ import annotations

import argparse
import os
import random as _random
import sys
import time

MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(MODULE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm   import PM
from pymoo.operators.sampling.rnd  import FloatRandomSampling
from pymoo.optimize                import minimize
from pymoo.termination             import get_termination
from pymoo.util.ref_dirs           import get_reference_directions
from scipy.stats import levene, mannwhitneyu, wilcoxon

from models.parametres         import load_instance
from Solvers.NSGA3.problem     import IRPProblem
from Solvers.NSGA3.metrics     import compute_pareto_metrics, build_empirical_reference_front
from Solvers.QINSGA3.algorithm import run_qinsga3
from Solvers.MOEAD._constrained_moead        import ConstrainedMOEAD
from Solvers.MOEAD._normalized_decomposition import NormalizedTchebycheff

from sensitivity.compare_2opt_fairness import (
    _stats, _wilcoxon_rank_biserial, _vargha_delaney_a12,
    _bootstrap_median_diff_ci, _holm_bonferroni, _config_F,
)

N_PARTITIONS  = 8
N_OBJ         = 4
QINSGA3_POP   = 200
DEFAULT_SEEDS = [42, 137, 271]

_INDICATORS       = ["HV", "GD", "IGD", "Spacing"]
_HIGHER_IS_BETTER = {"HV": True, "GD": False, "IGD": False, "Spacing": False}

ALPHA_MAX = 0.10 * np.pi
ALPHA_MIN = 0.001 * np.pi


def _run_moead_once(problem, ref_dirs, max_gen, seed) -> np.ndarray:
    np.random.seed(seed)
    _random.seed(seed)
    algorithm = ConstrainedMOEAD(
        ref_dirs      = ref_dirs,
        decomposition = NormalizedTchebycheff(),
        sampling      = FloatRandomSampling(),
        crossover     = SBX(prob=0.9, eta=20),
        mutation      = PM(prob=1.0, prob_var=1.0 / problem.n_var, eta=20),
    )
    result = minimize(
        problem, algorithm, get_termination("n_gen", max_gen),
        seed=seed, verbose=True,
    )
    return result.X


def _run_qinsga3_once(sets_, params_, ref_dirs, max_gen, seed) -> np.ndarray:
    pareto_X, _, _ = run_qinsga3(
        sets_=sets_, params_=params_, ref_dirs=ref_dirs, pop_size=QINSGA3_POP,
        max_gen=max_gen, alpha_max=ALPHA_MAX, alpha_min=ALPHA_MIN,
        p_cross=0.9, eta_cross=20.0, p_mut=None, eta_mut=20.0,
        migration_period=10, n_migrate=10, seed=seed,
        rotation_type="tanh", repair_final_front=False,
    )
    return pareto_X


def run_comparison(instance: str, seeds: list[int], max_gen: int) -> dict:
    data_path = os.path.join(PROJECT_DIR, "data", f"instance_{instance}_clients.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Instance non trouvee : {data_path}")
    sets_, params_ = load_instance(data_path)
    n_clients = len(sets_["clients"])

    problem   = IRPProblem(sets_, params_)
    ref_dirs  = get_reference_directions("das-dennis", N_OBJ, n_partitions=N_PARTITIONS)
    moead_pop = len(ref_dirs)

    print("=" * 88)
    print("  QI-NSGA-III vs MOEA/D -- comparaison statistique (IRP)")
    print("=" * 88)
    print(f"  Instance        : {n_clients} clients")
    print(f"  Seeds           : {seeds}")
    print(f"  Generations     : {max_gen} (partagees, meme budget nominal)")
    print(f"  Pop QI-NSGA-III : {QINSGA3_POP}")
    print(f"  Pop MOEA/D      : {moead_pop} (= len(ref_dirs), non reglable independamment)")
    print("=" * 88)

    per_seed_F = {"QI-NSGA-III": {}, "MOEA/D": {}}
    for seed in seeds:
        print(f"\n--- seed={seed} ---")
        t0 = time.time()
        qi_X = _run_qinsga3_once(sets_, params_, ref_dirs, max_gen, seed)
        n_qi = 0 if qi_X is None else len(qi_X)
        print(f"  QI-NSGA-III: {n_qi} solutions en {time.time() - t0:.1f}s")

        t0 = time.time()
        mo_X = _run_moead_once(problem, ref_dirs, max_gen, seed)
        n_mo = 0 if mo_X is None else len(mo_X)
        print(f"  MOEA/D     : {n_mo} solutions en {time.time() - t0:.1f}s")

        # Either search can legitimately return None/empty (zero feasible
        # solutions found at this seed, a realistic outcome given the IRP's
        # hard constraints -- see Solvers/MOEAD/main.py::run_moead's own
        # identical guard). Skip the WHOLE seed for BOTH algorithms rather
        # than only the failing one, so per_seed_F["QI-NSGA-III"] and
        # per_seed_F["MOEA/D"] stay seed-key-aligned for the LORO/quality
        # loop and the paired-Wilcoxon section below, both of which assume
        # matching seed sets between the two algorithms.
        if n_qi == 0 or n_mo == 0:
            print(f"  -- seed={seed}: aucune solution faisable pour au moins un algorithme, "
                  "seed ignoree pour les deux --")
            continue

        # Shared repair pipeline, current production defaults.
        per_seed_F["QI-NSGA-III"][seed] = _config_F(
            qi_X, sets_, params_, repair=True, use_two_opt=False, use_delivery_shift=True,
        )
        per_seed_F["MOEA/D"][seed] = _config_F(
            mo_X, sets_, params_, repair=True, use_two_opt=False, use_delivery_shift=True,
        )

    all_F = np.vstack([F for algo in per_seed_F.values() for F in algo.values()])
    g_ideal, g_nadir = all_F.min(axis=0), all_F.max(axis=0)

    # LORO (leave-one-run-out) reference front, pooled across BOTH algorithms
    # -- not per-algorithm. Scoring QI-NSGA-III's runs only against other
    # QI-NSGA-III runs (and MOEA/D's only against other MOEA/D runs) would
    # measure within-algorithm consistency, not solution quality, and would
    # put the two algorithms' GD/IGD on non-comparable scales, invalidating
    # the Mann-Whitney test on those two indicators. Follows the same
    # pooled-LORO pattern as sensitivity/compare_2opt_fairness.py's own
    # run_comparison (see its _flat_runs / "other_F" construction): every
    # run, from either algorithm, is a reference-front candidate for every
    # OTHER run -- only the exact (algo, seed) pair being scored is excluded.
    _flat = [(a, s, F) for a, by_seed in per_seed_F.items() for s, F in by_seed.items()]
    quality = {"QI-NSGA-III": {ind: [] for ind in _INDICATORS},
               "MOEA/D":      {ind: [] for ind in _INDICATORS}}
    for algo, by_seed in per_seed_F.items():
        for seed, F_run in by_seed.items():
            other_F = [F for a2, s2, F in _flat if not (a2 == algo and s2 == seed)]
            pf_ref  = build_empirical_reference_front(np.vstack(other_F)) if other_F else None
            q = compute_pareto_metrics(F_run, g_ideal, g_nadir, reference_front=pf_ref)
            for ind in _INDICATORS:
                quality[algo][ind].append(q[ind])

    print(f"\n{'-' * 88}")
    print("  RESULTATS")
    print(f"{'-' * 88}")
    pvalues_mw = {}
    for ind in _INDICATORS:
        arrow = "^" if _HIGHER_IS_BETTER[ind] else "v"
        qi_vals = quality["QI-NSGA-III"][ind]
        mo_vals = quality["MOEA/D"][ind]
        print(f"\n  {ind} ({arrow})")
        for algo in ("QI-NSGA-III", "MOEA/D"):
            s = _stats(quality[algo][ind])
            print(f"    {algo:<12} mean={s['mean']:.6f}  std={s['std']:.6f}")

        u_stat, p_mw = mannwhitneyu(qi_vals, mo_vals, alternative="two-sided")
        a12 = _vargha_delaney_a12(u_stat, len(qi_vals), len(mo_vals))
        pvalues_mw[ind] = p_mw
        print(f"    Mann-Whitney U = {u_stat:.1f}, p = {p_mw:.6f}  (A12={a12:.3f}) [PRIMAIRE]")

        if len(qi_vals) == len(mo_vals) and len(qi_vals) >= 2:
            try:
                w_stat, p_w = wilcoxon(qi_vals, mo_vals)
                r = _wilcoxon_rank_biserial(qi_vals, mo_vals)
                print(f"    Wilcoxon apparie (seed) = {w_stat:.1f}, p = {p_w:.6f}  "
                      f"(r={r:.3f}) [secondaire, exploratoire]")
                median_diff, ci_lo, ci_hi = _bootstrap_median_diff_ci(qi_vals, mo_vals)
                print(f"    Diff median (QI - MOEA/D) = {median_diff:.6f}  "
                      f"IC95%=[{ci_lo:.6f}, {ci_hi:.6f}]")
            except ValueError as e:
                print(f"    Wilcoxon apparie (seed): {e}")

        if len(qi_vals) >= 2 and len(mo_vals) >= 2:
            lev_stat, p_lev = levene(qi_vals, mo_vals, center="median")
            print(f"    Brown-Forsythe (dispersion) p = {p_lev:.6f}")

    holm = _holm_bonferroni(pvalues_mw)
    print(f"\n{'-' * 88}")
    print("  Correction Holm-Bonferroni (sur le test Mann-Whitney primaire)")
    print(f"{'-' * 88}")
    for ind, (p, threshold, sig) in holm.items():
        tag = "significatif" if sig else "non significatif"
        print(f"    {ind:<10} p={p:.6f}  seuil={threshold:.6f}  -> {tag} (apres correction)")

    print(f"\n{'=' * 88}\n")
    return {"quality": quality, "holm": holm}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare QI-NSGA-III vs MOEA/D on the IRP (Mann-Whitney primary)"
    )
    parser.add_argument("--instance", default="100",
                        choices=["3", "5", "15", "25", "30", "40", "100", "150"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--gen", type=int, default=300)
    args = parser.parse_args()
    run_comparison(args.instance, args.seeds, args.gen)
