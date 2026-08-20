"""NSGA-III vs QI-NSGA-III vs MOEA/D -- 3-way statistical comparison on the
real IRP.

Context: user asked to compare all 3 algorithms together "sur une base
solide" specifically to measure QI-NSGA-III's own impact/added value,
rather than reading that off two separate 2-way comparisons
(compare_qinsga3_vs_nsga3.py, compare_qinsga3_vs_moead.py) whose seeds,
instance, and reference fronts are never guaranteed to line up. This
script runs all three algorithms fresh, once per seed, on the SAME
instance, through the SAME decode/repair/report pipeline
(Solvers/NSGA3/report_builder.py::_evaluate_pareto) with explicit current
production defaults (use_two_opt=False, use_delivery_shift=True --
_config_F's own defaults predate the production default flip and are
deliberately left unchanged for compare_2opt_fairness.py's own
historical-campaign reproducibility, so they are overridden here exactly
as in compare_qinsga3_vs_moead.py).

Statistics: 3 pairwise Mann-Whitney U tests per indicator (QI vs NSGA-III,
QI vs MOEA/D, NSGA-III vs MOEA/D) x 4 indicators (HV/GD/IGD/Spacing) = 12
tests, Holm-Bonferroni corrected together as one family. This is an
"engine effect" comparison for all three pairs (three independent search
engines, no shared chromosome array between any two of them) -- Mann-
Whitney is the appropriate test for every pair, same reasoning as
compare_qinsga3_vs_moead.py's own primary test (see that file's docstring
for the fuller argument re: why not paired Wilcoxon here). Chosen over a
Kruskal-Wallis omnibus + post-hoc scheme: pairwise Mann-Whitney with a
single 12-test Holm correction gives the per-pair detail needed to answer
the actual question ("what does QI-NSGA-III specifically add against
EACH of the other two"), not just "do the 3 groups differ somewhere".

GD/IGD reference front: empirical leave-one-run-out (LORO), pooled across
ALL THREE algorithms -- not per-algorithm and not per-pair. Every run,
from any of the 3 algorithms, is a reference-front candidate for every
OTHER run; only the exact (algo, seed) pair being scored is excluded.
Scoring an algorithm's runs only against its own other runs (or only
against one other algorithm) would measure within-algorithm consistency
or a partial-pair scale, not solution quality on a scale comparable across
all three -- same reasoning, generalized to 3 algorithms, as the pooling
fix already applied in compare_qinsga3_vs_moead.py.

NOTE on statistical power at the default seed count: DEFAULT_SEEDS below
is a 3-seed smoke count, the same convention every other comparison script
in this project uses for its own default (compare_2opt_fairness.py,
compare_qinsga3_vs_moead.py) -- a smoke-test default meant to be
overridden via --seeds for a real campaign, not something to change here.
At n=3 per group, the exact two-sided Mann-Whitney U test has a floor of
p=0.10 -- it literally cannot reach p<0.05 no matter how large the real
effect is. With 12 tests in one Holm-Bonferroni family, the single most
lenient threshold (largest raw p-value) is 0.05/1=0.05, still below the
n=3 floor of 0.10 -- so at the DEFAULT seed count NO pairwise comparison
can ever be reported significant after correction, and a "non
significatif" verdict means nothing at this seed count. Use at least 5
seeds via --seeds before trusting a non-significant result (minimum
two-sided Mann-Whitney p=0.0079 at n=5 per group); this project's own real
campaigns typically use 7-30 seeds.

MOEA/D's population is fixed at len(ref_dirs) = 165 (Das-Dennis
N_PARTITIONS=8, 4 objectives) -- NOT independently settable, unlike
NSGA-III's and QI-NSGA-III's own pop_size=200. This is a structural
property of MOEA/D (pymoo's own MOEAD._setup(): one individual per
decomposed subproblem), not a tuning choice made here.

DELIBERATE, DOCUMENTED LIMITATION -- evaluation budget is NOT equalized
across the three algorithms. All three share the same --gen (generation
count), not the same total evaluation count: NSGA-III and QI-NSGA-III
evaluate ~pop_size (~200) individuals per generation, while MOEA/D
evaluates exactly one offspring per subproblem per generation
(len(ref_dirs) = 165) -- at gen=300 this is roughly 60000 evaluations for
NSGA-III, ~66000 for QI-NSGA-III (its own external archive adds real
evaluations beyond the base population -- see Solvers/QINSGA3/README.md),
but only ~49500 for MOEA/D, about 17.5% fewer than NSGA-III. This mirrors
the exact same equal-generations-vs-equal-evaluations question already
investigated for QI-NSGA-III vs NSGA-III on this project's 100-client
instance (see sensitivity/compare_2opt_fairness.py's own NOTE on
evaluation budget, and its --qinsga3-gen calibration mechanism) --
resolved there in favour of measuring both under BOTH conventions. Here,
the explicit choice (confirmed with the user rather than assumed) is to
keep generations equal across all three and read results with this bias
in mind, not to calibrate MOEA/D's own generation count upward to match
evaluation totals -- report any MOEA/D-unfavourable result from this
script alongside this caveat, the same way n_neighbors/
prob_neighbor_mating being left at pymoo's defaults (never IRP-tuned,
unlike NSGA-III's/QI-NSGA-III's own operators) is already flagged as a
limitation rather than silently absorbed into the headline numbers.

Usage:
    python -m sensitivity.compare_3algos --gen 5 --seeds 42            # smoke
    python -m sensitivity.compare_3algos --seeds 42 137 271 --gen 300  # real (3-seed)
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
from pymoo.algorithms.moo.nsga3    import NSGA3
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
NSGA3_POP     = 200
QINSGA3_POP   = 200
DEFAULT_SEEDS = [42, 137, 271]

_ALGOS      = ["NSGA-III", "QI-NSGA-III", "MOEA/D"]
_PAIRS      = [("QI-NSGA-III", "NSGA-III"), ("QI-NSGA-III", "MOEA/D"), ("NSGA-III", "MOEA/D")]
_INDICATORS = ["HV", "GD", "IGD", "Spacing"]
_HIGHER_IS_BETTER = {"HV": True, "GD": False, "IGD": False, "Spacing": False}

ALPHA_MAX = 0.10 * np.pi
ALPHA_MIN = 0.001 * np.pi


def _run_nsga3_once(sets_, params_, ref_dirs, max_gen, seed) -> np.ndarray:
    """Plain (no external archive) NSGA-III search, matching the project's
    production default (Solvers/NSGA3/main.py::run_nsga3's own
    use_archive=False default) -- same pymoo configuration
    sensitivity/compare_2opt_fairness.py::_run_nsga3_once uses for its own
    plain (use_archive=False) case."""
    np.random.seed(seed)
    _random.seed(seed)

    problem = IRPProblem(sets_, params_)
    n_genes = len(sets_["clients"]) * len(sets_["T"]) + len(sets_["clients"])

    algorithm = NSGA3(
        pop_size  = NSGA3_POP,
        ref_dirs  = ref_dirs,
        sampling  = FloatRandomSampling(),
        crossover = SBX(prob=0.9, eta=20),
        mutation  = PM(prob=1.0, prob_var=1.0 / n_genes, eta=20),
    )
    result = minimize(
        problem, algorithm, get_termination("n_gen", max_gen), seed=seed, verbose=True,
    )
    return result.X if result.X is not None else np.empty((0, n_genes))


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
    print("  NSGA-III vs QI-NSGA-III vs MOEA/D -- comparaison statistique a 3 (IRP)")
    print("=" * 88)
    print(f"  Instance          : {n_clients} clients")
    print(f"  Seeds             : {seeds}")
    print(f"  Generations       : {max_gen} (partagees, meme budget nominal)")
    print(f"  Pop NSGA-III      : {NSGA3_POP}")
    print(f"  Pop QI-NSGA-III   : {QINSGA3_POP}")
    print(f"  Pop MOEA/D        : {moead_pop} (= len(ref_dirs), non reglable independamment)")
    print("=" * 88)

    per_seed_F = {algo: {} for algo in _ALGOS}
    runtimes   = {algo: [] for algo in _ALGOS}
    for seed in seeds:
        print(f"\n--- seed={seed} ---")

        t0 = time.time()
        n3_X = _run_nsga3_once(sets_, params_, ref_dirs, max_gen, seed)
        elapsed_n3 = time.time() - t0
        n_n3 = 0 if n3_X is None else len(n3_X)
        print(f"  NSGA-III   : {n_n3} solutions en {elapsed_n3:.1f}s")

        t0 = time.time()
        qi_X = _run_qinsga3_once(sets_, params_, ref_dirs, max_gen, seed)
        elapsed_qi = time.time() - t0
        n_qi = 0 if qi_X is None else len(qi_X)
        print(f"  QI-NSGA-III: {n_qi} solutions en {elapsed_qi:.1f}s")

        t0 = time.time()
        mo_X = _run_moead_once(problem, ref_dirs, max_gen, seed)
        elapsed_mo = time.time() - t0
        n_mo = 0 if mo_X is None else len(mo_X)
        print(f"  MOEA/D     : {n_mo} solutions en {elapsed_mo:.1f}s")

        # Any of the 3 searches can legitimately return None/empty (zero
        # feasible solutions at this seed, a realistic outcome given the
        # IRP's hard constraints -- see Solvers/MOEAD/main.py::run_moead's
        # own identical guard). Skip the WHOLE seed for ALL THREE algorithms
        # rather than only the failing one(s), so every algorithm's
        # per_seed_F stays seed-key-aligned for the LORO/quality loop and
        # the paired-Wilcoxon section below, both of which assume matching
        # seed sets across all three algorithms.
        if n_n3 == 0 or n_qi == 0 or n_mo == 0:
            print(f"  -- seed={seed}: aucune solution faisable pour au moins un algorithme, "
                  "seed ignoree pour les trois --")
            continue

        # Shared repair pipeline, current production defaults.
        per_seed_F["NSGA-III"][seed] = _config_F(
            n3_X, sets_, params_, repair=True, use_two_opt=False, use_delivery_shift=True,
        )
        per_seed_F["QI-NSGA-III"][seed] = _config_F(
            qi_X, sets_, params_, repair=True, use_two_opt=False, use_delivery_shift=True,
        )
        per_seed_F["MOEA/D"][seed] = _config_F(
            mo_X, sets_, params_, repair=True, use_two_opt=False, use_delivery_shift=True,
        )
        runtimes["NSGA-III"].append(elapsed_n3)
        runtimes["QI-NSGA-III"].append(elapsed_qi)
        runtimes["MOEA/D"].append(elapsed_mo)

    all_F = np.vstack([F for algo in per_seed_F.values() for F in algo.values()])
    g_ideal, g_nadir = all_F.min(axis=0), all_F.max(axis=0)

    # LORO (leave-one-run-out) reference front, pooled across ALL THREE
    # algorithms -- see module docstring. Every run, from any of the 3
    # algorithms, is a reference-front candidate for every OTHER run; only
    # the exact (algo, seed) pair being scored is excluded.
    _flat = [(a, s, F) for a, by_seed in per_seed_F.items() for s, F in by_seed.items()]
    quality = {algo: {ind: [] for ind in _INDICATORS} for algo in _ALGOS}
    for algo, by_seed in per_seed_F.items():
        for seed, F_run in by_seed.items():
            other_F = [F for a2, s2, F in _flat if not (a2 == algo and s2 == seed)]
            pf_ref  = build_empirical_reference_front(np.vstack(other_F)) if other_F else None
            q = compute_pareto_metrics(F_run, g_ideal, g_nadir, reference_front=pf_ref)
            for ind in _INDICATORS:
                quality[algo][ind].append(q[ind])

    print(f"\n{'-' * 88}")
    print("  TEMPS DE CALCUL -- moyenne +/- ecart-type par algorithme, par run (secondes)")
    print("  (informatif -- pas un indicateur de qualite de front ; budgets d'evaluations")
    print("   non egalises entre les 3 algos, voir la NOTE dans l'entete du module)")
    print(f"{'-' * 88}")
    for algo in _ALGOS:
        s = _stats(runtimes[algo])
        print(f"    {algo:<12} mean={s['mean']:.1f}s  std={s['std']:.1f}s")
    rt_pvalues = {}
    for a, b in _PAIRS:
        a_t, b_t = runtimes[a], runtimes[b]
        u_stat, p_mw = mannwhitneyu(a_t, b_t, alternative="two-sided")
        rt_pvalues[f"{a} vs {b}"] = p_mw
        print(f"    {a:<12} vs {b:<12} : Mann-Whitney U={u_stat:.1f}  p={p_mw:.6f}")
    rt_holm = _holm_bonferroni(rt_pvalues) if rt_pvalues else {}
    for name, (p, threshold, sig) in rt_holm.items():
        tag = "significatif" if sig else "non significatif"
        print(f"      -> {name:<28} p={p:.6f}  seuil={threshold:.6f}  -> {tag} (Holm, famille de 3)")

    print(f"\n{'-' * 88}")
    print("  RESULTATS -- moyennes par algorithme")
    print(f"{'-' * 88}")
    for ind in _INDICATORS:
        arrow = "^" if _HIGHER_IS_BETTER[ind] else "v"
        print(f"\n  {ind} ({arrow})")
        for algo in _ALGOS:
            s = _stats(quality[algo][ind])
            print(f"    {algo:<12} mean={s['mean']:.6f}  std={s['std']:.6f}")

    print(f"\n{'-' * 88}")
    print("  TESTS PAR PAIRES (Mann-Whitney U, primaire) -- 3 paires x 4 indicateurs = 12 tests")
    print(f"{'-' * 88}")
    pvalues_mw = {}
    for ind in _INDICATORS:
        arrow = "^" if _HIGHER_IS_BETTER[ind] else "v"
        print(f"\n  {ind} ({arrow})")
        for a, b in _PAIRS:
            a_vals, b_vals = quality[a][ind], quality[b][ind]
            u_stat, p_mw = mannwhitneyu(a_vals, b_vals, alternative="two-sided")
            a12 = _vargha_delaney_a12(u_stat, len(a_vals), len(b_vals))
            pvalues_mw[f"{ind}::{a} vs {b}"] = p_mw
            print(f"    {a:<12} vs {b:<12} : U={u_stat:.1f}  p={p_mw:.6f}  (A12={a12:.3f})")

            if len(a_vals) == len(b_vals) and len(a_vals) >= 2:
                try:
                    w_stat, p_w = wilcoxon(a_vals, b_vals)
                    r = _wilcoxon_rank_biserial(a_vals, b_vals)
                    median_diff, ci_lo, ci_hi = _bootstrap_median_diff_ci(a_vals, b_vals)
                    print(f"      Wilcoxon apparie (seed) = {w_stat:.1f}, p = {p_w:.6f}  "
                          f"(r={r:.3f}) [secondaire, exploratoire]")
                    print(f"      Diff median ({a} - {b}) = {median_diff:.6f}  "
                          f"IC95%=[{ci_lo:.6f}, {ci_hi:.6f}]")
                except ValueError as e:
                    print(f"      Wilcoxon apparie (seed): {e}")

            if len(a_vals) >= 2 and len(b_vals) >= 2:
                lev_stat, p_lev = levene(a_vals, b_vals, center="median")
                print(f"      Brown-Forsythe (dispersion) p = {p_lev:.6f}")

    holm = _holm_bonferroni(pvalues_mw)
    print(f"\n{'-' * 88}")
    print("  Correction Holm-Bonferroni (famille unique des 12 tests Mann-Whitney)")
    print(f"{'-' * 88}")
    for name, (p, threshold, sig) in holm.items():
        tag = "significatif" if sig else "non significatif"
        print(f"    {name:<32} p={p:.6f}  seuil={threshold:.6f}  -> {tag} (apres correction)")

    print(f"\n{'=' * 88}\n")
    return {"quality": quality, "holm": holm, "runtimes": runtimes, "runtime_holm": rt_holm}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare NSGA-III vs QI-NSGA-III vs MOEA/D on the IRP "
                     "(pairwise Mann-Whitney, Holm-corrected as one 12-test family)"
    )
    parser.add_argument("--instance", default="100",
                        choices=["3", "5", "15", "25", "30", "40", "100", "150"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--gen", type=int, default=300)
    args = parser.parse_args()
    run_comparison(args.instance, args.seeds, args.gen)
