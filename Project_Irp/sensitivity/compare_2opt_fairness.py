"""2-opt fairness campaign -- NSGA-III vs QI-NSGA-III, with/without the
post-decode 2-opt repair applied to BOTH algorithms.

Context: docs/superpowers/specs/2026-08-10-qinsga3-2opt-fairness-campaign-
design.md. QI-NSGA-III's 2-opt repair (Solvers/QINSGA3/repair.py, "Remede
G") has so far only ever been applied to QI-NSGA-III's own front, then
compared against a plain (unrepaired) NSGA-III baseline
(sensitivity/compare_route_repair_final.py) -- conflating the
evolutionary-engine effect (quantum rotation gate vs. plain NSGA-III) with
the post-processing effect (2-opt vs. none). This script isolates the two
by running each algorithm's search ONCE per seed, then decoding the SAME
resulting chromosome array twice -- once without repair, once with (the
repair is Baldwinian: it never touches the chromosome, only the fitness
reported for it, via Solvers/NSGA3/report_builder.py::_evaluate_pareto's
repair flag) -- producing four configurations from two searches per seed,
all sharing identical initial fronts, seeds, and budgets:

    1. NSGA-III      without 2-opt
    2. NSGA-III      with 2-opt
    3. QI-NSGA-III   without 2-opt
    4. QI-NSGA-III   with 2-opt

Statistics separate the two effects:
    - 2-opt effect   (paired Wilcoxon, same front, same seed): 1 vs 2 (NSGA-
      III), 3 vs 4 (QI-NSGA-III).
    - Engine effect  (Mann-Whitney, independent fronts): 1 vs 3 (raw gap, no
      post-processing on either side -- reproduces what
      compare_qinsga3_vs_nsga3.py already measures) and 2 vs 4 (fair gap,
      both sides get the same 2-opt polish -- the number that actually
      answers "is the quantum mechanism itself better", isolated from the
      repair).

NOTE on statistical power at the default seed count: DEFAULT_SEEDS below is
a 3-seed smoke count. The exact two-sided Wilcoxon signed-rank test used for
the paired "Effet 2-opt" section has a floor of p=0.25 at n=3 -- it
literally cannot reach p<0.05 no matter how large the real effect is, so a
"non significatif" verdict from that section at the default seed count means
nothing and should not be read as "2-opt has no effect". This is a smaller-
sample floor than Mann-Whitney's own n=3 floor (p=0.10, see
Solvers/QINSGA3/README.md's "small-sample Mann-Whitney floor" discussion),
which affects the "Effet moteur" section instead. Use 6+ seeds before
trusting a non-significant "Effet 2-opt" result.

NOTE on evaluation budget: matching max_gen and pop_size does NOT match the
number of objective-function evaluations actually spent. NSGA-III evaluates
exactly effective_pop individuals per generation (pop_size infills each
iteration, pymoo's own GeneticAlgorithm default) -> effective_pop * max_gen
total. QI-NSGA-III's run_qinsga3 evaluates the PARENT population (needed
every generation since qpop.measure() redraws measurement noise, so a
survivor's cached F from the previous generation is stale) AND the offspring
population each generation, plus one final measurement pass after the loop
-> effective_pop * (2 * max_gen + 1) total -- essentially double. This
script reports both algorithms' real evaluation counts (n_eval) alongside
every result so the "Effet moteur quantique" section can be read with this
in mind: any QI-NSGA-III advantage there is confounded with QI-NSGA-III
having spent roughly 2x the function evaluations, not isolated to the
rotation-gate mechanism alone. The "Effet 2-opt" section is NOT affected by
this (it compares each algorithm only to itself, at whatever budget it
actually used). Pass --qinsga3-gen (e.g. --gen 300 --qinsga3-gen 150) to run
QI-NSGA-III at a different generation count than NSGA-III specifically to
match real evaluation budgets instead of max_gen -- 150 gives
effective_pop*(2*150+1), within 0.3% of NSGA-III's effective_pop*300 at the
project's own pop=200 (60200 vs 60000).

NSGA-III's search is replicated locally (same pymoo setup
Solvers/NSGA3/main.py::run_nsga3 uses) rather than calling run_nsga3
itself, since that function has no low-level mode that skips writing the
shared nsga3_chromosomes.json cache file. QI-NSGA-III's search uses the
existing low-level Solvers.QINSGA3.algorithm.run_qinsga3 directly, which
already returns raw arrays without touching any cache -- same pattern
sensitivity/compare_route_repair_final.py already uses.

Usage:
    python -m sensitivity.compare_2opt_fairness --gen 5 --seeds 42   # smoke
    python -m sensitivity.compare_2opt_fairness --seeds 42 137 271 --gen 300
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
from pymoo.algorithms.moo.nsga3    import NSGA3, ReferenceDirectionSurvival
from pymoo.util.ref_dirs           import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm   import PM
from pymoo.operators.sampling.rnd  import FloatRandomSampling
from pymoo.optimize                import minimize
from pymoo.termination             import get_termination
from pymoo.util.archive            import MultiObjectiveArchive, SurvivalTruncation
from scipy.stats import mannwhitneyu, wilcoxon

from models.parametres            import load_instance
from Solvers.NSGA3.metrics        import compute_pareto_metrics
from Solvers.NSGA3.problem        import IRPProblem
from Solvers.NSGA3.report_builder import _evaluate_pareto
from Solvers.QINSGA3.algorithm    import run_qinsga3

N_PARTITIONS = 8
N_OBJ        = 4
POP_SIZE     = 200
DEFAULT_SEEDS = [42, 137, 271]

_INDICATORS       = ["HV", "GD", "IGD", "Spacing"]
_HIGHER_IS_BETTER = {"HV": True, "GD": False, "IGD": False, "Spacing": False}

_CONFIGS = [
    "NSGA-III sans 2-opt", "NSGA-III avec 2-opt",
    "QI-NSGA-III sans 2-opt", "QI-NSGA-III avec 2-opt",
]


def _run_nsga3_once(sets_, params_, ref_dirs, effective_pop, max_gen, seed,
                     use_archive: bool = False):
    """Local replication of Solvers/NSGA3/main.py::run_nsga3's search setup,
    returning (pareto_X, n_eval) -- run_nsga3 itself has no low-level
    equivalent that skips writing the shared chromosome cache file, so this
    mirrors its exact pymoo configuration instead (same pattern
    sensitivity/compare_route_repair_final.py already uses for QI-NSGA-III's
    low-level run_qinsga3 call). n_eval is pymoo's own evaluation counter
    (result.algorithm.evaluator.n_eval), not a derived estimate.

    use_archive: mirrors Solvers/NSGA3/main.py::run_nsga3's own use_archive
    parameter -- see its docstring for the fairness-audit finding this
    tests (QI-NSGA-III returns its front from an external archive spanning
    the whole run, NSGA-III returns only its final generation's population
    by default)."""
    np.random.seed(seed)
    _random.seed(seed)

    problem = IRPProblem(sets_, params_)
    n_genes = len(sets_["clients"]) * len(sets_["T"]) + len(sets_["clients"])

    archive_survival = ReferenceDirectionSurvival(ref_dirs) if use_archive else None
    archive = (
        MultiObjectiveArchive(
            max_size=500,
            truncation=SurvivalTruncation(archive_survival, problem),
        )
        if use_archive else None
    )

    algorithm = NSGA3(
        pop_size  = effective_pop,
        ref_dirs  = ref_dirs,
        sampling  = FloatRandomSampling(),
        archive   = archive,
        crossover = SBX(prob=0.9, eta=20),
        # prob=1.0 disables pymoo's per-individual mutation gate, prob_var
        # sets the per-gene rate -- matches the fix in Solvers/NSGA3/
        # main.py::run_nsga3, which this function mirrors.
        mutation  = PM(prob=1.0, prob_var=1.0 / n_genes, eta=20),
    )
    result = minimize(
        problem, algorithm, get_termination("n_gen", max_gen), seed=seed, verbose=False,
    )

    if use_archive and result.archive is not None and len(result.archive) > 0:
        arch_pop = result.archive
        if len(arch_pop) > effective_pop:
            arch_pop = archive_survival.do(problem, arch_pop, n_survive=effective_pop)
        pareto_X = arch_pop.get("X")
    else:
        pareto_X = result.X if result.X is not None else np.empty((0, n_genes))

    return pareto_X, result.algorithm.evaluator.n_eval


def _qinsga3_n_eval(effective_pop: int, max_gen: int) -> int:
    """QI-NSGA-III's real evaluation count: run_qinsga3 (Solvers/QINSGA3/
    algorithm.py) calls _eval_batch on the parent population and the
    offspring population every generation (lines ~1297, ~1421), plus one
    final _eval_batch(X_final) after the loop (line ~1498) -- confirmed by
    grepping every _eval_batch( call site, not assumed."""
    return effective_pop * (2 * max_gen + 1)


def _config_F(pareto_X, sets_, params_, repair: bool) -> np.ndarray:
    """Decode (and optionally 2-opt-repair) a chromosome array via the
    shared _evaluate_pareto, returning its (f1,f2,f3,f4) objective matrix."""
    if pareto_X is None or len(pareto_X) == 0:
        return np.empty((0, N_OBJ))
    result = _evaluate_pareto(pareto_X, sets_, params_, {}, repair=repair)
    return np.array([[s["objectives"]["f1"], s["objectives"]["f2"],
                       s["objectives"]["f3"], s["objectives"]["f4"]]
                      for s in result["solutions"]])


def _stats(vals) -> dict:
    a = np.array(vals, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std())}


def run_comparison(
    instance: str, seeds: list[int], max_gen: int, pop_size: int,
    qinsga3_gen: int | None = None, noise_scale: float = 0.0,
    eliminate_duplicates: bool = False, use_archive: bool = False,
) -> None:
    """qinsga3_gen (default None = same as max_gen): lets QI-NSGA-III run at a
    DIFFERENT generation count than NSGA-III, specifically to match real
    evaluation budgets rather than max_gen (see the module docstring's "NOTE
    on evaluation budget" -- effective_pop*(2*qinsga3_gen+1) vs.
    effective_pop*max_gen). qinsga3_gen=max_gen//2 makes the two counts
    approximately equal (exactly equal up to the "+1" final-pass term).

    noise_scale (default 0.0, run_qinsga3's own current production default --
    was 0.02 until a fairness audit found it added ~10-12% of each
    normalised objective's range as noise to F every generation, breaking
    elitism, with no measured benefit at the project's own 20/30-seed
    gold-standard protocol; see Solvers/QINSGA3/README.md's parameter
    table). Pass --noise-scale 0.02 to reproduce the old default for
    comparison.

    eliminate_duplicates (default False -- NOT run_qinsga3's production
    default, ablation-only): rejects X-space duplicate offspring and
    regenerates them, matching pymoo's NSGA-III eliminate_duplicates=True.
    Tested at the project's own 30-seed protocol: no significant change on
    HV/GD/IGD/Spacing -- see Solvers/QINSGA3/algorithm.py's run_qinsga3
    docstring. Pass --eliminate-duplicates 1 to enable it for comparison.

    use_archive (default False -- NOT NSGA-III's production default,
    ablation-only until validated): reproduces QI-NSGA-III's own external
    non-dominated archive (spans the whole run, capped at 500, trimmed to
    effective_pop at the end) for NSGA-III too, via pymoo's own
    MultiObjectiveArchive -- see Solvers/NSGA3/main.py's own use_archive
    parameter docstring for the fairness-audit finding this tests: NSGA-III
    normally reports only its final generation's population, so a good
    solution found early and later lost to niching is gone for good, an
    asymmetry QI-NSGA-III's archive doesn't have. Pass --use-archive 1 to
    test whether this symmetry changes the verdict."""
    if qinsga3_gen is None:
        qinsga3_gen = max_gen

    data_path = os.path.join(PROJECT_DIR, "data", f"instance_{instance}_clients.json")
    sets_, params_ = load_instance(data_path)
    ref_dirs = get_reference_directions("das-dennis", N_OBJ, n_partitions=N_PARTITIONS)
    effective_pop = max(pop_size, len(ref_dirs))

    print("=" * 92)
    print("  2-OPT FAIRNESS CAMPAIGN -- NSGA-III vs QI-NSGA-III, 4 configurations")
    print("=" * 92)
    print(f"  Instance : {instance} clients | pop={effective_pop} | "
          f"gen(NSGA-III)={max_gen} | gen(QI-NSGA-III)={qinsga3_gen} | "
          f"noise_scale(QI)={noise_scale} | eliminate_duplicates(QI)={eliminate_duplicates} | "
          f"use_archive(NSGA-III)={use_archive}")
    if qinsga3_gen != max_gen:
        print("  NOTE: asymmetric generation counts -- budget-matched run, "
              "not a max_gen-matched run. See module docstring.")
    if noise_scale != 0.0:
        print("  NOTE: noise_scale != production default (0.0) -- measurement-noise "
              "ablation run, not a production-parameter run. See module docstring.")
    if not eliminate_duplicates:
        print("  NOTE: eliminate_duplicates disabled -- not the production default. "
              "See module docstring.")
    if use_archive:
        print("  NOTE: use_archive enabled for NSGA-III -- not its production default. "
              "See module docstring.")
    print(f"  Seeds    : {seeds}")
    print("=" * 92)

    F_by_config:       dict[str, list[np.ndarray]] = {c: [] for c in _CONFIGS}
    elapsed_by_config: dict[str, list[float]]      = {c: [] for c in _CONFIGS}
    n_eval_nsga3:   list[int] = []
    n_eval_qinsga3: list[int] = []

    for seed in seeds:
        print(f"\n>>> seed={seed}")

        t0 = time.time()
        X_nsga3, nsga3_n_eval = _run_nsga3_once(sets_, params_, ref_dirs, effective_pop, max_gen, seed,
                                                  use_archive=use_archive)
        t_nsga3 = time.time() - t0
        n_eval_nsga3.append(nsga3_n_eval)
        print(f"  NSGA-III    search done in {t_nsga3:.1f}s | front={len(X_nsga3)} | n_eval={nsga3_n_eval}")

        t0 = time.time()
        X_qinsga3, _, _ = run_qinsga3(
            sets_=sets_, params_=params_, ref_dirs=ref_dirs,
            pop_size=effective_pop, max_gen=qinsga3_gen, seed=seed,
            repair_final_front=False, noise_scale=noise_scale,
            eliminate_duplicates=eliminate_duplicates,
        )
        t_qinsga3 = time.time() - t0
        n_qi = len(X_qinsga3) if X_qinsga3 is not None else 0
        qinsga3_n_eval = _qinsga3_n_eval(effective_pop, qinsga3_gen)
        n_eval_qinsga3.append(qinsga3_n_eval)
        print(f"  QI-NSGA-III search done in {t_qinsga3:.1f}s | front={n_qi} | n_eval={qinsga3_n_eval}")

        for label, X, base_elapsed, repair in (
            ("NSGA-III sans 2-opt",    X_nsga3,   t_nsga3,   False),
            ("NSGA-III avec 2-opt",    X_nsga3,   t_nsga3,   True),
            ("QI-NSGA-III sans 2-opt", X_qinsga3, t_qinsga3, False),
            ("QI-NSGA-III avec 2-opt", X_qinsga3, t_qinsga3, True),
        ):
            t0 = time.time()
            F = _config_F(X, sets_, params_, repair=repair)
            t_repair = time.time() - t0
            F_by_config[label].append(F)
            elapsed_by_config[label].append(base_elapsed + t_repair)
            print(f"    {label:<24} front={len(F)}  decode+repair={t_repair:.2f}s")

    all_F = np.vstack([F for runs in F_by_config.values() for F in runs if len(F) > 0])
    g_ideal = all_F.min(axis=0)
    g_nadir = all_F.max(axis=0)
    print(f"\nIdeal global partage : {g_ideal}")
    print(f"Nadir global partage  : {g_nadir}")

    values: dict[str, dict[str, list]] = {c: {ind: [] for ind in _INDICATORS} for c in _CONFIGS}
    for label, runs in F_by_config.items():
        for F in runs:
            if len(F) == 0:
                continue
            q = compute_pareto_metrics(F, g_ideal, g_nadir)
            for ind in _INDICATORS:
                values[label][ind].append(q[ind])

    print(f"\n{'-'*92}")
    print("  RESULTATS (ideal/nadir global partage entre les 4 configurations)")
    print(f"{'-'*92}")
    for ind in _INDICATORS:
        arrow = "^" if _HIGHER_IS_BETTER[ind] else "v"
        print(f"\n  {ind} ({arrow})")
        for label in _CONFIGS:
            if not values[label][ind]:
                print(f"    {label:<24} (aucune donnee)")
                continue
            s = _stats(values[label][ind])
            print(f"    {label:<24} mean={s['mean']:.6f}  std={s['std']:.6f}")

    print(f"\n{'-'*92}")
    print("  Effet 2-opt (Wilcoxon apparie -- meme front initial, meme seed)")
    print(f"{'-'*92}")
    for algo, without_lbl, with_lbl in (
        ("NSGA-III",    "NSGA-III sans 2-opt",    "NSGA-III avec 2-opt"),
        ("QI-NSGA-III", "QI-NSGA-III sans 2-opt", "QI-NSGA-III avec 2-opt"),
    ):
        print(f"\n  {algo}")
        for ind in _INDICATORS:
            a, b = values[without_lbl][ind], values[with_lbl][ind]
            if len(a) < 2 or len(b) < 2 or len(a) != len(b):
                print(f"    {ind:<8} : pas assez de runs valides pour un test apparie")
                continue
            try:
                stat, p = wilcoxon(a, b)
            except ValueError:
                print(f"    {ind:<8} : difference nulle sur tous les seeds -- test non applicable")
                continue
            sig = "significatif (p<0.05)" if p < 0.05 else "non significatif"
            print(f"    {ind:<8} W={stat:.1f}  p={p:.6f}  -> {sig}")

    print(f"\n{'-'*92}")
    print("  Budget d'evaluations reel (max_gen egal ne veut PAS dire n_eval egal)")
    print(f"{'-'*92}")
    ns = _stats(n_eval_nsga3)
    qs = _stats(n_eval_qinsga3)
    ratio = qs["mean"] / ns["mean"] if ns["mean"] else float("nan")
    print(f"    NSGA-III     n_eval  mean={ns['mean']:.0f}  std={ns['std']:.0f}")
    print(f"    QI-NSGA-III  n_eval  mean={qs['mean']:.0f}  std={qs['std']:.0f}")
    print(f"    ratio QI-NSGA-III / NSGA-III : {ratio:.2f}x")
    if abs(ratio - 1.0) < 0.05:
        print("    -> budgets approximativement apparies (dans 5%) -- la section")
        print("       \"Effet moteur quantique\" ci-dessous compare les deux moteurs")
        print("       a evaluations comparables, pas seulement a max_gen egal.")
    else:
        print("    -> la section \"Effet moteur quantique\" ci-dessous compare deux")
        print("       moteurs a budget d'evaluations DIFFERENT (voir note dans le")
        print("       docstring du module) -- lire les p-values de cette section")
        print("       avec ce ratio en tete, surtout si QI-NSGA-III ressort meilleur.")

    print(f"\n{'-'*92}")
    print("  Effet moteur quantique (Mann-Whitney U -- fronts independants)")
    print(f"{'-'*92}")
    for title, lbl_a, lbl_b in (
        ("Sans 2-opt des deux cotes (ecart brut)",      "NSGA-III sans 2-opt", "QI-NSGA-III sans 2-opt"),
        ("Avec 2-opt des deux cotes (ecart equitable)", "NSGA-III avec 2-opt", "QI-NSGA-III avec 2-opt"),
    ):
        print(f"\n  {title}")
        for ind in _INDICATORS:
            a, b = values[lbl_a][ind], values[lbl_b][ind]
            if len(a) < 2 or len(b) < 2:
                print(f"    {ind:<8} : pas assez de runs valides pour un test")
                continue
            u, p = mannwhitneyu(a, b, alternative="two-sided")
            sig = "significatif (p<0.05)" if p < 0.05 else "non significatif"
            print(f"    {ind:<8} U={u:.1f}  p={p:.6f}  -> {sig}")

    print(f"\n{'-'*92}")
    print("  Temps moyen par configuration (recherche + decodage/repair)")
    print(f"{'-'*92}")
    for label in _CONFIGS:
        s = _stats(elapsed_by_config[label])
        print(f"    {label:<24} mean={s['mean']:.1f}s")

    print(f"\n{'='*92}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Campagne d'equite 2-opt (4 configurations) -- NSGA-III vs QI-NSGA-III"
    )
    parser.add_argument("--instance", default="100",
                        choices=["3", "5", "15", "25", "30", "40", "100"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--gen", type=int, default=300)
    parser.add_argument("--pop", type=int, default=POP_SIZE)
    parser.add_argument("--qinsga3-gen", type=int, default=None,
                        help="Generation count for QI-NSGA-III only, if different from "
                             "--gen (NSGA-III's count). Use max_gen//2 to roughly match "
                             "real evaluation budgets instead of matching max_gen -- see "
                             "the module docstring's evaluation-budget note. Default: "
                             "same as --gen (current behaviour, budgets differ ~2x).")
    parser.add_argument("--noise-scale", type=float, default=0.0,
                        help="QI-NSGA-III's measurement-noise magnitude (run_qinsga3's "
                             "own current production default: 0.0). Pass 0.02 to "
                             "reproduce the old default -- see the module docstring's "
                             "noise_scale note.")
    parser.add_argument("--eliminate-duplicates", type=int, choices=[0, 1], default=0,
                        help="QI-NSGA-III's X-space duplicate elimination, matching "
                             "pymoo's NSGA-III eliminate_duplicates=True (NOT "
                             "run_qinsga3's production default -- ablation-only). "
                             "Pass 1 to enable it -- see the module docstring's "
                             "eliminate_duplicates note.")
    parser.add_argument("--use-archive", type=int, choices=[0, 1], default=0,
                        help="Give NSGA-III an external non-dominated archive spanning "
                             "the whole run, matching QI-NSGA-III's own archive "
                             "(NOT NSGA-III's production default -- disabled by "
                             "default). Pass 1 to test whether this symmetry changes "
                             "the verdict -- see the module docstring's use_archive "
                             "note.")
    args = parser.parse_args()
    run_comparison(args.instance, args.seeds, args.gen, args.pop,
                    args.qinsga3_gen, args.noise_scale,
                    bool(args.eliminate_duplicates), bool(args.use_archive))
