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

NOTE on evaluation budget: matching max_gen and pop_size does NOT
automatically match the number of objective-function evaluations actually
spent. NSGA-III evaluates exactly effective_pop individuals per generation
(pop_size infills each iteration, pymoo's own GeneticAlgorithm default) ->
effective_pop * max_gen total. QI-NSGA-III's run_qinsga3 evaluates the
offspring population every generation, but the PARENT population only
when its cache is invalid -- with noise_scale=0.0 (this project's
production default), qpop.measure() is a deterministic function of theta,
so a survivor's F from the previous generation's own evaluation is still
exactly correct and is reused instead of recomputed (see Solvers/QINSGA3/
algorithm.py's "Performance" docstring section) -- invalidated only when
theta changes after being cached, i.e. generation 0 and every generation
right after an archive migration fires. This makes QI-NSGA-III's real
evaluation count data-dependent (not a fixed multiple of max_gen) and, at
equal max_gen, typically close to but still somewhat above NSGA-III's --
measured directly on this project's own 100-client instance at max_gen=300:
66400 vs NSGA-III's 60000 (~1.11x), not the ~2x a pre-cache-fix formula
would have predicted. This script reports both algorithms' REAL evaluation
counts (n_eval) alongside every result, counted directly (not estimated by
a formula -- see _run_qinsga3_counted), so the "Effet moteur quantique"
section can be read with this in mind. The "Effet 2-opt" section is NOT
affected by this (it compares each algorithm only to itself, at whatever
budget it actually used). Pass --qinsga3-gen to run QI-NSGA-III at a
different generation count than NSGA-III specifically to match real
evaluation budgets instead of max_gen -- qinsga3_gen=271 was found by
direct instrumentation to give exactly 60000 real evaluations on this
project's own 100-client instance at pop=200, an exact match to NSGA-III's
effective_pop*300, not merely an approximation.

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
from scipy.stats import levene, mannwhitneyu, rankdata, wilcoxon

from models.parametres            import load_instance
from Solvers.NSGA3.metrics        import compute_pareto_metrics, build_empirical_reference_front
from Solvers.NSGA3.problem        import IRPProblem
from Solvers.NSGA3.report_builder import _evaluate_pareto
from Solvers.QINSGA3.algorithm    import run_qinsga3, _archive_update_epsilon, _crowding_trim

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
                     use_archive: bool = False,
                     use_epsilon_archive: bool = False, epsilon_divisions: int = 20):
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
    by default).

    use_epsilon_archive (default False, mutually exclusive with use_archive):
    gives NSGA-III an external archive maintained by the LITERAL SAME
    epsilon-dominance function QI-NSGA-III's own use_epsilon_archive uses
    (Solvers/QINSGA3/algorithm.py::_archive_update_epsilon), via a per-
    generation pymoo callback reading algorithm.pop -- not pymoo's own
    MultiObjectiveArchive (plain dominance, what use_archive above gives).
    This is the true apples-to-apples equity test: both algorithms bounded
    by the SAME archiving rule, rather than one plain / one epsilon or one
    archived / one not. theta has no meaning for NSGA-III (it has no
    quantum encoding) -- X is passed in its place purely so
    _archive_update_epsilon's signature is satisfied; the returned
    arch_theta list is discarded. epsilon is computed once, the first
    generation algorithm.survival.norm's ideal/nadir are populated (mirrors
    run_qinsga3's own use_epsilon_archive timing exactly), then held fixed.
    """
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

    eps_arch_X: list = []
    eps_arch_F: list = []
    eps_arch_theta: list = []
    _eps_state = {"ideal": None, "epsilon": None}

    def _epsilon_archive_callback(algorithm):
        pop = algorithm.pop
        if pop is None or len(pop) == 0:
            return
        X = pop.get("X")
        F = pop.get("F")
        G = pop.get("G")
        if G is None or G.size == 0:
            G = np.zeros((len(pop), 0))
        if _eps_state["epsilon"] is None:
            norm = getattr(algorithm.survival, "norm", None)
            if norm is not None and norm.nadir_point is not None:
                ideal = np.asarray(norm.ideal_point, dtype=float)
                nadir = np.asarray(norm.nadir_point, dtype=float)
                _eps_state["ideal"]   = ideal
                _eps_state["epsilon"] = np.maximum(nadir - ideal, 1e-9) / epsilon_divisions
            else:
                return
        _archive_update_epsilon(
            X, F, G, X, eps_arch_X, eps_arch_F, eps_arch_theta,
            _eps_state["ideal"], _eps_state["epsilon"], max_size=500,
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
        callback  = _epsilon_archive_callback if use_epsilon_archive else None,
    )
    result = minimize(
        problem, algorithm, get_termination("n_gen", max_gen), seed=seed, verbose=False,
    )

    if use_epsilon_archive:
        if eps_arch_X:
            arch_X_arr = np.array(eps_arch_X)
            if len(arch_X_arr) > effective_pop:
                arch_X_arr, _, _ = _crowding_trim(
                    arch_X_arr, np.array(eps_arch_F), np.array(eps_arch_theta), effective_pop,
                )
            pareto_X = arch_X_arr
        else:
            pareto_X = result.X if result.X is not None else np.empty((0, n_genes))
    elif use_archive and result.archive is not None and len(result.archive) > 0:
        arch_pop = result.archive
        if len(arch_pop) > effective_pop:
            arch_pop = archive_survival.do(problem, arch_pop, n_survive=effective_pop)
        pareto_X = arch_pop.get("X")
    else:
        pareto_X = result.X if result.X is not None else np.empty((0, n_genes))

    return pareto_X, result.algorithm.evaluator.n_eval


def _run_qinsga3_counted(**kwargs):
    """Call run_qinsga3 while counting its REAL number of evaluated
    individuals, the same way _run_nsga3_once reads pymoo's own evaluator
    counter for NSGA-III -- no closed-form formula, an actual count.

    Needed because run_qinsga3's parent-population evaluation is no longer
    unconditional every generation: it is cached (skipped) whenever
    noise_scale == 0.0 and theta hasn't changed since the previous
    generation's own evaluation, invalidated only by archive migration or
    niche-recentring reset (see Solvers/QINSGA3/algorithm.py's "Performance"
    docstring section). The real count is therefore data-dependent (how
    often migration actually fires, which depends on how quickly the
    archive fills past 4 entries) -- verified by direct instrumentation on
    the project's own 100-client instance to be as low as ~1.0x NSGA-III's
    budget at a calibrated generation count (qinsga3_gen=271 -> exactly
    60000, matching NSGA-III's effective_pop*300 exactly, not the ~2x a
    closed-form pre-cache-fix formula would have predicted), NOT a fixed
    ~2x or ~1.1x ratio that a formula could capture in general.

    Counts every individual passed to ProcessPoolExecutor.map -- the exact
    call site _eval_batch uses (Solvers/QINSGA3/algorithm.py) -- across the
    whole run_qinsga3 call, in-process (the call itself, not its worker-
    process execution, so a plain module-level monkeypatch sees it).
    """
    from concurrent.futures import ProcessPoolExecutor
    count = {"n": 0}
    orig_map = ProcessPoolExecutor.map

    def _counted_map(self, fn, iterable, *a, **kw):
        items = list(iterable)
        count["n"] += len(items)
        return orig_map(self, fn, items, *a, **kw)

    ProcessPoolExecutor.map = _counted_map
    try:
        result = run_qinsga3(**kwargs)
    finally:
        ProcessPoolExecutor.map = orig_map
    return result, count["n"]


def _config_F(pareto_X, sets_, params_, repair: bool, use_or_opt: bool = False,
              use_single_relocation: bool = False, use_two_opt: bool = True,
              use_inter_route_relocate: bool = False, use_route_swap: bool = False,
              use_delivery_shift: bool = False) -> np.ndarray:
    """Decode (and optionally 2-opt-repair, optionally Or-opt-widened) a
    chromosome array via the shared _evaluate_pareto, returning its
    (f1,f2,f3,f4) objective matrix. use_or_opt/use_single_relocation/
    use_two_opt/use_inter_route_relocate/use_route_swap/use_delivery_shift
    only have an effect when repair=True (see _evaluate_pareto's own
    docstrings) -- applied identically to both algorithms' "avec 2-opt"
    configs by the caller, for a fair vs-not comparison matching how the
    2-opt-vs-not comparison itself works."""
    if pareto_X is None or len(pareto_X) == 0:
        return np.empty((0, N_OBJ))
    result = _evaluate_pareto(pareto_X, sets_, params_, {}, repair=repair, use_or_opt=use_or_opt,
                               use_single_relocation=use_single_relocation, use_two_opt=use_two_opt,
                               use_inter_route_relocate=use_inter_route_relocate,
                               use_route_swap=use_route_swap,
                               use_delivery_shift=use_delivery_shift)
    return np.array([[s["objectives"]["f1"], s["objectives"]["f2"],
                       s["objectives"]["f3"], s["objectives"]["f4"]]
                      for s in result["solutions"]])


def _stats(vals) -> dict:
    a = np.array(vals, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std())}


def _wilcoxon_rank_biserial(a, b) -> float:
    """Matched-pairs rank-biserial correlation for a paired Wilcoxon test --
    r = (W+ - W-) / (W+ + W-), where W+/W- are the summed ranks of the
    positive/negative paired differences (ties at exactly 0 excluded, same
    convention scipy.stats.wilcoxon uses by default). r in [-1, 1]: positive
    means a tends to exceed b (rank-weighted), 0 means no consistent
    direction, negative means b tends to exceed a. This is the effect-size
    companion to the Wilcoxon p-value -- p says whether the paired
    difference is distinguishable from chance, r says how one-sided it is.
    """
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    w_pos = ranks[nonzero > 0].sum()
    w_neg = ranks[nonzero < 0].sum()
    total = w_pos + w_neg
    return float((w_pos - w_neg) / total) if total > 0 else 0.0


def _vargha_delaney_a12(u_stat: float, n1: int, n2: int) -> float:
    """Vargha-Delaney A12 effect size from a Mann-Whitney U statistic (U for
    sample 1, scipy's default convention: mannwhitneyu(a, b) returns U_a).

    A12 = U_a / (n1*n2) is the probability that a randomly drawn value from
    group 1 exceeds a randomly drawn value from group 2 (ties count as a
    half-win) -- Vargha & Delaney (2000)'s common-language effect size.
    A12=0.5 means no difference; A12>0.5 means group 1 tends to be larger;
    A12<0.5 means group 1 tends to be smaller. Unlike the raw p-value, this
    is interpretable regardless of sample size (a p<0.05 with A12=0.51 is a
    real but practically tiny effect; the reverse can also happen at small n).
    """
    return float(u_stat) / (n1 * n2)


def _bootstrap_median_diff_ci(a, b, n_boot: int = 10000, ci: float = 0.95,
                               seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap CI for the median of the PAIRED differences (a[i]-b[i]) --
    resamples the vector of paired differences with replacement n_boot
    times, takes the median each time, and reports the empirical
    (1-ci)/2 / (1+ci)/2 percentiles. Complements the Wilcoxon p-value with a
    magnitude estimate that doesn't assume any particular distribution
    shape. Returns (median_diff, ci_lo, ci_hi).
    """
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_medians = np.empty(n_boot)
    for i in range(n_boot):
        sample = diffs[rng.integers(0, n, size=n)]
        boot_medians[i] = np.median(sample)
    lo, hi = np.percentile(boot_medians, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(np.median(diffs)), float(lo), float(hi)


def _holm_bonferroni(pvalues: dict, alpha: float = 0.05) -> dict:
    """Holm-Bonferroni step-down correction for multiple comparisons --
    controls the family-wise error rate across the len(pvalues) tests
    (here: HV/GD/IGD/Spacing tested together), unlike treating each
    indicator's p<0.05 as independently conclusive.

    Sorted ascending p_(1)<=...<=p_(m): reject H_(i) iff p_(i) <= alpha/(m-i+1)
    for every j<=i (step-down: stop rejecting at the first failure, every
    remaining larger p-value is then also treated as non-significant
    regardless of its own value, per the standard Holm procedure).

    Returns {name: (p, threshold, significant_after_correction)}.
    """
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    results = {}
    still_rejecting = True
    for i, (name, p) in enumerate(items):
        threshold = alpha / (m - i)
        significant = still_rejecting and p <= threshold
        if not significant:
            still_rejecting = False
        results[name] = (p, threshold, significant)
    return results


def run_comparison(
    instance: str, seeds: list[int], max_gen: int, pop_size: int,
    qinsga3_gen: int | None = None, noise_scale: float = 0.0,
    eliminate_duplicates: bool = False, use_archive: bool = False,
    compensate_dx_dtheta: bool = False, use_or_opt: bool = False,
    use_single_relocation: bool = False, use_two_opt: bool = True,
    use_inter_route_relocate: bool = False, use_route_swap: bool = False,
    use_delivery_shift: bool = False,
    use_epsilon_archive: bool = False, epsilon_divisions: int = 20,
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
    test whether this symmetry changes the verdict.

    compensate_dx_dtheta (default False -- ablation-only): scales
    QI-NSGA-III's quantum rotation step to counteract the non-uniform
    dx/dtheta mapping of its theta->X measurement (vanishes near theta=0/
    pi/2, where individuals actually converge) -- see
    Solvers/QINSGA3/chromosome.py's QuantumPopulation.rotate() docstring
    for the exact mechanism and the hypothesis this tests: that this
    uncompensated non-uniformity, not NSGA-III's SBX (which moves uniformly
    in X-space), may explain part of QI-NSGA-III's residual HV/IGD gap.
    Pass --compensate-dx-dtheta 1 to enable it for comparison.

    use_or_opt (default False -- ablation-only): widens the "avec 2-opt"
    configs' local search, for BOTH algorithms symmetrically, from pure
    2-opt (sequencing only) to also relocating 2-/3-client segments
    elsewhere in the same route -- see Solvers/QINSGA3/repair.py's
    _repair_route_result use_or_opt docstring for the full rationale (the
    MPIRP's decision space -- vehicle assignment, quantities, periods,
    frigo compatibility, stock levels -- is much richer than sequencing
    alone; Or-opt is the first, smallest step toward a richer
    neighbourhood). Applied identically to NSGA-III and QI-NSGA-III's
    "avec 2-opt" configs, matching how 2-opt itself was made fair. Pass
    --use-or-opt 1 to enable it for comparison.

    use_single_relocation (default False -- ablation-only, independent of
    use_or_opt): widens the "avec 2-opt" configs' local search, for BOTH
    algorithms symmetrically, to also relocate a SINGLE client -- the
    third lever from the advisor's Niveau-1 plan (2-opt; Or-opt;
    single-client relocation), tested in isolation against plain 2-opt
    rather than stacked with use_or_opt -- see
    Solvers/QINSGA3/repair.py's _repair_route_result
    use_single_relocation docstring. Pass --use-single-relocation 1 to
    enable it for comparison.

    use_two_opt (default True -- production behaviour unchanged): when
    False, disables the 2-opt scan entirely in the "avec 2-opt" configs,
    isolating use_or_opt/use_single_relocation as the ONLY neighbourhood
    searched -- answers "does Or-opt/relocation help on its own" rather
    than "does adding it to 2-opt help" -- see
    Solvers/QINSGA3/repair.py's _repair_route_result use_two_opt
    docstring. Pass --use-two-opt 0 together with --use-or-opt 1 (or
    --use-single-relocation 1) to test a neighbourhood in isolation.

    use_inter_route_relocate (default False -- ablation-only): widens the
    "avec 2-opt" configs' local search, for BOTH algorithms symmetrically,
    to also relocate a single client to a DIFFERENT truck's route within
    the same period -- the advisor's Niveau-2 "relocate d'un client vers
    un autre vehicule" move, a fundamentally different lever from the
    Niveau-1 moves (touches truck assignment, not just visit order) -- see
    Solvers/QINSGA3/repair.py's _repair_route_result
    use_inter_route_relocate docstring. Pass --use-inter-route-relocate 1
    to enable it for comparison.

    use_route_swap (default False -- ablation-only): widens the "avec
    2-opt" configs' local search, for BOTH algorithms symmetrically, to
    also exchange two clients between different routes within the same
    period -- the advisor's Niveau-2 "swap entre deux tournees" move --
    see Solvers/QINSGA3/repair.py's _repair_route_result use_route_swap
    docstring. Pass --use-route-swap 1 to enable it for comparison.

    use_delivery_shift (default False -- ablation-only): widens the "avec
    2-opt" configs' local search, for BOTH algorithms symmetrically, to
    also shift a client's delivered quantity between two ADJACENT periods
    it is already served in -- the advisor's Niveau-3 "deplacement partiel
    d'une livraison vers une periode voisine" move, an inventory-timing
    lever distinct from every Niveau-1/2 route-topology move above -- see
    Solvers/QINSGA3/repair.py's _repair_route_result use_delivery_shift
    docstring. Pass --use-delivery-shift 1 to enable it for comparison.

    use_epsilon_archive (default False -- ablation-only, search-time only,
    unlike every ablation above which acts at repair time on the final
    front): gives BOTH algorithms an external archive maintained by
    epsilon-dominance archiving (Laumanns, Thiele, Deb & Zitzler 2002)
    instead of their own respective defaults (QI-NSGA-III's own plain-
    dominance archive; NSGA-III's plain final-generation-only reporting) --
    see Solvers/QINSGA3/algorithm.py's run_qinsga3 use_epsilon_archive
    docstring and _run_nsga3_once's own use_epsilon_archive docstring
    (literally the same _archive_update_epsilon function, applied via a
    per-generation callback for NSGA-III). This is the true equity test:
    both algorithms bounded by the SAME archiving rule -- added after
    finding that giving NSGA-III a PLAIN archive (--use-archive) while
    QI-NSGA-III uses epsilon-dominance does not equalise front sizes, it
    just flips which side balloons (NSGA-III's own plain archive explodes
    toward pop_size too -- see sensitivity/2opt_fairness_100clients_3seed_
    archive_campaign_log.txt). Pass --use-epsilon-archive 1 to enable it for
    both; --epsilon-divisions N (default 20) controls the grid resolution.
    Mutually exclusive with --use-archive in practice (epsilon takes
    priority for NSGA-III if both are somehow set)."""
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
          f"use_archive(NSGA-III)={use_archive} | compensate_dx_dtheta(QI)={compensate_dx_dtheta} | "
          f"use_or_opt(both)={use_or_opt} | use_single_relocation(both)={use_single_relocation} | "
          f"use_two_opt(both)={use_two_opt} | use_inter_route_relocate(both)={use_inter_route_relocate} | "
          f"use_route_swap(both)={use_route_swap} | use_delivery_shift(both)={use_delivery_shift} | "
          f"use_epsilon_archive(both)={use_epsilon_archive}(div={epsilon_divisions})")
    if qinsga3_gen != max_gen:
        print("  NOTE: asymmetric generation counts -- budget-matched run, "
              "not a max_gen-matched run. See module docstring.")
    if noise_scale != 0.0:
        print("  NOTE: noise_scale != production default (0.0) -- measurement-noise "
              "ablation run, not a production-parameter run. See module docstring.")
    if eliminate_duplicates:
        print("  NOTE: eliminate_duplicates enabled -- not the production default. "
              "See module docstring.")
    if use_archive:
        print("  NOTE: use_archive enabled for NSGA-III -- not its production default. "
              "See module docstring.")
    if compensate_dx_dtheta:
        print("  NOTE: compensate_dx_dtheta enabled for QI-NSGA-III -- not the "
              "production default. See module docstring.")
    if use_or_opt:
        print("  NOTE: use_or_opt enabled for BOTH algorithms' 'avec 2-opt' configs -- "
              "not the production default. See module docstring.")
    if use_single_relocation:
        print("  NOTE: use_single_relocation enabled for BOTH algorithms' 'avec 2-opt' "
              "configs -- not the production default. See module docstring.")
    if not use_two_opt:
        print("  NOTE: use_two_opt DISABLED for BOTH algorithms' 'avec 2-opt' configs -- "
              "isolates use_or_opt/use_single_relocation as the only neighbourhood "
              "searched, not the production default. See module docstring.")
    if use_inter_route_relocate:
        print("  NOTE: use_inter_route_relocate enabled for BOTH algorithms' 'avec 2-opt' "
              "configs -- not the production default. See module docstring.")
    if use_route_swap:
        print("  NOTE: use_route_swap enabled for BOTH algorithms' 'avec 2-opt' configs -- "
              "not the production default. See module docstring.")
    if use_delivery_shift:
        print("  NOTE: use_delivery_shift enabled for BOTH algorithms' 'avec 2-opt' configs -- "
              "not the production default. See module docstring.")
    if use_epsilon_archive:
        print(f"  NOTE: use_epsilon_archive enabled for BOTH algorithms (divisions={epsilon_divisions}) -- "
              "not the production default for either. See module docstring.")
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
                                                  use_archive=use_archive,
                                                  use_epsilon_archive=use_epsilon_archive,
                                                  epsilon_divisions=epsilon_divisions)
        t_nsga3 = time.time() - t0
        n_eval_nsga3.append(nsga3_n_eval)
        print(f"  NSGA-III    search done in {t_nsga3:.1f}s | front={len(X_nsga3)} | n_eval={nsga3_n_eval}")

        t0 = time.time()
        (X_qinsga3, _, _), qinsga3_n_eval = _run_qinsga3_counted(
            sets_=sets_, params_=params_, ref_dirs=ref_dirs,
            pop_size=effective_pop, max_gen=qinsga3_gen, seed=seed,
            repair_final_front=False, noise_scale=noise_scale,
            eliminate_duplicates=eliminate_duplicates,
            compensate_dx_dtheta=compensate_dx_dtheta,
            use_epsilon_archive=use_epsilon_archive,
            epsilon_divisions=epsilon_divisions,
        )
        t_qinsga3 = time.time() - t0
        n_qi = len(X_qinsga3) if X_qinsga3 is not None else 0
        n_eval_qinsga3.append(qinsga3_n_eval)
        print(f"  QI-NSGA-III search done in {t_qinsga3:.1f}s | front={n_qi} | n_eval={qinsga3_n_eval}")

        for label, X, base_elapsed, repair in (
            ("NSGA-III sans 2-opt",    X_nsga3,   t_nsga3,   False),
            ("NSGA-III avec 2-opt",    X_nsga3,   t_nsga3,   True),
            ("QI-NSGA-III sans 2-opt", X_qinsga3, t_qinsga3, False),
            ("QI-NSGA-III avec 2-opt", X_qinsga3, t_qinsga3, True),
        ):
            t0 = time.time()
            F = _config_F(X, sets_, params_, repair=repair, use_or_opt=use_or_opt,
                          use_single_relocation=use_single_relocation, use_two_opt=use_two_opt,
                          use_inter_route_relocate=use_inter_route_relocate,
                          use_route_swap=use_route_swap,
                          use_delivery_shift=use_delivery_shift)
            t_repair = time.time() - t0
            F_by_config[label].append(F)
            elapsed_by_config[label].append(base_elapsed + t_repair)
            print(f"    {label:<24} front={len(F)}  decode+repair={t_repair:.2f}s")

    all_F = np.vstack([F for runs in F_by_config.values() for F in runs if len(F) > 0])
    g_ideal = all_F.min(axis=0)
    g_nadir = all_F.max(axis=0)
    print(f"\nIdeal global partage : {g_ideal}")
    print(f"Nadir global partage  : {g_nadir}")

    # PF_ref = ND(union of every solution, every config, every seed) -- an
    # empirical approximation of the true Pareto front, used for GD/IGD
    # instead of a Das-Dennis reference-direction grid (which is geometry,
    # not necessarily feasible MPIRP solutions). See
    # Solvers/NSGA3/metrics.py's build_empirical_reference_front docstring.
    # Printed here for display only (overall pool size) -- the metrics below
    # use a LEAVE-ONE-RUN-OUT variant, never this exact shared front, so a
    # run is never scored against a reference that includes its own
    # solutions (which would let it trivially score a zero self-distance).
    pf_ref = build_empirical_reference_front(all_F)
    print(f"Front de reference empirique (PF_ref, tous runs confondus) : {len(pf_ref)} solutions "
          f"non dominees (sur {len(all_F)} au total, {len(np.unique(all_F, axis=0))} apres "
          f"deduplication) -- affichage seul ; chaque run est evalue contre PF_ref^(-r), "
          f"reconstruit sans ses propres solutions (leave-one-run-out).")

    # Flatten to (label, algo, run_index, F) so each run's own contribution
    # to all_F can be excluded when building ITS reference front. algo is
    # label with the " sans 2-opt"/" avec 2-opt" suffix stripped -- "NSGA-III
    # sans 2-opt" and "NSGA-III avec 2-opt" for the SAME seed are Baldwinian
    # siblings decoded from the exact same chromosome population (only the
    # post-decode 2-opt repair differs, see this module's own top-of-file
    # docstring), so excluding only the exact (label, run_idx) pair leaves a
    # near-identical, usually-dominating twin of "this run" in its own
    # reference front -- self-leakage in substance even though the array
    # objects differ. Excluding by (algo, run_idx) instead removes every
    # repair-variant sibling of the same search together, while still
    # keeping the OTHER algorithm's same-seed runs in the pool (those come
    # from a genuinely independent search, not a repair variant of this one).
    _flat_runs = [
        (label, label.replace(" sans 2-opt", "").replace(" avec 2-opt", ""), i, F)
        for label, runs in F_by_config.items()
        for i, F in enumerate(runs)
        if len(F) > 0
    ]

    values: dict[str, dict[str, list]] = {c: {ind: [] for ind in _INDICATORS} for c in _CONFIGS}
    for label, algo, run_idx, F in _flat_runs:
        # PF_ref^(-r) = ND(union of every OTHER search's solutions) --
        # excludes every repair-variant of THIS run's own search, so none of
        # its own (or its 2-opt-sibling's) solutions can define the
        # reference points it is then measured against.
        other_F = [
            F2 for label2, algo2, i2, F2 in _flat_runs
            if not (algo2 == algo and i2 == run_idx)
        ]
        pf_ref_minus_r = build_empirical_reference_front(np.vstack(other_F))
        q = compute_pareto_metrics(F, g_ideal, g_nadir, reference_front=pf_ref_minus_r)
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
    print("  Effet moteur quantique -- Wilcoxon signed-rank (PRIMAIRE, seeds appariees)")
    print(f"{'-'*92}")
    print("  Les seeds sont communes aux deux algorithmes (seed r pour NSGA-III et seed r pour")
    print("  QI-NSGA-III) -- les donnees sont structurellement appariees, donc Wilcoxon exploite")
    print("  mieux le protocole que Mann-Whitney (qui reste ci-dessous comme verification de")
    print("  robustesse). Chaque indicateur est aussi accompagne d'une taille d'effet (le p ne dit")
    print("  QUE si l'ecart est distinguable du hasard, pas s'il est important en pratique) :")
    print("  correlation biserielle de rang r (Wilcoxon), IC bootstrap 95% de la mediane des")
    print("  differences appariees. Une correction de Holm (controle de l'erreur familiale sur les")
    print("  4 indicateurs testes ensemble) est appliquee a la fin de chaque bloc.")
    for title, lbl_a, lbl_b in (
        ("Sans 2-opt des deux cotes (ecart brut)",      "NSGA-III sans 2-opt", "QI-NSGA-III sans 2-opt"),
        ("Avec 2-opt des deux cotes (ecart equitable)", "NSGA-III avec 2-opt", "QI-NSGA-III avec 2-opt"),
    ):
        print(f"\n  {title}")
        wilcoxon_p = {}
        for ind in _INDICATORS:
            a, b = values[lbl_a][ind], values[lbl_b][ind]
            if len(a) < 2 or len(b) < 2 or len(a) != len(b):
                print(f"    {ind:<8} : pas assez de runs valides/apparies pour un test")
                continue
            try:
                stat, p = wilcoxon(a, b)
            except ValueError:
                print(f"    {ind:<8} : difference nulle sur tous les seeds -- test non applicable")
                continue
            wilcoxon_p[ind] = p
            r = _wilcoxon_rank_biserial(a, b)
            med_diff, ci_lo, ci_hi = _bootstrap_median_diff_ci(a, b)
            sig = "significatif (p<0.05)" if p < 0.05 else "non significatif"
            print(f"    {ind:<8} W={stat:.1f}  p={p:.6f}  -> {sig}  |  r_biserial={r:+.3f}  "
                  f"|  mediane(NSGA-QI)={med_diff:+.4f}  IC95%=[{ci_lo:+.4f}, {ci_hi:+.4f}]")

        if len(wilcoxon_p) == len(_INDICATORS):
            print(f"    Correction de Holm (alpha=0.05, {len(wilcoxon_p)} indicateurs testes ensemble) :")
            holm = _holm_bonferroni(wilcoxon_p)
            for ind, (p, threshold, sig) in sorted(holm.items(), key=lambda kv: kv[1][0]):
                sig_str = "significatif apres correction" if sig else "NON significatif apres correction"
                flip = "" if (sig == (wilcoxon_p[ind] < 0.05)) else "  <-- CHANGE DE VERDICT vs p brut"
                print(f"      {ind:<8} p={p:.6f}  seuil_ajuste={threshold:.5f}  -> {sig_str}{flip}")

    print(f"\n{'-'*92}")
    print("  Effet moteur quantique (Mann-Whitney U -- verification de robustesse)")
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
            a12 = _vargha_delaney_a12(u, len(a), len(b))
            sig = "significatif (p<0.05)" if p < 0.05 else "non significatif"
            print(f"    {ind:<8} U={u:.1f}  p={p:.6f}  -> {sig}  |  A12={a12:.3f}")

    print(f"\n{'-'*92}")
    print("  Test de dispersion (Brown-Forsythe -- egalite de variance/stabilite)")
    print(f"{'-'*92}")
    print("  Un ecart-type plus petit NE prouve PAS, a lui seul, une stabilite")
    print("  statistiquement differente -- il faut un test dedie a la dispersion,")
    print("  pas seulement comparer deux std bruts. Brown-Forsythe = test de Levene")
    print("  centre sur la MEDIANE (pas la moyenne), robuste si les distributions")
    print("  ne sont pas normales -- H0 : les deux groupes ont la meme variance.")
    for title, lbl_a, lbl_b in (
        ("Sans 2-opt des deux cotes",      "NSGA-III sans 2-opt", "QI-NSGA-III sans 2-opt"),
        ("Avec 2-opt des deux cotes (comparaison equitable)", "NSGA-III avec 2-opt", "QI-NSGA-III avec 2-opt"),
    ):
        print(f"\n  {title}")
        for ind in _INDICATORS:
            a, b = values[lbl_a][ind], values[lbl_b][ind]
            if len(a) < 2 or len(b) < 2:
                print(f"    {ind:<8} : pas assez de runs valides pour un test")
                continue
            stat, p = levene(a, b, center="median")
            std_a, std_b = float(np.std(a)), float(np.std(b))
            sig = "significatif (p<0.05)" if p < 0.05 else "non significatif"
            print(f"    {ind:<8} W={stat:.4f}  p={p:.6f}  -> {sig}  "
                  f"(std NSGA-III={std_a:.6f}, std QI-NSGA-III={std_b:.6f})")

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
    parser.add_argument("--compensate-dx-dtheta", type=int, choices=[0, 1], default=0,
                        help="Scale QI-NSGA-III's quantum rotation step to counteract "
                             "the non-uniform dx/dtheta mapping of its theta->X "
                             "measurement (NOT the production default). Pass 1 to test "
                             "whether this closes part of the residual HV/IGD gap -- "
                             "see the module docstring's compensate_dx_dtheta note.")
    parser.add_argument("--use-or-opt", type=int, choices=[0, 1], default=0,
                        help="Widen the 'avec 2-opt' configs' local search, for BOTH "
                             "algorithms symmetrically, to also relocate 2-/3-client "
                             "segments within the same route (NOT the production "
                             "default). Pass 1 to test whether this helps either "
                             "algorithm -- see the module docstring's use_or_opt note.")
    parser.add_argument("--use-single-relocation", type=int, choices=[0, 1], default=0,
                        help="Widen the 'avec 2-opt' configs' local search, for BOTH "
                             "algorithms symmetrically, to also relocate a SINGLE "
                             "client within the same route (NOT the production "
                             "default, independent of --use-or-opt). Pass 1 to test "
                             "whether this helps either algorithm -- see the module "
                             "docstring's use_single_relocation note.")
    parser.add_argument("--use-two-opt", type=int, choices=[0, 1], default=1,
                        help="Whether the 'avec 2-opt' configs include the 2-opt scan "
                             "(production default: 1). Pass 0 together with "
                             "--use-or-opt 1 or --use-single-relocation 1 to test that "
                             "neighbourhood in ISOLATION, without 2-opt riding along -- "
                             "see the module docstring's use_two_opt note.")
    parser.add_argument("--use-inter-route-relocate", type=int, choices=[0, 1], default=0,
                        help="Widen the 'avec 2-opt' configs' local search, for BOTH "
                             "algorithms symmetrically, to also relocate a client to a "
                             "DIFFERENT truck's route within the same period (NOT the "
                             "production default) -- the advisor's Niveau-2 move. Pass 1 "
                             "to test whether this helps either algorithm -- see the "
                             "module docstring's use_inter_route_relocate note.")
    parser.add_argument("--use-route-swap", type=int, choices=[0, 1], default=0,
                        help="Widen the 'avec 2-opt' configs' local search, for BOTH "
                             "algorithms symmetrically, to also exchange two clients "
                             "between different routes within the same period (NOT the "
                             "production default) -- the advisor's Niveau-2 swap move. "
                             "Pass 1 to test whether this helps either algorithm -- see "
                             "the module docstring's use_route_swap note.")
    parser.add_argument("--use-delivery-shift", type=int, choices=[0, 1], default=0,
                        help="Widen the 'avec 2-opt' configs' local search, for BOTH "
                             "algorithms symmetrically, to also shift a client's delivered "
                             "quantity between two ADJACENT periods it is already served "
                             "in (NOT the production default) -- the advisor's Niveau-3 "
                             "inventory-timing move. Pass 1 to test whether this helps "
                             "either algorithm -- see the module docstring's "
                             "use_delivery_shift note.")
    parser.add_argument("--use-epsilon-archive", type=int, choices=[0, 1], default=0,
                        help="Give BOTH algorithms an external archive bounded by the "
                             "SAME epsilon-dominance rule (NOT the production default for "
                             "either) -- search-time only, unlike every other ablation "
                             "flag above. Pass 1 to test the true equity config -- see "
                             "the module docstring's use_epsilon_archive note.")
    parser.add_argument("--epsilon-divisions", type=int, default=20,
                        help="Grid resolution for --use-epsilon-archive (default 20 "
                             "boxes per objective's observed range). Only used when "
                             "--use-epsilon-archive 1.")
    args = parser.parse_args()
    run_comparison(args.instance, args.seeds, args.gen, args.pop,
                    args.qinsga3_gen, args.noise_scale,
                    bool(args.eliminate_duplicates), bool(args.use_archive),
                    bool(args.compensate_dx_dtheta), bool(args.use_or_opt),
                    bool(args.use_single_relocation), bool(args.use_two_opt),
                    bool(args.use_inter_route_relocate), bool(args.use_route_swap),
                    bool(args.use_delivery_shift),
                    bool(args.use_epsilon_archive), args.epsilon_divisions)
