"""QINSGA-III runner for DTLZ/MaF benchmark validation.

Same run_single/run_experiment signatures as
Validation/Benchmarking/algorithms/nsga3/runner.py, so
Validation/Benchmarking/engine.py's validate() can call either
interchangeably. Reference directions and population size come from
nsga3.runner.get_run_config() — the Cui et al. (2025) protocol, identical to
the classic NSGA-III validation.

Mirrors the IRP production algorithm (Solvers/QINSGA3/algorithm.py): the
rotation gate stays in theta-space (the genuinely quantum-inspired
mechanism), but crossover (SBX) and mutation (PM) now happen in X-SPACE with
pymoo's own operators — identical to NSGA-III's — instead of the old
theta-space SBX + two-tier quantum mutation. An elitist survivor-selection
step (merge parent + offspring, keep the best pop_size via pymoo's own
ReferenceDirectionSurvival) replaces the old "mutate the whole population
unconditionally every generation" loop. See Solvers/QINSGA3/algorithm.py's
module docstring for the evidence (A/B tests on the IRP's 100-client
instance) behind this change.

P_CROSS/ETA_CROSS/P_MUT/ETA_MUT are set to the SAME values used by the
classic NSGA-III validation (pc=1.0, eta=20, p_mut=1/D, eta_mut=20) since
crossover/mutation now happen in the same X-space with the same operators —
there is no longer a separate "QINSGA3 default" to diverge from for these.
The remaining parameters below are intrinsic to the quantum encoding
(rotation gate, external-archive migration) and have no classic-NSGA-III
equivalent to align to — they stay at QINSGA3's own IRP-tuned defaults,
copied verbatim from Solvers/QINSGA3/main.py.

NOISE_SCALE is the one exception: QuantumPopulation's measurement noise
(Solvers/QINSGA3/chromosome.py) exists solely to stop nearby theta values
collapsing to identical integer routes after the IRP decoder rounds to
integers. DTLZ/MaF have no such rounding step, so the noise only blurs
convergence on continuous variables — set to 0 here rather than the
IRP-tuned 0.02. A 5-run ablation (pre-dating the elitist-selection/X-space
change) confirmed this substantially improves convergence on the harder
multimodal problems (DTLZ3 mean IGD 181.9 -> 31.7, MaF3 mean IGD 46985 ->
2407, at n_gen from get_problem()), at the cost of a regression on the two
disconnected-front problems (DTLZ7, MaF7) — an ablation over intermediate
noise_scale values (0.01, 0.005) found no favorable trade-off (both trends
are strictly monotone and opposed), so this is a documented limitation
rather than a tunable knob. Not re-validated since the elitist-selection/
X-space change — the mechanism this note describes is unaffected by it
(NOISE_SCALE only affects QuantumPopulation.measure(), unchanged here).

MIGRATION_PERIOD/N_MIGRATE were also re-tuned (10/10 -> 5/20, i.e. inject
archive solutions more often and in greater numbers) after a 5-run ablation
(also pre-dating the elitist-selection/X-space change) showed this improves
every problem tested simultaneously — DTLZ7 mean IGD 2.98 -> 1.85, MaF7 2.98
-> 1.85, DTLZ3 31.7 -> 12.0, DTLZ2 ~unchanged.

ESCAPE_PROB_FACTOR is new post-elitist-selection/X-space: DTLZ4 and MaF5
raise their "position" variables to a bias exponent of 100. Since
QuantumPopulation always starts every gene at theta=pi/4 (x ~= midpoint of
[xl, xu]), 0.5**100 ~= 8e-31 — 2 of 3(4) objectives collapse to 0 for the
ENTIRE initial population before any search dynamics run at all (confirmed
by instrumentation). The X-space PM mutation is a purely local perturbation
and can never climb from x~0.5 to the x~1 region these two problems need —
the old theta-space "strong" mutation (removed when crossover/mutation
moved to X-space) used to provide that escape by accident.

`escape_prob = ESCAPE_PROB_FACTOR / n_var` restores just that one
capability (a per-gene reset to Uniform(0, pi/2) after the X-space variation
step — see core.py), matching the old total per-gene "strong reset" rate
(p_mut=2/D times p_strong=0.15). Applied uniformly to every problem — same
mechanism, same parameters, across the whole suite, exactly like every
other hyperparameter on this page — rather than special-cased to the two
problems that most need it, since a general-purpose algorithm should not be
tuned per test problem. A full 30-run campaign confirms it fixes the two
collapsed problems (DTLZ4 M3 mean IGD 0.945922 -> 0.072017, MaF5 M3
4.888431 -> 0.327065, both now with real seed-to-seed variance), at the
cost of a regression on the two Rastrigin-style deceptive multimodal
problems that share DTLZ3's g-function (DTLZ3 M3 mean 0.106545 -> 0.200398,
worst-case 0.156 -> 1.191; MaF3 M3 mean 0.123875 -> 7.026855, worst-case
0.244 -> 95.165) — the random resets occasionally eject a run into a much
worse local optimum there. Documented as a known trade-off (same treatment
as NOISE_SCALE's DTLZ7/MaF7 regression above) rather than avoided via
per-problem special-casing.
"""
import numpy as np

from Validation.Benchmarking.algorithms.nsga3.runner import get_run_config
from Validation.Benchmarking.algorithms.seeds import SEEDS

from .core import run_qinsga3_generic

ALPHA_MAX = 0.10 * np.pi
ALPHA_MIN = 0.001 * np.pi
P_CROSS = 1.0     # aligned to NSGA-III's pc=1.0 (Cui et al. 2025)
ETA_CROSS = 20.0  # aligned to NSGA-III's eta=20 (Cui et al. 2025); X-space SBX
ETA_MUT = 20.0    # aligned to NSGA-III's eta=20 (Cui et al. 2025); X-space PM
MIGRATION_PERIOD = 5   # was 10 (Solvers/QINSGA3/main.py default) — see module docstring
N_MIGRATE = 20          # was 10 (Solvers/QINSGA3/main.py default) — see module docstring
ROTATION_TYPE = "tanh"
NOISE_SCALE = 0.0  # was 0.02 (Solvers/QINSGA3/main.py's IRP default) — see module docstring
ESCAPE_PROB_FACTOR = 2.0 * 0.15  # -> escape_prob = ESCAPE_PROB_FACTOR / n_var — see module docstring


def run_single(problem, n_gen: int, seed: int) -> np.ndarray:
    """Run one QINSGA-III optimisation and return the non-dominated objective values.

    Args:
        problem : pymoo Problem instance (n_obj determines ref_dirs and pop_size).
        n_gen   : number of generations.
        seed    : random seed for reproducibility.

    Returns:
        np.ndarray of shape (n_solutions, n_obj).
    """
    ref_dirs, pop_size = get_run_config(problem.n_obj)
    p_mut = 1.0 / problem.n_var  # aligned to NSGA-III's p_mut=1/D (Cui et al. 2025)
    escape_prob = ESCAPE_PROB_FACTOR / problem.n_var  # applied uniformly, every problem

    return run_qinsga3_generic(
        problem, ref_dirs, pop_size, max_gen=n_gen,
        alpha_max=ALPHA_MAX, alpha_min=ALPHA_MIN,
        p_cross=P_CROSS, eta_cross=ETA_CROSS,
        p_mut=p_mut, eta_mut=ETA_MUT,
        migration_period=MIGRATION_PERIOD, n_migrate=N_MIGRATE,
        seed=seed, rotation_type=ROTATION_TYPE, noise_scale=NOISE_SCALE,
        escape_prob=escape_prob,
    )


def run_experiment(problem_name: str, problem, n_gen: int, n_runs: int = 30) -> list:
    """
    Run QINSGA-III n_runs times with distinct seeds (same seeds as the NSGA-III runner).

    Args:
        problem_name : display name (e.g. "DTLZ1"), used for progress printing.
        problem      : pymoo Problem instance.
        n_gen        : number of generations per run.
        n_runs       : number of independent runs (default 30).

    Returns:
        List of np.ndarray, one per run, shape (n_solutions, n_obj).
    """
    if n_runs > len(SEEDS):
        raise ValueError(f"n_runs={n_runs} exceeds available seeds ({len(SEEDS)})")

    fronts = []
    for i in range(n_runs):
        seed = SEEDS[i]
        print(f"  [{problem_name}] Run {i+1:02d}/{n_runs}  seed={seed}", flush=True)
        front = run_single(problem, n_gen, seed)
        fronts.append(front)
        print(f"  [{problem_name}] Run {i+1:02d} done - front size: {len(front)}", flush=True)
    return fronts
