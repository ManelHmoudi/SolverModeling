"""
MOEA/D runner for Cui et al. (2025) DTLZ/MaF validation.

Same Das-Dennis partition table as the NSGA-III/QI-NSGA-III benchmark
runners (imported directly from nsga3.runner, single source of truth):
  n_obj=3 -> p=12 -> H=91
  n_obj=4 -> p=7  -> H=120

Unlike NSGA-III's runner, population size here is NOT rounded up to a
multiple of 4 -- pymoo's own MOEAD forces pop_size = len(ref_dirs)
exactly (one individual per decomposed subproblem), so this runner's
n_obj=3 population is 91, one less than NSGA-III's/QI-NSGA-III's 92. This
is a structural property of MOEA/D, not a tuning choice, and mirrors the
same 165-vs-200 population gap already documented on the IRP side (see
docs/superpowers/specs/2026-08-19-moead-integration-design.md).

Uses ConstrainedMOEAD + NormalizedTchebycheff (Solvers/MOEAD/) exactly as
the IRP side does, for a single MOEA/D wiring shared by both -- even
though DTLZ/MaF are themselves unconstrained, ConstrainedMOEAD's
feasibility rule reduces to vanilla MOEAD's own comparison whenever every
individual is feasible (see _constrained_moead.py's own docstring).

SBX/PM parameters (eta=20, pc=1.0, pm=1/n_var) match the NSGA-III/
QI-NSGA-III benchmark runners' own Cui et al. (2025) Table 2 settings.
"""
import numpy as np
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

from Validation.Benchmarking.algorithms.nsga3.runner import _N_OBJ_TO_P
from Validation.Benchmarking.algorithms.seeds import SEEDS as _SEEDS

from Solvers.MOEAD._constrained_moead import ConstrainedMOEAD
from Solvers.MOEAD._normalized_decomposition import NormalizedTchebycheff

# Module-level constants for n_obj=4 (backward compatibility and unit tests,
# same pattern as nsga3.runner's own N_REF_DIRS/POP_SIZE).
_ref_dirs_4 = get_reference_directions("das-dennis", 4, n_partitions=_N_OBJ_TO_P[4])
N_REF_DIRS = len(_ref_dirs_4)  # C(10,7) = 120
POP_SIZE = N_REF_DIRS  # MOEA/D: pop_size == len(ref_dirs), no rounding


def get_run_config(n_obj: int):
    """Return (ref_dirs, pop_size) for the given number of objectives.

    pop_size == len(ref_dirs) always -- MOEA/D cannot round this up
    independently, unlike NSGA-III's own get_run_config.
    """
    if n_obj not in _N_OBJ_TO_P:
        raise ValueError(
            f"n_obj={n_obj} not supported. Supported values: {sorted(_N_OBJ_TO_P)}"
        )
    p = _N_OBJ_TO_P[n_obj]
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=p)
    return ref_dirs, len(ref_dirs)


def run_single(problem, n_gen: int, seed: int) -> np.ndarray:
    """
    Run one MOEA/D optimisation and return the non-dominated objective values.

    Args:
        problem : pymoo Problem instance (n_obj determines ref_dirs and pop_size).
        n_gen   : number of generations.
        seed    : random seed for reproducibility.

    Returns:
        np.ndarray of shape (n_solutions, n_obj).
    """
    np.random.seed(seed)
    ref_dirs, _ = get_run_config(problem.n_obj)

    algorithm = ConstrainedMOEAD(
        ref_dirs=ref_dirs,
        decomposition=NormalizedTchebycheff(),
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=1.0, eta=20),
        mutation=PM(prob=1.0 / problem.n_var, eta=20),
    )

    result = minimize(
        problem,
        algorithm,
        get_termination("n_gen", n_gen),
        seed=seed,
        verbose=False,
    )

    F = result.F

    if F is None or len(F) == 0:
        raise RuntimeError(
            f"MOEA/D returned no solutions for {type(problem).__name__} "
            f"(n_obj={problem.n_obj}) with seed={seed}."
        )
    return np.asarray(F)


def run_experiment(problem_name: str, problem, n_gen: int, n_runs: int = 30) -> list:
    """
    Run MOEA/D n_runs times with distinct seeds (same seeds as the
    NSGA-III/QI-NSGA-III runners).

    Args:
        problem_name : display name (e.g. "DTLZ1"), used for progress printing.
        problem      : pymoo Problem instance.
        n_gen        : number of generations per run.
        n_runs       : number of independent runs (default 30).

    Returns:
        List of np.ndarray, one per run, shape (n_solutions, n_obj).
    """
    if n_runs > len(_SEEDS):
        raise ValueError(f"n_runs={n_runs} exceeds available seeds ({len(_SEEDS)})")

    fronts = []
    for i in range(n_runs):
        seed = _SEEDS[i]
        print(f"  [{problem_name}] Run {i+1:02d}/{n_runs}  seed={seed}", flush=True)
        front = run_single(problem, n_gen, seed)
        fronts.append(front)
        print(f"  [{problem_name}] Run {i+1:02d} done - front size: {len(front)}", flush=True)
    return fronts
