# validation/algorithms/nsga3_runner.py
"""
NSGA-III runner configured for Deb & Jain (2014) DTLZ validation.

Parameters:
  SBX crossover : eta=30, prob=1.0
  PM mutation   : eta=20, prob=1/n_var
  Ref dirs      : Das-Dennis p=6 → H=84 directions for 4 objectives
  Population    : N=88 (smallest multiple of 4 strictly > 84)
  Generations   : 400 (DTLZ1) or 600 (DTLZ2/3/4) — passed by caller
"""
import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

_P_DIVISIONS = 6
_N_OBJ = 4

_ref_dirs = get_reference_directions("das-dennis", _N_OBJ, n_partitions=_P_DIVISIONS)
N_REF_DIRS = len(_ref_dirs)  # C(9,6) = 84

# Smallest multiple of 4 strictly greater than N_REF_DIRS
_raw = N_REF_DIRS + (4 - N_REF_DIRS % 4) % 4
POP_SIZE = _raw if _raw > N_REF_DIRS else _raw + 4

# 20 distinct deterministic seeds
_SEEDS = [
    42, 137, 271, 491, 613, 733, 857, 977, 1009, 1123,
    1249, 1373, 1499, 1609, 1733, 1871, 1997, 2113, 2237, 2351,
]


def run_single(problem, n_gen: int, seed: int) -> np.ndarray:
    """
    Run one NSGA-III optimisation and return the non-dominated objective values.

    Args:
        problem : pymoo Problem instance.
        n_gen   : number of generations.
        seed    : random seed for reproducibility.

    Returns:
        np.ndarray of shape (n_solutions, n_obj) — objective values of final Pareto front.
    """
    np.random.seed(seed)

    algorithm = NSGA3(
        pop_size=POP_SIZE,
        ref_dirs=_ref_dirs,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=1.0, eta=30),
        mutation=PM(prob=1.0 / problem.n_var, eta=20),
    )

    result = minimize(
        problem,
        algorithm,
        get_termination("n_gen", n_gen),
        seed=seed,
        verbose=False,
    )

    if result.F is None or len(result.F) == 0:
        raise RuntimeError(
            f"NSGA-III returned no solutions for {type(problem).__name__} "
            f"with seed={seed}. Check problem definition."
        )
    return result.F


def run_experiment(problem_name: str, problem, n_gen: int, n_runs: int = 20) -> list:
    """
    Run NSGA-III n_runs times with distinct seeds.

    Args:
        problem_name : display name (e.g. "DTLZ1"), used for progress printing.
        problem      : pymoo Problem instance.
        n_gen        : number of generations per run.
        n_runs       : number of independent runs (default 20).

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
        print(f"  [{problem_name}] Run {i+1:02d} done — front size: {len(front)}", flush=True)
    return fronts
