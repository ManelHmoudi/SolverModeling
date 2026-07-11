"""
NSGA-III runner configured for Deb & Jain (2014) DTLZ validation.

Parameters:
  SBX crossover : eta=30, prob=1.0
  PM mutation   : eta=20, prob=1/n_var
  Ref dirs      : Das-Dennis, p depends on n_obj:
                    n_obj=3 -> p=12 -> H=C(14,12)=91 -> N=92
                    n_obj=4 -> p=6  -> H=C(9,6)=84   -> N=88
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

# Das-Dennis partition count per number of objectives (Deb & Jain 2014, Table I)
_N_OBJ_TO_P = {3: 12, 4: 6}

# Module-level constants for n_obj=4 (backward compatibility and unit tests)
_ref_dirs_4 = get_reference_directions("das-dennis", 4, n_partitions=6)
N_REF_DIRS = len(_ref_dirs_4)  # C(9,6) = 84
_raw = N_REF_DIRS + (4 - N_REF_DIRS % 4) % 4
POP_SIZE = _raw if _raw > N_REF_DIRS else _raw + 4  # 88

# 20 distinct deterministic seeds
_SEEDS = [
    42, 137, 271, 491, 613, 733, 857, 977, 1009, 1123,
    1249, 1373, 1499, 1609, 1733, 1871, 1997, 2113, 2237, 2351,
]


def _get_run_config(n_obj: int):
    """Return (ref_dirs, pop_size) for the given number of objectives.

    Uses Das-Dennis p from _N_OBJ_TO_P. pop_size is the smallest
    multiple of 4 strictly greater than len(ref_dirs).
    """
    if n_obj not in _N_OBJ_TO_P:
        raise ValueError(
            f"n_obj={n_obj} not supported. Supported values: {sorted(_N_OBJ_TO_P)}"
        )
    p = _N_OBJ_TO_P[n_obj]
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=p)
    h = len(ref_dirs)
    raw = h + (4 - h % 4) % 4
    pop_size = raw if raw > h else raw + 4
    return ref_dirs, pop_size


def run_single(problem, n_gen: int, seed: int) -> np.ndarray:
    """
    Run one NSGA-III optimisation and return the non-dominated objective values.

    Args:
        problem : pymoo Problem instance (n_obj determines ref_dirs and pop_size).
        n_gen   : number of generations.
        seed    : random seed for reproducibility.

    Returns:
        np.ndarray of shape (n_solutions, n_obj).
    """
    np.random.seed(seed)
    ref_dirs, pop_size = _get_run_config(problem.n_obj)

    algorithm = NSGA3(
        pop_size=pop_size,
        ref_dirs=ref_dirs,
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
            f"(n_obj={problem.n_obj}) with seed={seed}."
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
        print(f"  [{problem_name}] Run {i+1:02d} done - front size: {len(front)}", flush=True)
    return fronts
