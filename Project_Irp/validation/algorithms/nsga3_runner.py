"""
NSGA-III runner for Cui et al. (2025) DTLZ validation.

Parameters — Cui, Shi, Wang, Ding, Li & Li (2025), "Practice of an improved
many-objective route optimization algorithm in a multimodal transportation
case under uncertain demand", Complex & Intelligent Systems 11:136, Table 2
("Relevant parameter settings for the test problems", DTLZ 1-7 row):
  SBX crossover : eta=30, prob=1.0   (Deb & Jain 2014's classic NSGA-III value)
  PM mutation   : eta=20, prob=1/n_var
  Ref dirs      : Das-Dennis, p depends on n_obj — Cui et al. (2025) Table 2
                  studies exactly M=3 and M=4 (not M=5+), so those are the
                  only two objective counts supported here:
                    n_obj=3 -> p=12 -> H=C(14,12)=91  -> N=92
                    n_obj=4 -> p=7  -> H=C(10,7)=120  -> N=120
                  (p=12/H=91/N=92 for M=3 originate in Deb & Jain (2014),
                  IEEE TEVC 18(4), 577-601 — Cui et al. reuse that value and
                  extend the same Das-Dennis convention to M=4, which Deb &
                  Jain's own experiments never cover.)
  Pop size      : smallest multiple of 4 that is >= H (not necessarily > H —
                  e.g. H=120 for M=4 is already a multiple of 4, so N=120=H).
  Generations   : Tmax / N, with Tmax = 30000 evaluations (Cui et al. 2025's
                  fixed evaluation budget, Table 2), computed in
                  dtlz_problems.py. This is Cui et al.'s termination rule for
                  both M=3 and M=4 — Deb & Jain (2014) instead uses a
                  per-problem generation count that this project no longer
                  follows.
"""
import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

# Das-Dennis partition count per number of objectives (Cui et al. 2025, Table 2).
_N_OBJ_TO_P = {3: 12, 4: 7}

# Module-level constants for n_obj=4 (backward compatibility and unit tests)
_ref_dirs_4 = get_reference_directions("das-dennis", 4, n_partitions=7)
N_REF_DIRS = len(_ref_dirs_4)  # C(10,7) = 120
POP_SIZE = N_REF_DIRS + (4 - N_REF_DIRS % 4) % 4  # 120 (already a multiple of 4)

# 20 distinct deterministic seeds
_SEEDS = [
    42, 137, 271, 491, 613, 733, 857, 977, 1009, 1123,
    1249, 1373, 1499, 1609, 1733, 1871, 1997, 2113, 2237, 2351,
]


def get_run_config(n_obj: int):
    """Return (ref_dirs, pop_size) for the given number of objectives.

    Uses Das-Dennis p from _N_OBJ_TO_P. pop_size is the smallest
    multiple of 4 that is >= len(ref_dirs) (equal when H is itself
    already a multiple of 4, e.g. M=4's H=120).
    """
    if n_obj not in _N_OBJ_TO_P:
        raise ValueError(
            f"n_obj={n_obj} not supported. Supported values: {sorted(_N_OBJ_TO_P)}"
        )
    p = _N_OBJ_TO_P[n_obj]
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=p)
    h = len(ref_dirs)
    pop_size = h + (4 - h % 4) % 4
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
    ref_dirs, pop_size = get_run_config(problem.n_obj)

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
