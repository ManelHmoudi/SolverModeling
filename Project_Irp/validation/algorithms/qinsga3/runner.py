"""QINSGA-III runner for DTLZ/MaF benchmark validation.

Same run_single/run_experiment signatures as
validation/algorithms/nsga3/runner.py, so validation/engine.py's validate()
can call either interchangeably. Reference directions and population size
come from nsga3.runner.get_run_config() — the Cui et al. (2025) protocol,
identical to the classic NSGA-III validation. Only the quantum-operator
parameters below are QINSGA3-specific, copied verbatim from
QINSGA3/main.py's IRP defaults (there is no classic-NSGA-III equivalent to
compare them against).
"""
import numpy as np

from validation.algorithms.nsga3.runner import get_run_config
from validation.algorithms.seeds import SEEDS

from .core import run_qinsga3_generic

ALPHA_MAX = 0.10 * np.pi
ALPHA_MIN = 0.001 * np.pi
P_CROSS = 0.9
ETA_CROSS = 5.0
P_MUT_STRONG = 0.15
MUT_SIGMA = 0.05 * np.pi
MIGRATION_PERIOD = 10
N_MIGRATE = 10
ROTATION_TYPE = "tanh"


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
    p_mut = 2.0 / problem.n_var

    return run_qinsga3_generic(
        problem, ref_dirs, pop_size, max_gen=n_gen,
        alpha_max=ALPHA_MAX, alpha_min=ALPHA_MIN,
        p_mut=p_mut, p_mut_strong=P_MUT_STRONG, mut_sigma=MUT_SIGMA,
        p_cross=P_CROSS, eta_cross=ETA_CROSS,
        migration_period=MIGRATION_PERIOD, n_migrate=N_MIGRATE,
        seed=seed, rotation_type=ROTATION_TYPE,
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
