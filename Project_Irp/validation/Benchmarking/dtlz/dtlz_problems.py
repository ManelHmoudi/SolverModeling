"""
DTLZ1-7 problem definitions for NSGA-III validation.

n_var = n_obj + k - 1  (standard DTLZ k-parameter convention, Deb et al. 2002):
  DTLZ1: k=5
  DTLZ2-6: k=10
  DTLZ7: k=20

Termination — fixed evaluation budget (Cui et al. 2025, Complex & Intelligent
Systems 11:136, Table 2): Tmax = 30000 evaluations, n_gen = Tmax // pop_size.
This gives the same compute budget to every problem at a given M, and covers
DTLZ5-7 (not studied by Deb & Jain 2014's per-problem generation table, which
this project no longer follows). pop_size comes from nsga3.runner.get_run_config
(Das-Dennis reference directions; M=3 and M=4 only, matching Cui et al.'s own
scope), so the actual generation count depends on M: M=3 (N=92) -> 326 gens,
M=4 (N=120) -> 250 gens.

DEGENERATE_PROBLEMS: DTLZ5/6 have a degenerate (curve-shaped) true Pareto
front and DTLZ7 a disconnected one. pymoo only ships a precomputed reference
front for these three at M=3 (problem.pareto_front() raises "Not implemented
yet." otherwise) — so IGD cannot be scored for them at M=4 here. The
validation pipeline (main_validation.py) skips those combinations rather than
crashing after running the optimisation.
"""
from pymoo.problems import get_problem as _pymoo_get_problem

from Validation.Benchmarking.algorithms.nsga3.runner import get_run_config

PROBLEM_NAMES = ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7"]

DEGENERATE_PROBLEMS = {"DTLZ5", "DTLZ6", "DTLZ7"}  # IGD only scorable at M=3, see module docstring

_K = {
    "DTLZ1": 5,
    "DTLZ2": 10,
    "DTLZ3": 10,
    "DTLZ4": 10,
    "DTLZ5": 10,
    "DTLZ6": 10,
    "DTLZ7": 20,
}

TMAX = 30000  # evaluation budget (Cui et al. 2025)


def get_problem(name: str, n_obj: int = 4):
    """Return (pymoo Problem instance, n_generations) for the given DTLZ name.

    Args:
        name  : one of "DTLZ1".."DTLZ7" (case-insensitive).
        n_obj : number of objectives (default 4). Supported: 3, 4.

    Returns:
        (problem, n_gen) where n_gen = TMAX // pop_size for this n_obj.
    """
    name = name.upper()
    if name not in _K:
        raise ValueError(f"Unknown problem '{name}'. Choose from {PROBLEM_NAMES}")
    n_var = n_obj + _K[name] - 1
    problem = _pymoo_get_problem(name.lower(), n_var=n_var, n_obj=n_obj)
    _, pop_size = get_run_config(n_obj)  # raises ValueError for unsupported n_obj
    n_gen = TMAX // pop_size
    return problem, n_gen
