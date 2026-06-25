"""
DTLZ1-4 problem definitions for Deb & Jain (2014) NSGA-III validation.

n_var = n_obj + k - 1  (follows the k-parameter convention):
  DTLZ1: k=5  → n_var = n_obj + 4
  DTLZ2/3/4: k=10 → n_var = n_obj + 9
"""
from pymoo.problems import get_problem as _pymoo_get_problem

PROBLEM_NAMES = ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4"]

_K = {
    "DTLZ1": 5,
    "DTLZ2": 10,
    "DTLZ3": 10,
    "DTLZ4": 10,
}

_N_GEN = {
    "DTLZ1": 400,
    "DTLZ2": 600,
    "DTLZ3": 600,
    "DTLZ4": 600,
}


def get_problem(name: str, n_obj: int = 4):
    """Return (pymoo Problem instance, n_generations) for the given DTLZ name.

    Args:
        name  : one of "DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4" (case-insensitive).
        n_obj : number of objectives (default 4).

    Returns:
        (problem, n_gen) where n_gen is the generation count from Deb & Jain (2014).
    """
    name = name.upper()
    if name not in _K:
        raise ValueError(f"Unknown problem '{name}'. Choose from {PROBLEM_NAMES}")
    n_var = n_obj + _K[name] - 1
    problem = _pymoo_get_problem(name.lower(), n_var=n_var, n_obj=n_obj)
    return problem, _N_GEN[name]
