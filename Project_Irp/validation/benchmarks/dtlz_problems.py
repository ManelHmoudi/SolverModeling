"""
DTLZ1-4 problem definitions for Deb & Jain (2014) NSGA-III validation.

n_obj = 4 for all problems.
n_var follows k parameter:
  DTLZ1: k=5  → n_var = M + k - 1 = 4 + 5 - 1 = 8
  DTLZ2/3/4: k=10 → n_var = M + k - 1 = 4 + 10 - 1 = 13
"""
from pymoo.problems import get_problem as _pymoo_get_problem

PROBLEM_NAMES = ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4"]

_N_OBJ = 4

_CONFIGS = {
    "DTLZ1": {"n_var": 8,  "n_gen": 400},
    "DTLZ2": {"n_var": 13, "n_gen": 600},
    "DTLZ3": {"n_var": 13, "n_gen": 600},
    "DTLZ4": {"n_var": 13, "n_gen": 600},
}


def get_problem(name: str):
    """Return (pymoo Problem instance, n_generations) for the given DTLZ name."""
    name = name.upper()
    if name not in _CONFIGS:
        raise ValueError(f"Unknown problem '{name}'. Choose from {PROBLEM_NAMES}")
    cfg = _CONFIGS[name]
    problem = _pymoo_get_problem(name.lower(), n_var=cfg["n_var"], n_obj=_N_OBJ)
    return problem, cfg["n_gen"]
