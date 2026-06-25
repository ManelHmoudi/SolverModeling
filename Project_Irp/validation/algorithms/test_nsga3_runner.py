import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from validation.algorithms.nsga3_runner import run_single, run_experiment, POP_SIZE, N_REF_DIRS
from validation.benchmarks.dtlz_problems import get_problem

def test_ref_dirs_count():
    """Das-Dennis p=6, M=4 → H = C(9,6) = 84 reference directions."""
    assert N_REF_DIRS == 84, f"Expected 84 ref dirs, got {N_REF_DIRS}"

def test_pop_size():
    """N = smallest multiple of 4 strictly greater than 84 = 88."""
    assert POP_SIZE == 88, f"Expected pop_size=88, got {POP_SIZE}"

def test_run_single_output_shape():
    """run_single returns (n_solutions, 4) array."""
    problem, n_gen = get_problem("DTLZ2")
    # Use very few generations for speed in test
    front = run_single(problem, n_gen=5, seed=42)
    assert isinstance(front, np.ndarray), "Should return ndarray"
    assert front.ndim == 2, f"Should be 2D, got shape {front.shape}"
    assert front.shape[1] == 4, f"Should have 4 objectives, got {front.shape[1]}"
    assert front.shape[0] >= 1, "Should have at least 1 solution"

def test_run_experiment_returns_list():
    """run_experiment returns list of length n_runs."""
    problem, n_gen = get_problem("DTLZ1")
    fronts = run_experiment("DTLZ1", problem, n_gen=3, n_runs=2)
    assert isinstance(fronts, list), "Should return list"
    assert len(fronts) == 2, f"Expected 2 runs, got {len(fronts)}"
    for f in fronts:
        assert f.shape[1] == 4

if __name__ == "__main__":
    test_ref_dirs_count()
    test_pop_size()
    print("Running run_single (5 gen DTLZ2)...")
    test_run_single_output_shape()
    print("Running run_experiment (3 gen DTLZ1, 2 runs)...")
    test_run_experiment_returns_list()
    print("All tests passed.")
