import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from validation.algorithms.nsga3_runner import run_single, run_experiment, POP_SIZE, N_REF_DIRS
from validation.benchmarks.dtlz_problems import get_problem

def test_ref_dirs_count_4obj():
    """Das-Dennis p=6, M=4 -> H = C(9,6) = 84."""
    assert N_REF_DIRS == 84, f"Expected 84, got {N_REF_DIRS}"

def test_pop_size_4obj():
    """N = smallest multiple of 4 strictly > 84 = 88."""
    assert POP_SIZE == 88, f"Expected 88, got {POP_SIZE}"

def test_run_single_4obj_output_shape():
    problem, _ = get_problem("DTLZ2", n_obj=4)
    front = run_single(problem, n_gen=3, seed=42)
    assert front.ndim == 2
    assert front.shape[1] == 4

def test_run_single_3obj_output_shape():
    """run_single must adapt ref_dirs and pop_size for n_obj=3."""
    problem, _ = get_problem("DTLZ2", n_obj=3)
    front = run_single(problem, n_gen=3, seed=42)
    assert front.ndim == 2
    assert front.shape[1] == 3, f"Expected 3 objectives, got {front.shape[1]}"

def test_run_experiment_3obj_returns_list():
    problem, _ = get_problem("DTLZ1", n_obj=3)
    fronts = run_experiment("DTLZ1", problem, n_gen=2, n_runs=2)
    assert len(fronts) == 2
    for f in fronts:
        assert f.shape[1] == 3

if __name__ == "__main__":
    test_ref_dirs_count_4obj()
    test_pop_size_4obj()
    print("Testing run_single 4obj (3 gen)...")
    test_run_single_4obj_output_shape()
    print("Testing run_single 3obj (3 gen)...")
    test_run_single_3obj_output_shape()
    print("Testing run_experiment 3obj (2 gen, 2 runs)...")
    test_run_experiment_3obj_returns_list()
    print("All tests passed.")
