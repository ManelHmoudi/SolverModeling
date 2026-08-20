import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from Validation.Benchmarking.algorithms.moead.runner import (
    run_single, run_experiment, get_run_config, N_REF_DIRS, POP_SIZE,
)
from Validation.Benchmarking.dtlz.dtlz_problems import get_problem


def test_ref_dirs_count_4obj():
    """Das-Dennis p=7, M=4 -> H = C(10,7) = 120 (Cui et al. 2025) -- same
    ref_dirs as the NSGA-III/QI-NSGA-III benchmark runners."""
    assert N_REF_DIRS == 120, f"Expected 120, got {N_REF_DIRS}"


def test_pop_size_4obj_equals_ref_dirs_no_rounding():
    """Unlike NSGA-III's runner (rounds up to a multiple of 4), MOEA/D's
    population is exactly len(ref_dirs) -- pymoo's own MOEAD forces this,
    it cannot be rounded independently."""
    assert POP_SIZE == 120, f"Expected 120, got {POP_SIZE}"


def test_pop_size_3obj_differs_from_nsga3_by_one():
    """M=3: Das-Dennis p=12 -> H=91. NSGA-III's own runner rounds this up
    to 92 (smallest multiple of 4 >= 91); MOEA/D cannot round, so its
    population here is 91, not 92 -- a structural one-individual gap,
    documented rather than hidden."""
    ref_dirs, pop_size = get_run_config(3)
    assert len(ref_dirs) == 91
    assert pop_size == 91


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


def test_n_obj_5_unsupported():
    """Cui et al. (2025) Table 2 only covers M=3 and M=4."""
    try:
        get_run_config(5)
        assert False, "expected ValueError for unsupported n_obj=5"
    except ValueError:
        pass


if __name__ == "__main__":
    test_ref_dirs_count_4obj()
    test_pop_size_4obj_equals_ref_dirs_no_rounding()
    test_pop_size_3obj_differs_from_nsga3_by_one()
    print("Testing run_single 4obj (3 gen)...")
    test_run_single_4obj_output_shape()
    print("Testing run_single 3obj (3 gen)...")
    test_run_single_3obj_output_shape()
    print("Testing run_experiment 3obj (2 gen, 2 runs)...")
    test_run_experiment_3obj_returns_list()
    print("Testing n_obj=5 raises...")
    test_n_obj_5_unsupported()
    print("All tests passed.")
