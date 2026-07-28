import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from Validation.Benchmarking.algorithms.qinsga3.runner import run_single, run_experiment
from Validation.Benchmarking.dtlz.dtlz_problems import get_problem


def test_run_single_3obj_output_shape():
    problem, _ = get_problem("DTLZ2", n_obj=3)
    front = run_single(problem, n_gen=3, seed=42)
    assert front.ndim == 2
    assert front.shape[1] == 3


def test_run_single_4obj_output_shape():
    problem, _ = get_problem("DTLZ2", n_obj=4)
    front = run_single(problem, n_gen=3, seed=42)
    assert front.ndim == 2
    assert front.shape[1] == 4


def test_run_experiment_returns_one_front_per_run():
    problem, _ = get_problem("DTLZ1", n_obj=3)
    fronts = run_experiment("DTLZ1", problem, n_gen=2, n_runs=2)
    assert len(fronts) == 2
    for f in fronts:
        assert f.shape[1] == 3


if __name__ == "__main__":
    test_run_single_3obj_output_shape()
    print("Testing run_single 3obj (3 gen)... OK")
    test_run_single_4obj_output_shape()
    print("Testing run_single 4obj (3 gen)... OK")
    test_run_experiment_returns_one_front_per_run()
    print("Testing run_experiment (2 gen, 2 runs)... OK")
    print("All tests passed.")
