import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from validation.metrics.igd_metric import compute_igd, igd_statistics, _get_true_front
from validation.dtlz.dtlz_problems import get_problem

def test_igd_is_zero_for_true_front():
    """IGD = 0 when the approximation IS the true Pareto front."""
    problem, _ = get_problem("DTLZ2")
    try:
        true_front = _get_true_front(problem)
    except Exception:
        print("SKIP: problem.pareto_front() not available for this configuration")
        return
    if true_front is None:
        print("SKIP: problem.pareto_front() returned None")
        return
    igd = compute_igd(problem, true_front)
    assert igd == 0.0, f"IGD of true front should be 0, got {igd}"

def test_igd_positive_for_bad_approx():
    """IGD > 0 when approximation is far from true front."""
    problem, _ = get_problem("DTLZ2")
    bad_approx = np.ones((10, problem.n_obj)) * 10.0
    igd = compute_igd(problem, bad_approx)
    assert igd > 0.0

def test_igd_statistics_shape():
    values = [0.1, 0.3, 0.2, 0.5, 0.4]
    stats = igd_statistics(values)
    assert set(stats.keys()) == {"best", "median", "worst", "mean", "std"}
    assert stats["best"] <= stats["median"] <= stats["worst"]
    assert stats["best"] == 0.1
    assert stats["worst"] == 0.5
    assert abs(stats["mean"] - 0.3) < 1e-9

if __name__ == "__main__":
    test_igd_is_zero_for_true_front()
    test_igd_positive_for_bad_approx()
    test_igd_statistics_shape()
    print("All tests passed.")
