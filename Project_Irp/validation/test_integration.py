import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from validation.benchmarks.dtlz_problems import get_problem, PROBLEM_NAMES
from validation.algorithms.nsga3_runner import run_single
from validation.metrics.igd_metric import compute_igd, igd_statistics

def test_full_pipeline_one_run():
    """End-to-end: run 1 generation on each problem, compute IGD."""
    for name in PROBLEM_NAMES:
        problem, _ = get_problem(name)
        front = run_single(problem, n_gen=2, seed=42)
        igd = compute_igd(problem, front)
        assert isinstance(igd, float) and igd >= 0.0, \
            f"{name}: IGD should be non-negative float, got {igd}"
        stats = igd_statistics([igd, igd + 0.01, igd + 0.02])
        assert stats["best"] <= stats["median"] <= stats["worst"]
        print(f"  {name}: front_size={len(front)}, IGD={igd:.4f} OK")

if __name__ == "__main__":
    print("Integration test (2 gen per problem)...")
    test_full_pipeline_one_run()
    print("Integration test PASSED.")
