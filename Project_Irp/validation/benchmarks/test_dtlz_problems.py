import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from validation.benchmarks.dtlz_problems import get_problem, PROBLEM_NAMES

def test_dtlz1_params():
    prob, n_gen = get_problem("DTLZ1")
    assert prob.n_var == 8,  f"DTLZ1 n_var should be 8, got {prob.n_var}"
    assert prob.n_obj == 4,  f"DTLZ1 n_obj should be 4, got {prob.n_obj}"
    assert n_gen == 400,     f"DTLZ1 n_gen should be 400, got {n_gen}"

def test_dtlz2_params():
    prob, n_gen = get_problem("DTLZ2")
    assert prob.n_var == 13, f"DTLZ2 n_var should be 13, got {prob.n_var}"
    assert prob.n_obj == 4
    assert n_gen == 600

def test_dtlz3_params():
    prob, n_gen = get_problem("DTLZ3")
    assert prob.n_var == 13
    assert prob.n_obj == 4
    assert n_gen == 600

def test_dtlz4_params():
    prob, n_gen = get_problem("DTLZ4")
    assert prob.n_var == 13
    assert prob.n_obj == 4
    assert n_gen == 600

def test_problem_names():
    assert PROBLEM_NAMES == ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4"]

if __name__ == "__main__":
    test_dtlz1_params()
    test_dtlz2_params()
    test_dtlz3_params()
    test_dtlz4_params()
    test_problem_names()
    print("All tests passed.")
