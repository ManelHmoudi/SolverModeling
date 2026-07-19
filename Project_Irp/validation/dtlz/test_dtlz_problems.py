import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from validation.benchmarks.dtlz_problems import get_problem, PROBLEM_NAMES

# --- Tests n_obj=4 (comportement existant, doit rester inchangé) ---

def test_dtlz1_params_4obj():
    prob, n_gen = get_problem("DTLZ1", n_obj=4)
    assert prob.n_var == 8,  f"DTLZ1 n_obj=4: n_var should be 8, got {prob.n_var}"
    assert prob.n_obj == 4
    assert n_gen == 400

def test_dtlz2_params_4obj():
    prob, n_gen = get_problem("DTLZ2", n_obj=4)
    assert prob.n_var == 13, f"DTLZ2 n_obj=4: n_var should be 13, got {prob.n_var}"
    assert prob.n_obj == 4
    assert n_gen == 600

# --- Tests n_obj=3 (nouveau comportement) ---

def test_dtlz1_params_3obj():
    prob, n_gen = get_problem("DTLZ1", n_obj=3)
    assert prob.n_var == 7,  f"DTLZ1 n_obj=3: n_var should be 7, got {prob.n_var}"
    assert prob.n_obj == 3
    assert n_gen == 400

def test_dtlz2_params_3obj():
    prob, n_gen = get_problem("DTLZ2", n_obj=3)
    assert prob.n_var == 12, f"DTLZ2 n_obj=3: n_var should be 12, got {prob.n_var}"
    assert prob.n_obj == 3
    assert n_gen == 600

def test_dtlz3_params_3obj():
    prob, n_gen = get_problem("DTLZ3", n_obj=3)
    assert prob.n_var == 12
    assert prob.n_obj == 3
    assert n_gen == 600

def test_dtlz4_params_3obj():
    prob, n_gen = get_problem("DTLZ4", n_obj=3)
    assert prob.n_var == 12
    assert prob.n_obj == 3
    assert n_gen == 600

def test_default_is_4obj():
    """get_problem without n_obj defaults to 4."""
    prob, _ = get_problem("DTLZ2")
    assert prob.n_obj == 4

def test_problem_names():
    assert PROBLEM_NAMES == ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4"]

if __name__ == "__main__":
    test_dtlz1_params_4obj()
    test_dtlz2_params_4obj()
    test_dtlz1_params_3obj()
    test_dtlz2_params_3obj()
    test_dtlz3_params_3obj()
    test_dtlz4_params_3obj()
    test_default_is_4obj()
    test_problem_names()
    print("All tests passed.")
