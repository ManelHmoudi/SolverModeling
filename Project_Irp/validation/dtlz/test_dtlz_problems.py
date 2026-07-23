import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from validation.dtlz.dtlz_problems import get_problem, PROBLEM_NAMES

# n_gen = TMAX(30000) // pop_size, pop_size from nsga3.runner.get_run_config:
#   M=3 -> N=92 -> n_gen=326 ; M=4 -> N=120 -> n_gen=250
# (Cui et al. 2025, Table 2, only studies M=3 and M=4.)

def test_problem_names():
    assert PROBLEM_NAMES == ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7"]

def test_default_is_4obj():
    prob, _ = get_problem("DTLZ2")
    assert prob.n_obj == 4

def test_dtlz1_n_var_and_gens():
    for n_obj, n_var, n_gen in [(3, 7, 326), (4, 8, 250)]:
        prob, gen = get_problem("DTLZ1", n_obj=n_obj)
        assert prob.n_var == n_var, f"DTLZ1 n_obj={n_obj}: n_var should be {n_var}, got {prob.n_var}"
        assert prob.n_obj == n_obj
        assert gen == n_gen, f"DTLZ1 n_obj={n_obj}: n_gen should be {n_gen}, got {gen}"

def test_dtlz2to6_n_var_and_gens():
    for name in ["DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6"]:
        for n_obj, n_var, n_gen in [(3, 12, 326), (4, 13, 250)]:
            prob, gen = get_problem(name, n_obj=n_obj)
            assert prob.n_var == n_var, f"{name} n_obj={n_obj}: n_var should be {n_var}, got {prob.n_var}"
            assert prob.n_obj == n_obj
            assert gen == n_gen, f"{name} n_obj={n_obj}: n_gen should be {n_gen}, got {gen}"

def test_dtlz7_n_var_and_gens():
    for n_obj, n_var, n_gen in [(3, 22, 326), (4, 23, 250)]:
        prob, gen = get_problem("DTLZ7", n_obj=n_obj)
        assert prob.n_var == n_var, f"DTLZ7 n_obj={n_obj}: n_var should be {n_var}, got {prob.n_var}"
        assert prob.n_obj == n_obj
        assert gen == n_gen, f"DTLZ7 n_obj={n_obj}: n_gen should be {n_gen}, got {gen}"

def test_unknown_problem_raises():
    try:
        get_problem("DTLZ8", n_obj=3)
        assert False, "expected ValueError for unknown problem name"
    except ValueError:
        pass

def test_unsupported_n_obj_raises():
    try:
        get_problem("DTLZ1", n_obj=5)
        assert False, "expected ValueError for unsupported n_obj"
    except ValueError:
        pass

if __name__ == "__main__":
    test_problem_names()
    test_default_is_4obj()
    test_dtlz1_n_var_and_gens()
    test_dtlz2to6_n_var_and_gens()
    test_dtlz7_n_var_and_gens()
    test_unknown_problem_raises()
    test_unsupported_n_obj_raises()
    print("All tests passed.")
