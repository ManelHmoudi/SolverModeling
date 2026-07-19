import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from validation.maf.maf_problems import get_problem, PROBLEM_NAMES
from validation.metrics.igd_metric import compute_igd

# n_gen = TMAX(30000) // pop_size, pop_size from nsga3_runner.get_run_config:
#   M=3 -> N=92 -> n_gen=326 ; M=4 -> N=120 -> n_gen=250
# (same protocol as DTLZ, Cui et al. 2025, Table 2 — identical row for MaF 1-7.)

def test_problem_names():
    assert PROBLEM_NAMES == ["MaF1", "MaF2", "MaF3"]

def test_default_is_4obj():
    prob, _ = get_problem("MaF1")
    assert prob.n_obj == 4

def test_n_var_and_gens():
    # n_var = n_obj + 9 for all three (k=10 distance variables)
    for name in PROBLEM_NAMES:
        for n_obj, n_var, n_gen in [(3, 12, 326), (4, 13, 250)]:
            prob, gen = get_problem(name, n_obj=n_obj)
            assert prob.n_var == n_var, f"{name} n_obj={n_obj}: n_var should be {n_var}, got {prob.n_var}"
            assert prob.n_obj == n_obj
            assert gen == n_gen, f"{name} n_obj={n_obj}: n_gen should be {n_gen}, got {gen}"

def test_unknown_problem_raises():
    try:
        get_problem("MaF4", n_obj=3)
        assert False, "expected ValueError for unknown problem name"
    except ValueError:
        pass

def test_unsupported_n_obj_raises():
    try:
        get_problem("MaF1", n_obj=5)
        assert False, "expected ValueError for unsupported n_obj"
    except ValueError:
        pass

def test_fronts_have_no_nan_or_negative_values():
    for name in PROBLEM_NAMES:
        for n_obj in (3, 4):
            prob, _ = get_problem(name, n_obj=n_obj)
            pf = prob.pareto_front()
            assert len(pf) > 10, f"{name} M={n_obj}: suspiciously few reference points ({len(pf)})"
            assert not np.isnan(pf).any(), f"{name} M={n_obj}: NaN in reference front"
            assert (pf >= -1e-9).all(), f"{name} M={n_obj}: negative objective in reference front"

def test_maf2_front_is_on_unit_sphere():
    """MaF2's front has no power transform, so it must lie exactly on the unit sphere."""
    prob, _ = get_problem("MaF2", n_obj=3)
    pf = prob.pareto_front()
    norms = np.sum(pf ** 2, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6), f"MaF2 front not on unit sphere: min={norms.min()}, max={norms.max()}"

def test_objective_function_agrees_with_reference_front():
    """Sampling decision vectors on the g=0 manifold (distance vars = 0.5) should
    land close to the true front — a large gap would indicate a translation bug
    (CalObj and _calc_pareto_front disagreeing on the front's geometry)."""
    rng = np.random.default_rng(0)
    for name in PROBLEM_NAMES:
        for n_obj in (3, 4):
            prob, _ = get_problem(name, n_obj=n_obj)
            X = rng.random((2000, prob.n_var))
            X[:, n_obj - 1:] = 0.5
            out = {}
            prob._evaluate(X, out)
            igd = compute_igd(prob, out["F"])
            assert igd < 0.1, f"{name} M={n_obj}: IGD={igd:.4f} too high — CalObj/front mismatch?"

if __name__ == "__main__":
    test_problem_names()
    test_default_is_4obj()
    test_n_var_and_gens()
    test_unknown_problem_raises()
    test_unsupported_n_obj_raises()
    test_fronts_have_no_nan_or_negative_values()
    test_maf2_front_is_on_unit_sphere()
    test_objective_function_agrees_with_reference_front()
    print("All tests passed.")
