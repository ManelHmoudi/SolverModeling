import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
from pymoo.indicators.igd import IGD
from pymoo.util.ref_dirs import get_reference_directions
from Validation.Benchmarking.maf.maf_problems import get_problem, PROBLEM_NAMES

# n_gen = TMAX(30000) // pop_size, pop_size from nsga3.runner.get_run_config:
#   M=3 -> N=92 -> n_gen=326 ; M=4 -> N=120 -> n_gen=250
# (same protocol as DTLZ, Cui et al. 2025, Table 2 — identical row for MaF 1-7.)

def test_problem_names():
    assert PROBLEM_NAMES == ["MaF1", "MaF2", "MaF3", "MaF4", "MaF5", "MaF6", "MaF7"]

def test_default_is_4obj():
    prob, _ = get_problem("MaF1")
    assert prob.n_obj == 4

def test_n_var_and_gens():
    # n_var = n_obj + k - 1: k=10 for MaF1-6, k=20 for MaF7 (same convention as DTLZ7)
    for name in PROBLEM_NAMES:
        k = 20 if name == "MaF7" else 10
        for n_obj, n_gen in [(3, 326), (4, 250)]:
            n_var = n_obj + k - 1
            prob, gen = get_problem(name, n_obj=n_obj)
            assert prob.n_var == n_var, f"{name} n_obj={n_obj}: n_var should be {n_var}, got {prob.n_var}"
            assert prob.n_obj == n_obj
            assert gen == n_gen, f"{name} n_obj={n_obj}: n_gen should be {n_gen}, got {gen}"

def test_unknown_problem_raises():
    try:
        get_problem("MaF8", n_obj=3)
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
    """Sampling decision vectors on the optimal-distance manifold should land close
    to the true front — a large gap would indicate a translation bug (CalObj and
    _calc_pareto_front disagreeing on the front's geometry). Distance vars = 0.5
    zeroes MaF1-6's g; MaF7 (DTLZ7-style) instead needs distance vars = 0 (g=1).

    IGD is normalised by the front's own scale before comparing to the threshold:
    MaF4/MaF5 scale objectives up to 2^M (~16 at M=4), so a fixed angular sampling
    gap that's negligible on MaF1-3/6-7's O(1) fronts reads as a large *absolute*
    IGD there — dividing it out isolates genuine translation bugs from this scale
    artifact. MaF5 additionally carries DTLZ4's alpha=100 diversity bias, which by
    design makes naive uniform-random x sampling cover the front poorly (that's
    the whole point of DTLZ4/MaF5 as a diversity stress test) — its relative IGD
    sits close to the threshold (~0.11) for that reason, not a bug; the 0.15
    threshold gives it headroom while still catching real translation errors
    (which are an order of magnitude larger, not a few extra percentage points)."""
    rng = np.random.default_rng(0)
    for name in PROBLEM_NAMES:
        for n_obj in (3, 4):
            prob, _ = get_problem(name, n_obj=n_obj)
            X = rng.random((2000, prob.n_var))
            X[:, n_obj - 1:] = 0.0 if name == "MaF7" else 0.5
            out = {}
            prob._evaluate(X, out)
            ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
            pf = prob._calc_pareto_front(ref_dirs)
            igd = float(IGD(pf)(out["F"]))
            scale = max(float(np.max(np.abs(pf))), 1.0)
            rel_igd = igd / scale
            assert rel_igd < 0.15, (
                f"{name} M={n_obj}: IGD={igd:.4f} (rel={rel_igd:.4f}, scale={scale:.2f}) "
                "too high — CalObj/front mismatch?"
            )

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
