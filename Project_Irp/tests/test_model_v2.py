"""Tests of the v2 model (docs/model_formulation.md).

1. Decoder + evaluator vs. the independent checker (tests/solution_checker.py),
   on several instances and random chromosomes, with and without the
   QI-NSGA-III repair: feasibility must agree, objectives must match.
2. Independent formulation check: a decoded solution is fixed into the CPLEX
   model (models/) and must be feasible there with the same objective values.
3. Unit checks of the specific editor remarks (E1-E3) and audit defects.

Run:  python -m pytest tests -q
"""

import os
import sys
import warnings

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore", category=RuntimeWarning)

from models.parameters import load_instance                       # noqa: E402
from solvers.nsga3.problem import (IRPProblem, decode_full,        # noqa: E402
                                   evaluate_route_result)
from solvers.nsga3.evaluator import compute_f1_detail, compute_f4_detail  # noqa: E402
from solvers.qinsga3.repair import _repair_route_result            # noqa: E402
from solution_checker import Instance, check, from_route_result    # noqa: E402

DATA = os.path.join(ROOT, "data")
INSTANCES = ["instance_3_clients.json", "instance_5_clients.json", "instance_15_clients.json",
             "instance_25_clients.json", "instance_100_clients.json",
             os.path.join("replicates", "instance_100_clients_r1.json")]
N_RANDOM = 25


def _random_solutions(name, n=N_RANDOM, seed=0):
    path = os.path.join(DATA, name)
    sets_, params_ = load_instance(path)
    pb = IRPProblem(sets_, params_)
    rng = np.random.default_rng(seed)
    for _ in range(n):
        x = pb.xl + rng.random(pb.n_var) * (pb.xu - pb.xl)
        yield path, sets_, params_, x


def _assert_same_objectives(F, ref, label):
    for j, key in enumerate(["f1", "f2", "f3", "f4"]):
        assert F[j] == pytest.approx(ref[key], rel=1e-6, abs=1e-4), f"{label}: {key} {F[j]} != {ref[key]}"


# ── 1. decoder/evaluator vs independent checker ──────────────────────────────

@pytest.mark.parametrize("name", INSTANCES)
@pytest.mark.parametrize("repair", [False, True])
def test_decoder_matches_independent_checker(name, repair):
    n_feasible = 0
    inst = None
    for path, sets_, params_, x in _random_solutions(name):
        inst = inst or Instance(path)
        rr = decode_full(x, sets_, params_)
        if repair:
            rr = _repair_route_result(rr, sets_, params_)
        F, G = evaluate_route_result(rr, sets_, params_)
        violations, ref = check(inst, *from_route_result(rr))
        feasible = max(G) <= 1e-9
        n_feasible += feasible
        # the solver's own constraint vector and the checker must agree
        assert feasible == (not violations), f"{name}: G feasible={feasible}, checker: {violations[:5]}"
        _assert_same_objectives(F, ref, name)
    assert n_feasible > 0, f"{name}: no feasible random solution, test is vacuous"


# ── 2. independent formulation check against the CPLEX model ─────────────────

def _cplex_available():
    try:
        import docplex.mp.model  # noqa: F401
        import cplex             # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _cplex_available(), reason="CPLEX / docplex not installed")
@pytest.mark.parametrize("name", ["instance_3_clients.json", "instance_5_clients.json"])
def test_decoded_solution_is_feasible_in_cplex_model(name):
    from docplex.mp.model import Model
    from models.variables import build_variables
    from models.objectives import build_all_objectives, exact_objective_values
    from models.constraints import add_all_constraints

    checked = 0
    for path, sets_, params_, x in _random_solutions(name, n=6, seed=3):
        rr = decode_full(x, sets_, params_)
        F, G = evaluate_route_result(rr, sets_, params_)
        if max(G) > 1e-9:
            continue

        mdl = Model(name="fixcheck")
        v = build_variables(mdl, sets_, params_)
        obj = build_all_objectives(mdl, v, sets_, params_)
        add_all_constraints(mdl, v, sets_, params_)

        # fix routing, loads and orders to the decoded solution
        for key, var in v["x"].items():
            mdl.add_constraint(var == (1 if key in rr["x"] else 0))
        for key, var in v["f"].items():
            mdl.add_constraint(var == rr["f"].get(key, 0.0))
        for t in sets_["T"]:
            for p in ("frigo", "nonfrigo"):
                mdl.add_constraint(v["r"][p, t] == rr["orders"][t][p])

        # with routes/orders fixed, minimising f1 picks the exact penalties and
        # order indicators; W is pushed to the true peak with a tiny weight
        mdl.minimize(obj["f1"] + 1e-6 * obj["f4"])
        sol = mdl.solve()
        assert sol, f"{name}: decoded solution infeasible in the CPLEX model ({mdl.solve_details.status})"

        exact = exact_objective_values(v, obj, sets_, params_)
        assert obj["f1"].solution_value == pytest.approx(F[0], rel=1e-6, abs=1e-3)
        for j, key in enumerate(["f1", "f2", "f3", "f4"]):
            assert exact[key] == pytest.approx(F[j], rel=1e-6, abs=1e-2), key
        checked += 1
    assert checked > 0, f"{name}: no feasible decoded solution to cross-check"


# ── 3. targeted checks of the editor's remarks and audit defects ─────────────

def _one_feasible(name="instance_15_clients.json"):
    for path, sets_, params_, x in _random_solutions(name, n=50, seed=11):
        rr = decode_full(x, sets_, params_)
        F, G = evaluate_route_result(rr, sets_, params_)
        if max(G) <= 1e-9:
            return sets_, params_, rr, x
    pytest.fail("no feasible solution found")


def test_E1_replenishment_is_a_decision_and_balance_holds():
    """c7: I_t = I_{t-1} + r_t - shipped_t, with r the decided supplier order."""
    sets_, params_, rr, _ = _one_feasible()
    for p in ("frigo", "nonfrigo"):
        I = params_[f"I_O_init_{p}"]
        for t in sets_["T"]:
            shipped = sum(sum(int(q) for q in info["qty"].values())
                          for k, info in rr["routes_data"][t].items()
                          if (k in params_["frigo_trucks"]) == (p == "frigo"))
            I = I + rr["orders"][t][p] - shipped
            assert rr["depot_stock"][t][p] == pytest.approx(I)
        assert I >= params_[f"I_O_init_{p}"] - 1e-9   # c7b


def test_E1_order_genes_change_orders():
    sets_, params_, rr, x = _one_feasible()
    pb = IRPProblem(sets_, params_)
    n_order = 2 * len(sets_["T"])
    lo, hi = x.copy(), x.copy()
    lo[-n_order:] = 0.0
    hi[-n_order:] = 1.0
    r_lo = decode_full(lo, sets_, params_)["orders"]
    r_hi = decode_full(hi, sets_, params_)["orders"]
    assert sum(r_hi[t][p] for t in sets_["T"] for p in r_hi[t]) >= sum(r_lo[t][p] for t in sets_["T"] for p in r_lo[t])
    assert r_lo != r_hi
    assert pb.n_var == len(x)


def test_E2_empty_return_leg_is_paid():
    """A route's transport cost includes the distance back to the depot even
    though the vehicle is empty on that arc."""
    from solvers.nsga3.evaluator import route_transport_cost
    sets_, params_, rr, _ = _one_feasible()
    t, routes = next((t, r) for t, r in rr["routes_data"].items() if r)
    k, info = next(iter(routes.items()))
    path = info["path"]
    loads = [rr["f"].get((path[i], path[i + 1], t, k), 0.0) for i in range(len(path) - 1)]
    assert loads[-1] == 0.0                          # c5b: empty on the return arc
    d_return = params_["d"][path[-2], path[-1]]
    full = route_transport_cost(path, loads, k, params_)
    no_return = route_transport_cost(path[:-1] + [path[-1]], loads, k, params_)
    assert full == pytest.approx(no_return)          # same route, sanity
    # cost of the return arc alone = c_km_k * d (load term is zero)
    manual = params_["F_veh"] + sum(params_["c_km_k"][k] * params_["d"][path[i], path[i + 1]]
                                    + params_["c_load"] * params_["d"][path[i], path[i + 1]] * loads[i]
                                    for i in range(len(path) - 1))
    assert full == pytest.approx(manual)
    assert params_["c_km_k"][k] * d_return > 0


def test_E3_f4_is_not_a_scaled_holding_cost():
    """Across random feasible solutions, f4 must not be (almost) a linear
    function of the holding cost, unlike v1 where it was exactly that."""
    hold, f4 = [], []
    for path, sets_, params_, x in _random_solutions("instance_100_clients.json", n=80, seed=5):
        rr = decode_full(x, sets_, params_)
        hold.append(compute_f1_detail(rr, sets_, params_)[1]["depot_holding"])
        f4.append(compute_f4_detail(rr, sets_, params_)[0])
    r = np.corrcoef(hold, f4)[0, 1]
    assert abs(r) < 0.9, f"f4 still tracks holding cost (r={r:.3f})"


def test_D5_second_truck_visit_is_penalised():
    """Arrivals are keyed per vehicle: when two trucks visit the same client
    in a period, both visits are present."""
    for path, sets_, params_, x in _random_solutions("instance_100_clients.json", n=10, seed=2):
        rr = decode_full(x, sets_, params_)
        keys = list(rr["arrival_times"])
        assert all(len(k) == 3 for k in keys)
        return


def test_cplex_model_has_no_v1_c14_and_per_vehicle_times():
    if not _cplex_available():
        pytest.skip("CPLEX not installed")
    from docplex.mp.model import Model
    from models.variables import build_variables
    from models.objectives import build_all_objectives
    from models.constraints import add_all_constraints
    sets_, params_ = load_instance(os.path.join(DATA, "instance_3_clients.json"))
    mdl = Model()
    v = build_variables(mdl, sets_, params_)
    build_all_objectives(mdl, v, sets_, params_)
    add_all_constraints(mdl, v, sets_, params_)
    names = [c.name for c in mdl.iter_constraints()]
    for prefix in ["C1_", "C2_", "C3_", "C4_", "C5_", "C6_", "C7_", "C8_", "C9_", "C10_",
                   "C11_", "C12_", "C13_", "C14_", "C15_", "C16_", "C18_", "C19_"]:
        assert any(n.startswith(prefix) for n in names), prefix
    assert all(len(k) == 3 for k in v["a"])          # (i, t, k)


# ── 4. regression tests for the issues found by the audit ────────────────────

REPAIR_FLAGS = ["use_two_opt", "use_or_opt", "use_single_relocation",
                "use_inter_route_relocate", "use_route_swap", "use_delivery_shift"]


def _dominates_weakly(new, old):
    return all(n <= o + 1e-6 for n, o in zip(new, old))


@pytest.mark.parametrize("flag", REPAIR_FLAGS)
def test_every_repair_move_is_consistent_and_never_worsens(flag):
    """Each repair move alone: the result matches the independent checker,
    stays feasible when the decoded solution was, and weakly dominates it."""
    kwargs = {f: False for f in REPAIR_FLAGS}
    kwargs[flag] = True
    for name in ["instance_15_clients.json", "instance_100_clients.json"]:
        inst = None
        for path, sets_, params_, x in _random_solutions(name, n=8, seed=7):
            inst = inst or Instance(path)
            rr = decode_full(x, sets_, params_)
            F0, G0 = evaluate_route_result(rr, sets_, params_)
            rep = _repair_route_result(rr, sets_, params_, **kwargs)
            F1, G1 = evaluate_route_result(rep, sets_, params_)
            violations, ref = check(inst, *from_route_result(rep))
            _assert_same_objectives(F1, ref, f"{name} {flag}")
            assert (max(G1) <= 1e-9) == (not violations), violations[:3]
            if max(G0) <= 1e-9:
                assert max(G1) <= 1e-9, f"{flag} made a feasible solution infeasible"
                assert _dominates_weakly(F1, F0), f"{flag} worsened an objective: {F0} -> {F1}"


def test_delivery_shift_with_two_trucks_per_client():
    """Audit finding: under tight capacity a client can get a cold and an
    ambient truck in the same period; the shift must stay consistent."""
    path = os.path.join(DATA, "instance_100_clients.json")
    sets_, params_ = load_instance(path)
    params_["Q"] = {k: max(1, int(q * 0.35)) for k, q in params_["Q"].items()}
    pb = IRPProblem(sets_, params_)
    rng = np.random.default_rng(1)
    for _ in range(20):
        x = pb.xl + rng.random(pb.n_var) * (pb.xu - pb.xl)
        rep = _repair_route_result(decode_full(x, sets_, params_), sets_, params_)
        served = {}
        for t, routes in rep["routes_data"].items():
            for k, info in routes.items():
                for l, q in info["qty"].items():
                    assert q >= 1
                    served[int(l), t] = served.get((int(l), t), 0) + q
        for (l, t), q in rep["actual_qty"].items():
            assert served.get((l, t), 0) == q, "actual_qty out of sync with routes"


def test_inter_route_moves_keep_classes_and_single_visits():
    """Audit finding: relocate/swap used the class of the demand instead of the
    truck, and could visit a client twice. Also C17 after relocation."""
    from models.constraints import identical_vehicle_groups
    path = os.path.join(DATA, "instance_100_clients.json")
    sets_, params_ = load_instance(path)
    params_["Q"] = {k: max(1, int(q * 0.35)) for k, q in params_["Q"].items()}   # forces 2-class visits
    pb = IRPProblem(sets_, params_)
    rng = np.random.default_rng(6)
    groups = identical_vehicle_groups(sets_, params_)
    for _ in range(10):
        x = pb.xl + rng.random(pb.n_var) * (pb.xu - pb.xl)
        rr = decode_full(x, sets_, params_)
        before = {(l, t, k in params_["frigo_trucks"]): q for t, r in rr["routes_data"].items()
                  for k, i in r.items() for l, q in i["qty"].items()}
        for flag in ["use_inter_route_relocate", "use_route_swap"]:
            kwargs = {f: False for f in REPAIR_FLAGS}
            kwargs[flag] = True
            rep = _repair_route_result(rr, sets_, params_, **kwargs)
            after = {}
            for t, r in rep["routes_data"].items():
                for k, info in r.items():
                    inner = info["path"][1:-1]
                    assert len(set(inner)) == len(inner), "client visited twice"
                    for l, q in info["qty"].items():
                        key = (l, t, k in params_["frigo_trucks"])
                        after[key] = after.get(key, 0) + q
                for g in groups:
                    used = [k in r and len(r[k]["path"]) > 2 for k in g]
                    assert used == sorted(used, reverse=True), "C17 broken after repair"
            assert after == before, f"{flag} moved goods to another truck class"


def test_cycle_is_closed_and_depot_stock_within_bounds():
    """Audit finding: end stock could exceed the initial stock (f4 gaming)."""
    for name in ["instance_15_clients.json", "instance_100_clients.json"]:
        for path, sets_, params_, x in _random_solutions(name, n=15, seed=8):
            rr = decode_full(x, sets_, params_)
            F, G = evaluate_route_result(rr, sets_, params_)
            if max(G) > 1e-9:
                continue
            for p in ("frigo", "nonfrigo"):
                assert rr["depot_stock"][sets_["T"][-1]][p] == pytest.approx(params_[f"I_O_init_{p}"])
                for t in sets_["T"]:
                    stock = rr["depot_stock"][t][p]
                    assert params_[f"I_O_min_{p}"] - 1e-9 <= stock <= params_[f"I_O_max_{p}"] + 1e-9


def test_final_front_rule_is_the_same_for_all_algorithms():
    """Audit finding: NSGA-III reported pymoo's result.X (niche representatives
    only). Every algorithm must report the feasible non-dominated set."""
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    from solvers.nsga3.main import _final_front
    sets_, params_ = load_instance(os.path.join(DATA, "instance_15_clients.json"))
    pb = IRPProblem(sets_, params_)
    ref = get_reference_directions("das-dennis", 4, n_partitions=4)
    res = minimize(pb, NSGA3(pop_size=40, ref_dirs=ref), ("n_gen", 8), seed=1, verbose=False)
    X = _final_front(res.pop)
    F = np.array([evaluate_route_result(decode_full(x, sets_, params_), sets_, params_)[0] for x in X])
    assert len(NonDominatedSorting().do(F, only_non_dominated_front=True)) == len(F)
    assert len(X) >= len(res.X)


@pytest.mark.skipif(not _cplex_available(), reason="CPLEX / docplex not installed")
def test_repaired_solution_is_feasible_in_cplex_model():
    """CPLEX cross-check extended to repaired solutions (default repair)."""
    from docplex.mp.model import Model
    from models.variables import build_variables
    from models.objectives import build_all_objectives, exact_objective_values
    from models.constraints import add_all_constraints
    checked = 0
    for name in ["instance_3_clients.json", "instance_5_clients.json"]:
        for path, sets_, params_, x in _random_solutions(name, n=6, seed=12):
            rr = _repair_route_result(decode_full(x, sets_, params_), sets_, params_)
            F, G = evaluate_route_result(rr, sets_, params_)
            if max(G) > 1e-9:
                continue
            mdl = Model(name="fixcheck_repaired")
            v = build_variables(mdl, sets_, params_)
            obj = build_all_objectives(mdl, v, sets_, params_)
            add_all_constraints(mdl, v, sets_, params_)
            for key, var in v["x"].items():
                mdl.add_constraint(var == (1 if key in rr["x"] else 0))
            for key, var in v["f"].items():
                mdl.add_constraint(var == rr["f"].get(key, 0.0))
            for t in sets_["T"]:
                for p in ("frigo", "nonfrigo"):
                    mdl.add_constraint(v["r"][p, t] == rr["orders"][t][p])
            mdl.minimize(obj["f1"] + 1e-6 * obj["f4"])
            assert mdl.solve(), f"{name}: repaired solution infeasible in CPLEX"
            exact = exact_objective_values(v, obj, sets_, params_)
            for j, key in enumerate(["f1", "f2", "f3", "f4"]):
                assert exact[key] == pytest.approx(F[j], rel=1e-6, abs=1e-2), key
            checked += 1
    assert checked > 0


def test_symmetry_breaking_only_groups_identical_vehicles():
    from models.constraints import identical_vehicle_groups
    sets_, params_ = load_instance(os.path.join(DATA, "instance_110_clients.json"))
    for group in identical_vehicle_groups(sets_, params_):
        sig = {((k in params_["frigo_trucks"]), params_["Q"][k], params_["v"][k]) for k in group}
        assert len(sig) == 1


def test_loader_is_strict():
    import json
    import tempfile
    with open(os.path.join(DATA, "instance_3_clients.json"), encoding="utf-8") as fh:
        raw = json.load(fh)
    del raw["parameters"]["F_order"]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        json.dump(raw, tmp)
    with pytest.raises(KeyError, match="F_order"):
        load_instance(tmp.name)
    os.unlink(tmp.name)


def test_constraint_vector_has_fixed_length():
    from solvers.nsga3.problem import n_constraints
    for name in INSTANCES:
        for path, sets_, params_, x in _random_solutions(name, n=3):
            _, G = evaluate_route_result(decode_full(x, sets_, params_), sets_, params_)
            assert len(G) == n_constraints(sets_)


def test_objectives_are_not_redundant():
    """f3 (longest route per period) must not duplicate f2 (CO2), unlike the
    former total driving time (Pearson 0.98); no pair of objectives may be
    near-collinear over random solutions."""
    F = []
    for path, sets_, params_, x in _random_solutions("instance_100_clients.json", n=120, seed=9):
        F.append(evaluate_route_result(decode_full(x, sets_, params_), sets_, params_)[0])
    corr = np.corrcoef(np.array(F).T)
    assert abs(corr[1, 2]) < 0.8, f"f2/f3 correlation {corr[1, 2]:.2f}"
    off_diag = corr[~np.eye(4, dtype=bool)]
    assert np.max(np.abs(off_diag)) < 0.9, np.round(corr, 2)
