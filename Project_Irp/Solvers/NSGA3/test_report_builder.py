"""Unit tests for report_builder._evaluate_pareto's repair flag -- confirms
NSGA3 and QINSGA3 share one code path for optionally applying the
post-decode 2-opt repair (Solvers/QINSGA3/repair.py) before objectives are
computed, so a repair_final_front flag means the same thing for either
algorithm."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.parametres import load_instance
from Solvers.NSGA3.decoder import decode_chromosome, build_routes
from Solvers.NSGA3.evaluator import compute_f1
from Solvers.NSGA3.problem import IRPProblem
from Solvers.NSGA3.report_builder import _evaluate_pareto
from Solvers.QINSGA3.repair import _repair_route_result

PROJECT_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TINY_INSTANCE  = os.path.join(PROJECT_DIR, "data", "instance_3_clients.json")
_SMALL_INSTANCE = os.path.join(PROJECT_DIR, "data", "instance_5_clients.json")


def _improvable_chromosome(sets_, params_):
    """A chromosome (instance_5_clients, rng seed 7's first draw) whose
    decoded route the 2-opt repair provably improves -- hand-confirmed
    (f1 604.96 -> 512.785) before writing this test, so repair=True vs.
    repair=False are guaranteed to differ, not just both trivially agree
    (real IRP routes on tiny instances are often already too short for any
    2-opt swap to apply -- see instance_3_clients's routes, all <=3 nodes)."""
    problem = IRPProblem(sets_, params_)
    rng = np.random.default_rng(7)
    return problem.xl + rng.random(len(problem.xl)) * (problem.xu - problem.xl)


def test_evaluate_pareto_repair_false_matches_plain_decode():
    sets_, params_ = load_instance(_TINY_INSTANCE)
    chromosome = IRPProblem(sets_, params_).xu

    result = _evaluate_pareto(np.array([chromosome]), sets_, params_, {}, repair=False)

    quantities, priorities = decode_chromosome(chromosome, sets_)
    route_result = build_routes(quantities, sets_, params_, priorities)
    expected_f1 = round(compute_f1(route_result, sets_, params_), 4)

    assert result["solutions"][0]["objectives"]["f1"] == expected_f1


def test_evaluate_pareto_repair_true_applies_two_opt():
    sets_, params_ = load_instance(_SMALL_INSTANCE)
    chromosome = _improvable_chromosome(sets_, params_)

    unrepaired = _evaluate_pareto(np.array([chromosome]), sets_, params_, {}, repair=False)
    repaired   = _evaluate_pareto(np.array([chromosome]), sets_, params_, {}, repair=True)

    assert repaired["solutions"][0]["objectives"]["f1"] < unrepaired["solutions"][0]["objectives"]["f1"]


def test_evaluate_pareto_repair_true_matches_manual_repair_call():
    sets_, params_ = load_instance(_SMALL_INSTANCE)
    chromosome = _improvable_chromosome(sets_, params_)

    result = _evaluate_pareto(np.array([chromosome]), sets_, params_, {}, repair=True)

    quantities, priorities = decode_chromosome(chromosome, sets_)
    route_result = build_routes(quantities, sets_, params_, priorities)
    repaired = _repair_route_result(route_result, sets_, params_)
    expected_f1 = round(compute_f1(repaired, sets_, params_), 4)

    assert result["solutions"][0]["objectives"]["f1"] == expected_f1
