"""Unit tests for QINSGA3.repair's pure helpers and orchestrator."""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Solvers.QINSGA3.repair import _two_opt_candidates, _route_traversal_time

# ── _two_opt_candidates ──────────────────────────────────────────────────

def test_two_opt_candidates_two_client_path_yields_single_swap():
    path = [0, 1, 2, 0]
    candidates = list(_two_opt_candidates(path))
    assert candidates == [(1, 2, [0, 2, 1, 0])]


def test_two_opt_candidates_three_client_path_yields_three_swaps():
    path = [0, 1, 2, 3, 0]
    candidates = list(_two_opt_candidates(path))
    assert len(candidates) == 3
    assert (1, 2, [0, 2, 1, 3, 0]) in candidates
    assert (1, 3, [0, 3, 2, 1, 0]) in candidates
    assert (2, 3, [0, 1, 3, 2, 0]) in candidates


def test_two_opt_candidates_single_client_path_yields_nothing():
    """len(path) == 3 (one client) has no interior segment to reverse --
    the caller (_repair_route_result) also short-circuits on this case, but
    the generator itself must be safe to call regardless."""
    path = [0, 1, 0]
    assert list(_two_opt_candidates(path)) == []


# ── _route_traversal_time ────────────────────────────────────────────────

def test_route_traversal_time_sums_distance_over_speed_plus_service():
    path = [0, 1, 2, 0]
    params_ = {
        "d": {(0, 1): 5.0, (1, 2): 3.0, (2, 0): 2.0},
        "v": {1: 2.0},
        "s": {0: 0.0, 1: 1.0, 2: 1.0},
    }
    result = _route_traversal_time(path, 1, params_)
    expected = (0.0 + 5.0 / 2.0) + (1.0 + 3.0 / 2.0) + (1.0 + 2.0 / 2.0)
    assert result == expected
