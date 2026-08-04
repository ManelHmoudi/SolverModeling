"""Unit tests for QINSGA3.repair's pure helpers and orchestrator."""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Solvers.QINSGA3.repair import (
    _two_opt_candidates, _route_traversal_time,
    _rebuild_route_arcs, _replace_route_arcs,
)

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


# ── _rebuild_route_arcs ──────────────────────────────────────────────────

def test_rebuild_route_arcs_matches_decoder_suffix_sum_and_arrival_formula():
    """Hand-verified against decoder.py's own suffix-sum (build_routes,
    lines 353-368) and arrival-time accumulation (_nearest_neighbour,
    line 342) formulas:
      suf: idx2(depot)=0; idx1(node=2)=0+qty[2]=20; idx0(node=1)=20+qty[1]=30
      arrivals: at node 1, time=0+d[0,1]/1=5.0; at node 2, time=5+d[1,2]/1=8.0
    """
    path = [0, 1, 2, 0]
    qty_on_route = {1: 10, 2: 20}
    params_ = {
        "d": {(0, 1): 5.0, (1, 2): 3.0, (2, 0): 2.0},
        "v": {1: 1.0},
        "s": {},
    }
    x_vars, f_vars, arrivals = _rebuild_route_arcs(path, qty_on_route, t=1, k=1, params_=params_)

    assert x_vars == {(0, 1, 1, 1): 1, (1, 2, 1, 1): 1, (2, 0, 1, 1): 1}
    assert f_vars == {(0, 1, 1, 1): 30, (1, 2, 1, 1): 20, (2, 0, 1, 1): 0}
    assert arrivals == {(1, 1): 5.0, (2, 1): 8.0}


def test_rebuild_route_arcs_total_time_matches_route_traversal_time():
    """Sanity cross-check: the cumulative time reached at the final depot
    return (not itself returned by _rebuild_route_arcs, but derivable by
    re-running the same accumulation) must equal _route_traversal_time's
    independent computation of the same path -- both formulas are meant to
    describe the same physical quantity."""
    path = [0, 1, 2, 0]
    params_ = {
        "d": {(0, 1): 5.0, (1, 2): 3.0, (2, 0): 2.0},
        "v": {1: 1.0},
        "s": {},
    }
    # last arrival (node 2) plus the final return leg d[2,0]/v == total traversal time
    _, _, arrivals = _rebuild_route_arcs(path, {1: 10, 2: 20}, t=1, k=1, params_=params_)
    total_via_arrivals = arrivals[2, 1] + params_["d"][2, 0] / params_["v"][1]
    assert total_via_arrivals == _route_traversal_time(path, 1, params_)


# ── _replace_route_arcs ──────────────────────────────────────────────────

def test_replace_route_arcs_removes_old_and_adds_new_for_same_route_only():
    arc_dict = {
        (0, 1, 1, 1): 1, (1, 2, 1, 1): 1, (2, 0, 1, 1): 1,   # route (t=1,k=1)
        (0, 3, 1, 2): 1,                                       # different truck, same period -- untouched
    }
    old_path = [0, 1, 2, 0]
    new_entries = {(0, 2, 1, 1): 1, (2, 1, 1, 1): 1, (1, 0, 1, 1): 1}

    result = _replace_route_arcs(arc_dict, old_path, t=1, k=1, new_entries=new_entries)

    assert result == {
        (0, 2, 1, 1): 1, (2, 1, 1, 1): 1, (1, 0, 1, 1): 1,
        (0, 3, 1, 2): 1,
    }
