"""Unit tests for QINSGA3.repair's pure helpers and orchestrator."""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Solvers.QINSGA3.repair import (
    _two_opt_candidates, _route_traversal_time,
    _rebuild_route_arcs, _replace_route_arcs,
    _repair_route_result, _route_f1_contribution,
)
from Solvers.NSGA3.evaluator import compute_f1

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


# ── _repair_route_result ─────────────────────────────────────────────────

def test_repair_route_result_improves_f1_via_two_opt_swap():
    """3-client route with a known-bad visiting order; the (i=1,j=2) 2-opt
    swap strictly reduces f1 by cutting total flow-weighted distance, with
    no time-window pressure (ET=0, LT huge) so the improvement is entirely
    attributable to the 2-opt reordering. Unlisted arcs default to a large
    distance (1000.0) so later, unplanned-for candidates the search may
    explore on a second pass are automatically rejected by the tau_return
    guard rather than raising KeyError.

    Hand-verified:
      original y1 = d[0,1]*35 + d[1,2]*25 + d[2,3]*5 + d[3,0]*0
                   = 5*35 + 5*25 + 1*5 + 1*0 = 175+125+5+0 = 305
      candidate y1 = d[0,2]*35 + d[2,1]*15 + d[1,3]*5 + d[3,0]*0
                    = 1*35 + 1*15 + 1*5 + 1*0 = 35+15+5+0 = 55
    """
    sets_ = {"clients": [1, 2, 3], "T": [1]}
    d = defaultdict(lambda: 1000.0)
    d.update({
        (0, 1): 5.0, (1, 2): 5.0, (2, 3): 1.0, (3, 0): 1.0,
        (0, 2): 1.0, (2, 1): 1.0, (1, 3): 1.0,
    })
    params_ = {
        "d": d,
        "v": {1: 1.0},
        "s": {},
        "c_ijk": defaultdict(lambda: 1.0),
        "h_O": 0.0, "c1": 0.0, "c2": 0.0,
        "ET": defaultdict(lambda: 0.0),
        "LT": defaultdict(lambda: 1e9),
    }
    route_result = {
        "x": {(0, 1, 1, 1): 1, (1, 2, 1, 1): 1, (2, 3, 1, 1): 1, (3, 0, 1, 1): 1},
        "f": {(0, 1, 1, 1): 35, (1, 2, 1, 1): 25, (2, 3, 1, 1): 5, (3, 0, 1, 1): 0},
        "depot_stock": {1: {"frigo": 0.0, "nonfrigo": 0.0}},
        "arrival_times": {(1, 1): 5.0, (2, 1): 10.0, (3, 1): 11.0},
        "truck_assign": {(1, 1): 1, (2, 1): 1, (3, 1): 1},
        "actual_qty": {(1, 1): 10, (2, 1): 20, (3, 1): 5},
        "routes_data": {1: {1: {"path": [0, 1, 2, 3, 0], "qty": {"1": 10, "2": 20, "3": 5}}}},
        "tau_return": {1: 12.0},
    }

    repaired = _repair_route_result(route_result, sets_, params_)

    assert repaired["routes_data"][1][1]["path"] == [0, 2, 1, 3, 0]
    assert compute_f1(repaired, sets_, params_) < compute_f1(route_result, sets_, params_)


def test_repair_route_result_rejects_swap_that_would_increase_tau_return():
    """2-client route where the only 2-opt swap strictly improves f1 (moves
    the heavy-flow leg onto a much shorter arc) but would more than triple
    the route's total traversal time -- the safety guard must reject it and
    leave the path unchanged, even though f1 would otherwise improve.

    Hand-verified:
      original: tau_return_before = d[0,1]+d[1,2]+d[2,0] = 1+1+1 = 3.0
                y1 = d[0,1]*101 + d[1,2]*1 + d[2,0]*0 = 101+1+0 = 102
      candidate: time = d[0,2]+d[2,1]+d[1,0] = 0.001+0.001+10 = 10.002 > 3.0 -- rejected
                (y1 would have been 0.001*101+0.001*100+10*0 = 0.201, an
                 improvement, but the guard fires before f1 is even checked)
    """
    sets_ = {"clients": [1, 2], "T": [1]}
    params_ = {
        "d": {
            (0, 1): 1.0, (1, 2): 1.0, (2, 0): 1.0,
            (0, 2): 0.001, (2, 1): 0.001, (1, 0): 10.0,
        },
        "v": {1: 1.0},
        "s": {},
        "c_ijk": defaultdict(lambda: 1.0),
        "h_O": 0.0, "c1": 0.0, "c2": 0.0,
        "ET": defaultdict(lambda: 0.0),
        "LT": defaultdict(lambda: 1e9),
    }
    route_result = {
        "x": {(0, 1, 1, 1): 1, (1, 2, 1, 1): 1, (2, 0, 1, 1): 1},
        "f": {(0, 1, 1, 1): 101, (1, 2, 1, 1): 1, (2, 0, 1, 1): 0},
        "depot_stock": {1: {"frigo": 0.0, "nonfrigo": 0.0}},
        "arrival_times": {(1, 1): 1.0, (2, 1): 2.0},
        "truck_assign": {(1, 1): 1, (2, 1): 1},
        "actual_qty": {(1, 1): 100, (2, 1): 1},
        "routes_data": {1: {1: {"path": [0, 1, 2, 0], "qty": {"1": 100, "2": 1}}}},
        "tau_return": {1: 3.0},
    }

    repaired = _repair_route_result(route_result, sets_, params_)

    assert repaired["routes_data"][1][1]["path"] == [0, 1, 2, 0]


def test_repair_route_result_leaves_already_optimal_route_unchanged():
    """2-client route with fully symmetric distances and quantities -- both
    visiting orders give identical f1 (30 == 30, hand-verified), so no
    STRICT improvement exists and the path must be left untouched (no
    spurious 'improvement' on a tie, no infinite loop)."""
    sets_ = {"clients": [1, 2], "T": [1]}
    params_ = {
        "d": {(0, 1): 1.0, (1, 2): 1.0, (2, 0): 1.0, (0, 2): 1.0, (2, 1): 1.0, (1, 0): 1.0},
        "v": {1: 1.0},
        "s": {},
        "c_ijk": defaultdict(lambda: 1.0),
        "h_O": 0.0, "c1": 0.0, "c2": 0.0,
        "ET": defaultdict(lambda: 0.0),
        "LT": defaultdict(lambda: 1e9),
    }
    route_result = {
        "x": {(0, 1, 1, 1): 1, (1, 2, 1, 1): 1, (2, 0, 1, 1): 1},
        "f": {(0, 1, 1, 1): 20, (1, 2, 1, 1): 10, (2, 0, 1, 1): 0},
        "depot_stock": {1: {"frigo": 0.0, "nonfrigo": 0.0}},
        "arrival_times": {(1, 1): 1.0, (2, 1): 2.0},
        "truck_assign": {(1, 1): 1, (2, 1): 1},
        "actual_qty": {(1, 1): 10, (2, 1): 10},
        "routes_data": {1: {1: {"path": [0, 1, 2, 0], "qty": {"1": 10, "2": 10}}}},
        "tau_return": {1: 3.0},
    }

    repaired = _repair_route_result(route_result, sets_, params_)

    assert repaired["routes_data"][1][1]["path"] == [0, 1, 2, 0]


# ── _route_f1_contribution ───────────────────────────────────────────────
# Performance fix: _repair_route_result originally called compute_f1 on a
# full scratch route_result per candidate (correct, but O(whole network)
# per candidate -- measured ~29x slower than baseline on the real IRP,
# sensitivity/compare_route_repair.py). _route_f1_contribution computes
# only the ONE repaired route's y1+y3 contribution, O(route length). This
# test proves the two are exactly equivalent as an acceptance criterion,
# using the real compute_f1 formula on a route_result with a SECOND,
# untouched route present -- proving the untouched route's contribution
# (and y2, holding cost) cancels exactly in the delta, not just that the
# two formulas agree when only one route exists in the whole network.

def test_route_f1_contribution_delta_matches_compute_f1_delta():
    """Same repairable route (path [0,1,2,3,0] -> [0,2,1,3,0]) as
    test_repair_route_result_improves_f1_via_two_opt_swap, PLUS a second,
    untouched route (truck 2, client 4, path [0,4,0] -- length 3, never
    repaired) with its own nonzero transport cost and time-window penalty
    (LT[4,1]=1.0, arrival=6.0 -> late by 5.0). If the isolation claim in
    _route_f1_contribution's docstring is correct, this second route's
    contribution -- and y2 -- must cancel exactly in the delta regardless
    of their actual values.
    """
    sets_ = {"clients": [1, 2, 3, 4], "T": [1]}
    d = defaultdict(lambda: 1000.0)
    d.update({
        (0, 1): 5.0, (1, 2): 5.0, (2, 3): 1.0, (3, 0): 1.0,
        (0, 2): 1.0, (2, 1): 1.0, (1, 3): 1.0,
        (0, 4): 2.0, (4, 0): 2.0,
    })
    params_ = {
        "d": d,
        "v": {1: 1.0, 2: 1.0},
        "s": {},
        "c_ijk": defaultdict(lambda: 1.0),
        "h_O": 0.0, "c1": 0.0, "c2": 1.0,
        "ET": defaultdict(lambda: 0.0),
        "LT": defaultdict(lambda: 1e9, {(1, 1): 3.0, (2, 1): 8.0, (3, 1): 12.0, (4, 1): 1.0}),
    }
    route_result = {
        "x": {
            (0, 1, 1, 1): 1, (1, 2, 1, 1): 1, (2, 3, 1, 1): 1, (3, 0, 1, 1): 1,
            (0, 4, 1, 2): 1, (4, 0, 1, 2): 1,
        },
        "f": {
            (0, 1, 1, 1): 35, (1, 2, 1, 1): 25, (2, 3, 1, 1): 5, (3, 0, 1, 1): 0,
            (0, 4, 1, 2): 8, (4, 0, 1, 2): 0,
        },
        "depot_stock": {1: {"frigo": 0.0, "nonfrigo": 0.0}},
        "arrival_times": {(1, 1): 5.0, (2, 1): 10.0, (3, 1): 11.0, (4, 1): 6.0},
        "truck_assign": {(1, 1): 1, (2, 1): 1, (3, 1): 1, (4, 1): 2},
        "actual_qty": {(1, 1): 10, (2, 1): 20, (3, 1): 5, (4, 1): 8},
        "routes_data": {1: {
            1: {"path": [0, 1, 2, 3, 0], "qty": {"1": 10, "2": 20, "3": 5}},
            2: {"path": [0, 4, 0], "qty": {"4": 8}},
        }},
        "tau_return": {1: 12.0},
    }

    repaired = _repair_route_result(route_result, sets_, params_)
    assert repaired["routes_data"][1][1]["path"] == [0, 2, 1, 3, 0]   # sanity: repair still fires
    assert repaired["routes_data"][1][2]["path"] == [0, 4, 0]          # sanity: route 2 untouched

    full_delta = compute_f1(repaired, sets_, params_) - compute_f1(route_result, sets_, params_)

    original_path = route_result["routes_data"][1][1]["path"]
    repaired_path = repaired["routes_data"][1][1]["path"]
    contribution_delta = (
        _route_f1_contribution(repaired_path, repaired["f"], repaired["arrival_times"], 1, 1, params_)
        - _route_f1_contribution(original_path, route_result["f"], route_result["arrival_times"], 1, 1, params_)
    )

    assert full_delta == contribution_delta
