"""Unit tests for QINSGA3.repair's pure helpers and orchestrator."""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Solvers.QINSGA3.repair import (
    _two_opt_candidates, _route_traversal_time,
    _rebuild_route_arcs, _replace_route_arcs,
    _repair_route_result, _route_f1_contribution,
    _evaluate_candidate, _or_opt_candidates,
    _inter_route_relocate_candidates, _repair_inter_route_relocate,
    _route_swap_candidates, _repair_route_swap,
    _delivery_shift_candidates, _repair_delivery_shift,
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


def test_two_opt_candidates_default_max_window_is_unbounded():
    """max_window=None (the default) must yield exactly the same pairs as
    calling without the parameter at all -- confirms the windowed search
    added later is purely additive, not a behaviour change to the
    unbounded default every existing caller/test relies on."""
    path = [0, 1, 2, 3, 4, 5, 0]   # 5 interior positions -> C(5,2) = 10 pairs
    assert list(_two_opt_candidates(path)) == list(_two_opt_candidates(path, max_window=None))
    assert len(list(_two_opt_candidates(path))) == 10


def test_two_opt_candidates_max_window_excludes_distant_pairs():
    """Same 7-node path (5 interior positions, 1..5), max_window=2: only
    pairs with j - i <= 2 survive. Hand-enumerated from the 10 unbounded
    pairs (i,j) with distance d=j-i: (1,2)d1 (1,3)d2 (1,4)d3 (1,5)d4
    (2,3)d1 (2,4)d2 (2,5)d3 (3,4)d1 (3,5)d2 (4,5)d1 -- excluding d>2 leaves
    exactly 7: (1,2) (1,3) (2,3) (2,4) (3,4) (3,5) (4,5)."""
    path = [0, 1, 2, 3, 4, 5, 0]

    pairs = [(i, j) for i, j, _ in _two_opt_candidates(path, max_window=2)]

    assert pairs == [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)]
    assert all(j - i <= 2 for i, j in pairs)


# ── _or_opt_candidates ───────────────────────────────────────────────────

def test_or_opt_candidates_segment_length_2_hand_verified():
    """5-node path (0,1,2,3,4,0), segment_lengths=(2,) only: two possible
    2-client segments ([1,2] at start=1, [2,3] at start=2), each with one
    valid non-no-op reinsertion point (the other candidate insertion point
    for each is skipped as the original position)."""
    path = [0, 1, 2, 3, 0]
    candidates = list(_or_opt_candidates(path, segment_lengths=(2,)))
    assert candidates == [
        (1, 1, [0, 3, 1, 2, 0]),
        (2, 0, [0, 2, 3, 1, 0]),
    ]


def test_or_opt_candidates_segment_length_3_hand_verified():
    """6-node path (0,1,2,3,4,0... wait 4 interior), segment_lengths=(3,):
    two possible 3-client segments ([1,2,3] at start=1, [2,3,4] at
    start=2), each with exactly one valid non-no-op reinsertion point."""
    path = [0, 1, 2, 3, 4, 0]
    candidates = list(_or_opt_candidates(path, segment_lengths=(3,)))
    assert candidates == [
        (1, 1, [0, 4, 1, 2, 3, 0]),
        (2, 0, [0, 2, 3, 4, 1, 0]),
    ]


def test_or_opt_candidates_too_short_path_yields_nothing():
    """Only 1 interior client -- no segment of length >= 2 fits."""
    path = [0, 1, 0]
    assert list(_or_opt_candidates(path, segment_lengths=(2, 3))) == []


def test_or_opt_candidates_never_reinserts_at_original_position():
    """The (q == start - 1) case (segment landing back where it started)
    must never appear among the yielded candidates, for any segment
    length -- it is a no-op, not a real move."""
    path = [0, 1, 2, 3, 4, 5, 0]
    for start, q, candidate in _or_opt_candidates(path, segment_lengths=(2, 3)):
        assert q != start - 1
        assert candidate != path


def test_or_opt_candidates_preserves_segment_order_no_reversal():
    """Every candidate must contain the moved segment as a contiguous,
    UN-reversed subsequence somewhere in the new path (Or-opt relocates,
    it does not reverse -- that is 2-opt's job). Checked directly against
    each known segment for this path (start position -> the 2- or 3-client
    chunk originally at that position)."""
    path = [0, 1, 2, 3, 4, 0]
    for seg, start in ([1, 2], 1), ([2, 3], 2), ([3, 4], 3), ([1, 2, 3], 1), ([2, 3, 4], 2):
        for s, q, candidate in _or_opt_candidates(path, segment_lengths=(len(seg),)):
            if s == start:
                # find seg as a contiguous run in candidate
                found = any(
                    candidate[i:i + len(seg)] == seg
                    for i in range(len(candidate) - len(seg) + 1)
                )
                assert found, f"segment {seg} not found intact in {candidate}"


def test_or_opt_candidates_max_window_bounds_reinsertion_distance():
    """max_window=1: every yielded q must satisfy |q - (start-1)| <= 1."""
    path = [0, 1, 2, 3, 4, 5, 6, 0]
    candidates = list(_or_opt_candidates(path, segment_lengths=(2,), max_window=1))
    assert len(candidates) > 0
    assert all(abs(q - (start - 1)) <= 1 for start, q, _ in candidates)


def test_or_opt_candidates_default_max_window_is_unbounded():
    """max_window=None (the default) must yield exactly the same candidates
    as calling without the parameter at all."""
    path = [0, 1, 2, 3, 4, 5, 0]
    assert (list(_or_opt_candidates(path, segment_lengths=(2, 3)))
            == list(_or_opt_candidates(path, segment_lengths=(2, 3), max_window=None)))


def test_or_opt_candidates_segment_length_1_hand_verified():
    """4-node path (0,1,2,3,0), segment_lengths=(1,) only (single-client
    relocation): 3 possible single clients (1, 2, 3), each with 2 valid
    non-no-op reinsertion points (out of 3 possible interior slots, minus
    the original one). Hand-verified by tracing _or_opt_candidates'
    rest/q construction for each start position."""
    path = [0, 1, 2, 3, 0]
    candidates = list(_or_opt_candidates(path, segment_lengths=(1,)))
    assert candidates == [
        (1, 1, [0, 2, 1, 3, 0]),
        (1, 2, [0, 2, 3, 1, 0]),
        (2, 0, [0, 2, 1, 3, 0]),
        (2, 2, [0, 1, 3, 2, 0]),
        (3, 0, [0, 3, 1, 2, 0]),
        (3, 1, [0, 1, 3, 2, 0]),
    ]


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

    repaired = _repair_route_result(route_result, sets_, params_,
                                     use_two_opt=True, use_delivery_shift=False)

    assert repaired["routes_data"][1][1]["path"] == [0, 2, 1, 3, 0]
    assert compute_f1(repaired, sets_, params_) < compute_f1(route_result, sets_, params_)


def test_repair_route_result_use_two_opt_false_disables_2opt_scan():
    """Same rigged route as test_repair_route_result_improves_f1_via_two_opt_swap
    (the only improving move is the 2-opt swap (i=1,j=2)) -- with
    use_two_opt=False and no other neighbourhood enabled, no candidate is
    ever generated, so the route must be returned completely unchanged
    even though an improving 2-opt move exists."""
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

    repaired = _repair_route_result(route_result, sets_, params_,
                                     use_two_opt=False, use_delivery_shift=False)

    assert repaired["routes_data"][1][1]["path"] == [0, 1, 2, 3, 0]
    assert compute_f1(repaired, sets_, params_) == compute_f1(route_result, sets_, params_)


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

    repaired = _repair_route_result(route_result, sets_, params_,
                                     use_two_opt=True, use_delivery_shift=False)

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

    repaired = _repair_route_result(route_result, sets_, params_,
                                     use_two_opt=True, use_delivery_shift=False)

    assert repaired["routes_data"][1][1]["path"] == [0, 1, 2, 0]


def test_repair_route_result_or_opt_finds_improvement_two_opt_alone_cannot():
    """5-node route (clients 1,2,3,4, one unit each), distances rigged so
    relocating the 2-client segment [1,2] to the front (equivalently: [3,4]
    to the back) strictly improves f1 (64 -> 10), while NONE of the six
    possible 2-opt segment reversals of this path improve on the original
    -- verified numerically: every reversal introduces at least one arc
    outside the small "cheap" set (explicit 1.0/10.0 entries below), which
    defaults to 1000.0, so all six cost 3043-10000, far above the
    original's 64. use_or_opt=False must therefore leave the path
    unchanged (2-opt alone cannot find this improvement); use_or_opt=True
    must find it.

    Hand-verified original: y1 = 4*d[0,1] + 3*d[1,2] + 2*d[2,3] + 1*d[3,4] + 0*d[4,0]
                                = 4*10 + 3*1 + 2*10 + 1*1 + 0*10 = 40+3+20+1+0 = 64
    Or-opt target [0,3,4,1,2,0]: y1 = 4*d[0,3] + 3*d[3,4] + 2*d[4,1] + 1*d[1,2] + 0*d[2,0]
                                     = 4*1 + 3*1 + 2*1 + 1*1 + 0*1 = 4+3+2+1+0 = 10
    """
    sets_ = {"clients": [1, 2, 3, 4], "T": [1]}
    d = defaultdict(lambda: 1000.0)
    d.update({
        (0, 3): 1.0, (3, 4): 1.0, (4, 1): 1.0, (1, 2): 1.0, (2, 0): 1.0,
        (0, 1): 10.0, (2, 3): 10.0, (4, 0): 10.0,
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
        "x": {(0, 1, 1, 1): 1, (1, 2, 1, 1): 1, (2, 3, 1, 1): 1, (3, 4, 1, 1): 1, (4, 0, 1, 1): 1},
        "f": {(0, 1, 1, 1): 4, (1, 2, 1, 1): 3, (2, 3, 1, 1): 2, (3, 4, 1, 1): 1, (4, 0, 1, 1): 0},
        "depot_stock": {1: {"frigo": 0.0, "nonfrigo": 0.0}},
        "arrival_times": {(1, 1): 10.0, (2, 1): 11.0, (3, 1): 21.0, (4, 1): 22.0},
        "truck_assign": {(1, 1): 1, (2, 1): 1, (3, 1): 1, (4, 1): 1},
        "actual_qty": {(1, 1): 1, (2, 1): 1, (3, 1): 1, (4, 1): 1},
        "routes_data": {1: {1: {"path": [0, 1, 2, 3, 4, 0],
                                 "qty": {"1": 1, "2": 1, "3": 1, "4": 1}}}},
        "tau_return": {1: 32.0},
    }

    repaired_two_opt_only = _repair_route_result(route_result, sets_, params_, use_or_opt=False,
                                                  use_two_opt=True, use_delivery_shift=False)
    assert repaired_two_opt_only["routes_data"][1][1]["path"] == [0, 1, 2, 3, 4, 0]

    repaired_with_or_opt = _repair_route_result(route_result, sets_, params_, use_or_opt=True,
                                                 use_two_opt=True, use_delivery_shift=False)
    assert repaired_with_or_opt["routes_data"][1][1]["path"] == [0, 3, 4, 1, 2, 0]
    assert compute_f1(repaired_with_or_opt, sets_, params_) < compute_f1(route_result, sets_, params_)


def test_repair_route_result_or_opt_default_false_matches_two_opt_only():
    """use_or_opt is a keyword-only-by-convention parameter defaulting to
    False -- calling without it at all must behave identically to
    explicitly passing False, on the same route this file's other 2-opt
    tests already use."""
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

    default_call    = _repair_route_result(route_result, sets_, params_,
                                            use_two_opt=True, use_delivery_shift=False)
    explicit_false  = _repair_route_result(route_result, sets_, params_, use_or_opt=False,
                                            use_two_opt=True, use_delivery_shift=False)
    assert default_call["routes_data"][1][1]["path"] == explicit_false["routes_data"][1][1]["path"]


def test_repair_route_result_single_relocation_finds_valid_improvement():
    """6-node route (clients 1-5, one unit each), distances rigged so
    relocating client 3 alone to just before the final depot strictly
    improves f1 (5010 -> 15). Verified numerically (script, not shown
    here) that a SINGLE best-improvement step of pure 2-opt only reaches
    3012 (an improvement, but not this one) -- use_single_relocation=True
    must reach the lower-cost target within the iteration budget.

    Unlike the 2-/3-client Or-opt case, plain 2-opt (use_or_opt=False,
    use_single_relocation=False), given its full iteration budget, was
    independently verified to ALSO eventually reach this same final path
    on this particular example, via a sequence of smaller reversals -- so
    this test only asserts correctness (the flag finds and applies a
    real, valid, lower-cost move), not that 2-opt could never reach it;
    see _repair_route_result's use_single_relocation docstring note.

    Hand-verified target [0,1,2,4,5,3,0]: y1 = 5*d[0,1] + 4*d[1,2] + 3*d[2,4]
                                              + 2*d[4,5] + 1*d[5,3] + 0*d[3,0]
                                             = 5+4+3+2+1+0 = 15
    """
    sets_ = {"clients": [1, 2, 3, 4, 5], "T": [1]}
    d = defaultdict(lambda: 1000.0)
    d.update({
        (0, 1): 1.0, (1, 2): 1.0, (2, 4): 1.0, (4, 5): 1.0, (5, 3): 1.0, (3, 0): 1.0,
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
        "x": {(0, 1, 1, 1): 1, (1, 2, 1, 1): 1, (2, 3, 1, 1): 1, (3, 4, 1, 1): 1, (4, 5, 1, 1): 1, (5, 0, 1, 1): 1},
        "f": {(0, 1, 1, 1): 5, (1, 2, 1, 1): 4, (2, 3, 1, 1): 3, (3, 4, 1, 1): 2, (4, 5, 1, 1): 1, (5, 0, 1, 1): 0},
        "depot_stock": {1: {"frigo": 0.0, "nonfrigo": 0.0}},
        "arrival_times": {(1, 1): 1.0, (2, 1): 1001.0, (3, 1): 2001.0, (4, 1): 3001.0, (5, 1): 3002.0},
        "truck_assign": {(1, 1): 1, (2, 1): 1, (3, 1): 1, (4, 1): 1, (5, 1): 1},
        "actual_qty": {(1, 1): 1, (2, 1): 1, (3, 1): 1, (4, 1): 1, (5, 1): 1},
        "routes_data": {1: {1: {"path": [0, 1, 2, 3, 4, 5, 0],
                                 "qty": {"1": 1, "2": 1, "3": 1, "4": 1, "5": 1}}}},
        "tau_return": {1: 1e9},
    }

    repaired = _repair_route_result(route_result, sets_, params_,
                                     use_or_opt=False, use_single_relocation=True,
                                     use_two_opt=True, use_delivery_shift=False)
    assert repaired["routes_data"][1][1]["path"] == [0, 1, 2, 4, 5, 3, 0]
    assert compute_f1(repaired, sets_, params_) < compute_f1(route_result, sets_, params_)


def test_repair_route_result_single_relocation_default_false_matches_two_opt_only():
    """use_single_relocation is a keyword-only-by-convention parameter
    defaulting to False -- calling without it at all must behave
    identically to explicitly passing False, on the same route this
    file's other 2-opt tests already use."""
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

    default_call   = _repair_route_result(route_result, sets_, params_,
                                           use_two_opt=True, use_delivery_shift=False)
    explicit_false = _repair_route_result(route_result, sets_, params_, use_single_relocation=False,
                                           use_two_opt=True, use_delivery_shift=False)
    assert default_call["routes_data"][1][1]["path"] == explicit_false["routes_data"][1][1]["path"]


# ── _route_f1_contribution ───────────────────────────────────────────────
# Performance fix: _repair_route_result originally called compute_f1 on a
# full scratch route_result per candidate (correct, but O(whole network)
# per candidate -- measured ~43x slower than baseline on the real IRP,
# sensitivity/route_repair_timing_check_iter5.txt, instance 100, gen=10).
# _route_f1_contribution computes
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

    repaired = _repair_route_result(route_result, sets_, params_,
                                     use_two_opt=True, use_delivery_shift=False)
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


# ── _evaluate_candidate ──────────────────────────────────────────────────
# Performance fix: _repair_route_result originally called
# _route_traversal_time, then _rebuild_route_arcs, then _route_f1_contribution
# separately per candidate -- correct, but walking the path 5 times total
# (1 + 3 internal + 1) when 3 passes (matching _rebuild_route_arcs's own
# count) suffice. _evaluate_candidate merges all three into one pass with
# an incremental tau_return short-circuit. These tests prove it produces
# EXACTLY what the three separate calls would have, on both an accepted
# and a guard-rejected candidate -- reusing the exact fixtures from
# test_repair_route_result_improves_f1_via_two_opt_swap and
# test_repair_route_result_rejects_swap_that_would_increase_tau_return.

def test_evaluate_candidate_matches_separate_calls_on_accepted_candidate():
    """Same fixture as test_repair_route_result_improves_f1_via_two_opt_swap:
    candidate [0,2,1,3,0] passes the tau_return guard (time=4.0 <= 12.0) and
    strictly improves f1 (55 < 305). _evaluate_candidate must return the
    same (x_vars, f_vars, arrivals) _rebuild_route_arcs would, plus a
    contribution equal to what _route_f1_contribution would compute from
    those same values."""
    t, k = 1, 1
    qty_on_route = {1: 10, 2: 20, 3: 5}
    tau_return_before = 12.0
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
    candidate = [0, 2, 1, 3, 0]

    reference_x, reference_f, reference_arrivals = _rebuild_route_arcs(
        candidate, qty_on_route, t, k, params_
    )
    reference_contrib = _route_f1_contribution(
        candidate, reference_f, reference_arrivals, t, k, params_
    )

    result = _evaluate_candidate(candidate, qty_on_route, t, k, tau_return_before, params_)

    assert result is not None
    x_vars, f_vars, arrivals, contrib = result
    assert x_vars == reference_x
    assert f_vars == reference_f
    assert arrivals == reference_arrivals
    assert contrib == reference_contrib
    assert contrib == 55.0   # matches the hand-verified y1 from the reused fixture


def test_evaluate_candidate_returns_none_when_tau_return_guard_fails():
    """Same fixture as test_repair_route_result_rejects_swap_that_would_
    increase_tau_return: the only candidate's traversal time (10.002) far
    exceeds tau_return_before (3.0) -- _route_traversal_time would compute
    the same 10.002 if called separately; _evaluate_candidate must reject
    via the same threshold without needing that separate call, returning
    None before ever computing x_vars/f_vars/contribution."""
    t, k = 1, 1
    qty_on_route = {1: 100, 2: 1}
    tau_return_before = 3.0
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
    candidate = [0, 2, 1, 0]

    reference_time = _route_traversal_time(candidate, k, params_)
    assert reference_time > tau_return_before   # sanity: the reference call agrees it should be rejected

    result = _evaluate_candidate(candidate, qty_on_route, t, k, tau_return_before, params_)

    assert result is None


# ── _inter_route_relocate_candidates ─────────────────────────────────────

def test_inter_route_relocate_candidates_hand_verified():
    """2 trucks, both frigo, period 1: truck 1 carries clients 1 (qty 5) and
    2 (qty 3), truck 2 carries client 3 (qty 2). Capacity (Q=20 each) is
    never a constraint here. Hand-traced: each of the 3 clients can move to
    the OTHER truck, inserted at every interior position of that truck's
    path (2 positions for the 2-node target paths, 3 for the 3-node one)."""
    routes = {
        1: {"path": [0, 1, 2, 0], "qty": {"1": 5, "2": 3}},
        2: {"path": [0, 3, 0],    "qty": {"3": 2}},
    }
    requires_cold = defaultdict(lambda: True)
    frigo_trucks  = {1, 2}
    Q             = {1: 20, 2: 20}

    candidates = list(_inter_route_relocate_candidates(routes, requires_cold, frigo_trucks, Q, 1))

    assert candidates == [
        (1, 1, 2, [0, 2, 0], [0, 1, 3, 0], 5),
        (1, 1, 2, [0, 2, 0], [0, 3, 1, 0], 5),
        (1, 2, 2, [0, 1, 0], [0, 2, 3, 0], 3),
        (1, 2, 2, [0, 1, 0], [0, 3, 2, 0], 3),
        (2, 3, 1, [0, 0],    [0, 3, 1, 2, 0], 2),
        (2, 3, 1, [0, 0],    [0, 1, 3, 2, 0], 2),
        (2, 3, 1, [0, 0],    [0, 1, 2, 3, 0], 2),
    ]


def test_inter_route_relocate_candidates_blocks_over_capacity_target():
    """Client's quantity would push the target truck over its capacity --
    that (k_from, client, k_to, ...) combination must never be yielded,
    regardless of how cheap the resulting route would look distance-wise."""
    routes = {
        1: {"path": [0, 1, 0], "qty": {"1": 5}},
        2: {"path": [0, 2, 0], "qty": {"2": 17}},   # 17 + 5 = 22 > Q[2] = 20
    }
    requires_cold = defaultdict(lambda: True)
    frigo_trucks  = {1, 2}
    Q             = {1: 20, 2: 20}

    candidates = list(_inter_route_relocate_candidates(routes, requires_cold, frigo_trucks, Q, 1))

    # client 1 (qty 5) -> truck 2 (already 17/20): 17+5=22 > Q[2] -- blocked.
    # client 2 (qty 17) -> truck 1 (already 5/20): 5+17=22 > Q[1] -- blocked too.
    # so no candidates at all should exist between these two near-full trucks.
    assert candidates == []


def test_inter_route_relocate_candidates_blocks_frigo_nonfrigo_mismatch():
    """A client whose period-t demand requires cold storage can never be
    yielded as relocatable onto a non-frigo truck, and vice versa -- even
    when capacity would otherwise allow it."""
    routes = {
        1: {"path": [0, 1, 0], "qty": {"1": 1}},   # truck 1: frigo
        2: {"path": [0, 2, 0], "qty": {"2": 1}},   # truck 2: non-frigo
    }
    requires_cold = {(1, 1): True, (2, 1): False}
    frigo_trucks  = {1}
    Q             = {1: 100, 2: 100}

    candidates = list(_inter_route_relocate_candidates(routes, requires_cold, frigo_trucks, Q, 1))

    assert candidates == []


# ── _repair_inter_route_relocate ─────────────────────────────────────────

def test_repair_inter_route_relocate_finds_improvement_2opt_alone_cannot():
    """2 trucks, both frigo, period 1: truck 1 = [0,1,2,0] (clients 1,2, qty
    1 each), truck 2 = [0,3,0] (client 3, qty 1). Distances rigged so moving
    client 2 from truck 1 to truck 2 (inserted after 3) strictly improves
    combined f1 -- a move NO intra-route neighbourhood (2-opt, Or-opt,
    single-relocation) could ever find, since it changes which TRUCK serves
    a client, not just the visit order within one truck's route.

    Hand-verified (suffix-sum flow weights, qty=1 each):
      original: truck1 y1 = 2*d[0,1] + 1*d[1,2] + 0*d[2,0] = 2+1000+0 = 1002
                truck2 y1 = 1*d[0,3] + 0*d[3,0]             = 1+0     = 1
                combined = 1003
      target:   truck1 [0,1,0]:   y1 = 1*d[0,1] + 0*d[1,0]           = 1
                truck2 [0,3,2,0]: y1 = 2*d[0,3]+1*d[3,2]+0*d[2,0]    = 2+1+0 = 3
                combined = 4
    """
    sets_ = {"clients": [1, 2, 3], "T": [1]}
    d = defaultdict(lambda: 1000.0)
    d.update({(0, 1): 1.0, (1, 0): 1.0, (0, 3): 1.0, (3, 2): 1.0, (2, 0): 1.0})
    params_ = {
        "d": d,
        "v": {1: 1.0, 2: 1.0},
        "s": {},
        "c_ijk": defaultdict(lambda: 1.0),
        "h_O": 0.0, "c1": 0.0, "c2": 0.0,
        "ET": defaultdict(lambda: 0.0),
        "LT": defaultdict(lambda: 1e9),
        "requires_cold": defaultdict(lambda: True),
        "frigo_trucks":  {1, 2},
        "Q":             {1: 10, 2: 10},
    }
    route_result = {
        "x": {(0, 1, 1, 1): 1, (1, 2, 1, 1): 1, (2, 0, 1, 1): 1,
              (0, 3, 1, 2): 1, (3, 0, 1, 2): 1},
        "f": {(0, 1, 1, 1): 2, (1, 2, 1, 1): 1, (2, 0, 1, 1): 0,
              (0, 3, 1, 2): 1, (3, 0, 1, 2): 0},
        "depot_stock": {1: {"frigo": 0.0, "nonfrigo": 0.0}},
        "arrival_times": {(1, 1): 1.0, (2, 1): 2.0, (3, 1): 1.0},
        "truck_assign": {(1, 1): 1, (2, 1): 1, (3, 1): 2},
        "actual_qty": {(1, 1): 1, (2, 1): 1, (3, 1): 1},
        "routes_data": {1: {1: {"path": [0, 1, 2, 0], "qty": {"1": 1, "2": 1}},
                            2: {"path": [0, 3, 0],    "qty": {"3": 1}}}},
        "tau_return": {1: 1e9},
    }

    orig_f1 = compute_f1(route_result, sets_, params_)
    assert orig_f1 == 1003.0

    repaired = _repair_inter_route_relocate(route_result, sets_, params_)

    assert repaired["routes_data"][1][1]["path"] == [0, 1, 0]
    assert repaired["routes_data"][1][2]["path"] == [0, 3, 2, 0]
    assert repaired["truck_assign"][2, 1] == 2
    assert compute_f1(repaired, sets_, params_) == 4.0


def test_repair_inter_route_relocate_single_route_period_is_noop():
    """Only one truck active in the period -- nothing to relocate to, must
    return the route unchanged (not raise, not loop forever)."""
    sets_ = {"clients": [1], "T": [1]}
    params_ = {
        "d": defaultdict(lambda: 1.0),
        "v": {1: 1.0}, "s": {},
        "c_ijk": defaultdict(lambda: 1.0),
        "h_O": 0.0, "c1": 0.0, "c2": 0.0,
        "ET": defaultdict(lambda: 0.0), "LT": defaultdict(lambda: 1e9),
        "requires_cold": defaultdict(lambda: True),
        "frigo_trucks": {1},
        "Q": {1: 10},
    }
    route_result = {
        "x": {(0, 1, 1, 1): 1, (1, 0, 1, 1): 1},
        "f": {(0, 1, 1, 1): 1, (1, 0, 1, 1): 0},
        "depot_stock": {1: {"frigo": 0.0, "nonfrigo": 0.0}},
        "arrival_times": {(1, 1): 1.0},
        "truck_assign": {(1, 1): 1},
        "actual_qty": {(1, 1): 1},
        "routes_data": {1: {1: {"path": [0, 1, 0], "qty": {"1": 1}}}},
        "tau_return": {1: 1e9},
    }

    repaired = _repair_inter_route_relocate(route_result, sets_, params_)

    assert repaired["routes_data"][1][1]["path"] == [0, 1, 0]


# ── _repair_route_result(use_inter_route_relocate=...) ──────────────────

def test_repair_route_result_inter_route_relocate_default_false_matches_baseline():
    """use_inter_route_relocate is a keyword-only-by-convention parameter
    defaulting to False -- calling without it must behave identically to
    explicitly passing False, on the same 2-truck route this file's own
    inter-route test uses."""
    sets_ = {"clients": [1, 2, 3], "T": [1]}
    d = defaultdict(lambda: 1000.0)
    d.update({(0, 1): 1.0, (1, 0): 1.0, (0, 3): 1.0, (3, 2): 1.0, (2, 0): 1.0})
    params_ = {
        "d": d,
        "v": {1: 1.0, 2: 1.0},
        "s": {},
        "c_ijk": defaultdict(lambda: 1.0),
        "h_O": 0.0, "c1": 0.0, "c2": 0.0,
        "ET": defaultdict(lambda: 0.0),
        "LT": defaultdict(lambda: 1e9),
        "requires_cold": defaultdict(lambda: True),
        "frigo_trucks":  {1, 2},
        "Q":             {1: 10, 2: 10},
    }
    route_result = {
        "x": {(0, 1, 1, 1): 1, (1, 2, 1, 1): 1, (2, 0, 1, 1): 1,
              (0, 3, 1, 2): 1, (3, 0, 1, 2): 1},
        "f": {(0, 1, 1, 1): 2, (1, 2, 1, 1): 1, (2, 0, 1, 1): 0,
              (0, 3, 1, 2): 1, (3, 0, 1, 2): 0},
        "depot_stock": {1: {"frigo": 0.0, "nonfrigo": 0.0}},
        "arrival_times": {(1, 1): 1.0, (2, 1): 2.0, (3, 1): 1.0},
        "truck_assign": {(1, 1): 1, (2, 1): 1, (3, 1): 2},
        "actual_qty": {(1, 1): 1, (2, 1): 1, (3, 1): 1},
        "routes_data": {1: {1: {"path": [0, 1, 2, 0], "qty": {"1": 1, "2": 1}},
                            2: {"path": [0, 3, 0],    "qty": {"3": 1}}}},
        "tau_return": {1: 1e9},
    }

    default_call   = _repair_route_result(route_result, sets_, params_, use_delivery_shift=False)
    explicit_false = _repair_route_result(route_result, sets_, params_, use_inter_route_relocate=False,
                                           use_delivery_shift=False)
    assert default_call["routes_data"][1] == explicit_false["routes_data"][1]


def test_repair_route_result_inter_route_relocate_true_applies_the_move():
    """Same rigged route as test_repair_inter_route_relocate_finds_improvement
    -- with use_inter_route_relocate=True passed through the main entry
    point (_repair_route_result), the inter-route move must be found and
    applied before the (here irrelevant) intra-route loop runs."""
    sets_ = {"clients": [1, 2, 3], "T": [1]}
    d = defaultdict(lambda: 1000.0)
    d.update({(0, 1): 1.0, (1, 0): 1.0, (0, 3): 1.0, (3, 2): 1.0, (2, 0): 1.0})
    params_ = {
        "d": d,
        "v": {1: 1.0, 2: 1.0},
        "s": {},
        "c_ijk": defaultdict(lambda: 1.0),
        "h_O": 0.0, "c1": 0.0, "c2": 0.0,
        "ET": defaultdict(lambda: 0.0),
        "LT": defaultdict(lambda: 1e9),
        "requires_cold": defaultdict(lambda: True),
        "frigo_trucks":  {1, 2},
        "Q":             {1: 10, 2: 10},
    }
    route_result = {
        "x": {(0, 1, 1, 1): 1, (1, 2, 1, 1): 1, (2, 0, 1, 1): 1,
              (0, 3, 1, 2): 1, (3, 0, 1, 2): 1},
        "f": {(0, 1, 1, 1): 2, (1, 2, 1, 1): 1, (2, 0, 1, 1): 0,
              (0, 3, 1, 2): 1, (3, 0, 1, 2): 0},
        "depot_stock": {1: {"frigo": 0.0, "nonfrigo": 0.0}},
        "arrival_times": {(1, 1): 1.0, (2, 1): 2.0, (3, 1): 1.0},
        "truck_assign": {(1, 1): 1, (2, 1): 1, (3, 1): 2},
        "actual_qty": {(1, 1): 1, (2, 1): 1, (3, 1): 1},
        "routes_data": {1: {1: {"path": [0, 1, 2, 0], "qty": {"1": 1, "2": 1}},
                            2: {"path": [0, 3, 0],    "qty": {"3": 1}}}},
        "tau_return": {1: 1e9},
    }

    repaired = _repair_route_result(route_result, sets_, params_, use_inter_route_relocate=True,
                                     use_delivery_shift=False)

    assert repaired["routes_data"][1][1]["path"] == [0, 1, 0]
    assert repaired["routes_data"][1][2]["path"] == [0, 3, 2, 0]
    assert compute_f1(repaired, sets_, params_) == 4.0


# ── _route_swap_candidates ───────────────────────────────────────────────

def test_route_swap_candidates_hand_verified():
    """2 trucks, both frigo, period 1: truck 1 = clients 1, 2 (qty 1 each),
    truck 2 = client 3 (qty 1). Capacity (Q=10 each) never binds. Hand-traced:
    swapping client 1 (truck 1) <-> client 3 (truck 2) puts 3 at 1's exact
    slot and 1 at 3's slot; swapping client 2 <-> client 3 likewise -- 2
    candidates total, one per client in truck 1's route paired with the
    (only) client in truck 2's route."""
    routes = {
        1: {"path": [0, 1, 2, 0], "qty": {"1": 1, "2": 1}},
        2: {"path": [0, 3, 0],    "qty": {"3": 1}},
    }
    requires_cold = defaultdict(lambda: True)
    frigo_trucks  = {1, 2}
    Q             = {1: 10, 2: 10}

    candidates = list(_route_swap_candidates(routes, requires_cold, frigo_trucks, Q, 1))

    assert candidates == [
        (1, 1, 2, 3, [0, 3, 2, 0], [0, 1, 0], 1, 1),
        (1, 2, 2, 3, [0, 1, 3, 0], [0, 2, 0], 1, 1),
    ]


def test_route_swap_candidates_visits_each_route_pair_once():
    """3 trucks -- the (k1, k2) route pair must be visited in one direction
    only (k2 > k1 by truck id), never both (1,2) and (2,1), since a swap is
    symmetric and visiting both would just yield the same logical moves
    twice with the tuple order flipped."""
    routes = {
        1: {"path": [0, 1, 0], "qty": {"1": 1}},
        2: {"path": [0, 2, 0], "qty": {"2": 1}},
        3: {"path": [0, 3, 0], "qty": {"3": 1}},
    }
    requires_cold = defaultdict(lambda: True)
    frigo_trucks  = {1, 2, 3}
    Q             = {1: 10, 2: 10, 3: 10}

    candidates = list(_route_swap_candidates(routes, requires_cold, frigo_trucks, Q, 1))
    pairs = [(c[0], c[2]) for c in candidates]

    assert pairs == [(1, 2), (1, 3), (2, 3)]   # each unordered pair exactly once, k1 < k2


def test_route_swap_candidates_blocks_over_capacity():
    """Swapping equal quantities never changes either truck's load, so this
    uses DELIBERATELY UNEQUAL quantities: truck 1 carries only client 1
    (qty 1, Q[1]=10); truck 2 carries only client 2 (qty 15, Q[2]=20 -- 15
    fits truck 2 itself, but swapping it onto truck 1 would need a load of
    15 there, over Q[1]=10) -- blocked, so no candidates at all."""
    routes = {
        1: {"path": [0, 1, 0], "qty": {"1": 1}},
        2: {"path": [0, 2, 0], "qty": {"2": 15}},
    }
    requires_cold = defaultdict(lambda: True)
    frigo_trucks  = {1, 2}
    Q             = {1: 10, 2: 20}

    candidates = list(_route_swap_candidates(routes, requires_cold, frigo_trucks, Q, 1))

    assert candidates == []


def test_route_swap_candidates_blocks_frigo_nonfrigo_mismatch():
    """Client 1 (frigo-required) can never swap onto non-frigo truck 2, and
    client 2 (non-frigo) can never swap onto frigo truck 1 -- even though
    capacity would allow it either way."""
    routes = {
        1: {"path": [0, 1, 0], "qty": {"1": 1}},   # truck 1: frigo
        2: {"path": [0, 2, 0], "qty": {"2": 1}},   # truck 2: non-frigo
    }
    requires_cold = {(1, 1): True, (2, 1): False}
    frigo_trucks  = {1}
    Q             = {1: 100, 2: 100}

    candidates = list(_route_swap_candidates(routes, requires_cold, frigo_trucks, Q, 1))

    assert candidates == []


# ── _repair_route_swap ───────────────────────────────────────────────────

def test_repair_route_swap_finds_improvement():
    """2 trucks, both frigo, period 1: truck 1 = [0,1,2,0] (clients 1,2, qty
    1 each), truck 2 = [0,3,0] (client 3, qty 1). Distances rigged so
    swapping clients 2 and 3 (each taking the other's exact slot) strictly
    improves combined f1 -- a move neither pure relocate (which searches
    insertion positions, not slot-preserving exchanges) nor any intra-route
    neighbourhood could replicate in one atomic step.

    Hand-verified (suffix-sum flow weights, qty=1 each):
      original: truck1 [0,1,2,0] y1 = 2*d[0,1]+1*d[1,2]+0*d[2,0] = 2+1000+0=1002
                truck2 [0,3,0]   y1 = 1*d[0,3]+0*d[3,0]           = 1000+0 =1000
                combined = 2002
      target:   truck1 [0,1,3,0] y1 = 2*d[0,1]+1*d[1,3]+0*d[3,0] = 2+1+0=3
                truck2 [0,2,0]   y1 = 1*d[0,2]+0*d[2,0]           = 1+0  =1
                combined = 4
    """
    sets_ = {"clients": [1, 2, 3], "T": [1]}
    d = defaultdict(lambda: 1000.0)
    d.update({(0, 1): 1.0, (1, 3): 1.0, (3, 0): 1.0, (0, 2): 1.0, (2, 0): 1.0})
    params_ = {
        "d": d,
        "v": {1: 1.0, 2: 1.0},
        "s": {},
        "c_ijk": defaultdict(lambda: 1.0),
        "h_O": 0.0, "c1": 0.0, "c2": 0.0,
        "ET": defaultdict(lambda: 0.0),
        "LT": defaultdict(lambda: 1e9),
        "requires_cold": defaultdict(lambda: True),
        "frigo_trucks":  {1, 2},
        "Q":             {1: 10, 2: 10},
    }
    route_result = {
        "x": {(0, 1, 1, 1): 1, (1, 2, 1, 1): 1, (2, 0, 1, 1): 1,
              (0, 3, 1, 2): 1, (3, 0, 1, 2): 1},
        "f": {(0, 1, 1, 1): 2, (1, 2, 1, 1): 1, (2, 0, 1, 1): 0,
              (0, 3, 1, 2): 1, (3, 0, 1, 2): 0},
        "depot_stock": {1: {"frigo": 0.0, "nonfrigo": 0.0}},
        "arrival_times": {(1, 1): 1.0, (2, 1): 2.0, (3, 1): 1.0},
        "truck_assign": {(1, 1): 1, (2, 1): 1, (3, 1): 2},
        "actual_qty": {(1, 1): 1, (2, 1): 1, (3, 1): 1},
        "routes_data": {1: {1: {"path": [0, 1, 2, 0], "qty": {"1": 1, "2": 1}},
                            2: {"path": [0, 3, 0],    "qty": {"3": 1}}}},
        "tau_return": {1: 1e9},
    }

    orig_f1 = compute_f1(route_result, sets_, params_)
    assert orig_f1 == 2002.0

    repaired = _repair_route_swap(route_result, sets_, params_)

    assert repaired["routes_data"][1][1]["path"] == [0, 1, 3, 0]
    assert repaired["routes_data"][1][2]["path"] == [0, 2, 0]
    assert repaired["truck_assign"][2, 1] == 2
    assert repaired["truck_assign"][3, 1] == 1
    assert compute_f1(repaired, sets_, params_) == 4.0


def test_repair_route_swap_single_route_period_is_noop():
    """Only one truck active in the period -- nothing to swap with, must
    return the route unchanged (not raise, not loop forever)."""
    sets_ = {"clients": [1], "T": [1]}
    params_ = {
        "d": defaultdict(lambda: 1.0),
        "v": {1: 1.0}, "s": {},
        "c_ijk": defaultdict(lambda: 1.0),
        "h_O": 0.0, "c1": 0.0, "c2": 0.0,
        "ET": defaultdict(lambda: 0.0), "LT": defaultdict(lambda: 1e9),
        "requires_cold": defaultdict(lambda: True),
        "frigo_trucks": {1},
        "Q": {1: 10},
    }
    route_result = {
        "x": {(0, 1, 1, 1): 1, (1, 0, 1, 1): 1},
        "f": {(0, 1, 1, 1): 1, (1, 0, 1, 1): 0},
        "depot_stock": {1: {"frigo": 0.0, "nonfrigo": 0.0}},
        "arrival_times": {(1, 1): 1.0},
        "truck_assign": {(1, 1): 1},
        "actual_qty": {(1, 1): 1},
        "routes_data": {1: {1: {"path": [0, 1, 0], "qty": {"1": 1}}}},
        "tau_return": {1: 1e9},
    }

    repaired = _repair_route_swap(route_result, sets_, params_)

    assert repaired["routes_data"][1][1]["path"] == [0, 1, 0]


# ── _repair_route_result(use_route_swap=...) ─────────────────────────────

def test_repair_route_result_route_swap_default_false_matches_baseline():
    """use_route_swap is a keyword-only-by-convention parameter defaulting
    to False -- calling without it must behave identically to explicitly
    passing False, on the same 2-truck route this file's own swap test
    uses."""
    sets_ = {"clients": [1, 2, 3], "T": [1]}
    d = defaultdict(lambda: 1000.0)
    d.update({(0, 1): 1.0, (1, 3): 1.0, (3, 0): 1.0, (0, 2): 1.0, (2, 0): 1.0})
    params_ = {
        "d": d,
        "v": {1: 1.0, 2: 1.0},
        "s": {},
        "c_ijk": defaultdict(lambda: 1.0),
        "h_O": 0.0, "c1": 0.0, "c2": 0.0,
        "ET": defaultdict(lambda: 0.0),
        "LT": defaultdict(lambda: 1e9),
        "requires_cold": defaultdict(lambda: True),
        "frigo_trucks":  {1, 2},
        "Q":             {1: 10, 2: 10},
    }
    route_result = {
        "x": {(0, 1, 1, 1): 1, (1, 2, 1, 1): 1, (2, 0, 1, 1): 1,
              (0, 3, 1, 2): 1, (3, 0, 1, 2): 1},
        "f": {(0, 1, 1, 1): 2, (1, 2, 1, 1): 1, (2, 0, 1, 1): 0,
              (0, 3, 1, 2): 1, (3, 0, 1, 2): 0},
        "depot_stock": {1: {"frigo": 0.0, "nonfrigo": 0.0}},
        "arrival_times": {(1, 1): 1.0, (2, 1): 2.0, (3, 1): 1.0},
        "truck_assign": {(1, 1): 1, (2, 1): 1, (3, 1): 2},
        "actual_qty": {(1, 1): 1, (2, 1): 1, (3, 1): 1},
        "routes_data": {1: {1: {"path": [0, 1, 2, 0], "qty": {"1": 1, "2": 1}},
                            2: {"path": [0, 3, 0],    "qty": {"3": 1}}}},
        "tau_return": {1: 1e9},
    }

    default_call   = _repair_route_result(route_result, sets_, params_, use_delivery_shift=False)
    explicit_false = _repair_route_result(route_result, sets_, params_, use_route_swap=False,
                                           use_delivery_shift=False)
    assert default_call["routes_data"][1] == explicit_false["routes_data"][1]


def test_repair_route_result_route_swap_true_applies_the_move():
    """Same rigged route as test_repair_route_swap_finds_improvement -- with
    use_route_swap=True passed through the main entry point
    (_repair_route_result), the swap must be found and applied."""
    sets_ = {"clients": [1, 2, 3], "T": [1]}
    d = defaultdict(lambda: 1000.0)
    d.update({(0, 1): 1.0, (1, 3): 1.0, (3, 0): 1.0, (0, 2): 1.0, (2, 0): 1.0})
    params_ = {
        "d": d,
        "v": {1: 1.0, 2: 1.0},
        "s": {},
        "c_ijk": defaultdict(lambda: 1.0),
        "h_O": 0.0, "c1": 0.0, "c2": 0.0,
        "ET": defaultdict(lambda: 0.0),
        "LT": defaultdict(lambda: 1e9),
        "requires_cold": defaultdict(lambda: True),
        "frigo_trucks":  {1, 2},
        "Q":             {1: 10, 2: 10},
    }
    route_result = {
        "x": {(0, 1, 1, 1): 1, (1, 2, 1, 1): 1, (2, 0, 1, 1): 1,
              (0, 3, 1, 2): 1, (3, 0, 1, 2): 1},
        "f": {(0, 1, 1, 1): 2, (1, 2, 1, 1): 1, (2, 0, 1, 1): 0,
              (0, 3, 1, 2): 1, (3, 0, 1, 2): 0},
        "depot_stock": {1: {"frigo": 0.0, "nonfrigo": 0.0}},
        "arrival_times": {(1, 1): 1.0, (2, 1): 2.0, (3, 1): 1.0},
        "truck_assign": {(1, 1): 1, (2, 1): 1, (3, 1): 2},
        "actual_qty": {(1, 1): 1, (2, 1): 1, (3, 1): 1},
        "routes_data": {1: {1: {"path": [0, 1, 2, 0], "qty": {"1": 1, "2": 1}},
                            2: {"path": [0, 3, 0],    "qty": {"3": 1}}}},
        "tau_return": {1: 1e9},
    }

    repaired = _repair_route_result(route_result, sets_, params_, use_route_swap=True,
                                     use_delivery_shift=False)

    assert repaired["routes_data"][1][1]["path"] == [0, 1, 3, 0]
    assert repaired["routes_data"][1][2]["path"] == [0, 2, 0]
    assert compute_f1(repaired, sets_, params_) == 4.0


# ── _delivery_shift_candidates / _repair_delivery_shift ──────────────────

def test_delivery_shift_candidates_hand_verified():
    """Client 1 served at t=1 (qty=5, truck 1) and t=2 (qty=5, truck 2),
    both frigo. q_lt (nominal per-period demand) = 3 at each period. Q=10
    both trucks (load=qty=5, headroom=5 each). I_O_max_frigo=10,
    I_O_min_frigo=0, depot_stock[1]["frigo"]=2.0.

    Case A (delay, donor=t1/truck1, receiver=t2/truck2), I[t1] += delta:
      min(headroom=Q[2]-load2=10-5=5, floor=q1-q_lt[1,1]=5-3=2,
          stock=I_max-I[t1]=10-2=8, q1-1=4) = 2
    Case B (advance, donor=t2/truck2, receiver=t1/truck1), I[t1] -= delta:
      min(headroom=Q[1]-load1=10-5=5, floor=q2-q_lt[1,2]=5-3=2,
          stock=I[t1]-I_min=2-0=2, q2-1=4) = 2
    """
    sets_ = {"clients": [1], "T": [1, 2]}
    params_ = {
        "requires_cold": defaultdict(lambda: True),
        "q_lt": {(1, 1): 3, (1, 2): 3},
        "Q": {1: 10, 2: 10},
        "I_O_min_frigo": 0, "I_O_max_frigo": 10,
        "I_O_min_nonfrigo": 0, "I_O_max_nonfrigo": 10,
    }
    route_result = {
        "actual_qty": {(1, 1): 5, (1, 2): 5},
        "depot_stock": {1: {"frigo": 2.0, "nonfrigo": 0.0}, 2: {"frigo": 0.0, "nonfrigo": 0.0}},
        "routes_data": {
            1: {1: {"path": [0, 1, 0], "qty": {"1": 5}}},
            2: {2: {"path": [0, 1, 0], "qty": {"1": 5}}},
        },
    }

    candidates = list(_delivery_shift_candidates(route_result, sets_, params_))
    assert len(candidates) == 2
    assert (1, 1, 2, 1, 2, 1, +1, 2) in candidates   # case A
    assert (1, 2, 1, 2, 1, 1, -1, 2) in candidates   # case B


def test_delivery_shift_candidates_blocks_frigo_nonfrigo_mismatch():
    """Client served both periods, but requires_cold differs between them
    (a real early-dispatch scenario where the coldness label itself changes
    period to period) -- out of scope for this move (would need a
    different truck TYPE, not just a different truck), must yield nothing."""
    sets_ = {"clients": [1], "T": [1, 2]}
    params_ = {
        "requires_cold": {(1, 1): True, (1, 2): False},
        "q_lt": {(1, 1): 1, (1, 2): 1},
        "Q": {1: 10},
        "I_O_min_frigo": 0, "I_O_max_frigo": 10,
        "I_O_min_nonfrigo": 0, "I_O_max_nonfrigo": 10,
    }
    route_result = {
        "actual_qty": {(1, 1): 5, (1, 2): 5},
        "depot_stock": {1: {"frigo": 2.0, "nonfrigo": 0.0}, 2: {"frigo": 0.0, "nonfrigo": 0.0}},
        "routes_data": {
            1: {1: {"path": [0, 1, 0], "qty": {"1": 5}}},
            2: {1: {"path": [0, 1, 0], "qty": {"1": 5}}},
        },
    }
    assert list(_delivery_shift_candidates(route_result, sets_, params_)) == []


def test_delivery_shift_candidates_skips_when_not_served_both_periods():
    """Client served at t=1 only (actual_qty at t=2 is 0) -- no existing
    stop at t=2 to shift quantity to/from, must yield nothing (this move
    never inserts a new stop)."""
    sets_ = {"clients": [1], "T": [1, 2]}
    params_ = {
        "requires_cold": defaultdict(lambda: True),
        "q_lt": {(1, 1): 1, (1, 2): 1},
        "Q": {1: 10},
        "I_O_min_frigo": 0, "I_O_max_frigo": 10,
        "I_O_min_nonfrigo": 0, "I_O_max_nonfrigo": 10,
    }
    route_result = {
        "actual_qty": {(1, 1): 5, (1, 2): 0},
        "depot_stock": {1: {"frigo": 2.0, "nonfrigo": 0.0}, 2: {"frigo": 0.0, "nonfrigo": 0.0}},
        "routes_data": {
            1: {1: {"path": [0, 1, 0], "qty": {"1": 5}}},
            2: {},
        },
    }
    assert list(_delivery_shift_candidates(route_result, sets_, params_)) == []


def test_repair_delivery_shift_finds_improvement_via_holding_cost():
    """Client 1 served at t=1 (qty=3) and t=2 (qty=5), same truck 1, zero
    distance (d=0 everywhere) so transport cost cannot move -- isolates the
    holding-cost mechanism. h_O=1.0, depot_stock[1]["frigo"]=10.0,
    I_O_min_frigo=0 (headroom for advancing), q_lt=1 at both periods (floor
    room), Q=100 (capacity never binds).

    Case B (advance, donor=t2, receiver=t1) is the only improving direction:
    shipping MORE at t1 (earlier) lowers how long that stock sits, so
    holding cost drops. Hand-verified max delta:
      min(headroom=Q[1]-load1=100-3=97, floor=q2-q_lt[1,2]=5-1=4,
          stock=I[t1]-I_min=10-0=10, q2-1=4) = 4
    New actual_qty: t1 -> 3+4=7, t2 -> 5-4=1. depot_stock[1]["frigo"]
    -> 10-4=6.0 (t2's stock, already 0.0, is untouched -- proven invariant
    by the module docstring's cumulative-sum argument).

    f1 = y1 (transport, 0 throughout, d=0) + y2 (holding, h_O*sum(stock))
       + y3 (0, c1=c2=0). Before: y2 = 1.0*(10+0+0+0) = 10.0. After:
    y2 = 1.0*(6+0+0+0) = 6.0. Delta = -4.0, matching sign*delta*h_O =
    -1*4*1.0 = -4.0 exactly (no transport-cost term, since d=0).
    """
    sets_ = {"clients": [1], "T": [1, 2]}
    d = defaultdict(lambda: 0.0)
    params_ = {
        "d": d,
        "v": {1: 1.0},
        "s": {},
        "c_ijk": defaultdict(lambda: 1.0),
        "h_O": 1.0, "c1": 0.0, "c2": 0.0,
        "ET": defaultdict(lambda: 0.0),
        "LT": defaultdict(lambda: 1e9),
        "requires_cold": defaultdict(lambda: True),
        "frigo_trucks":  {1},
        "Q":             {1: 100},
        "q_lt":          {(1, 1): 1, (1, 2): 1},
        "I_O_min_frigo": 0, "I_O_max_frigo": 100,
        "I_O_min_nonfrigo": 0, "I_O_max_nonfrigo": 100,
    }
    route_result = {
        "x": {(0, 1, 1, 1): 1, (1, 0, 1, 1): 1, (0, 1, 2, 1): 1, (1, 0, 2, 1): 1},
        "f": {(0, 1, 1, 1): 3, (1, 0, 1, 1): 0, (0, 1, 2, 1): 5, (1, 0, 2, 1): 0},
        "depot_stock": {1: {"frigo": 10.0, "nonfrigo": 0.0}, 2: {"frigo": 0.0, "nonfrigo": 0.0}},
        "arrival_times": {(1, 1): 0.0, (1, 2): 0.0},
        "truck_assign": {(1, 1): 1, (1, 2): 1},
        "actual_qty": {(1, 1): 3, (1, 2): 5},
        "routes_data": {
            1: {1: {"path": [0, 1, 0], "qty": {"1": 3}}},
            2: {1: {"path": [0, 1, 0], "qty": {"1": 5}}},
        },
        "tau_return": {1: 0.0, 2: 0.0},
    }

    orig_f1 = compute_f1(route_result, sets_, params_)
    assert orig_f1 == 10.0

    repaired = _repair_delivery_shift(route_result, sets_, params_)

    assert repaired["actual_qty"][1, 1] == 7
    assert repaired["actual_qty"][1, 2] == 1
    assert repaired["depot_stock"][1]["frigo"] == 6.0
    assert repaired["depot_stock"][2]["frigo"] == 0.0
    assert repaired["routes_data"][1][1]["qty"] == {"1": 7}
    assert repaired["routes_data"][2][1]["qty"] == {"1": 1}
    assert compute_f1(repaired, sets_, params_) == 6.0


# ── _repair_route_result(use_delivery_shift=...) ─────────────────────────

def test_repair_route_result_delivery_shift_default_true_matches_explicit():
    """use_delivery_shift defaults to True (production default, matching
    the validated delivery-shift-only campaign -- see
    Livrables_Prof/IRP_100clients_RESULTATS_DELIVERYSHIFT.html) -- calling
    without it must behave identically to explicitly passing True, on the
    same rigged route this file's own delivery-shift test uses, and must
    actually apply the shift (not silently no-op)."""
    sets_ = {"clients": [1], "T": [1, 2]}
    d = defaultdict(lambda: 0.0)
    params_ = {
        "d": d,
        "v": {1: 1.0},
        "s": {},
        "c_ijk": defaultdict(lambda: 1.0),
        "h_O": 1.0, "c1": 0.0, "c2": 0.0,
        "ET": defaultdict(lambda: 0.0),
        "LT": defaultdict(lambda: 1e9),
        "requires_cold": defaultdict(lambda: True),
        "frigo_trucks":  {1},
        "Q":             {1: 100},
        "q_lt":          {(1, 1): 1, (1, 2): 1},
        "I_O_min_frigo": 0, "I_O_max_frigo": 100,
        "I_O_min_nonfrigo": 0, "I_O_max_nonfrigo": 100,
    }
    route_result = {
        "x": {(0, 1, 1, 1): 1, (1, 0, 1, 1): 1, (0, 1, 2, 1): 1, (1, 0, 2, 1): 1},
        "f": {(0, 1, 1, 1): 3, (1, 0, 1, 1): 0, (0, 1, 2, 1): 5, (1, 0, 2, 1): 0},
        "depot_stock": {1: {"frigo": 10.0, "nonfrigo": 0.0}, 2: {"frigo": 0.0, "nonfrigo": 0.0}},
        "arrival_times": {(1, 1): 0.0, (1, 2): 0.0},
        "truck_assign": {(1, 1): 1, (1, 2): 1},
        "actual_qty": {(1, 1): 3, (1, 2): 5},
        "routes_data": {
            1: {1: {"path": [0, 1, 0], "qty": {"1": 3}}},
            2: {1: {"path": [0, 1, 0], "qty": {"1": 5}}},
        },
        "tau_return": {1: 0.0, 2: 0.0},
    }

    default_call  = _repair_route_result(route_result, sets_, params_, use_two_opt=False)
    explicit_true = _repair_route_result(route_result, sets_, params_, use_two_opt=False,
                                          use_delivery_shift=True)
    assert default_call["actual_qty"] == explicit_true["actual_qty"]
    assert default_call["actual_qty"][1, 1] == 7
    assert default_call["actual_qty"][1, 2] == 1


def test_repair_route_result_delivery_shift_true_applies_the_move():
    """Same rigged route as test_repair_delivery_shift_finds_improvement_
    via_holding_cost -- with use_delivery_shift=True passed through the
    main entry point (_repair_route_result), the shift must be found and
    applied. use_two_opt=False isolates the delivery-shift pass (both
    routes here are single-client, 2-opt/Or-opt would find nothing anyway,
    but disabling it keeps the test's intent explicit)."""
    sets_ = {"clients": [1], "T": [1, 2]}
    d = defaultdict(lambda: 0.0)
    params_ = {
        "d": d,
        "v": {1: 1.0},
        "s": {},
        "c_ijk": defaultdict(lambda: 1.0),
        "h_O": 1.0, "c1": 0.0, "c2": 0.0,
        "ET": defaultdict(lambda: 0.0),
        "LT": defaultdict(lambda: 1e9),
        "requires_cold": defaultdict(lambda: True),
        "frigo_trucks":  {1},
        "Q":             {1: 100},
        "q_lt":          {(1, 1): 1, (1, 2): 1},
        "I_O_min_frigo": 0, "I_O_max_frigo": 100,
        "I_O_min_nonfrigo": 0, "I_O_max_nonfrigo": 100,
    }
    route_result = {
        "x": {(0, 1, 1, 1): 1, (1, 0, 1, 1): 1, (0, 1, 2, 1): 1, (1, 0, 2, 1): 1},
        "f": {(0, 1, 1, 1): 3, (1, 0, 1, 1): 0, (0, 1, 2, 1): 5, (1, 0, 2, 1): 0},
        "depot_stock": {1: {"frigo": 10.0, "nonfrigo": 0.0}, 2: {"frigo": 0.0, "nonfrigo": 0.0}},
        "arrival_times": {(1, 1): 0.0, (1, 2): 0.0},
        "truck_assign": {(1, 1): 1, (1, 2): 1},
        "actual_qty": {(1, 1): 3, (1, 2): 5},
        "routes_data": {
            1: {1: {"path": [0, 1, 0], "qty": {"1": 3}}},
            2: {1: {"path": [0, 1, 0], "qty": {"1": 5}}},
        },
        "tau_return": {1: 0.0, 2: 0.0},
    }

    repaired = _repair_route_result(route_result, sets_, params_, use_two_opt=False,
                                     use_delivery_shift=True)

    assert repaired["actual_qty"][1, 1] == 7
    assert repaired["actual_qty"][1, 2] == 1
    assert compute_f1(repaired, sets_, params_) == 6.0
