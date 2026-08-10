"""Unit tests for sensitivity.prins_split's DP-optimal giant-tour split."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensitivity.prins_split import _segment_metrics, _prins_split_from_order


# ── _prins_split_from_order: DP beats a hand-verified greedy-maximal-fill baseline ──

def test_dp_split_prefers_globally_cheaper_partition_over_greedy_maximal_fill():
    """3 clients (order fixed: 1, 2, 3), capacity 2 per truck (forces >=2
    trucks). Client 1 is close to the depot but far from 2; clients 2 and 3
    are close to each other but far from 1. A left-to-right greedy-maximal
    fill (`_giant_tour_split`'s own strategy: keep adding while capacity
    allows) would pack truck A with {1, 2} (both fit under cap=2) and leave
    truck B with {3} alone -- hand-computed distance 101 + 2 = 103. The
    DP-optimal partition instead splits after client 1, pairing {2, 3}
    (geographically close) on truck B -- hand-computed distance 2 + 52 = 54,
    strictly better. This is exactly the failure mode
    `diagnose_giant_tour_continuity.py` diagnosed in its own greedy fill."""
    d = {
        (0, 1): 1, (1, 0): 1,
        (0, 2): 50, (2, 0): 50,
        (0, 3): 1, (3, 0): 1,
        (1, 2): 50, (2, 1): 50,
        (2, 3): 1, (3, 2): 1,
        (1, 3): 999, (3, 1): 999,
    }
    order    = [1, 2, 3]
    qty_dict = {1: 1, 2: 1, 3: 1}
    trucks   = ["A", "B"]
    Q        = {"A": 2, "B": 2}
    v        = {"A": 1, "B": 1}

    routes, x_vars, f_vars, arrivals, assign = _prins_split_from_order(
        order, qty_dict, trucks, t=0, d=d, v=v, s={}, Q=Q, O=0, tau_max=None, floors={})

    assert routes["A"]["path"] == [0, 1, 0]
    assert routes["A"]["qty"]  == {"1": 1}
    assert routes["B"]["path"] == [0, 2, 3, 0]
    assert routes["B"]["qty"]  == {"2": 1, "3": 1}

    total_distance = (d[0, 1] + d[1, 0]) + (d[0, 2] + d[2, 3] + d[3, 0])
    assert total_distance == 54
    greedy_maximal_fill_distance = (d[0, 1] + d[1, 2] + d[2, 0]) + (d[0, 3] + d[3, 0])
    assert greedy_maximal_fill_distance == 103
    assert total_distance < greedy_maximal_fill_distance

    assert x_vars[0, 1, 0, "A"] == 1
    assert x_vars[1, 0, 0, "A"] == 1
    assert x_vars[0, 2, 0, "B"] == 1
    assert x_vars[2, 3, 0, "B"] == 1
    assert x_vars[3, 0, 0, "B"] == 1
    assert assign[1, 0] == "A"
    assert assign[2, 0] == "B"
    assert assign[3, 0] == "B"


# ── _segment_metrics: mandatory floor fallback ──────────────────────────

def test_segment_metrics_falls_back_to_mandatory_floor_when_full_qty_too_heavy():
    """Client 1 is mandatory with floor=2 but full requested qty=4; client 2
    is optional with qty=3. Full load (4+3=7) exceeds no capacity check
    inside _segment_metrics itself (that's the DP caller's job) -- this test
    checks the metrics are computed correctly so the caller CAN make that
    capacity decision: load_full=7, load_floor=5 (client 1 reduced to its
    floor, client 2 unchanged)."""
    d = {(0, 1): 1, (1, 0): 1, (1, 2): 1, (2, 1): 1, (0, 2): 1, (2, 0): 1}
    order    = [1, 2]
    qty_dict = {1: 4, 2: 3}
    floors   = {1: 2}

    metrics = _segment_metrics(order, 0, 2, qty_dict, floors, d, v_k=1, s={}, O=0, tau_max=None)

    assert metrics is not None
    distance, load_full, qty_full, load_floor, qty_floor = metrics
    assert distance == 3
    assert load_full == 7
    assert qty_full == {1: 4, 2: 3}
    assert load_floor == 5
    assert qty_floor == {1: 2, 2: 3}


# ── _segment_metrics: mandatory clients bypass the tau_max check ────────

def test_segment_metrics_mandatory_client_bypasses_tau_max():
    """Client 2 is mandatory (floor=1); its return-trip projection (12)
    exceeds tau_max=5, but mandatory clients bypass the check -- same rule
    as `_nearest_neighbour` ("Mandatory clients ... bypass the tau_max check
    to preserve delivery deadlines"). Segment must stay feasible."""
    d = {(0, 1): 1, (1, 0): 1, (1, 2): 10, (2, 1): 10, (0, 2): 1, (2, 0): 1}
    order    = [1, 2]
    qty_dict = {1: 1, 2: 1}
    floors   = {2: 1}

    metrics = _segment_metrics(order, 0, 2, qty_dict, floors, d, v_k=1, s={}, O=0, tau_max=5)

    assert metrics is not None


def test_segment_metrics_non_mandatory_client_blocked_by_tau_max():
    """Same geometry as above, but client 2 is NOT mandatory this time --
    the tau_max check must apply and reject the segment (projection=12 > 5)."""
    d = {(0, 1): 1, (1, 0): 1, (1, 2): 10, (2, 1): 10, (0, 2): 1, (2, 0): 1}
    order    = [1, 2]
    qty_dict = {1: 1, 2: 1}
    floors   = {}

    metrics = _segment_metrics(order, 0, 2, qty_dict, floors, d, v_k=1, s={}, O=0, tau_max=5)

    assert metrics is None
