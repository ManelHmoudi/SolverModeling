"""Unit tests for sensitivity.zone_locked's Sweep zoning and zone-locked decoder."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensitivity.zone_locked import _zone_assignment, _make_zone_locked_decoder


# ── _zone_assignment ─────────────────────────────────────────────────────

def test_zone_assignment_splits_by_angle_proportional_to_capacity():
    """4 clients at the 4 cardinal directions around the depot, equal
    demand, equal truck capacity -> a clean 50/50 angular split. Sorted by
    atan2(y, x): client 4 (0,-10, angle -pi/2), client 1 (10,0, angle 0),
    client 2 (0,10, angle pi/2), client 3 (-10,0, angle pi) -- in that
    order. Cumulative demand target for truck A is 2 (half of the total 4)
    -- reached exactly after clients 4 and 1 (cum=2), so truck A gets
    {4, 1} and truck B gets {2, 3}."""
    coords = {0: (0, 0), 1: (10, 0), 2: (0, 10), 3: (-10, 0), 4: (0, -10)}
    client_ids = [1, 2, 3, 4]
    trucks = ["A", "B"]
    Q = {"A": 1, "B": 1}
    total_demand = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}

    zone = _zone_assignment(coords, 0, client_ids, trucks, Q, total_demand)

    assert zone == {4: 0, 1: 0, 2: 1, 3: 1}


def test_zone_assignment_respects_heterogeneous_capacity_share():
    """Same 4 clients, but truck A has 3x truck B's capacity -- truck A's
    target share is 3/4 of total demand (=3), so it should absorb 3 of the
    4 equal-demand clients (angle order 4, 1, 2, 3) before truck B gets the
    last one."""
    coords = {0: (0, 0), 1: (10, 0), 2: (0, 10), 3: (-10, 0), 4: (0, -10)}
    client_ids = [1, 2, 3, 4]
    trucks = ["A", "B"]
    Q = {"A": 3, "B": 1}
    total_demand = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}

    zone = _zone_assignment(coords, 0, client_ids, trucks, Q, total_demand)

    assert zone == {4: 0, 1: 0, 2: 0, 3: 1}


# ── zone-locked decoder: isolation ───────────────────────────────────────

def test_zone_locked_decoder_never_lets_a_truck_cross_into_another_zone():
    """Client 3 is geographically much closer to client 1 (d=0.1) than
    client 2 is (d=5) -- an unrestricted greedy decoder would prefer client
    3 over client 2 once at client 1's position. Client 3 is locked to
    truck B's zone though, so truck A must still pick client 2, proving the
    zone restriction actually holds even under a strong distance
    incentive to violate it."""
    d = {
        (0, 1): 1, (1, 0): 1,
        (0, 2): 2, (2, 0): 2,
        (0, 3): 1, (3, 0): 1,
        (1, 2): 5, (2, 1): 5,
        (1, 3): 0.1, (3, 1): 0.1,
        (2, 3): 5, (3, 2): 5,
    }
    qty_dict = {1: 1, 2: 1, 3: 1}
    trucks   = ["A", "B"]
    Q        = {"A": 10, "B": 10}
    v        = {"A": 1, "B": 1}
    zone     = {1: 0, 2: 0, 3: 1}
    zone_by_trucks = {tuple(sorted(trucks)): zone}

    decoder = _make_zone_locked_decoder(zone_by_trucks)
    routes, x_vars, f_vars, arrivals, assign = decoder(
        qty_dict, trucks, t=0, d=d, v=v, s={}, Q=Q, O=0, tau_max=None, floors={}, priorities={})

    assert routes["A"]["path"] == [0, 1, 2, 0]
    assert routes["B"]["path"] == [0, 3, 0]
    assert assign[1, 0] == "A"
    assert assign[2, 0] == "A"
    assert assign[3, 0] == "B"


def test_zone_locked_decoder_leaves_infeasible_zone_overflow_unserved():
    """Both clients 1 and 2 are locked to truck A's zone, but truck A's
    capacity (1) can only fit one of them (qty 1 each) -- client 2 has no
    cross-zone fallback (truck B is a different zone entirely, zone={}) and
    must stay unserved (absent from any route's qty)."""
    d = {(0, 1): 1, (1, 0): 1, (0, 2): 1, (2, 0): 1, (1, 2): 1, (2, 1): 1}
    qty_dict = {1: 1, 2: 1}
    trucks   = ["A", "B"]
    Q        = {"A": 1, "B": 1}
    v        = {"A": 1, "B": 1}
    zone     = {1: 0, 2: 0}   # both in truck A's zone; nobody in truck B's
    zone_by_trucks = {tuple(sorted(trucks)): zone}

    decoder = _make_zone_locked_decoder(zone_by_trucks)
    routes, x_vars, f_vars, arrivals, assign = decoder(
        qty_dict, trucks, t=0, d=d, v=v, s={}, Q=Q, O=0, tau_max=None, floors={}, priorities={})

    served = {l for r in routes.values() for l in r["qty"]}
    assert served == {"1"}
    assert "B" not in routes
