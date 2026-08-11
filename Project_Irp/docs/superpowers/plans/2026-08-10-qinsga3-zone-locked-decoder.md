# QINSGA3 zone-locked decoder — continuity screening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a continuity smoke test checking whether locking clients to fixed geographic zones (one per truck, computed once and independent of theta) produces a real theta→route continuity gradient, extending the existing 3-way comparison (original / giant-tour / Prins-split) to 4 columns.

**Architecture:** A standalone, testable module (`sensitivity/zone_locked.py`) computes a Sweep-style angular zoning once per truck group and produces a drop-in `_nearest_neighbour`-compatible decoder that restricts each truck to its own zone's pending clients. A diagnose script (`sensitivity/diagnose_zone_locked_continuity.py`), mirroring the existing `diagnose_giant_tour_continuity.py`/`diagnose_prins_split_continuity.py`, monkey-patches it in and measures Jaccard route-distance under theta perturbation across the real production alpha schedule, alongside the three existing decoders.

**Tech Stack:** Python, numpy, pytest. No new dependencies (uses `math.atan2`, stdlib).

## Global Constraints

- `Solvers/NSGA3/decoder.py` must never be modified on disk — monkey-patched fork only, restored in `finally` (same isolation convention as every prior remedy).
- No change to `Solvers/QINSGA3/algorithm.py` — ablation-only in `sensitivity/` regardless of outcome.
- Zone assignment is computed once, from static coordinates and aggregate per-client demand only — never recomputed per generation/theta/priority.
- A client whose zone's truck can't fit it this period has no cross-zone fallback (documented simplification, tracked via a counter, not verified non-degenerate beforehand).
- `any_mandatory_pending` (the rule that blocks non-mandatory candidates while a mandatory client is still pending) is scoped to the truck's own zone, not globally across all zones — a deliberate adaptation to keep zones isolated, different from `_nearest_neighbour`'s global rule.

Spec: `docs/superpowers/specs/2026-08-10-qinsga3-zone-locked-decoder-design.md`

---

### Task 1: Zone assignment + zone-locked decoder (`sensitivity/zone_locked.py`)

**Files:**
- Create: `sensitivity/zone_locked.py`
- Create: `sensitivity/test_zone_locked.py`

**Interfaces:**
- Produces:
  - `_zone_assignment(coords, O, client_ids, trucks, Q, total_demand)` → `dict {client_id: zone_index}` (`zone_index` in `0..len(trucks)-1`).
  - `_make_zone_locked_decoder(zone_by_trucks)` → a function `(qty_dict, trucks, t, d, v, s, Q, O, tau_max=None, floors=None, priorities=None)` with the exact same signature and 5-tuple return shape as `Solvers.NSGA3.decoder._nearest_neighbour` — a drop-in monkey-patch target. `zone_by_trucks` is a `dict {tuple(sorted(trucks)): zone_dict}`, one entry per truck group.

- [ ] **Step 1: Write `sensitivity/zone_locked.py`**

```python
"""Sweep-partition (Gillett & Miller 1974) zone-locked decoder for QINSGA3
continuity screening: clients are assigned once to a fixed geographic zone
(one per truck, by polar angle around the depot), independent of theta.
Construction reuses production's exact greedy scoring
(`Solvers/NSGA3/decoder.py::_nearest_neighbour`), with one added
restriction: the truck at position `truck_idx` only ever considers pending
clients in its own zone -- a theta perturbation can still reorder or drop
clients within a zone, but can never move one into a different truck's
route. See
docs/superpowers/specs/2026-08-10-qinsga3-zone-locked-decoder-design.md.
"""
from __future__ import annotations

import math

_unserved_count = [0]   # mutable counter: clients whose zone's truck couldn't
                          # fit them this call (no cross-zone fallback), for
                          # the smoke test's own reporting -- not used by the
                          # decoder itself.


def _zone_assignment(coords, O, client_ids, trucks, Q, total_demand):
    """Partition `client_ids` into `len(trucks)` contiguous angular sectors
    around `coords[O]`, one sector per truck in the given fixed order. Cut
    thresholds are proportional to each truck's capacity `Q[truck]` against
    cumulative aggregate demand (`total_demand`, a static per-client
    weight -- NOT the smaller per-period quantity seen at any single decode
    call), keeping zones roughly capacity-balanced despite a heterogeneous
    fleet. Returns {client_id: zone_index}, zone_index in 0..len(trucks)-1.
    """
    ox, oy = coords[O]
    order = sorted(
        client_ids,
        key=lambda l: (math.atan2(coords[l][1] - oy, coords[l][0] - ox), l),
    )
    total_Q = sum(Q[truck] for truck in trucks) or 1.0
    total_d = sum(total_demand.get(l, 0.0) for l in client_ids)

    zone, cum, k_pos = {}, 0.0, 0
    cum_target = total_d * (Q[trucks[0]] / total_Q)
    for l in order:
        while k_pos < len(trucks) - 1 and cum >= cum_target:
            k_pos += 1
            cum_target += total_d * (Q[trucks[k_pos]] / total_Q)
        zone[l] = k_pos
        cum += total_demand.get(l, 0.0)
    return zone


def _make_zone_locked_decoder(zone_by_trucks):
    """Returns a `_nearest_neighbour`-compatible decoder closed over
    `zone_by_trucks` (`{tuple(sorted(trucks)): zone_dict}`, one entry per
    truck group -- looked up by truck-id set, so the same returned function
    works for both the frigo and non-frigo groups)."""

    def _zone_locked_nearest_neighbour(qty_dict, trucks, t, d, v, s, Q, O,
                                        tau_max=None, floors=None, priorities=None):
        zone = zone_by_trucks[tuple(sorted(trucks))]
        prios = priorities or {}
        floors = floors or {}
        pending = dict(qty_dict)
        truck_idx = 0

        x_vars, f_vars, arrivals, assign, routes = {}, {}, {}, {}, {}

        while pending and truck_idx < len(trucks):
            k     = trucks[truck_idx]
            cap   = Q[k]
            speed = v[k]

            path, qty_on_route, load, current_time, current = [O], {}, 0, 0.0, O

            while True:
                best, best_score, best_qty = None, float("inf"), 0
                has_mandatory = False

                zone_pending = {l: q for l, q in pending.items() if zone.get(l) == truck_idx}
                any_mandatory_pending = any(floors.get(l2, 0) > 0 for l2 in zone_pending)

                for l, q in zone_pending.items():
                    floor_l      = floors.get(l, 0)
                    is_mandatory = floor_l > 0

                    if not is_mandatory and any_mandatory_pending:
                        continue

                    if load + q <= cap:
                        q_effective = q
                    elif is_mandatory and load + floor_l <= cap:
                        q_effective = floor_l
                    else:
                        continue

                    dist = d[current, l]

                    if tau_max is not None and not is_mandatory:
                        projected = (current_time + s.get(current, 0.0) + dist / speed
                                     + s.get(l, 0.0) + d[l, O] / speed)
                        if projected > tau_max:
                            continue

                    score = dist / (0.5 + prios.get(l, 0.5))
                    if is_mandatory:
                        if not has_mandatory or score < best_score:
                            best_score, best, best_qty = score, l, q_effective
                            has_mandatory = True
                    elif not has_mandatory and score < best_score:
                        best_score, best, best_qty = score, l, q_effective

                if best is None:
                    break

                current_time      += s.get(current, 0.0) + d[current, best] / speed
                path.append(best)
                pending.pop(best)
                qty_on_route[best] = best_qty
                load              += qty_on_route[best]
                arrivals[best, t]  = current_time
                assign[best, t]    = k
                current            = best

            path.append(O)

            if len(path) > 2:
                n   = len(path)
                suf = [0] * (n + 1)
                for idx in range(n - 2, -1, -1):
                    node    = path[idx + 1]
                    suf[idx] = suf[idx + 1] + (qty_on_route.get(node, 0) if node != O else 0)

                for idx in range(n - 1):
                    i, j               = path[idx], path[idx + 1]
                    x_vars[i, j, t, k] = 1
                    f_vars[i, j, t, k] = suf[idx]

                routes[k] = {
                    "path": path,
                    "qty":  {str(l): q for l, q in qty_on_route.items()},
                }

            truck_idx += 1

        _unserved_count[0] += len(pending)
        return routes, x_vars, f_vars, arrivals, assign

    return _zone_locked_nearest_neighbour
```

- [ ] **Step 2: Write `sensitivity/test_zone_locked.py`**

```python
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
```

- [ ] **Step 3: Run the tests**

Run: `cd sensitivity && python -m pytest test_zone_locked.py -v`
Expected: `4 passed`

- [ ] **Step 4: Commit**

```bash
git add sensitivity/zone_locked.py sensitivity/test_zone_locked.py
git commit -m "$(cat <<'EOF'
feat: zone-locked decoder for QINSGA3 continuity screening

Sweep-partition (Gillett & Miller 1974) zoning, computed once and
independent of theta, restricting each truck to its own geographic zone
-- bounds the reach of a theta perturbation instead of changing the
greedy decision rule, unlike every remedy tried so far.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Continuity smoke test script and run

**Files:**
- Create: `sensitivity/diagnose_zone_locked_continuity.py`
- Modify: `docs/superpowers/specs/2026-08-10-qinsga3-zone-locked-decoder-design.md` (append Result section)

**Interfaces:**
- Consumes: `_zone_assignment`, `_make_zone_locked_decoder` from `sensitivity/zone_locked.py` (Task 1); `decode_chromosome`, `Solvers.NSGA3.decoder` (`build_routes`, `_nearest_neighbour`); `_build_routes_giant_tour` from `sensitivity/diagnose_giant_tour_continuity.py`; `_prins_split` from `sensitivity/prins_split.py`; `_route_arcset`/`_jaccard_distance` from `sensitivity/test_route_diversity.py`; `_encode_theta` from `Solvers/QINSGA3/algorithm.py`.
- Produces: printed 5-column comparison table (gen, alpha, Jaccard original, giant-tour, Prins-split, zone-locked) and correlation summary.

- [ ] **Step 1: Write `sensitivity/diagnose_zone_locked_continuity.py`**

```python
"""Smoke test (not a performance comparison): does locking clients to fixed
geographic zones (`sensitivity/zone_locked.py`) restore theta -> route
continuity where greedy-fill giant-tour and Prins DP-split both failed?

Context: `docs/superpowers/specs/2026-08-10-qinsga3-zone-locked-decoder-
design.md`. Same real cached NSGA-III chromosomes, same sparse alpha
sweep, and same Jaccard route-distance metric as
`diagnose_giant_tour_continuity.py`/`diagnose_prins_split_continuity.py`,
extended to a four-way comparison (original greedy / giant-tour
greedy-fill / Prins DP-split / zone-locked).

Isolation: `Solvers/NSGA3/decoder.py` is NEVER modified on disk -- each
alternative decoder is monkey-patched onto `_nearest_neighbour` only for
the duration of its own `build_routes` call, restored in `finally`.

Usage:
    python -m sensitivity.diagnose_zone_locked_continuity
"""
from __future__ import annotations

import json
import os
import sys

MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(MODULE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np

from models.parametres          import load_instance
import Solvers.NSGA3.decoder as decoder_mod
from Solvers.NSGA3.decoder      import decode_chromosome
from Solvers.NSGA3.problem      import IRPProblem
from Solvers.QINSGA3.algorithm  import _encode_theta
from sensitivity.test_route_diversity           import _route_arcset, _jaccard_distance
from sensitivity.diagnose_giant_tour_continuity import _build_routes_giant_tour
from sensitivity.prins_split                    import _prins_split
from sensitivity.zone_locked                    import _zone_assignment, _make_zone_locked_decoder

INSTANCE   = "100"
SEEDS      = [42, 137, 271, 491, 613]
MAX_GEN    = 300
ALPHA_MAX  = 0.10 * np.pi
ALPHA_MIN  = 0.001 * np.pi
GENS_TO_TEST = [0, 50, 100, 150, 200, 250, 300]
RNG_SEED   = 12345

_NSGA3_CACHE = os.path.join(PROJECT_DIR, "Solvers", "NSGA3", "nsga3_chromosomes.json")


def _build_routes_prins_split(quantities, sets_, params_, priorities):
    original = decoder_mod._nearest_neighbour
    decoder_mod._nearest_neighbour = _prins_split
    try:
        return decoder_mod.build_routes(quantities, sets_, params_, priorities)
    finally:
        decoder_mod._nearest_neighbour = original


def _load_coordinates(data_path):
    with open(data_path, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): tuple(v) for k, v in raw["sets"]["coordinates"].items()}


def _zone_locked_builder(sets_, params_, coords):
    O                = sets_["O"]
    clients          = sets_["clients"]
    T                = sets_["T"]
    Q                = params_["Q"]
    frigo_trucks     = params_["frigo_trucks"]
    frigo_list       = sorted(frigo_trucks)
    non_frigo_trucks = [k for k in sets_["M"] if k not in frigo_trucks]
    q_lt             = params_["q_lt"]

    total_demand = {l: sum(q_lt.get((l, t), 0) for t in T) for l in clients}

    zone_by_trucks = {}
    for trucks in (frigo_list, non_frigo_trucks):
        if trucks:
            zone_by_trucks[tuple(sorted(trucks))] = _zone_assignment(
                coords, O, clients, trucks, Q, total_demand)

    zone_locked_decoder = _make_zone_locked_decoder(zone_by_trucks)

    def _build_routes_zone_locked(quantities, sets_inner, params_inner, priorities):
        original = decoder_mod._nearest_neighbour
        decoder_mod._nearest_neighbour = zone_locked_decoder
        try:
            return decoder_mod.build_routes(quantities, sets_inner, params_inner, priorities)
        finally:
            decoder_mod._nearest_neighbour = original

    return _build_routes_zone_locked


def main() -> None:
    data_path = os.path.join(PROJECT_DIR, "data", f"instance_{INSTANCE}_clients.json")
    sets_, params_ = load_instance(data_path)
    coords = _load_coordinates(data_path)
    build_routes_zone_locked = _zone_locked_builder(sets_, params_, coords)

    problem = IRPProblem(sets_, params_)
    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)

    with open(_NSGA3_CACHE, encoding="utf-8") as f:
        cache = json.load(f)
    by_seed = {r["seed"]: r for r in cache["runs"]}
    baseline_X = []
    for sd in SEEDS:
        baseline_X.extend(by_seed[sd]["chromosomes"])
    baseline_X = np.array(baseline_X, dtype=float)
    N = len(baseline_X)
    print(f"Baseline : {N} solutions reelles (NSGA-III cache, {len(SEEDS)} seeds)")

    theta = _encode_theta(baseline_X, xl, xu)
    rng = np.random.default_rng(RNG_SEED)

    print("\nDecodage des baselines (quatre decodeurs, routes de reference)...")
    base_o, base_g, base_p, base_z = [], [], [], []
    for i in range(N):
        q, p = decode_chromosome(baseline_X[i], sets_)
        base_o.append(_route_arcset(decoder_mod.build_routes(q, sets_, params_, p)))
        base_g.append(_route_arcset(_build_routes_giant_tour(q, sets_, params_, p)))
        base_p.append(_route_arcset(_build_routes_prins_split(q, sets_, params_, p)))
        base_z.append(_route_arcset(build_routes_zone_locked(q, sets_, params_, p)))

    print("\n" + "=" * 140)
    print("  CONTINUITE theta -> route : original vs giant-tour vs Prins-split vs zone-locked")
    print("=" * 140)
    print(f"  {'gen':<6} {'alpha':<10} {'original':<14} {'giant-tour':<14} {'prins-split':<14} {'zone-locked':<14}")
    print("-" * 140)

    results = []
    for gen in GENS_TO_TEST:
        alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * (1.0 - gen / MAX_GEN)

        guide_idx = rng.integers(0, N, size=N)
        same = guide_idx == np.arange(N)
        guide_idx[same] = (guide_idx[same] + 1) % N
        theta_guide = theta[guide_idx]

        diff = theta_guide - theta
        theta_pert = np.clip(theta + alpha * np.tanh(diff / (np.pi / 8.0)), 0.0, np.pi / 2.0)
        p_pert = np.cos(theta_pert) ** 2
        X_pert = np.clip(xl + p_pert * (xu - xl), xl, xu)

        d_o, d_g, d_p, d_z = [], [], [], []
        for i in range(N):
            q, p = decode_chromosome(X_pert[i], sets_)
            d_o.append(_jaccard_distance(base_o[i], _route_arcset(decoder_mod.build_routes(q, sets_, params_, p))))
            d_g.append(_jaccard_distance(base_g[i], _route_arcset(_build_routes_giant_tour(q, sets_, params_, p))))
            d_p.append(_jaccard_distance(base_p[i], _route_arcset(_build_routes_prins_split(q, sets_, params_, p))))
            d_z.append(_jaccard_distance(base_z[i], _route_arcset(build_routes_zone_locked(q, sets_, params_, p))))

        m_o, m_g, m_p, m_z = (float(np.mean(x)) for x in (d_o, d_g, d_p, d_z))
        results.append((gen, alpha, m_o, m_g, m_p, m_z))
        print(f"  {gen:<6} {alpha:<10.5f} {m_o:<14.4f} {m_g:<14.4f} {m_p:<14.4f} {m_z:<14.4f}")

    print("-" * 140)
    alphas = np.array([r[1] for r in results])
    cols   = {"original": np.array([r[2] for r in results]),
              "giant-tour": np.array([r[3] for r in results]),
              "prins-split": np.array([r[4] for r in results]),
              "zone-locked": np.array([r[5] for r in results])}

    def _corr(y):
        return float(np.corrcoef(alphas, y)[0, 1]) if len(alphas) >= 3 else float("nan")

    i_min, i_max = int(np.argmin(alphas)), int(np.argmax(alphas))
    print(f"\n  Jaccard au plus petit alpha teste ({alphas[i_min]:.5f}):")
    for name, y in cols.items():
        print(f"    {name:<14}: {y[i_min]:.4f}")
    print(f"  Jaccard au plus grand alpha teste  ({alphas[i_max]:.5f}):")
    for name, y in cols.items():
        print(f"    {name:<14}: {y[i_max]:.4f}")
    print("  Correlation (alpha, Jaccard):")
    for name, y in cols.items():
        print(f"    {name:<14}: r={_corr(y):.4f}")

    print("\nINTERPRETATION")
    print("  Continuite restauree si : (1) Jaccard zone-locked au plus petit alpha est")
    print("  nettement plus proche de 0 que les trois autres au meme alpha, ET (2) la")
    print("  correlation (alpha, Jaccard) est nettement positive pour zone-locked (le Jaccard")
    print("  grandit avec alpha, un vrai gradient) alors que les autres saturent deja au")
    print("  plus petit alpha teste.")
    print(f"{'='*140}\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the smoke test**

Run: `python -m sensitivity.diagnose_zone_locked_continuity`
Expected: completes without error, prints the four-way Jaccard table across `GENS_TO_TEST` and the correlation summary.

- [ ] **Step 3: Record the result**

Append a `## Result` section to
`docs/superpowers/specs/2026-08-10-qinsga3-zone-locked-decoder-design.md`
with the printed table and correlation values, and state the decision:
scale to `sensitivity/compare_zone_locked.py` (a follow-up plan, not part
of this one) only if zone-locked shows a real gradient unlike the other
three; otherwise mark it a null result, same convention as Prins-split and
every other rejected remedy in `Solvers/IRP_results_summary.md`.

- [ ] **Step 4: Commit**

```bash
git add sensitivity/diagnose_zone_locked_continuity.py \
        docs/superpowers/specs/2026-08-10-qinsga3-zone-locked-decoder-design.md
git commit -m "$(cat <<'EOF'
test: four-way continuity smoke test for the zone-locked decoder

Compares original greedy / giant-tour greedy-fill / Prins DP-split /
zone-locked under theta perturbation across the real production alpha
schedule. Records the result in the design spec to decide whether a full
quality campaign (compare_zone_locked.py) is worth running.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

## Non-goals (this plan)

No `sensitivity/compare_zone_locked.py` quality campaign — conditional on
this plan's Task 2 result, and if warranted, a follow-up plan of its own
(same protocol: 3 seeds first, scale on positive signal). No change to any
file outside `sensitivity/` and the design spec.
