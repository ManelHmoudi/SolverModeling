# QINSGA3 Post-Decode Route Repair (Remède G) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and empirically validate a post-decode 2-opt route repair for QI-NSGA-III, scoped entirely to QINSGA3, that redoes an earlier abandoned repair attempt correctly — evaluated against the real f1 cost function instead of raw distance.

**Architecture:** A new `Solvers/QINSGA3/repair.py` module implements a Baldwinian (evaluation-only, no re-encoding into the chromosome) 2-opt local search: pure candidate generation and traversal-time helpers, plus an orchestrator that runs a first-improvement search per route, guarded so no swap can increase a period's worst-case travel time beyond its pre-repair value, accepting only swaps that strictly reduce f1 (recomputed via the project's own `compute_f1`). A new `_evaluate_with_repair` function in `Solvers/QINSGA3/algorithm.py` wires this into a parallel worker-evaluation path, gated behind a new `use_route_repair: bool = False` parameter on `run_qinsga3` — `Solvers/NSGA3/decoder.py`, `evaluator.py`, and `problem.py` (shared with NSGA3) are never modified. Validated with the project's established 3-seed fast-reject protocol, scaling to 5 seeds on a positive signal, documented as "Remède G" regardless of outcome.

**Tech Stack:** Python, NumPy, the project's existing `decode_chromosome`/`build_routes`/`compute_f1..f4` (reused unchanged), pymoo (unchanged), plain-assert pytest-style tests matching `Solvers/QINSGA3/test_algorithm.py`'s established style.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-04-qinsga3-route-repair-design.md` — every task below implements one part of it; do not deviate from its function signatures or the G-list duplication rationale.
- **Never modify** `Solvers/NSGA3/decoder.py`, `Solvers/NSGA3/evaluator.py`, or `Solvers/NSGA3/problem.py` — these are shared with NSGA3, and this whole design exists specifically to avoid touching them. Only read from them for reference.
- `use_route_repair` must default to `False` — production `run_qinsga3()` behavior is unchanged unless a caller explicitly opts in.
- No re-encoding of the repaired route back into the chromosome (θ/X) — repair only affects the fitness value assigned to an individual, never its genotype.
- Run all commands from the repository root (`c:\Users\Mariem\OneDrive\Bureau\SolverModeling\Project_Irp`).
- Before staging any commit, run `git status`/`git diff --stat` and confirm only the files this task is supposed to touch are modified — this branch has repeatedly hit an issue where a broad `git add` swept up unrelated pre-existing uncommitted work. If anything unexpected shows as modified before you've written anything, stop and report NEEDS_CONTEXT rather than committing over it.

---

### Task 1: `repair.py` — pure candidate generation and traversal-time helpers

**Files:**
- Create: `Solvers/QINSGA3/repair.py`
- Test: `Solvers/QINSGA3/test_repair.py`

**Interfaces:**
- Produces: `_two_opt_candidates(path: list) -> Iterator[tuple[int, int, list]]` and `_route_traversal_time(path: list, k, params_: dict) -> float` — both consumed by Task 3.

- [ ] **Step 1: Write the failing tests**

Create `Solvers/QINSGA3/test_repair.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Solvers/QINSGA3/test_repair.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'Solvers.QINSGA3.repair'`.

- [ ] **Step 3: Implement the two pure helpers**

Create `Solvers/QINSGA3/repair.py`:

```python
"""Post-decode 2-opt route repair for QI-NSGA-III (remedy G) -- see
docs/superpowers/specs/2026-08-04-qinsga3-route-repair-design.md.

Baldwinian repair: improves the FITNESS assigned to a decoded route_result,
never re-encoded back into the chromosome (theta/X) -- the visit order a
2-opt swap produces has no defined inverse into priority genes. Scoped
entirely to QINSGA3: Solvers/NSGA3/decoder.py, evaluator.py, and problem.py
(shared with NSGA3) are never imported for modification here, only their
pure functions (decode_chromosome, build_routes, compute_f1) are reused
unchanged, from Solvers/QINSGA3/algorithm.py's _evaluate_with_repair.
"""


def _two_opt_candidates(path: list):
    """Yield (i, j, candidate_path) for every 2-opt segment reversal of the
    interior of path (positions 1..len(path)-2 -- the depot at both ends,
    index 0 and index len(path)-1, is never moved). Pure: never inspects
    distances, costs, or any domain state.
    """
    n = len(path)
    for i in range(1, n - 2):
        for j in range(i + 1, n - 1):
            candidate = path[:i] + path[i:j + 1][::-1] + path[j + 1:]
            yield i, j, candidate


def _route_traversal_time(path: list, k, params_: dict) -> float:
    """Total round-trip time for one truck's path (depot to depot), summing
    service time + travel time over every consecutive pair -- the exact
    formula Solvers/NSGA3/decoder.py's build_routes uses for its own
    per-route tau_return computation (decoder.py, the block right after
    the main per-period loop), factored out here as a standalone helper
    since build_routes doesn't expose it.
    """
    d = params_["d"]
    v = params_["v"]
    s = params_["s"]
    speed = v[k]
    return sum(
        s.get(path[idx], 0.0) + d[path[idx], path[idx + 1]] / speed
        for idx in range(len(path) - 1)
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Solvers/QINSGA3/test_repair.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add Solvers/QINSGA3/repair.py Solvers/QINSGA3/test_repair.py
git commit -m "feat: add 2-opt candidate generation and traversal-time helpers (remede G, part 1/4)"
```

---

### Task 2: `repair.py` — arc-rebuilding helpers

**Files:**
- Modify: `Solvers/QINSGA3/repair.py`
- Test: `Solvers/QINSGA3/test_repair.py`

**Interfaces:**
- Consumes: nothing from Task 1 directly (these are independent pure functions operating on a path/quantities, not on candidates).
- Produces: `_rebuild_route_arcs(path: list, qty_on_route: dict, t, k, params_: dict) -> tuple[dict, dict, dict]` (returns `(x_vars, f_vars, arrivals)`) and `_replace_route_arcs(arc_dict: dict, old_path: list, t, k, new_entries: dict) -> dict` — both consumed by Task 3.

- [ ] **Step 1: Write the failing tests**

Append to `Solvers/QINSGA3/test_repair.py` (add the new import names to the existing import line):

```python
from Solvers.QINSGA3.repair import (
    _two_opt_candidates, _route_traversal_time,
    _rebuild_route_arcs, _replace_route_arcs,
)
```

Then append:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Solvers/QINSGA3/test_repair.py -v`
Expected: FAIL — `ImportError: cannot import name '_rebuild_route_arcs'`.

- [ ] **Step 3: Implement the two helpers**

Append to `Solvers/QINSGA3/repair.py`:

```python
def _rebuild_route_arcs(path: list, qty_on_route: dict, t, k, params_: dict):
    """Recompute one route's x/f arc entries and per-client arrival times
    for a (possibly reordered) path, given the SAME qty_on_route mapping
    the original decode produced -- a 2-opt swap never changes which
    clients are visited or their delivered quantities, only the order.
    Mirrors decoder.py's own arrival-time accumulation (line 342) and
    suffix-sum arc-flow construction (lines 353-368) exactly, factored out
    here since build_routes doesn't expose either as a standalone function.

    Returns (x_vars, f_vars, arrivals) for THIS route only -- the caller
    merges these into the full route_result's dicts via _replace_route_arcs.
    """
    d = params_["d"]
    v = params_["v"]
    s = params_["s"]
    speed = v[k]
    depot = path[0]

    x_vars = {}
    f_vars = {}
    arrivals = {}

    current_time = 0.0
    current = depot
    for node in path[1:]:
        current_time += s.get(current, 0.0) + d[current, node] / speed
        if node != depot:
            arrivals[node, t] = current_time
        current = node

    n = len(path)
    suf = [0] * (n + 1)
    for idx in range(n - 2, -1, -1):
        node = path[idx + 1]
        suf[idx] = suf[idx + 1] + (qty_on_route.get(node, 0) if node != depot else 0)

    for idx in range(n - 1):
        i, j = path[idx], path[idx + 1]
        x_vars[i, j, t, k] = 1
        f_vars[i, j, t, k] = suf[idx]

    return x_vars, f_vars, arrivals


def _replace_route_arcs(arc_dict: dict, old_path: list, t, k, new_entries: dict) -> dict:
    """Remove old_path's (i, j, t, k) arc keys from arc_dict, then merge in
    new_entries -- used to update route_result["x"]/["f"] for one repaired
    route without disturbing other routes' entries in the same shared dict.
    """
    result = dict(arc_dict)
    for idx in range(len(old_path) - 1):
        i, j = old_path[idx], old_path[idx + 1]
        result.pop((i, j, t, k), None)
    result.update(new_entries)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Solvers/QINSGA3/test_repair.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add Solvers/QINSGA3/repair.py Solvers/QINSGA3/test_repair.py
git commit -m "feat: add arc-rebuilding helpers for route repair (remede G, part 2/4)"
```

---

### Task 3: `repair.py` — `_repair_route_result` orchestrator

**Files:**
- Modify: `Solvers/QINSGA3/repair.py`
- Test: `Solvers/QINSGA3/test_repair.py`

**Interfaces:**
- Consumes: `_two_opt_candidates`, `_route_traversal_time`, `_rebuild_route_arcs`, `_replace_route_arcs` (Tasks 1-2), and `compute_f1` from `Solvers.NSGA3.evaluator` (existing, unmodified).
- Produces: `_repair_route_result(route_result: dict, sets_: dict, params_: dict) -> dict` — consumed by Task 4.

- [ ] **Step 1: Write the failing tests**

Append to `Solvers/QINSGA3/test_repair.py` (add the new import name, and a new top-level import):

```python
from Solvers.QINSGA3.repair import (
    _two_opt_candidates, _route_traversal_time,
    _rebuild_route_arcs, _replace_route_arcs,
    _repair_route_result,
)
from Solvers.NSGA3.evaluator import compute_f1
```

Then append:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Solvers/QINSGA3/test_repair.py -v`
Expected: FAIL — `ImportError: cannot import name '_repair_route_result'`.

- [ ] **Step 3: Implement `_repair_route_result`**

Append to `Solvers/QINSGA3/repair.py`:

```python
_MAX_REPAIR_ITER = 20   # internal constant, not exposed -- see design doc's "New parameters"


def _repair_route_result(route_result: dict, sets_: dict, params_: dict) -> dict:
    """First-improvement 2-opt local search per route: for every truck's
    path longer than 3 nodes (more than 1 client), repeatedly applies the
    first candidate swap that (a) does not push this period's worst-case
    travel time above its pre-repair value, and (b) strictly reduces f1
    for the whole individual -- until no such candidate exists or
    _MAX_REPAIR_ITER is reached. Baldwinian: returns a NEW route_result
    with updated x/f/arrival_times/routes_data/tau_return; the chromosome
    that produced the original route_result is never touched by the caller.
    """
    from Solvers.NSGA3.evaluator import compute_f1

    working = dict(route_result)
    working["x"] = dict(route_result["x"])
    working["f"] = dict(route_result["f"])
    working["arrival_times"] = dict(route_result["arrival_times"])
    working["tau_return"] = dict(route_result["tau_return"])
    working["routes_data"] = {
        t: dict(routes) for t, routes in route_result["routes_data"].items()
    }

    for t, routes in route_result["routes_data"].items():
        tau_return_before = route_result["tau_return"].get(t, 0.0)

        for k, info in routes.items():
            path = list(info["path"])
            if len(path) <= 3:
                continue
            qty_on_route = {int(l): q for l, q in info["qty"].items()}

            current_f1 = compute_f1(working, sets_, params_)

            for _ in range(_MAX_REPAIR_ITER):
                improved = False
                for i, j, candidate in _two_opt_candidates(path):
                    candidate_time = _route_traversal_time(candidate, k, params_)
                    if candidate_time > tau_return_before:
                        continue

                    trial_x, trial_f, trial_arrivals = _rebuild_route_arcs(
                        candidate, qty_on_route, t, k, params_
                    )
                    scratch = dict(working)
                    scratch["x"] = _replace_route_arcs(working["x"], path, t, k, trial_x)
                    scratch["f"] = _replace_route_arcs(working["f"], path, t, k, trial_f)
                    scratch["arrival_times"] = {**working["arrival_times"], **trial_arrivals}

                    trial_f1 = compute_f1(scratch, sets_, params_)
                    if trial_f1 < current_f1:
                        working["x"] = scratch["x"]
                        working["f"] = scratch["f"]
                        working["arrival_times"] = scratch["arrival_times"]
                        working["routes_data"][t][k] = {"path": candidate, "qty": info["qty"]}
                        working["tau_return"][t] = max(
                            _route_traversal_time(r["path"], k2, params_)
                            for k2, r in working["routes_data"][t].items()
                        )
                        path = candidate
                        current_f1 = trial_f1
                        improved = True
                        break
                if not improved:
                    break

    return working
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Solvers/QINSGA3/test_repair.py -v`
Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add Solvers/QINSGA3/repair.py Solvers/QINSGA3/test_repair.py
git commit -m "feat: add _repair_route_result orchestrator (remede G, part 3/4)"
```

---

### Task 4: Wire repair into `Solvers/QINSGA3/algorithm.py`

**Files:**
- Modify: `Solvers/QINSGA3/algorithm.py`
- Test: `Solvers/QINSGA3/test_algorithm.py`

**Interfaces:**
- Consumes: `_repair_route_result` (Task 3), `decode_chromosome`/`build_routes` (`Solvers.NSGA3.decoder`, unmodified), `compute_f1`..`compute_f4` (`Solvers.NSGA3.evaluator`, unmodified).
- Produces: `_build_g_constraints(route_result, sets_, params_) -> list`, `_evaluate_with_repair(x, sets_, params_) -> (F, G)`, `_worker_eval_repaired(x) -> (F, G)`, and `run_qinsga3(..., use_route_repair: bool = False, ...)`.

- [ ] **Step 1: Write the failing test for `_build_g_constraints`**

Add to `Solvers/QINSGA3/test_algorithm.py`'s existing import block (find the line starting `from Solvers.QINSGA3.algorithm import (` and add `_build_g_constraints` to the list of imported names — do not create a second import line).

Then append this new section to the end of `Solvers/QINSGA3/test_algorithm.py`:

```python
# ── _build_g_constraints ─────────────────────────────────────────────────
# Remedy G: duplicates IRPProblem._evaluate's G-list construction
# (Solvers/NSGA3/problem.py:67-85) so _evaluate_with_repair can compute G
# for a REPAIRED route_result without calling IRPProblem._evaluate itself
# (which decodes and builds routes internally, with no repair hook). This
# test hand-verifies the duplicate against the same published formula, to
# catch transcription drift.

def test_build_g_constraints_matches_hand_verified_formula():
    """2 periods, 1 client. Hand-computed (mirrors problem.py:67-85 exactly):
      t=1: ret=20 -> [20-100=-80, 0-20=-20]
      t=2: ret=30 -> [30-100=-70, 0-30=-30]
      client 1: t=1: cum_del=8,  cum_dem=10 -> [10-8=2]
                t=2: cum_del=15, cum_dem=15 -> [15-15=0]
      t=1: [depot_stock[1].frigo-50=10-50=-40, depot_stock[1].nonfrigo-50=5-50=-45]
      t=2: [depot_stock[2].frigo-50=60-50=10, depot_stock[2].nonfrigo-50=2-50=-48]
    """
    sets_ = {"clients": [1], "T": [1, 2]}
    params_ = {
        "q_lt": {(1, 1): 10, (1, 2): 5},
        "tau_min": 0.0,
        "tau_max": 100.0,
        "I_O_max_frigo": 50.0,
        "I_O_max_nonfrigo": 50.0,
    }
    route_result = {
        "tau_return": {1: 20.0, 2: 30.0},
        "actual_qty": {(1, 1): 8, (1, 2): 7},
        "depot_stock": {
            1: {"frigo": 10.0, "nonfrigo": 5.0},
            2: {"frigo": 60.0, "nonfrigo": 2.0},
        },
    }

    G = _build_g_constraints(route_result, sets_, params_)

    assert G == [-80, -20, -70, -30, 2, 0, -40, -45, 10, -48]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest Solvers/QINSGA3/test_algorithm.py -k build_g_constraints -v`
Expected: FAIL — `ImportError: cannot import name '_build_g_constraints'`.

- [ ] **Step 3: Implement `_build_g_constraints`, `_evaluate_with_repair`, `_worker_eval_repaired`**

In `Solvers/QINSGA3/algorithm.py`, locate `_worker_eval` (search for `def _worker_eval`, currently just after `_worker_init`):

```python
def _worker_eval(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one solution in a worker process. Returns (F, G)."""
    out: dict = {}
    _g_problem._evaluate(x, out)
    return out["F"], out["G"]
```

Insert immediately after it:

```python
def _build_g_constraints(route_result: dict, sets_: dict, params_: dict) -> list:
    """Mirrors IRPProblem._evaluate's G-list construction exactly
    (Solvers/NSGA3/problem.py:67-85) -- duplicated here, not imported,
    since calling IRPProblem._evaluate directly would decode and build
    routes itself with no repair hook. Kept in sync manually; see
    docs/superpowers/specs/2026-08-04-qinsga3-route-repair-design.md.
    """
    clients  = sets_["clients"]
    T        = sets_["T"]
    q_lt     = params_["q_lt"]
    tau_min  = params_["tau_min"]
    tau_max  = params_["tau_max"]
    I_max_f  = params_["I_O_max_frigo"]
    I_max_nf = params_["I_O_max_nonfrigo"]
    actual      = route_result["actual_qty"]
    depot_stock = route_result["depot_stock"]

    G = []
    for t in T:
        ret = route_result["tau_return"].get(t, 0.0)
        G.append(ret - tau_max)
        G.append(tau_min - ret)

    for l in clients:
        cum_del = cum_dem = 0
        for t in T:
            cum_del += actual.get((l, t), 0)
            cum_dem += q_lt[l, t]
            G.append(cum_dem - cum_del)

    for t in T:
        G.append(depot_stock[t]["frigo"]    - I_max_f)
        G.append(depot_stock[t]["nonfrigo"] - I_max_nf)

    return G


def _evaluate_with_repair(x: np.ndarray, sets_: dict, params_: dict) -> tuple[np.ndarray, np.ndarray]:
    """Remedy G: decode + repair (2-opt, Baldwinian -- see repair.py) +
    evaluate, replacing IRPProblem._evaluate for QINSGA3 only, when
    use_route_repair=True. See
    docs/superpowers/specs/2026-08-04-qinsga3-route-repair-design.md.
    """
    from Solvers.NSGA3.decoder import decode_chromosome, build_routes
    from Solvers.NSGA3.evaluator import compute_f1, compute_f2, compute_f3, compute_f4
    from Solvers.QINSGA3.repair import _repair_route_result

    quantities, priorities = decode_chromosome(x, sets_)
    route_result = build_routes(quantities, sets_, params_, priorities)
    route_result = _repair_route_result(route_result, sets_, params_)

    F = np.array([
        compute_f1(route_result, sets_, params_),
        compute_f2(route_result, sets_, params_),
        compute_f3(route_result, sets_, params_),
        compute_f4(route_result, sets_, params_),
    ])
    G = np.array(_build_g_constraints(route_result, sets_, params_))
    return F, G


def _worker_eval_repaired(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same per-process pattern as _worker_eval, but routes through
    _evaluate_with_repair (remedy G) instead of _g_problem._evaluate."""
    return _evaluate_with_repair(x, _g_problem.sets_, _g_problem.params_)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest Solvers/QINSGA3/test_algorithm.py -k build_g_constraints -v`
Expected: PASS.

- [ ] **Step 5: Wire `use_route_repair` into `run_qinsga3`**

In `Solvers/QINSGA3/algorithm.py`, find `run_qinsga3`'s signature (search for `use_chaotic_rotation: bool = False,`) and add the new parameter right after it:

```python
    use_chaotic_rotation: bool = False,
    use_route_repair: bool = False,
```

Find the docstring paragraph for `use_chaotic_rotation` (ends `"...Mutually exclusive with the other rotation variants."`) and insert a new paragraph immediately after it, still inside the docstring:

```python

    use_route_repair (disabled by default) replaces each worker's call to
    _worker_eval with _worker_eval_repaired, applying a 2-opt local-search
    repair (see Solvers/QINSGA3/repair.py) to every individual's decoded
    route before scoring it, for both parent and offspring populations
    every generation. Baldwinian: the repair never changes the chromosome,
    only the fitness it is scored with. See
    docs/superpowers/specs/2026-08-04-qinsga3-route-repair-design.md.
```

Find `_eval_batch`'s definition inside the `with ProcessPoolExecutor(...) as pool:` block:

```python
        def _eval_batch(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Evaluate all individuals in X in parallel across worker processes."""
            results = list(pool.map(_worker_eval, list(X), chunksize=chunksize))
            return (
                np.array([r[0] for r in results]),
                np.array([r[1] for r in results]),
            )
```

Replace with:

```python
        def _eval_batch(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Evaluate all individuals in X in parallel across worker processes."""
            worker_fn = _worker_eval_repaired if use_route_repair else _worker_eval
            results = list(pool.map(worker_fn, list(X), chunksize=chunksize))
            return (
                np.array([r[0] for r in results]),
                np.array([r[1] for r in results]),
            )
```

- [ ] **Step 6: Sanity-check the module imports cleanly and the parameter is wired**

Run: `python -c "from Solvers.QINSGA3.algorithm import run_qinsga3; import inspect; assert 'use_route_repair' in inspect.signature(run_qinsga3).parameters; print('OK')"`
Expected: prints `OK` with no traceback.

- [ ] **Step 7: Smoke-test `_evaluate_with_repair` end-to-end on a real small instance**

Run this one-off check (not a committed test — just a manual verification that the wiring produces valid F/G shapes without crashing on real data, since a full formal test would need a large, instance-specific fixture):

```bash
python -c "
from models.parametres import load_instance
from Solvers.NSGA3.problem import IRPProblem
from Solvers.QINSGA3.algorithm import _evaluate_with_repair
import numpy as np

sets_, params_ = load_instance('data/instance_3_clients.json')
problem = IRPProblem(sets_, params_)
x = np.array(problem.xl, dtype=float)
F, G = _evaluate_with_repair(x, sets_, params_)
print('F shape:', F.shape, 'G shape:', G.shape)
print('F:', F)
assert F.shape == (4,)
assert len(G) == problem.n_ieq_constr
print('OK')
"
```

Expected: prints `F shape: (4,) G shape: (N,)` (N matching `problem.n_ieq_constr`), the F values, and `OK`, with no traceback. If this raises an exception, do not proceed to Step 8 — report BLOCKED with the full traceback instead.

- [ ] **Step 8: Run the full test suite to confirm no regressions**

Run: `python -m pytest Solvers/QINSGA3/test_algorithm.py Solvers/QINSGA3/test_repair.py -v`
Expected: all tests PASS (existing tests untouched, new ones added).

- [ ] **Step 9: Commit**

```bash
git add Solvers/QINSGA3/algorithm.py Solvers/QINSGA3/test_algorithm.py
git commit -m "feat: wire use_route_repair into run_qinsga3 (remede G, part 4/4)"
```

---

### Task 5: `sensitivity/compare_route_repair.py`

**Files:**
- Create: `sensitivity/compare_route_repair.py`

**Interfaces:**
- Consumes: `run_qinsga3(..., use_route_repair=True)` (Task 4), `Solvers/NSGA3/nsga3_chromosomes.json` cache, `Solvers/NSGA3/{decoder,evaluator,metrics,problem}.py` (all pre-existing, unchanged).
- Produces: a runnable script printing HV/GD/IGD/Spacing, chromosome diversity, and two Mann-Whitney U passes — consumed by Task 6.

- [ ] **Step 1: Create the script**

Structural copy of `sensitivity/compare_crowding_guides.py` (same helpers, same statistical protocol), with the crowding-specific parts swapped for the route-repair variant:

```python
"""Post-decode route repair test for QINSGA-III (remedy G) -- see
docs/superpowers/specs/2026-08-04-qinsga3-route-repair-design.md.

Context: six prior remedies (A-F, see Solvers/IRP_results_summary.md's
"Remedes testes" section) all targeted the rotation gate's guide-selection
mechanism -- reset frequency/target/magnitude, the rotation rule itself,
the guide-selection criterion -- and all were rejected. The project's
retained hypothesis: the limiting factor is Solvers/NSGA3/decoder.py's
greedy, irrevocable construction heuristic, not the rotation mechanism --
a small theta perturbation can flip an early construction choice and
cascade into a wildly different route with a disproportionate objective
jump. This remedy targets that directly: a 2-opt local search repairs each
individual's decoded route (evaluated against the real f1 cost, not raw
distance -- correcting an earlier abandoned 2-opt attempt that used the
wrong metric) before scoring it, every generation, for both parent and
offspring populations. Baldwinian: the chromosome is never modified, only
the fitness assigned to it. Scoped entirely to QINSGA3 -- decoder.py,
evaluator.py, and problem.py (shared with NSGA3) are untouched.

Already wired into production run_qinsga3 as use_route_repair (default
False) -- no separate reimplementation needed.

Usage:
    python -m sensitivity.compare_route_repair
    python -m sensitivity.compare_route_repair --seeds 42 137 271
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(MODULE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np
from pymoo.util.ref_dirs import get_reference_directions
from scipy.stats import mannwhitneyu

from models.parametres         import load_instance
from Solvers.NSGA3.decoder      import decode_chromosome, build_routes
from Solvers.NSGA3.evaluator    import compute_f1, compute_f2, compute_f3, compute_f4
from Solvers.NSGA3.metrics      import compute_pareto_metrics
from Solvers.NSGA3.problem      import IRPProblem
from Solvers.QINSGA3.algorithm  import run_qinsga3

N_PARTITIONS = 8
N_OBJ        = 4
POP_SIZE     = 200
DEFAULT_SEEDS = [42, 137, 271]

_NSGA3_CACHE = os.path.join(PROJECT_DIR, "Solvers", "NSGA3", "nsga3_chromosomes.json")


def _decode_to_F(chromosomes, sets_, params_) -> np.ndarray:
    F = []
    for chrom in chromosomes:
        quantities, priorities = decode_chromosome(np.array(chrom), sets_)
        route_result = build_routes(quantities, sets_, params_, priorities)
        F.append([
            compute_f1(route_result, sets_, params_),
            compute_f2(route_result, sets_, params_),
            compute_f3(route_result, sets_, params_),
            compute_f4(route_result, sets_, params_),
        ])
    return np.array(F)


def _load_nsga3_reference(sets_, params_, instance: str) -> list[np.ndarray]:
    with open(_NSGA3_CACHE, encoding="utf-8") as f:
        cache = json.load(f)
    expected = f"{instance}_clients"
    cached   = cache["runs"][0]["meta_base"]["instance"]
    if cached != expected:
        raise ValueError(f"NSGA-III cache is for '{cached}', not '{expected}'")
    return [_decode_to_F(r["chromosomes"], sets_, params_) for r in cache["runs"]]


def _stats(vals):
    arr = np.array([v for v in vals if v is not None], dtype=float)
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(arr.mean()), "std": float(arr.std())}


def _chrom_diversity(X_list, xl, xu):
    X = np.array(X_list, dtype=float)
    denom = np.where(xu - xl > 1e-9, xu - xl, 1.0)
    P = (X - xl) / denom
    return float(P.var(axis=0).mean())


def run_comparison(instance: str, seeds: list[int], max_gen: int, pop_size: int) -> None:
    data_path = os.path.join(PROJECT_DIR, "data", f"instance_{instance}_clients.json")
    sets_, params_ = load_instance(data_path)
    problem_ref = IRPProblem(sets_, params_)
    xl = np.asarray(problem_ref.xl, dtype=float)
    xu = np.asarray(problem_ref.xu, dtype=float)
    ref_dirs = get_reference_directions("das-dennis", N_OBJ, n_partitions=N_PARTITIONS)
    effective_pop = max(pop_size, len(ref_dirs))

    print("=" * 92)
    print("  ROUTE REPAIR — QINSGA-III (baseline vs NSGA-III cache)")
    print("=" * 92)
    print(f"  Instance : {instance} clients | pop={effective_pop} | gen={max_gen}")
    print(f"  Seeds    : {seeds}")
    print("=" * 92)

    print("\nDecodage du front NSGA-III (cache, reference fixe)...")
    nsga3_F_runs = _load_nsga3_reference(sets_, params_, instance)
    print(f"  NSGA-III : {len(nsga3_F_runs)} runs, "
          f"{sum(len(f) for f in nsga3_F_runs)} solutions")

    baseline_lbl = "baseline (no repair)"
    test_lbl     = "Route repair (test)"
    raw:     dict[str, list] = {baseline_lbl: [], test_lbl: []}
    chrom_X: dict[str, list] = {baseline_lbl: [], test_lbl: []}

    print(f"\n>>> {baseline_lbl} (prod. actuelle)")
    for seed in seeds:
        print(f"  seed={seed} ... ", end="", flush=True)
        t0 = time.time()
        X, pareto_F, _ = run_qinsga3(
            sets_=sets_, params_=params_, ref_dirs=ref_dirs,
            pop_size=effective_pop, max_gen=max_gen, seed=seed,
        )
        elapsed = round(time.time() - t0, 1)
        raw[baseline_lbl].append((seed, pareto_F, elapsed))
        chrom_X[baseline_lbl].extend(X.tolist())
        print(f"front={len(pareto_F) if pareto_F is not None else 0}  time={elapsed}s", flush=True)

    print(f"\n>>> {test_lbl}")
    for seed in seeds:
        print(f"  seed={seed} ... ", end="", flush=True)
        t0 = time.time()
        X, pareto_F, _ = run_qinsga3(
            sets_=sets_, params_=params_, ref_dirs=ref_dirs,
            pop_size=effective_pop, max_gen=max_gen, seed=seed, use_route_repair=True,
        )
        elapsed = round(time.time() - t0, 1)
        raw[test_lbl].append((seed, pareto_F, elapsed))
        chrom_X[test_lbl].extend(X.tolist())
        print(f"front={len(pareto_F) if pareto_F is not None else 0}  time={elapsed}s", flush=True)

    all_F = list(nsga3_F_runs)
    for lbl in raw:
        all_F.extend(F for (_, F, _) in raw[lbl] if F is not None and len(F) > 0)
    all_F_stack  = np.vstack(all_F)
    global_ideal = all_F_stack.min(axis=0)
    global_nadir = all_F_stack.max(axis=0)

    print(f"\nIdeal global partage : {global_ideal}")
    print(f"Nadir global partage  : {global_nadir}")

    groups: dict[str, dict[str, list]] = {}

    nsga3_vals = {"HV": [], "GD": [], "IGD": [], "Spacing": []}
    for F_run in nsga3_F_runs:
        q = compute_pareto_metrics(F_run, global_ideal, global_nadir)
        for k in nsga3_vals:
            nsga3_vals[k].append(q[k])
    groups["NSGA-III (reference)"] = nsga3_vals

    for lbl in (baseline_lbl, test_lbl):
        vals = {"HV": [], "GD": [], "IGD": [], "Spacing": [], "elapsed_s": [], "front_size": []}
        for seed, F, elapsed in raw[lbl]:
            if F is None or len(F) == 0:
                continue
            q = compute_pareto_metrics(F, global_ideal, global_nadir)
            for k in ("HV", "GD", "IGD", "Spacing"):
                vals[k].append(q[k])
            vals["elapsed_s"].append(elapsed)
            vals["front_size"].append(len(F))
        groups[lbl] = vals

    print(f"\n{'-'*92}")
    print("  RESULTATS (ideal/nadir global partage NSGA-III + baseline + test)")
    print(f"{'-'*92}")
    for metric, higher in (("HV", True), ("GD", False), ("IGD", False), ("Spacing", False)):
        arrow = "^" if higher else "v"
        print(f"\n  {metric} ({arrow})")
        for name, vals in groups.items():
            s = _stats(vals[metric])
            print(f"    {name:<32} mean={s['mean']:.6f}  std={s['std']:.6f}")

    print(f"\n{'-'*92}")
    print("  Diversite chromosome  Var(X_norm)  (pooled sur tous les runs testes)")
    print(f"{'-'*92}")
    for lbl in (baseline_lbl, test_lbl):
        d = _chrom_diversity(chrom_X[lbl], xl, xu)
        print(f"    {lbl:<32} {d:.6f}")

    print(f"\n{'-'*92}")
    print(f"  Mann-Whitney U : baseline vs {test_lbl}")
    print(f"{'-'*92}")
    for metric in ("HV", "GD", "IGD", "Spacing"):
        a, b = groups[baseline_lbl][metric], groups[test_lbl][metric]
        if len(a) < 2 or len(b) < 2:
            print(f"  {metric:<8} : pas assez de runs valides pour un test")
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        sig = "significatif (p<0.05)" if p < 0.05 else "non significatif"
        print(f"  {metric:<8} U={u:.1f}  p={p:.6f}  -> {sig}")

    print(f"\n{'-'*92}")
    print(f"  Mann-Whitney U : {test_lbl} vs NSGA-III (l'objectif final)")
    print(f"{'-'*92}")
    for metric in ("HV", "GD", "IGD", "Spacing"):
        a, b = groups[test_lbl][metric], groups["NSGA-III (reference)"][metric]
        if len(a) < 2 or len(b) < 2:
            print(f"  {metric:<8} : pas assez de runs valides pour un test")
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        sig = "significatif (p<0.05)" if p < 0.05 else "non significatif"
        print(f"  {metric:<8} U={u:.1f}  p={p:.6f}  -> {sig}")

    print(f"\n{'-'*92}")
    print("  Temps moyen par run")
    print(f"{'-'*92}")
    for lbl in (baseline_lbl, test_lbl):
        s = _stats(groups[lbl]["elapsed_s"])
        print(f"    {lbl:<32} mean={s['mean']:.1f}s")

    print(f"\n{'='*92}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test de la reparation de tournee post-decodage — QINSGA-III")
    parser.add_argument("--instance", default="100",
                        choices=["3", "5", "15", "25", "30", "40", "100"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--gen", type=int, default=300)
    parser.add_argument("--pop", type=int, default=POP_SIZE)
    args = parser.parse_args()
    run_comparison(args.instance, args.seeds, args.gen, args.pop)
```

- [ ] **Step 2: Smoke-test the script on a tiny, fast configuration**

Run: `python -m sensitivity.compare_route_repair --instance 3 --seeds 42 --gen 3 --pop 20`
Expected: completes without traceback (may take noticeably longer than a non-repair run of the same size, given the repair's per-candidate f1 recomputation — that is expected and exactly what Task 6's timing check quantifies before committing to the full campaign). Prints all report sections; Mann-Whitney blocks will say "pas assez de runs valides pour un test" with only 1 seed, which is correct at this smoke-test scale.

- [ ] **Step 3: Commit**

```bash
git add sensitivity/compare_route_repair.py
git commit -m "test: add compare_route_repair.py sensitivity script (remede G)"
```

---

### Task 6: Timing check, empirical validation, and documentation

**Files:**
- Modify: `Solvers/IRP_results_summary.md`
- Create: `sensitivity/route_repair_campaign_log.txt` (raw run output)

**Interfaces:**
- Consumes: `sensitivity/compare_route_repair.py` (Task 5).
- Produces: a new "Remède G" section in `Solvers/IRP_results_summary.md`, following the exact structure of Remèdes A-F.

- [ ] **Step 1: Run the small-scale timing check (spec's Risk section)**

Before committing to a full 300-generation campaign, quantify the repair's runtime cost, per the design spec's explicit risk callout:

```bash
python -m sensitivity.compare_route_repair --instance 5 --seeds 42 --gen 10 --pop 50
```

Read the printed "Temps moyen par run" section. Compute the ratio: `test_lbl`'s mean elapsed time ÷ `baseline_lbl`'s mean elapsed time.

- If the ratio is under ~3x: proceed to Step 2 at full scale.
- If the ratio is 3x-10x: proceed to Step 2, but note in your final report that the full 3-seed/300-gen campaign may take proportionally longer than prior remedies' ~208-220s/run baseline (so a 5x ratio implies roughly 1000-1100s/run, i.e. the whole 3-seed campaign could take on the order of 1-2 hours) — still proceed, just flag the expected wall-clock time.
- If the ratio is over 10x: STOP. Do not run the full campaign. Report BLOCKED with the measured ratio and ask the controller how to proceed (e.g., lowering `_MAX_REPAIR_ITER` in `Solvers/QINSGA3/repair.py`, or accepting the cost) rather than launching a run that could take many hours unsupervised.

- [ ] **Step 2: Run the full-scale 3-seed validation**

Assuming Step 1 did not block:

```bash
python -m sensitivity.compare_route_repair --instance 100 --seeds 42 137 271 --gen 300 --pop 200 > sensitivity/route_repair_campaign_log.txt 2>&1
```

Run this in the background and wait for it to finish — expected wall-clock time is at least the ~208-220s × 3 seeds × 2 arms (~21-22 min) baseline scale, scaled by the ratio measured in Step 1.

- [ ] **Step 3: Read the log and extract the numbers**

Read `sensitivity/route_repair_campaign_log.txt`. Extract: HV/GD/IGD/Spacing means for `baseline (no repair)` and `Route repair (test)`; both chromosome-diversity values; Mann-Whitney U/p for both comparison blocks.

- [ ] **Step 4: If there's a positive signal, scale to 5 seeds**

If HV shows a higher mean for the test arm AND the design spec's precedent applies (Remède F's own validation showed 3 seeds cannot reach p<0.05 against baseline — exact floor p=0.10 at n=3), do not treat a merely-higher-mean-but-p=0.10-boundary result as inconclusively negative. Re-run with 5 seeds to get a real read, mirroring exactly how Remède F was re-validated:

```bash
python -m sensitivity.compare_route_repair --instance 100 --seeds 42 137 271 314 512 --gen 300 --pop 200 > sensitivity/route_repair_5seed_campaign_log.txt 2>&1
```

If the 3-seed result is not even nominally favorable on HV (test mean <= baseline mean), skip this step — there's no signal to confirm.

- [ ] **Step 5: Insert the "Remède G" section into `Solvers/IRP_results_summary.md`**

Find the Remède F section (search for `### Remède F`) and its Conclusion bullet (search for `Le remède F, qui change`). Insert a new subsection immediately after Remède F's "Confirmation (5 seeds...)" block and before `## Conclusion`, using this structure (fill in `<VALUE>` from your actual run's numbers — every other word is final text):

```markdown
### Remède G -- réparation de tournée post-décodage

Contrairement aux remèdes A-F (tous centrés sur le mécanisme de
guide/rotation), ce remède s'attaque directement à la cause racine
identifiée par le diagnostic : le décodeur glouton et irrévocable
(`Solvers/NSGA3/decoder.py::_nearest_neighbour`). Une réparation locale
2-opt, appliquée après décodage sur chaque individu (parents ET enfants, à
chaque génération), corrige un défaut d'une tentative abandonnée très tôt
dans ce projet (voir "Premières tentatives" ci-dessus) qui optimisait la
distance brute au lieu du vrai coût f1 (transport + stockage + pénalités
de fenêtres de temps). Réparation baldwinienne (améliore uniquement la
fitness évaluée, jamais le chromosome) et scopée entièrement à QINSGA3
(`Solvers/QINSGA3/repair.py`, nouveau module -- `decoder.py`/`evaluator.py`/
`problem.py`, partagés avec NSGA3, ne sont pas modifiés). Design complet :
`docs/superpowers/specs/2026-08-04-qinsga3-route-repair-design.md`.

**Résultat (3 seeds, 300 générations, instance 100 clients,
`sensitivity/compare_route_repair.py`)** :

| Indicateur | Baseline (sans réparation) | Test (réparation 2-opt) | Mann-Whitney |
|---|---|---|---|
| HV ↑ | <VALUE> | <VALUE> | U=<VALUE>, p=<VALUE> |
| GD ↓ | <VALUE> | <VALUE> | U=<VALUE>, p=<VALUE> |
| IGD ↓ | <VALUE> | <VALUE> | U=<VALUE>, p=<VALUE> |
| Diversité chromosome | <VALUE> | <VALUE> | -- |

Script conservé : `sensitivity/compare_route_repair.py`. Log complet :
`sensitivity/route_repair_campaign_log.txt`.
```

Then append ONE interpretation paragraph, choosing the template that matches your actual Mann-Whitney result:

If non-significant on HV (same pattern as A-F):

```markdown
**Septième remède indépendant, même verdict** : contrairement à A-F, celui-ci
s'attaquait directement au décodeur plutôt qu'au mécanisme de guide/rotation
-- et pourtant le résultat reste non significatif face au baseline. Ceci
renforce encore l'hypothèse retenue pour le mémoire : même une réparation
locale du symptôme le plus directement identifié (les tournées elles-mêmes,
via le vrai coût f1) ne suffit pas à combler l'écart avec NSGA-III sur
l'IRP -- la limite structurelle touche quelque chose de plus profond que la
qualité locale d'une tournée individuelle.
```

If significant and favorable on HV (p<0.05, higher mean than baseline) — using either the 3-seed or 5-seed result, whichever crossed the threshold:

```markdown
**Résultat positif** : contrairement aux six remèdes précédents, s'attaquer
directement au décodeur (plutôt qu'au mécanisme de guide/rotation) améliore
significativement la qualité du front. Ceci confirme directement
l'hypothèse retenue : le décodeur chaotique, pas le mécanisme quantique,
était le facteur limitant sur l'IRP.
```

If you used 5 seeds (Step 4), also append a short paragraph analogous to Remède F's "Confirmation (5 seeds)" subsection with the 5-seed table and log filename, following that section's exact format as a template.

- [ ] **Step 6: Update the Conclusion's bullet list**

In the same file's `## Conclusion` section, after the existing Remède F bullet, add one more bullet consistent with your actual outcome — if rejected:

```markdown
- Le remède G, la première tentative à s'attaquer directement au décodeur
  plutôt qu'au mécanisme de guide/rotation (réparation locale 2-opt
  post-décodage, évaluée contre le vrai coût f1), a vu <décrire la
  direction réelle observée sur la diversité chromosome et la qualité du
  front> -- septième mécanisme indépendant, et le premier à cibler la
  cause racine identifiée par le diagnostic plutôt que ses symptômes en
  aval, avec le même résultat : pas de gain significatif face à NSGA-III
  sur l'IRP.
```

(replace the `<...>` clause with the real observed direction; remove the angle brackets). If the result was positive instead, do not write a scripted bullet — flag this prominently in your final report to the controller, since a positive Remède G outcome would be the first success across seven remedies and directly validates the project's retained root-cause hypothesis, meaningfully changing the thesis narrative in a way that needs the project owner's own judgment.

- [ ] **Step 7: Commit**

```bash
git add Solvers/IRP_results_summary.md sensitivity/route_repair_campaign_log.txt
# also add sensitivity/route_repair_5seed_campaign_log.txt if Step 4 ran
git commit -m "docs: document remede G (post-decode route repair) result on the IRP"
```
