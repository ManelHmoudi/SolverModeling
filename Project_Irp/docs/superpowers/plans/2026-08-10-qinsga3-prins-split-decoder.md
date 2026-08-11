# QINSGA3 Prins-split decoder — continuity screening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a continuity smoke test comparing the production greedy decoder, the existing giant-tour+greedy-fill prototype, and a new Prins (2004) DP-optimal split decoder, to check whether the DP split restores a real theta→route continuity gradient.

**Architecture:** A standalone, testable DP module (`sensitivity/prins_split.py`) computes the optimal partition of a fixed client visit order into per-truck routes. A diagnose script (`sensitivity/diagnose_prins_split_continuity.py`), mirroring the existing `diagnose_giant_tour_continuity.py`, monkey-patches this decoder in for the duration of each decode call and measures Jaccard route-distance under theta perturbation at several points along the real production alpha schedule, alongside the two existing decoders for a three-way comparison.

**Tech Stack:** Python, numpy, pytest. No new dependencies.

## Global Constraints

- `Solvers/NSGA3/decoder.py` must never be modified on disk — every decoder variant is a private, monkey-patched fork, restored in a `finally` block (same isolation convention as every prior remedy in `sensitivity/`).
- No change to `Solvers/QINSGA3/algorithm.py` — this stays ablation-only in `sensitivity/` regardless of outcome (see spec's Non-goals).
- Split minimizes distance only (no multi-objective DP) — matches both the production greedy score and the giant-tour prototype's greedy fill.
- Segment cost uses each client's full requested quantity, with a floor-quantity fallback for mandatory clients only when the full-quantity segment doesn't fit capacity — documented simplification, not full production-style context-dependent quantity reduction.

Spec: `docs/superpowers/specs/2026-08-10-qinsga3-prins-split-decoder-design.md`

---

### Task 1: Prins DP-split core (`sensitivity/prins_split.py`)

**Files:**
- Create: `sensitivity/prins_split.py`
- Create: `sensitivity/test_prins_split.py`

**Interfaces:**
- Consumes: `_giant_tour_order(qty_dict, floors, priorities)` from `sensitivity/diagnose_giant_tour_continuity.py` (existing, unmodified — returns `list` of client ids, mandatory-first then priority-descending).
- Produces:
  - `_segment_metrics(order, i, j, qty_dict, floors, d, v_k, s, O, tau_max)` → `None` or `(distance: float, load_full: float, qty_full: dict, load_floor: float, qty_floor: dict)`.
  - `_prins_split_from_order(order, qty_dict, trucks, t, d, v, s, Q, O, tau_max=None, floors=None)` → `(routes, x_vars, f_vars, arrivals, assign)`, same shape as `Solvers.NSGA3.decoder._nearest_neighbour`.
  - `_prins_split(qty_dict, trucks, t, d, v, s, Q, O, tau_max=None, floors=None, priorities=None)` → same shape, drop-in replacement for `_nearest_neighbour` (computes `order` internally via `_giant_tour_order`).

- [ ] **Step 1: Write `sensitivity/prins_split.py`**

```python
"""Prins (2004) DP-optimal split of a fixed giant-tour order into vehicle
routes, adapted to production's heterogeneous ordered fleet via a layered
DP: the k-th contiguous segment in the fixed order is always assigned to
`trucks[k-1]` -- mirrors `Solvers/NSGA3/decoder.py::_nearest_neighbour`'s
sequential truck consumption (truck k+1 is only ever used once truck k has
been tried, including the case where truck k ends up serving nobody).

Unlike a single left-to-right greedy fill (`diagnose_giant_tour_continuity.
py::_giant_tour_split`, which always extends the current truck's route as
far as capacity/tau_max allow before moving on), this DP evaluates ALL
contiguous partitions of the fixed order and picks the one minimising total
distance -- a "maximal fill" is not always distance-optimal (e.g. it can
pair a far-away client with a near one on the same truck just because
capacity allows it, when leaving that far client for the next truck --
paired with clients actually close to it -- would cost much less overall).

See docs/superpowers/specs/2026-08-10-qinsga3-prins-split-decoder-design.md.
"""
from __future__ import annotations

from sensitivity.diagnose_giant_tour_continuity import _giant_tour_order


def _segment_metrics(order, i, j, qty_dict, floors, d, v_k, s, O, tau_max):
    """Metrics for assigning order[i:j] (contiguous client ids) to one truck
    of speed v_k, visited in the given fixed order. Returns None if a
    non-mandatory client's return-trip projection exceeds tau_max --
    mandatory clients (floor > 0) bypass the check, same as
    `_nearest_neighbour`. Otherwise returns (distance, load_full, qty_full,
    load_floor, qty_floor): *_full uses each client's full requested
    quantity; *_floor substitutes the mandatory floor for mandatory clients
    only (non-mandatory quantities are identical in both) -- the caller
    picks whichever fits the truck's capacity.
    """
    floors = floors or {}
    distance = 0.0
    current_time = 0.0
    current = O
    qty_full, qty_floor = {}, {}
    load_full = load_floor = 0

    for idx in range(i, j):
        l = order[idx]
        floor_l = floors.get(l, 0)
        is_mandatory = floor_l > 0
        dist = d[current, l]

        if tau_max is not None and not is_mandatory:
            projected = (current_time + s.get(current, 0.0) + dist / v_k
                         + s.get(l, 0.0) + d[l, O] / v_k)
            if projected > tau_max:
                return None

        current_time += s.get(current, 0.0) + dist / v_k
        distance += dist

        q_full  = qty_dict[l]
        q_floor = floor_l if is_mandatory else q_full
        qty_full[l]  = q_full
        qty_floor[l] = q_floor
        load_full  += q_full
        load_floor += q_floor
        current = l

    distance += d[current, O]
    return distance, load_full, qty_full, load_floor, qty_floor


def _prins_split_from_order(order, qty_dict, trucks, t, d, v, s, Q, O,
                             tau_max=None, floors=None):
    """DP-optimal partition of `order` into contiguous per-truck segments,
    `trucks[k-1]` serving the k-th segment. Returns the same 5-tuple shape
    as `_nearest_neighbour`: (routes, x_vars, f_vars, arrivals, assign).
    """
    floors = floors or {}
    n = len(order)
    K = len(trucks)
    INF = float("inf")

    dp     = [[INF] * (n + 1) for _ in range(K + 1)]
    origin = [[None] * (n + 1) for _ in range(K + 1)]
    dp[0][0] = 0.0

    for k in range(1, K + 1):
        truck = trucks[k - 1]
        v_k, cap = v[truck], Q[truck]
        dp[k]     = list(dp[k - 1])       # truck k serves nobody (carry forward)
        origin[k] = list(origin[k - 1])

        for j in range(0, n + 1):
            if dp[k - 1][j] == INF:
                continue
            for j2 in range(j + 1, n + 1):
                metrics = _segment_metrics(order, j, j2, qty_dict, floors, d, v_k, s, O, tau_max)
                if metrics is None:
                    break
                distance, load_full, qty_full, load_floor, qty_floor = metrics
                if load_full <= cap:
                    qty_on_route = qty_full
                elif load_floor <= cap:
                    qty_on_route = qty_floor
                else:
                    break
                candidate = dp[k - 1][j] + distance
                if candidate < dp[k][j2]:
                    dp[k][j2]     = candidate
                    origin[k][j2] = (j, qty_on_route)

    served_upto = n if dp[K][n] < INF else max(
        (j for j in range(n + 1) if dp[K][j] < INF), default=0)

    segments = []
    k, j = K, served_upto
    while j > 0:
        if origin[k][j] is None:
            k -= 1
            continue
        prev_j, qty_on_route = origin[k][j]
        segments.append((trucks[k - 1], order[prev_j:j], qty_on_route))
        j = prev_j
        k -= 1
    segments.reverse()

    routes, x_vars, f_vars, arrivals, assign = {}, {}, {}, {}, {}
    for truck, clients_in_segment, qty_on_route in segments:
        speed = v[truck]
        path  = [O] + list(clients_in_segment) + [O]
        m     = len(path)
        suf   = [0] * (m + 1)
        for pos in range(m - 2, -1, -1):
            node = path[pos + 1]
            suf[pos] = suf[pos + 1] + (qty_on_route.get(node, 0) if node != O else 0)

        current_time = 0.0
        current = O
        for pos in range(m - 1):
            i_node, j_node = path[pos], path[pos + 1]
            x_vars[i_node, j_node, t, truck] = 1
            f_vars[i_node, j_node, t, truck] = suf[pos]
            if j_node != O:
                current_time += s.get(current, 0.0) + d[current, j_node] / speed
                arrivals[j_node, t] = current_time
                assign[j_node, t]   = truck
            current = j_node

        routes[truck] = {
            "path": path,
            "qty":  {str(l): q for l, q in qty_on_route.items()},
        }

    return routes, x_vars, f_vars, arrivals, assign


def _prins_split(qty_dict, trucks, t, d, v, s, Q, O, tau_max=None, floors=None,
                  priorities=None):
    """Same call signature as `_nearest_neighbour` -- drop-in monkey-patch
    target. Computes the fixed giant-tour order internally, then delegates
    to `_prins_split_from_order`."""
    order = _giant_tour_order(qty_dict, floors, priorities)
    return _prins_split_from_order(order, qty_dict, trucks, t, d, v, s, Q, O,
                                    tau_max=tau_max, floors=floors)
```

- [ ] **Step 2: Write `sensitivity/test_prins_split.py`**

```python
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
```

- [ ] **Step 3: Run the tests and confirm they fail before implementation exists**

Skip — implementation is written in Step 1 alongside the tests (both are new files with no prior partial state to regress from). Proceed directly to Step 4.

- [ ] **Step 4: Run the tests**

Run: `cd sensitivity && python -m pytest test_prins_split.py -v`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add sensitivity/prins_split.py sensitivity/test_prins_split.py
git commit -m "$(cat <<'EOF'
feat: Prins DP-optimal split decoder for QINSGA3 continuity screening

Layered DP over the existing giant-tour fixed order, adapted to
production's heterogeneous ordered fleet. Targets the diagnosed failure
of the greedy-fill giant-tour attempt (split step still cascading).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Continuity smoke test script and run

**Files:**
- Create: `sensitivity/diagnose_prins_split_continuity.py`
- Modify: `docs/superpowers/specs/2026-08-10-qinsga3-prins-split-decoder-design.md` (append Result section)

**Interfaces:**
- Consumes: `_prins_split` from `sensitivity/prins_split.py` (Task 1); `decode_chromosome`, `Solvers.NSGA3.decoder` module (`build_routes`, `_nearest_neighbour`) from production; `_build_routes_giant_tour` from `sensitivity/diagnose_giant_tour_continuity.py`; `_route_arcset`/`_jaccard_distance` from `sensitivity/test_route_diversity.py`; `_encode_theta` from `Solvers/QINSGA3/algorithm.py`.
- Produces: printed 4-column comparison table (gen, alpha, Jaccard original, Jaccard giant-tour+greedy, Jaccard Prins-split) and a correlation summary — this is the decision gate for Task 3 (a full quality campaign), not built in this plan.

- [ ] **Step 1: Write `sensitivity/diagnose_prins_split_continuity.py`**

```python
"""Smoke test (not a performance comparison): does the Prins DP-optimal
split (`sensitivity/prins_split.py`) restore theta -> route continuity
where the greedy-fill giant-tour attempt
(`diagnose_giant_tour_continuity.py`) failed?

Context: `docs/superpowers/specs/2026-08-10-qinsga3-prins-split-decoder-
design.md`. Same real cached NSGA-III chromosomes, same sparse alpha sweep,
and same Jaccard route-distance metric as `diagnose_giant_tour_continuity.
py`, extended to a three-way comparison (original greedy / giant-tour
greedy-fill / Prins DP-split).

Isolation: `Solvers/NSGA3/decoder.py` is NEVER modified on disk -- each
alternative decoder is monkey-patched onto `_nearest_neighbour` only for
the duration of its own `build_routes` call, restored in `finally`.

Usage:
    python -m sensitivity.diagnose_prins_split_continuity
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
from sensitivity.test_route_diversity          import _route_arcset, _jaccard_distance
from sensitivity.diagnose_giant_tour_continuity import _build_routes_giant_tour
from sensitivity.prins_split                    import _prins_split

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


def main() -> None:
    data_path = os.path.join(PROJECT_DIR, "data", f"instance_{INSTANCE}_clients.json")
    sets_, params_ = load_instance(data_path)
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

    print("\nDecodage des baselines (trois decodeurs, routes de reference)...")
    base_arcs_original, base_arcs_giant_tour, base_arcs_prins = [], [], []
    for i in range(N):
        q, p = decode_chromosome(baseline_X[i], sets_)
        rr_o = decoder_mod.build_routes(q, sets_, params_, p)
        rr_g = _build_routes_giant_tour(q, sets_, params_, p)
        rr_p = _build_routes_prins_split(q, sets_, params_, p)
        base_arcs_original.append(_route_arcset(rr_o))
        base_arcs_giant_tour.append(_route_arcset(rr_g))
        base_arcs_prins.append(_route_arcset(rr_p))

    print("\n" + "=" * 120)
    print("  CONTINUITE theta -> route : original (NN glouton) vs giant-tour+greedy-fill vs Prins DP-split")
    print("=" * 120)
    print(f"  {'gen':<6} {'alpha':<10} {'Jaccard original':<20} {'Jaccard giant-tour':<20} {'Jaccard Prins-split':<20}")
    print("-" * 120)

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

        d_original, d_giant_tour, d_prins = [], [], []
        for i in range(N):
            q, p = decode_chromosome(X_pert[i], sets_)
            rr_o = decoder_mod.build_routes(q, sets_, params_, p)
            rr_g = _build_routes_giant_tour(q, sets_, params_, p)
            rr_p = _build_routes_prins_split(q, sets_, params_, p)
            d_original.append(_jaccard_distance(base_arcs_original[i],   _route_arcset(rr_o)))
            d_giant_tour.append(_jaccard_distance(base_arcs_giant_tour[i], _route_arcset(rr_g)))
            d_prins.append(_jaccard_distance(base_arcs_prins[i],          _route_arcset(rr_p)))

        m_o, m_g, m_p = float(np.mean(d_original)), float(np.mean(d_giant_tour)), float(np.mean(d_prins))
        results.append((gen, alpha, m_o, m_g, m_p))
        print(f"  {gen:<6} {alpha:<10.5f} {m_o:<20.4f} {m_g:<20.4f} {m_p:<20.4f}")

    print("-" * 120)
    alphas  = np.array([r[1] for r in results])
    m_orig  = np.array([r[2] for r in results])
    m_giant = np.array([r[3] for r in results])
    m_prins = np.array([r[4] for r in results])

    def _corr(y):
        return float(np.corrcoef(alphas, y)[0, 1]) if len(alphas) >= 3 else float("nan")

    i_min, i_max = int(np.argmin(alphas)), int(np.argmax(alphas))
    print(f"\n  Jaccard au plus petit alpha teste ({alphas[i_min]:.5f}) : "
          f"original={m_orig[i_min]:.4f}  giant-tour={m_giant[i_min]:.4f}  prins-split={m_prins[i_min]:.4f}")
    print(f"  Jaccard au plus grand alpha teste  ({alphas[i_max]:.5f}) : "
          f"original={m_orig[i_max]:.4f}  giant-tour={m_giant[i_max]:.4f}  prins-split={m_prins[i_max]:.4f}")
    print(f"  Correlation (alpha, Jaccard) -- original     : r={_corr(m_orig):.4f}")
    print(f"  Correlation (alpha, Jaccard) -- giant-tour   : r={_corr(m_giant):.4f}")
    print(f"  Correlation (alpha, Jaccard) -- prins-split  : r={_corr(m_prins):.4f}")

    print("\nINTERPRETATION")
    print("  Continuite restauree si : (1) Jaccard prins-split au plus petit alpha est")
    print("  nettement plus proche de 0 que original/giant-tour au meme alpha, ET (2) la")
    print("  correlation (alpha, Jaccard) est nettement positive pour prins-split (le Jaccard")
    print("  grandit avec alpha, un vrai gradient) alors que original/giant-tour saturent deja au")
    print("  plus petit alpha teste.")
    print(f"{'='*120}\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the smoke test**

Run: `python -m sensitivity.diagnose_prins_split_continuity`
Expected: completes without error, prints the three-way Jaccard table across `GENS_TO_TEST` and the correlation summary.

- [ ] **Step 3: Record the result**

Append a `## Result` section to
`docs/superpowers/specs/2026-08-10-qinsga3-prins-split-decoder-design.md`
with the printed table and correlation values, and state the decision:
scale to `sensitivity/compare_prins_split.py` (Task 2 of the design's Scope
section — a follow-up plan, not part of this one) only if Prins-split shows
a real gradient unlike the other two; otherwise mark it a null result, same
convention as every rejected remedy in `Solvers/IRP_results_summary.md`.

- [ ] **Step 4: Commit**

```bash
git add sensitivity/diagnose_prins_split_continuity.py \
        docs/superpowers/specs/2026-08-10-qinsga3-prins-split-decoder-design.md
git commit -m "$(cat <<'EOF'
test: three-way continuity smoke test for the Prins-split decoder

Compares original greedy / giant-tour greedy-fill / Prins DP-split under
theta perturbation across the real production alpha schedule. Records the
result in the design spec to decide whether a full quality campaign
(compare_prins_split.py) is worth running.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

## Non-goals (this plan)

No `sensitivity/compare_prins_split.py` quality campaign — that is
conditional on this plan's Task 2 result and, if warranted, is a follow-up
plan of its own (same protocol as every other remedy: 3 seeds first, scale
on positive signal). No change to any file outside `sensitivity/` and the
design spec.
