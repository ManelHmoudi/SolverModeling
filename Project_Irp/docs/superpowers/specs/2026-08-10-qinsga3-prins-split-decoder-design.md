# QINSGA3 Prins-split decoder — continuity screening design

## Context

`Solvers/IRP_results_summary.md`'s diagnostic chapter established that the
production decoder (`_nearest_neighbour`, `Solvers/NSGA3/decoder.py:331`) is
discontinuous, not just insensitive: a small theta perturbation flips one
early greedy choice, which cascades through every downstream distance
(`test_theta_route_sensitivity_weighted.py`: only 1.12% of routes unchanged
on average across a full run's alpha schedule). Eleven independent remedies
(A-J) that tuned the rotation mechanism or the decoder's selection rule
failed or were neutral — the only production-adopted fix (`repair_final_front`)
patches the symptom after decoding rather than restoring continuity.

`sensitivity/diagnose_giant_tour_continuity.py` (uncommitted, this session)
tried a genuinely different construction principle: a fixed priority-only
visit order (the "giant tour"), split into truck routes by a single
left-to-right greedy capacity/tau_max fill. It failed — its own docstring
diagnoses why: the split step is "just as sequential/cascading as the
original, just relocated from distance to capacity-boundary." Two further
attempts this session (`prototype_incremental_decoder.py`, incremental
relocation from the parent route; `compare_damped_priority.py`, damping the
priority term in the greedy score) also failed to produce a significant,
adoptable improvement.

This design targets the specific, already-diagnosed failure mode of the
giant-tour attempt: keep its fixed order (that part was never shown to be
the problem), replace only the greedy fill with the actual Prins (2004)
split — the DP-optimal partition of a fixed giant tour into vehicle routes,
via shortest path over an auxiliary DAG. This is the standard reason
BRKGA/random-key VRP decoders are smooth in practice: an optimal split is
computed globally from a fixed order, with no "current position" feedback
loop to cascade errors.

## Goal

Check whether replacing greedy capacity-fill with Prins' DP-optimal split
(same fixed giant-tour order) restores a real theta -> route continuity
gradient — Jaccard route-distance shrinking toward 0 as alpha shrinks,
instead of saturating near its ceiling at the smallest production alpha, as
both the original decoder and the giant-tour+greedy-fill attempt do.

## Design

### Order (reused, unchanged)

`_giant_tour_order(qty_dict, floors, priorities)`, imported directly from
`sensitivity/diagnose_giant_tour_continuity.py` (same cross-import
convention already used by `prototype_incremental_decoder.py`, which imports
`_relocate_client` from `diagnose_incremental_decoder_continuity.py`).
Mandatory clients (floor > 0) first, then the rest, both sorted by priority
descending, tie-broken by client id — never modified by this design.

### Split (new): layered DP, one layer per truck

Classical Prins split assumes a homogeneous fleet; production (and the
giant-tour prototype) uses a heterogeneous, ordered fleet (`trucks`, each
with its own `Q[k]`, `v[k]`, consumed in sequence — truck `i+1` is only used
once truck `i` is exhausted or infeasible). The split is adapted to a
layered DP to preserve this:

- `dp[k][j]` = minimum total distance to have served the first `j` clients
  of the fixed order using exactly the first `k` trucks of `trucks`, in
  order.
- Transition: `dp[k][j] = min over i < j of dp[k-1][i] + segment_cost(i, j, trucks[k-1])`,
  defined only when segment `order[i+1..j]` is feasible for truck `trucks[k-1]`
  (capacity `Q[trucks[k-1]]`, `tau_max` for non-mandatory legs, mandatory
  floor-quantity fallback when a mandatory client's full quantity doesn't
  fit — same feasibility rules production and the giant-tour prototype
  already apply).
- `segment_cost(i, j, k)` = depot-to-first + consecutive + last-to-depot
  distance for `order[i+1..j]` under truck `k`'s speed, using each client's
  full requested quantity except mandatory floor fallback when needed.
- Answer: `dp[K][n]` where `K = len(trucks)`, `n = len(order)`; clients
  beyond what the full fleet can reach stay unserved this period (same
  documented approximation the giant-tour prototype already uses and
  accepts).
- Complexity O(n²·K) per (period, frigo/nonfrigo group) — n ≲ ~100,
  K ≲ ~6 at this instance's scale, cheap for a screening script.

Known simplification, documented rather than hidden (same project
convention as every prior remedy): segment cost uses each client's full
requested quantity, not a load-context-dependent effective quantity the way
the greedy version's mid-route "does it still fit" check works — a segment
either fits as given (with mandatory floor fallback) or the edge doesn't
exist in the DAG.

### Isolation

Same monkey-patch pattern as every remedy so far: a private
`_prins_split(...)` matching `_nearest_neighbour`'s call signature, swapped
onto `Solvers.NSGA3.decoder._nearest_neighbour` only for the duration of a
`build_routes` call, restored in `finally`. `Solvers/NSGA3/decoder.py` is
never modified on disk.

## Scope

### Step 1 — continuity smoke test (decision gate)

`sensitivity/diagnose_prins_split_continuity.py`, mirroring
`diagnose_giant_tour_continuity.py` exactly: same real cached NSGA-III
100-client chromosomes, same sparse alpha sweep
(`GENS_TO_TEST = [0, 50, 100, 150, 200, 250, 300]`), same Jaccard
route-distance metric (`test_route_diversity.py`'s `_route_arcset`/
`_jaccard_distance`, unmodified), same interpretation criteria (gradient
correlating with alpha, ceiling at smallest alpha broken). Adds the
Prins-split column alongside the existing original/giant-tour-greedy
columns for direct three-way comparison.

### Step 2 — quality campaign (conditional)

Only if step 1 shows a real gradient (unlike both existing decoders, which
saturate immediately): `sensitivity/compare_prins_split.py`, same standard
protocol as every other remedy in this project (3 seeds first, shared
ideal/nadir with the NSGA-III cache, HV/GD/IGD/Spacing, chromosome
diversity, Mann-Whitney U vs. baseline and vs. NSGA-III, scale to more seeds
only on positive signal).

If step 1 shows no real gradient (saturates like giant-tour+greedy-fill
did), stop there and document the null result in
`Solvers/IRP_results_summary.md` alongside the other rejected remedies —
step 2 is not worth running.

## Non-goals

No change to `Solvers/QINSGA3/algorithm.py` or `Solvers/NSGA3/decoder.py`
regardless of outcome at this stage — this stays ablation-only in
`sensitivity/`, same convention as every prior remedy (production adoption,
if ever warranted, is a separate later decision after a significant,
scaled result — see how `repair_final_front` was adopted only after 20-seed
confirmation). No multi-objective DP: the split minimizes distance only,
consistent with both the production greedy score (`dist / (0.5 + priority)`)
and the giant-tour prototype's greedy fill, neither of which optimizes
CO2/travel-time during construction either — all four objectives are
measured after decoding, in `evaluator.py`, unchanged. No attempt to make
segment cost load-context-dependent (documented simplification above).
