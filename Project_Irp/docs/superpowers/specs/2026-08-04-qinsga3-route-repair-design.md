# QINSGA3 post-decode route repair (Remède G) — design

## Context

`Solvers/IRP_results_summary.md`'s diagnostic chapter found the root cause of
QI-NSGA-III underperforming NSGA-III on the real IRP (100-client instance),
despite winning 17/25 DTLZ/MaF benchmark cases: `Solvers/NSGA3/decoder.py`'s
`_nearest_neighbour()` construction is a greedy, irrevocable heuristic — a
small θ perturbation can flip an early construction choice and cascade into
a wildly different route, with a disproportionate objective jump
(`sensitivity/test_theta_route_sensitivity_weighted.py`). On DTLZ/MaF,
x=f(θ) directly (continuous); on the IRP, θ→quantities→priorities→decoder
(discrete, chaotic)→routes→objectives — the rotation gate's implicit
assumption ("a small step toward the guide improves the solution a little")
breaks down at the decoder, not in the quantum mechanism itself.

Six independent remedies (A-F), all targeting *how the rotation gate picks
or moves toward its target* (reset frequency/magnitude, dual-attractor,
momentum, chaotic magnitude, and — Remède F — the guide-selection criterion
itself, convergence vs. crowding-distance diversity), were tried and
rejected. Remède F's own empirical validation (5 seeds, real measurement)
confirmed the null result is robust, not a statistical-power artifact. The
project's retained hypothesis: the limiting factor is not *which* point in
θ/objective space becomes the rotation's target, but the decoder's
discontinuity itself, downstream of any θ-space mechanism.

This design targets that root cause directly instead of the rotation
mechanism. A local-repair-after-decoding idea was tried once already, very
early in this project (see `IRP_results_summary.md`'s "Premières tentatives"
table), using 2-opt — abandoned because it optimised the wrong metric (raw
route distance) instead of the project's actual multi-objective cost
function, which includes time-window penalties. This design redoes that
attempt correctly: 2-opt evaluated against f1 (the metric the abandonment
note itself named as the one that should have been used), not distance.

## Goals

- Test whether repairing each individual's decoded route (2-opt, evaluated
  against the real cost function) before scoring it changes QI-NSGA-III's
  result on the real IRP — isolating the decoder's contribution to solution
  quality, independent of the rotation/guide mechanism already exhausted by
  remedies A-F.
- Apply the repair every generation, to both parent and offspring
  populations — this is the only way to test the retained hypothesis
  properly: if the decoder's chaos is corrupting the fitness signal that
  drives guide selection and elitist survival throughout the run (not just
  the final reported front), only an every-generation repair exercises that
  claim. A final-front-only repair would "polish" the report without
  touching the search itself.
- Follow the project's established fast-reject protocol: 3 seeds first,
  shared ideal/nadir with `Solvers/NSGA3`'s cache, Mann-Whitney U, documented
  as "Remède G" in `Solvers/IRP_results_summary.md` regardless of outcome.

## Scope

**New file `Solvers/QINSGA3/repair.py`, plus a new evaluation entry point in
`Solvers/QINSGA3/algorithm.py`.** `Solvers/NSGA3/decoder.py`,
`Solvers/NSGA3/evaluator.py`, and `Solvers/NSGA3/problem.py` are **not
modified** — `IRPProblem._evaluate` (in `problem.py`) is the exact evaluation
path both NSGA3 and QINSGA3 currently share (QINSGA3's `_worker_eval` calls
it directly), so touching it would risk regressing NSGA3's already-validated
baseline for an unproven QINSGA3-only fix — the same reasoning every prior
remedy (A-F) has used to stay QINSGA3-scoped. `decode_chromosome`,
`build_routes`, and `compute_f1`..`compute_f4` are reused unchanged (pure
function calls, not modified) — only their orchestration is new, and only
inside QINSGA3's own module.

## Non-goals

No change to `decoder.py`'s construction heuristic, to NSGA3, to the
rotation gate, θ-encoding/measurement, or `_select_guides`/guide-selection
machinery (remedies A-F's territory, already exhausted). No re-encoding of
the repaired route back into the chromosome (θ/X) — the visit order a 2-opt
repair produces has no defined inverse into priority genes, so this stays a
Baldwinian (evaluation-only) repair: the genotype is unchanged, only the
fitness assigned to it reflects the repaired phenotype. No Or-opt or other
move types in this pass — this design redoes the abandoned 2-opt attempt
correctly, not a broader neighbourhood search; Or-opt is a natural follow-up
if 2-opt alone shows no effect, matching the project's one-variable-at-a-time
convention.

## Design

### `Solvers/QINSGA3/repair.py` (new)

Two functions, splitting "generate a structural candidate" from "decide
whether to keep it" (the latter needs whole-individual context the former
doesn't):

`_two_opt_candidates(path: list) -> Iterator[tuple[int, int, list]]` — pure,
no domain knowledge. For a path (list of node ids, starting/ending at the
depot `O`) with more than 3 nodes, yields `(i, j, candidate_path)` for every
non-adjacent edge pair `(i, j)` with `1 <= i < j <= len(path) - 2` (interior
positions only — the depot at index 0 and the last index are never moved),
`candidate_path` being `path` with the `[i, j]` segment reversed. Never
changes the multiset of visited nodes, only their order.

`_route_traversal_time(path: list, t, k, sets_, params_) -> float` — pure,
reuses the exact arrival-time accumulation formula
`decoder.py::_nearest_neighbour` already computes internally
(`current_time += s.get(current, 0.0) + d[current, next] / speed`, using
`params_["d"]`/`params_["v"]`/`params_["s"]` and truck `k`'s speed), summed
over the whole path from depot to depot. Same formula `build_routes`'s
`tau_return` block already uses per-truck (`decoder.py:200-207`) — factored
out here as a standalone helper since `build_routes` doesn't expose it.

`_repair_route_result(route_result: dict, sets_: dict, params_: dict) -> dict`
— the orchestrator, and the only function called from `algorithm.py`. For
every `routes_data[t][k]["path"]` with more than 3 nodes:

- Computes `tau_return_before`, the period `t`'s current `tau_return` (max
  traversal time over all trucks active that period, from
  `route_result["tau_return"][t]`).
- Runs a first-improvement local search: iterate
  `_two_opt_candidates(path)`; for each candidate, compute its
  `_route_traversal_time`; **reject** if it would push this period's
  `tau_return` (recomputed as the max over all trucks that period, this
  route's candidate time replacing its current one) above
  `tau_return_before` — avoids introducing a new C13 (`tau_return` window)
  violation without needing to know whether the route contains a mandatory
  (`floor > 0`) client, information not preserved in `build_routes`'s return
  value. This is more conservative than the decoder's own mandatory-client
  tau_max exemption, but it guarantees repair never makes a period's
  worst-case travel time worse than what construction already produced — it
  can only hold steady or improve.
- For each candidate that passes the guard, apply it to compute trial
  `x`/`f`/`arrival_times` via `_rebuild_route_arcs` (same suffix-sum/arc-flow
  logic `build_routes` uses at its tail end, `decoder.py:353-368`, factored
  out as a third small local helper), then compute ONLY that route's
  contribution to f1 via `_route_f1_contribution(candidate_path, trial_f,
  trial_arrivals, t, k, params_)` — O(route length), not O(whole network) —
  and compare it against the same route's pre-swap contribution (computed
  the same way). **Accept** the first candidate whose contribution is
  strictly lower than the pre-swap contribution, apply the update via
  `_replace_route_arcs`, and restart the scan on the modified path. This is
  provably exact: every other term in f1 (holding cost, every other route's
  transport cost and time-window penalty) is unchanged by a single-route
  swap and cancels exactly in the delta — proven by
  `test_route_f1_contribution_delta_matches_compute_f1_delta` in
  `Solvers/QINSGA3/test_repair.py`, which checks this against the real
  `compute_f1` on a route_result with a second, untouched route present (not
  just a single-route fixture, which would trivially agree). An earlier
  version of this design called `compute_f1` on a full scratch
  `route_result` per candidate — correct but O(whole network) — and was
  replaced after being measured at ~43x slower than baseline
  (`sensitivity/route_repair_timing_check_iter5.txt`).
- Stop when no improving-and-guard-passing candidate exists for the current
  path, or after `_MAX_REPAIR_ITER` iterations (internal constant, not
  exposed — see New Parameters), then move to the next route.
- `depot_stock` and `actual_qty` are untouched — order-independent
  (aggregate per-period totals, unaffected by within-period visit order) —
  and capacity, C14 cumulative-demand, and C6 stock-ceiling constraints are
  structurally unaffected by any 2-opt swap (same visited clients, same
  quantities, only order changes), so none of those need re-checking.

### `Solvers/QINSGA3/algorithm.py` (modified)

New function `_evaluate_with_repair(x, sets_, params_) -> (F, G)`, called
from a new worker-eval path parallel to the existing `_worker_eval`:

```
quantities, priorities = decode_chromosome(x, sets_)        # unchanged, reused
route_result            = build_routes(quantities, sets_, params_, priorities)  # unchanged, reused
route_result             = _repair_route_result(route_result, sets_, params_)   # new
F = [compute_f1(route_result, ...), compute_f2(...), compute_f3(...), compute_f4(...)]  # unchanged, reused
G = <same G-list construction as IRPProblem._evaluate, duplicated here>          # see below
```

The G-list construction (~10 lines: C13 `tau_return` window, C14 cumulative
delivery deadlines, C6 stock-ceiling safety net — `problem.py:67-85`) is
duplicated into this new function rather than calling `IRPProblem._evaluate`
itself, since that method doesn't expose a repair hook and modifying it
would reintroduce the shared-file risk this design avoids. This is the one
deliberate duplication in this design — kept small (~10 lines, no branching
logic of its own) and commented with a pointer to `problem.py:67-85` as the
source of truth it mirrors, so the two stay in sync if `problem.py`'s
G-list ever changes.

New parameter `use_route_repair: bool = False` on `run_qinsga3` (disabled by
default, same convention as every prior remedy). When `True`, `_eval_batch`'s
worker function is `_evaluate_with_repair` instead of `_worker_eval` — since
both parent and offspring evaluation already go through the same
`_eval_batch` helper each generation, this one flag applies the repair to
both without any other loop change.

## New parameters

`use_route_repair` (bool, default `False`) — the only new `run_qinsga3`
parameter. No new tunables beyond it in this pass (the 2-opt iteration cap
is an internal constant in `repair.py`, not exposed, to keep the ablation
surface minimal — matching how e.g. `_crowding_distance`'s internals aren't
parameterised either).

## Testing

`Solvers/QINSGA3/test_repair.py` (new file):

- A hand-constructed 3+ stop route where a known 2-opt swap strictly
  improves f1 (e.g. two stops visited in a distance-inefficient order with
  no time-window pressure) — assert the repaired path differs and its f1 is
  lower than the original.
- A route where the only available 2-opt swap would increase the period's
  `tau_return` beyond its pre-repair value — assert the swap is rejected and
  the path is unchanged (safety guard fires).
- An already 2-opt-optimal route (no improving swap exists) — assert the
  repair leaves it byte-identical (no spurious "improvement", no infinite
  loop; also bounds the iteration cap's correctness).
- `_evaluate_with_repair`'s G-list output matches `IRPProblem._evaluate`'s G
  formula exactly when repair is a no-op (i.e., on a chromosome whose
  decoded route has no improving 2-opt swap) — guards against the
  duplicated G-list construction silently drifting from `problem.py`'s.

## Empirical validation

`sensitivity/compare_route_repair.py` (new file), same structure as
`sensitivity/compare_crowding_guides.py`: instance 100 clients, 3 seeds
first (`[42, 137, 271]`), 300 generations, shared ideal/nadir with
`Solvers/NSGA3`'s cache, Mann-Whitney U on HV/GD/IGD/Spacing, chromosome
diversity reported. If a positive signal appears, scale to 5 seeds minimum
(matching Remède F's precedent that 3 seeds alone can't reach p<0.05 on a
3-vs-3 comparison — exact two-sided Mann-Whitney floor is p=0.10 at n=3,
~0.008 at n=5) before treating any significance claim as final. Documented
as "Remède G" in `Solvers/IRP_results_summary.md` regardless of outcome,
same format as A-F.

## Risk

- **Runtime cost**: this materialized. The whole-network approach (recomputing
  whole-individual f1 per candidate 2-opt swap, not a route-local delta) was
  tried first and measured at ~43x slower than baseline
  (`sensitivity/route_repair_timing_check_iter5.txt`) — well past the 10x
  threshold this bullet anticipated — and was replaced by the route-local
  delta (`_route_f1_contribution`), which brought a small-scale timing check
  down to ~5.7x (`sensitivity/route_repair_timing_check_delta.txt`). Even
  after that fix, the real full-scale campaign (3 seeds, 300 generations)
  still measured ~15.2x (`sensitivity/route_repair_campaign_log.txt`), since
  the ratio grows with generation count rather than staying constant. This
  is exactly the practical disqualification the project's final
  documentation records for this remedy.
- **Baldwinian repair may have limited effect on the search itself**: since
  the repaired route never feeds back into the chromosome, a repair that
  reliably improves fitness might still not change *which* chromosomes get
  selected as guides or survive, if the ranking among individuals is largely
  preserved by repair (i.e., if repair improves all individuals roughly
  proportionally, relative order — and thus every downstream selection
  decision — may not shift much). This is a real possibility worth watching
  for in the 3-seed result, not just whether HV improves.
- **f1-only acceptance criterion**: a swap that improves f1 is not
  guaranteed to improve f2 (CO2) or f3 (travel time), which also depend on
  route order — the design doc's own reasoning is that these are usually
  correlated with f1 (distance-driven), but this is an assumption, not a
  guarantee. Worth checking f2/f3 alongside HV/GD/IGD/Spacing in the
  validation output, even though they aren't the repair's optimisation
  target.
