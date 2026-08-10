# QINSGA3 vs NSGA3 — 2-opt fairness campaign — design

## Context

`Solvers/QINSGA3/repair.py` (Remède G, see
`docs/superpowers/specs/2026-08-04-qinsga3-route-repair-design.md`) adds a
post-decode 2-opt local search to QI-NSGA-III's returned Pareto front,
enabled by default (`repair_final_front=True` on `run_qinsga3`). `Solvers/
NSGA3/` — the classic NSGA-III baseline QI-NSGA-III is benchmarked against —
has no equivalent post-processing.

A reviewer flagged the resulting equity problem directly: any comparison
that reports "QI-NSGA-III (with 2-opt) vs NSGA-III (without 2-opt)"
conflates two different effects — the evolutionary engine itself (quantum
rotation gate vs. plain real-coded NSGA-III) and the 2-opt post-processing
applied to only one side. This is exactly what
`sensitivity/compare_route_repair_final.py` currently does at its "test_lbl
vs NSGA-III (l'objectif final)" comparison (lines 213-223): it decodes
NSGA-III's cached chromosomes with the plain, unmodified decoder
(`_load_nsga3_reference` → `_decode_to_F`, no repair step) and compares that
against QI-NSGA-III's *repaired* front.

Investigating the repair's wiring also surfaced a second, independent
finding: `Solvers/QINSGA3/main.py::run_qinsga3_solver` calls `run_qinsga3`
(which computes a correctly-repaired `pareto_F` when `repair_final_front=
True`) but then discards that `pareto_F` and rebuilds the reported/cached
objectives via `_evaluate_pareto(raw["pareto_X"], ...)`
(`Solvers/NSGA3/report_builder.py`), which re-decodes from the chromosome
array with the plain decoder and never applies the repair. So the
production report (`qinsga3_report.html`) and cache
(`qinsga3_chromosomes.json`) never reflect the 2-opt repair at all, despite
`repair_final_front=True` being the default. The dedicated ablation scripts
(`compare_route_repair_final.py`, etc.) sidestep this bug because they read
`pareto_F` straight from `run_qinsga3`'s return value, not from the
cache/report path — which is why their validated numbers (HV +29.2% etc.,
see `Solvers/IRP_results_summary.md`) are correct despite the bug.

## Goals

- Make 2-opt post-processing available to NSGA-III on the same terms as
  QI-NSGA-III, so a fair 4-way comparison is possible.
- Build a campaign script that isolates the two effects a naive 2-config
  comparison conflates: the evolutionary engine (quantum vs. plain) and the
  2-opt post-processing, using the same initial fronts, seeds, and budgets
  across all four configurations:
  1. NSGA-III without 2-opt
  2. NSGA-III with 2-opt
  3. QI-NSGA-III without 2-opt
  4. QI-NSGA-III with 2-opt
- Fix the latent bug so `repair_final_front`'s effect is correctly reflected
  in both algorithms' reports/caches when enabled, instead of silently
  fixing it for QINSGA3 alone while leaving NSGA3 to replicate the same
  discard-on-report bug.

## Scope

- `Solvers/NSGA3/report_builder.py::_evaluate_pareto` — new `repair: bool =
  False` parameter.
- `Solvers/NSGA3/main.py::run_nsga3` — new `repair_final_front: bool =
  False` parameter (default unchanged: the live app and its report keep
  their current behaviour unless a caller explicitly opts in).
- `Solvers/QINSGA3/main.py::run_qinsga3_solver` — wire its existing
  `repair_final_front` parameter through to `_evaluate_pareto`'s new
  `repair` flag (bug fix; the parameter already exists and already defaults
  to `True`, so this changes what the production report/cache show under
  the existing default — see Risk).
- New file `sensitivity/compare_2opt_fairness.py` — the 4-config campaign.
- One-line caveat added to `sensitivity/compare_route_repair_final.py`'s
  NSGA-III comparison section, pointing at the new script.

## Non-goals

- No change to the 2-opt engine itself (`_repair_route_result` and its
  helpers in `Solvers/QINSGA3/repair.py` stay exactly as they are — they are
  already algorithm-agnostic, operating only on a decoded `route_result`
  dict).
- No change to `run_nsga3`'s or `run_qinsga3`'s default behaviour as seen by
  the app (`repair_final_front` defaults to `False` for NSGA-III, `True`
  for QI-NSGA-III — both unchanged from today).
- No retroactive rewrite of `Solvers/QINSGA3/README.md` or
  `Solvers/IRP_results_summary.md`'s existing claims in this pass — those
  get updated once the campaign has produced real 4-config numbers, as a
  follow-up.
- No CLI flag for `repair_final_front` on either solver's `__main__` block —
  QINSGA3 doesn't expose one today either; stays a function parameter,
  consistent with the existing convention.

## Design

### `Solvers/NSGA3/report_builder.py::_evaluate_pareto` (modified)

```python
def _evaluate_pareto(pareto_X, sets_, params_, meta_base, repair: bool = False):
    ...
    for i, chromosome in enumerate(pareto_X):
        quantities, priorities = decode_chromosome(chromosome, sets_)
        route_result = build_routes(quantities, sets_, params_, priorities)
        if repair:
            from Solvers.QINSGA3.repair import _repair_route_result
            route_result = _repair_route_result(route_result, sets_, params_)
        f1 = compute_f1(route_result, sets_, params_)
        ...
```

The import stays function-local (lazy), matching the existing convention
`Solvers/QINSGA3/algorithm.py::_evaluate_with_repair` already uses for the
same reason: `Solvers/QINSGA3/__init__.py` imports `.main`, which imports
`Solvers.NSGA3.report_builder` — a module-level import of
`Solvers.QINSGA3.repair` from inside `report_builder.py` would run during
that same import chain. `Solvers/QINSGA3/repair.py` has no imports of its
own (verified), so this does not actually deadlock either way, but a
function-local import avoids relying on that fact and keeps the dependency
direction explicit: `Solvers/NSGA3` (the shared base) reaches into
`Solvers/QINSGA3` (the derived package) only inside the one function that
needs it, only when the caller opts in.

Everything downstream of `route_result` in `_evaluate_pareto` (routes
report, deliveries, depot_stock) already reads from `route_result`, so it
automatically reflects the repaired routes too — no separate plumbing
needed for the fuller report, not just the four objectives.

### `Solvers/NSGA3/main.py::run_nsga3` (modified)

New parameter `repair_final_front: bool = False`, added to `base_meta`
(so `render_from_instance` can read it back from the cache — see below) and
passed as `repair=repair_final_front` to every `_evaluate_pareto(...)` call
in `run_nsga3`. `render_from_instance` passes
`repair=run_cache["meta_base"].get("repair_final_front", False)` per run,
so refreshing a report from a stale cache re-applies the same repair
decision the original run used, rather than silently defaulting to one
behaviour regardless of how the cache was produced.

### `Solvers/QINSGA3/main.py::run_qinsga3_solver` (modified)

Both `_evaluate_pareto(raw["pareto_X"], sets_, params_, per_run_meta)`
call sites (the post-run build and `render_from_instance`) gain
`repair=repair_final_front` / `repair=run_cache["meta_base"].get(
"repair_final_front", False)`, mirroring NSGA3's change exactly. This is
the bug fix: `repair_final_front`'s existing default (`True`) now actually
reaches the report/cache, instead of being computed by `run_qinsga3` and
then discarded.

`run_qinsga3`'s own `_repair_pareto_front` post-processing (algorithm.py:
1471-1472) and its returned `pareto_F` are unaffected by this change — they
are simply no longer the only path that sees the repair's effect.

### `sensitivity/compare_2opt_fairness.py` (new)

Same shape as `sensitivity/compare_zone_locked.py` /
`compare_damped_priority.py`: `argparse` for `--instance` (default `100`),
`--seeds` (default `[42, 137, 271]`), `--gen` (default `300`), `--pop`
(default `200`).

Per seed:

1. Run NSGA-III's search once — the exact `IRPProblem` + `NSGA3` + SBX/PM +
   `minimize()` setup `Solvers/NSGA3/main.py::run_nsga3` uses, replicated
   locally in the script (not calling `run_nsga3` itself, since that writes
   a shared cache file and returns a report dict, not raw arrays) — to get
   `pareto_X_nsga3`.
2. Run QI-NSGA-III's search once via `run_qinsga3(sets_=sets_, params_=
   params_, ref_dirs=ref_dirs, pop_size=effective_pop, max_gen=max_gen,
   seed=seed, repair_final_front=False)` (the low-level function, already
   returning raw arrays — same call `compare_route_repair_final.py` uses)
   to get `pareto_X_qinsga3`.
3. Decode each chromosome array **twice**, using `_evaluate_pareto`-shaped
   logic (repair is Baldwinian — the chromosome array is untouched by
   `repair_final_front`, so the same `pareto_X` can be decoded with and
   without repair to get two configurations from one search run):
   - `pareto_X_nsga3` → Config 1 (`repair=False`), Config 2 (`repair=True`)
   - `pareto_X_qinsga3` → Config 3 (`repair=False`), Config 4 (`repair=True`)

   This guarantees identical initial fronts between the "without" and
   "with" configuration of each algorithm — only the post-processing
   differs, exactly matching "mêmes fronts initiaux, seeds et budgets".

4. `compute_pareto_metrics` (HV/GD/IGD/Spacing) for all four configurations
   using one global ideal/nadir computed across all 4 configs × all seeds
   combined (the project's established convention, e.g.
   `compare_qinsga3_vs_nsga3.py`).
5. Statistics, explicitly separating the two effects:
   - **2-opt effect** (paired Wilcoxon signed-rank — same front, same
     seed, only the post-processing differs): Config 1 vs Config 2
     (NSGA-III), Config 3 vs Config 4 (QI-NSGA-III). Matches the paired
     design already used in `Solvers/QINSGA3/repair.py`'s own docstring
     precedent (best-improvement vs first-improvement validation).
   - **Engine effect** (Mann-Whitney U — independent fronts, matches every
     other cross-algorithm comparison in this project): Config 1 vs
     Config 3 ("raw" gap, no post-processing on either side — reproduces
     what `compare_qinsga3_vs_nsga3.py` already measures) and Config 2 vs
     Config 4 ("fair" gap — both sides get the same 2-opt polish; the
     number that actually answers "is the quantum mechanism itself
     better", isolated from the repair).
6. Printed summary table (mean/std per config per indicator) plus both
   statistical sections, plus per-config elapsed time (2-opt adds decode
   cost only to the final front, not the search loop, so timing should be
   near-identical within an algorithm's two configs — worth confirming,
   not assuming).

Fast smoke-test invocation: `--gen 5 --seeds 42`, to validate the pipeline
end-to-end before a full run, matching every prior campaign script's usage
docstring convention.

## New parameters

- `Solvers/NSGA3/report_builder.py::_evaluate_pareto(..., repair: bool =
  False)`.
- `Solvers/NSGA3/main.py::run_nsga3(..., repair_final_front: bool =
  False)`.

No new tunables inside the 2-opt engine itself — it is reused exactly as
tuned for QI-NSGA-III (`_TWO_OPT_WINDOW=8`, `_MAX_REPAIR_ITER=20`,
best-improvement acceptance), since this design's question is about parity
of *whether* 2-opt runs, not re-tuning it per algorithm.

## Testing

- `Solvers/NSGA3/test_*` (extend existing report_builder/main test coverage
  if present, else a small new test): `_evaluate_pareto(..., repair=True)`
  on a hand-built chromosome whose decoded route has a known improving
  2-opt swap — assert the returned `f1` is lower than `repair=False`'s, and
  that `repair=False`'s output is byte-identical to today's (no behaviour
  change for the existing default).
- `run_nsga3(..., repair_final_front=True)` end-to-end on a small instance
  (3 or 5 clients) — assert it runs without error and the cached
  `meta_base` records `repair_final_front: True`.
- `run_qinsga3_solver`'s report now reflecting the repair: a regression
  check that a small-instance run with `repair_final_front=True` produces a
  report whose `f1` values are `<=` the `repair_final_front=False` run's
  (repair only ever improves or holds f1 steady, never worsens it — the
  same guarantee `repair.py`'s own guard already proves).

## Empirical validation

`python -m sensitivity.compare_2opt_fairness --gen 5 --seeds 42` (smoke),
then `--seeds 42 137 271 --gen 300` (full 3-seed pass, instance 100) as the
first real campaign run. Results get folded into
`Solvers/IRP_results_summary.md` as a new dated entry once available — not
fabricated ahead of time in this design doc.

## Risk

- **Changes what today's default QI-NSGA-III report/cache show.**
  `repair_final_front=True` is already the production default; this design
  makes it actually take effect for the first time. Anyone who has been
  reading `qinsga3_report.html`'s numbers as "the current production
  behaviour" will see them shift (in the improving direction, per Remède
  G's own validated numbers) after this change ships — this is a
  correctness fix, not a regression, but worth calling out explicitly since
  it changes previously-observed output for an unchanged default flag.
- **Local replication of NSGA-III's pymoo setup in the campaign script**
  (rather than calling `run_nsga3` and reading its cache back) means two
  copies of "how NSGA-III is configured" exist if `Solvers/NSGA3/main.py`'s
  hyperparameters ever change. Matches the precedent
  `compare_route_repair_final.py` already set for QI-NSGA-III's low-level
  `run_qinsga3` call, and is necessary here since `run_nsga3` has no
  low-level equivalent returning raw arrays without writing the shared
  cache file — accepted as consistent with existing project convention
  rather than introduced fresh by this design.
- **3 seeds is underpowered for the paired Wilcoxon and Mann-Whitney tests**
  (matches every other first-pass campaign in this project — e.g. exact
  two-sided Mann-Whitney floor is p=0.10 at n=3). Treated as a fast-reject
  first pass; a positive or ambiguous signal should scale to more seeds
  before any significance claim is treated as final, same convention as
  every prior remedy.
