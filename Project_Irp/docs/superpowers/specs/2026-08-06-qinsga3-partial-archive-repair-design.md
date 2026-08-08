# QINSGA3 partial in-search archive repair — screening design

## Context

Nine independent remedies (A-I, see `Solvers/IRP_results_summary.md`) all
targeted chromosome or route DIVERSITY, and all failed to close the gap with
NSGA-III on the real IRP -- confirming the project's own diagnostic
conclusion: the limiting factor is not a diversity deficit but the fragility
of the greedy, irrevocable nearest-neighbour decoder (small θ change →
disproportionately different route). The only remedy that produced a real,
eventually significant gain was Remedy G (2-opt post-decode route repair,
`Solvers/QINSGA3/repair.py`) -- but repairing every individual every
generation (`use_route_repair`) costs ~15x baseline runtime, disqualifying
it in practice. Its adopted practical variant (`repair_final_front`,
production default) repairs only the returned Pareto front, ONCE, after the
search loop ends -- cheap (near-zero added cost, 211.4s vs 219.8s at 7
seeds) but the repaired signal never reaches guide selection or the archive
DURING the search.

This design tests the untested middle ground between those two: repairing
only the individuals that actually enter the archive each generation (the
current generation's own rank-0 Pareto front, not the whole population).
Measured directly on a real 100-generation run (`run_qinsga3`'s own
`off_pareto_idx`, instrumented via the existing `callback` hook): this front
is typically 10-20 individuals out of a population of 200 (5-10%), far
smaller than `use_route_repair`'s full-population target.

## Goal

Check whether letting the archive (and, via `_supplement_from_archive`,
niche-guide selection) see repair-corrected fitness THROUGHOUT the search --
not just once at the end -- produces a quality gain beyond what
`repair_final_front` alone already gives, at an affordable runtime cost.

## Design

### Mechanism

Wherever the production loop (`run_qinsga3`) calls `_archive_update` with
raw `(X, F, G)` for the current generation's Pareto-front candidates (parent
`pareto_idx` and offspring `off_pareto_idx`, two call sites per generation,
plus the one-off final-generation call), this variant first repairs those
candidates' `F`/`G` via `_evaluate_with_repair` (the exact same Baldwinian
2-opt mechanism Remedy G already uses) before they enter the archive:

```python
pareto_idx = sorter.do(F_pen_parent)[0]
F_arch, G_arch = _repair_candidates(X_parent[pareto_idx], sets_, params_)
_archive_update(
    X_parent[pareto_idx], F_arch, G_arch,
    theta_parent[pareto_idx], arch_X, arch_F, arch_theta, _MAX_ARCHIVE,
)
```

where `_repair_candidates` loops `_evaluate_with_repair` (decode → 2-opt
repair → evaluate) over the small candidate subset.

### What does NOT change

`pareto_idx`/`off_pareto_idx` selection, `F_norm`/`assoc`, `_select_guides`,
survival, SBX/PM, and rotation all stay driven by the RAW (unrepaired)
fitness -- identical to production. The chromosomes themselves are never
touched (Baldwinian, same convention as Remedy G). The only place this can
change downstream behaviour is `_supplement_from_archive` (already in
production, unmodified) reading a now-more-accurate `arch_F` when filling in
niche guides that lack a population-native champion -- so a repaired archive
can only make supplemented guides MORE reliable, it cannot destabilise the
rest of the search.

Both baseline and test keep `repair_final_front=True` (production default)
at the end, so the final front is repaired identically in both arms -- the
only isolated variable is whether archive-feeding candidates were ALSO
repaired throughout the search, not just at the very end.

### Cost

Smoke-tested (1 seed, 8 generations): ~1.7x baseline runtime (21.6s →
36.3s) -- consistent with the ~10-20/200 (5-10%) candidate fraction being
repaired each generation, versus `use_route_repair`'s ~15x for repairing
100%. Well within a usable range, unlike the disqualified full variant.

### Scope

Standalone `sensitivity/compare_archive_repair.py`, mirroring the
established pattern of every other remedy script in this project (private
copy of the generation loop, no changes to `Solvers/QINSGA3/algorithm.py`
until/unless validated). Baseline = production `run_qinsga3()` unmodified.
Test = `run_qinsga3_archive_repair()`, identical except for the archive-
candidate repair calls described above.

## Testing / validation

3 seeds first (`[42, 137, 271]`), full 300 generations, instance 100 clients
-- the project's own fast-reject protocol. Report HV/GD/IGD/Spacing vs.
baseline and vs. the NSGA-III cache, plus per-run elapsed time (to confirm
the cost estimate holds at full scale). If there's a positive signal, scale
to 20 seeds, matching the validated-result table. If null, document
alongside the nine already-rejected remedies in
`Solvers/IRP_results_summary.md`.

## Non-goals

No change to production code. No change to `pareto_idx`/`off_pareto_idx`
selection, guide selection, survival, or rotation. No repair-frequency
ablation (every-K-generations) in this first pass -- every generation is the
starting point, given the measured cost is already affordable; a frequency
knob is a natural follow-up only if this first screening shows a positive
signal worth tuning further.
