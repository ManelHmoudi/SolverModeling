# QINSGA3 elitist per-niche guide — design

## Context

The QINSGA3 benchmark validation (see
`docs/superpowers/specs/2026-07-23-qinsga3-benchmark-validation-design.md`)
compared QINSGA3 against classic NSGA-III (Cui et al. 2025 protocol) on
DTLZ1-7 and MaF1-7, M=3/M=4, 30 runs each. After aligning crossover/mutation
parameters and re-tuning migration, QINSGA3 matches or beats NSGA-III on most
problems, but stays clearly worse — with very high Best/Worst spread — on
DTLZ1, DTLZ3, MaF3, MaF4 (e.g. DTLZ3 M3: mean IGD 8.12 vs NSGA-III's 7.85,
worst-case 24.5 vs best-case 0.56 across seeds).

Root cause identified by reading `QINSGA3/algorithm.py` and
`QINSGA3/chromosome.py`: `_select_guides` recomputes, every generation, the
best member of the *current* Pareto front per reference-direction niche, with
no memory. If a mutation ("strong" reset, applied with probability
`p_mut_strong=0.15` per gene) destroys the best individual of a niche, next
generation's guide for that niche can regress relative to what was already
found — there is no elitism protecting a niche's best-known solution, unlike
classic NSGA-III's environmental selection (which never lets a discovered
good solution disappear from the population) or the classic QEA elitism
principle (Han & Kim 2002; formalised as "attractor replaced only if better"
in Zhang, *Quantum-inspired evolutionary algorithms: a survey and empirical
study*, Journal of Heuristics 17(3), 2011). This is a plausible explanation
for the high seed-to-seed variance on DTLZ1/DTLZ3 (multimodal problems where
losing a good niche representative is costly) — see the results tables in
`validation/dtlz/results/qinsga3/DTLZ_results_summary_qinsga3.md`.

QINSGA3 already maintains an external archive (`arch_X`/`arch_F`/`arch_theta`
in both `QINSGA3/algorithm.py::run_qinsga3` and
`validation/algorithms/qinsga3/core.py::run_qinsga3_generic`) that is
elitist by construction: `_archive_update` only keeps non-dominated feasible
solutions and removes archive members later dominated by a new candidate — it
never regresses. Today this archive is only consulted for niches with **no**
representative in the current Pareto front (`_supplement_from_archive`);
niches that *are* covered by the current front never check whether the
archive holds a better representative than the current front's pick.

## Goals

- Make guide selection elitist per niche: for every reference-direction
  niche (covered or not by the current Pareto front), compare the current
  front's best representative against the archive's best representative, and
  guide the niche's population members toward whichever is closer
  (perpendicular distance to the reference ray) — reusing the existing
  archive as the elitist store, no new persistent state.
- Fix a normalization inconsistency this surfaces: today `F_norm`
  (population) and `arch_F_norm` (archive) are each normalised with their
  *own* ideal/nadir, so their perpendicular distances aren't on the same
  scale. The comparison this design introduces requires both to be measured
  in the same normalised frame.
- Apply the fix in the shared `QINSGA3/algorithm.py` (used by both the real
  IRP production path and the DTLZ/MaF benchmark harness) — confirmed with
  the user: this is intended to benefit both paths, not just the benchmark.
- Add unit test coverage for the changed guide-selection logic —
  `QINSGA3/algorithm.py` currently has zero unit tests.
- Validate with the same small-ablation-then-full-campaign methodology
  already used for the crossover/migration/noise tuning: 5 runs on DTLZ1 +
  DTLZ3 first, then (if neutral-or-better) a fresh 30-run campaign across
  DTLZ+MaF, M3+M4.

## Non-goals

- No change to the rotation gate, crossover, or mutation operators
  themselves.
- No change to `MIGRATION_PERIOD`/`N_MIGRATE`/`NOISE_SCALE` or any other
  already-tuned parameter — this is an independent mechanism change.
- No attempt to fix DTLZ7/MaF7 (disjoint-front regression) — that is a
  separately documented, unrelated limitation.
- No multi-guide-per-niche mechanism (the PSO-inspired "multiple guides"
  idea discussed earlier) — out of scope for this pass; single elitist guide
  per niche only.
- No full recalibration of the real IRP's own tuned defaults
  (`QINSGA3/main.py`) — only a smoke test that production still runs
  end-to-end without errors after the shared-code change.

## Design

### Guide selection mechanism

Merge today's two-step process (`_select_guides` then, conditionally,
`_supplement_from_archive`) into one elitist selection:

For each niche present in the population's `assoc` (not just those in
`pareto_assoc` today):

1. If the niche has a representative in the current Pareto front, compute
   its perpendicular distance to the reference ray (as today).
2. If the archive (once it has ≥4 entries, same threshold as today) has a
   representative in the same niche, compute its perpendicular distance too,
   **in the same normalised frame as step 1**.
3. Guide the niche = whichever of the two has the smaller distance. If only
   one side has a representative, use it. If neither does, fall back to the
   existing global-fallback rule (closest-to-origin Pareto member).

This is a direct generalisation of what `_supplement_from_archive` already
does for empty niches — now applied uniformly, so a niche's guide can never
regress below the best solution ever found for it, exactly mirroring the
elitist archive's own invariant.

### Normalisation fix

`_select_guides`/the new merged function will normalise the archive's F
using the **population's** ideal point and nadir (the same ones used to
produce `F_norm`), instead of recomputing ideal/nadir from the archive alone.
Concretely: `_normalise_F` gains an optional way to reuse externally-supplied
ideal/denom (e.g. an internal helper split out of `_normalise_F` that takes
`ideal`/`denom` explicitly, with `_normalise_F(F)` becoming a thin wrapper
that computes them from `F` itself as it does today). All existing call sites
(`F_norm = _normalise_F(F_pen)`, the standalone archive normalisation used by
`_migrate`) keep working unchanged; only the new merged guide-selection code
uses the explicit-ideal/denom path to project the archive into the
population's frame.

### API changes

- `_select_guides(assoc, pareto_idx, F_norm, ref_dirs, qpop_theta, arch_theta=None, arch_F=None)`
  — two new optional parameters, default `None` (preserves today's
  front-only behaviour when the archive isn't available yet, i.e.
  `len(arch_X) < 4`, same guard as today).
- `_supplement_from_archive` is deleted — its logic is absorbed into the
  merged `_select_guides`.
- Both call sites (`QINSGA3/algorithm.py::run_qinsga3`,
  `validation/algorithms/qinsga3/core.py::run_qinsga3_generic`) are updated
  to a single guide-selection call instead of the current two-step
  sequence. `_migrate`'s own archive normalisation is untouched (separate
  concern, not part of this design).

### Testing

New `QINSGA3/test_algorithm.py` (first unit tests for this module), covering
the merged guide-selection function with small synthetic arrays:

- Archive holds a strictly better (smaller perpendicular distance)
  representative than the current front for an occupied niche → guide comes
  from the archive.
- Archive representative is worse than the front's → guide stays on the
  front's pick (no regression from today's behaviour).
- Niche uncovered by the front, with/without an archive representative →
  same fallback behaviour as today's `_supplement_from_archive`.
- Archive below the 4-entry threshold → guide selection ignores the archive
  entirely, identical output to the current `_select_guides` alone.

### Empirical validation

1. 5-run ablation on DTLZ1 and DTLZ3 (highest variance / most multimodal),
   same seeds subset and same protocol as the earlier migration/noise
   ablations documented in `DTLZ_results_summary_qinsga3.md`, compared
   against the current adopted config (crossover aligned, noise=0,
   migration 5/20).
2. If mean IGD and Best/Worst spread are neutral-or-better on both problems
   with no regression elsewhere in the 5-run sample (spot-check DTLZ2, DTLZ7
   since guide selection touches every niche including disjoint-front
   niches) → run the full 30-run × (DTLZ+MaF) × (M3+M4) campaign and update
   both `*_results_summary_qinsga3.md` reports the same way the earlier
   tuning passes did.
3. Smoke-test the real IRP path (`QINSGA3/main.py`) end-to-end after the
   shared-code change — confirms no crash/regression in the production
   caller, not a full recalibration.

## Migration steps

1. Split an ideal/denom-explicit helper out of `_normalise_F` in
   `QINSGA3/algorithm.py`; keep `_normalise_F(F)`'s existing signature and
   behaviour for all current callers.
2. Extend `_select_guides` with the optional `arch_theta`/`arch_F`
   parameters and the merged front-vs-archive comparison; delete
   `_supplement_from_archive`.
3. Update `QINSGA3/algorithm.py::run_qinsga3`'s generational loop to the
   single guide-selection call.
4. Update `validation/algorithms/qinsga3/core.py::run_qinsga3_generic`'s
   loop the same way.
5. Add `QINSGA3/test_algorithm.py` with the four cases above; run it.
6. Run the existing `validation/algorithms/qinsga3/test_runner.py` (still
   must pass unchanged — same public shape) plus a `--runs 2` smoke run on
   one suite.
7. Run the 5-run DTLZ1/DTLZ3 ablation, compare against current numbers.
8. If adopted: full 30-run campaign, update both `*_qinsga3.md` reports with
   a new "guide élitiste" column, same style as the existing 4-step
   evolution table.
9. Smoke-test `QINSGA3/main.py` on the real IRP.

## Open risk / thing to watch

Because the archive is capped at 500 entries and trimmed by crowding
distance (`_MAX_ARCHIVE`, `_crowding_trim`), on high-dimensional problems
(M=4) a niche's best-ever solution could in principle be evicted from the
archive by the crowding trim before the population's own front loses it —
in that case the elitism guarantee is only as strong as the archive's
retention, not absolute. Not expected to matter in practice (archive holds
non-dominated solutions across all niches, crowding trim spreads evictions
roughly evenly), but worth checking if the M=4 ablation results look
inconsistent with M=3.
