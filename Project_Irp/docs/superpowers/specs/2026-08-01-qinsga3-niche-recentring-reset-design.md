# QINSGA3 niche-recentring reset operator — design

## Context

`Solvers/IRP_results_summary.md`'s diagnostic chapter found that QI-NSGA-III's
chromosome diversity on the real IRP is ~13x lower than NSGA-III's *before any
decoding happens* — the guided rotation gate pulls the whole population toward
a single guide per niche every generation, with no counterforce strong enough
to stop it over 300 generations. Four remedies targeting the rotation's
frequency/target/magnitude, plus the decoder, were tried and rejected (see
that file's "Quatre remedes testes et rejetes" table).

A fifth candidate existed in git history: the DPQiEA diversity-preserving
operator [Tayarani-N & Akbarzadeh-T 2014, *Evolutionary Intelligence*
7:219-239, §5], ported for QINSGA3 in commits `d70b396`..`1b7e369`, tested on
DTLZ1/DTLZ3, and reverted (not adopted — its trigger almost never fired).
`sensitivity/diagnose_diversity_preserving_trigger.py` (this session) checked
whether restoring it unmodified would fire on the real IRP, using the same
pooled Pareto-front chromosomes as the diversity measurement above (20 seeds,
1016 chromosomes, 43 occupied niches): **0% trigger rate at every γ from 0.99
down to 0.50.** Root cause, confirmed by a direct θ measurement: the paper's
convergence formula (`mean|cos(2θ)|`, eq. 11) detects collapse *toward a
classical bit value* (θ→0 or π/2) — but QINSGA3 initialises at θ=π/4±0.05
("maximum superposition") and its rotation gate pulls individuals *toward each
other*, not toward the boundaries. Measured: mean θ per individual =
0.7786 rad vs π/4=0.7854 rad, std across individuals = 0.0034; 78% of all gene
values sit within 0.1 rad of π/4. The population collapses at exactly the
point eq. 11 calls "least converged." The formula is structurally blind to
this algorithm's actual failure mode — restoring it unmodified would very
likely be a 5th rejection, for a new reason.

The underlying idea (detect a converged cluster in a stagnant niche, keep the
best, reset the rest to reintroduce exploration) is still sound. This design
keeps that structure and corrects the two parts eq. 11 got wrong for
QINSGA3's real-valued, guide-clustering collapse mode.

## What changes vs. the original (reverted) DPQiEA port

| Original DPQiEA port (git history) | This design | Why |
|---|---|---|
| Convergence gate: eq. 11, `mean\|cos(2θ_ik)\| > γ` (detects θ near 0/π/2) | **Dropped.** | Measured 0% trigger on the IRP at every γ from 0.99 to 0.50 (max observed score 0.155) — the population never approaches the boundaries this formula checks for, regardless of how permissive γ is. |
| Similarity clustering: eq. 12, `mean\|θ_ik−θ_jk\|/(π/2) < δ`, applied only to individuals that passed eq. 11 | **Kept, but now the sole convergence signal** — a mutually-close pair/cluster in a stagnant niche *is* the trigger, independent of where in [0,π/2] it sits. | This formula was never the broken part; it correctly measures "these individuals are near-identical," which is exactly QINSGA3's actual collapse symptom. |
| Reset value: `θ ← π/4` (eq. 14, "return to maximum superposition") | **`θ ← U(0, π/2)` per gene, freshly drawn** | π/4 is where QINSGA3 initialises *and* where the population is already measured to sit (mean 0.7786 rad, 78% of genes within 0.1 rad of it). Resetting there would reintroduce ~0 diversity — an empirically-grounded correction, not a stylistic one. |
| "Best observed" `b_i`: the niche's elitist guide (from the separate, also-unadopted elitist-guide change) | **The niche's regular guide** (`_select_guides`, already in production) | Isolates this operator's own effect, same reasoning the original design doc already used for its own DTLZ ablation ("validated on top of the currently adopted config, not on top of the un-adopted elitist guide"). |
| Stagnation gate: eq. 13, guide unchanged for `T_STAGNATION` generations | **Unchanged.** | Not implicated by the diagnostic — kept as a cheap guard against resetting a niche that is still actively improving. |
| Keep-best selection: individual closest to the niche's reference ray | **Unchanged.** | Not implicated. |
| Archive untouched by the operator | **Unchanged.** | Not implicated — same reasoning as the original design (`_archive_update`'s own dominance check self-corrects if a reset niche finds something better). |

## Goals

- Give QINSGA3 a mechanism that can actually detect and break up the
  X-space-diversity collapse the diagnostic chapter measured, instead of one
  that is mathematically incapable of seeing it.
- Reuse as much of the original (reviewed, tested, git-history) DPQiEA
  scaffolding as still applies: per-niche stagnation tracking (eq. 13),
  keep-best-by-reference-ray selection, archive non-interference.
- Validate with the same protocol as every other QINSGA3 change in this
  project: 3 seeds first (shared ideal/nadir vs. `Solvers/NSGA3`'s cache,
  Mann-Whitney U), scale to 20 only if there is a positive signal.

## Scope

**`Solvers/QINSGA3/algorithm.py::run_qinsga3` only** — the real IRP solver.
Unlike the original DPQiEA port, this is *not* wired into
`validation/algorithms/qinsga3/core.py::run_qinsga3_generic` (the DTLZ/MaF
benchmark runner). Motivation: the diagnostic that justifies this operator is
IRP-specific (13x chromosome-diversity gap, measured on the IRP only); QI-NSGA-III
already wins 15/25 DTLZ/MaF cases with the current config, so touching that
code path risks regressing an already-validated result for no diagnosed
problem there. Also, `run_qinsga3`'s loop structure changed materially since
the original DPQiEA port (elitist survival + X-space SBX/PM were adopted
afterward) — see "Wiring" below for how the insertion point maps onto the
*current* loop, not the pre-refactor one the original port was written
against.

## Non-goals

Same as the original design's non-goal list: no change to the rotation gate,
θ-encoding/measurement, `_select_guides`, `_archive_update`, or the archive
itself; no change to parameters shared with NSGA-III (`p_cross`, `eta_cross`,
`p_mut`, `eta_mut`). This stays an additive, disable-by-default operator
(`T_STAGNATION > 0` idiom, matching `migration_period`/`noise_scale`).

## Design

### State (per generation loop, `run_qinsga3` and `run_qinsga3_generic`)

- `niche_guide_history: dict[int, np.ndarray]` — last generation's guide θ per niche.
- `niche_stagnation: dict[int, int]` — consecutive-generations-unchanged counter per niche.

### Helpers (`Solvers/QINSGA3/algorithm.py`)

- `_update_niche_stagnation(assoc, guides_theta, history, counters) -> (history, counters, stagnant_niches)`
  — unchanged from the original port (eq. 13).
- `_recentring_reset_mask(assoc, qpop_theta, F_norm, ref_dirs, stagnant_niches, delta) -> reset_mask`
  — for each stagnant niche with ≥2 members: pairwise θ-distance (eq. 12
  formula, unmodified) among *all* niche members (no eq. 11 pre-filter);
  members within `delta` of at least one other member form a cluster; within
  each cluster, keep the member closest to the reference ray, mark the rest
  in `reset_mask`.

### Loop wiring

The current `run_qinsga3` loop (unlike the pre-refactor version the original
DPQiEA port was written against) computes `assoc`/`guides_theta`/`F_norm` from
the **parent** population early in the generation (around the existing
`_select_guides` call), then later replaces `qpop.theta` wholesale via elitist
survival (`survival.do()`, merging parent+offspring). By the time survival has
run, there is no fresh niche assignment for the *survived* population — this
is the same approximation `_migrate` (already in production) already makes:
it applies the parent-population's `assoc` positionally against the
post-survival `qpop.theta`. This operator follows that same, already-accepted
convention rather than inventing a stricter one, and sits at the same point in
the loop as `_migrate`, right after the elitist survival step:

```python
niche_guide_history, niche_stagnation, stagnant = _update_niche_stagnation(
    assoc, guides_theta, niche_guide_history, niche_stagnation, t_stagnation)

if t_stagnation > 0 and stagnant:
    reset_mask = _recentring_reset_mask(
        assoc, qpop.theta, F_norm, ref_dirs, stagnant, delta_similar)
    n_reset = int(reset_mask.sum())
    if n_reset:
        qpop.theta[reset_mask] = rng.uniform(0.0, np.pi / 2.0, size=(n_reset, qpop.theta.shape[1]))

if (arch_F_norm is not None and migration_period > 0 and gen % migration_period == 0):
    _migrate(...)
```

`niche_guide_history`/`niche_stagnation` are updated using this generation's
`guides_theta` (computed earlier, before rotation) — same semantics as the
original port: it tracks whether the niche's guide changed generation to
generation, independent of when in the generation the reset itself is applied.

### New parameters

`DELTA` (cluster-similarity threshold) and `T_STAGNATION` (generations of
unchanged guide before a niche is eligible). No `GAMMA` — the convergence gate
it controlled is dropped. Starting values:

- `T_STAGNATION = 5` — paper's own best-performing values cluster at 2 or 5; no
  evidence from this diagnostic to deviate.
- `DELTA = 0.05` — the diagnostic measured 47% of gene values within 0.05 rad
  of π/4 (78% within 0.1 rad) on the *already-filtered final Pareto front*,
  which under-estimates raw working-population tightness. Starting at the
  tighter end avoids the operator firing on nearly the whole population every
  stagnant generation, which would just be repeated full-niche
  re-randomisation rather than a targeted correction. Flagged, like in the
  original design, as the primary ablation target if the 3-seed result is
  ambiguous.

### Testing

`Solvers/QINSGA3/test_algorithm.py` (new file, mirrors the original port's
now-removed test file): `_update_niche_stagnation` unit tests carried over
unchanged; new tests for `_recentring_reset_mask` — a niche with 2 members
within `delta` of each other resets the one farther from the reference ray to
a value in `[0, π/2)` different from its pre-reset value; a niche with members
farther apart than `delta` resets nothing; a non-stagnant niche is untouched
regardless of clustering.

### Empirical validation

3 seeds first (project's established fast-reject protocol), shared ideal/nadir
with `Solvers/NSGA3`'s cache, Mann-Whitney U on HV/GD/IGD/Spacing
(`sensitivity/compare_qinsga3_vs_nsga3.py`'s methodology). If there's a
positive signal, scale to 20 seeds to match the validated-result table in
`Solvers/IRP_results_summary.md`. If not: document as rejected in that same
file, same as the four prior remedies, with the chromosome-diversity number
(Var(X norm.)) recorded before/after so a null result is still informative
about whether the operator fired at all.

## Risk

`DELTA=0.05` is a guess anchored to the final-front measurement, not the raw
working population — the actual right order of magnitude could be off in
either direction. If the 3-seed run shows the operator essentially never
firing (check via a printed `n_reset` count per generation, not just the final
metrics), that is itself informative and worth widening `DELTA` for a second
3-seed pass before concluding the trigger design is still wrong.

## Amendment (found during implementation): the stagnation gate (eq. 13) is
## also a mismatch for this architecture — dropped

Before running the 3-seed validation, an instrumented run on the real
100-client instance (60 generations, `t_stagnation=5`, pop_size=200) checked
how often `_recentring_reset_mask` was even *called* (i.e. how often any niche
reached 5 consecutive generations with an unchanged guide, eq. 13's own
condition, ported unmodified from the original DPQiEA design). Result: **0
calls in 60 generations** — the gate never opened.

Direct measurement of guide-to-guide change per niche per generation
(931 comparisons) explains why: only 3.1% of consecutive generations have an
*exactly* unchanged guide (`np.array_equal`), and when the guide does change,
the median shift is 0.033 rad — comparable in magnitude to `DELTA` itself, not
numerical noise. Cause: unlike the original binary QEA the paper targets,
QINSGA3's elitist survival re-selects the whole population from a freshly
merged parent+offspring pool (400 individuals) every single generation — there
is structurally no stable "niche champion" for multiple generations running,
even when the niche's members are otherwise tightly clustered and going
nowhere. Eq. 13's persistence requirement was written for an algorithm without
this per-generation full-population churn; here it just disables the operator
almost entirely, independent of the eq. 11 fix already made.

**Change**: drop the stagnation gate (`_update_niche_stagnation`,
`niche_guide_history`, `niche_stagnation`, `t_stagnation`) entirely. The
operator now runs the clustering check (`_recentring_reset_mask`, unchanged
logic otherwise) on **every occupied niche, every generation**, gated only by
`delta_similar > 0.0` (same enable-by-nonzero-parameter idiom as
`migration_period`/`p_cross` elsewhere in this codebase) — no separate
boolean flag or persistence counter. Re-instrumented with the gate removed
(`delta_similar=0.05`, same run): 2163 individuals reset over 60 generations
(mostly 10-100 per generation, generation 0 resets ~194/200 as expected since
initialisation places everyone within `delta` of each other) — the operator
now demonstrably engages throughout the run, not just at initialisation.
