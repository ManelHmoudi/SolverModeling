# QINSGA3 crowding-distance guide selection — design

## Context

`Solvers/IRP_results_summary.md`'s diagnostic chapter established that QI-NSGA-III's
chromosome diversity on the real IRP is ~13x lower than NSGA-III's before any
decoding happens, and that the guided rotation gate (`Δθ = α(g) ×
tanh((θ_guide − θ)/(π/8))`) pulling the whole population toward a single guide
per niche every generation is the cause. Five independent remedies (A-E),
each grounded in a distinct literature source, were tried and rejected — see
that file's "Remèdes testés" section. The consistent pattern across all five:
none of them changed *how* the niche's "best" member (the guide) is chosen —
only its frequency, magnitude, the rotation rule itself (dual-attractor,
momentum, chaotic), or a disruptive external reset. `_select_guides` has
always picked the champion by `d_perp2.argmin()` — the population member
whose projection onto the niche's reference ray is closest to the ray itself
(a **convergence** criterion, from Deb & Jain 2014 §IV-B, borrowed directly
from NSGA-III's own environmental selection).

An external reviewer of the thesis (the diagnostic chapter and remedies A-E)
agreed diversity is not the limiting factor by itself (remedies A/B showed
forcing diversity up degrades quality; C/D showed forcing it down via a
second attractor or momentum degrades quality just as much) but proposed a
different lever, not yet tried: change **which** chromosome becomes the
guide, using an objective-space **quality/diversity** signal instead of a
**convergence-to-reference-ray** signal. Of the three criteria named
(local hypervolume contribution, front contribution, crowding distance),
crowding distance is the one to test first — `_crowding_distance` (NSGA-II,
Deb et al. 2002 §III-B) already exists in `algorithm.py` (line 733, currently
used only for external-archive trimming), is cheap to compute per niche, and
is a well-defined, already-validated-elsewhere-in-this-project diversity
signal. Local HV contribution has no existing implementation and is
materially more expensive per niche per generation; "front contribution" has
no operational definition in the reviewer's note.

## Goals

- Test whether replacing the niche guide's selection criterion — from
  "closest to the reference ray" to "highest crowding distance within the
  niche" — changes QI-NSGA-III's result on the real IRP, isolating this one
  variable exactly as remedies A-E each isolated theirs.
- Reuse `_crowding_distance` unmodified — no new diversity metric invented.
- Follow the project's established fast-reject protocol: 3 seeds first,
  shared ideal/nadir with `Solvers/NSGA3`'s cache, Mann-Whitney U on
  HV/GD/IGD/Spacing, chromosome diversity Var(X norm.) reported regardless of
  outcome. Scale to 20 seeds only on a positive signal.

## Scope

**`Solvers/QINSGA3/algorithm.py::run_qinsga3` only** — the real IRP solver.
Not wired into `validation/algorithms/qinsga3/core.py` (the DTLZ/MaF
benchmark runner), matching the same reasoning already used for the
niche-recentring-reset design: the motivating diagnostic (13x chromosome-
diversity gap, decoder discontinuity) is IRP-specific, and QI-NSGA-III
already wins 17/25 DTLZ/MaF cases with the current guide-selection criterion
— there is no diagnosed problem there to fix, and touching that code path
risks regressing an already-validated result.

## Non-goals

No change to the rotation gate formula, θ-encoding/measurement, the
niche-recentring-reset operator, the archive itself, or `_migrate` (which
also currently uses `d_perp2.argmin()` to pick an archive candidate, but
serves a different purpose — archive→population injection, not per-
generation guide selection — and was left untouched by every prior remedy
for the same reason). No change to parameters shared with NSGA-III.

## Design

### New helpers (`Solvers/QINSGA3/algorithm.py`, next to `_select_guides` /
`_supplement_from_archive`)

- `_select_guides_crowding(assoc, pareto_idx, F_norm, qpop_theta) ->
  np.ndarray` — same shape and iteration structure as `_select_guides`
  (vectorised over occupied niches), but within each niche with ≥2 Pareto
  members, the champion is `qpop_theta[same_idx[_crowding_distance(F_norm[same_idx]).argmax()]]`
  instead of the `d_perp2.argmin()` member. `ref_dirs` is not needed by this
  function (crowding distance doesn't reference the reference direction) —
  `assoc`/`pareto_idx` already encode niche membership. Niches with exactly 1
  Pareto member: unchanged, guide is that member. The global fallback for
  niches with no Pareto representative (closest-to-origin Pareto member)
  stays exactly as `_select_guides` computes it — it is a rare, last-resort
  path (superseded by archive supplementation whenever the archive holds ≥4
  members), not the criterion under test.
- `_supplement_from_archive_crowding(guides_theta, assoc, pareto_assoc,
  arch_theta, arch_F_norm, ref_dirs) -> np.ndarray` — same structure as
  `_supplement_from_archive`, same `d_perp2.argmax()` → crowding-distance
  substitution for uncovered-niche candidates drawn from the archive.
  `ref_dirs` is still needed here, only to compute `arch_assoc` via
  `_assign_ref_dirs` (niche membership itself is still by reference-ray
  association — only the in-niche "best" tie-break criterion changes).

Both variants are added, not substituted in place, mirroring how
`_select_guides_ring` sits alongside `_select_guides` — the baseline stays
available and default.

### Wiring into `run_qinsga3`

New parameter `use_crowding_guides: bool = False`, default off. In the
per-generation guide-selection block:

```python
if use_ring_guides:
    guides_theta = _select_guides_ring(assoc, F_norm, ref_dirs, qpop.theta)
elif use_crowding_guides:
    guides_theta = _select_guides_crowding(assoc, pareto_idx, F_norm, qpop.theta)
else:
    guides_theta = _select_guides(assoc, pareto_idx, F_norm, ref_dirs, qpop.theta)
```

and, at the existing archive-supplementation call site further down the same
generation:

```python
if arch_theta_arr is not None:
    supplement_fn = _supplement_from_archive_crowding if use_crowding_guides else _supplement_from_archive
    guides_theta = supplement_fn(guides_theta, assoc, pareto_assoc, arch_theta_arr, arch_F_norm, ref_dirs)
```

(exact call-site variable names to match whatever the current supplementation
call already uses — no behavioural change to when supplementation runs, only
which criterion it uses when `use_crowding_guides=True`).

Docstring addition on `run_qinsga3`, same style as the existing
`use_ring_guides` paragraph: what it replaces, that it's mutually exclusive
with `use_ring_guides` (both replace the same guide-selection step), and that
it is combinable in principle with the rotation-rule remedies
(RQPSO/PSO/chaotic) but validated alone first.

### Testing (`Solvers/QINSGA3/test_algorithm.py`)

New unit tests for `_select_guides_crowding`:

- A niche constructed so the reference-ray-closest member and the
  highest-crowding-distance member are two different, known individuals
  (e.g. one clustered point plus one boundary point on an objective) —
  assert the function returns the boundary point's θ, and that this differs
  from what `_select_guides` would return on the same inputs.
- A niche with exactly 1 Pareto member — guide is that member (trivial case,
  same as `_select_guides`).
- A niche with exactly 2 Pareto members — `_crowding_distance` assigns `inf`
  to both boundary points in a 2-point set; assert the function still
  returns one of the two deterministically (via `argmax`'s first-index tie
  break) without raising.

New unit tests for `_supplement_from_archive_crowding`, mirroring the
existing `_supplement_from_archive` tests: an uncovered niche with ≥2 archive
candidates picks the highest-crowding-distance one, not the reference-ray-
closest one, on a constructed example where they differ.

### Empirical validation (`sensitivity/compare_crowding_guides.py`, new file)

Direct copy of `sensitivity/compare_ring_guides.py`'s structure (same
imports, same `_decode_to_F`/`_load_nsga3_reference`/`_chrom_diversity`
helpers, same CLI args): instance 100 clients, `DEFAULT_SEEDS = [42, 137,
271]`, 300 generations, pop 200. Runs baseline (`use_crowding_guides=False`,
implicit default) vs test (`use_crowding_guides=True`) vs the NSGA-III cache,
with a shared global ideal/nadir across all three. Reports HV/GD/IGD/Spacing
means, chromosome diversity Var(X norm.) for both QINSGA3 variants, and two
Mann-Whitney U passes (baseline vs test; test vs NSGA-III) — identical
metric set and statistical protocol to every prior remedy comparison script.

### Documentation

New section "Remède F — guide de niche par crowding distance" in
`Solvers/IRP_results_summary.md`, following the exact format of remedies A-E
(what changed, the result table, Mann-Whitney outcome, chromosome-diversity
before/after, an interpretation paragraph) — written up regardless of
outcome, matching the project's established practice of documenting
rejections with the same rigor as adoptions.

## Risk

Crowding distance rewards *boundary/extreme* solutions within a niche (`inf`
at each objective's min/max), not solutions that are merely more spread out
than the ray-closest one — so this is not a generic "prefer diverse guides"
change, it specifically pulls the whole niche toward whichever member is most
extreme on some objective. If most niches are small (~4-5 members across
~43 occupied niches, per the ring-guides finding in the diagnostic chapter),
a large fraction of members may receive `inf` and the criterion degenerates
to `argmax`'s first-index tie-break — worth checking with a printed
per-generation count of niches hitting this degenerate case, not just the
final HV/GD/IGD numbers, so a null result is informative about whether the
criterion was even meaningfully exercised.
