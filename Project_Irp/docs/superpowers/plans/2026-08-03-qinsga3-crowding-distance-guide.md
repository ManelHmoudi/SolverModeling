# QINSGA3 Crowding-Distance Guide Selection (Remède F) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and empirically validate a new QI-NSGA-III guide-selection variant that picks each niche's rotation target by highest crowding distance instead of lowest perpendicular distance to the reference ray, per the approved design spec.

**Architecture:** Two new pure helper functions in `Solvers/QINSGA3/algorithm.py` mirror the existing `_select_guides` / `_supplement_from_archive` pair exactly, swapping the in-niche "best" criterion from `d_perp2.argmin()` to `_crowding_distance(...).argmax()` (an existing NSGA-II helper already used for archive trimming). A new `use_crowding_guides` boolean parameter on `run_qinsga3` switches to the new pair at both call sites where `guides_theta` is built, mutually exclusive with the existing `use_ring_guides` switch. Validated with the project's standing 3-seed fast-reject protocol via a new `sensitivity/compare_crowding_guides.py` script (a structural copy of `compare_ring_guides.py`), then documented in `Solvers/IRP_results_summary.md` as "Remède F" regardless of outcome.

**Tech Stack:** Python, NumPy, pymoo (`ReferenceDirectionSurvival`, SBX/PM), pytest-style plain-assert test functions (no framework — see `Solvers/QINSGA3/test_algorithm.py`'s existing style), scipy.stats.mannwhitneyu.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-03-qinsga3-crowding-distance-guide-design.md` — every task below implements one part of it; do not deviate from its function signatures.
- Scope: `Solvers/QINSGA3/algorithm.py`, `Solvers/QINSGA3/test_algorithm.py`, `sensitivity/compare_crowding_guides.py`, `Solvers/IRP_results_summary.md` only. No changes to `validation/algorithms/qinsga3/core.py`, `_migrate`, the rotation gate, θ-encoding, or NSGA-III-shared parameters.
- `_migrate` is explicitly out of scope — do not touch it in any task.
- Every new function must carry a docstring citing the spec file, matching the style already used throughout `algorithm.py` (e.g. `_select_guides_ring`'s docstring).
- Run all commands from the repository root (`c:\Users\Mariem\OneDrive\Bureau\SolverModeling\Project_Irp`).

---

### Task 1: `_select_guides_crowding` helper + unit tests

**Files:**
- Modify: `Solvers/QINSGA3/algorithm.py` (add new function directly after `_select_guides_ring`, i.e. after its closing `return guides_theta` around line 282, before the `_supplement_from_archive` section)
- Test: `Solvers/QINSGA3/test_algorithm.py`

**Interfaces:**
- Consumes: `_crowding_distance(F: np.ndarray) -> np.ndarray` (already defined in `algorithm.py:733`, no changes needed — Python resolves the call at run time, so it is safe to call before its own definition line in the file).
- Produces: `_select_guides_crowding(assoc: np.ndarray, pareto_idx: np.ndarray, F_norm: np.ndarray, qpop_theta: np.ndarray) -> np.ndarray` — consumed by Task 3.

- [ ] **Step 1: Write the failing tests**

Add to `Solvers/QINSGA3/test_algorithm.py`, first extending the import block at the top of the file:

```python
from Solvers.QINSGA3.algorithm import (
    _normalise_F, _compute_nadir, _recentring_reset_mask,
    _update_pbest, _rqpso_rotate, _select_guides_ring,
    _max_min_density, _domination_counts, _adaptive_inertia, _pso_rotate,
    _elite_rms_distance, _chaotic_lambda_seed, _chaotic_lambda_step, _chaotic_rotate,
    _select_guides_crowding, _supplement_from_archive_crowding,
)
```

Then append this new section at the end of the file:

```python
# ── _select_guides_crowding ─────────────────────────────────────────────
# Remedy F: niche champion chosen by HIGHEST crowding distance instead of
# LOWEST perpendicular distance to the reference ray. See
# docs/superpowers/specs/2026-08-03-qinsga3-crowding-distance-guide-design.md.

def test_select_guides_crowding_single_pareto_member_guides_toward_itself():
    assoc      = np.array([0])
    pareto_idx = np.array([0])
    F_norm     = np.array([[1.0, 0.0]])
    theta      = np.array([[0.3, 0.3]])

    guides = _select_guides_crowding(assoc, pareto_idx, F_norm, theta)

    assert np.allclose(guides[0], theta[0])


def test_select_guides_crowding_two_member_niche_picks_deterministically_without_crash():
    """With exactly 2 Pareto members and 2 objectives, _crowding_distance
    assigns inf to both (both are boundary points on every objective) --
    argmax's first-index tie-break must still return a valid, non-crashing
    result (the degenerate case flagged in the design doc's Risk section)."""
    assoc      = np.array([0, 0])
    pareto_idx = np.array([0, 1])
    F_norm     = np.array([[0.0, 1.0], [1.0, 0.0]])
    theta      = np.array([[0.2, 0.2], [0.8, 0.8]])

    guides = _select_guides_crowding(assoc, pareto_idx, F_norm, theta)

    assert np.allclose(guides[0], theta[0])
    assert np.allclose(guides[1], theta[0])


def test_select_guides_crowding_differs_from_ray_closest_champion():
    """4-member niche where the crowding-distance winner and the
    reference-ray-closest member (what _select_guides would pick) are
    different, known individuals -- proves the criterion swap actually
    changes which chromosome becomes the guide.

    Dataset (verified by hand against both formulas):
      P0=[.5,.5]  P1=[.1,.6]  P2=[.9,.4]  P3=[.3,.55]
      d_perp2 to ref_dir [1,1]: P0=0.0 (ray-closest) < P3=.0313 < P1=.125 ~= P2=.1251
      crowding distance (2 objectives): P0=1.5 (finite), P1=inf, P2=inf, P3=1.0
      -> argmax picks P1 (first index at the max/inf value) -- NOT P0.
    """
    assoc      = np.array([0, 0, 0, 0])
    pareto_idx = np.array([0, 1, 2, 3])
    F_norm     = np.array([
        [0.5, 0.5],
        [0.1, 0.6],
        [0.9, 0.4],
        [0.3, 0.55],
    ])
    theta = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ])

    guides = _select_guides_crowding(assoc, pareto_idx, F_norm, theta)

    assert np.allclose(guides[0], theta[1])   # P1's theta, not P0's
    assert not np.allclose(guides[0], theta[0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Solvers/QINSGA3/test_algorithm.py -k select_guides_crowding -v`
Expected: FAIL/ERROR — `ImportError: cannot import name '_select_guides_crowding'` (and the sibling `_supplement_from_archive_crowding`, added together for import-block convenience but implemented in Task 2 — expect this whole run to error at collection until Task 2 also lands; that is fine, this step only confirms the *new* test names are wired up and would fail for the right reason once the function exists). If the collection error blocks even seeing the 3 new test names, temporarily comment out the `_supplement_from_archive_crowding` import to confirm these 3 fail on `_select_guides_crowding` alone, then restore it before Step 3.

- [ ] **Step 3: Implement `_select_guides_crowding`**

In `Solvers/QINSGA3/algorithm.py`, insert immediately after `_select_guides_ring`'s closing `return guides_theta` (the blank line before the `_supplement_from_archive` block):

```python
def _select_guides_crowding(
    assoc:      np.ndarray,
    pareto_idx: np.ndarray,
    F_norm:     np.ndarray,
    qpop_theta: np.ndarray,
) -> np.ndarray:
    """Remedy F: same niche-champion broadcast as _select_guides, but the
    champion is chosen by HIGHEST crowding distance within the niche
    [Deb et al. 2002, §III-B -- _crowding_distance, already used elsewhere
    in this module for archive trimming] instead of LOWEST perpendicular
    distance to the reference ray. See docs/superpowers/specs/
    2026-08-03-qinsga3-crowding-distance-guide-design.md.

    ref_dirs is not needed here -- crowding distance doesn't reference the
    niche's ray, and assoc/pareto_idx already encode niche membership. The
    global fallback (closest-to-origin Pareto member, for niches with no
    Pareto representative) is unchanged from _select_guides -- it is not the
    criterion under test.
    """
    N            = len(assoc)
    F_par_n      = F_norm[pareto_idx]
    global_fb    = qpop_theta[pareto_idx[np.linalg.norm(F_par_n, axis=1).argmin()]]
    pareto_assoc = assoc[pareto_idx]

    guides_theta = np.tile(global_fb, (N, 1))

    for rd in np.unique(pareto_assoc):
        same_mask = pareto_assoc == rd
        same_idx  = pareto_idx[same_mask]

        if len(same_idx) == 1:
            best_theta = qpop_theta[same_idx[0]]
        else:
            F_same     = F_norm[same_idx]
            cd         = _crowding_distance(F_same)
            best_theta = qpop_theta[same_idx[cd.argmax()]]

        guides_theta[assoc == rd] = best_theta

    return guides_theta
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Solvers/QINSGA3/test_algorithm.py -k select_guides_crowding -v`
Expected: the 3 tests from Step 1 PASS (an import error for `_supplement_from_archive_crowding` is still expected/acceptable at this point if Task 2 hasn't landed yet — if so, run with `-k "select_guides_crowding and not supplement"` to isolate).

- [ ] **Step 5: Commit**

```bash
git add Solvers/QINSGA3/algorithm.py Solvers/QINSGA3/test_algorithm.py
git commit -m "feat: add crowding-distance niche-guide selection (remede F, part 1/2)"
```

---

### Task 2: `_supplement_from_archive_crowding` helper + unit tests

**Files:**
- Modify: `Solvers/QINSGA3/algorithm.py` (add new function directly after `_supplement_from_archive`'s closing `return guides_theta`, before the `# Migration` section header)
- Test: `Solvers/QINSGA3/test_algorithm.py`

**Interfaces:**
- Consumes: `_assign_ref_dirs(F_norm, ref_dirs) -> np.ndarray` and `_crowding_distance(F) -> np.ndarray` (both already defined in `algorithm.py`).
- Produces: `_supplement_from_archive_crowding(guides_theta, assoc, pareto_assoc, arch_theta, arch_F_norm, ref_dirs) -> np.ndarray` — consumed by Task 3.

- [ ] **Step 1: Write the failing test**

Append to `Solvers/QINSGA3/test_algorithm.py` (the import for `_supplement_from_archive_crowding` was already added in Task 1's Step 1):

```python
# ── _supplement_from_archive_crowding ───────────────────────────────────

def test_supplement_from_archive_crowding_fills_uncovered_niche_by_crowding():
    """Niche 1 has no Pareto representative (pareto_assoc only covers niche
    0); 4 archive candidates are all associated to niche 1. Reuses the exact
    dataset from test_select_guides_crowding_differs_from_ray_closest_champion
    (verified: crowding winner = index 1, ray-closest winner = index 0) to
    show the archive-fallback path picks the same, different champion the
    base _supplement_from_archive would not."""
    guides_theta = np.array([[9.0, 9.0]])   # placeholder, must be overwritten
    assoc        = np.array([1])
    pareto_assoc = np.array([0])            # niche 0 covered, niche 1 is not
    ref_dirs     = np.array([[1.0, 0.0], [1.0, 1.0]])
    arch_F_norm  = np.array([
        [0.5, 0.5],
        [0.1, 0.6],
        [0.9, 0.4],
        [0.3, 0.55],
    ])
    arch_theta = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ])

    result = _supplement_from_archive_crowding(
        guides_theta, assoc, pareto_assoc, arch_theta, arch_F_norm, ref_dirs,
    )

    assert np.allclose(result[0], arch_theta[1])   # crowding winner
    assert not np.allclose(result[0], arch_theta[0])  # not the ray-closest winner


def test_supplement_from_archive_crowding_leaves_covered_niches_untouched():
    guides_theta = np.array([[7.0, 7.0]])
    assoc        = np.array([0])
    pareto_assoc = np.array([0])   # niche 0 IS covered -> no supplementation
    ref_dirs     = np.array([[1.0, 0.0]])
    arch_F_norm  = np.array([[0.2, 0.3]])
    arch_theta   = np.array([[9.0, 9.0]])

    result = _supplement_from_archive_crowding(
        guides_theta, assoc, pareto_assoc, arch_theta, arch_F_norm, ref_dirs,
    )

    assert np.allclose(result[0], guides_theta[0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Solvers/QINSGA3/test_algorithm.py -k supplement_from_archive_crowding -v`
Expected: FAIL — `ImportError: cannot import name '_supplement_from_archive_crowding'`.

- [ ] **Step 3: Implement `_supplement_from_archive_crowding`**

In `Solvers/QINSGA3/algorithm.py`, insert immediately after `_supplement_from_archive`'s closing `return guides_theta`, before the `# Migration` section comment block:

```python
def _supplement_from_archive_crowding(
    guides_theta: np.ndarray,
    assoc:        np.ndarray,
    pareto_assoc: np.ndarray,
    arch_theta:   np.ndarray,
    arch_F_norm:  np.ndarray,
    ref_dirs:     np.ndarray,
) -> np.ndarray:
    """Remedy F counterpart to _supplement_from_archive: fills the same
    uncovered-niche guides from the external archive, but the archive
    candidate is chosen by highest crowding distance within the niche
    instead of lowest perpendicular distance to the reference ray -- kept
    consistent with _select_guides_crowding so no single generation mixes
    the two criteria across niches. See docs/superpowers/specs/
    2026-08-03-qinsga3-crowding-distance-guide-design.md.

    ref_dirs is still needed here (only) to compute arch_assoc via
    _assign_ref_dirs -- niche MEMBERSHIP is still by reference-ray
    association; only the in-niche tie-break criterion changes.
    """
    arch_assoc = _assign_ref_dirs(arch_F_norm, ref_dirs)
    covered    = set(pareto_assoc.tolist())

    pop_rds           = np.unique(assoc)
    uncovered_pop_rds = pop_rds[~np.isin(pop_rds, list(covered))]

    for rd in uncovered_pop_rds:
        in_niche = np.where(arch_assoc == rd)[0]
        if len(in_niche) == 0:
            continue
        if len(in_niche) == 1:
            best_theta = arch_theta[in_niche[0]]
        else:
            F_cand     = arch_F_norm[in_niche]
            cd         = _crowding_distance(F_cand)
            best_theta = arch_theta[in_niche[cd.argmax()]]

        guides_theta[assoc == rd] = best_theta

    return guides_theta
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Solvers/QINSGA3/test_algorithm.py -k "select_guides_crowding or supplement_from_archive_crowding" -v`
Expected: all 5 tests (3 from Task 1 + 2 from this task) PASS.

- [ ] **Step 5: Run the full existing test suite to confirm no regressions**

Run: `python -m pytest Solvers/QINSGA3/test_algorithm.py -v`
Expected: all tests PASS (existing tests untouched, new ones added).

- [ ] **Step 6: Commit**

```bash
git add Solvers/QINSGA3/algorithm.py Solvers/QINSGA3/test_algorithm.py
git commit -m "feat: add crowding-distance archive supplementation (remede F, part 2/2)"
```

---

### Task 3: Wire `use_crowding_guides` into `run_qinsga3`

**Files:**
- Modify: `Solvers/QINSGA3/algorithm.py` (function signature ~line 878, docstring ~line 920-924, two call sites ~line 1017-1020 and ~line 1032-1036)

**Interfaces:**
- Consumes: `_select_guides_crowding` (Task 1), `_supplement_from_archive_crowding` (Task 2), both already-existing `_select_guides`/`_supplement_from_archive`.
- Produces: `run_qinsga3(..., use_crowding_guides: bool = False, ...)` — consumed by Task 4's comparison script.

- [ ] **Step 1: Add the new parameter to `run_qinsga3`'s signature**

In `Solvers/QINSGA3/algorithm.py`, find:

```python
    use_rqpso_rotation: bool = False,
    use_ring_guides:  bool   = False,
    use_pso_rotation: bool   = False,
```

Replace with:

```python
    use_rqpso_rotation: bool = False,
    use_ring_guides:  bool   = False,
    use_crowding_guides: bool = False,
    use_pso_rotation: bool   = False,
```

- [ ] **Step 2: Add the docstring paragraph**

Find the end of the `use_ring_guides` docstring paragraph:

```python
    use_ring_guides (disabled by default) replaces _select_guides's single
    niche-wide champion with the Ring-structured local guide from
    Tayarani-N & Akbarzadeh-T (2014) §3 -- see _select_guides_ring's
    docstring. Combinable with use_rqpso_rotation (gbest becomes the ring
    guide instead of the niche champion) but validated independently first.
```

Insert immediately after it (still inside the same docstring, before `use_pso_rotation`'s paragraph):

```python

    use_crowding_guides (disabled by default) replaces _select_guides's
    reference-ray-closest niche champion (and _supplement_from_archive's
    archive fallback) with the HIGHEST-crowding-distance member instead --
    see _select_guides_crowding's docstring for the full rationale and
    docs/superpowers/specs/2026-08-03-qinsga3-crowding-distance-guide-design.md
    for the design. Mutually exclusive with use_ring_guides (both replace
    the same guide-selection step) -- combinable in principle with the
    rotation-rule variants (use_rqpso_rotation/use_pso_rotation/
    use_chaotic_rotation) but validated alone first, same as every other
    remedy in this module.
```

- [ ] **Step 3: Wire the guide-selection call site**

Find:

```python
            if use_ring_guides:
                guides_theta = _select_guides_ring(assoc, F_norm, ref_dirs, qpop.theta)
            else:
                guides_theta = _select_guides(assoc, pareto_idx, F_norm, ref_dirs, qpop.theta)
```

Replace with:

```python
            if use_ring_guides:
                guides_theta = _select_guides_ring(assoc, F_norm, ref_dirs, qpop.theta)
            elif use_crowding_guides:
                guides_theta = _select_guides_crowding(assoc, pareto_idx, F_norm, qpop.theta)
            else:
                guides_theta = _select_guides(assoc, pareto_idx, F_norm, ref_dirs, qpop.theta)
```

- [ ] **Step 4: Wire the archive-supplementation call site**

Find:

```python
                pareto_assoc   = assoc[pareto_idx]
                guides_theta   = _supplement_from_archive(
                    guides_theta, assoc, pareto_assoc,
                    arch_theta_arr, arch_F_norm, ref_dirs,
                )
```

Replace with:

```python
                pareto_assoc   = assoc[pareto_idx]
                supplement_fn  = (_supplement_from_archive_crowding if use_crowding_guides
                                   else _supplement_from_archive)
                guides_theta   = supplement_fn(
                    guides_theta, assoc, pareto_assoc,
                    arch_theta_arr, arch_F_norm, ref_dirs,
                )
```

- [ ] **Step 5: Sanity-check the module still imports cleanly**

Run: `python -c "from Solvers.QINSGA3.algorithm import run_qinsga3; import inspect; assert 'use_crowding_guides' in inspect.signature(run_qinsga3).parameters; print('OK')"`
Expected: prints `OK` with no traceback.

- [ ] **Step 6: Run the full unit test suite again to confirm no regressions from the wiring edit**

Run: `python -m pytest Solvers/QINSGA3/test_algorithm.py -v`
Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add Solvers/QINSGA3/algorithm.py
git commit -m "feat: wire use_crowding_guides into run_qinsga3 (remede F)"
```

---

### Task 4: `sensitivity/compare_crowding_guides.py`

**Files:**
- Create: `sensitivity/compare_crowding_guides.py`

**Interfaces:**
- Consumes: `run_qinsga3(..., use_crowding_guides=True)` (Task 3), `Solvers/NSGA3/nsga3_chromosomes.json` cache, `Solvers/NSGA3/{decoder,evaluator,metrics,problem}.py` (all pre-existing, unchanged).
- Produces: a runnable script printing HV/GD/IGD/Spacing, chromosome diversity, and two Mann-Whitney U passes — consumed by Task 5 (run it, capture its output).

- [ ] **Step 1: Create the script**

This is a structural copy of `sensitivity/compare_ring_guides.py` (same helpers, same statistical protocol) with the ring-specific parts swapped for the crowding variant:

```python
"""Crowding-distance guide selection test for QINSGA-III (remedy F, an
external thesis reviewer's proposal -- see
docs/superpowers/specs/2026-08-03-qinsga3-crowding-distance-guide-design.md).

Context: five prior structural remedies (A-E, see
Solvers/IRP_results_summary.md's "Remedes testes" section) each isolated a
different variable -- reset frequency/target/magnitude, the rotation rule
itself (dual attractor, momentum, chaos) -- but none changed WHO a niche's
single champion is. _select_guides has always picked the champion by
d_perp2.argmin() (closest to the reference ray -- a CONVERGENCE criterion,
borrowed from NSGA-III's own environmental selection). This remedy replaces
that criterion with the champion of HIGHEST crowding distance within the
niche [Deb et al. 2002, Section III-B] -- a DIVERSITY/QUALITY criterion,
already implemented in this project's own _crowding_distance (used
elsewhere for archive trimming) and named directly by the reviewer.

Already wired into production run_qinsga3 as use_crowding_guides (default
False) -- no separate reimplementation needed.

Usage:
    python -m sensitivity.compare_crowding_guides
    python -m sensitivity.compare_crowding_guides --seeds 42 137 271
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
    print("  CROWDING-DISTANCE GUIDES — QINSGA-III (baseline ray-closest vs NSGA-III cache)")
    print("=" * 92)
    print(f"  Instance : {instance} clients | pop={effective_pop} | gen={max_gen}")
    print(f"  Seeds    : {seeds}")
    print("=" * 92)

    print("\nDecodage du front NSGA-III (cache, reference fixe)...")
    nsga3_F_runs = _load_nsga3_reference(sets_, params_, instance)
    print(f"  NSGA-III : {len(nsga3_F_runs)} runs, "
          f"{sum(len(f) for f in nsga3_F_runs)} solutions")

    baseline_lbl = "baseline (ray-closest)"
    test_lbl     = "Crowding-distance (test)"
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
            pop_size=effective_pop, max_gen=max_gen, seed=seed, use_crowding_guides=True,
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
    parser = argparse.ArgumentParser(description="Test des guides par crowding distance — QINSGA-III")
    parser.add_argument("--instance", default="100",
                        choices=["3", "5", "15", "25", "30", "40", "100"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--gen", type=int, default=300)
    parser.add_argument("--pop", type=int, default=POP_SIZE)
    args = parser.parse_args()
    run_comparison(args.instance, args.seeds, args.gen, args.pop)
```

- [ ] **Step 2: Smoke-test the script on a tiny, fast configuration**

Run: `python -m sensitivity.compare_crowding_guides --instance 5 --seeds 42 --gen 3 --pop 20`
Expected: completes without traceback in well under a minute, prints all report sections (RESULTATS, Diversite chromosome, both Mann-Whitney blocks — these will say "pas assez de runs valides pour un test" with only 1 seed, which is correct/expected at this smoke-test scale, not a bug). This only proves the script runs end-to-end; it is not the empirical validation itself (that's Task 5, at full scale).

- [ ] **Step 3: Commit**

```bash
git add sensitivity/compare_crowding_guides.py
git commit -m "test: add compare_crowding_guides.py sensitivity script (remede F)"
```

---

### Task 5: Empirical validation + documentation

**Files:**
- Modify: `Solvers/IRP_results_summary.md`
- Create: `sensitivity/crowding_guides_campaign_log.txt` (raw run output, matching the existing convention of `sensitivity/pso_rotation_campaign_log.txt` / `sensitivity/chaotic_rotation_campaign_log.txt`)

**Interfaces:**
- Consumes: `sensitivity/compare_crowding_guides.py` (Task 4).
- Produces: a new "Remède F" section in `Solvers/IRP_results_summary.md`, following the exact structure of Remèdes A-E.

- [ ] **Step 1: Run the full-scale validation**

Run (this mirrors remedies A-E's own validation scale — 3 seeds, 300 generations, the real 100-client instance — and will take a comparable amount of wall-clock time to those prior runs, likely tens of minutes; run in the background and capture output to a log file):

```bash
python -m sensitivity.compare_crowding_guides > sensitivity/crowding_guides_campaign_log.txt 2>&1
```

Expected: the command exits 0 and `sensitivity/crowding_guides_campaign_log.txt` contains the full report (RESULTATS table for HV/GD/IGD/Spacing across NSGA-III/baseline/test, chromosome diversity for both QINSGA3 variants, two Mann-Whitney U blocks).

- [ ] **Step 2: Read the captured log and extract the numbers**

Read `sensitivity/crowding_guides_campaign_log.txt`. You need, from its printed tables: HV/GD/IGD/Spacing means for `baseline (ray-closest)` and `Crowding-distance (test)`; the two chromosome-diversity values; the Mann-Whitney U/p for both comparison blocks (baseline vs test, and test vs NSGA-III).

- [ ] **Step 3: Insert the "Remède F" section into `Solvers/IRP_results_summary.md`**

Find the end of the `### Autre tentative testée sans effet mesurable` section (its last paragraph, ending "...le guide local et le champion global finissent souvent par être les mêmes individus.") and the following `## Conclusion` heading. Insert a new section between them, using this exact structure (fill in `<VALUE>` placeholders from Step 2's real numbers — every other word is final text, not a placeholder):

```markdown
### Remède F -- guide de niche par crowding distance

Proposition d'un relecteur externe de la thèse : les remèdes A-E changent
tous la fréquence, la cible ou la règle de la rotation, mais aucun ne touche
le critère utilisé pour choisir LEQUEL des membres d'une niche devient le
champion (`_select_guides` a toujours pris le membre le plus proche du rayon
de référence -- un critère de **convergence**, `d_perp2.argmin()`). Ce
remède le remplace par le membre à plus forte **crowding distance** dans la
niche [Deb et al. 2002, §III-B] -- un critère de diversité déjà implémenté
dans ce module (`_crowding_distance`, utilisé jusque-là pour l'élagage de
l'archive externe). Design complet :
`docs/superpowers/specs/2026-08-03-qinsga3-crowding-distance-guide-design.md`.

**Résultat (3 seeds, 300 générations, instance 100 clients,
`sensitivity/compare_crowding_guides.py`)** :

| Indicateur | Baseline (ray-closest) | Test (crowding) | Mann-Whitney |
|---|---|---|---|
| HV ↑ | <VALUE> | <VALUE> | U=<VALUE>, p=<VALUE> |
| GD ↓ | <VALUE> | <VALUE> | U=<VALUE>, p=<VALUE> |
| IGD ↓ | <VALUE> | <VALUE> | U=<VALUE>, p=<VALUE> |
| Diversité chromosome | <VALUE> | <VALUE> | -- |

Script conservé : `sensitivity/compare_crowding_guides.py`. Log complet :
`sensitivity/crowding_guides_campaign_log.txt`.
```

If the Mann-Whitney result shows a significant improvement on at least HV (p<0.05, higher mean than baseline), append this interpretation paragraph instead of writing your own:

```markdown
**Résultat positif** : contrairement aux remèdes A-E, changer le critère de
sélection du champion (sans changer sa fréquence, sa magnitude, ni la
formule de rotation elle-même) améliore significativement la qualité du
front. Ceci confirme l'hypothèse du relecteur : le problème n'était pas la
notion même de "guide unique par niche", mais le fait que le critère de
convergence (`d_perp2.argmin()`) sélectionne un point qui n'est pas
nécessairement informatif pour un décodeur combinatoire discontinu -- un
critère de diversité objective (crowding distance) l'est davantage.
```

Otherwise (not significant, or significant in the wrong direction), append this one instead:

```markdown
**Sixième remède indépendant, même verdict que A-E** : changer uniquement le
critère de sélection du champion de niche (sans toucher sa fréquence, sa
magnitude, ni la formule de rotation) ne suffit pas non plus à combler
l'écart avec NSGA-III sur l'IRP. Ceci renforce l'hypothèse retenue pour le
mémoire : la limite n'est pas dans le CHOIX du point cible (que ce soit par
convergence ou par diversité), mais dans le principe même de tirer chaque
génération vers UN point unique dans un espace θ dont la géométrie n'est pas
régulière une fois passée par le décodeur.
```

- [ ] **Step 4: Update the Conclusion's bullet list**

In the same file's `## Conclusion` section, after the existing bullet that starts "Le remède E, qui module la magnitude...", add one more bullet consistent with whichever outcome Step 3 recorded — if rejected:

```markdown
- Le remède F, qui change pour la première fois le CRITÈRE de choix du
  champion (crowding distance plutôt que distance au rayon de référence)
  plutôt que sa fréquence, sa cible ou la règle de rotation, a vu <la
  diversité chromosome ET la qualité du front reproduire le même pattern que
  les cinq remèdes précédents> -- sixième mécanisme indépendant, même
  conclusion : ce n'est pas le choix du point cible qui limite QI-NSGA-III
  sur l'IRP.
```

(replace the `<...>` clause with the actual direction observed in Step 2 -- e.g. "la diversité chromosome a légèrement augmenté mais la qualité du front s'est dégradée" or the equivalent phrase matching the real numbers; do not leave the angle brackets in the final text). If Step 3 recorded a positive result instead, do not add a bullet here — instead flag this explicitly in your final report to the user, since a positive Remède F outcome invalidates part of the existing "Hypothèse retenue pour le mémoire" paragraph and that rewrite needs the user's own judgment, not a scripted template.

- [ ] **Step 5: Commit**

```bash
git add Solvers/IRP_results_summary.md sensitivity/crowding_guides_campaign_log.txt
git commit -m "docs: document remede F (crowding-distance guides) result on the IRP"
```
