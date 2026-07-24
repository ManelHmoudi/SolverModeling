# QINSGA3 elitist per-niche guide — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make QINSGA3's per-niche rotation guide elitist by consulting the
existing non-dominated archive for every niche (not just niches empty in the
current Pareto front), fixing a normalisation mismatch that currently makes
population-vs-archive distance comparisons invalid, so a niche's guide can
never regress below the best solution ever found for it.

**Architecture:** `_select_guides` (QINSGA3/algorithm.py) is extended with
two optional parameters (`arch_theta`, `arch_F_norm`) and, for every niche,
now compares the current front's best representative against the archive's
best representative — both measured in the same ideal/nadir frame — keeping
whichever is closer to the reference ray. `_supplement_from_archive` is
deleted; its behaviour (guide from archive when the front has no
representative) falls out of the same comparison as a special case (one side
is `None`). `_normalise_F` is split into a stats helper (`_normalise_stats`,
returns ideal/denom) plus the existing thin wrapper, so callers can project
the archive into the population's exact normalisation frame instead of the
archive's own. Both call sites — the production IRP loop in
`QINSGA3/algorithm.py::run_qinsga3` and the benchmark loop in
`validation/algorithms/qinsga3/core.py::run_qinsga3_generic` — are updated
identically since they already share this code by import, not duplication.

**Tech Stack:** Python, NumPy, pymoo (`NonDominatedSorting`), pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-24-qinsga3-elitist-guide-design.md`.
- `QINSGA3/algorithm.py` is shared by the real IRP production path
  (`QINSGA3/main.py`) and the benchmark harness
  (`validation/algorithms/qinsga3/core.py`) — every change here affects both;
  confirmed intended with the user.
- No change to rotation gate, crossover, mutation operators, or any already-
  tuned parameter (`MIGRATION_PERIOD`, `N_MIGRATE`, `NOISE_SCALE`, etc.).
- `_migrate`'s own code is untouched; it is only affected indirectly because
  the `arch_F_norm` value the callers now pass it is computed in the
  population's frame instead of the archive's own frame (a natural
  consequence of fixing the normalisation mismatch at its source, not a
  separate change).
- Cleanup pass (removing dead code / unnecessary comments) is scoped to the
  files this plan touches only, per user decision — not a project-wide
  sweep.

## File Structure

- Modify `QINSGA3/algorithm.py`: split `_normalise_stats` out of
  `_normalise_F`; merge `_select_guides` + `_supplement_from_archive` into
  one elitist `_select_guides`; delete `_supplement_from_archive`; update
  `run_qinsga3`'s generational loop and module docstring.
- Modify `validation/algorithms/qinsga3/core.py`: update
  `run_qinsga3_generic`'s generational loop and module docstring to match.
- Create `QINSGA3/test_algorithm.py`: first unit tests for this module —
  covers the merged `_select_guides` and the `_normalise_stats`/
  `_normalise_F` split.
- No changes to `QINSGA3/chromosome.py`, `validation/algorithms/qinsga3/
  runner.py`, or any results/report files until the ablation (Task 6)
  and, conditionally, the full campaign (Task 7).

---

### Task 1: Split `_normalise_stats` out of `_normalise_F`

**Files:**
- Modify: `QINSGA3/algorithm.py:103-108`
- Test: `QINSGA3/test_algorithm.py` (new file)

**Interfaces:**
- Produces: `_normalise_stats(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]`
  returning `(ideal, denom)`, and `_normalise_F(F: np.ndarray) -> np.ndarray`
  unchanged in behaviour (now a thin wrapper over `_normalise_stats`).

- [ ] **Step 1: Write the failing test**

Create `QINSGA3/test_algorithm.py` with:

```python
import numpy as np

from QINSGA3.algorithm import _normalise_F, _normalise_stats


def test_normalise_F_matches_normalise_stats_composition():
    """_normalise_F(F) must equal (F - ideal) / denom for the ideal/denom
    _normalise_stats(F) returns -- guards the refactor that splits them apart."""
    F = np.array([[0.0, 2.0], [1.0, 0.0], [2.0, 1.0]])

    ideal, denom = _normalise_stats(F)
    expected = (F - ideal) / denom

    np.testing.assert_allclose(_normalise_F(F), expected)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest QINSGA3/test_algorithm.py -v`
Expected: FAIL — `ImportError: cannot import name '_normalise_stats'`

- [ ] **Step 3: Split the helper**

In `QINSGA3/algorithm.py`, replace the current `_normalise_F` (lines 103-108):

```python
def _normalise_F(F: np.ndarray) -> np.ndarray:
    """Normalise F: ideal point + nadir from hyperplane (Deb & Jain 2014, §IV-A)."""
    ideal = F.min(axis=0)
    nadir = _compute_nadir(F, ideal)
    denom = np.where(nadir - ideal > 1e-9, nadir - ideal, 1.0)
    return (F - ideal) / denom
```

with:

```python
def _normalise_stats(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ideal point + nadir-minus-ideal span for F (Deb & Jain 2014, §IV-A).

    Split out of _normalise_F so callers that need to project a SECOND array
    (e.g. an external archive) into the exact same normalised frame as F can
    reuse (ideal, denom) instead of recomputing their own — required for a
    valid distance comparison between the two arrays.
    """
    ideal = F.min(axis=0)
    nadir = _compute_nadir(F, ideal)
    denom = np.where(nadir - ideal > 1e-9, nadir - ideal, 1.0)
    return ideal, denom


def _normalise_F(F: np.ndarray) -> np.ndarray:
    """Normalise F: ideal point + nadir from hyperplane (Deb & Jain 2014, §IV-A)."""
    ideal, denom = _normalise_stats(F)
    return (F - ideal) / denom
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest QINSGA3/test_algorithm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add QINSGA3/algorithm.py QINSGA3/test_algorithm.py
git commit -m "refactor: split _normalise_stats out of _normalise_F

Lets callers project a second array (the archive) into the exact
normalisation frame used for F, needed for a valid guide-selection
distance comparison in the next commit."
```

---

### Task 2: Elitist `_select_guides` (merge in archive comparison)

**Files:**
- Modify: `QINSGA3/algorithm.py:128-212` (replaces `_select_guides` and
  deletes `_supplement_from_archive`)
- Test: `QINSGA3/test_algorithm.py`

**Interfaces:**
- Consumes: `_assign_ref_dirs(F_norm, ref_dirs) -> np.ndarray` (unchanged,
  from Task 1's file, already defined above `_select_guides`).
- Produces:
  `_select_guides(assoc, pareto_idx, F_norm, ref_dirs, qpop_theta, arch_theta=None, arch_F_norm=None) -> np.ndarray`
  — `arch_theta`/`arch_F_norm` are optional (default `None`); when provided,
  `arch_F_norm` MUST already be normalised in the same frame as `F_norm`
  (the caller's responsibility, wired in Task 3/4). Returns guide theta per
  population member, shape `(len(assoc), n_genes)` — same shape/semantics as
  today.
- `_supplement_from_archive` no longer exists — any remaining caller must be
  updated to pass `arch_theta`/`arch_F_norm` to `_select_guides` instead
  (done in Task 3 and Task 4).

- [ ] **Step 1: Write the failing tests**

Append to `QINSGA3/test_algorithm.py`:

```python
from QINSGA3.algorithm import _select_guides

REF_DIRS_2 = np.array([[1.0, 0.0], [0.0, 1.0]])


def test_select_guides_archive_better_than_front_wins():
    """Archive holds a strictly closer representative for niche 0 than the
    current Pareto front -> the guide for niche 0 comes from the archive."""
    assoc      = np.array([0, 0, 1, 1])
    pareto_idx = np.array([0, 2])
    qpop_theta = np.array([[0.1], [0.2], [0.3], [0.4]])
    F_norm     = np.array([
        [0.5, 0.9],   # individual 0: front's niche-0 pick, d_perp^2 to ray0 = 0.81
        [0.0, 0.0],
        [0.9, 0.5],   # individual 2: front's niche-1 pick, d_perp^2 to ray1 = 0.81
        [0.0, 0.0],
    ])
    arch_theta  = np.array([[0.9]])
    arch_F_norm = np.array([[0.5, 0.1]])   # d_perp^2 to ray0 = 0.01 < front's 0.81

    guides = _select_guides(
        assoc, pareto_idx, F_norm, REF_DIRS_2, qpop_theta,
        arch_theta=arch_theta, arch_F_norm=arch_F_norm,
    )

    assert guides[0, 0] == 0.9   # niche 0 -> archive wins
    assert guides[1, 0] == 0.9
    assert guides[2, 0] == 0.3   # niche 1 -> unaffected, front's own pick
    assert guides[3, 0] == 0.3


def test_select_guides_front_beats_worse_archive():
    """Archive's representative for niche 0 is farther than the front's ->
    guide stays on the front's pick (no regression from today's behaviour)."""
    assoc      = np.array([0, 0])
    pareto_idx = np.array([0])
    qpop_theta = np.array([[0.1], [0.2]])
    F_norm     = np.array([
        [0.5, 0.9],   # front pick, d_perp^2 to ray0 = 0.81
        [0.0, 0.0],
    ])
    arch_theta  = np.array([[0.9]])
    arch_F_norm = np.array([[0.5, 0.95]])   # d_perp^2 = 0.9025 > front's 0.81

    guides = _select_guides(
        assoc, pareto_idx, F_norm, REF_DIRS_2, qpop_theta,
        arch_theta=arch_theta, arch_F_norm=arch_F_norm,
    )

    assert guides[0, 0] == 0.1
    assert guides[1, 0] == 0.1


def test_select_guides_uncovered_niche_uses_archive():
    """A niche with no Pareto-front representative but an archive one ->
    guide comes from the archive (matches old _supplement_from_archive)."""
    ref_dirs   = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assoc      = np.array([0, 2, 2])
    pareto_idx = np.array([0])       # only niche 0 covered by the front
    qpop_theta = np.array([[0.1], [0.2], [0.3]])
    F_norm     = np.array([
        [0.5, 0.9],
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    arch_theta  = np.array([[0.7]])
    arch_F_norm = np.array([[1.0, 1.0]])   # exactly on ray2 = (1,1)

    guides = _select_guides(
        assoc, pareto_idx, F_norm, ref_dirs, qpop_theta,
        arch_theta=arch_theta, arch_F_norm=arch_F_norm,
    )

    assert guides[1, 0] == 0.7
    assert guides[2, 0] == 0.7


def test_select_guides_without_archive_matches_front_only_behaviour():
    """No archive passed (None) -> identical output to front-only selection,
    i.e. the behaviour used before the archive reaches its 4-entry threshold."""
    assoc      = np.array([0, 0, 1, 1])
    pareto_idx = np.array([0, 2])
    qpop_theta = np.array([[0.1], [0.2], [0.3], [0.4]])
    F_norm     = np.array([
        [0.5, 0.9],
        [0.0, 0.0],
        [0.9, 0.5],
        [0.0, 0.0],
    ])

    guides = _select_guides(assoc, pareto_idx, F_norm, REF_DIRS_2, qpop_theta)

    assert guides[0, 0] == 0.1
    assert guides[2, 0] == 0.3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest QINSGA3/test_algorithm.py -v`
Expected: the four new tests FAIL (`TypeError: _select_guides() got an
unexpected keyword argument 'arch_theta'` for the first three; the fourth
currently passes already since it matches today's behaviour — that one is
allowed to pass at this step, the other three must fail).

- [ ] **Step 3: Replace `_select_guides` and delete `_supplement_from_archive`**

In `QINSGA3/algorithm.py`, replace the full block from the start of
`_select_guides` (currently line 128) through the end of
`_supplement_from_archive` (currently line 212) with:

```python
def _select_guides(
    assoc:       np.ndarray,
    pareto_idx:  np.ndarray,
    F_norm:      np.ndarray,
    ref_dirs:    np.ndarray,
    qpop_theta:  np.ndarray,
    arch_theta:  np.ndarray | None = None,
    arch_F_norm: np.ndarray | None = None,
) -> np.ndarray:
    """Return elitist guide theta angles for each individual, one per niche.

    For every niche present in `assoc`, compares the current Pareto front's
    best representative (smallest perpendicular distance to the reference
    ray) against the archive's best representative in the same niche, and
    keeps whichever is closer. A niche with a representative on only one
    side uses that side; a niche with neither uses the global fallback
    (closest-to-origin Pareto member) — same fallback as before.

    `arch_theta`/`arch_F_norm` are optional (default None, meaning "no
    archive yet" — same output as before this function had archive support).
    When provided, `arch_F_norm` MUST already be normalised in the SAME
    ideal/nadir frame as `F_norm` — the caller's responsibility — otherwise
    the two perpendicular distances are not on a comparable scale.

    This generalises the old split between front-only `_select_guides` and
    archive-only-for-empty-niches `_supplement_from_archive`: the archive is
    elitist by construction (`_archive_update` never lets a dominated
    solution survive), so consulting it for every niche — not just
    uncovered ones — stops a niche's guide from regressing across
    generations [Han & Kim 2002 elitism principle; Zhang 2011 survey,
    "attractor replaced only if better"].
    """
    N         = len(assoc)
    F_par_n   = F_norm[pareto_idx]
    global_fb = qpop_theta[pareto_idx[np.linalg.norm(F_par_n, axis=1).argmin()]]

    ref_norms = np.linalg.norm(ref_dirs, axis=1, keepdims=True)
    ref_unit  = ref_dirs / np.where(ref_norms > 1e-9, ref_norms, 1.0)

    have_archive = arch_theta is not None and arch_F_norm is not None and len(arch_theta) > 0
    arch_assoc   = _assign_ref_dirs(arch_F_norm, ref_dirs) if have_archive else None

    def _best_in_niche(cand_theta: np.ndarray, cand_F: np.ndarray, rd: int):
        proj    = cand_F @ ref_unit[rd]
        d_perp2 = np.maximum((cand_F ** 2).sum(axis=1) - proj ** 2, 0.0)
        best    = d_perp2.argmin()
        return cand_theta[best], d_perp2[best]

    guides_theta = np.tile(global_fb, (N, 1))   # default: global fallback

    for rd in np.unique(assoc):
        front_idx = pareto_idx[assoc[pareto_idx] == rd]
        front_theta, front_dist = (
            _best_in_niche(qpop_theta[front_idx], F_norm[front_idx], rd)
            if len(front_idx) > 0 else (None, None)
        )

        arch_theta_best, arch_dist = (None, None)
        if have_archive:
            in_niche = np.where(arch_assoc == rd)[0]
            if len(in_niche) > 0:
                arch_theta_best, arch_dist = _best_in_niche(
                    arch_theta[in_niche], arch_F_norm[in_niche], rd
                )

        if front_theta is None and arch_theta_best is None:
            continue   # keep the global fallback already in guides_theta
        if front_theta is None:
            best_theta = arch_theta_best
        elif arch_theta_best is None:
            best_theta = front_theta
        else:
            best_theta = front_theta if front_dist <= arch_dist else arch_theta_best

        guides_theta[assoc == rd] = best_theta   # broadcast to whole niche at once

    return guides_theta
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest QINSGA3/test_algorithm.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Update the module docstring's step list**

In `QINSGA3/algorithm.py`, in the module docstring (top of file), replace:

```
  7. Guide selection: best Pareto member in same niche; archive fills empty niches
```

with:

```
  7. Guide selection: elitist per niche — best of (current Pareto front,
     archive), archive normalised in the same ideal/nadir frame as the
     population [Han & Kim 2002 elitism principle]
```

and replace the "Performance" bullet list entries:

```
  - _crowding_distance, _select_guides, _supplement_from_archive, and
    _archive_update are fully vectorised with NumPy broadcasting — no Python
    inner loops over population or archive members.
  - arch_F_norm is computed once per generation and shared by both
    _supplement_from_archive and _migrate, removing a redundant _normalise_F call.
```

with:

```
  - _crowding_distance, _select_guides, and _archive_update are fully
    vectorised with NumPy broadcasting — no Python inner loops over
    population or archive members.
  - arch_F_norm is computed once per generation (projected into the
    population's ideal/nadir frame via _normalise_stats) and shared by both
    _select_guides and _migrate.
```

- [ ] **Step 6: Commit**

```bash
git add QINSGA3/algorithm.py QINSGA3/test_algorithm.py
git commit -m "feat: elitist per-niche guide selection in QINSGA3

_select_guides now compares the current Pareto front's best-in-niche
against the archive's best-in-niche for every niche (not just empty
ones), keeping whichever is closer to the reference ray. The archive
is elitist by construction, so this stops a niche's guide from
regressing across generations. Merges _supplement_from_archive's
behaviour into _select_guides (deleted as a separate function)."
```

---

### Task 3: Wire elitist guide selection into the production loop

**Files:**
- Modify: `QINSGA3/algorithm.py:442-491` (inside `run_qinsga3`)

**Interfaces:**
- Consumes: `_normalise_stats` (Task 1), `_select_guides` with
  `arch_theta`/`arch_F_norm` params (Task 2).
- Produces: no new public interface — `run_qinsga3`'s own signature and
  return type are unchanged.

- [ ] **Step 1: Replace the normalisation + guide-selection block**

In `QINSGA3/algorithm.py`, inside the `for gen in range(max_gen):` loop of
`run_qinsga3`, replace:

```python
            F_norm = _normalise_F(F_pen)
            assoc  = _assign_ref_dirs(F_norm, ref_dirs)

            guides_theta = _select_guides(assoc, pareto_idx, F_norm, ref_dirs, qpop.theta)

            # arch_F_norm computed once and shared by both _supplement_from_archive
            # and _migrate — avoids a redundant _normalise_F call per generation
            arch_theta_arr = None
            arch_F_norm    = None
            if len(arch_X) >= 4:
                arch_theta_arr = np.array(arch_theta)
                arch_F_norm    = _normalise_F(np.array(arch_F))
                pareto_assoc   = assoc[pareto_idx]
                guides_theta   = _supplement_from_archive(
                    guides_theta, assoc, pareto_assoc,
                    arch_theta_arr, arch_F_norm, ref_dirs,
                )
```

with:

```python
            ideal, denom = _normalise_stats(F_pen)
            F_norm = (F_pen - ideal) / denom
            assoc  = _assign_ref_dirs(F_norm, ref_dirs)

            # arch_F_norm projected into the SAME ideal/denom as F_norm (not its
            # own) so _select_guides can validly compare front vs archive
            # distances; also reused by _migrate below.
            arch_theta_arr = None
            arch_F_norm    = None
            if len(arch_X) >= 4:
                arch_theta_arr = np.array(arch_theta)
                arch_F_norm    = (np.array(arch_F) - ideal) / denom

            guides_theta = _select_guides(
                assoc, pareto_idx, F_norm, ref_dirs, qpop.theta,
                arch_theta=arch_theta_arr, arch_F_norm=arch_F_norm,
            )
```

- [ ] **Step 2: Confirm no leftover references**

Run: `python -c "import ast,sys; src=open('QINSGA3/algorithm.py').read(); ast.parse(src); print('_supplement_from_archive' in src)"`
Expected: `False`

- [ ] **Step 3: Smoke-test the module imports and parses cleanly**

Run: `python -c "import QINSGA3.algorithm"`
Expected: no output, exit code 0 (no `NameError`/`ImportError`)

- [ ] **Step 4: Commit**

```bash
git add QINSGA3/algorithm.py
git commit -m "refactor: wire elitist guide selection into run_qinsga3's loop"
```

---

### Task 4: Wire elitist guide selection into the benchmark loop

**Files:**
- Modify: `validation/algorithms/qinsga3/core.py`

**Interfaces:**
- Consumes: same as Task 3 — `_normalise_stats`, `_select_guides` with
  `arch_theta`/`arch_F_norm`.
- Produces: no change to `run_qinsga3_generic`'s public signature or return
  type.

- [ ] **Step 1: Update the import list**

In `validation/algorithms/qinsga3/core.py`, replace:

```python
from QINSGA3.algorithm import (
    _archive_update,
    _assign_ref_dirs,
    _crowding_trim,
    _migrate,
    _normalise_F,
    _penalised_F,
    _select_guides,
    _supplement_from_archive,
)
```

with:

```python
from QINSGA3.algorithm import (
    _archive_update,
    _assign_ref_dirs,
    _crowding_trim,
    _migrate,
    _normalise_stats,
    _penalised_F,
    _select_guides,
)
```

- [ ] **Step 2: Replace the normalisation + guide-selection block**

Replace:

```python
        F_norm = _normalise_F(F_pen)
        assoc = _assign_ref_dirs(F_norm, ref_dirs)

        guides_theta = _select_guides(assoc, pareto_idx, F_norm, ref_dirs, qpop.theta)

        arch_theta_arr = None
        arch_F_norm = None
        if len(arch_X) >= 4:
            arch_theta_arr = np.array(arch_theta)
            arch_F_norm = _normalise_F(np.array(arch_F))
            pareto_assoc = assoc[pareto_idx]
            guides_theta = _supplement_from_archive(
                guides_theta, assoc, pareto_assoc,
                arch_theta_arr, arch_F_norm, ref_dirs,
            )
```

with:

```python
        ideal, denom = _normalise_stats(F_pen)
        F_norm = (F_pen - ideal) / denom
        assoc = _assign_ref_dirs(F_norm, ref_dirs)

        arch_theta_arr = None
        arch_F_norm = None
        if len(arch_X) >= 4:
            arch_theta_arr = np.array(arch_theta)
            arch_F_norm = (np.array(arch_F) - ideal) / denom

        guides_theta = _select_guides(
            assoc, pareto_idx, F_norm, ref_dirs, qpop.theta,
            arch_theta=arch_theta_arr, arch_F_norm=arch_F_norm,
        )
```

- [ ] **Step 3: Update the module docstring's generation-order line**

Replace:

```
Mirrors QINSGA3/algorithm.py::run_qinsga3()'s algorithm exactly (same
generation order: measure -> evaluate -> penalise -> non-dominated sort ->
archive update -> normalise -> assign ref dirs -> select guides -> supplement
from archive -> rotate -> crossover -> mutate -> migrate), but:
```

with:

```
Mirrors QINSGA3/algorithm.py::run_qinsga3()'s algorithm exactly (same
generation order: measure -> evaluate -> penalise -> non-dominated sort ->
archive update -> normalise -> assign ref dirs -> select guides (elitist:
front + archive) -> rotate -> crossover -> mutate -> migrate), but:
```

- [ ] **Step 4: Run the existing benchmark-harness tests**

Run: `python -m pytest validation/algorithms/qinsga3/test_runner.py -v`
Expected: PASS (3 tests — these test output shape only, so they must still
pass unchanged)

- [ ] **Step 5: Commit**

```bash
git add validation/algorithms/qinsga3/core.py
git commit -m "refactor: wire elitist guide selection into the benchmark loop"
```

---

### Task 5: Regression smoke run on the benchmark CLI

**Files:**
- None modified — verification only.

- [ ] **Step 1: Run the full QINSGA3 unit test suite**

Run: `python -m pytest QINSGA3/test_algorithm.py validation/algorithms/qinsga3/test_runner.py -v`
Expected: PASS (9 tests total: 5 from Task 1+2, 3 from
`validation/algorithms/qinsga3/test_runner.py`, plus any pre-existing tests
already in those files)

- [ ] **Step 2: Run a 2-run smoke test on DTLZ via the CLI**

Run: `python -m validation.dtlz.main_validation --algorithm qinsga3 --runs 2 --n_obj 3`
Expected: completes without exceptions, prints a summary table with
non-NaN, non-empty IGD values for DTLZ1-7, writes CSVs under
`validation/dtlz/results/qinsga3/`.

- [ ] **Step 3: Confirm no working-tree changes from the smoke run**

Run: `git status --porcelain validation/dtlz/results`
Expected: shows the smoke-run CSVs as modified (2-run numbers, not the
30-run campaign ones) — do NOT commit these; they get overwritten by the
real campaign in Task 7. Discard them:

```bash
git checkout -- validation/dtlz/results
```

---

### Task 6: 5-run ablation on DTLZ1 + DTLZ3

**Files:**
- Create: `C:\Users\Mariem\AppData\Local\Temp\claude\c--Users-Mariem-OneDrive-Bureau-SolverModeling-Project-Irp\30c9548c-7d3f-453d-9305-4c8a98219a80\scratchpad\qinsga3_elitist_guide_ablation.py`
  (temporary, not committed — this session's scratchpad directory, not the
  repo; if executed in a different session, use that session's scratchpad
  directory instead and adjust the path below accordingly)

**Interfaces:**
- Consumes: `validation.algorithms.qinsga3.runner.run_experiment` (existing,
  unchanged signature: `(problem_name, problem, n_gen, n_runs) -> list[np.ndarray]`),
  `validation.dtlz.dtlz_problems.get_problem`, `validation.metrics.igd_metric.compute_igd`.

- [ ] **Step 1: Write the ablation script**

Create the file with:

```python
"""One-off ablation: elitist guide selection vs the currently-adopted
QINSGA3 config, on DTLZ1 and DTLZ3 (highest seed-to-seed variance / most
multimodal problems in the benchmark suite), 5 runs each, same 5 seeds
used by every prior QINSGA3 ablation (see DTLZ_results_summary_qinsga3.md).

This script does not change any algorithm code — it just runs the already
-elitist run_experiment (post Task 1-4) and compares the printed numbers
against the pre-elitism figures already recorded in
validation/dtlz/results/qinsga3/DTLZ_results_summary_qinsga3.md.

Usage: python qinsga3_elitist_guide_ablation.py
"""
import os
import sys

PROJECT_DIR = r"c:\Users\Mariem\OneDrive\Bureau\SolverModeling\Project_Irp"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from validation.algorithms.qinsga3.runner import run_experiment
from validation.algorithms.seeds import SEEDS
from validation.dtlz.dtlz_problems import get_problem
from validation.metrics.igd_metric import compute_igd, igd_statistics

N_RUNS = 5

for name in ("DTLZ1", "DTLZ3"):
    problem, n_gen = get_problem(name, n_obj=3)
    fronts = run_experiment(name, problem, n_gen, n_runs=N_RUNS)
    igd_values = [compute_igd(problem, front) for front in fronts]
    stats = igd_statistics(igd_values)
    print(f"\n{name} (M3, {N_RUNS} runs, elitist guide)")
    print(f"  seeds used: {SEEDS[:N_RUNS]}")
    print(f"  best={stats['best']:.6f}  median={stats['median']:.6f}  "
          f"worst={stats['worst']:.6f}  mean={stats['mean']:.6f}  "
          f"std={stats['std']:.6f}")

print("\nCompare against DTLZ_results_summary_qinsga3.md (current, "
      "pre-elitism, 30-run M3 numbers): DTLZ1 mean=0.228982, "
      "DTLZ3 mean=8.116732.")
```

- [ ] **Step 2: Run it**

Run: `python "C:\Users\Mariem\AppData\Local\Temp\claude\c--Users-Mariem-OneDrive-Bureau-SolverModeling-Project-Irp\30c9548c-7d3f-453d-9305-4c8a98219a80\scratchpad\qinsga3_elitist_guide_ablation.py"`

Expected: prints best/median/worst/mean/std for DTLZ1 and DTLZ3 with 5 runs.

- [ ] **Step 3: Compare and decide go/no-go**

Compare the printed `mean` (5-run) against the reference 30-run means quoted
in the script's final print (`DTLZ1: 0.228982`, `DTLZ3: 8.116732`) and,
qualitatively, the Best/Worst spread against the same rows in
`validation/dtlz/results/qinsga3/DTLZ_results_summary_qinsga3.md` (DTLZ1
worst=1.536838, DTLZ3 worst=24.535356). A 5-run sample will not match the
30-run mean exactly — look for a materially tighter Best/Worst spread and a
mean in the same ballpark or better, not an exact match.

- If neutral-or-better on both problems: proceed to Task 7 (full campaign).
- If clearly worse on either problem: stop, do not run Task 7 — report the
  numbers back and reassess the design (do not silently tweak the elitism
  comparison rule without discussing it first).

---

### Task 7: Full 30-run campaign (conditional on Task 6's go decision)

**Files:**
- Modify (regenerated, not hand-edited): `validation/dtlz/results/qinsga3/*.csv`,
  `validation/maf/results/qinsga3/*.csv`
- Modify (hand-edited): `validation/dtlz/results/qinsga3/DTLZ_results_summary_qinsga3.md`,
  `validation/maf/results/qinsga3/MaF_results_summary_qinsga3.md`

- [ ] **Step 1: Run the full DTLZ campaign, both objective counts**

Run (may take a while — run in background if your environment supports it):
```bash
python -m validation.dtlz.main_validation --algorithm qinsga3 --runs 30 --n_obj 3
python -m validation.dtlz.main_validation --algorithm qinsga3 --runs 30 --n_obj 4
```
Expected: completes, prints summary tables, overwrites CSVs under
`validation/dtlz/results/qinsga3/`.

- [ ] **Step 2: Run the full MaF campaign, both objective counts**

Run:
```bash
python -m validation.maf.main_validation --algorithm qinsga3 --runs 30 --n_obj 3
python -m validation.maf.main_validation --algorithm qinsga3 --runs 30 --n_obj 4
```

- [ ] **Step 3: Update the two summary reports**

Add a fifth column ("(5) + guide élitiste") to the "Comparaison avec
NSGA-III classique" evolution tables in both
`validation/dtlz/results/qinsga3/DTLZ_results_summary_qinsga3.md` and
`validation/maf/results/qinsga3/MaF_results_summary_qinsga3.md`, using the
new mean IGD values from Steps 1-2, following the exact same table format
and per-problem prose notes already used for the prior 4-step evolution
(bold the values that materially close the gap with NSGA-III or beat it, as
the existing rows already do). Also update the per-problem
Best/Median/Worst/Mean/Std tables and per-run tables earlier in each file to
the new numbers.

- [ ] **Step 4: Commit**

```bash
git add validation/dtlz/results/qinsga3 validation/maf/results/qinsga3
git commit -m "docs: QINSGA3 30-run campaign with elitist per-niche guide"
```

---

### Task 8: Smoke-test the real IRP production path

**Files:**
- None modified — verification only.

- [ ] **Step 1: Run QINSGA3 on a small IRP instance**

Run: `python -m QINSGA3.main --instance 5 --gen 20 --runs 1`

(small instance, 20 generations instead of the production default of 300 —
this is a smoke test for crashes/NaNs, not a quality benchmark, so a short
run is enough and keeps it fast)

Expected: completes without exceptions, produces a Pareto front output
(`qinsga3_chromosomes.json`/`qinsga3_report.html`, whichever `main.py`
currently writes) with a non-empty, non-NaN result — not a full
recalibration, just confirmation the shared-code change didn't break the
production path.

- [ ] **Step 2: Report the result**

Note the run completed cleanly (or any error) back to the user before
considering this plan done — do not silently proceed if this step errors.

---

### Task 9: Cleanup pass (scoped to touched files only)

**Files:**
- Review only: `QINSGA3/algorithm.py`, `QINSGA3/test_algorithm.py`,
  `validation/algorithms/qinsga3/core.py`

- [ ] **Step 1: Run the `/simplify` skill**

Invoke the `simplify` skill against the diff introduced by Tasks 1-4 (guide
elitism + normalisation split), scoped to `QINSGA3/algorithm.py`,
`QINSGA3/test_algorithm.py`, and `validation/algorithms/qinsga3/core.py`
only — per the user's explicit decision to scope cleanup to files this
feature touches, not a project-wide sweep.

- [ ] **Step 2: Review and commit any resulting cleanup**

If `/simplify` proposes changes, review them, then:

```bash
git add QINSGA3/algorithm.py QINSGA3/test_algorithm.py validation/algorithms/qinsga3/core.py
git commit -m "refactor: simplify guide-elitism diff (dead code / comment cleanup)"
```

If it proposes nothing, note that and skip the commit.
