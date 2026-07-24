# QINSGA3 per-niche diversity-preserving operator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port Tayarani-N & Akbarzadeh-T (2014, Evol. Intel. 7:219-239, §5)'s
Diversity Preserving operator to QINSGA3, so that when a niche's elitist
guide (already implemented, committed, not yet adopted) has been stuck on
the same solution for several generations, converged individuals clustered
around it are detected and reinitialised — breaking the premature-convergence
trap the elitist-guide ablation exposed on DTLZ3, without touching the
archive, the guide-selection logic, or any existing quantum operator.

**Architecture:** Two new pure helpers in `QINSGA3/algorithm.py`:
`_update_niche_stagnation` (tracks, per niche, how many consecutive
generations the guide has been unchanged — eq. 13) and
`_diversity_preserve_mask` (eq. 11 convergence + eq. 12 similarity,
restricted to stagnant niches, returns which individuals to reset). Both
generational loops (`QINSGA3/algorithm.py::run_qinsga3` and
`validation/algorithms/qinsga3/core.py::run_qinsga3_generic`) call them
right after `qpop.mutate(...)` and reset qualifying individuals' θ to π/4
(QINSGA3's own "maximum superposition" constant, the same value the paper's
eq. 14 reinitialises to). Three new parameters (`gamma_converge`,
`delta_similar`, `t_stagnation`) are threaded through both loops and
`validation/algorithms/qinsga3/runner.py`.

**Tech Stack:** Python, NumPy, pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-24-qinsga3-diversity-preserving-design.md`.
- **No change to the benchmark protocol**: DTLZ1-7/MaF1-7 problems, the 30
  shared seeds, Das-Dennis reference directions, population size, and the
  Cui et al. (2025) generation budget stay exactly as they are.
- **No change to the "quantum spirit"**: `rotate()`, `measure()`, and the
  existing crossover/mutation operators in `QINSGA3/chromosome.py` are
  untouched. This is a new, additive operator.
- **No change to the parameters shared with NSGA-III**: `P_CROSS=1.0`,
  `ETA_CROSS=20`, `p_mut=1/D` stay exactly as adopted.
- **No change to `_select_guides`, `_archive_update`, or the archive
  itself.** The new operator only ever writes to `qpop.theta` for
  individuals it targets — it never touches `arch_X`/`arch_F`/`arch_theta`.
- Every formula is ported from the paper's §5 (eq. 11-14) with only the
  adaptations the spec's "Formula mapping" table lists — no invented
  mechanics.

## File Structure

- Modify `QINSGA3/algorithm.py`: add `_update_niche_stagnation` and
  `_diversity_preserve_mask`; wire both into `run_qinsga3`'s loop and
  signature; update the module docstring's algorithm list.
- Modify `validation/algorithms/qinsga3/core.py`: same wiring into
  `run_qinsga3_generic`; update its import list and module docstring.
- Modify `validation/algorithms/qinsga3/runner.py`: add
  `GAMMA_CONVERGE`/`DELTA_SIMILAR`/`T_STAGNATION` constants, thread them
  through `run_single`'s call, document the rationale.
- Modify `QINSGA3/test_algorithm.py`: new tests for both helpers.
- No changes to `QINSGA3/chromosome.py`, `QINSGA3/main.py`, or any
  results/report files until the ablation (Task 7) and, conditionally, the
  full campaign (Task 8).

---

### Task 1: `_update_niche_stagnation` helper

**Files:**
- Modify: `QINSGA3/algorithm.py` (new function, added after `_migrate`,
  before the "Crowding distance and archive trimming" section)
- Test: `QINSGA3/test_algorithm.py`

**Interfaces:**
- Produces: `_update_niche_stagnation(assoc, guides_theta, history, counters, t_stagnation) -> (new_history, new_counters, stagnant_niches)`
  where `history: dict[int, np.ndarray]`, `counters: dict[int, int]`,
  `stagnant_niches: set[int]`.

- [ ] **Step 1: Write the failing tests**

Append to `QINSGA3/test_algorithm.py`:

```python
from QINSGA3.algorithm import _update_niche_stagnation


def test_update_niche_stagnation_triggers_after_t_generations():
    """A niche's guide unchanged for exactly t_stagnation consecutive calls
    ends up in the returned stagnant set; one call short does not."""
    assoc = np.array([0, 0])
    guide_theta = np.array([[0.5], [0.5]])
    history, counters = {}, {}
    t_stagnation = 3

    history, counters, stagnant = _update_niche_stagnation(
        assoc, guide_theta, history, counters, t_stagnation)
    assert counters[0] == 0
    assert 0 not in stagnant

    for _ in range(2):
        history, counters, stagnant = _update_niche_stagnation(
            assoc, guide_theta, history, counters, t_stagnation)
    assert counters[0] == 2
    assert 0 not in stagnant   # t_stagnation=3 not yet reached

    history, counters, stagnant = _update_niche_stagnation(
        assoc, guide_theta, history, counters, t_stagnation)
    assert counters[0] == 3
    assert 0 in stagnant


def test_update_niche_stagnation_resets_when_guide_changes():
    """The counter resets to 0 the moment a niche's guide theta changes."""
    assoc = np.array([0, 0])
    history, counters = {}, {}
    t_stagnation = 2

    guide_a = np.array([[0.5], [0.5]])
    guide_b = np.array([[0.7], [0.7]])

    history, counters, _ = _update_niche_stagnation(
        assoc, guide_a, history, counters, t_stagnation)
    history, counters, stagnant = _update_niche_stagnation(
        assoc, guide_a, history, counters, t_stagnation)
    assert counters[0] == 1
    assert 0 not in stagnant

    history, counters, stagnant = _update_niche_stagnation(
        assoc, guide_b, history, counters, t_stagnation)
    assert counters[0] == 0
    assert 0 not in stagnant
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest QINSGA3/test_algorithm.py -v`
Expected: FAIL — `ImportError: cannot import name '_update_niche_stagnation'`

- [ ] **Step 3: Add the helper**

In `QINSGA3/algorithm.py`, after `_migrate`'s closing line
(`qpop.theta = np.clip(qpop.theta, 0.0, np.pi / 2.0)`) and before the
`# Crowding distance and archive trimming (vectorised)` section comment,
insert:

```python
# ---------------------------------------------------------------------------
# Diversity preserving operator [Tayarani-N & Akbarzadeh-T 2014, Evol.
# Intel. 7:219-239, Section 5]
# ---------------------------------------------------------------------------

def _update_niche_stagnation(
    assoc:        np.ndarray,
    guides_theta: np.ndarray,
    history:      dict,
    counters:     dict,
    t_stagnation: int,
) -> tuple[dict, dict, set]:
    """Track how many consecutive generations each niche's guide has been
    unchanged [Tayarani-N & Akbarzadeh-T 2014, eq. 13: b_i^{t-T} = b_i^t].

    guides_theta is the (pop_size, n_genes) array _select_guides just
    returned this generation -- every member of a niche shares the same
    guide theta by construction (_select_guides broadcasts it), so
    comparing one representative row per niche to the stored history is
    enough. Returns (new_history, new_counters, stagnant_niches), where
    stagnant_niches is the set of niche ids whose guide has been unchanged
    for >= t_stagnation consecutive calls.
    """
    new_history  = dict(history)
    new_counters = dict(counters)
    stagnant: set = set()

    for rd in np.unique(assoc):
        rd = int(rd)
        rep_theta = guides_theta[np.where(assoc == rd)[0][0]]
        prev = new_history.get(rd)
        if prev is not None and np.array_equal(prev, rep_theta):
            new_counters[rd] = new_counters.get(rd, 0) + 1
        else:
            new_counters[rd] = 0
        new_history[rd] = rep_theta
        if new_counters[rd] >= t_stagnation:
            stagnant.add(rd)

    return new_history, new_counters, stagnant
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest QINSGA3/test_algorithm.py -v`
Expected: PASS (all tests, including the 2 new ones)

- [ ] **Step 5: Commit**

```bash
git add QINSGA3/algorithm.py QINSGA3/test_algorithm.py
git commit -m "feat: add per-niche stagnation tracking for QINSGA3

Ports eq. 13 of Tayarani-N & Akbarzadeh-T (2014)'s diversity-preserving
operator: tracks how many consecutive generations each niche's elitist
guide has stayed unchanged, the trigger condition for the diversity
preserving operator added in the next commit."
```

---

### Task 2: `_diversity_preserve_mask` helper

**Files:**
- Modify: `QINSGA3/algorithm.py` (new function, added directly after
  `_update_niche_stagnation`)
- Test: `QINSGA3/test_algorithm.py`

**Interfaces:**
- Consumes: nothing new — operates on plain NumPy arrays.
- Produces: `_diversity_preserve_mask(assoc, qpop_theta, F_norm, ref_dirs, stagnant_niches, gamma, delta) -> np.ndarray`
  (boolean mask, shape `(pop_size,)`).

- [ ] **Step 1: Write the failing tests**

Append to `QINSGA3/test_algorithm.py`:

```python
from QINSGA3.algorithm import _diversity_preserve_mask


def test_diversity_preserve_mask_resets_similar_converged_neighbour():
    """Within a stagnant niche: the champion (closest to the reference ray)
    is kept; a converged individual similar to it (eq. 12) is reset; a
    converged individual NOT similar to it is left alone; an unconverged
    individual (eq. 11) is left alone regardless of proximity."""
    assoc = np.array([0, 0, 0, 0])
    ref_dirs = np.array([[1.0, 0.0], [0.0, 1.0]])
    qpop_theta = np.array([
        [0.05],   # champion: converged, closest to ray0
        [0.06],   # converged AND similar to champion -> reset
        [1.52],   # converged but NOT similar to champion -> kept
        [0.78],   # NOT converged (near pi/4) -> kept regardless of distance
    ])
    F_norm = np.array([
        [0.9, 0.05],
        [0.5, 0.5],
        [0.5, 0.6],
        [0.0, 0.0],
    ])

    mask = _diversity_preserve_mask(
        assoc, qpop_theta, F_norm, ref_dirs, stagnant_niches={0},
        gamma=0.99, delta=0.1,
    )

    assert mask.tolist() == [False, True, False, False]


def test_diversity_preserve_mask_ignores_non_stagnant_niches():
    """A niche not in stagnant_niches is never touched, even if its
    individuals would otherwise satisfy the converged+similar criteria."""
    assoc = np.array([0, 0])
    ref_dirs = np.array([[1.0, 0.0], [0.0, 1.0]])
    qpop_theta = np.array([[0.05], [0.06]])
    F_norm = np.array([[0.9, 0.05], [0.5, 0.5]])

    mask = _diversity_preserve_mask(
        assoc, qpop_theta, F_norm, ref_dirs, stagnant_niches=set(),
        gamma=0.99, delta=0.1,
    )

    assert mask.tolist() == [False, False]


def test_diversity_preserve_mask_skips_niche_with_fewer_than_two_converged():
    """A stagnant niche with fewer than 2 converged individuals has nothing
    to compare, so nothing is reset."""
    assoc = np.array([0, 0])
    ref_dirs = np.array([[1.0, 0.0], [0.0, 1.0]])
    qpop_theta = np.array([[0.05], [0.78]])   # only one converged (0.78 is not)
    F_norm = np.array([[0.9, 0.05], [0.0, 0.0]])

    mask = _diversity_preserve_mask(
        assoc, qpop_theta, F_norm, ref_dirs, stagnant_niches={0},
        gamma=0.99, delta=0.1,
    )

    assert mask.tolist() == [False, False]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest QINSGA3/test_algorithm.py -v`
Expected: FAIL — `ImportError: cannot import name '_diversity_preserve_mask'`

- [ ] **Step 3: Add the helper**

In `QINSGA3/algorithm.py`, directly after `_update_niche_stagnation`'s
closing line, insert:

```python
def _diversity_preserve_mask(
    assoc:           np.ndarray,
    qpop_theta:      np.ndarray,
    F_norm:          np.ndarray,
    ref_dirs:        np.ndarray,
    stagnant_niches: set,
    gamma:           float,
    delta:           float,
) -> np.ndarray:
    """Diversity Preserving operator [Tayarani-N & Akbarzadeh-T 2014, eq.
    11-14], restricted to niches whose guide has been stagnant for
    t_stagnation generations (see _update_niche_stagnation).

    Convergence (eq. 11, exact port -- QINSGA3 already uses
    |alpha_ik|^2 = cos^2(theta_ik), and 1 - 2cos^2(theta) == -cos(2*theta)):
        (1/n_genes) * sum_k |cos(2*theta_ik)| > gamma

    Similarity (eq. 12, adapted from Hamming distance on observed bits to
    normalised distance on continuous theta):
        (1/n_genes) * sum_k |theta_ik - theta_jk| / (pi/2) < delta

    Within each stagnant niche, "best" (kept, eq. 14) is the converged
    individual closest to the niche's reference ray -- the same criterion
    _select_guides already uses to pick a niche's own representative.
    Similarity is checked against this best individual specifically
    (rather than fully general pairwise clustering): the champion IS the
    attractor this operator exists to help individuals escape from, so
    comparing everyone else in the niche against it directly targets the
    diagnosed failure mode.

    Returns a boolean mask of shape (pop_size,): True for individuals the
    caller should reset to pi/4 (eq. 14's reinitialisation value, which is
    QINSGA3's own "maximum superposition" constant -- see
    QuantumPopulation.__init__).
    """
    pop_size = len(assoc)
    reset    = np.zeros(pop_size, dtype=bool)

    if not stagnant_niches:
        return reset

    conv_score = np.abs(np.cos(2.0 * qpop_theta)).mean(axis=1)
    converged  = conv_score > gamma

    ref_norms = np.linalg.norm(ref_dirs, axis=1, keepdims=True)
    ref_unit  = ref_dirs / np.where(ref_norms > 1e-9, ref_norms, 1.0)

    for rd in stagnant_niches:
        niche_idx = np.where((assoc == rd) & converged)[0]
        if len(niche_idx) < 2:
            continue

        proj    = F_norm[niche_idx] @ ref_unit[rd]
        d_perp2 = np.maximum((F_norm[niche_idx] ** 2).sum(axis=1) - proj ** 2, 0.0)
        best_local = niche_idx[d_perp2.argmin()]

        best_theta   = qpop_theta[best_local]
        dist         = np.abs(qpop_theta[niche_idx] - best_theta).mean(axis=1) / (np.pi / 2.0)
        similar_mask = dist < delta

        losers = niche_idx[similar_mask & (niche_idx != best_local)]
        reset[losers] = True

    return reset
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest QINSGA3/test_algorithm.py -v`
Expected: PASS (all tests, including the 3 new ones — 10 total)

- [ ] **Step 5: Commit**

```bash
git add QINSGA3/algorithm.py QINSGA3/test_algorithm.py
git commit -m "feat: add diversity-preserving reset mask for stagnant QINSGA3 niches

Ports eq. 11 (convergence) and eq. 12 (similarity, adapted to continuous
theta) of Tayarani-N & Akbarzadeh-T (2014)'s diversity-preserving
operator: within a stagnant niche, keeps the individual closest to the
reference ray and flags converged individuals similar to it for reset."
```

---

### Task 3: Wire the operator into the production loop

**Files:**
- Modify: `QINSGA3/algorithm.py` (`run_qinsga3`'s signature, state init,
  and generational loop; module docstring)

**Interfaces:**
- Consumes: `_update_niche_stagnation`, `_diversity_preserve_mask` (Tasks 1-2).
- Produces: `run_qinsga3` gains three new optional parameters
  (`gamma_converge: float = 0.99`, `delta_similar: float = 0.1`,
  `t_stagnation: int = 5`) — existing callers that don't pass them keep
  working, now with the operator active at these defaults.

- [ ] **Step 1: Add the three new parameters to `run_qinsga3`'s signature**

In `QINSGA3/algorithm.py`, replace:

```python
    migration_period: int   = 10,
    n_migrate:        int   = 10,
    seed:             int   = 42,
    rotation_type:    str   = "tanh",
    callback          = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
```

with:

```python
    migration_period: int   = 10,
    n_migrate:        int   = 10,
    gamma_converge:   float = 0.99,
    delta_similar:    float = 0.1,
    t_stagnation:     int   = 5,
    seed:             int   = 42,
    rotation_type:    str   = "tanh",
    callback          = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
```

- [ ] **Step 2: Add the stagnation-tracking state**

Replace:

```python
    arch_X:    list[np.ndarray] = []
    arch_F:    list[np.ndarray] = []
    arch_theta: list[np.ndarray] = []
    _MAX_ARCHIVE = 500

    n_workers = min(os.cpu_count() or 1, pop_size)
```

with:

```python
    arch_X:    list[np.ndarray] = []
    arch_F:    list[np.ndarray] = []
    arch_theta: list[np.ndarray] = []
    _MAX_ARCHIVE = 500

    niche_guide_history: dict = {}
    niche_stagnation:    dict = {}

    n_workers = min(os.cpu_count() or 1, pop_size)
```

- [ ] **Step 3: Call the operator after mutation, before migration**

Replace:

```python
            qpop.mutate(p_mut, p_mut_strong, mut_sigma)

            if (arch_F_norm is not None
                    and migration_period > 0
                    and gen % migration_period == 0):
                _migrate(
                    qpop, arch_theta_arr, arch_F_norm,
                    assoc, ref_dirs, rng, n_migrate=n_migrate,
                )
```

with:

```python
            qpop.mutate(p_mut, p_mut_strong, mut_sigma)

            niche_guide_history, niche_stagnation, stagnant = _update_niche_stagnation(
                assoc, guides_theta, niche_guide_history, niche_stagnation, t_stagnation,
            )
            if t_stagnation > 0 and stagnant:
                reset_mask = _diversity_preserve_mask(
                    assoc, qpop.theta, F_norm, ref_dirs, stagnant,
                    gamma_converge, delta_similar,
                )
                qpop.theta[reset_mask] = np.pi / 4.0

            if (arch_F_norm is not None
                    and migration_period > 0
                    and gen % migration_period == 0):
                _migrate(
                    qpop, arch_theta_arr, arch_F_norm,
                    assoc, ref_dirs, rng, n_migrate=n_migrate,
                )
```

- [ ] **Step 4: Update the module docstring's algorithm list**

Replace:

```
  9. SBX crossover + quantum mutation
```

with:

```
  9. SBX crossover + quantum mutation
  10. Diversity preserving: in niches whose guide has been stagnant for
      t_stagnation generations, converged individuals similar to the
      niche's best are reinitialised to pi/4, breaking premature
      convergence [Tayarani-N & Akbarzadeh-T 2014, §5]
```

- [ ] **Step 5: Smoke-test the module still imports and parses cleanly**

Run: `python -c "import QINSGA3.algorithm"`
Expected: no output, exit code 0

- [ ] **Step 6: Commit**

```bash
git add QINSGA3/algorithm.py
git commit -m "feat: wire diversity-preserving operator into run_qinsga3's loop"
```

---

### Task 4: Wire the operator into the benchmark loop

**Files:**
- Modify: `validation/algorithms/qinsga3/core.py` (import list,
  `run_qinsga3_generic`'s signature, state init, and loop; module
  docstring)

**Interfaces:**
- Consumes: same as Task 3.
- Produces: `run_qinsga3_generic` gains three new required parameters
  (`gamma_converge: float`, `delta_similar: float`, `t_stagnation: int`,
  no defaults — matching how `migration_period`/`n_migrate` are already
  required here even though `run_qinsga3` gives them defaults).

- [ ] **Step 1: Update the import list**

In `validation/algorithms/qinsga3/core.py`, replace:

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

with:

```python
from QINSGA3.algorithm import (
    _archive_update,
    _assign_ref_dirs,
    _crowding_trim,
    _diversity_preserve_mask,
    _migrate,
    _normalise_stats,
    _penalised_F,
    _select_guides,
    _update_niche_stagnation,
)
```

- [ ] **Step 2: Add the three new parameters to `run_qinsga3_generic`'s signature**

Replace:

```python
    migration_period: int,
    n_migrate: int,
    seed: int,
    rotation_type: str = "tanh",
    noise_scale: float = 0.02,
) -> np.ndarray:
```

with:

```python
    migration_period: int,
    n_migrate: int,
    gamma_converge: float,
    delta_similar: float,
    t_stagnation: int,
    seed: int,
    rotation_type: str = "tanh",
    noise_scale: float = 0.02,
) -> np.ndarray:
```

- [ ] **Step 3: Add the stagnation-tracking state**

Replace:

```python
    arch_X: list[np.ndarray] = []
    arch_F: list[np.ndarray] = []
    arch_theta: list[np.ndarray] = []
    _MAX_ARCHIVE = 500

    def _eval_batch(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
```

with:

```python
    arch_X: list[np.ndarray] = []
    arch_F: list[np.ndarray] = []
    arch_theta: list[np.ndarray] = []
    _MAX_ARCHIVE = 500

    niche_guide_history: dict = {}
    niche_stagnation:    dict = {}

    def _eval_batch(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
```

- [ ] **Step 4: Call the operator after mutation, before migration**

Replace:

```python
        qpop.mutate(p_mut, p_mut_strong, mut_sigma)

        if (arch_F_norm is not None
                and migration_period > 0
                and gen % migration_period == 0):
            _migrate(
                qpop, arch_theta_arr, arch_F_norm,
                assoc, ref_dirs, rng, n_migrate=n_migrate,
            )
```

with:

```python
        qpop.mutate(p_mut, p_mut_strong, mut_sigma)

        niche_guide_history, niche_stagnation, stagnant = _update_niche_stagnation(
            assoc, guides_theta, niche_guide_history, niche_stagnation, t_stagnation,
        )
        if t_stagnation > 0 and stagnant:
            reset_mask = _diversity_preserve_mask(
                assoc, qpop.theta, F_norm, ref_dirs, stagnant,
                gamma_converge, delta_similar,
            )
            qpop.theta[reset_mask] = np.pi / 4.0

        if (arch_F_norm is not None
                and migration_period > 0
                and gen % migration_period == 0):
            _migrate(
                qpop, arch_theta_arr, arch_F_norm,
                assoc, ref_dirs, rng, n_migrate=n_migrate,
            )
```

- [ ] **Step 5: Update the module docstring's generation-order line**

Replace:

```
Mirrors QINSGA3/algorithm.py::run_qinsga3()'s algorithm exactly (same
generation order: measure -> evaluate -> penalise -> non-dominated sort ->
archive update -> normalise -> assign ref dirs -> select guides (elitist:
front + archive) -> rotate -> crossover -> mutate -> migrate), but:
```

with:

```
Mirrors QINSGA3/algorithm.py::run_qinsga3()'s algorithm exactly (same
generation order: measure -> evaluate -> penalise -> non-dominated sort ->
archive update -> normalise -> assign ref dirs -> select guides (elitist:
front + archive) -> rotate -> crossover -> mutate -> diversity preserve ->
migrate), but:
```

- [ ] **Step 6: This file has no dedicated unit tests of its own** — it is
  exercised via `validation/algorithms/qinsga3/test_runner.py` in Task 6.
  Do not add tests here; the pure logic is already tested in
  `QINSGA3/test_algorithm.py` (Tasks 1-2).

- [ ] **Step 7: Commit**

```bash
git add validation/algorithms/qinsga3/core.py
git commit -m "feat: wire diversity-preserving operator into the benchmark loop"
```

---

### Task 5: Add the new parameters to the benchmark runner

**Files:**
- Modify: `validation/algorithms/qinsga3/runner.py`

**Interfaces:**
- Consumes: `run_qinsga3_generic`'s new required parameters (Task 4).
- Produces: no change to `run_single`/`run_experiment`'s own signatures —
  the new constants are internal to this module, same as
  `MIGRATION_PERIOD`/`N_MIGRATE`/`NOISE_SCALE` already are.

- [ ] **Step 1: Add the new constants**

In `validation/algorithms/qinsga3/runner.py`, replace:

```python
ROTATION_TYPE = "tanh"
NOISE_SCALE = 0.0  # was 0.02 (QINSGA3/main.py's IRP default) — see module docstring
```

with:

```python
ROTATION_TYPE = "tanh"
NOISE_SCALE = 0.0  # was 0.02 (QINSGA3/main.py's IRP default) — see module docstring
GAMMA_CONVERGE = 0.99   # Tayarani-N & Akbarzadeh-T (2014) diversity-preserving operator — see module docstring
DELTA_SIMILAR = 0.1     # most problem-sensitive parameter per the paper's own Table 4 — primary ablation target
T_STAGNATION = 5        # generations a niche's guide must be unchanged before the operator applies
```

- [ ] **Step 2: Thread the new constants through `run_single`**

Replace:

```python
    return run_qinsga3_generic(
        problem, ref_dirs, pop_size, max_gen=n_gen,
        alpha_max=ALPHA_MAX, alpha_min=ALPHA_MIN,
        p_mut=p_mut, p_mut_strong=P_MUT_STRONG, mut_sigma=MUT_SIGMA,
        p_cross=P_CROSS, eta_cross=ETA_CROSS,
        migration_period=MIGRATION_PERIOD, n_migrate=N_MIGRATE,
        seed=seed, rotation_type=ROTATION_TYPE, noise_scale=NOISE_SCALE,
    )
```

with:

```python
    return run_qinsga3_generic(
        problem, ref_dirs, pop_size, max_gen=n_gen,
        alpha_max=ALPHA_MAX, alpha_min=ALPHA_MIN,
        p_mut=p_mut, p_mut_strong=P_MUT_STRONG, mut_sigma=MUT_SIGMA,
        p_cross=P_CROSS, eta_cross=ETA_CROSS,
        migration_period=MIGRATION_PERIOD, n_migrate=N_MIGRATE,
        gamma_converge=GAMMA_CONVERGE, delta_similar=DELTA_SIMILAR,
        t_stagnation=T_STAGNATION,
        seed=seed, rotation_type=ROTATION_TYPE, noise_scale=NOISE_SCALE,
    )
```

- [ ] **Step 3: Add a module-docstring paragraph documenting the addition**

At the end of the module docstring (after the `MIGRATION_PERIOD/N_MIGRATE`
paragraph, before the closing `"""`), add:

```
GAMMA_CONVERGE/DELTA_SIMILAR/T_STAGNATION are new parameters for the
per-niche diversity-preserving operator [Tayarani-N & Akbarzadeh-T 2014,
Evol. Intel. 7:219-239, §5], added to address a premature-convergence trap
found in a 5-run ablation of the elitist per-niche guide (see
docs/superpowers/specs/2026-07-24-qinsga3-elitist-guide-design.md and the
"Piste explorée et écartée" note in DTLZ_results_summary_qinsga3.md):
DTLZ3's mean/worst/std regressed 3-5x when the elitist guide alone was
tested. Starting values are the paper's own empirically-found good ranges
(Table 4), not yet validated for QINSGA3 — see
docs/superpowers/specs/2026-07-24-qinsga3-diversity-preserving-design.md
for the full formula derivation and the ablation methodology used to
validate (or reject) them before any adoption.
```

- [ ] **Step 4: Run the existing benchmark-harness tests**

Run: `python -m pytest validation/algorithms/qinsga3/test_runner.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add validation/algorithms/qinsga3/runner.py
git commit -m "feat: add diversity-preserving operator parameters to the QINSGA3 benchmark runner"
```

---

### Task 6: Regression smoke run

**Files:**
- None modified — verification only.

- [ ] **Step 1: Run the full QINSGA3 unit test suite**

Run: `python -m pytest QINSGA3/test_algorithm.py validation/algorithms/qinsga3/test_runner.py -v`
Expected: PASS (13 tests: 10 from `QINSGA3/test_algorithm.py` after Tasks
1-2, 3 from `validation/algorithms/qinsga3/test_runner.py`)

- [ ] **Step 2: Run a 2-run smoke test on DTLZ via the CLI**

Run: `python -m validation.dtlz.main_validation --algorithm qinsga3 --runs 2 --n_obj 3`
Expected: completes without exceptions, prints a summary table with
non-NaN, non-empty IGD values for DTLZ1-7.

- [ ] **Step 3: Discard the smoke-run CSVs**

Run: `git status --porcelain validation/dtlz/results` — expect the 2-run
CSVs listed as modified. Discard them (they'd otherwise overwrite the
official 30-run numbers with 2-run noise):

```bash
git checkout -- validation/dtlz/results
```

---

### Task 7: 5-run ablation on DTLZ1 + DTLZ3

**Files:**
- Create: `C:\Users\Mariem\AppData\Local\Temp\claude\c--Users-Mariem-OneDrive-Bureau-SolverModeling-Project-Irp\30c9548c-7d3f-453d-9305-4c8a98219a80\scratchpad\qinsga3_diversity_preserve_ablation.py`
  (temporary, not committed — this session's scratchpad directory, not the
  repo; if executed in a different session, use that session's scratchpad
  directory instead and adjust the path below accordingly)

**Interfaces:**
- Consumes: `validation.algorithms.qinsga3.runner.run_experiment`
  (unchanged signature), `validation.dtlz.dtlz_problems.get_problem`,
  `validation.metrics.igd_metric.compute_igd`/`igd_statistics`.

**Context on the comparison:** `_select_guides` no longer has an
"elitist off" mode — Tasks 1-4 of the earlier plan merged the archive
comparison into it unconditionally, so this ablation inherently measures
"elitist guide + diversity-preserving operator" together, not the new
operator in isolation. That is fine: the elitist-guide-alone numbers are
already recorded (in `DTLZ_results_summary_qinsga3.md`'s "Piste explorée
et écartée" note), so this ablation's own numbers, compared against BOTH
the original baseline and the elitist-alone figures, tell us the new
operator's marginal contribution just as clearly, without needing to
re-add a toggle for a configuration that is no longer the code's default
shape.

- [ ] **Step 1: Write the ablation script**

Create the file with:

```python
"""One-off ablation: elitist guide + diversity-preserving operator vs the
currently-adopted QINSGA3 config, on DTLZ1 and DTLZ3, 5 runs each, same 5
seeds used by every prior QINSGA3 ablation.

Compare against BOTH reference points already recorded:
  - original adopted baseline (DTLZ_results_summary_qinsga3.md):
      DTLZ1 mean=0.228982, worst=1.536838
      DTLZ3 mean=8.116732,  worst=24.535356
  - elitist-guide-alone 5-run ablation (no diversity-preserving operator):
      DTLZ1 mean=0.240365, worst=0.346592
      DTLZ3 mean=24.765429, worst=71.955994

Usage: python qinsga3_diversity_preserve_ablation.py
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
    print(f"\n{name} (M3, {N_RUNS} runs, elitist guide + diversity preserving)")
    print(f"  seeds used: {SEEDS[:N_RUNS]}")
    print(f"  best={stats['best']:.6f}  median={stats['median']:.6f}  "
          f"worst={stats['worst']:.6f}  mean={stats['mean']:.6f}  "
          f"std={stats['std']:.6f}")

print("\nCompare against baseline (DTLZ1 mean=0.228982/worst=1.536838, "
      "DTLZ3 mean=8.116732/worst=24.535356) and elitist-alone (DTLZ1 "
      "mean=0.240365/worst=0.346592, DTLZ3 mean=24.765429/worst=71.955994).")
```

- [ ] **Step 2: Run it**

Run: `python "C:\Users\Mariem\AppData\Local\Temp\claude\c--Users-Mariem-OneDrive-Bureau-SolverModeling-Project-Irp\30c9548c-7d3f-453d-9305-4c8a98219a80\scratchpad\qinsga3_diversity_preserve_ablation.py"`

Expected: prints best/median/worst/mean/std for DTLZ1 and DTLZ3 with 5 runs.

- [ ] **Step 3: Compare and decide go/no-go**

Success condition: DTLZ3's mean and worst move back toward (or below) the
**baseline** figures (8.116732 / 24.535356) relative to the elitist-alone
regression (24.765429 / 71.955994), **without** DTLZ1 losing the
elitist-alone improvement (mean ≈0.24, worst ≈0.35, i.e. staying well
below the baseline's worst of 1.536838).

- If both hold (DTLZ3 recovers, DTLZ1 stays improved): proceed to Task 8
  (full campaign).
- If DTLZ3 doesn't recover, or DTLZ1 regresses back toward/past the
  original baseline's spread: stop, do not run Task 8 — report the
  numbers back and reassess rather than silently tweaking
  `gamma_converge`/`delta_similar`/`t_stagnation` without discussing it
  first (same discipline as the elitist-guide plan's Task 6 gate).

---

### Task 8: Full 30-run campaign (conditional on Task 7's go decision)

**Files:**
- Modify (regenerated, not hand-edited): `validation/dtlz/results/qinsga3/*.csv`,
  `validation/maf/results/qinsga3/*.csv`
- Modify (hand-edited): `validation/dtlz/results/qinsga3/DTLZ_results_summary_qinsga3.md`,
  `validation/maf/results/qinsga3/MaF_results_summary_qinsga3.md`

- [ ] **Step 1: Run the full DTLZ campaign, both objective counts**

Run:
```bash
python -m validation.dtlz.main_validation --algorithm qinsga3 --runs 30 --n_obj 3
python -m validation.dtlz.main_validation --algorithm qinsga3 --runs 30 --n_obj 4
```

- [ ] **Step 2: Run the full MaF campaign, both objective counts**

Run:
```bash
python -m validation.maf.main_validation --algorithm qinsga3 --runs 30 --n_obj 3
python -m validation.maf.main_validation --algorithm qinsga3 --runs 30 --n_obj 4
```

- [ ] **Step 3: Update the two summary reports**

Add a new column/paragraph to both
`validation/dtlz/results/qinsga3/DTLZ_results_summary_qinsga3.md` and
`validation/maf/results/qinsga3/MaF_results_summary_qinsga3.md`, following
the exact format the "Piste explorée et écartée : guide élitiste par
niche" note already established (this note now gets a follow-up
paragraph documenting the diversity-preserving operator's own 30-run
outcome, replacing the "non retenu" conclusion if this campaign confirms
the fix). Update the per-problem Best/Median/Worst/Mean/Std tables and
per-run tables to the new numbers.

- [ ] **Step 4: Commit**

```bash
git add validation/dtlz/results/qinsga3 validation/maf/results/qinsga3
git commit -m "docs: QINSGA3 30-run campaign with diversity-preserving operator"
```

---

### Task 9: Smoke-test the real IRP production path

**Files:**
- None modified — verification only.

- [ ] **Step 1: Run QINSGA3 on a small IRP instance**

Run: `python -m QINSGA3.main --instance 5 --gen 20 --runs 1`

(small instance, 20 generations instead of the production default of
300 — a smoke test for crashes/NaNs, not a quality benchmark)

Expected: completes without exceptions, produces a Pareto front output
with a non-empty, non-NaN result.

- [ ] **Step 2: Report the result**

Note the run completed cleanly (or any error) back to the user before
considering this plan done.

---

### Task 10: Cleanup pass (scoped to touched files only)

**Files:**
- Review only: `QINSGA3/algorithm.py`, `QINSGA3/test_algorithm.py`,
  `validation/algorithms/qinsga3/core.py`,
  `validation/algorithms/qinsga3/runner.py`

- [ ] **Step 1: Run the `/simplify` skill**

Invoke the `simplify` skill against the diff introduced by Tasks 1-5,
scoped to the four files above only — per the same user decision made for
the elitist-guide plan (cleanup scoped to touched files, not a
project-wide sweep).

- [ ] **Step 2: Review and commit any resulting cleanup**

If `/simplify` proposes changes, review them, then:

```bash
git add QINSGA3/algorithm.py QINSGA3/test_algorithm.py validation/algorithms/qinsga3/core.py validation/algorithms/qinsga3/runner.py
git commit -m "refactor: simplify diversity-preserving operator diff (dead code / comment cleanup)"
```

If it proposes nothing, note that and skip the commit.
