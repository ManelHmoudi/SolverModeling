# MOEA/D Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add MOEA/D as a third algorithm for the many-objective IRP, compared against QI-NSGA-III the same way NSGA-III already is (full app integration, DTLZ/MaF benchmark runner, dedicated comparison script), reusing pymoo's own `MOEAD` and this project's existing decode/repair/report pipeline unchanged.

**Architecture:** New `Solvers/MOEAD/` module mirrors `Solvers/NSGA3/main.py`'s three entry points (`run_moead`, `render_from_instance`, `run_moead_report`) and reuses `Solvers/NSGA3/`'s `IRPProblem`/`report.py`/`report_builder.py` unmodified. Two small pymoo subclasses close gaps found in pymoo's stock `MOEAD` that block direct reuse: `ConstrainedMOEAD` (Deb's feasibility rule, since `IRPProblem` has hard constraints pymoo's `MOEAD` outright rejects) and `NormalizedTchebycheff` (objective-range normalization, since none of pymoo's decomposition variants have it and the IRP's 4 objectives have heterogeneous scales). `app.py` gets a third report card/routes mirroring the existing NSGA-III/QI-NSGA-III ones. A new benchmark runner and a new comparison script follow the same patterns as their NSGA-III/QI-NSGA-III siblings.

**Tech Stack:** Python, pymoo 0.6.1.6, numpy, scipy (stats), Flask, pytest.

**Spec:** `docs/superpowers/specs/2026-08-19-moead-integration-design.md`

## Global Constraints

- Reuse pymoo's own `MOEAD` structure unchanged except for the two subclasses above — no reimplementation of MOEA/D's core algorithm.
- No changes to `IRPProblem`, `Solvers/NSGA3/report.py`, `Solvers/NSGA3/decoder.py`, `Solvers/NSGA3/evaluator.py`, `Solvers/NSGA3/metrics.py`, or `Solvers/NSGA3/report_builder.py`.
- No changes to NSGA-III's or QI-NSGA-III's own algorithm code.
- No changes to `sensitivity/compare_2opt_fairness.py` (its pure statistical helpers are imported, not modified).
- MOEA/D's population size is `len(ref_dirs)` (165 for the IRP's 4-objective Das-Dennis N_PARTITIONS=8 scheme) — not independently settable, not rounded to match NSGA-III/QI-NSGA-III's 200. Do not add a `pop_size` parameter to `run_moead()`.
- Current production repair defaults (`use_two_opt=False`, `use_delivery_shift=True`) come for free from the shared `_evaluate_pareto`/`_repair_route_result` path — no MOEA/D-specific wiring needed for this.
- No new `Livrables_Prof/` HTML pages as part of this plan.

---

## Task 1: `NormalizedTchebycheff` decomposition class

**Files:**
- Create: `Solvers/MOEAD/__init__.py`
- Create: `Solvers/MOEAD/_normalized_decomposition.py`
- Test: `Solvers/MOEAD/test_main.py` (created here, extended by later tasks)

**Interfaces:**
- Produces: `Solvers.MOEAD._normalized_decomposition.NormalizedTchebycheff` — a `pymoo.core.decomposition.Decomposition` subclass with `update_nadir(F)` and the inherited `do(F, weights, ideal_point=..., ...)` / `_do(F, weights, **kwargs)`.

- [ ] **Step 1: Create the empty package marker**

Create `Solvers/MOEAD/__init__.py` with no content (matches `Solvers/NSGA3/__init__.py` / `Solvers/QINSGA3/__init__.py`, both empty).

- [ ] **Step 2: Write the failing tests**

Create `Solvers/MOEAD/test_main.py`:

```python
"""Tests for the MOEA/D IRP integration: end-to-end run_moead smoke tests
(mirroring Solvers/NSGA3/test_main.py / Solvers/QINSGA3/test_main.py), plus
unit tests for the two MOEA/D-specific helper classes this project adds on
top of pymoo's own MOEAD: NormalizedTchebycheff (objective-range
normalization, closing a gap in pymoo's own decomposition variants) and
ConstrainedMOEAD (Deb's feasibility rule, since IRPProblem has hard
constraints pymoo's own MOEAD rejects outright). See
docs/superpowers/specs/2026-08-19-moead-integration-design.md for the
full rationale behind both."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Solvers.MOEAD._normalized_decomposition import NormalizedTchebycheff

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TINY_INSTANCE = os.path.join(PROJECT_DIR, "data", "instance_3_clients.json")


# ── NormalizedTchebycheff ───────────────────────────────────────────────

def test_normalized_tchebycheff_matches_hand_computed_value():
    dec = NormalizedTchebycheff()
    F = np.array([[10.0, 100.0], [20.0, 50.0]])
    weights = np.array([[0.5, 0.5], [0.5, 0.5]])
    ideal = np.array([0.0, 0.0])

    result = dec.do(F, weights=weights, ideal_point=ideal)

    # update_nadir has not been called yet -> _nadir_running is None ->
    # _do falls back to span=ones(n_obj), i.e. no normalization this call.
    # F_norm = F (unnormalized). |F_norm| * weights = [[5,50],[10,25]].
    # max(axis=1) = [50, 25].
    np.testing.assert_allclose(result, [50.0, 25.0])


def test_normalized_tchebycheff_normalizes_after_update_nadir():
    dec = NormalizedTchebycheff()
    F = np.array([[10.0, 100.0], [20.0, 50.0]])
    weights = np.array([[0.5, 0.5], [0.5, 0.5]])
    ideal = np.array([0.0, 0.0])

    dec.update_nadir(F)
    result = dec.do(F, weights=weights, ideal_point=ideal)

    # nadir_running = batch max = [20, 100]. span = [20, 100].
    # F_norm = F / span = [[0.5, 1.0], [1.0, 0.5]].
    # |F_norm| * weights = [[0.25, 0.5], [0.5, 0.25]]. max(axis=1) = [0.5, 0.5].
    np.testing.assert_allclose(result, [0.5, 0.5])


def test_normalized_tchebycheff_nadir_estimate_is_monotonic():
    """The running nadir estimate only ever grows -- a later call with a
    smaller batch max must not shrink the normalization span."""
    dec = NormalizedTchebycheff()

    dec.update_nadir(np.array([[100.0, 100.0]]))
    assert dec._nadir_running.tolist() == [100.0, 100.0]

    dec.update_nadir(np.array([[10.0, 10.0]]))
    assert dec._nadir_running.tolist() == [100.0, 100.0]

    dec.update_nadir(np.array([[500.0, 1.0]]))
    assert dec._nadir_running.tolist() == [500.0, 100.0]
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest Solvers/MOEAD/test_main.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'Solvers.MOEAD._normalized_decomposition'`

- [ ] **Step 4: Implement `NormalizedTchebycheff`**

Create `Solvers/MOEAD/_normalized_decomposition.py`:

```python
"""Tchebycheff decomposition normalized by a running nadir estimate --
closes a gap in pymoo's own MOEAD found while integrating it into this
project (see docs/superpowers/specs/2026-08-19-moead-integration-design.md,
"Objective normalization in decomposition"): MOEAD._replace() only ever
passes ideal_point to the decomposition, never nadir_point, and none of
pymoo's built-in decomposition variants (Tchebicheff, PBI, ASF) normalize
by objective range -- all operate on raw F magnitudes after only
subtracting the ideal/utopian point. Left unfixed, the IRP's raw cost
scale (~15000-25000 DNT) would dominate every replacement decision
regardless of the reference-direction weights, collapsing MOEA/D's
decomposition to a cost-only comparison.

Mirrors how NSGA-III's own ReferenceDirectionSurvival tracks nadir_point
automatically across generations for the same normalization purpose, and
how QI-NSGA-III's _normalise_F (Solvers/QINSGA3/algorithm.py) reuses that
same tracked ideal/nadir pair -- MOEA/D has no equivalent built-in
mechanism, so this class provides one specific to it.

update_nadir() is a separate method from _do(), not folded into it:
ConstrainedMOEAD._replace() (see _constrained_moead.py) calls do() twice
per replacement step -- once for the neighborhood's F, once for the
offspring's F. Updating the running estimate inside _do() would let the
second call silently use a different, more-informed span than the first,
biasing the comparison between them. Callers must call update_nadir()
once, on the union of both F sets, before either do() call, so both use
the same frozen snapshot for that replacement step.
"""
import numpy as np
from pymoo.core.decomposition import Decomposition


class NormalizedTchebycheff(Decomposition):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._nadir_running = None

    def update_nadir(self, F):
        batch_max = np.asarray(F, dtype=float).max(axis=0)
        self._nadir_running = batch_max if self._nadir_running is None \
            else np.maximum(self._nadir_running, batch_max)

    def _do(self, F, weights, **kwargs):
        if self._nadir_running is None:
            span = np.ones(F.shape[1])
        else:
            span = np.maximum(self._nadir_running - self.utopian_point, 1e-9)
        F_norm = (F - self.utopian_point) / span
        return (np.abs(F_norm) * weights).max(axis=1)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest Solvers/MOEAD/test_main.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add Solvers/MOEAD/__init__.py Solvers/MOEAD/_normalized_decomposition.py Solvers/MOEAD/test_main.py
git commit -m "Add NormalizedTchebycheff decomposition for MOEA/D objective-range normalization"
```

---

## Task 2: `ConstrainedMOEAD` — Deb's feasibility rule

**Files:**
- Create: `Solvers/MOEAD/_constrained_moead.py`
- Modify: `Solvers/MOEAD/test_main.py`

**Interfaces:**
- Consumes: `Solvers.MOEAD._normalized_decomposition.NormalizedTchebycheff` (Task 1) — used in tests only, `ConstrainedMOEAD` itself accepts any `Decomposition`.
- Produces: `Solvers.MOEAD._constrained_moead.ConstrainedMOEAD` — `pymoo.algorithms.moo.moead.MOEAD` subclass, same constructor signature as the parent (`ref_dirs`, `n_neighbors=20`, `decomposition=None`, `prob_neighbor_mating=0.9`, `sampling`, `crossover`, `mutation`).

- [ ] **Step 1: Write the failing tests**

Append to `Solvers/MOEAD/test_main.py`:

```python
from pymoo.core.individual import Individual
from pymoo.core.population import Population

from Solvers.MOEAD._constrained_moead import ConstrainedMOEAD


# ── ConstrainedMOEAD._replace: Deb's feasibility rule ───────────────────

def _make_pop(F_list, CV_list):
    return Population.new(
        "F", np.array(F_list, dtype=float),
        "CV", np.array([[cv] for cv in CV_list], dtype=float),
    )


def _make_ind(F, CV):
    ind = Individual()
    ind.set("F", np.array(F, dtype=float))
    ind.set("CV", np.array([CV], dtype=float))
    return ind


def _bare_moead():
    """A ConstrainedMOEAD instance with just enough state for _replace --
    bypasses the full pymoo run loop (which needs a real Problem)."""
    algo = ConstrainedMOEAD(
        ref_dirs=np.array([[1.0, 0.0], [0.0, 1.0]]),
        decomposition=NormalizedTchebycheff(),
    )
    algo.neighbors = np.array([[0, 1]])
    algo.ideal = np.array([0.0, 0.0])
    return algo


def test_replace_feasible_offspring_beats_infeasible_neighbors():
    algo = _bare_moead()
    algo.pop = _make_pop([[5.0, 5.0], [5.0, 5.0]], [1.0, 1.0])  # both infeasible
    off = _make_ind([1.0, 1.0], 0.0)  # feasible

    algo._replace(0, off)

    np.testing.assert_allclose(algo.pop.get("F"), [[1.0, 1.0], [1.0, 1.0]])


def test_replace_feasible_vs_feasible_uses_decomposition():
    algo = _bare_moead()
    # Neighbor 0/1: F=[1,1] (good under the decomposition). Offspring:
    # F=[9,9] (worse). Both feasible -> decomposition value decides.
    algo.pop = _make_pop([[1.0, 1.0], [1.0, 1.0]], [0.0, 0.0])
    off = _make_ind([9.0, 9.0], 0.0)

    algo._replace(0, off)

    np.testing.assert_allclose(algo.pop.get("F"), [[1.0, 1.0], [1.0, 1.0]])


def test_replace_infeasible_vs_infeasible_smaller_violation_wins():
    algo = _bare_moead()
    algo.pop = _make_pop([[5.0, 5.0], [5.0, 5.0]], [2.0, 2.0])  # CV=2.0
    off = _make_ind([9.0, 9.0], 0.5)  # worse F but much smaller CV

    algo._replace(0, off)

    np.testing.assert_allclose(algo.pop.get("F"), [[9.0, 9.0], [9.0, 9.0]])


def test_replace_infeasible_offspring_never_beats_feasible_neighbor():
    algo = _bare_moead()
    algo.pop = _make_pop([[9.0, 9.0], [9.0, 9.0]], [0.0, 0.0])  # feasible
    off = _make_ind([1.0, 1.0], 0.5)  # better F but infeasible

    algo._replace(0, off)

    np.testing.assert_allclose(algo.pop.get("F"), [[9.0, 9.0], [9.0, 9.0]])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest Solvers/MOEAD/test_main.py -v -k replace`
Expected: FAIL with `ModuleNotFoundError: No module named 'Solvers.MOEAD._constrained_moead'`

- [ ] **Step 3: Implement `ConstrainedMOEAD`**

Create `Solvers/MOEAD/_constrained_moead.py`:

```python
"""ConstrainedMOEAD -- pymoo's own MOEAD, unmodified except for _replace(),
which applies Deb (2000)'s parameter-free feasibility rule instead of a
raw decomposition-value comparison. Needed because IRPProblem
(Solvers/NSGA3/problem.py) declares real hard inequality constraints
(n_ieq_constr = 2*|T| + |clients|*|T| + 2*|T|: tau_return window, delivery
deadlines, depot stock ceiling), and pymoo's own MOEAD._setup() contains
`assert not problem.has_constraints()` -- running vanilla MOEAD against
IRPProblem crashes immediately at setup. Everything else (neighbor
structure, weight vectors, _infill, crossover/mutation) is inherited from
pymoo's MOEAD unchanged. See
docs/superpowers/specs/2026-08-19-moead-integration-design.md,
"Constraint handling", for the full rationale, including why a static
penalty-weight reformulation was rejected (it would add a tuning
parameter with no counterpart on the NSGA-III/QI-NSGA-III side).

Deb's rule (no tuning parameter, matches the "feasibility-first"
principle NSGA-III/QI-NSGA-III already apply via pymoo's own
constraint-domination):
  - one feasible, one infeasible -> the feasible one always wins
  - both feasible                -> normal decomposition value decides
  - both infeasible               -> the smaller total constraint violation wins

Also used, unchanged, for the DTLZ/MaF benchmark runner
(Validation/Benchmarking/algorithms/moead/runner.py) even though those
problems are themselves unconstrained: when every individual is feasible
(CV <= 0 throughout), the rule below reduces exactly to `off_FV < FV`,
i.e. vanilla MOEAD's own comparison -- reusing one class in both places
avoids two parallel MOEA/D wirings.
"""
import numpy as np
from scipy.spatial.distance import cdist

from pymoo.algorithms.moo.moead import MOEAD, default_decomp
from pymoo.util.reference_direction import default_ref_dirs


class ConstrainedMOEAD(MOEAD):

    def _setup(self, problem, **kwargs):
        if self.ref_dirs is None:
            self.ref_dirs = default_ref_dirs(problem.n_obj)
        self.pop_size = len(self.ref_dirs)
        self.neighbors = np.argsort(
            cdist(self.ref_dirs, self.ref_dirs), axis=1, kind="quicksort"
        )[:, : self.n_neighbors]
        if self.decomposition is None:
            self.decomposition = default_decomp(problem)

    def _replace(self, k, off):
        pop = self.pop
        N = self.neighbors[k]

        neighbor_F = pop[N].get("F")
        # Freeze the normalization span for this replacement step -- see
        # NormalizedTchebycheff.update_nadir's own docstring for why this
        # must happen once, before either do() call below, not inside _do().
        if hasattr(self.decomposition, "update_nadir"):
            self.decomposition.update_nadir(np.vstack([neighbor_F, off.F[None, :]]))

        FV = self.decomposition.do(
            neighbor_F, weights=self.ref_dirs[N, :], ideal_point=self.ideal,
        )
        off_FV = self.decomposition.do(
            off.F[None, :], weights=self.ref_dirs[N, :], ideal_point=self.ideal,
        )

        CV = pop[N].get("CV")[:, 0]
        off_CV = float(off.CV[0])

        # Deb (2000) feasibility rule:
        #  - both feasible    -> decomposition value decides (unchanged MOEAD)
        #  - one feasible     -> the feasible one always wins
        #  - both infeasible  -> smaller total violation wins
        off_wins = np.where(
            off_CV <= 0,
            np.where(CV <= 0, off_FV < FV, True),
            np.where(CV <= 0, False, off_CV < CV),
        )
        I = np.where(off_wins)[0]
        pop[N[I]] = off
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest Solvers/MOEAD/test_main.py -v -k replace`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add Solvers/MOEAD/_constrained_moead.py Solvers/MOEAD/test_main.py
git commit -m "Add ConstrainedMOEAD: Deb's feasibility rule for MOEA/D on constrained problems"
```

---

## Task 3: `Solvers/MOEAD/main.py` — the three entry points

**Files:**
- Create: `Solvers/MOEAD/main.py`
- Modify: `Solvers/MOEAD/test_main.py`

**Interfaces:**
- Consumes: `Solvers.MOEAD._constrained_moead.ConstrainedMOEAD` (Task 2), `Solvers.MOEAD._normalized_decomposition.NormalizedTchebycheff` (Task 1), `Solvers.NSGA3.problem.IRPProblem`, `Solvers.NSGA3.report.write_report`/`generate_and_open`, `Solvers.NSGA3.report_builder._evaluate_pareto`/`_build_report_data`, `models.parametres.load_instance`.
- Produces: `Solvers.MOEAD.main.run_moead(data_path=None, n_gen=300, crossover_prob=0.9, mutation_prob=None, n_runs=1, repair_final_front=True) -> dict`, `Solvers.MOEAD.main.render_from_instance(data_path: str) -> dict`, `Solvers.MOEAD.main.run_moead_report(output_path=DEFAULT_REPORT_PATH, data_path=None, n_gen=300, crossover_prob=0.9, mutation_prob=None, n_runs=1, repair_final_front=True) -> str`, `Solvers.MOEAD.main.DEFAULT_REPORT_PATH` (module-level `str`).

- [ ] **Step 1: Write the failing tests**

Append to `Solvers/MOEAD/test_main.py`:

```python
from Solvers.MOEAD import main as moead_main


# ── run_moead end-to-end ─────────────────────────────────────────────────

def test_run_moead_repair_final_front_defaults_to_true(tmp_path, monkeypatch):
    monkeypatch.setattr(moead_main, "_CHROM_CACHE_PATH", str(tmp_path / "moead_chromosomes.json"))

    data = moead_main.run_moead(data_path=_TINY_INSTANCE, n_gen=2, n_runs=1)

    assert data["runs"][0]["meta"]["repair_final_front"] is True


def test_run_moead_repair_final_front_false_is_recorded(tmp_path, monkeypatch):
    cache_path = tmp_path / "moead_chromosomes.json"
    monkeypatch.setattr(moead_main, "_CHROM_CACHE_PATH", str(cache_path))

    data = moead_main.run_moead(
        data_path=_TINY_INSTANCE, n_gen=2, n_runs=1, repair_final_front=False,
    )

    assert data["runs"][0]["meta"]["repair_final_front"] is False
    with open(cache_path, encoding="utf-8") as f:
        cache = json.load(f)
    assert cache["runs"][0]["meta_base"]["repair_final_front"] is False


def test_render_from_instance_reapplies_stored_repair_flag(tmp_path, monkeypatch):
    cache_path = tmp_path / "moead_chromosomes.json"
    monkeypatch.setattr(moead_main, "_CHROM_CACHE_PATH", str(cache_path))

    moead_main.run_moead(
        data_path=_TINY_INSTANCE, n_gen=2, n_runs=1, repair_final_front=True,
    )
    refreshed = moead_main.render_from_instance(_TINY_INSTANCE)

    assert refreshed["runs"][0]["meta"]["repair_final_front"] is True


def test_run_moead_population_equals_ref_dirs_count_not_200(tmp_path, monkeypatch):
    """MOEA/D's population is len(ref_dirs), not independently settable --
    see the population-size discussion in docs/superpowers/specs/
    2026-08-19-moead-integration-design.md. N_PARTITIONS=8, n_obj=4 -> 165,
    regardless of instance size."""
    monkeypatch.setattr(moead_main, "_CHROM_CACHE_PATH", str(tmp_path / "moead_chromosomes.json"))

    data = moead_main.run_moead(data_path=_TINY_INSTANCE, n_gen=2, n_runs=1)

    assert data["meta"]["pop_size"] == 165


def test_run_moead_report_writes_html_file(tmp_path, monkeypatch):
    monkeypatch.setattr(moead_main, "_CHROM_CACHE_PATH", str(tmp_path / "moead_chromosomes.json"))
    out_path = str(tmp_path / "moead_report.html")

    path = moead_main.run_moead_report(output_path=out_path, data_path=_TINY_INSTANCE, n_gen=2, n_runs=1)

    assert os.path.exists(path)
    with open(path, encoding="utf-8") as f:
        html = f.read()
    assert "MOEA/D" in html
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest Solvers/MOEAD/test_main.py -v -k run_moead`
Expected: FAIL with `ModuleNotFoundError: No module named 'Solvers.MOEAD.main'`

- [ ] **Step 3: Implement `main.py`**

Create `Solvers/MOEAD/main.py`:

```python
"""MOEAD -- Many-Objective IRP solved with MOEA/D (pymoo), constraint-
handled via Deb's feasibility rule (see _constrained_moead.py) and
objective-range-normalized via a custom Tchebycheff decomposition (see
_normalized_decomposition.py) -- both needed because vanilla pymoo MOEAD
cannot run against IRPProblem unmodified. See
docs/superpowers/specs/2026-08-19-moead-integration-design.md for the
full rationale.

Usage (standalone):
    python -m Solvers.MOEAD.main                          # instance_25_clients.json
    python -m Solvers.MOEAD.main --instance 30
    python -m Solvers.MOEAD.main --instance 25 --gen 300 --runs 5

Called from app.py via run_moead_report().
Report format is identical to NSGA3 -- NSGA3/report.py is reused unchanged.
"""

import argparse
import json
import os
import random as _random
import sys
import time
import threading

import numpy as np

MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(MODULE_DIR))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from pymoo.util.ref_dirs           import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm   import PM
from pymoo.operators.sampling.rnd  import FloatRandomSampling
from pymoo.optimize                import minimize
from pymoo.termination             import get_termination

from models.parametres            import load_instance
from Solvers.NSGA3.problem        import IRPProblem
from Solvers.NSGA3.report         import write_report, generate_and_open
from Solvers.NSGA3.report_builder import _evaluate_pareto, _build_report_data
from ._constrained_moead          import ConstrainedMOEAD
from ._normalized_decomposition   import NormalizedTchebycheff

DEFAULT_REPORT_PATH = os.path.join(MODULE_DIR, "moead_report.html")
_CHROM_CACHE_PATH   = os.path.join(MODULE_DIR, "moead_chromosomes.json")

N_GEN          = 300  # matches NSGA3/QINSGA3.N_GEN -- equal search budget for a fair comparison
CROSSOVER_PROB = 0.9
MUTATION_PROB  = None  # pm = 1/D, computed dynamically once the instance is loaded
# N_PARTITIONS=8 -> 165 reference directions (das-dennis, M=4), shared with
# NSGA3/QINSGA3's own N_PARTITIONS. Unlike those two, MOEA/D's population
# size is NOT independently settable (pymoo's own MOEAD._setup(): one
# individual per decomposed subproblem) -- this fixes it at 165, not the
# 200 NSGA3/QINSGA3 use. See docs/superpowers/specs/
# 2026-08-19-moead-integration-design.md's population-size discussion for
# why 200 is not reachable for MOEA/D at any Das-Dennis partition count.
N_PARTITIONS = 8

SEEDS = [42, 137, 271, 491, 613, 733, 857, 977, 1009, 1123,
         1249, 1373, 1499, 1609, 1733, 1871, 1997, 2113, 2237, 2351]

_run_lock = threading.Lock()


def render_from_instance(data_path):
    """Re-evaluate cached MOEA/D chromosomes with the current instance JSON."""
    if not os.path.exists(_CHROM_CACHE_PATH):
        raise FileNotFoundError(
            "No cached MOEA/D chromosomes. Run the algorithm at least once first."
        )
    with open(_CHROM_CACHE_PATH, encoding="utf-8") as f:
        cache = json.load(f)

    sets_, params_ = load_instance(data_path)
    n_genes_expected = len(sets_["clients"]) * len(sets_["T"]) + len(sets_["clients"])

    runs_data = []
    for i, run_cache in enumerate(cache.get("runs", [])):
        pareto_X = np.array(run_cache["chromosomes"])
        if pareto_X.shape[1] != n_genes_expected:
            raise ValueError(
                f"Run {i+1}: cached chromosomes have {pareto_X.shape[1]} genes but the "
                f"current instance requires {n_genes_expected}. Re-run the algorithm."
            )
        repair = run_cache["meta_base"].get("repair_final_front", False)
        runs_data.append(_evaluate_pareto(pareto_X, sets_, params_, run_cache["meta_base"], repair=repair))

    return _build_report_data(runs_data)


def run_moead(data_path=None, n_gen=N_GEN,
              crossover_prob=CROSSOVER_PROB, mutation_prob=MUTATION_PROB,
              n_runs=1, repair_final_front: bool = True):
    """Run MOEA/D n_runs times with distinct seeds, cache all Pareto fronts,
    return report data.

    No pop_size parameter (unlike run_nsga3/run_qinsga3_solver): MOEA/D's
    population size is fixed at len(ref_dirs) by pymoo's own MOEAD._setup()
    (one individual per decomposed subproblem) -- it cannot be set
    independently. With the project's shared Das-Dennis N_PARTITIONS=8
    scheme (4 objectives), this is 165, not the 200 NSGA-III/QI-NSGA-III
    use -- see docs/superpowers/specs/2026-08-19-moead-integration-design.md.

    repair_final_front (default True, matches NSGA-III/QI-NSGA-III): see
    run_nsga3's own docstring -- identical meaning here, same shared
    _evaluate_pareto path (current production defaults use_two_opt=False,
    use_delivery_shift=True apply automatically).
    """
    n_runs = max(1, min(20, int(n_runs)))

    if data_path is None:
        data_path = os.path.join(PROJECT_DIR, "data", "instance_25_clients.json")

    sets_, params_ = load_instance(data_path)
    n_clients      = len(sets_["clients"])

    if mutation_prob is None:
        n_genes       = n_clients * len(sets_["T"]) + n_clients
        mutation_prob = 1.0 / n_genes
        print(f"[MOEAD] mutation_prob = 1/D = 1/{n_genes} = {mutation_prob:.6f}", flush=True)

    problem  = IRPProblem(sets_, params_)
    ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=N_PARTITIONS)
    effective_pop = len(ref_dirs)

    print(f"[MOEAD] {n_clients} clients | {len(sets_['T'])} periods | "
          f"{len(sets_['M'])} vehicles | {n_clients * len(sets_['T'])} genes | "
          f"pop={effective_pop} (=len(ref_dirs), not independently settable) "
          f"gen={n_gen} | runs={n_runs}", flush=True)

    if not _run_lock.acquire(blocking=False):
        raise RuntimeError(
            "[MOEAD] Another run is already in progress. "
            "Wait for it to finish before launching a new one."
        )

    raw_runs = []
    try:
        for run_idx in range(n_runs):
            seed = SEEDS[run_idx % len(SEEDS)]
            print(f"\n[MOEAD] === Run {run_idx + 1}/{n_runs}  seed={seed} ===", flush=True)

            np.random.seed(seed)
            _random.seed(seed)

            algorithm = ConstrainedMOEAD(
                ref_dirs       = ref_dirs,
                decomposition  = NormalizedTchebycheff(),
                sampling       = FloatRandomSampling(),
                crossover      = SBX(prob=crossover_prob, eta=20),
                # prob=1.0, prob_var=mutation_prob -- same per-gene-rate
                # convention as Solvers/NSGA3/main.py::run_nsga3 (see its
                # own long comment on prob vs prob_var).
                mutation       = PM(prob=1.0, prob_var=mutation_prob, eta=20),
            )

            t_start = time.time()
            result  = minimize(
                problem,
                algorithm,
                get_termination("n_gen", n_gen),
                verbose = True,
                seed    = seed,
            )
            elapsed = time.time() - t_start

            pareto_X = result.X

            if pareto_X is None or len(pareto_X) == 0:
                print(f"[MOEAD] Run {run_idx + 1}: no feasible solutions — skipping.", flush=True)
                continue

            print(f"[MOEAD] Run {run_idx + 1} done in {elapsed:.1f}s | "
                  f"Pareto front: {len(pareto_X)} solutions", flush=True)
            raw_runs.append({"seed": seed, "elapsed": elapsed, "pareto_X": pareto_X})
    finally:
        _run_lock.release()

    if not raw_runs:
        raise RuntimeError(
            "[MOEAD] No feasible solutions found in any run — all solutions violate "
            f"the hard constraint (tau_return > tau_max={params_['tau_max']}). "
            "Check tau_max in the instance JSON or increase gen."
        )

    base_meta = {
        "instance":           f"{n_clients}_clients",
        "pop_size":           effective_pop,
        "n_gen":              n_gen,
        "crossover_prob":     crossover_prob,
        "mutation_prob":      mutation_prob,
        "n_runs":             n_runs,
        "n_completed":        len(raw_runs),
        "algorithm":          "MOEAD",
        "repair_final_front": repair_final_front,
    }

    runs_data = []
    for i, raw in enumerate(raw_runs):
        per_run_meta = {
            **base_meta,
            "run_id":    i + 1,
            "seed":      raw["seed"],
            "elapsed_s": round(raw["elapsed"], 1),
        }
        runs_data.append(_evaluate_pareto(
            raw["pareto_X"], sets_, params_, per_run_meta, repair=repair_final_front,
        ))

    cache = {
        "algorithm": "MOEAD",
        "n_runs": len(raw_runs),
        "runs": [
            {
                "seed":        raw["seed"],
                "chromosomes": raw["pareto_X"].tolist(),
                "meta_base":   runs_data[i]["meta"],
            }
            for i, raw in enumerate(raw_runs)
        ],
    }
    with open(_CHROM_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)

    return _build_report_data(runs_data)


def run_moead_report(output_path=DEFAULT_REPORT_PATH, data_path=None,
                     n_gen=N_GEN, crossover_prob=CROSSOVER_PROB,
                     mutation_prob=MUTATION_PROB, n_runs=1,
                     repair_final_front: bool = True):
    data = run_moead(
        data_path, n_gen, crossover_prob, mutation_prob, n_runs,
        repair_final_front=repair_final_front,
    )
    return write_report(data, output_path, algo_label="MOEA/D")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MOEA/D solver for the many-objective IRP")
    parser.add_argument("--instance", default="25", choices=["3", "5", "15", "25", "30", "40", "100"])
    parser.add_argument("--gen",  type=int,   default=N_GEN)
    parser.add_argument("--cx",   type=float, default=CROSSOVER_PROB)
    parser.add_argument("--mut",  type=float, default=None,
                        help="Mutation probability (default: 1/D, D=number of decision variables)")
    parser.add_argument("--runs", type=int,   default=1)
    args = parser.parse_args()

    dp   = os.path.join(PROJECT_DIR, "data", f"instance_{args.instance}_clients.json")
    out  = os.path.join(MODULE_DIR,  f"moead_report_{args.instance}clients.html")
    data = run_moead(dp, args.gen, args.cx, args.mut, args.runs)
    path = write_report(data, out, algo_label="MOEA/D")
    print(f"[MOEAD] Report → {path}", flush=True)
    generate_and_open(path)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest Solvers/MOEAD/test_main.py -v`
Expected: PASS (all tests so far — 3 from Task 1, 4 from Task 2, 5 from this task)

- [ ] **Step 5: Commit**

```bash
git add Solvers/MOEAD/main.py Solvers/MOEAD/test_main.py
git commit -m "Add Solvers/MOEAD/main.py: run_moead/render_from_instance/run_moead_report"
```

---

## Task 4: `app.py` integration

**Files:**
- Modify: `app.py:19-21` (imports), `app.py:140` (`_BENCHMARK_ALGOS`), `app.py:425-429` (JS `ALGO_LABEL`/`ALGO_ORDER`), `app.py:1299-1340` (new card, mirroring the QI-NSGA-III card block), `app.py:1379-1428` (JS run-button/instance-switch wiring), `app.py:1581-1656` (new Flask routes, mirroring `/qinsga3/run` and `/qinsga3/report`).

**Interfaces:**
- Consumes: `Solvers.MOEAD.main.DEFAULT_REPORT_PATH`, `run_moead_report`, `render_from_instance` (Task 3).

- [ ] **Step 1: Add the import**

In `app.py`, after line 21 (`from Solvers.QINSGA3.main import ...`):

```python
from Solvers.MOEAD.main    import DEFAULT_REPORT_PATH as MOEAD_REPORT_PATH, run_moead_report, render_from_instance as moead_render_from_instance
```

- [ ] **Step 2: Register MOEA/D in the benchmark algorithm list**

Change line 140:

```python
_BENCHMARK_ALGOS = ["nsga3", "qinsga3", "moead"]
```

- [ ] **Step 3: Add the MOEA/D card to the menu template**

In the `MENU_TEMPLATE` string, immediately after the closing `</article>` of the "04 QI-NSGA-III" card (after line 1340) and before `</section>`, insert:

```html
    <!-- ── 05 MOEA/D ── -->
    <article class="card" style="--card-color:var(--c4);--card-icon-bg:var(--c4-bg)">
      <div class="card-body">
        <div class="card-top">
          <div class="card-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="var(--c4)" stroke-width="1.8" stroke-linecap="round">
              <circle cx="5" cy="6" r="2.2"/><circle cx="19" cy="6" r="2.2"/><circle cx="12" cy="18" r="2.2"/>
              <path d="M5 6L12 18M19 6L12 18M5 6L19 6"/>
            </svg>
          </div>
          <div class="card-badges">
            <span class="badge-num">05</span>
            <span class="badge-status active">Active</span>
          </div>
        </div>
        <h2 class="card-title">MOEA/D</h2>
        <p class="card-desc">Decomposition-based many-objective evolutionary algorithm. Splits the front into scalarized subproblems along Das-Dennis reference directions, evolved jointly via neighborhood replacement.</p>
        <div class="card-tags">
          <span class="card-tag">Decomposition</span>
          <span class="card-tag">Neighborhood replacement</span>
        </div>
        <div class="runs-row">
          <span class="runs-label">Runs</span>
          <div class="runs-group" id="moeadRunsBtns">
            <button class="runs-btn selected" data-runs="1"  onclick="setMoeadRuns(1)">1×</button>
            <button class="runs-btn"          data-runs="3"  onclick="setMoeadRuns(3)">3×</button>
            <button class="runs-btn"          data-runs="5"  onclick="setMoeadRuns(5)">5×</button>
            <button class="runs-btn"          data-runs="10" onclick="setMoeadRuns(10)">10×</button>
            <button class="runs-btn"          data-runs="20" onclick="setMoeadRuns(20)">20×</button>
          </div>
        </div>
      </div>
      <div class="card-footer">
        <a class="btn btn-primary" id="moead-run" href="{{ url_for('run_moead_route') }}?instance=25&runs=1" target="_blank" rel="noopener noreferrer">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
          Run
        </a>
        <a class="btn btn-secondary" id="moead-report" href="{{ url_for('moead_report') }}?instance=25" target="_blank" rel="noopener noreferrer">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
          Last report
        </a>
      </div>
    </article>
```

Note: reuses the `--c4`/`--c4-bg` CSS variables already used by the QI-NSGA-III card (no new color token needed — check the template's `:root` block already defines `--c4`/`--c4-bg`; if a 5th distinct card color is preferred, confirm with the user before adding a new token, since this plan does not add one).

- [ ] **Step 4: Wire the JS run-button state and instance-switch handling**

After the `setQi3Runs` function (around line 1397), add:

```javascript
let moeadRuns = 1;
function setMoeadRuns(n) {
  moeadRuns = n;
  document.querySelectorAll('#moeadRunsBtns .runs-btn').forEach(b => {
    b.classList.toggle('selected', parseInt(b.dataset.runs) === n);
  });
  const el = document.getElementById('moead-run');
  if (el) el.href = '{{ url_for("run_moead_route") }}?instance=' + selectedInstance + '&runs=' + n;
}
```

In the `selectInstance` function's `pairs` array (around line 1412-1419), add a new entry after `['qi3-report','{{ url_for("qinsga3_report") }}'],`:

```javascript
    ['moead-report', '{{ url_for("moead_report") }}'],
```

After the `if (qi3run) qi3run.href = ...` line (around line 1427), add:

```javascript
  const moeadrun = document.getElementById('moead-run');
  if (moeadrun) moeadrun.href = '{{ url_for("run_moead_route") }}?instance=' + key + '&runs=' + moeadRuns;
```

- [ ] **Step 5: Add the ALGO_LABEL/ALGO_ORDER entries for the benchmarking page**

Change lines 425-429:

```javascript
const ALGO_LABEL = {
  nsga3:   'NSGA-III (classique)',
  qinsga3: 'QI-NSGA-III (quantum-inspired)',
  moead:   'MOEA/D (décomposition)',
};
const ALGO_ORDER = ['nsga3', 'qinsga3', 'moead'];
```

- [ ] **Step 6: Add the Flask routes**

After the `qinsga3_report` route (after line 1656, before `def render_error`), add:

```python
@app.route("/moead/run")
def run_moead_route():
    data_path, inst_key = _resolve_instance()
    n_runs = max(1, min(20, int(request.args.get("runs", 1))))
    job_id = _new_job("MOEA/D")

    def _run():
        lock = _get_run_lock("moead", inst_key)
        if not lock.acquire(blocking=False):
            _job_error(job_id, f"A MOEA/D run for instance {inst_key} is already in progress -- wait for it to finish.")
            return
        try:
            run_moead_report(output_path=MOEAD_REPORT_PATH, data_path=data_path, n_runs=n_runs)
            _job_done(job_id, f"/moead/report?instance={inst_key}")
        except Exception:
            tb = traceback.format_exc()
            print(tb, flush=True)
            _job_error(job_id, tb)
        finally:
            lock.release()

    threading.Thread(target=_run, daemon=True).start()
    return _JOB_PAGE.format(algo="MOEA/D", job_id=job_id), 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/moead/report")
def moead_report():
    data_path, _ = _resolve_instance()
    try:
        try:
            data = moead_render_from_instance(data_path)
        except FileNotFoundError:
            run_moead_report(output_path=MOEAD_REPORT_PATH, data_path=data_path)
            data = moead_render_from_instance(data_path)
        return nsga3_render_html(data, algo_label="MOEA/D"), 200, {"Content-Type": "text/html; charset=utf-8"}
    except Exception:
        return render_error(traceback.format_exc()), 500
```

- [ ] **Step 7: Manual smoke test**

Run: `python app.py`, open the menu in a browser, click "Run" on the new MOEA/D card with the smallest instance, wait for the job page to redirect, confirm the report renders (Pareto chart, route network, etc.) with "MOEA/D" in the title. Then click "Last report" to confirm the cached-report path also works. Stop the server (Ctrl+C).

- [ ] **Step 8: Commit**

```bash
git add app.py
git commit -m "Add MOEA/D card, routes, and benchmarking-page entry to app.py"
```

---

## Task 5: DTLZ/MaF benchmark runner

**Files:**
- Create: `Validation/Benchmarking/algorithms/moead/__init__.py`
- Create: `Validation/Benchmarking/algorithms/moead/runner.py`
- Create: `Validation/Benchmarking/algorithms/moead/test_runner.py`

**Interfaces:**
- Consumes: `Solvers.MOEAD._constrained_moead.ConstrainedMOEAD`, `Solvers.MOEAD._normalized_decomposition.NormalizedTchebycheff` (Tasks 1-2), `Validation.Benchmarking.algorithms.nsga3.runner._N_OBJ_TO_P`, `Validation.Benchmarking.algorithms.seeds.SEEDS`, `Validation.Benchmarking.dtlz.dtlz_problems.get_problem`.
- Produces: `Validation.Benchmarking.algorithms.moead.runner.get_run_config(n_obj: int) -> (ref_dirs, pop_size)`, `run_single(problem, n_gen: int, seed: int) -> np.ndarray`, `run_experiment(problem_name: str, problem, n_gen: int, n_runs: int = 30) -> list`. Same signatures as the `nsga3`/`qinsga3` sibling runners, so `Validation/Benchmarking/engine.py::validate()` can call it interchangeably via its existing `run_experiment_fn` parameter.

- [ ] **Step 1: Create the empty package marker**

Create `Validation/Benchmarking/algorithms/moead/__init__.py` with no content.

- [ ] **Step 2: Write the failing tests**

Create `Validation/Benchmarking/algorithms/moead/test_runner.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from Validation.Benchmarking.algorithms.moead.runner import (
    run_single, run_experiment, get_run_config, N_REF_DIRS, POP_SIZE,
)
from Validation.Benchmarking.dtlz.dtlz_problems import get_problem


def test_ref_dirs_count_4obj():
    """Das-Dennis p=7, M=4 -> H = C(10,7) = 120 (Cui et al. 2025) -- same
    ref_dirs as the NSGA-III/QI-NSGA-III benchmark runners."""
    assert N_REF_DIRS == 120, f"Expected 120, got {N_REF_DIRS}"


def test_pop_size_4obj_equals_ref_dirs_no_rounding():
    """Unlike NSGA-III's runner (rounds up to a multiple of 4), MOEA/D's
    population is exactly len(ref_dirs) -- pymoo's own MOEAD forces this,
    it cannot be rounded independently."""
    assert POP_SIZE == 120, f"Expected 120, got {POP_SIZE}"


def test_pop_size_3obj_differs_from_nsga3_by_one():
    """M=3: Das-Dennis p=12 -> H=91. NSGA-III's own runner rounds this up
    to 92 (smallest multiple of 4 >= 91); MOEA/D cannot round, so its
    population here is 91, not 92 -- a structural one-individual gap,
    documented rather than hidden."""
    ref_dirs, pop_size = get_run_config(3)
    assert len(ref_dirs) == 91
    assert pop_size == 91


def test_run_single_4obj_output_shape():
    problem, _ = get_problem("DTLZ2", n_obj=4)
    front = run_single(problem, n_gen=3, seed=42)
    assert front.ndim == 2
    assert front.shape[1] == 4


def test_run_single_3obj_output_shape():
    """run_single must adapt ref_dirs and pop_size for n_obj=3."""
    problem, _ = get_problem("DTLZ2", n_obj=3)
    front = run_single(problem, n_gen=3, seed=42)
    assert front.ndim == 2
    assert front.shape[1] == 3, f"Expected 3 objectives, got {front.shape[1]}"


def test_run_experiment_3obj_returns_list():
    problem, _ = get_problem("DTLZ1", n_obj=3)
    fronts = run_experiment("DTLZ1", problem, n_gen=2, n_runs=2)
    assert len(fronts) == 2
    for f in fronts:
        assert f.shape[1] == 3


def test_n_obj_5_unsupported():
    """Cui et al. (2025) Table 2 only covers M=3 and M=4."""
    try:
        get_run_config(5)
        assert False, "expected ValueError for unsupported n_obj=5"
    except ValueError:
        pass


if __name__ == "__main__":
    test_ref_dirs_count_4obj()
    test_pop_size_4obj_equals_ref_dirs_no_rounding()
    test_pop_size_3obj_differs_from_nsga3_by_one()
    print("Testing run_single 4obj (3 gen)...")
    test_run_single_4obj_output_shape()
    print("Testing run_single 3obj (3 gen)...")
    test_run_single_3obj_output_shape()
    print("Testing run_experiment 3obj (2 gen, 2 runs)...")
    test_run_experiment_3obj_returns_list()
    print("Testing n_obj=5 raises...")
    test_n_obj_5_unsupported()
    print("All tests passed.")
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest Validation/Benchmarking/algorithms/moead/test_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'Validation.Benchmarking.algorithms.moead.runner'`

- [ ] **Step 4: Implement the runner**

Create `Validation/Benchmarking/algorithms/moead/runner.py`:

```python
"""
MOEA/D runner for Cui et al. (2025) DTLZ/MaF validation.

Same Das-Dennis partition table as the NSGA-III/QI-NSGA-III benchmark
runners (imported directly from nsga3.runner, single source of truth):
  n_obj=3 -> p=12 -> H=91
  n_obj=4 -> p=7  -> H=120

Unlike NSGA-III's runner, population size here is NOT rounded up to a
multiple of 4 -- pymoo's own MOEAD forces pop_size = len(ref_dirs)
exactly (one individual per decomposed subproblem), so this runner's
n_obj=3 population is 91, one less than NSGA-III's/QI-NSGA-III's 92. This
is a structural property of MOEA/D, not a tuning choice, and mirrors the
same 165-vs-200 population gap already documented on the IRP side (see
docs/superpowers/specs/2026-08-19-moead-integration-design.md).

Uses ConstrainedMOEAD + NormalizedTchebycheff (Solvers/MOEAD/) exactly as
the IRP side does, for a single MOEA/D wiring shared by both -- even
though DTLZ/MaF are themselves unconstrained, ConstrainedMOEAD's
feasibility rule reduces to vanilla MOEAD's own comparison whenever every
individual is feasible (see _constrained_moead.py's own docstring).

SBX/PM parameters (eta=20, pc=1.0, pm=1/n_var) match the NSGA-III/
QI-NSGA-III benchmark runners' own Cui et al. (2025) Table 2 settings.
"""
import numpy as np
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

from Validation.Benchmarking.algorithms.nsga3.runner import _N_OBJ_TO_P
from Validation.Benchmarking.algorithms.seeds import SEEDS as _SEEDS

from Solvers.MOEAD._constrained_moead import ConstrainedMOEAD
from Solvers.MOEAD._normalized_decomposition import NormalizedTchebycheff

# Module-level constants for n_obj=4 (backward compatibility and unit tests,
# same pattern as nsga3.runner's own N_REF_DIRS/POP_SIZE).
_ref_dirs_4 = get_reference_directions("das-dennis", 4, n_partitions=_N_OBJ_TO_P[4])
N_REF_DIRS = len(_ref_dirs_4)  # C(10,7) = 120
POP_SIZE = N_REF_DIRS  # MOEA/D: pop_size == len(ref_dirs), no rounding


def get_run_config(n_obj: int):
    """Return (ref_dirs, pop_size) for the given number of objectives.

    pop_size == len(ref_dirs) always -- MOEA/D cannot round this up
    independently, unlike NSGA-III's own get_run_config.
    """
    if n_obj not in _N_OBJ_TO_P:
        raise ValueError(
            f"n_obj={n_obj} not supported. Supported values: {sorted(_N_OBJ_TO_P)}"
        )
    p = _N_OBJ_TO_P[n_obj]
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=p)
    return ref_dirs, len(ref_dirs)


def run_single(problem, n_gen: int, seed: int) -> np.ndarray:
    """
    Run one MOEA/D optimisation and return the non-dominated objective values.

    Args:
        problem : pymoo Problem instance (n_obj determines ref_dirs and pop_size).
        n_gen   : number of generations.
        seed    : random seed for reproducibility.

    Returns:
        np.ndarray of shape (n_solutions, n_obj).
    """
    np.random.seed(seed)
    ref_dirs, _ = get_run_config(problem.n_obj)

    algorithm = ConstrainedMOEAD(
        ref_dirs=ref_dirs,
        decomposition=NormalizedTchebycheff(),
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=1.0, eta=20),
        mutation=PM(prob=1.0 / problem.n_var, eta=20),
    )

    result = minimize(
        problem,
        algorithm,
        get_termination("n_gen", n_gen),
        seed=seed,
        verbose=False,
    )

    F = result.F

    if F is None or len(F) == 0:
        raise RuntimeError(
            f"MOEA/D returned no solutions for {type(problem).__name__} "
            f"(n_obj={problem.n_obj}) with seed={seed}."
        )
    return np.asarray(F)


def run_experiment(problem_name: str, problem, n_gen: int, n_runs: int = 30) -> list:
    """
    Run MOEA/D n_runs times with distinct seeds (same seeds as the
    NSGA-III/QI-NSGA-III runners).

    Args:
        problem_name : display name (e.g. "DTLZ1"), used for progress printing.
        problem      : pymoo Problem instance.
        n_gen        : number of generations per run.
        n_runs       : number of independent runs (default 30).

    Returns:
        List of np.ndarray, one per run, shape (n_solutions, n_obj).
    """
    if n_runs > len(_SEEDS):
        raise ValueError(f"n_runs={n_runs} exceeds available seeds ({len(_SEEDS)})")

    fronts = []
    for i in range(n_runs):
        seed = _SEEDS[i]
        print(f"  [{problem_name}] Run {i+1:02d}/{n_runs}  seed={seed}", flush=True)
        front = run_single(problem, n_gen, seed)
        fronts.append(front)
        print(f"  [{problem_name}] Run {i+1:02d} done - front size: {len(front)}", flush=True)
    return fronts
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest Validation/Benchmarking/algorithms/moead/test_runner.py -v`
Expected: PASS (7 tests)

- [ ] **Step 6: Commit**

```bash
git add Validation/Benchmarking/algorithms/moead/
git commit -m "Add MOEA/D DTLZ/MaF benchmark runner"
```

---

## Task 6: `sensitivity/compare_qinsga3_vs_moead.py`

**Files:**
- Create: `sensitivity/compare_qinsga3_vs_moead.py`

**Interfaces:**
- Consumes: `sensitivity.compare_2opt_fairness._stats`, `_wilcoxon_rank_biserial`, `_vargha_delaney_a12`, `_bootstrap_median_diff_ci`, `_holm_bonferroni`, `_config_F` (all pure functions, imported not duplicated); `Solvers.QINSGA3.algorithm.run_qinsga3`; `Solvers.MOEAD._constrained_moead.ConstrainedMOEAD`; `Solvers.MOEAD._normalized_decomposition.NormalizedTchebycheff`; `Solvers.NSGA3.problem.IRPProblem`; `Solvers.NSGA3.metrics.compute_pareto_metrics`/`build_empirical_reference_front`; `models.parametres.load_instance`.
- Produces: `sensitivity.compare_qinsga3_vs_moead.run_comparison(instance: str, seeds: list[int], max_gen: int) -> dict` (keys: `"quality"`, `"holm"`), plus a `__main__` CLI.

This script has no fast unit-test target of its own (it runs real pymoo searches, like its siblings `compare_2opt_fairness.py`/`compare_qinsga3_vs_nsga3.py`, neither of which has a dedicated test file) — verification is the smoke run in Step 2 below, matching how those scripts were verified when first written.

- [ ] **Step 1: Implement the script**

Create `sensitivity/compare_qinsga3_vs_moead.py`:

```python
"""QI-NSGA-III vs MOEA/D -- statistical comparison on the real IRP.

Context: docs/superpowers/specs/2026-08-19-moead-integration-design.md.
Runs both algorithms' search fresh, once per seed, through the SAME
decode/repair/report pipeline (Solvers/NSGA3/report_builder.py::
_evaluate_pareto) every other algorithm comparison in this project uses.
Current production repair defaults (use_two_opt=False,
use_delivery_shift=True) are passed explicitly below -- _config_F's own
defaults (imported from compare_2opt_fairness.py) predate the production
default flip and are deliberately left unchanged there for that script's
own historical-campaign reproducibility, so they must be overridden here.

Statistics: this is an "engine effect" comparison (two independent search
engines, no shared chromosome array between them, unlike
sensitivity/compare_2opt_fairness.py's 2-opt-on/off pairing) -- Mann-Whitney
U (independent samples) is PRIMARY, matching
sensitivity/compare_qinsga3_vs_nsga3.py's own precedent for this same
comparison category. A same-seed-index paired Wilcoxon is also reported as
a SECONDARY, exploratory view -- do not read it as the headline result.
Brown-Forsythe (dispersion) and Holm-Bonferroni (multiple-comparison
correction across HV/GD/IGD/Spacing) apply to the primary Mann-Whitney
test. GD/IGD reference front: empirical leave-one-run-out (LORO), same as
every other comparison here (Solvers/NSGA3/metrics.py::
build_empirical_reference_front) -- each run scored against a front built
from every OTHER run only.

MOEA/D's population is fixed at len(ref_dirs) = 165 (Das-Dennis
N_PARTITIONS=8, 4 objectives) -- NOT independently settable, unlike
QI-NSGA-III's pop_size=200. This is a structural property of MOEA/D
(pymoo's own MOEAD._setup(): one individual per decomposed subproblem),
not a tuning choice made here.

Usage:
    python -m sensitivity.compare_qinsga3_vs_moead --gen 5 --seeds 42   # smoke
    python -m sensitivity.compare_qinsga3_vs_moead --seeds 42 137 271 --gen 300
"""
from __future__ import annotations

import argparse
import os
import random as _random
import sys
import time

MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(MODULE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm   import PM
from pymoo.operators.sampling.rnd  import FloatRandomSampling
from pymoo.optimize                import minimize
from pymoo.termination             import get_termination
from pymoo.util.ref_dirs           import get_reference_directions
from scipy.stats import levene, mannwhitneyu, wilcoxon

from models.parametres         import load_instance
from Solvers.NSGA3.problem     import IRPProblem
from Solvers.NSGA3.metrics     import compute_pareto_metrics, build_empirical_reference_front
from Solvers.QINSGA3.algorithm import run_qinsga3
from Solvers.MOEAD._constrained_moead        import ConstrainedMOEAD
from Solvers.MOEAD._normalized_decomposition import NormalizedTchebycheff

from sensitivity.compare_2opt_fairness import (
    _stats, _wilcoxon_rank_biserial, _vargha_delaney_a12,
    _bootstrap_median_diff_ci, _holm_bonferroni, _config_F,
)

N_PARTITIONS  = 8
N_OBJ         = 4
QINSGA3_POP   = 200
DEFAULT_SEEDS = [42, 137, 271]

_INDICATORS       = ["HV", "GD", "IGD", "Spacing"]
_HIGHER_IS_BETTER = {"HV": True, "GD": False, "IGD": False, "Spacing": False}

ALPHA_MAX = 0.10 * np.pi
ALPHA_MIN = 0.001 * np.pi


def _run_moead_once(problem, ref_dirs, max_gen, seed) -> np.ndarray:
    np.random.seed(seed)
    _random.seed(seed)
    algorithm = ConstrainedMOEAD(
        ref_dirs      = ref_dirs,
        decomposition = NormalizedTchebycheff(),
        sampling      = FloatRandomSampling(),
        crossover     = SBX(prob=0.9, eta=20),
        mutation      = PM(prob=1.0, prob_var=1.0 / problem.n_var, eta=20),
    )
    result = minimize(
        problem, algorithm, get_termination("n_gen", max_gen),
        seed=seed, verbose=True,
    )
    return result.X


def _run_qinsga3_once(sets_, params_, ref_dirs, max_gen, seed) -> np.ndarray:
    pareto_X, _, _ = run_qinsga3(
        sets_=sets_, params_=params_, ref_dirs=ref_dirs, pop_size=QINSGA3_POP,
        max_gen=max_gen, alpha_max=ALPHA_MAX, alpha_min=ALPHA_MIN,
        p_cross=0.9, eta_cross=20.0, p_mut=None, eta_mut=20.0,
        migration_period=10, n_migrate=10, seed=seed,
        rotation_type="tanh", repair_final_front=False,
    )
    return pareto_X


def run_comparison(instance: str, seeds: list[int], max_gen: int) -> dict:
    data_path = os.path.join(PROJECT_DIR, "data", f"instance_{instance}_clients.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Instance non trouvee : {data_path}")
    sets_, params_ = load_instance(data_path)
    n_clients = len(sets_["clients"])

    problem   = IRPProblem(sets_, params_)
    ref_dirs  = get_reference_directions("das-dennis", N_OBJ, n_partitions=N_PARTITIONS)
    moead_pop = len(ref_dirs)

    print("=" * 88)
    print("  QI-NSGA-III vs MOEA/D -- comparaison statistique (IRP)")
    print("=" * 88)
    print(f"  Instance        : {n_clients} clients")
    print(f"  Seeds           : {seeds}")
    print(f"  Generations     : {max_gen} (partagees, meme budget nominal)")
    print(f"  Pop QI-NSGA-III : {QINSGA3_POP}")
    print(f"  Pop MOEA/D      : {moead_pop} (= len(ref_dirs), non reglable independamment)")
    print("=" * 88)

    per_seed_F = {"QI-NSGA-III": {}, "MOEA/D": {}}
    for seed in seeds:
        print(f"\n--- seed={seed} ---")
        t0 = time.time()
        qi_X = _run_qinsga3_once(sets_, params_, ref_dirs, max_gen, seed)
        print(f"  QI-NSGA-III: {len(qi_X)} solutions en {time.time() - t0:.1f}s")

        t0 = time.time()
        mo_X = _run_moead_once(problem, ref_dirs, max_gen, seed)
        print(f"  MOEA/D     : {len(mo_X)} solutions en {time.time() - t0:.1f}s")

        # Shared repair pipeline, current production defaults.
        per_seed_F["QI-NSGA-III"][seed] = _config_F(
            qi_X, sets_, params_, repair=True, use_two_opt=False, use_delivery_shift=True,
        )
        per_seed_F["MOEA/D"][seed] = _config_F(
            mo_X, sets_, params_, repair=True, use_two_opt=False, use_delivery_shift=True,
        )

    all_F = np.vstack([F for algo in per_seed_F.values() for F in algo.values()])
    g_ideal, g_nadir = all_F.min(axis=0), all_F.max(axis=0)

    quality = {"QI-NSGA-III": {ind: [] for ind in _INDICATORS},
               "MOEA/D":      {ind: [] for ind in _INDICATORS}}
    for algo, by_seed in per_seed_F.items():
        for seed in seeds:
            F_run   = by_seed[seed]
            other_F = [by_seed[s] for s in seeds if s != seed]
            pf_ref  = build_empirical_reference_front(np.vstack(other_F)) if other_F else None
            q = compute_pareto_metrics(F_run, g_ideal, g_nadir, reference_front=pf_ref)
            for ind in _INDICATORS:
                quality[algo][ind].append(q[ind])

    print(f"\n{'-' * 88}")
    print("  RESULTATS")
    print(f"{'-' * 88}")
    pvalues_mw = {}
    for ind in _INDICATORS:
        arrow = "^" if _HIGHER_IS_BETTER[ind] else "v"
        qi_vals = quality["QI-NSGA-III"][ind]
        mo_vals = quality["MOEA/D"][ind]
        print(f"\n  {ind} ({arrow})")
        for algo in ("QI-NSGA-III", "MOEA/D"):
            s = _stats(quality[algo][ind])
            print(f"    {algo:<12} mean={s['mean']:.6f}  std={s['std']:.6f}")

        u_stat, p_mw = mannwhitneyu(qi_vals, mo_vals, alternative="two-sided")
        a12 = _vargha_delaney_a12(u_stat, len(qi_vals), len(mo_vals))
        pvalues_mw[ind] = p_mw
        print(f"    Mann-Whitney U = {u_stat:.1f}, p = {p_mw:.6f}  (A12={a12:.3f}) [PRIMAIRE]")

        if len(qi_vals) == len(mo_vals) and len(qi_vals) >= 2:
            try:
                w_stat, p_w = wilcoxon(qi_vals, mo_vals)
                r = _wilcoxon_rank_biserial(qi_vals, mo_vals)
                print(f"    Wilcoxon apparie (seed) = {w_stat:.1f}, p = {p_w:.6f}  "
                      f"(r={r:.3f}) [secondaire, exploratoire]")
            except ValueError as e:
                print(f"    Wilcoxon apparie (seed): {e}")

        if len(qi_vals) >= 2 and len(mo_vals) >= 2:
            lev_stat, p_lev = levene(qi_vals, mo_vals, center="median")
            print(f"    Brown-Forsythe (dispersion) p = {p_lev:.6f}")

    holm = _holm_bonferroni(pvalues_mw)
    print(f"\n{'-' * 88}")
    print("  Correction Holm-Bonferroni (sur le test Mann-Whitney primaire)")
    print(f"{'-' * 88}")
    for ind, (p, threshold, sig) in holm.items():
        tag = "significatif" if sig else "non significatif"
        print(f"    {ind:<10} p={p:.6f}  seuil={threshold:.6f}  -> {tag} (apres correction)")

    print(f"\n{'=' * 88}\n")
    return {"quality": quality, "holm": holm}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare QI-NSGA-III vs MOEA/D on the IRP (Mann-Whitney primary)"
    )
    parser.add_argument("--instance", default="100",
                        choices=["3", "5", "15", "25", "30", "40", "100", "150"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--gen", type=int, default=300)
    args = parser.parse_args()
    run_comparison(args.instance, args.seeds, args.gen)
```

- [ ] **Step 2: Smoke test**

Run: `python -m sensitivity.compare_qinsga3_vs_moead --instance 3 --gen 3 --seeds 42`
Expected: completes without error, prints per-seed solution counts for both algorithms, then the HV/GD/IGD/Spacing table with Mann-Whitney/Wilcoxon/Brown-Forsythe lines and the Holm-Bonferroni table. (A single seed makes the paired Wilcoxon section raise inside its own `try/except ValueError` and print a message instead of a p-value — expected at n=1, not a bug.)

- [ ] **Step 3: Commit**

```bash
git add sensitivity/compare_qinsga3_vs_moead.py
git commit -m "Add sensitivity/compare_qinsga3_vs_moead.py"
```

---

## Task 7: Full-suite verification

**Files:** None modified — verification only.

- [ ] **Step 1: Run the full project test suite**

Run: `python -m pytest -q`
Expected: every previously-passing test still passes, plus the new tests from Tasks 1-5 (12 in `Solvers/MOEAD/test_main.py`, 7 in `Validation/Benchmarking/algorithms/moead/test_runner.py`). No regressions.

- [ ] **Step 2: If any test fails, fix and re-run**

Diagnose against this plan's own file contents first (a mismatched import path or signature is the most likely cause, not a deeper design issue, since Tasks 1-6 were already checked against pymoo 0.6.1.6's actual source during plan-writing). Re-run Step 1 until green.

- [ ] **Step 3: Report final state**

Confirm to the user: test count before vs. after, and that no real IRP or benchmark campaign has been run yet (per the spec's Non-goals — that's a deliberate next step, not part of this plan).
