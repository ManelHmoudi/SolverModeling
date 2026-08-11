# QINSGA3 vs NSGA3 2-opt fairness campaign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the post-decode 2-opt repair (Solvers/QINSGA3/repair.py) applicable to NSGA-III on the same terms as QI-NSGA-III, fix the latent bug where QI-NSGA-III's own report/cache silently ignore the repair, and add a 4-configuration campaign script that isolates the evolutionary-engine effect from the 2-opt post-processing effect.

**Architecture:** One shared function (`Solvers/NSGA3/report_builder.py::_evaluate_pareto`) gains a `repair: bool` flag; both `Solvers/NSGA3/main.py::run_nsga3` and `Solvers/QINSGA3/main.py::run_qinsga3_solver`/`run_qinsga3_report` gain a `repair_final_front` parameter that threads down to it. A new campaign script runs each algorithm's search once per seed and decodes the resulting chromosome array twice (with/without repair) via that same shared function, guaranteeing identical initial fronts between a "without" and "with" pair.

**Tech Stack:** Python, numpy, pymoo (NSGA3, SBX, PM, minimize), scipy.stats (mannwhitneyu, wilcoxon), pytest.

## Global Constraints

- No change to `Solvers/QINSGA3/repair.py`'s 2-opt engine itself — reused exactly as-is (`_TWO_OPT_WINDOW=8`, `_MAX_REPAIR_ITER=20`, best-improvement acceptance).
- `run_nsga3`'s default behaviour is unchanged: `repair_final_front` defaults to `False`.
- `run_qinsga3_solver`/`run_qinsga3_report`'s default behaviour is unchanged: `repair_final_front` defaults to `True` (matches today's effective, if buggy, default).
- No new CLI flags on either solver's `__main__` block — `repair_final_front` stays a function parameter only, matching the existing convention (QINSGA3 doesn't expose it via CLI today either).
- All new/modified imports of `Solvers.QINSGA3.repair` stay function-local (lazy), not module-level — matches the existing convention in `Solvers/QINSGA3/algorithm.py::_evaluate_with_repair` and avoids relying on import-order details of `Solvers/QINSGA3/__init__.py`.
- Tests that call a full `run_nsga3`/`run_qinsga3_solver` must monkeypatch `_CHROM_CACHE_PATH` to a `tmp_path` — never let a test overwrite the real `nsga3_chromosomes.json` / `qinsga3_chromosomes.json` the live app reads.
- On this Windows environment, running a solver end-to-end via the Bash tool requires `PYTHONIOENCODING=utf-8` (the default `cp1252` console encoding cannot print the `α` character `Solvers/QINSGA3/main.py` logs) — set it when running any command below that exercises `run_qinsga3_solver`.

---

### Task 1: `_evaluate_pareto` gains a `repair` flag (shared by both algorithms)

**Files:**
- Modify: `Solvers/NSGA3/report_builder.py:41-56`
- Test: `Solvers/NSGA3/test_report_builder.py` (new)

**Interfaces:**
- Produces: `_evaluate_pareto(pareto_X, sets_, params_, meta_base, repair: bool = False) -> dict` — same return shape as before (`{"meta": {...}, "solutions": [...]}`); when `repair=True`, each solution's `route_result` is passed through `Solvers.QINSGA3.repair._repair_route_result` before objectives/routes/deliveries are computed. `repair=False` is byte-identical to the function's current behaviour.

- [ ] **Step 1: Write the failing tests**

Create `Solvers/NSGA3/test_report_builder.py`:

```python
"""Unit tests for report_builder._evaluate_pareto's repair flag -- confirms
NSGA3 and QINSGA3 share one code path for optionally applying the
post-decode 2-opt repair (Solvers/QINSGA3/repair.py) before objectives are
computed, so a repair_final_front flag means the same thing for either
algorithm."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.parametres import load_instance
from Solvers.NSGA3.decoder import decode_chromosome, build_routes
from Solvers.NSGA3.evaluator import compute_f1
from Solvers.NSGA3.problem import IRPProblem
from Solvers.NSGA3.report_builder import _evaluate_pareto
from Solvers.QINSGA3.repair import _repair_route_result

PROJECT_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TINY_INSTANCE  = os.path.join(PROJECT_DIR, "data", "instance_3_clients.json")
_SMALL_INSTANCE = os.path.join(PROJECT_DIR, "data", "instance_5_clients.json")


def _improvable_chromosome(sets_, params_):
    """A chromosome (instance_5_clients, rng seed 7's first draw) whose
    decoded route the 2-opt repair provably improves -- hand-confirmed
    (f1 604.96 -> 512.785) before writing this test, so repair=True vs.
    repair=False are guaranteed to differ, not just both trivially agree
    (real IRP routes on tiny instances are often already too short for any
    2-opt swap to apply -- see instance_3_clients's routes, all <=3 nodes)."""
    problem = IRPProblem(sets_, params_)
    rng = np.random.default_rng(7)
    return problem.xl + rng.random(len(problem.xl)) * (problem.xu - problem.xl)


def test_evaluate_pareto_repair_false_matches_plain_decode():
    sets_, params_ = load_instance(_TINY_INSTANCE)
    chromosome = IRPProblem(sets_, params_).xu

    result = _evaluate_pareto(np.array([chromosome]), sets_, params_, {}, repair=False)

    quantities, priorities = decode_chromosome(chromosome, sets_)
    route_result = build_routes(quantities, sets_, params_, priorities)
    expected_f1 = round(compute_f1(route_result, sets_, params_), 4)

    assert result["solutions"][0]["objectives"]["f1"] == expected_f1


def test_evaluate_pareto_repair_true_applies_two_opt():
    sets_, params_ = load_instance(_SMALL_INSTANCE)
    chromosome = _improvable_chromosome(sets_, params_)

    unrepaired = _evaluate_pareto(np.array([chromosome]), sets_, params_, {}, repair=False)
    repaired   = _evaluate_pareto(np.array([chromosome]), sets_, params_, {}, repair=True)

    assert repaired["solutions"][0]["objectives"]["f1"] < unrepaired["solutions"][0]["objectives"]["f1"]


def test_evaluate_pareto_repair_true_matches_manual_repair_call():
    sets_, params_ = load_instance(_SMALL_INSTANCE)
    chromosome = _improvable_chromosome(sets_, params_)

    result = _evaluate_pareto(np.array([chromosome]), sets_, params_, {}, repair=True)

    quantities, priorities = decode_chromosome(chromosome, sets_)
    route_result = build_routes(quantities, sets_, params_, priorities)
    repaired = _repair_route_result(route_result, sets_, params_)
    expected_f1 = round(compute_f1(repaired, sets_, params_), 4)

    assert result["solutions"][0]["objectives"]["f1"] == expected_f1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Solvers/NSGA3/test_report_builder.py -v`
Expected: the first test passes (repair=False already matches today's behaviour), but
`test_evaluate_pareto_repair_true_applies_two_opt` and
`test_evaluate_pareto_repair_true_matches_manual_repair_call` FAIL with
`TypeError: _evaluate_pareto() got an unexpected keyword argument 'repair'`.

- [ ] **Step 3: Implement the `repair` flag**

In `Solvers/NSGA3/report_builder.py`, replace lines 41-56:

```python
def _evaluate_pareto(pareto_X, sets_, params_, meta_base):
    """Evaluate Pareto chromosomes and return the run data dict.

    Called after a fresh solver run and on every report refresh
    (with potentially updated instance data).
    """
    solutions = []
    for i, chromosome in enumerate(pareto_X):
        quantities, priorities = decode_chromosome(chromosome, sets_)
        route_result = build_routes(quantities, sets_, params_, priorities)

        f1         = compute_f1(route_result, sets_, params_)
        f2         = compute_f2(route_result, sets_, params_)
        f3         = compute_f3(route_result, sets_, params_)
        f4, f4_sub = compute_f4_detail(route_result, sets_, params_)
```

with:

```python
def _evaluate_pareto(pareto_X, sets_, params_, meta_base, repair: bool = False):
    """Evaluate Pareto chromosomes and return the run data dict.

    Called after a fresh solver run and on every report refresh
    (with potentially updated instance data). When repair=True, each
    decoded route is passed through QINSGA3's post-decode 2-opt local
    search (Solvers/QINSGA3/repair.py, Baldwinian: the chromosome itself
    is unaffected, only the fitness/routes reported) before objectives
    are computed -- shared by both NSGA3 and QINSGA3's report pipelines so
    a repair_final_front flag means the same thing for either algorithm.
    """
    solutions = []
    for i, chromosome in enumerate(pareto_X):
        quantities, priorities = decode_chromosome(chromosome, sets_)
        route_result = build_routes(quantities, sets_, params_, priorities)
        if repair:
            from Solvers.QINSGA3.repair import _repair_route_result
            route_result = _repair_route_result(route_result, sets_, params_)

        f1         = compute_f1(route_result, sets_, params_)
        f2         = compute_f2(route_result, sets_, params_)
        f3         = compute_f3(route_result, sets_, params_)
        f4, f4_sub = compute_f4_detail(route_result, sets_, params_)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Solvers/NSGA3/test_report_builder.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Run the existing full test suite to check for regressions**

Run: `python -m pytest Solvers/NSGA3 Solvers/QINSGA3 -v`
Expected: all previously-passing tests still PASS (this change is additive — `repair` defaults to `False`, matching every existing call site's current behaviour).

- [ ] **Step 6: Commit**

```bash
git add Solvers/NSGA3/report_builder.py Solvers/NSGA3/test_report_builder.py
git commit -m "$(cat <<'EOF'
feat: add optional 2-opt repair to the shared _evaluate_pareto

report_builder._evaluate_pareto now accepts repair=True to route decoded
solutions through QINSGA3's post-decode 2-opt local search before scoring
them -- one shared implementation both NSGA3 and QINSGA3's report
pipelines can opt into, instead of duplicating the decode+repair+evaluate
glue in each. Defaults to False; no behaviour change for existing callers.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `run_nsga3` gains a `repair_final_front` parameter

**Files:**
- Modify: `Solvers/NSGA3/main.py`
- Test: `Solvers/NSGA3/test_main.py` (new)

**Interfaces:**
- Consumes: `_evaluate_pareto(pareto_X, sets_, params_, meta_base, repair: bool = False)` from Task 1.
- Produces: `run_nsga3(data_path=None, pop_size=POP_SIZE, n_gen=N_GEN, crossover_prob=CROSSOVER_PROB, mutation_prob=MUTATION_PROB, n_runs=1, repair_final_front: bool = False) -> dict`. `run_nsga3_report(..., repair_final_front: bool = False)` forwards it. The returned report dict's `data["runs"][i]["meta"]["repair_final_front"]` records which value was used; the cached JSON's `runs[i]["meta_base"]["repair_final_front"]` does too, and `render_from_instance` re-reads it from there on refresh.

- [ ] **Step 1: Write the failing tests**

Create `Solvers/NSGA3/test_main.py`:

```python
"""End-to-end tests for run_nsga3's repair_final_front parameter, on the
tiny 3-client instance with a minimal generation count -- fast enough to
run as a regular test, still exercising the real pymoo search loop."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Solvers.NSGA3 import main as nsga3_main

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TINY_INSTANCE = os.path.join(PROJECT_DIR, "data", "instance_3_clients.json")


def test_run_nsga3_repair_final_front_defaults_to_false(tmp_path, monkeypatch):
    monkeypatch.setattr(nsga3_main, "_CHROM_CACHE_PATH", str(tmp_path / "nsga3_chromosomes.json"))

    data = nsga3_main.run_nsga3(data_path=_TINY_INSTANCE, n_gen=2, n_runs=1)

    assert data["runs"][0]["meta"]["repair_final_front"] is False


def test_run_nsga3_repair_final_front_true_runs_and_is_recorded(tmp_path, monkeypatch):
    cache_path = tmp_path / "nsga3_chromosomes.json"
    monkeypatch.setattr(nsga3_main, "_CHROM_CACHE_PATH", str(cache_path))

    data = nsga3_main.run_nsga3(
        data_path=_TINY_INSTANCE, n_gen=2, n_runs=1, repair_final_front=True,
    )

    assert data["runs"][0]["meta"]["repair_final_front"] is True
    with open(cache_path, encoding="utf-8") as f:
        cache = json.load(f)
    assert cache["runs"][0]["meta_base"]["repair_final_front"] is True


def test_render_from_instance_reapplies_stored_repair_flag(tmp_path, monkeypatch):
    cache_path = tmp_path / "nsga3_chromosomes.json"
    monkeypatch.setattr(nsga3_main, "_CHROM_CACHE_PATH", str(cache_path))

    nsga3_main.run_nsga3(
        data_path=_TINY_INSTANCE, n_gen=2, n_runs=1, repair_final_front=True,
    )
    refreshed = nsga3_main.render_from_instance(_TINY_INSTANCE)

    assert refreshed["runs"][0]["meta"]["repair_final_front"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Solvers/NSGA3/test_main.py -v`
Expected: `test_run_nsga3_repair_final_front_defaults_to_false` FAILS with `KeyError: 'repair_final_front'` (the key doesn't exist in `meta` yet); the other two FAIL with `TypeError: run_nsga3() got an unexpected keyword argument 'repair_final_front'`.

- [ ] **Step 3: Implement `repair_final_front` on `run_nsga3`**

In `Solvers/NSGA3/main.py`, replace the `render_from_instance` function (lines 61-90):

```python
def render_from_instance(data_path):
    """Re-evaluate cached NSGA-III chromosomes with the current instance JSON.
    Supports both old single-run cache and new multi-run cache formats.
    """
    if not os.path.exists(_CHROM_CACHE_PATH):
        raise FileNotFoundError(
            "No cached chromosomes. Run the algorithm at least once first."
        )
    with open(_CHROM_CACHE_PATH, encoding="utf-8") as f:
        cache = json.load(f)

    sets_, params_ = load_instance(data_path)
    n_genes_expected = len(sets_["clients"]) * len(sets_["T"]) + len(sets_["clients"])

    if "runs" in cache:
        runs_cache = cache["runs"]
    else:
        runs_cache = [{"chromosomes": cache["chromosomes"], "meta_base": cache["meta_base"]}]

    runs_data = []
    for i, run_cache in enumerate(runs_cache):
        pareto_X = np.array(run_cache["chromosomes"])
        if pareto_X.shape[1] != n_genes_expected:
            raise ValueError(
                f"Run {i+1}: cached chromosomes have {pareto_X.shape[1]} genes but the "
                f"current instance requires {n_genes_expected}. Re-run the algorithm."
            )
        runs_data.append(_evaluate_pareto(pareto_X, sets_, params_, run_cache["meta_base"]))

    return _build_report_data(runs_data)
```

with:

```python
def render_from_instance(data_path):
    """Re-evaluate cached NSGA-III chromosomes with the current instance JSON.
    Supports both old single-run cache and new multi-run cache formats.
    """
    if not os.path.exists(_CHROM_CACHE_PATH):
        raise FileNotFoundError(
            "No cached chromosomes. Run the algorithm at least once first."
        )
    with open(_CHROM_CACHE_PATH, encoding="utf-8") as f:
        cache = json.load(f)

    sets_, params_ = load_instance(data_path)
    n_genes_expected = len(sets_["clients"]) * len(sets_["T"]) + len(sets_["clients"])

    if "runs" in cache:
        runs_cache = cache["runs"]
    else:
        runs_cache = [{"chromosomes": cache["chromosomes"], "meta_base": cache["meta_base"]}]

    runs_data = []
    for i, run_cache in enumerate(runs_cache):
        pareto_X = np.array(run_cache["chromosomes"])
        if pareto_X.shape[1] != n_genes_expected:
            raise ValueError(
                f"Run {i+1}: cached chromosomes have {pareto_X.shape[1]} genes but the "
                f"current instance requires {n_genes_expected}. Re-run the algorithm."
            )
        repair = run_cache["meta_base"].get("repair_final_front", False)
        runs_data.append(_evaluate_pareto(pareto_X, sets_, params_, run_cache["meta_base"], repair=repair))

    return _build_report_data(runs_data)
```

Then replace the `run_nsga3` signature and body (lines 93-210):

```python
def run_nsga3(data_path=None, pop_size=POP_SIZE, n_gen=N_GEN,
              crossover_prob=CROSSOVER_PROB, mutation_prob=MUTATION_PROB,
              n_runs=1):
    """Run NSGA-III n_runs times with distinct seeds, cache all Pareto fronts, return report data."""
```

with:

```python
def run_nsga3(data_path=None, pop_size=POP_SIZE, n_gen=N_GEN,
              crossover_prob=CROSSOVER_PROB, mutation_prob=MUTATION_PROB,
              n_runs=1, repair_final_front: bool = False):
    """Run NSGA-III n_runs times with distinct seeds, cache all Pareto fronts, return report data.

    repair_final_front (default False, unlike QINSGA3's True): when True,
    each returned run's Pareto front is decoded through the same
    post-decode 2-opt local search (Solvers/QINSGA3/repair.py) QI-NSGA-III
    uses for its own repair_final_front, via _evaluate_pareto(repair=True).
    Exists so a fair NSGA-III-vs-QI-NSGA-III comparison can apply the same
    post-processing to both sides -- see
    sensitivity/compare_2opt_fairness.py.
    """
```

Then, inside `run_nsga3`, replace the `base_meta` block:

```python
    base_meta = {
        "instance":       f"{n_clients}_clients",
        "pop_size":       effective_pop,
        "n_gen":          n_gen,
        "crossover_prob": crossover_prob,
        "mutation_prob":  mutation_prob,
        "n_runs":         n_runs,
        "n_completed":    len(raw_runs),
    }
```

with:

```python
    base_meta = {
        "instance":           f"{n_clients}_clients",
        "pop_size":           effective_pop,
        "n_gen":              n_gen,
        "crossover_prob":     crossover_prob,
        "mutation_prob":      mutation_prob,
        "n_runs":             n_runs,
        "n_completed":        len(raw_runs),
        "repair_final_front": repair_final_front,
    }
```

Then replace the per-run evaluation call:

```python
        runs_data.append(_evaluate_pareto(raw["pareto_X"], sets_, params_, per_run_meta))
```

with:

```python
        runs_data.append(_evaluate_pareto(
            raw["pareto_X"], sets_, params_, per_run_meta, repair=repair_final_front,
        ))
```

Finally, replace `run_nsga3_report`:

```python
def run_nsga3_report(output_path=DEFAULT_REPORT_PATH, data_path=None,
                     pop_size=POP_SIZE, n_gen=N_GEN,
                     crossover_prob=CROSSOVER_PROB, mutation_prob=MUTATION_PROB,
                     n_runs=1):
    data = run_nsga3(data_path, pop_size, n_gen, crossover_prob, mutation_prob, n_runs)
    return write_report(data, output_path)
```

with:

```python
def run_nsga3_report(output_path=DEFAULT_REPORT_PATH, data_path=None,
                     pop_size=POP_SIZE, n_gen=N_GEN,
                     crossover_prob=CROSSOVER_PROB, mutation_prob=MUTATION_PROB,
                     n_runs=1, repair_final_front: bool = False):
    data = run_nsga3(
        data_path, pop_size, n_gen, crossover_prob, mutation_prob, n_runs,
        repair_final_front=repair_final_front,
    )
    return write_report(data, output_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Solvers/NSGA3/test_main.py -v`
Expected: all 3 tests PASS. (Each test runs a real, tiny `run_nsga3` — allow up to ~10s per test.)

- [ ] **Step 5: Run the full NSGA3/QINSGA3 test suite to check for regressions**

Run: `python -m pytest Solvers/NSGA3 Solvers/QINSGA3 -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add Solvers/NSGA3/main.py Solvers/NSGA3/test_main.py
git commit -m "$(cat <<'EOF'
feat: expose repair_final_front on run_nsga3 (default False)

NSGA-III can now optionally have its final Pareto front pass through the
same post-decode 2-opt repair QI-NSGA-III already uses -- needed so the
two algorithms can be compared on equal post-processing terms. Default
behaviour and every existing caller are unchanged (defaults to False).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: fix `run_qinsga3_solver`'s latent bug (repair_final_front never reaches the report)

**Files:**
- Modify: `Solvers/QINSGA3/main.py`
- Test: `Solvers/QINSGA3/test_main.py` (new)

**Interfaces:**
- Consumes: `_evaluate_pareto(pareto_X, sets_, params_, meta_base, repair: bool = False)` from Task 1.
- Produces: `run_qinsga3_solver(..., repair_final_front: bool = True) -> dict` (new parameter; default preserves today's *intended* behaviour). `run_qinsga3_report(..., repair_final_front: bool = True)` forwards it. `data["runs"][i]["meta"]["repair_final_front"]` and the cached `meta_base["repair_final_front"]` record which value was used, and are correctly reflected in the computed objectives (fixing the bug). `render_from_instance` re-reads it from the cache, defaulting to `False` for pre-existing cache files that predate this field (so old caches keep showing the same numbers they always have, rather than silently changing on next refresh).

- [ ] **Step 1: Write the failing tests**

First, note the actual bug precisely: `run_qinsga3_solver`'s call to the low-level `run_qinsga3(...)` (around line 135) never passes `repair_final_front`, so it silently always uses `run_qinsga3`'s own default (`True`) -- there is currently no way to run the *solver* with repair disabled at all, and the resulting `pareto_F` (which IS correctly repaired) is discarded because `_evaluate_pareto(raw["pareto_X"], ...)` re-decodes from scratch without repair. This task fixes both: expose the parameter, thread it into the low-level call, AND make `_evaluate_pareto` apply it too.

Create `Solvers/QINSGA3/test_main.py`:

```python
"""End-to-end tests for run_qinsga3_solver's repair_final_front parameter --
covers the bug where the solver computed a correctly-repaired pareto_F
internally but then discarded it, re-decoding the report/cache from raw
chromosomes without ever applying the 2-opt repair. Uses the tiny 3-client
instance with a minimal generation count."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Solvers.QINSGA3 import main as qinsga3_main

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TINY_INSTANCE = os.path.join(PROJECT_DIR, "data", "instance_3_clients.json")


def test_run_qinsga3_solver_repair_final_front_defaults_to_true(tmp_path, monkeypatch):
    monkeypatch.setattr(qinsga3_main, "_CHROM_CACHE_PATH", str(tmp_path / "qinsga3_chromosomes.json"))

    data = qinsga3_main.run_qinsga3_solver(data_path=_TINY_INSTANCE, n_gen=2, n_runs=1)

    assert data["runs"][0]["meta"]["repair_final_front"] is True


def test_run_qinsga3_solver_repair_final_front_false_is_recorded(tmp_path, monkeypatch):
    cache_path = tmp_path / "qinsga3_chromosomes.json"
    monkeypatch.setattr(qinsga3_main, "_CHROM_CACHE_PATH", str(cache_path))

    data = qinsga3_main.run_qinsga3_solver(
        data_path=_TINY_INSTANCE, n_gen=2, n_runs=1, repair_final_front=False,
    )

    assert data["runs"][0]["meta"]["repair_final_front"] is False
    with open(cache_path, encoding="utf-8") as f:
        cache = json.load(f)
    assert cache["runs"][0]["meta_base"]["repair_final_front"] is False


def test_render_from_instance_reapplies_stored_repair_flag(tmp_path, monkeypatch):
    cache_path = tmp_path / "qinsga3_chromosomes.json"
    monkeypatch.setattr(qinsga3_main, "_CHROM_CACHE_PATH", str(cache_path))

    qinsga3_main.run_qinsga3_solver(
        data_path=_TINY_INSTANCE, n_gen=2, n_runs=1, repair_final_front=False,
    )
    refreshed = qinsga3_main.render_from_instance(_TINY_INSTANCE)

    assert refreshed["runs"][0]["meta"]["repair_final_front"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONIOENCODING=utf-8 python -m pytest Solvers/QINSGA3/test_main.py -v`
Expected: `test_run_qinsga3_solver_repair_final_front_defaults_to_true` FAILS with
`KeyError: 'repair_final_front'`; the other two FAIL with
`TypeError: run_qinsga3_solver() got an unexpected keyword argument 'repair_final_front'`.

- [ ] **Step 3: Implement the fix**

In `Solvers/QINSGA3/main.py`, replace the `run_qinsga3_solver` signature (lines 55-69):

```python
def run_qinsga3_solver(
    data_path:        str | None = None,
    pop_size:         int        = POP_SIZE,
    n_gen:            int        = N_GEN,
    alpha_max:        float      = ALPHA_MAX,
    alpha_min:        float      = ALPHA_MIN,
    p_cross:          float      = 0.9,
    eta_cross:        float      = 20.0,
    p_mut:            float | None = None,
    eta_mut:          float      = 20.0,
    migration_period: int        = 10,
    n_migrate:        int        = 10,
    n_runs:           int        = 1,
    rotation_type:    str        = "tanh",
) -> dict:
```

with:

```python
def run_qinsga3_solver(
    data_path:        str | None = None,
    pop_size:         int        = POP_SIZE,
    n_gen:            int        = N_GEN,
    alpha_max:        float      = ALPHA_MAX,
    alpha_min:        float      = ALPHA_MIN,
    p_cross:          float      = 0.9,
    eta_cross:        float      = 20.0,
    p_mut:            float | None = None,
    eta_mut:          float      = 20.0,
    migration_period: int        = 10,
    n_migrate:        int        = 10,
    n_runs:           int        = 1,
    rotation_type:    str        = "tanh",
    repair_final_front: bool     = True,
) -> dict:
```

In the same function's docstring (right after the `Args:` block, before `n_runs:`), add:

```python
        repair_final_front: Post-decode 2-opt repair on the returned Pareto
                             front only (Baldwinian -- see
                             Solvers/QINSGA3/repair.py). Default True.
```

Replace the `run_qinsga3(...)` call:

```python
            pareto_X, pareto_F, pareto_G = run_qinsga3(
                sets_             = sets_,
                params_           = params_,
                ref_dirs          = ref_dirs,
                pop_size          = effective_pop,
                max_gen           = n_gen,
                alpha_max         = alpha_max,
                alpha_min         = alpha_min,
                p_cross           = p_cross,
                eta_cross         = eta_cross,
                p_mut             = p_mut,
                eta_mut           = eta_mut,
                migration_period  = migration_period,
                n_migrate         = n_migrate,
                seed              = seed,
                rotation_type     = rotation_type,
                callback          = _progress,
            )
```

with:

```python
            pareto_X, pareto_F, pareto_G = run_qinsga3(
                sets_               = sets_,
                params_             = params_,
                ref_dirs            = ref_dirs,
                pop_size            = effective_pop,
                max_gen             = n_gen,
                alpha_max           = alpha_max,
                alpha_min           = alpha_min,
                p_cross             = p_cross,
                eta_cross           = eta_cross,
                p_mut               = p_mut,
                eta_mut             = eta_mut,
                migration_period    = migration_period,
                n_migrate           = n_migrate,
                seed                = seed,
                rotation_type       = rotation_type,
                callback            = _progress,
                repair_final_front  = repair_final_front,
            )
```

Replace `base_meta`:

```python
    base_meta = {
        "instance":       f"{n_clients}_clients",
        "pop_size":       effective_pop,
        "n_gen":          n_gen,
        "alpha_max":      round(alpha_max, 6),
        "alpha_min":      round(alpha_min, 6),
        "crossover_prob": round(p_cross, 6),
        "eta_cross":      round(float(eta_cross), 6),
        "mutation_prob":  round(p_mut, 6),
        "eta_mut":        round(float(eta_mut), 6),
        "n_runs":         n_runs,
        "n_completed":    len(raw_runs),
        "algorithm":      "QINSGA3",
    }
```

with:

```python
    base_meta = {
        "instance":           f"{n_clients}_clients",
        "pop_size":           effective_pop,
        "n_gen":              n_gen,
        "alpha_max":          round(alpha_max, 6),
        "alpha_min":          round(alpha_min, 6),
        "crossover_prob":     round(p_cross, 6),
        "eta_cross":          round(float(eta_cross), 6),
        "mutation_prob":      round(p_mut, 6),
        "eta_mut":            round(float(eta_mut), 6),
        "n_runs":             n_runs,
        "n_completed":        len(raw_runs),
        "algorithm":          "QINSGA3",
        "repair_final_front": repair_final_front,
    }
```

Replace the per-run evaluation call:

```python
        runs_data.append(_evaluate_pareto(raw["pareto_X"], sets_, params_, per_run_meta))
```

with:

```python
        runs_data.append(_evaluate_pareto(
            raw["pareto_X"], sets_, params_, per_run_meta, repair=repair_final_front,
        ))
```

Replace `render_from_instance`'s evaluation call:

```python
        runs_data.append(_evaluate_pareto(pareto_X, sets_, params_, run_cache["meta_base"]))
```

with:

```python
        repair = run_cache["meta_base"].get("repair_final_front", False)
        runs_data.append(_evaluate_pareto(pareto_X, sets_, params_, run_cache["meta_base"], repair=repair))
```

Replace `run_qinsga3_report`:

```python
def run_qinsga3_report(
    output_path:      str        = DEFAULT_REPORT_PATH,
    data_path:        str | None = None,
    pop_size:         int        = POP_SIZE,
    n_gen:            int        = N_GEN,
    alpha_max:        float      = ALPHA_MAX,
    alpha_min:        float      = ALPHA_MIN,
    p_cross:          float      = 0.9,
    eta_cross:        float      = 20.0,
    p_mut:            float | None = None,
    eta_mut:          float      = 20.0,
    migration_period: int        = 10,
    n_migrate:        int        = 10,
    n_runs:           int        = 1,
    rotation_type:    str        = "tanh",
) -> str:
    data = run_qinsga3_solver(
        data_path, pop_size, n_gen, alpha_max, alpha_min,
        p_cross, eta_cross, p_mut, eta_mut,
        migration_period=migration_period,
        n_migrate=n_migrate,
        n_runs=n_runs,
        rotation_type=rotation_type,
    )
    return write_report(data, output_path, algo_label="QI-NSGA-III")
```

with:

```python
def run_qinsga3_report(
    output_path:      str        = DEFAULT_REPORT_PATH,
    data_path:        str | None = None,
    pop_size:         int        = POP_SIZE,
    n_gen:            int        = N_GEN,
    alpha_max:        float      = ALPHA_MAX,
    alpha_min:        float      = ALPHA_MIN,
    p_cross:          float      = 0.9,
    eta_cross:        float      = 20.0,
    p_mut:            float | None = None,
    eta_mut:          float      = 20.0,
    migration_period: int        = 10,
    n_migrate:        int        = 10,
    n_runs:           int        = 1,
    rotation_type:    str        = "tanh",
    repair_final_front: bool     = True,
) -> str:
    data = run_qinsga3_solver(
        data_path, pop_size, n_gen, alpha_max, alpha_min,
        p_cross, eta_cross, p_mut, eta_mut,
        migration_period=migration_period,
        n_migrate=n_migrate,
        n_runs=n_runs,
        rotation_type=rotation_type,
        repair_final_front=repair_final_front,
    )
    return write_report(data, output_path, algo_label="QI-NSGA-III")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONIOENCODING=utf-8 python -m pytest Solvers/QINSGA3/test_main.py -v`
Expected: all 3 tests PASS. (Each test runs a real, tiny `run_qinsga3_solver`; allow up to ~15s per test.)

- [ ] **Step 5: Run the full NSGA3/QINSGA3 test suite to check for regressions**

Run: `PYTHONIOENCODING=utf-8 python -m pytest Solvers/NSGA3 Solvers/QINSGA3 -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add Solvers/QINSGA3/main.py Solvers/QINSGA3/test_main.py
git commit -m "$(cat <<'EOF'
fix: wire repair_final_front through to QINSGA3's report/cache

run_qinsga3_solver computed a correctly-repaired pareto_F internally but
discarded it, rebuilding the report/cache from raw chromosomes via
_evaluate_pareto without ever applying the 2-opt repair -- so
qinsga3_report.html and qinsga3_chromosomes.json never reflected
repair_final_front's effect despite it defaulting to True. Exposes
repair_final_front as an explicit, overridable parameter (default True,
preserving today's intended behaviour) and threads it into
_evaluate_pareto's new repair flag.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: caveat note on `compare_route_repair_final.py`'s NSGA-III comparison

**Files:**
- Modify: `sensitivity/compare_route_repair_final.py`

**Interfaces:**
- None (docstring/print text only, no behaviour change).

- [ ] **Step 1: Add the caveat to the module docstring**

In `sensitivity/compare_route_repair_final.py`, after the existing docstring's last paragraph (ending `"...no separate reimplementation needed."`, right before the `Usage:` section), insert:

```python
CAVEAT: the "vs NSGA-III (l'objectif final)" comparison this script prints
loads Solvers/NSGA3/nsga3_chromosomes.json's cache and decodes it with the
plain, unmodified decoder -- NSGA-III never receives the 2-opt repair here.
That comparison therefore conflates the evolutionary-engine effect with the
2-opt post-processing effect. See sensitivity/compare_2opt_fairness.py for
a 4-configuration campaign that isolates the two by applying (or not
applying) the same repair to both algorithms.
```

- [ ] **Step 2: Add the same caveat to the printed output**

Replace:

```python
    print(f"\n{'-'*92}")
    print(f"  Mann-Whitney U : {test_lbl} vs NSGA-III (l'objectif final)")
    print(f"{'-'*92}")
```

with:

```python
    print(f"\n{'-'*92}")
    print(f"  Mann-Whitney U : {test_lbl} vs NSGA-III (l'objectif final)")
    print("  NOTE: NSGA-III ci-dessous n'a PAS recu le 2-opt -- comparaison non")
    print("  equitable a elle seule (effet moteur + effet 2-opt confondus). Voir")
    print("  sensitivity/compare_2opt_fairness.py (4 configurations).")
    print(f"{'-'*92}")
```

- [ ] **Step 3: Verify the script still runs (smoke check, no assertion needed)**

Run: `python -c "import ast; ast.parse(open('sensitivity/compare_route_repair_final.py', encoding='utf-8').read())"`
Expected: no output (valid syntax).

- [ ] **Step 4: Commit**

```bash
git add sensitivity/compare_route_repair_final.py
git commit -m "$(cat <<'EOF'
docs: caveat compare_route_repair_final.py's NSGA-III comparison

Its NSGA-III reference is decoded without the 2-opt repair QI-NSGA-III's
side already has, conflating the engine effect with the post-processing
effect. Points at the new compare_2opt_fairness.py, which isolates them.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: 4-configuration fairness campaign script

**Files:**
- Create: `sensitivity/compare_2opt_fairness.py`

**Interfaces:**
- Consumes: `_evaluate_pareto(pareto_X, sets_, params_, meta_base, repair: bool = False)` from Task 1; `Solvers.QINSGA3.algorithm.run_qinsga3` (existing, unchanged); `Solvers.NSGA3.problem.IRPProblem`, `Solvers.NSGA3.metrics.compute_pareto_metrics` (existing, unchanged).
- Produces: a standalone script, no importable interface consumed elsewhere.

- [ ] **Step 1: Write the script**

Create `sensitivity/compare_2opt_fairness.py`:

```python
"""2-opt fairness campaign -- NSGA-III vs QI-NSGA-III, with/without the
post-decode 2-opt repair applied to BOTH algorithms.

Context: docs/superpowers/specs/2026-08-10-qinsga3-2opt-fairness-campaign-
design.md. QI-NSGA-III's 2-opt repair (Solvers/QINSGA3/repair.py, "Remede
G") has so far only ever been applied to QI-NSGA-III's own front, then
compared against a plain (unrepaired) NSGA-III baseline
(sensitivity/compare_route_repair_final.py) -- conflating the
evolutionary-engine effect (quantum rotation gate vs. plain NSGA-III) with
the post-processing effect (2-opt vs. none). This script isolates the two
by running each algorithm's search ONCE per seed, then decoding the SAME
resulting chromosome array twice -- once without repair, once with (the
repair is Baldwinian: it never touches the chromosome, only the fitness
reported for it, via Solvers/NSGA3/report_builder.py::_evaluate_pareto's
repair flag) -- producing four configurations from two searches per seed,
all sharing identical initial fronts, seeds, and budgets:

    1. NSGA-III      without 2-opt
    2. NSGA-III      with 2-opt
    3. QI-NSGA-III   without 2-opt
    4. QI-NSGA-III   with 2-opt

Statistics separate the two effects:
    - 2-opt effect   (paired Wilcoxon, same front, same seed): 1 vs 2 (NSGA-
      III), 3 vs 4 (QI-NSGA-III).
    - Engine effect  (Mann-Whitney, independent fronts): 1 vs 3 (raw gap, no
      post-processing on either side -- reproduces what
      compare_qinsga3_vs_nsga3.py already measures) and 2 vs 4 (fair gap,
      both sides get the same 2-opt polish -- the number that actually
      answers "is the quantum mechanism itself better", isolated from the
      repair).

NSGA-III's search is replicated locally (same pymoo setup
Solvers/NSGA3/main.py::run_nsga3 uses) rather than calling run_nsga3
itself, since that function has no low-level mode that skips writing the
shared nsga3_chromosomes.json cache file. QI-NSGA-III's search uses the
existing low-level Solvers.QINSGA3.algorithm.run_qinsga3 directly, which
already returns raw arrays without touching any cache -- same pattern
sensitivity/compare_route_repair_final.py already uses.

Usage:
    python -m sensitivity.compare_2opt_fairness --gen 5 --seeds 42   # smoke
    python -m sensitivity.compare_2opt_fairness --seeds 42 137 271 --gen 300
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
from pymoo.algorithms.moo.nsga3    import NSGA3
from pymoo.util.ref_dirs           import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm   import PM
from pymoo.operators.sampling.rnd  import FloatRandomSampling
from pymoo.optimize                import minimize
from pymoo.termination             import get_termination
from scipy.stats import mannwhitneyu, wilcoxon

from models.parametres            import load_instance
from Solvers.NSGA3.metrics        import compute_pareto_metrics
from Solvers.NSGA3.problem        import IRPProblem
from Solvers.NSGA3.report_builder import _evaluate_pareto
from Solvers.QINSGA3.algorithm    import run_qinsga3

N_PARTITIONS = 8
N_OBJ        = 4
POP_SIZE     = 200
DEFAULT_SEEDS = [42, 137, 271]

_INDICATORS       = ["HV", "GD", "IGD", "Spacing"]
_HIGHER_IS_BETTER = {"HV": True, "GD": False, "IGD": False, "Spacing": False}

_CONFIGS = [
    "NSGA-III sans 2-opt", "NSGA-III avec 2-opt",
    "QI-NSGA-III sans 2-opt", "QI-NSGA-III avec 2-opt",
]


def _run_nsga3_once(sets_, params_, ref_dirs, effective_pop, max_gen, seed):
    """Local replication of Solvers/NSGA3/main.py::run_nsga3's search setup,
    returning the raw final chromosome array -- run_nsga3 itself has no
    low-level equivalent that skips writing the shared chromosome cache
    file, so this mirrors its exact pymoo configuration instead (same
    pattern sensitivity/compare_route_repair_final.py already uses for
    QI-NSGA-III's low-level run_qinsga3 call)."""
    np.random.seed(seed)
    _random.seed(seed)

    problem = IRPProblem(sets_, params_)
    n_genes = len(sets_["clients"]) * len(sets_["T"]) + len(sets_["clients"])
    algorithm = NSGA3(
        pop_size  = effective_pop,
        ref_dirs  = ref_dirs,
        sampling  = FloatRandomSampling(),
        crossover = SBX(prob=0.9, eta=20),
        mutation  = PM(prob=1.0 / n_genes, eta=20),
    )
    result = minimize(
        problem, algorithm, get_termination("n_gen", max_gen), seed=seed, verbose=False,
    )
    return result.X if result.X is not None else np.empty((0, n_genes))


def _config_F(pareto_X, sets_, params_, repair: bool) -> np.ndarray:
    """Decode (and optionally 2-opt-repair) a chromosome array via the
    shared _evaluate_pareto, returning its (f1,f2,f3,f4) objective matrix."""
    if pareto_X is None or len(pareto_X) == 0:
        return np.empty((0, N_OBJ))
    result = _evaluate_pareto(pareto_X, sets_, params_, {}, repair=repair)
    return np.array([[s["objectives"]["f1"], s["objectives"]["f2"],
                       s["objectives"]["f3"], s["objectives"]["f4"]]
                      for s in result["solutions"]])


def _stats(vals) -> dict:
    a = np.array(vals, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std())}


def run_comparison(instance: str, seeds: list[int], max_gen: int, pop_size: int) -> None:
    data_path = os.path.join(PROJECT_DIR, "data", f"instance_{instance}_clients.json")
    sets_, params_ = load_instance(data_path)
    ref_dirs = get_reference_directions("das-dennis", N_OBJ, n_partitions=N_PARTITIONS)
    effective_pop = max(pop_size, len(ref_dirs))

    print("=" * 92)
    print("  2-OPT FAIRNESS CAMPAIGN -- NSGA-III vs QI-NSGA-III, 4 configurations")
    print("=" * 92)
    print(f"  Instance : {instance} clients | pop={effective_pop} | gen={max_gen}")
    print(f"  Seeds    : {seeds}")
    print("=" * 92)

    F_by_config:       dict[str, list[np.ndarray]] = {c: [] for c in _CONFIGS}
    elapsed_by_config: dict[str, list[float]]      = {c: [] for c in _CONFIGS}

    for seed in seeds:
        print(f"\n>>> seed={seed}")

        t0 = time.time()
        X_nsga3 = _run_nsga3_once(sets_, params_, ref_dirs, effective_pop, max_gen, seed)
        t_nsga3 = time.time() - t0
        print(f"  NSGA-III    search done in {t_nsga3:.1f}s | front={len(X_nsga3)}")

        t0 = time.time()
        X_qinsga3, _, _ = run_qinsga3(
            sets_=sets_, params_=params_, ref_dirs=ref_dirs,
            pop_size=effective_pop, max_gen=max_gen, seed=seed,
            repair_final_front=False,
        )
        t_qinsga3 = time.time() - t0
        n_qi = len(X_qinsga3) if X_qinsga3 is not None else 0
        print(f"  QI-NSGA-III search done in {t_qinsga3:.1f}s | front={n_qi}")

        for label, X, base_elapsed, repair in (
            ("NSGA-III sans 2-opt",    X_nsga3,   t_nsga3,   False),
            ("NSGA-III avec 2-opt",    X_nsga3,   t_nsga3,   True),
            ("QI-NSGA-III sans 2-opt", X_qinsga3, t_qinsga3, False),
            ("QI-NSGA-III avec 2-opt", X_qinsga3, t_qinsga3, True),
        ):
            t0 = time.time()
            F = _config_F(X, sets_, params_, repair=repair)
            t_repair = time.time() - t0
            F_by_config[label].append(F)
            elapsed_by_config[label].append(base_elapsed + t_repair)
            print(f"    {label:<24} front={len(F)}  decode+repair={t_repair:.2f}s")

    all_F = np.vstack([F for runs in F_by_config.values() for F in runs if len(F) > 0])
    g_ideal = all_F.min(axis=0)
    g_nadir = all_F.max(axis=0)
    print(f"\nIdeal global partage : {g_ideal}")
    print(f"Nadir global partage  : {g_nadir}")

    values: dict[str, dict[str, list]] = {c: {ind: [] for ind in _INDICATORS} for c in _CONFIGS}
    for label, runs in F_by_config.items():
        for F in runs:
            if len(F) == 0:
                continue
            q = compute_pareto_metrics(F, g_ideal, g_nadir)
            for ind in _INDICATORS:
                values[label][ind].append(q[ind])

    print(f"\n{'-'*92}")
    print("  RESULTATS (ideal/nadir global partage entre les 4 configurations)")
    print(f"{'-'*92}")
    for ind in _INDICATORS:
        arrow = "^" if _HIGHER_IS_BETTER[ind] else "v"
        print(f"\n  {ind} ({arrow})")
        for label in _CONFIGS:
            if not values[label][ind]:
                print(f"    {label:<24} (aucune donnee)")
                continue
            s = _stats(values[label][ind])
            print(f"    {label:<24} mean={s['mean']:.6f}  std={s['std']:.6f}")

    print(f"\n{'-'*92}")
    print("  Effet 2-opt (Wilcoxon apparie -- meme front initial, meme seed)")
    print(f"{'-'*92}")
    for algo, without_lbl, with_lbl in (
        ("NSGA-III",    "NSGA-III sans 2-opt",    "NSGA-III avec 2-opt"),
        ("QI-NSGA-III", "QI-NSGA-III sans 2-opt", "QI-NSGA-III avec 2-opt"),
    ):
        print(f"\n  {algo}")
        for ind in _INDICATORS:
            a, b = values[without_lbl][ind], values[with_lbl][ind]
            if len(a) < 2 or len(b) < 2 or len(a) != len(b):
                print(f"    {ind:<8} : pas assez de runs valides pour un test apparie")
                continue
            try:
                stat, p = wilcoxon(a, b)
            except ValueError:
                print(f"    {ind:<8} : difference nulle sur tous les seeds -- test non applicable")
                continue
            sig = "significatif (p<0.05)" if p < 0.05 else "non significatif"
            print(f"    {ind:<8} W={stat:.1f}  p={p:.6f}  -> {sig}")

    print(f"\n{'-'*92}")
    print("  Effet moteur quantique (Mann-Whitney U -- fronts independants)")
    print(f"{'-'*92}")
    for title, lbl_a, lbl_b in (
        ("Sans 2-opt des deux cotes (ecart brut)",      "NSGA-III sans 2-opt", "QI-NSGA-III sans 2-opt"),
        ("Avec 2-opt des deux cotes (ecart equitable)", "NSGA-III avec 2-opt", "QI-NSGA-III avec 2-opt"),
    ):
        print(f"\n  {title}")
        for ind in _INDICATORS:
            a, b = values[lbl_a][ind], values[lbl_b][ind]
            if len(a) < 2 or len(b) < 2:
                print(f"    {ind:<8} : pas assez de runs valides pour un test")
                continue
            u, p = mannwhitneyu(a, b, alternative="two-sided")
            sig = "significatif (p<0.05)" if p < 0.05 else "non significatif"
            print(f"    {ind:<8} U={u:.1f}  p={p:.6f}  -> {sig}")

    print(f"\n{'-'*92}")
    print("  Temps moyen par configuration (recherche + decodage/repair)")
    print(f"{'-'*92}")
    for label in _CONFIGS:
        s = _stats(elapsed_by_config[label])
        print(f"    {label:<24} mean={s['mean']:.1f}s")

    print(f"\n{'='*92}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Campagne d'equite 2-opt (4 configurations) -- NSGA-III vs QI-NSGA-III"
    )
    parser.add_argument("--instance", default="100",
                        choices=["3", "5", "15", "25", "30", "40", "100"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--gen", type=int, default=300)
    parser.add_argument("--pop", type=int, default=POP_SIZE)
    args = parser.parse_args()
    run_comparison(args.instance, args.seeds, args.gen, args.pop)
```

- [ ] **Step 2: Smoke-run the script end-to-end on the tiny 3-client instance**

Run: `PYTHONIOENCODING=utf-8 python -m sensitivity.compare_2opt_fairness --instance 3 --gen 2 --seeds 42`
Expected: completes without a traceback, prints all four sections (RESULTATS, Effet 2-opt, Effet moteur quantique, Temps moyen). With only 1 seed, the Wilcoxon/Mann-Whitney sections will correctly print "pas assez de runs valides pour un test" (n=1 everywhere) rather than crashing -- this is expected, not a bug.

- [ ] **Step 3: Smoke-run with the project's standard fast-reject smoke params**

Run: `PYTHONIOENCODING=utf-8 python -m sensitivity.compare_2opt_fairness --gen 5 --seeds 42`
Expected: completes without a traceback on the default 100-client instance (this run takes longer than Step 2 -- allow a few minutes).

- [ ] **Step 4: Commit**

```bash
git add sensitivity/compare_2opt_fairness.py
git commit -m "$(cat <<'EOF'
feat: 4-configuration 2-opt fairness campaign (NSGA-III vs QI-NSGA-III)

Runs each algorithm's search once per seed, then decodes the same
resulting chromosome array with and without the post-decode 2-opt repair
-- guaranteeing identical initial fronts, seeds, and budgets across all
four configurations. Separates the 2-opt effect (paired Wilcoxon, same
front) from the evolutionary-engine effect (Mann-Whitney, independent
fronts), reporting both the raw gap (no repair either side) and the fair
gap (repair on both sides) so the quantum mechanism's own contribution can
be read off without the post-processing confound.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## After this plan: running the real campaign

Once all 5 tasks are merged, the actual 3-seed, 300-generation, 100-client
campaign (`python -m sensitivity.compare_2opt_fairness`, no smoke-mode
flags) is a separate follow-up action, not part of this plan -- it produces
real numbers to fold into `Solvers/QINSGA3/README.md` and
`Solvers/IRP_results_summary.md` as a new dated entry, per the design doc's
Non-goals section.
