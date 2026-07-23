# QINSGA3 Benchmark Validation on DTLZ/MaF — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate QINSGA3 (Quantum-Inspired NSGA-III) on the DTLZ1-7/MaF1-7 benchmark suites using the same Cui et al. (2025) protocol (population, generations, reference directions, seeds) already used to validate classic NSGA-III, reorganizing `validation/` so NSGA-III, QINSGA3, and shared code/results are clearly separated, and extending the `/benchmarking` dashboard to show both algorithms side by side.

**Architecture:** A new `validation/algorithms/qinsga3/` package provides a `run_experiment(problem_name, problem, n_gen, n_runs)` function with the exact same signature as the existing `validation/algorithms/nsga3/runner.py` (after it is moved from its current flat location), so `validation/engine.py` can drive either algorithm through one `run_experiment_fn` parameter. QINSGA3's benchmark loop reuses — by import, not duplication — the already problem-agnostic helpers in `QINSGA3/algorithm.py` (normalisation, niching, archive, crowding) and `QINSGA3/chromosome.py`'s `QuantumPopulation`, replacing only the IRP-specific multiprocessing evaluation with a single vectorised `problem.evaluate()` call per generation. Results land in per-algorithm subfolders (`results/nsga3/`, `results/qinsga3/`) under the same suite directories, and the dashboard reads both.

**Tech Stack:** Python, pymoo (NSGA-III, reference directions, IGD, non-dominated sorting), NumPy, Flask (`app.py` dashboard), pytest-less inline test scripts (matching the existing `if __name__ == "__main__":` test style in `validation/`).

## Global Constraints

- Population size, generation count, and Das-Dennis reference directions for QINSGA3 on DTLZ/MaF MUST come from the existing `get_run_config(n_obj)` (Cui et al. 2025 protocol: M=3 → 91/92, M=4 → 120/120) — identical to the classic NSGA-III validation already completed.
- QINSGA3's quantum-operator parameters (alpha_max=0.10π, alpha_min=0.001π, p_cross=0.9, eta_cross=5.0, p_mut=2/D, p_mut_strong=0.15, mut_sigma=0.05π, migration_period=10, n_migrate=10, rotation_type="tanh") MUST match `QINSGA3/main.py`'s existing IRP defaults verbatim.
- All 30 runs MUST use the same 30 deterministic seeds already used for the NSGA-III validation (currently `_SEEDS` in `validation/algorithms/nsga3_runner.py`).
- `QINSGA3/algorithm.py`, `QINSGA3/main.py`, and any IRP-production code path MUST NOT be modified.
- No new third-party dependencies.

---

### Task 1: Extract the shared seed list into its own module

**Files:**
- Create: `validation/algorithms/seeds.py`
- Modify: `validation/algorithms/nsga3_runner.py:45-49`
- Test: `validation/algorithms/test_nsga3_runner.py` (existing file, run unchanged — this task must not break it)

**Interfaces:**
- Produces: `validation.algorithms.seeds.SEEDS` — a `list[int]` of 30 deterministic seeds, single source of truth for every algorithm's runner.

- [ ] **Step 1: Create `validation/algorithms/seeds.py`**

```python
"""Shared deterministic seeds for benchmark validation runs.

Every algorithm's runner (NSGA-III, QINSGA3, ...) must import SEEDS from
here rather than defining its own list, so that "run N" always means the
exact same seed across algorithms — required for a fair run-by-run
comparison.

First 20 match Cui et al. (2025)'s protocol; 10 more appended to extend the
run count beyond the article without invalidating the original 20 runs.
"""

SEEDS = [
    42, 137, 271, 491, 613, 733, 857, 977, 1009, 1123,
    1249, 1373, 1499, 1609, 1733, 1871, 1997, 2113, 2237, 2351,
    2467, 2593, 2711, 2837, 2953, 3079, 3191, 3313, 3433, 3557,
]
```

- [ ] **Step 2: Update `validation/algorithms/nsga3_runner.py` to import SEEDS instead of defining it**

Replace lines 45-49 (the `_SEEDS = [...]` block and its comment) with:

```python
from validation.algorithms.seeds import SEEDS as _SEEDS
```

Leave every other line of the file untouched — `_SEEDS` is still the name used everywhere else in this file (`run_experiment`), so no other line needs to change.

- [ ] **Step 3: Run the existing NSGA-III runner tests to confirm nothing broke**

Run: `cd Project_Irp && ./.venv/Scripts/python.exe -m validation.algorithms.test_nsga3_runner`
Expected: `All tests passed.` (same output as before this change)

- [ ] **Step 4: Commit**

```bash
git add validation/algorithms/seeds.py validation/algorithms/nsga3_runner.py
git commit -m "refactor: extract shared seed list into validation/algorithms/seeds.py"
```

---

### Task 2: Move the NSGA-III runner into its own `nsga3/` package

**Files:**
- Create: `validation/algorithms/nsga3/__init__.py` (empty)
- Create: `validation/algorithms/nsga3/runner.py` (moved content of `nsga3_runner.py`, with the seeds import path updated)
- Create: `validation/algorithms/nsga3/test_runner.py` (moved content of `test_nsga3_runner.py`, imports updated)
- Delete: `validation/algorithms/nsga3_runner.py`
- Delete: `validation/algorithms/test_nsga3_runner.py`
- Modify: `validation/engine.py:18`

**Interfaces:**
- Consumes: `validation.algorithms.seeds.SEEDS` (from Task 1).
- Produces: `validation.algorithms.nsga3.runner.get_run_config(n_obj)`, `.run_single(problem, n_gen, seed)`, `.run_experiment(problem_name, problem, n_gen, n_runs=30)` — same three names/signatures as the old `nsga3_runner.py`, just at a new import path. `validation/engine.py` and any future algorithm runner rely on this exact `run_experiment` signature.

- [ ] **Step 1: Create the package directory and move the runner file**

```bash
cd Project_Irp
mkdir -p validation/algorithms/nsga3
touch validation/algorithms/nsga3/__init__.py
git mv validation/algorithms/nsga3_runner.py validation/algorithms/nsga3/runner.py
git mv validation/algorithms/test_nsga3_runner.py validation/algorithms/nsga3/test_runner.py
```

- [ ] **Step 2: Fix the seeds import inside the moved runner**

In `validation/algorithms/nsga3/runner.py`, the line added in Task 1 currently reads:

```python
from validation.algorithms.seeds import SEEDS as _SEEDS
```

This path is still correct after the move (it's an absolute import from the repo root), so no change is needed here — just confirm it's present.

- [ ] **Step 3: Fix imports inside the moved test file**

In `validation/algorithms/nsga3/test_runner.py`, change:

```python
from validation.algorithms.nsga3_runner import run_single, run_experiment, POP_SIZE, N_REF_DIRS
```

to:

```python
from validation.algorithms.nsga3.runner import run_single, run_experiment, POP_SIZE, N_REF_DIRS
```

Leave the rest of the file (`from validation.dtlz.dtlz_problems import get_problem` and all test functions) unchanged.

- [ ] **Step 4: Run the moved tests**

Run: `cd Project_Irp && ./.venv/Scripts/python.exe -m validation.algorithms.nsga3.test_runner`
Expected: `All tests passed.`

- [ ] **Step 5: Update `validation/engine.py`'s import**

In `validation/engine.py:18`, change:

```python
from validation.algorithms.nsga3_runner import run_experiment, _SEEDS
```

to:

```python
from validation.algorithms.nsga3.runner import run_experiment as _default_run_experiment
from validation.algorithms.seeds import SEEDS as _SEEDS
```

- [ ] **Step 6: Update `validate()`'s call site to use the renamed default**

In `validation/engine.py`, inside `def validate(...)`, find the line:

```python
        fronts = run_experiment(name, problem, n_gen, n_runs)
```

Change it to:

```python
        fronts = run_experiment_fn(name, problem, n_gen, n_runs)
```

(This adds the `run_experiment_fn` parameter, defaulting to `_default_run_experiment` so `engine.py` behaves identically to before for every existing call site. Task 4 only adds a docstring line for it and starts passing it explicitly from the CLIs — the signature itself is final here.)

Change the `validate()` signature line from:

```python
def validate(problems_module, results_dir: str, suite_label: str, n_runs: int = 30, n_obj: int = 4):
```

to:

```python
def validate(problems_module, results_dir: str, suite_label: str, n_runs: int = 30, n_obj: int = 4,
             run_experiment_fn=_default_run_experiment):
```

- [ ] **Step 7: Smoke-test the full DTLZ validation CLI still works end to end**

Run: `cd Project_Irp && ./.venv/Scripts/python.exe -m validation.dtlz.main_validation --runs 2 --n_obj 3`
Expected: prints a results table for DTLZ1-7 exactly as before, writes CSVs under `validation/dtlz/results/` (still the old flat location at this point — Task 3 moves this).

- [ ] **Step 8: Clean up the CSVs written by the smoke test (they're 2-run test artifacts, not real validation data)**

```bash
rm -f validation/dtlz/results/igd_*.csv validation/dtlz/results/summary_M*.csv
```

- [ ] **Step 9: Commit**

```bash
git add validation/algorithms/nsga3/ validation/engine.py
git commit -m "refactor: move NSGA-III runner into validation/algorithms/nsga3/ package"
```

---

### Task 3: Move existing NSGA-III result files into a `results/nsga3/` subfolder

**Files:**
- Move: `validation/dtlz/results/*.csv` → `validation/dtlz/results/nsga3/*.csv`
- Move: `validation/dtlz/results/DTLZ_results_summary.md` → `validation/dtlz/results/nsga3/DTLZ_results_summary_nsga3.md`
- Move: `validation/maf/results/*.csv` → `validation/maf/results/nsga3/*.csv`
- Move: `validation/maf/results/MaF_results_summary.md` → `validation/maf/results/nsga3/MaF_results_summary_nsga3.md`
- Modify: `app.py:122-134` (`_read_igd_runs`)
- Modify: `validation/dtlz/main_validation.py:19` (`RESULTS_DIR`)
- Modify: `validation/maf/main_validation.py:19` (`RESULTS_DIR`)

**Interfaces:**
- Produces: every NSGA-III result file now lives under a suite's `results/nsga3/` subfolder instead of directly in `results/`. `app.py`'s dashboard and both `main_validation.py` scripts are updated to match — this task keeps the dashboard showing exactly the same (single-algorithm) data as before, just from the new path.

- [ ] **Step 1: Move the DTLZ result files**

```bash
cd Project_Irp
mkdir -p validation/dtlz/results/nsga3
git mv validation/dtlz/results/igd_DTLZ1_M3.csv validation/dtlz/results/nsga3/igd_DTLZ1_M3.csv
git mv validation/dtlz/results/igd_DTLZ1_M4.csv validation/dtlz/results/nsga3/igd_DTLZ1_M4.csv
git mv validation/dtlz/results/igd_DTLZ2_M3.csv validation/dtlz/results/nsga3/igd_DTLZ2_M3.csv
git mv validation/dtlz/results/igd_DTLZ2_M4.csv validation/dtlz/results/nsga3/igd_DTLZ2_M4.csv
git mv validation/dtlz/results/igd_DTLZ3_M3.csv validation/dtlz/results/nsga3/igd_DTLZ3_M3.csv
git mv validation/dtlz/results/igd_DTLZ3_M4.csv validation/dtlz/results/nsga3/igd_DTLZ3_M4.csv
git mv validation/dtlz/results/igd_DTLZ4_M3.csv validation/dtlz/results/nsga3/igd_DTLZ4_M3.csv
git mv validation/dtlz/results/igd_DTLZ4_M4.csv validation/dtlz/results/nsga3/igd_DTLZ4_M4.csv
git mv validation/dtlz/results/igd_DTLZ5_M3.csv validation/dtlz/results/nsga3/igd_DTLZ5_M3.csv
git mv validation/dtlz/results/igd_DTLZ6_M3.csv validation/dtlz/results/nsga3/igd_DTLZ6_M3.csv
git mv validation/dtlz/results/igd_DTLZ7_M3.csv validation/dtlz/results/nsga3/igd_DTLZ7_M3.csv
git mv validation/dtlz/results/summary_M3.csv validation/dtlz/results/nsga3/summary_M3.csv
git mv validation/dtlz/results/summary_M4.csv validation/dtlz/results/nsga3/summary_M4.csv
git mv validation/dtlz/results/DTLZ_results_summary.md validation/dtlz/results/nsga3/DTLZ_results_summary_nsga3.md
```

- [ ] **Step 2: Move the MaF result files**

```bash
mkdir -p validation/maf/results/nsga3
git mv validation/maf/results/igd_MaF1_M3.csv validation/maf/results/nsga3/igd_MaF1_M3.csv
git mv validation/maf/results/igd_MaF1_M4.csv validation/maf/results/nsga3/igd_MaF1_M4.csv
git mv validation/maf/results/igd_MaF2_M3.csv validation/maf/results/nsga3/igd_MaF2_M3.csv
git mv validation/maf/results/igd_MaF2_M4.csv validation/maf/results/nsga3/igd_MaF2_M4.csv
git mv validation/maf/results/igd_MaF3_M3.csv validation/maf/results/nsga3/igd_MaF3_M3.csv
git mv validation/maf/results/igd_MaF3_M4.csv validation/maf/results/nsga3/igd_MaF3_M4.csv
git mv validation/maf/results/igd_MaF4_M3.csv validation/maf/results/nsga3/igd_MaF4_M3.csv
git mv validation/maf/results/igd_MaF4_M4.csv validation/maf/results/nsga3/igd_MaF4_M4.csv
git mv validation/maf/results/igd_MaF5_M3.csv validation/maf/results/nsga3/igd_MaF5_M3.csv
git mv validation/maf/results/igd_MaF5_M4.csv validation/maf/results/nsga3/igd_MaF5_M4.csv
git mv validation/maf/results/igd_MaF6_M3.csv validation/maf/results/nsga3/igd_MaF6_M3.csv
git mv validation/maf/results/igd_MaF6_M4.csv validation/maf/results/nsga3/igd_MaF6_M4.csv
git mv validation/maf/results/igd_MaF7_M3.csv validation/maf/results/nsga3/igd_MaF7_M3.csv
git mv validation/maf/results/igd_MaF7_M4.csv validation/maf/results/nsga3/igd_MaF7_M4.csv
git mv validation/maf/results/summary_M3.csv validation/maf/results/nsga3/summary_M3.csv
git mv validation/maf/results/summary_M4.csv validation/maf/results/nsga3/summary_M4.csv
git mv validation/maf/results/MaF_results_summary.md validation/maf/results/nsga3/MaF_results_summary_nsga3.md
```

- [ ] **Step 3: Update `app.py`'s `_read_igd_runs` to read from the `nsga3/` subfolder**

In `app.py:122-134`, change:

```python
def _read_igd_runs(results_dir: str, problem: str, n_obj: int):
    path = os.path.join(results_dir, f"igd_{problem}_M{n_obj}.csv")
```

to:

```python
def _read_igd_runs(results_dir: str, problem: str, n_obj: int):
    path = os.path.join(results_dir, "nsga3", f"igd_{problem}_M{n_obj}.csv")
```

(This is a temporary single-algorithm shape — Task 5 replaces it with a per-algorithm loop. Keeping this task's diff minimal lets the dashboard be verified working before the bigger restructure.)

- [ ] **Step 4: Update both `main_validation.py` scripts' output directory**

In `validation/dtlz/main_validation.py:19`, change:

```python
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
```

to:

```python
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "nsga3")
```

Apply the identical change to `validation/maf/main_validation.py:19`.

- [ ] **Step 5: Smoke-test the dashboard reads the moved files correctly**

Run: `cd Project_Irp && ./.venv/Scripts/python.exe -c "import app; d = app._build_benchmark_data(); print(d['dtlz']['DTLZ2']['M3'])"`
Expected: prints a dict with `best`, `median`, `worst`, `mean`, `std`, `n_runs: 30` (same values as before the move — only the source path changed).

- [ ] **Step 6: Smoke-test the CLI writes to the new location**

Run: `cd Project_Irp && ./.venv/Scripts/python.exe -m validation.dtlz.main_validation --runs 2 --n_obj 3`
Expected: results table printed; `validation/dtlz/results/nsga3/summary_M3.csv` is overwritten (confirm via `git status` that only files already inside `nsga3/` show as modified, nothing new appears directly under `results/`).

- [ ] **Step 7: Revert the smoke-test's 2-run overwrite**

```bash
git checkout -- validation/dtlz/results/nsga3/
```

- [ ] **Step 8: Commit**

```bash
git add -A validation/dtlz/results validation/maf/results app.py validation/dtlz/main_validation.py validation/maf/main_validation.py
git commit -m "refactor: move NSGA-III benchmark results into results/nsga3/ subfolders"
```

---

### Task 4: Add the `--algorithm` flag and finalize `engine.py`'s `run_experiment_fn` parameter

**Files:**
- Modify: `validation/engine.py` (finalize the parameter added provisionally in Task 2, add docstring)
- Modify: `validation/dtlz/main_validation.py`
- Modify: `validation/maf/main_validation.py`

**Interfaces:**
- Consumes: `validation.algorithms.nsga3.runner.run_experiment` (Task 2).
- Produces: `validate(..., run_experiment_fn=...)` — any callable matching `(problem_name, problem, n_gen, n_runs) -> list[np.ndarray]` can be passed. Both `main_validation.py` CLIs gain `--algorithm {nsga3,qinsga3}` (default `nsga3`); passing `qinsga3` before Task 7 exists will raise `ModuleNotFoundError` — expected and fine at this point in the plan.

- [ ] **Step 1: Add a docstring line for the new parameter in `validation/engine.py`**

In the `validate()` docstring (the `Args:` block), add this line after the `n_obj` line:

```python
        run_experiment_fn : callable(problem_name, problem, n_gen, n_runs) -> list of fronts;
                             defaults to the classic NSGA-III runner.
```

- [ ] **Step 2: Add the `--algorithm` flag to `validation/dtlz/main_validation.py`**

Replace the whole `if __name__ == "__main__":` block with:

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate NSGA-III/QINSGA3 on DTLZ1-7 (Cui et al. 2025 protocol, M=3/M=4)"
    )
    parser.add_argument(
        "--runs", type=int, default=30,
        help="Number of independent runs per problem (default: 30)"
    )
    parser.add_argument(
        "--n_obj", type=int, default=4, choices=[3, 4],
        help="Number of objectives (default: 4)"
    )
    parser.add_argument(
        "--algorithm", default="nsga3", choices=["nsga3", "qinsga3"],
        help="Which algorithm to validate (default: nsga3)"
    )
    args = parser.parse_args()

    if args.runs < 1 or args.runs > 30:
        print("--runs must be between 1 and 30")
        sys.exit(1)

    if args.algorithm == "qinsga3":
        from validation.algorithms.qinsga3.runner import run_experiment
    else:
        from validation.algorithms.nsga3.runner import run_experiment

    RESULTS_DIR_ALGO = os.path.join(os.path.dirname(__file__), "results", args.algorithm)

    validate(
        problems_module=dtlz_problems,
        results_dir=RESULTS_DIR_ALGO,
        suite_label="DTLZ1-7",
        n_runs=args.runs,
        n_obj=args.n_obj,
        run_experiment_fn=run_experiment,
    )
```

Also update the module docstring's usage examples at the top of the file to mention `--algorithm`:

```python
"""
DTLZ validation CLI — runs NSGA-III/QINSGA3 on DTLZ1-7 (Cui et al. 2025 protocol, M=3/M=4).

Usage:
    python -m validation.dtlz.main_validation                        # NSGA-III, 30 runs, 4 objectives
    python -m validation.dtlz.main_validation --algorithm qinsga3    # QINSGA3, 30 runs, 4 objectives
    python -m validation.dtlz.main_validation --runs 3               # 3 runs, 4 objectives
    python -m validation.dtlz.main_validation --n_obj 3              # 30 runs, 3 objectives
    python -m validation.dtlz.main_validation --n_obj 3 --runs 30    # full 3-obj validation
"""
```

Remove the now-unused module-level `RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "nsga3")` line added in Task 3 — it is superseded by `RESULTS_DIR_ALGO` computed from `args.algorithm` inside `__main__`.

- [ ] **Step 3: Apply the identical change to `validation/maf/main_validation.py`**

Same edits, with `"DTLZ1-7"` → `"MaF1-7"` and `dtlz_problems` → `maf_problems` (these are already what the file uses — just add the `--algorithm` argument, the `if/else` import, and the `RESULTS_DIR_ALGO` computation the same way).

- [ ] **Step 4: Smoke-test the default (nsga3) path still works identically**

Run: `cd Project_Irp && ./.venv/Scripts/python.exe -m validation.dtlz.main_validation --runs 2 --n_obj 3`
Expected: same results table as Task 3's Step 6; writes to `validation/dtlz/results/nsga3/`.

- [ ] **Step 5: Revert the smoke test's overwrite**

```bash
git checkout -- validation/dtlz/results/nsga3/
```

- [ ] **Step 6: Confirm `--algorithm qinsga3` fails with the expected (temporary) error**

Run: `cd Project_Irp && ./.venv/Scripts/python.exe -m validation.dtlz.main_validation --algorithm qinsga3 --runs 2 --n_obj 3`
Expected: `ModuleNotFoundError: No module named 'validation.algorithms.qinsga3'` — this is expected until Task 7; it confirms the flag correctly attempts the qinsga3 import path.

- [ ] **Step 7: Commit**

```bash
git add validation/engine.py validation/dtlz/main_validation.py validation/maf/main_validation.py
git commit -m "feat: add --algorithm flag to DTLZ/MaF validation CLIs"
```

---

### Task 5: Restructure the dashboard backend and front end for per-algorithm comparison

**Files:**
- Modify: `app.py:122-158` (`_read_igd_runs`, `_build_benchmark_data`)
- Modify: `app.py:243` (CSS: add `.algo-compare`)
- Modify: `app.py:389-392` (`M_TITLE` — no longer needed as a fixed NSGA-III label)
- Modify: `app.py:475-499` (`renderCard` → per-algorithm card)
- Modify: `app.py:551-568` (`renderResultsForM`)

**Interfaces:**
- Produces: `DATA[suite][problem]["M3"]` is now `{"nsga3": {...stats...}, "qinsga3": {...stats...}}` (only algorithms with data present as keys) instead of a flat stats dict. The front end renders one card per available algorithm, side by side.

- [ ] **Step 1: Update `_read_igd_runs` to take an `algo` argument**

In `app.py:122-134`, change:

```python
def _read_igd_runs(results_dir: str, problem: str, n_obj: int):
    path = os.path.join(results_dir, "nsga3", f"igd_{problem}_M{n_obj}.csv")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "run":  int(row["run"]),
                "seed": int(row["seed"]),
                "igd":  float(row["igd"]),
            })
    return rows
```

to:

```python
_BENCHMARK_ALGOS = ["nsga3", "qinsga3"]


def _read_igd_runs(results_dir: str, algo: str, problem: str, n_obj: int):
    path = os.path.join(results_dir, algo, f"igd_{problem}_M{n_obj}.csv")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "run":  int(row["run"]),
                "seed": int(row["seed"]),
                "igd":  float(row["igd"]),
            })
    return rows
```

- [ ] **Step 2: Nest `_build_benchmark_data` by algorithm**

Immediately below, change:

```python
def _build_benchmark_data():
    data = {}
    for suite, cfg in _BENCHMARK_SUITES.items():
        data[suite] = {}
        for p in cfg["problems"]:
            data[suite][p] = {}
            for m in _BENCHMARK_M_VALUES:
                details = _read_igd_runs(cfg["results_dir"], p, m)
                if not details:
                    continue
                igds = [r["igd"] for r in details]
                data[suite][p][f"M{m}"] = {
                    "best":   min(igds),
                    "median": float(_statistics.median(igds)),
                    "worst":  max(igds),
                    "mean":   float(_statistics.mean(igds)),
                    "std":    float(_statistics.pstdev(igds)),
                    "runs":   igds,
                    "run_details": details,
                    "n_runs": len(igds),
                }
    return data
```

to:

```python
def _build_benchmark_data():
    data = {}
    for suite, cfg in _BENCHMARK_SUITES.items():
        data[suite] = {}
        for p in cfg["problems"]:
            data[suite][p] = {}
            for m in _BENCHMARK_M_VALUES:
                m_data = {}
                for algo in _BENCHMARK_ALGOS:
                    details = _read_igd_runs(cfg["results_dir"], algo, p, m)
                    if not details:
                        continue
                    igds = [r["igd"] for r in details]
                    m_data[algo] = {
                        "best":   min(igds),
                        "median": float(_statistics.median(igds)),
                        "worst":  max(igds),
                        "mean":   float(_statistics.mean(igds)),
                        "std":    float(_statistics.pstdev(igds)),
                        "runs":   igds,
                        "run_details": details,
                        "n_runs": len(igds),
                    }
                if m_data:
                    data[suite][p][f"M{m}"] = m_data
    return data
```

- [ ] **Step 3: Verify the new data shape**

Run: `cd Project_Irp && ./.venv/Scripts/python.exe -c "import app; d = app._build_benchmark_data(); print(d['dtlz']['DTLZ2']['M3'].keys()); print(d['dtlz']['DTLZ2']['M3']['nsga3']['n_runs'])"`
Expected: `dict_keys(['nsga3'])` (qinsga3 has no CSVs yet) and `30`.

- [ ] **Step 4: Add the `.algo-compare` CSS rule**

In `app.py`, right after line 243 (`.grid2 { display:flex;flex-direction:column;gap:20px; }`), add:

```css
.algo-compare { display:flex;flex-wrap:wrap;gap:16px; }
.algo-compare .rcard { flex:1 1 380px;min-width:320px; }
```

- [ ] **Step 5: Replace the fixed `M_TITLE` map with an algorithm label map**

Change:

```javascript
const M_TITLE = {
  M3: 'NSGA-III &mdash; M <span class="param-lbl">(objectifs)</span> = 3',
  M4: 'NSGA-III &mdash; M <span class="param-lbl">(objectifs)</span> = 4',
};
```

to:

```javascript
const ALGO_LABEL = {
  nsga3:   'NSGA-III (classique)',
  qinsga3: 'QI-NSGA-III (quantum-inspired)',
};
const ALGO_ORDER = ['nsga3', 'qinsga3'];
```

- [ ] **Step 6: Update `renderCard` to render one algorithm's card and key its chart by `M:algo`**

Change:

```javascript
function renderCard(key, d) {
  return `<div class="rcard">
    <div class="rcard-hdr">
      <h3>${M_TITLE[key]}</h3>
      <div class="rcard-sub">N=${M_META[key].N}, G=${M_META[key].G} &nbsp;|&nbsp; ${d.n_runs} runs</div>
    </div>
    <div class="rcard-body">
      <div class="stat-row">
        <div class="stat-box best"><div class="stat-lbl">Best</div><div class="stat-val">${d.best.toExponential(3)}<span class="stat-val-dec">${fmtDec(d.best)}</span></div></div>
        <div class="stat-box med"><div class="stat-lbl">Median</div><div class="stat-val">${d.median.toExponential(3)}<span class="stat-val-dec">${fmtDec(d.median)}</span></div></div>
        <div class="stat-box mean"><div class="stat-lbl">Mean <span class="param-lbl">(&plusmn;&sigma;)</span></div><div class="stat-val">${d.mean.toExponential(3)}<span class="stat-val-dec">${fmtDec(d.mean)} &plusmn; ${fmtDec(d.std)}</span></div></div>
        <div class="stat-box worst"><div class="stat-lbl">Worst</div><div class="stat-val">${d.worst.toExponential(3)}<span class="stat-val-dec">${fmtDec(d.worst)}</span></div></div>
      </div>
      <div class="stat-note">Mean = statistique compar&eacute;e dans Cui et al. 2025 (PlatEMO) &mdash; Median = statistique la plus robuste aux runs rat&eacute;s</div>
      <div class="chart-lbl">Run distribution &mdash; ${d.n_runs} runs, tri&eacute;s par IGD (rang &times; valeur, &eacute;chelle log)</div>
      <svg class="dot-svg" viewBox="0 0 360 74" style="height:74px" data-key="${key}"></svg>
      <div class="chart-legend">
        <span><i style="background:rgb(${IGD_GREEN.join(',')})"></i>Succ&egrave;s (IGD &le; 0.06)</span>
        <span><i style="background:rgb(${IGD_ORANGE.join(',')})"></i>Optimum local (0.25&ndash;0.55)</span>
        <span><i style="background:rgb(${IGD_RED.join(',')})"></i>&Eacute;chec de convergence (&gt; 0.75)</span>
      </div>
      ${renderRunsTable(d.run_details)}
    </div>
  </div>`;
}
```

to:

```javascript
function renderCard(mKey, algo, d) {
  const dataKey = `${mKey}:${algo}`;
  const nObj = mKey === 'M3' ? 3 : 4;
  return `<div class="rcard">
    <div class="rcard-hdr">
      <h3>${ALGO_LABEL[algo]} &mdash; M <span class="param-lbl">(objectifs)</span> = ${nObj}</h3>
      <div class="rcard-sub">N=${M_META[mKey].N}, G=${M_META[mKey].G} &nbsp;|&nbsp; ${d.n_runs} runs</div>
    </div>
    <div class="rcard-body">
      <div class="stat-row">
        <div class="stat-box best"><div class="stat-lbl">Best</div><div class="stat-val">${d.best.toExponential(3)}<span class="stat-val-dec">${fmtDec(d.best)}</span></div></div>
        <div class="stat-box med"><div class="stat-lbl">Median</div><div class="stat-val">${d.median.toExponential(3)}<span class="stat-val-dec">${fmtDec(d.median)}</span></div></div>
        <div class="stat-box mean"><div class="stat-lbl">Mean <span class="param-lbl">(&plusmn;&sigma;)</span></div><div class="stat-val">${d.mean.toExponential(3)}<span class="stat-val-dec">${fmtDec(d.mean)} &plusmn; ${fmtDec(d.std)}</span></div></div>
        <div class="stat-box worst"><div class="stat-lbl">Worst</div><div class="stat-val">${d.worst.toExponential(3)}<span class="stat-val-dec">${fmtDec(d.worst)}</span></div></div>
      </div>
      <div class="stat-note">Mean = statistique compar&eacute;e dans Cui et al. 2025 (PlatEMO) &mdash; Median = statistique la plus robuste aux runs rat&eacute;s</div>
      <div class="chart-lbl">Run distribution &mdash; ${d.n_runs} runs, tri&eacute;s par IGD (rang &times; valeur, &eacute;chelle log)</div>
      <svg class="dot-svg" viewBox="0 0 360 74" style="height:74px" data-key="${dataKey}"></svg>
      <div class="chart-legend">
        <span><i style="background:rgb(${IGD_GREEN.join(',')})"></i>Succ&egrave;s (IGD &le; 0.06)</span>
        <span><i style="background:rgb(${IGD_ORANGE.join(',')})"></i>Optimum local (0.25&ndash;0.55)</span>
        <span><i style="background:rgb(${IGD_RED.join(',')})"></i>&Eacute;chec de convergence (&gt; 0.75)</span>
      </div>
      ${renderRunsTable(d.run_details)}
    </div>
  </div>`;
}
```

- [ ] **Step 7: Update `renderResultsForM` to render one card per available algorithm and re-key the chart lookup**

Change:

```javascript
function renderResultsForM() {
  const results = document.getElementById('results');
  const d = currentPd[currentM];
  const isDegenerateM4 = DEGENERATE_PROBLEMS.has(currentProb) && currentM === 'M4';
  let html;
  if (d) {
    html = renderCard(currentM, d);
  } else if (isDegenerateM4) {
    html = `<div class="rcard"><div class="nodata">M=4 is not shown for ${currentProb}: it has a degenerate/disconnected true Pareto front, and pymoo only ships a reference front for it at M=3 &mdash; IGD can&rsquo;t be scored otherwise.</div></div>`;
  } else {
    html = `<div class="rcard"><div class="nodata">No benchmark results yet for this problem &mdash; run <code>python -m validation.${currentSuite}.main_validation</code> to generate them.</div></div>`;
  }
  results.innerHTML = html;
  document.querySelectorAll('[data-key]').forEach(svg => {
    const dd = currentPd[svg.dataset.key];
    if (dd) drawDots(dd.runs, svg);
  });
}
```

to:

```javascript
function renderResultsForM() {
  const results = document.getElementById('results');
  const mData = currentPd[currentM] || {};
  const isDegenerateM4 = DEGENERATE_PROBLEMS.has(currentProb) && currentM === 'M4';
  const availableAlgos = ALGO_ORDER.filter(a => mData[a]);
  let html;
  if (availableAlgos.length) {
    html = `<div class="algo-compare">${availableAlgos.map(a => renderCard(currentM, a, mData[a])).join('')}</div>`;
  } else if (isDegenerateM4) {
    html = `<div class="rcard"><div class="nodata">M=4 is not shown for ${currentProb}: it has a degenerate/disconnected true Pareto front, and pymoo only ships a reference front for it at M=3 &mdash; IGD can&rsquo;t be scored otherwise.</div></div>`;
  } else {
    html = `<div class="rcard"><div class="nodata">No benchmark results yet for this problem &mdash; run <code>python -m validation.${currentSuite}.main_validation</code> to generate them.</div></div>`;
  }
  results.innerHTML = html;
  document.querySelectorAll('[data-key]').forEach(svg => {
    const [mKey, algo] = svg.dataset.key.split(':');
    const dd = (currentPd[mKey] || {})[algo];
    if (dd) drawDots(dd.runs, svg);
  });
}
```

- [ ] **Step 8: Start the app and verify the benchmarking page renders**

Run: `cd Project_Irp && ./.venv/Scripts/python.exe main.py` (in background), then in another shell: `curl -s http://127.0.0.1:5000/benchmarking | grep -o "NSGA-III (classique)" | head -1`
Expected: `NSGA-III (classique)` printed (confirms the template rendered without a Jinja/Python error and the new label appears). Stop the app afterward.

- [ ] **Step 9: Commit**

```bash
git add app.py
git commit -m "feat: restructure benchmarking dashboard for per-algorithm comparison"
```

---

### Task 6: Build the generic QINSGA3 benchmark core loop

**Files:**
- Create: `validation/algorithms/qinsga3/__init__.py` (empty)
- Create: `validation/algorithms/qinsga3/core.py`

**Interfaces:**
- Consumes: `QINSGA3.algorithm._normalise_F`, `._assign_ref_dirs`, `._select_guides`, `._supplement_from_archive`, `._migrate`, `._archive_update`, `._crowding_trim`, `._penalised_F` (all already problem-agnostic, imported unmodified); `QINSGA3.chromosome.QuantumPopulation`.
- Produces: `run_qinsga3_generic(problem, ref_dirs, pop_size, max_gen, alpha_max, alpha_min, p_mut, p_mut_strong, mut_sigma, p_cross, eta_cross, migration_period, n_migrate, seed, rotation_type="tanh") -> np.ndarray` returning the final Pareto front's objective values, shape `(n_solutions, n_obj)` — this exact signature and return shape is what `validation/algorithms/qinsga3/runner.py` (Task 7) calls.

- [ ] **Step 1: Create the package directory**

```bash
cd Project_Irp
mkdir -p validation/algorithms/qinsga3
touch validation/algorithms/qinsga3/__init__.py
```

- [ ] **Step 2: Write `validation/algorithms/qinsga3/core.py`**

```python
"""Lean QINSGA-III generational loop for synthetic benchmark problems (DTLZ/MaF).

Mirrors QINSGA3/algorithm.py::run_qinsga3()'s algorithm exactly (same
generation order: measure -> evaluate -> penalise -> non-dominated sort ->
archive update -> normalise -> assign ref dirs -> select guides -> supplement
from archive -> rotate -> crossover -> mutate -> migrate), but:

  - evaluates the whole population in one vectorised pymoo
    Problem.evaluate() call per generation instead of a multiprocessing
    pool — DTLZ/MaF evaluation is cheap, so a process pool would only add
    overhead (unlike the IRP's expensive simulation-based evaluation).
  - takes any pymoo Problem instance instead of the IRP-specific
    IRPProblem.

All non-dominated-sorting, normalisation, niching, archive and crowding
helpers are imported directly from QINSGA3/algorithm.py — not duplicated —
so a future fix to that shared math applies to both the IRP path and this
benchmark path automatically. QINSGA3/algorithm.py itself is never modified
by this module.
"""

from __future__ import annotations

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

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
from QINSGA3.chromosome import QuantumPopulation


def run_qinsga3_generic(
    problem,
    ref_dirs: np.ndarray,
    pop_size: int,
    max_gen: int,
    alpha_max: float,
    alpha_min: float,
    p_mut: float,
    p_mut_strong: float,
    mut_sigma: float,
    p_cross: float,
    eta_cross: float,
    migration_period: int,
    n_migrate: int,
    seed: int,
    rotation_type: str = "tanh",
) -> np.ndarray:
    """Run one QINSGA-III instance on a pymoo Problem. Returns the final Pareto front's F."""
    rng = np.random.default_rng(seed)

    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)
    n_genes = problem.n_var

    qpop = QuantumPopulation(pop_size, n_genes, xl, xu, rng=rng, rotation_type=rotation_type)
    sorter = NonDominatedSorting()

    arch_X: list[np.ndarray] = []
    arch_F: list[np.ndarray] = []
    arch_theta: list[np.ndarray] = []
    _MAX_ARCHIVE = 500

    def _eval_batch(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        F, G = problem.evaluate(X, return_values_of=["F", "G"])
        if G is None or (hasattr(G, "size") and G.size == 0):
            G = np.zeros((X.shape[0], 0))
        return np.asarray(F), np.asarray(G)

    for gen in range(max_gen):
        X = qpop.measure()
        F, G = _eval_batch(X)

        F_pen = _penalised_F(F, G)
        fronts = sorter.do(F_pen)
        pareto_idx = fronts[0]

        _archive_update(
            X[pareto_idx], F[pareto_idx], G[pareto_idx], qpop.theta[pareto_idx],
            arch_X, arch_F, arch_theta, _MAX_ARCHIVE,
        )

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

        alpha = alpha_min + (alpha_max - alpha_min) * (1.0 - gen / max_gen)
        qpop.rotate(guides_theta, alpha)

        if p_cross > 0.0:
            qpop.crossover(p_cross, eta_cross)

        qpop.mutate(p_mut, p_mut_strong, mut_sigma)

        if (arch_F_norm is not None
                and migration_period > 0
                and gen % migration_period == 0):
            _migrate(
                qpop, arch_theta_arr, arch_F_norm,
                assoc, ref_dirs, rng, n_migrate=n_migrate,
            )

    X_final = qpop.measure()
    F_final, G_final = _eval_batch(X_final)
    F_pen_final = _penalised_F(F_final, G_final)
    final_pareto_idx = sorter.do(F_pen_final)[0]
    _archive_update(
        X_final[final_pareto_idx], F_final[final_pareto_idx],
        G_final[final_pareto_idx], qpop.theta[final_pareto_idx],
        arch_X, arch_F, arch_theta, _MAX_ARCHIVE,
    )

    if arch_X:
        _, arch_F_arr, _ = _crowding_trim(
            np.array(arch_X), np.array(arch_F), np.array(arch_theta), pop_size,
        )
        return arch_F_arr

    return F_final[final_pareto_idx]
```

- [ ] **Step 3: Verify it runs on DTLZ2, M=3 and M=4, with a tiny generation count**

Run:

```bash
cd Project_Irp
./.venv/Scripts/python.exe -c "
import numpy as np
from validation.algorithms.qinsga3.core import run_qinsga3_generic
from validation.algorithms.nsga3.runner import get_run_config
from validation.dtlz.dtlz_problems import get_problem
from validation.metrics.igd_metric import compute_igd

for n_obj in (3, 4):
    problem, _ = get_problem('DTLZ2', n_obj=n_obj)
    ref_dirs, pop_size = get_run_config(n_obj)
    F = run_qinsga3_generic(
        problem, ref_dirs, pop_size, max_gen=3,
        alpha_max=0.10*np.pi, alpha_min=0.001*np.pi,
        p_mut=2.0/problem.n_var, p_mut_strong=0.15, mut_sigma=0.05*np.pi,
        p_cross=0.9, eta_cross=5.0, migration_period=10, n_migrate=10, seed=42,
    )
    print(n_obj, 'F shape', F.shape, 'igd', compute_igd(problem, F))
"
```

Expected: two lines printed, e.g. `3 F shape (92, 3) igd <some float>` and `4 F shape (120, 4) igd <some float>` — no exceptions, `F.shape[1]` equal to `n_obj`, `F.shape[0] <= pop_size`, IGD a finite positive float.

- [ ] **Step 4: Commit**

```bash
git add validation/algorithms/qinsga3/__init__.py validation/algorithms/qinsga3/core.py
git commit -m "feat: add generic QINSGA3 core loop for benchmark problems"
```

---

### Task 7: Build the QINSGA3 runner adapter and its tests

**Files:**
- Create: `validation/algorithms/qinsga3/runner.py`
- Create: `validation/algorithms/qinsga3/test_runner.py`

**Interfaces:**
- Consumes: `validation.algorithms.nsga3.runner.get_run_config` (Task 2), `validation.algorithms.seeds.SEEDS` (Task 1), `validation.algorithms.qinsga3.core.run_qinsga3_generic` (Task 6).
- Produces: `run_single(problem, n_gen, seed) -> np.ndarray` and `run_experiment(problem_name, problem, n_gen, n_runs=30) -> list[np.ndarray]` — identical signatures to `validation.algorithms.nsga3.runner`'s, so `main_validation.py` (Task 4's `--algorithm` dispatch) and `validation.engine.validate()` can call either interchangeably.

- [ ] **Step 1: Write the failing test file first**

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from validation.algorithms.qinsga3.runner import run_single, run_experiment
from validation.dtlz.dtlz_problems import get_problem


def test_run_single_3obj_output_shape():
    problem, _ = get_problem("DTLZ2", n_obj=3)
    front = run_single(problem, n_gen=3, seed=42)
    assert front.ndim == 2
    assert front.shape[1] == 3


def test_run_single_4obj_output_shape():
    problem, _ = get_problem("DTLZ2", n_obj=4)
    front = run_single(problem, n_gen=3, seed=42)
    assert front.ndim == 2
    assert front.shape[1] == 4


def test_run_experiment_returns_one_front_per_run():
    problem, _ = get_problem("DTLZ1", n_obj=3)
    fronts = run_experiment("DTLZ1", problem, n_gen=2, n_runs=2)
    assert len(fronts) == 2
    for f in fronts:
        assert f.shape[1] == 3


if __name__ == "__main__":
    test_run_single_3obj_output_shape()
    print("Testing run_single 3obj (3 gen)... OK")
    test_run_single_4obj_output_shape()
    print("Testing run_single 4obj (3 gen)... OK")
    test_run_experiment_returns_one_front_per_run()
    print("Testing run_experiment (2 gen, 2 runs)... OK")
    print("All tests passed.")
```

Save this as `validation/algorithms/qinsga3/test_runner.py`.

- [ ] **Step 2: Run it to confirm it fails (runner.py doesn't exist yet)**

Run: `cd Project_Irp && ./.venv/Scripts/python.exe -m validation.algorithms.qinsga3.test_runner`
Expected: `ModuleNotFoundError: No module named 'validation.algorithms.qinsga3.runner'`

- [ ] **Step 3: Write `validation/algorithms/qinsga3/runner.py`**

```python
"""QINSGA-III runner for DTLZ/MaF benchmark validation.

Same run_single/run_experiment signatures as
validation/algorithms/nsga3/runner.py, so validation/engine.py's validate()
can call either interchangeably. Reference directions and population size
come from nsga3.runner.get_run_config() — the Cui et al. (2025) protocol,
identical to the classic NSGA-III validation. Only the quantum-operator
parameters below are QINSGA3-specific, copied verbatim from
QINSGA3/main.py's IRP defaults (there is no classic-NSGA-III equivalent to
compare them against).
"""
import numpy as np

from validation.algorithms.nsga3.runner import get_run_config
from validation.algorithms.seeds import SEEDS

from .core import run_qinsga3_generic

ALPHA_MAX = 0.10 * np.pi
ALPHA_MIN = 0.001 * np.pi
P_CROSS = 0.9
ETA_CROSS = 5.0
P_MUT_STRONG = 0.15
MUT_SIGMA = 0.05 * np.pi
MIGRATION_PERIOD = 10
N_MIGRATE = 10
ROTATION_TYPE = "tanh"


def run_single(problem, n_gen: int, seed: int) -> np.ndarray:
    """Run one QINSGA-III optimisation and return the non-dominated objective values.

    Args:
        problem : pymoo Problem instance (n_obj determines ref_dirs and pop_size).
        n_gen   : number of generations.
        seed    : random seed for reproducibility.

    Returns:
        np.ndarray of shape (n_solutions, n_obj).
    """
    ref_dirs, pop_size = get_run_config(problem.n_obj)
    p_mut = 2.0 / problem.n_var

    return run_qinsga3_generic(
        problem, ref_dirs, pop_size, max_gen=n_gen,
        alpha_max=ALPHA_MAX, alpha_min=ALPHA_MIN,
        p_mut=p_mut, p_mut_strong=P_MUT_STRONG, mut_sigma=MUT_SIGMA,
        p_cross=P_CROSS, eta_cross=ETA_CROSS,
        migration_period=MIGRATION_PERIOD, n_migrate=N_MIGRATE,
        seed=seed, rotation_type=ROTATION_TYPE,
    )


def run_experiment(problem_name: str, problem, n_gen: int, n_runs: int = 30) -> list:
    """
    Run QINSGA-III n_runs times with distinct seeds (same seeds as the NSGA-III runner).

    Args:
        problem_name : display name (e.g. "DTLZ1"), used for progress printing.
        problem      : pymoo Problem instance.
        n_gen        : number of generations per run.
        n_runs       : number of independent runs (default 30).

    Returns:
        List of np.ndarray, one per run, shape (n_solutions, n_obj).
    """
    if n_runs > len(SEEDS):
        raise ValueError(f"n_runs={n_runs} exceeds available seeds ({len(SEEDS)})")

    fronts = []
    for i in range(n_runs):
        seed = SEEDS[i]
        print(f"  [{problem_name}] Run {i+1:02d}/{n_runs}  seed={seed}", flush=True)
        front = run_single(problem, n_gen, seed)
        fronts.append(front)
        print(f"  [{problem_name}] Run {i+1:02d} done - front size: {len(front)}", flush=True)
    return fronts
```

- [ ] **Step 4: Run the test file again to confirm it passes**

Run: `cd Project_Irp && ./.venv/Scripts/python.exe -m validation.algorithms.qinsga3.test_runner`
Expected:

```
Testing run_single 3obj (3 gen)... OK
Testing run_single 4obj (3 gen)... OK
Testing run_experiment (2 gen, 2 runs)... OK
All tests passed.
```

- [ ] **Step 5: Commit**

```bash
git add validation/algorithms/qinsga3/runner.py validation/algorithms/qinsga3/test_runner.py
git commit -m "feat: add QINSGA3 runner adapter for DTLZ/MaF validation"
```

---

### Task 8: Wire QINSGA3 into the CLI and smoke-test end to end

**Files:**
- No file changes — `--algorithm qinsga3` in `validation/dtlz/main_validation.py` / `validation/maf/main_validation.py` (Task 4) already dispatches to `validation.algorithms.qinsga3.runner` (Task 7), which now exists.

**Interfaces:**
- Consumes: everything from Tasks 1-7.
- Produces: `validation/dtlz/results/qinsga3/*.csv` and `validation/maf/results/qinsga3/*.csv` written by the CLI.

- [ ] **Step 1: Smoke-test QINSGA3 on DTLZ, M=3, 2 runs, 3 generations worth of budget**

Since `main_validation.py` derives `n_gen` from the problem's own `get_problem()` (326 generations for M=3 in the real protocol), a true "2 runs" smoke test still runs full-length generations — that's fine for a correctness smoke test (no NaNs, sane IGD), just slower than a `--n_obj`-only toggle. Run:

Run: `cd Project_Irp && ./.venv/Scripts/python.exe -m validation.dtlz.main_validation --algorithm qinsga3 --runs 2 --n_obj 3`
Expected: prints `[QINSGA3]`-prefixed... actually no `[QINSGA3]` prefix here (that's the IRP solver's print style) — expect the same progress print style as the NSGA-III runner (`  [DTLZ1] Run 01/2  seed=42`, etc.), followed by a results table, for all of DTLZ1-7. This will take a few minutes (7 problems × 2 runs × up to 326 generations of quantum-loop overhead) — that is expected and acceptable for a one-time smoke test.

- [ ] **Step 2: Inspect one generated CSV for sane values**

Run: `cd Project_Irp && head -5 validation/dtlz/results/qinsga3/igd_DTLZ2_M3.csv`
Expected:

```
run,seed,igd
1,42,<some positive float>
2,137,<some positive float>
```

No `nan`, no negative values.

- [ ] **Step 3: Clean up the 2-run smoke-test artifacts (not the real 30-run data)**

```bash
rm -rf validation/dtlz/results/qinsga3
```

- [ ] **Step 4: Repeat Steps 1-3 for MaF, M=3**

Run: `cd Project_Irp && ./.venv/Scripts/python.exe -m validation.maf.main_validation --algorithm qinsga3 --runs 2 --n_obj 3`
Then inspect `validation/maf/results/qinsga3/igd_MaF2_M3.csv` the same way, then:

```bash
rm -rf validation/maf/results/qinsga3
```

- [ ] **Step 5: No commit for this task** (smoke-test only, no file changes to keep — cleanup already done in Steps 3/4).

---

### Task 9: Run the full QINSGA3 campaign and consolidate results

**Files:**
- Generates: `validation/dtlz/results/qinsga3/*.csv`, `validation/maf/results/qinsga3/*.csv`
- Create: `validation/dtlz/results/qinsga3/DTLZ_results_summary_qinsga3.md`
- Create: `validation/maf/results/qinsga3/MaF_results_summary_qinsga3.md`

**Interfaces:**
- Consumes: everything from Tasks 1-8.
- Produces: the same Markdown report shape already used for NSGA-III (`nsga3/DTLZ_results_summary_nsga3.md` / `nsga3/MaF_results_summary_nsga3.md`) — per-problem Best/Median/Worst/Mean/Std plus all 30 per-run IGD values, one section per problem, M=3 and M=4 subsections, plus a "Récapitulatif global" table at the end.

- [ ] **Step 1: Launch the four full campaigns in the background**

```bash
cd Project_Irp
./.venv/Scripts/python.exe -m validation.dtlz.main_validation --algorithm qinsga3 --n_obj 4 > /tmp/qinsga3_dtlz_m4.log 2>&1 &
./.venv/Scripts/python.exe -m validation.dtlz.main_validation --algorithm qinsga3 --n_obj 3 > /tmp/qinsga3_dtlz_m3.log 2>&1 &
./.venv/Scripts/python.exe -m validation.maf.main_validation  --algorithm qinsga3 --n_obj 4 > /tmp/qinsga3_maf_m4.log 2>&1 &
./.venv/Scripts/python.exe -m validation.maf.main_validation  --algorithm qinsga3 --n_obj 3 > /tmp/qinsga3_maf_m3.log 2>&1 &
wait
```

Expected: all four processes exit 0. This is the slow step — QINSGA3's per-generation overhead (guide selection, archive maintenance, migration) is higher than plain pymoo NSGA-III, so budget real wall-clock time here (likely longer than the NSGA-III campaign's runtime). If it proves impractically slow, reduce run count for QINSGA3 only and say so explicitly in the consolidated `.md` files' header (e.g. "20 runs" instead of "30") rather than silently changing algorithm parameters to speed it up.

- [ ] **Step 2: Verify all expected CSVs exist**

Run: `cd Project_Irp && ls validation/dtlz/results/qinsga3/ && ls validation/maf/results/qinsga3/`
Expected: `igd_DTLZ{1..7}_M3.csv`, `igd_DTLZ{1,2,3,4}_M4.csv`, `summary_M3.csv`, `summary_M4.csv` (DTLZ); `igd_MaF{1..7}_M3.csv`, `igd_MaF{1..7}_M4.csv`, `summary_M3.csv`, `summary_M4.csv` (MaF) — matching exactly the file set already produced for NSGA-III in `results/nsga3/`.

- [ ] **Step 3: Build `DTLZ_results_summary_qinsga3.md`**

Read every `igd_DTLZ*_M*.csv` and `summary_M*.csv` under `validation/dtlz/results/qinsga3/`, and write `validation/dtlz/results/qinsga3/DTLZ_results_summary_qinsga3.md` following the exact same structure as `validation/dtlz/results/nsga3/DTLZ_results_summary_nsga3.md` (per-problem `## DTLZn` sections with `### M3`/`### M4` subsections, each showing the Best/Median/Worst/Mean/Std table followed by the full per-run table, plus a final "Récapitulatif global" section) — only the protocol line at the top changes from "NSGA-III" to "QINSGA3" and from "moteur `validation/algorithms/nsga3_runner.py`" to "moteur `validation/algorithms/qinsga3/runner.py` + `core.py`".

- [ ] **Step 4: Build `MaF_results_summary_qinsga3.md`** the same way, from `validation/maf/results/qinsga3/`.

- [ ] **Step 5: Verify the dashboard now shows both algorithms**

Run: `cd Project_Irp && ./.venv/Scripts/python.exe -c "import app; d = app._build_benchmark_data(); print(sorted(d['dtlz']['DTLZ2']['M3'].keys())); print(sorted(d['maf']['MaF2']['M4'].keys()))"`
Expected: `['nsga3', 'qinsga3']` printed twice.

- [ ] **Step 6: Commit**

```bash
git add validation/dtlz/results/qinsga3 validation/maf/results/qinsga3
git commit -m "data: add QINSGA3 30-run validation results on DTLZ/MaF"
```

---

## Post-plan note

After Task 9, `validation/` has a fully symmetric structure: `algorithms/{nsga3,qinsga3}/` for algorithm code, `{dtlz,maf}/results/{nsga3,qinsga3}/` for results, one shared `engine.py`/`metrics/`/`{dtlz,maf}_problems.py` per suite, and a dashboard that shows both algorithms on every problem card. Extending to a third algorithm later means: add `algorithms/<name>/runner.py` with the same `run_experiment` signature, add `<name>` to `app.py`'s `_BENCHMARK_ALGOS` and `ALGO_LABEL`/`ALGO_ORDER`, and add the CLI's `--algorithm` choice — no other file needs to change.
