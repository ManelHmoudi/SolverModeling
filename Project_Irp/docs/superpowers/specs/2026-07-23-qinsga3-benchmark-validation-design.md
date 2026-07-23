# QINSGA3 benchmark validation on DTLZ/MaF — design

## Context

The project already validates classic NSGA-III (pymoo) on the DTLZ1-7 and
MaF1-7 benchmark suites, reproducing the experimental protocol of Cui et al.
(2025) — same population size, generation count (via a fixed 30,000-evaluation
budget), Das-Dennis reference directions, SBX/PM operators, IGD metric with
P*≈10,000 reference points, and 30 independent runs on 30 shared deterministic
seeds. Results live in `validation/dtlz/results/` and `validation/maf/results/`
as per-run CSVs (`igd_{Problem}_M{n}.csv`), per-dimension summary CSVs
(`summary_M{n}.csv`), and a hand-consolidated Markdown report per suite. A
Flask dashboard (`app.py`, route `/benchmarking`) reads those CSVs live and
renders per-problem cards with Best/Median/Worst/Mean/Std and a run-by-run
chart.

QINSGA3 (Quantum-Inspired NSGA-III) is a separate algorithm implemented for
solving the project's real many-objective IRP. Its quantum chromosome
population (`QINSGA3/chromosome.py`, class `QuantumPopulation`) and its
generational helper functions (`QINSGA3/algorithm.py`: `_normalise_F`,
`_assign_ref_dirs`, `_select_guides`, `_supplement_from_archive`, `_migrate`,
`_archive_update`, `_crowding_trim`, `_penalised_F`) are already
problem-agnostic — they operate purely on `X`/`F`/`G`/`theta` NumPy arrays and
bounds `xl`/`xu`. The only IRP coupling is in `run_qinsga3()`'s
`ProcessPoolExecutor` workers, which hardcode `IRPProblem` for parallel
evaluation (needed because IRP evaluation is an expensive simulation).

The goal of this project is to run the **same DTLZ/MaF validation protocol**
against QINSGA3 instead of classic NSGA-III, so the two algorithms can be
compared problem-by-problem, run-by-run, on identical benchmarks — without
touching the existing NSGA-III validation or the production IRP path.

## Goals

- Validate QINSGA3 on DTLZ1-7 and MaF1-7, M=3 and M=4, 30 independent runs,
  using the **same 30 seeds, same population size, same generation count, and
  same reference directions** as the existing NSGA-III validation (Cui et al.
  protocol) — so only the algorithm differs, not the experimental budget.
- Keep QINSGA3's own quantum-operator parameters (rotation step decay,
  crossover, mutation) as already tuned for the IRP in `QINSGA3/main.py`,
  since classic NSGA-III has no equivalent to compare against.
- Reorganize `validation/` so NSGA-III-specific, QINSGA3-specific, and shared
  code/results are clearly separated — current NSGA-III files get moved, not
  just added alongside.
- Extend the `/benchmarking` dashboard to show both algorithms side by side
  per problem, without a visual redesign.
- Zero changes to `QINSGA3/algorithm.py`, `QINSGA3/main.py`, or any
  IRP-production code path.

## Non-goals

- No changes to how QINSGA3 solves the real IRP (pop=200, gen=300, IRP
  chromosome encoding, multiprocessing evaluation) — untouched.
- No new visual design for the dashboard — same layout, an added
  algorithm-comparison block per problem card.
- No automated statistical significance testing (Wilcoxon, etc.) between
  algorithms — out of scope for this pass; Best/Median/Worst/Mean/Std side by
  side is enough for now.

## File/folder reorganization

```
validation/
├── engine.py                        # SHARED — run/save/print loop, agnostic to suite AND algorithm
├── metrics/
│   ├── igd_metric.py                 # SHARED
│   └── test_igd_metric.py
├── algorithms/
│   ├── seeds.py                      # SHARED — the 30 deterministic seeds (single source of truth)
│   ├── nsga3/
│   │   ├── __init__.py
│   │   ├── runner.py                  # classic NSGA-III (pymoo) — moved from algorithms/nsga3_runner.py
│   │   └── test_runner.py             # moved from algorithms/test_nsga3_runner.py
│   └── qinsga3/
│       ├── __init__.py
│       ├── core.py                    # new: lean generational loop, no multiprocessing
│       ├── runner.py                  # new: run_experiment() adapter matching nsga3/runner.py's signature
│       └── test_runner.py             # new
├── dtlz/
│   ├── dtlz_problems.py               # SHARED (both algorithms test the same problems)
│   ├── test_dtlz_problems.py
│   ├── main_validation.py             # SHARED CLI — new --algorithm {nsga3,qinsga3} flag (default nsga3)
│   └── results/
│       ├── nsga3/                     # igd_DTLZx_My.csv, summary_My.csv — MOVED from results/ directly
│       ├── qinsga3/                   # same filenames, new algorithm
│       ├── DTLZ_results_summary_nsga3.md    # renamed from DTLZ_results_summary.md
│       └── DTLZ_results_summary_qinsga3.md  # new
└── maf/                                # mirrors dtlz/ exactly
    ├── maf_problems.py
    ├── test_maf_problems.py
    ├── main_validation.py
    └── results/
        ├── nsga3/
        ├── qinsga3/
        ├── MaF_results_summary_nsga3.md
        └── MaF_results_summary_qinsga3.md
```

Rationale: `algorithms/nsga3/` and `algorithms/qinsga3/` are symmetric, so
anyone reading the tree immediately sees "one folder per algorithm, shared
code lives above them." `seeds.py` is extracted from `nsga3_runner.py` (where
it currently lives as `_SEEDS`) so both algorithms import the exact same list
— no risk of the two runners silently drifting to different seeds.

## Components

### `validation/algorithms/seeds.py` (new)

Just the 30-seed list, extracted verbatim from the current
`nsga3_runner.py`. Both `nsga3/runner.py` and `qinsga3/runner.py` import
`SEEDS` from here.

### `validation/algorithms/nsga3/runner.py` (moved, no logic changes)

Exact move of the current `validation/algorithms/nsga3_runner.py`, with the
`_SEEDS` definition replaced by `from ..seeds import SEEDS`. All existing
tests must still pass unchanged (import path updated).

### `validation/algorithms/qinsga3/core.py` (new)

A generational loop mirroring `QINSGA3/algorithm.py::run_qinsga3()`, but:

- No `ProcessPoolExecutor` / worker processes — evaluates the full population
  each generation via `F, G = problem.evaluate(X, return_values_of=["F", "G"])`
  directly (DTLZ/MaF evaluation is cheap and already vectorized in pymoo;
  multiprocessing overhead would only slow this down).
- Reuses, by import, the already problem-agnostic helpers from
  `QINSGA3/algorithm.py`: `_normalise_F`, `_assign_ref_dirs`,
  `_select_guides`, `_supplement_from_archive`, `_migrate`,
  `_archive_update`, `_crowding_trim`, `_penalised_F`. These are not
  duplicated — imported directly, so any future bugfix to the shared math
  benefits both the IRP path and the benchmark path automatically.
- Uses `QINSGA3.chromosome.QuantumPopulation` directly (already generic).
- Same generation order as the IRP algorithm: measure → evaluate → penalise →
  non-dominated sort → archive update → normalise → assign ref dirs → select
  guides → supplement from archive → rotate → crossover → mutate → migrate.
- Returns the final external archive (trimmed to `pop_size` by crowding
  distance), exactly like `run_qinsga3()` does — so it plugs into
  `compute_igd()` the same way an NSGA-III front does.

### `validation/algorithms/qinsga3/runner.py` (new)

Mirrors `validation/algorithms/nsga3/runner.py`'s public shape exactly:

```python
def get_run_config(n_obj: int): ...     # delegates to nsga3.runner.get_run_config
                                          # (same ref_dirs/pop_size, Cui et al. protocol)

def run_single(problem, n_gen: int, seed: int) -> np.ndarray: ...
def run_experiment(problem_name, problem, n_gen: int, n_runs: int = 30) -> list: ...
```

`run_single` builds `xl`/`xu` from the pymoo `problem` (DTLZ/MaF problems
already expose these), gets `ref_dirs`/`pop_size` from
`nsga3.runner.get_run_config(problem.n_obj)` (so both algorithms use
identical reference directions and population sizes), and calls
`qinsga3.core`'s loop with QINSGA3's own quantum-operator defaults, copied
from `QINSGA3/main.py`: `alpha_max=0.10*pi`, `alpha_min=0.001*pi`,
`p_cross=0.9`, `eta_cross=5.0`, `p_mut=2.0/problem.n_var`,
`p_mut_strong=0.15`, `mut_sigma=0.05*pi`, `migration_period=10`,
`n_migrate=10`, `rotation_type="tanh"`.

Because `run_experiment` has the exact same signature as the NSGA-III
runner's, `validation/engine.py`'s `validate()` can call either one
interchangeably.

### `validation/engine.py` (modified, backward-compatible)

One new optional parameter on `validate()`:

- `run_experiment_fn` (default: `nsga3.runner.run_experiment`) — which
  algorithm's `run_experiment` to call.

`validate()` keeps taking `results_dir` as-is — the caller (suite's
`main_validation.py`) is responsible for pointing it at the
already-algorithm-specific directory (`results/nsga3` or `results/qinsga3`
based on the `--algorithm` flag) and passing the matching
`run_experiment_fn`. This keeps `engine.py` simple: it never needs to know
algorithm names, just "a results dir and a callable".

No existing call sites need to change behavior: the two existing calls in
`dtlz/main_validation.py` / `maf/main_validation.py` keep working once those
files are updated to pass `results/nsga3` explicitly (see below) — this is
the one behavior-visible change (output path), covered in Migration steps.

### `validation/dtlz/main_validation.py` / `validation/maf/main_validation.py` (modified)

New `--algorithm {nsga3,qinsga3}` argument, default `nsga3` (so running the
script with no flags behaves exactly as before, except results now land in
`results/nsga3/` instead of `results/`). Based on the flag, the script:

- picks `validation.algorithms.nsga3.runner` or
  `validation.algorithms.qinsga3.runner` as the `run_experiment_fn` source,
- points `results_dir` at `results/nsga3` or `results/qinsga3`.

### Dashboard (`app.py`)

- `_BENCHMARK_SUITES["dtlz"/"maf"]["results_dir"]` now points at the suite
  root (`.../dtlz/results`); `_read_igd_runs` is updated to read from
  `{results_dir}/{algo}/igd_{problem}_M{n}.csv` for `algo in ("nsga3",
  "qinsga3")`.
- `_build_benchmark_data()` returns, per `(suite, problem, "M{n}")`, a dict
  keyed by algorithm: `{"nsga3": {...stats...}, "qinsga3": {...stats...}}`
  (a problem/M combination with only one algorithm's CSVs present just omits
  the missing key — the front end shows "no data yet" for that half, same
  message style as today).
- Front end: each problem card gets a second column/table for QINSGA3 next
  to the existing NSGA-III one, using the same Best/Median/Worst/Mean/Std
  layout already in place — no new chart types, no layout redesign.

## Data flow

```
main_validation.py --algorithm qinsga3
        │
        ▼
engine.validate(problems_module, results_dir="results/qinsga3",
                run_experiment_fn=qinsga3.runner.run_experiment, n_runs=30, n_obj=...)
        │
        ▼  (per problem)
qinsga3.runner.run_experiment(name, problem, n_gen, n_runs=30)
        │  for each of the 30 shared seeds:
        ▼
qinsga3.runner.run_single(problem, n_gen, seed)
        │  ref_dirs, pop_size = nsga3.runner.get_run_config(problem.n_obj)
        ▼
qinsga3.core.run_qinsga3_generic(problem, ref_dirs, pop_size, n_gen, seed, <quantum params>)
        │  (QuantumPopulation + shared helpers from QINSGA3/algorithm.py)
        ▼
returns final archive (X, F) → compute_igd(problem, F) in engine.py, same as NSGA-III today
        │
        ▼
igd_{Problem}_M{n}.csv, summary_M{n}.csv written under results/qinsga3/
```

## Migration steps (existing NSGA-III files/results)

1. Move `validation/algorithms/nsga3_runner.py` →
   `validation/algorithms/nsga3/runner.py` (extract `_SEEDS` into
   `algorithms/seeds.py` first, update the import).
2. Move `validation/algorithms/test_nsga3_runner.py` →
   `validation/algorithms/nsga3/test_runner.py`, update imports.
3. Move existing result CSVs: `validation/dtlz/results/*.csv` →
   `validation/dtlz/results/nsga3/*.csv` (same for `maf/`).
4. Rename `DTLZ_results_summary.md` → `DTLZ_results_summary_nsga3.md` (same
   for MaF) — content unchanged.
5. Update `app.py`'s `_read_igd_runs`/`_build_benchmark_data` for the new
   nested path and per-algorithm keys.
6. Update `dtlz/main_validation.py` / `maf/main_validation.py` for the new
   `--algorithm` flag and results subfolder.
7. Run the existing NSGA-III test suite + a quick `--runs 2` smoke run for
   both suites to confirm nothing broke from the move.

## Testing

- `validation/algorithms/qinsga3/test_runner.py`: mirrors
  `nsga3/test_runner.py`'s test shapes — `run_single` returns an
  `(n_solutions, n_obj)` array for both M=3 and M=4 on a cheap problem
  (DTLZ2, `n_gen=3`), `run_experiment` with `n_runs=2` returns 2 fronts.
- Manual smoke test before the full 30-run campaign: `--algorithm qinsga3
  --runs 2 --n_obj 3` on one suite, inspect the printed summary table and the
  generated CSV for sane (non-NaN, non-empty) values.
- After migration, re-run `validation/algorithms/nsga3/test_runner.py` and
  the existing `dtlz`/`maf` test files to confirm the move didn't break
  anything.
- Full campaign: 30 runs × (M=3, M=4) × (DTLZ, MaF) × qinsga3, same shape as
  the NSGA-III campaign already completed — run in background, consolidate
  into `*_qinsga3.md` files the same way the NSGA-III ones were built.

## Open risk / thing to watch

QINSGA3's generational loop does more per-generation NumPy work than plain
pymoo NSGA-III (guide selection, archive maintenance, migration), so wall-clock
time per run may be noticeably higher even without the multiprocessing
overhead removed here. If the full 30-run × 4-grid campaign turns out to be
too slow, the fallback is reducing run count for QINSGA3 only (documented
explicitly in the results, not silently) rather than changing the algorithm's
parameters to be faster.
