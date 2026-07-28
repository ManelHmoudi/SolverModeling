# Project architecture — Many-Objective Inventory Routing Problem (IRP)

Map of the project. For algorithm-specific detail (steps,
parameters, references).

## The problem

A single depot supplies a set of clients over several periods, using a
fleet of refrigerated/non-refrigerated trucks, under stock, time-window and
routing constraints. Four conflicting objectives are optimised jointly:

- **f1** — logistics cost (transport + depot holding + time-window penalties)
- **f2** — CO₂ emissions (CMEM model, Bektas & Laporte 2011)
- **f3** — total travel time
- **f4** — working capital (BFR: stock + receivables − payables)

`models/` defines the shared problem data model (`parametres.py` loads an
instance JSON into `sets_`/`params_` dicts; `constraints.py`/`objectives.py`/
`variables.py` document the exact math, C1–C18 and f1–f4, independently of
any specific solver).

## Solvers

| Module | Approach | Objectives handled | Notes |
|---|---|---|---|
| `ObjectiveCalibration/` | Exact (CPLEX/DOcplex), one objective at a time | f1..f4 individually | Establishes each objective's own best/worst bound (ideal/nadir anchors). |
| `FunctionMerge/` | Exact (CPLEX/DOcplex), scalarised | min(f1+f2+f3+f4) | Single combined objective — a baseline, not Pareto-aware. |
| `NSGA3/` | Metaheuristic — NSGA-III (pymoo) | All 4, Pareto front | Reference many-objective solver; see `NSGA3/README.md`. |
| `NSGA3/` | Metaheuristic — quantum-inspired NSGA-III (this project's contribution) | All 4, Pareto front | Quantum-angle representation + rotation gate, combined with NSGA-III's own elitist reference-direction survival and real-coded (SBX/PM) variation; see `QINSGA3/README.md`. |

`NSGA3/` and `QINSGA3/` share the same problem definition (`NSGA3/problem.py`,
imported by both) and the same report/caching plumbing
(`NSGA3/report_builder.py`, `NSGA3/report.py`), so their outputs are directly
comparable and rendered through an identical HTML dashboard.

## Validation (against literature, not the IRP itself)

- `validation/` — runs NSGA-III and QI-NSGA-III on standard synthetic
  many-objective benchmark suites (DTLZ, MaF) to validate the algorithms
  themselves against published results, independently of the IRP.
  `validation/algorithms/{nsga3,qinsga3}/` hold benchmark-only copies of the
  generational loops (kept separate from `NSGA3/`/`QINSGA3/` so the real IRP
  solvers are never affected by benchmark-only experiments);
  `validation/engine.py` is the shared run/save/print harness;
  `validation/dtlz/` and `validation/maf/` each add their own problem set +
  CLI. Results (CSVs + `*_results_summary_*.md`) live under
  `validation/{dtlz,maf}/results/`.
- `validation_archetti/` — validates the mono-objective cost function (f1)
  against Archetti (2007)'s published IRP benchmark instances and Vadseth
  (2021)'s reference solutions, via exact CPLEX solving.

## Experimentation

- `sensitivity/` — one-off diagnostic/tuning scripts (parameter sweeps,
  A/B comparisons between algorithm variants), always evaluated against a
  **shared** ideal/nadir so results are comparable across runs/algorithms.
  `compare_qinsga3_vs_nsga3.py` is the main head-to-head comparison; the
  others test individual hyperparameters or mechanism changes. These are
  throwaway/diagnostic by nature — scripts here get deleted once their
  question is answered and the finding is either integrated into production
  or recorded as a rejected path.

## App / entry points

- `main.py` → `app.py`: a local Flask menu that launches each module (exact
  solvers, NSGA-III, QI-NSGA-III) against a selectable instance from `data/`
  and serves the resulting HTML reports.
- Each solver module is also independently runnable from the CLI via
  `python -m <Module>.main` (see each module's own README/docstring for
  flags).

## Data & docs

- `data/instance_{N}_clients.json` — problem instances at various sizes
  (3 to 100 clients).
- `docs/superpowers/` — design specs, implementation plans, and dated
  records of mechanisms that were tried and their outcome (including ones
  not adopted) — a durable history of what's been explored beyond what's
  currently in the code.
