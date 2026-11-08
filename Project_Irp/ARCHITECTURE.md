# Project Architecture

**Many-Objective Inventory Routing Problem (IRP)** — optimisation project comparing
exact, scalarised, and metaheuristic solvers, including a quantum-inspired
NSGA-III variant developed in this project.

This is a high-level map. For algorithm-specific detail (steps, parameters,
literature references), see each module's own `README.md`.

## Layout

```
Project_Irp/
├── models/                  Shared problem definition (sets, params, constraints, objectives)
├── Solvers/                 The four solvers (see below)
│   ├── ObjectiveCalibration/
│   ├── FunctionMerge/
│   ├── NSGA3/
│   └── QINSGA3/
├── Validation/               Validation against external literature, not the IRP itself
│   ├── Benchmarking/         DTLZ/MaF synthetic benchmarks (Cui et al. 2025 protocol)
│   └── Archetti/             Archetti (2007) IRP benchmark instances
├── sensitivity/              Final comparison campaigns (2-opt fairness, 3 algorithms) + their reference logs
├── data/                     Problem instances (instance_{N}_clients.json)
├── Livrables_Prof/           Deliverables (Colab notebooks + HTML result reports)
├── docs/superpowers/         Design specs, plans, and history of explored mechanisms
├── app.py / main.py          Local Flask UI and its launcher
└── requirements.txt
```

## The problem

A single depot supplies a set of clients over several periods, using a fleet
of refrigerated/non-refrigerated trucks, under stock, time-window and
routing constraints. Four conflicting objectives are optimised jointly:

| Objective | Description |
|---|---|
| **f1** | Logistics cost (transport + depot holding + time-window penalties) |
| **f2** | CO₂ emissions (CMEM model, Bektas & Laporte 2011) |
| **f3** | Total travel time |
| **f4** | Working capital / BFR (stock + receivables − payables) |

`models/` defines the shared problem data model: `parametres.py` loads an
instance JSON into `sets_`/`params_` dicts; `constraints.py`, `objectives.py`
and `variables.py` document the exact math (constraints C1–C18, objectives
f1–f4) independently of any specific solver.

## Solvers — `Solvers/`

| Module | Approach | Objectives | Notes |
|---|---|---|---|
| `ObjectiveCalibration/` | Exact (CPLEX/DOcplex), one objective at a time | f1..f4 individually | Establishes each objective's own best/worst bound (ideal/nadir anchors) |
| `FunctionMerge/` | Exact (CPLEX/DOcplex), scalarised | min(f1+f2+f3+f4) | Single combined objective — a baseline, not Pareto-aware |
| `NSGA3/` | Metaheuristic — NSGA-III (pymoo) | All 4, Pareto front | Reference many-objective solver — see [`Solvers/NSGA3/README.md`](Solvers/NSGA3/README.md) |
| `QINSGA3/` | Metaheuristic — quantum-inspired NSGA-III | All 4, Pareto front | This project's contribution — see [`Solvers/QINSGA3/README.md`](Solvers/QINSGA3/README.md) |

`NSGA3/` and `QINSGA3/` share the same problem definition (`NSGA3/problem.py`,
imported by both, never duplicated) and the same report/caching plumbing
(`NSGA3/report_builder.py`, `NSGA3/report.py`), so their outputs are directly
comparable and rendered through an identical HTML dashboard.

## Validation — `Validation/`

Validates the algorithms themselves against published external results —
distinct from running them on the IRP.

| Module | Validates against | Method |
|---|---|---|
| `Benchmarking/` | DTLZ/MaF synthetic many-objective suites (Cui et al. 2025 protocol) | Runs NSGA-III and QI-NSGA-III via benchmark-only copies of the generational loops (`Benchmarking/algorithms/{nsga3,qinsga3}/`), kept separate from `Solvers/` so IRP solvers are never affected. `Benchmarking/engine.py` is the shared run/save/print harness; `dtlz/` and `maf/` each add their own problem set + CLI. Results live under `Benchmarking/{dtlz,maf}/results/`. |
| `Archetti/` | Archetti (2007) published IRP instances, Vadseth (2021) reference solutions | Exact CPLEX solving of the mono-objective cost function (f1) |

## Experimentation — `sensitivity/`

Only the final comparison campaigns are kept, always evaluated against a
**shared** ideal/nadir so results are comparable across algorithms:

- `compare_2opt_fairness.py` — main campaign (NSGA-III vs QI-NSGA-III, with/without 2-opt, budget-matched).
- `compare_3algos.py` — NSGA-III vs QI-NSGA-III vs MOEA/D.
- `compare_qinsga3_vs_moead.py`, `compare_qinsga3_vs_nsga3.py` — pairwise head-to-head tests.
- `*_log.txt` — raw logs of the campaigns reported in `Livrables_Prof/`.

One-off tuning/diagnostic scripts were removed once their question was
answered; the findings are in `Solvers/IRP_results_summary.md` and
`docs/superpowers/`, and the scripts remain available in git history.

## Entry points

- `main.py` → `app.py` — a local Flask menu that launches any solver against
  a selectable instance from `data/` and serves the resulting HTML reports.
- Each solver is also independently runnable from the CLI:
  `python -m Solvers.<Module>.main` (see each module's README/docstring for flags).

## Data & history

- `data/instance_{N}_clients.json` — problem instances, 3 to 200 clients.
- `docs/superpowers/` — design specs, implementation plans, and dated
  records of what's been tried and why, including mechanisms not adopted —
  a durable history that goes beyond what's currently in the code.
