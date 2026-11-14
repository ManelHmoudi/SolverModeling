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
├── Solvers/                 The five solvers (see below)
│   ├── ObjectiveCalibration/
│   ├── FunctionMerge/
│   ├── NSGA3/
│   ├── QINSGA3/
│   └── MOEAD/
├── Validation/Benchmarking/  DTLZ/MaF benchmark results (IGD CSVs) displayed by the UI
├── data/                     Problem instances (instance_{N}_clients.json)
├── helpers/                  Synthetic instance generator
├── Livrables_Prof/           Deliverables (Colab notebooks + HTML result reports)
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
| `MOEAD/` | Metaheuristic — MOEA/D (pymoo) | All 4, Pareto front | Decomposition-based comparison baseline — see [`Solvers/MOEAD/README.md`](Solvers/MOEAD/README.md) |

`NSGA3/` and `QINSGA3/` share the same problem definition (`NSGA3/problem.py`,
imported by both, never duplicated) and the same report/caching plumbing
(`NSGA3/report_builder.py`, `NSGA3/report.py`), so their outputs are directly
comparable and rendered through an identical HTML dashboard.

## Benchmark results — `Validation/Benchmarking/`

`{dtlz,maf}/results/{nsga3,qinsga3,moead}/igd_<problem>_M<m>.csv` hold the
per-run IGD values of the DTLZ/MaF benchmark campaigns (Cui et al. 2025
protocol). The UI's benchmarking page reads them directly; the generating
code is available in git history.

## Entry points

- `main.py` → `app.py` — a local Flask menu that launches any solver against
  a selectable instance from `data/` and serves the resulting HTML reports.
- Each solver is also independently runnable from the CLI:
  `python -m Solvers.<Module>.main` (see each module's README/docstring for flags).

## Data

- `data/instance_{N}_clients.json` — problem instances, 3 to 800 clients.
- `helpers/generate_synthetic_instances.py` — regenerates the synthetic
  instances (110–800 clients) with varied periods, fleet and quantities,
  sized so a full truck route respects `tau_max`.
