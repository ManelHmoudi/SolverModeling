# Project Architecture

**Many-Objective Inventory Routing Problem (IRP)** — optimisation project comparing
exact, scalarised, and metaheuristic solvers, including a quantum-inspired
NSGA-III variant developed in this project.

This is a high-level map. For algorithm-specific detail (steps, parameters,
literature references), see each module's own `README.md`.

## Layout

```
Project_Irp/
├── main.py                      Launcher (starts the local Flask UI)
├── app.py                       Local Flask UI
├── requirements.txt
├── models/                      Shared problem definition
│   ├── parameters.py            Loads an instance JSON into sets/params
│   ├── variables.py
│   ├── constraints.py
│   └── objectives.py
├── solvers/                     The five solvers (see below)
│   ├── objective_calibration/
│   ├── function_merge/
│   ├── nsga3/
│   ├── qinsga3/
│   └── moead/
├── data/                        Problem instances (instance_{N}_clients.json)
├── scripts/                     Utility scripts (synthetic instance generator)
├── benchmarks/                  DTLZ/MaF benchmark results (IGD CSVs) shown by the UI
│   ├── dtlz/{nsga3,qinsga3,moead}/igd_<problem>_M<m>.csv
│   └── maf/{nsga3,qinsga3,moead}/igd_<problem>_M<m>.csv
└── deliverables/                Deliverables (Colab notebooks + HTML result reports)
    ├── irp_100_clients/
    │   ├── nsga3_vs_qinsga3_colab.ipynb
    │   ├── results.html
    │   ├── results_30_seeds.html
    │   └── results_delivery_shift.html
    └── benchmarks_dtlz_maf/
        ├── nsga3_vs_qinsga3_colab.ipynb
        └── results.html
```

**Naming convention:** every folder and file name is English, lowercase,
`snake_case` (PEP 8). Only `README.md` keeps its conventional capitals.

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

`models/` defines the shared problem data model: `parameters.py` loads an
instance JSON into `sets_`/`params_` dicts; `constraints.py`, `objectives.py`
and `variables.py` document the exact math (constraints C1–C18, objectives
f1–f4) independently of any specific solver.

## Solvers — `solvers/`

| Module | Approach | Objectives | Notes |
|---|---|---|---|
| `objective_calibration/` | Exact (CPLEX/DOcplex), one objective at a time | f1..f4 individually | Establishes each objective's own best/worst bound (ideal/nadir anchors) |
| `function_merge/` | Exact (CPLEX/DOcplex), scalarised | min(f1+f2+f3+f4) | Single combined objective — a baseline, not Pareto-aware |
| `nsga3/` | Metaheuristic — NSGA-III (pymoo) | All 4, Pareto front | Reference many-objective solver — see [`solvers/nsga3/README.md`](solvers/nsga3/README.md) |
| `qinsga3/` | Metaheuristic — quantum-inspired NSGA-III | All 4, Pareto front | This project's contribution — see [`solvers/qinsga3/README.md`](solvers/qinsga3/README.md) |
| `moead/` | Metaheuristic — MOEA/D (pymoo) | All 4, Pareto front | Decomposition-based comparison baseline — see [`solvers/moead/README.md`](solvers/moead/README.md) |

`nsga3/` and `qinsga3/` share the same problem definition (`nsga3/problem.py`,
imported by both, never duplicated) and the same report/caching plumbing
(`nsga3/report_builder.py`, `nsga3/report.py`), so their outputs are directly
comparable and rendered through an identical HTML dashboard.

## Benchmark results — `benchmarks/`

`{dtlz,maf}/{nsga3,qinsga3,moead}/igd_<problem>_M<m>.csv` hold the
per-run IGD values of the DTLZ/MaF benchmark campaigns (Cui et al. 2025
protocol). The UI's benchmarking page reads them directly; the generating
code is available in git history.

## Entry points

- `main.py` → `app.py` — a local Flask menu that launches any solver against
  a selectable instance from `data/` and serves the resulting HTML reports.
- Each solver is also independently runnable from the CLI:
  `python -m solvers.<module>.main` (see each module's README/docstring for flags).

## Data

- `data/instance_{N}_clients.json` — problem instances, 3 to 800 clients.
- `scripts/generate_synthetic_instances.py` — regenerates the synthetic
  instances (110–800 clients) with varied periods, fleet and quantities,
  sized so a full truck route respects `tau_max`.
