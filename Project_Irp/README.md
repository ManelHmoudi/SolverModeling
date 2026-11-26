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
│   └── replicates/              Independent replicates per size (instance_{N}_clients_r{r}.json)
├── docs/
│   ├── model_formulation.md     Full formulation (C1–C19, f1–f4) + answers to the editor's remarks
│   └── modele_IRP_corrige.docx  The same model in French, with Word equations (for the paper)
├── tests/                       Independent solution checker + model tests (pytest)
├── scripts/                     Instance generator (+ --migrate, --replicates), ablation, plan export
└── benchmarks/                  DTLZ/MaF benchmark results (IGD CSVs) shown by the UI
    ├── dtlz/{nsga3,qinsga3,moead}/igd_<problem>_M<m>.csv
    └── maf/{nsga3,qinsga3,moead}/igd_<problem>_M<m>.csv
```

**Naming convention:** every folder and file name is English, lowercase,
`snake_case` (PEP 8). Only `README.md` keeps its conventional capitals.

## The problem

A single depot supplies a set of clients over several periods, using a fleet
of refrigerated/non-refrigerated trucks, under stock, time-window and
routing constraints. Four conflicting objectives are optimised jointly:

| Objective | Description |
|---|---|
| **f1** | Logistics cost (€): dispatch + distance per vehicle-km (empty return included) + load term + depot holding + time-window penalties + supplier ordering |
| **f2** | CO₂ emissions (kg), linearised CMEM (Bektas & Laporte 2011) |
| **f3** | Longest route of each period, summed over the periods (h): workload of the most loaded driver |
| **f4** | Peak net working capital (€): stock value + receivables − payables per period, cyclic accounting |

Supplier orders are decisions of the model. The full formulation, with the editor's
remarks and the audit findings it answers, is in
[`docs/model_formulation.md`](docs/model_formulation.md). It is checked by
`python -m pytest tests -q`.

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
