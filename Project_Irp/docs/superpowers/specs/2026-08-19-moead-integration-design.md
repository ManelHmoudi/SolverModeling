# MOEA/D integration — design spec

Date: 2026-08-19
Status: approved by user, pending implementation plan

## Context

This project already compares two algorithms on the many-objective IRP
(Tunisian pharmaceutical cold-chain inventory routing, 4 objectives: cost,
CO2, time, working capital/BFR): NSGA-III (`Solvers/NSGA3/`, pymoo's own
implementation) and QI-NSGA-III (`Solvers/QINSGA3/`, this project's
quantum-inspired theta-encoded variant). Both are wired into the Flask app
(`app.py`), share a decode/repair/report pipeline
(`Solvers/NSGA3/{problem,decoder,evaluator,metrics,report_builder,report}.py`),
have dedicated `sensitivity/` comparison campaign scripts, and have their
own runners in the DTLZ/MaF benchmark suite (`Validation/Benchmarking/`).

The user wants to add a third algorithm, MOEA/D (Multi-Objective
Evolutionary Algorithm based on Decomposition), compared against QI-NSGA-III
the same way NSGA-III already is — both on the IRP and on the DTLZ/MaF
benchmarks, with full app integration.

## Goals

- A new `Solvers/MOEAD/` module producing IRP results through the exact
  same decode/repair/report pipeline NSGA-III and QI-NSGA-III already use
  (so the current production defaults — delivery-shift on, 2-opt off —
  apply to it automatically, with no separate wiring).
- Full app.py/GUI integration: a third report button, alongside the
  existing NSGA-III and QI-NSGA-III ones, generating its own HTML report
  via the already-shared `Solvers/NSGA3/report.py`.
- A dedicated `sensitivity/` campaign script comparing QI-NSGA-III against
  MOEA/D, following the same statistical protocol already validated for
  QI-NSGA-III vs NSGA-III (paired Wilcoxon primary, Mann-Whitney secondary,
  Brown-Forsythe dispersion test, Holm-Bonferroni correction across the 4
  indicators, empirical leave-one-run-out reference front for GD/IGD).
- A DTLZ/MaF benchmark runner for MOEA/D, mirroring the existing NSGA-III
  benchmark runner (same Cui et al. 2025 protocol, same problems).
- Tests mirroring the existing NSGA-III/QI-NSGA-III test patterns.

## Non-goals

- No changes to NSGA-III's or QI-NSGA-III's own algorithm code.
- No reimplementation of MOEA/D — use pymoo's own
  `pymoo.algorithms.moo.moead.MOEAD` exactly as `Solvers/NSGA3/main.py`
  uses pymoo's own `NSGA3`, not a custom/rewritten variant.
- No changes to `sensitivity/compare_2opt_fairness.py`'s existing
  2-algorithm structure — a new, separate script handles the 3rd
  algorithm instead of generalizing the existing one.
- No new HTML pages in `Livrables_Prof/` as part of this implementation —
  those get built after real comparison campaigns are run, as a separate,
  later step (same pattern already used for QI-NSGA-III vs NSGA-III: code
  first, campaigns second, deliverable pages third).

## Architecture

New module `Solvers/MOEAD/`:
- `main.py` — `run_moead()`, `render_from_instance()`,
  `run_moead_report()`, mirroring `Solvers/NSGA3/main.py`'s own three
  entry points exactly (same signature shape, same caching-to-JSON
  pattern for `render_from_instance` refresh support).
- No new `report.py` — reuses `Solvers/NSGA3/report.py` unchanged
  (already reused as-is for QI-NSGA-III; same reuse here).
- No new `problem.py`/`decoder.py`/`evaluator.py`/`metrics.py` —
  reuses the existing `Solvers/NSGA3/` versions of all four (same
  `IRPProblem`, same decode/repair/metric logic every algorithm in this
  project already shares).
- No new repair-layer code — goes through the already-shared
  `Solvers/NSGA3/report_builder.py::_evaluate_pareto` /
  `Solvers/QINSGA3/repair.py::_repair_route_result`, so the current
  production defaults (`use_two_opt=False`, `use_delivery_shift=True`,
  set earlier this session) apply automatically with zero additional
  wiring.

## MOEA/D algorithm wiring

`pymoo.algorithms.moo.moead.MOEAD(ref_dirs, n_neighbors=20,
decomposition=None, prob_neighbor_mating=0.9, sampling=..., crossover=...,
mutation=...)`. Wiring:
- `ref_dirs`: same Das-Dennis scheme already used for NSGA-III and
  QI-NSGA-III (`get_reference_directions("das-dennis", 4,
  n_partitions=8)` → 165 directions for the project's 4 objectives) — kept
  identical across all three algorithms for consistency, not tuned
  per-algorithm.
- `crossover`/`mutation`: SBX/PM with the same eta/probability defaults
  NSGA-III and QI-NSGA-III already use, for a fair operator-level
  comparison.
- `n_neighbors`/`prob_neighbor_mating`: pymoo's own defaults (20 / 0.9) —
  not tuned as part of this integration; can be revisited later as its
  own ablation if warranted.
- **Population size constraint (confirmed, accepted by user)**: unlike
  NSGA-III/QI-NSGA-III, which take an independent `pop_size` parameter
  (currently 200, only overridden upward if `len(ref_dirs)` would exceed
  it), MOEA/D's population size is NOT independent — it equals
  `len(ref_dirs)` by construction (one individual per decomposed
  subproblem). With the shared 165-direction Das-Dennis scheme, MOEA/D's
  population is therefore **165, not 200**. This is documented explicitly
  everywhere MOEA/D results are reported (reports, campaign logs, any
  future `Livrables_Prof/` pages) as a structural property of the
  algorithm, not an arbitrary or hidden choice. Rejected alternative:
  tuning a MOEA/D-specific partition count to approach 200 would break
  the shared reference-direction scheme's consistency across all three
  algorithms — decided against.

## App integration (`app.py`)

- New import: `from Solvers.MOEAD.main import DEFAULT_REPORT_PATH as
  MOEAD_REPORT_PATH, run_moead_report, render_from_instance as
  moead_render_from_instance` (mirrors the existing NSGA3/QINSGA3 import
  lines).
- New Flask routes mirroring the existing `run_nsga3_route`/`nsga3_report`
  pair: a run route and a report route for MOEA/D, using
  `MOEAD_REPORT_PATH`.
- New button in the embedded HTML template, alongside the existing
  `n3-report`/`qi3-report` buttons (`id="moead-report"`), and the matching
  entry in the report-path JS array (mirrors lines 1417-1418's pattern).
- The report-refresh-on-instance-change code path (around the existing
  `nsga3_report`/`qinsga3_report` cache-refresh handling) gets the same
  treatment for MOEA/D.

## Comparison script (`sensitivity/`)

New `sensitivity/compare_qinsga3_vs_moead.py`, structurally mirroring
`sensitivity/compare_2opt_fairness.py`'s statistical machinery (paired
Wilcoxon primary + effect sizes + bootstrap CI, Mann-Whitney secondary,
Brown-Forsythe dispersion, Holm-Bonferroni correction across HV/GD/IGD/
Spacing, empirical LORO reference front) but comparing QI-NSGA-III against
MOEA/D instead of NSGA-III — a new, independent script rather than a
generalization of the existing one, consistent with this project's
existing pattern of one dedicated script per pairwise comparison.
Exact CLI surface (which ablation flags, if any, get exposed) is left to
the implementation plan.

## Benchmark integration (`Validation/Benchmarking/`)

New `Validation/Benchmarking/algorithms/moead/` package, mirroring
`algorithms/nsga3/`'s structure: a `runner.py` using pymoo's `MOEAD`
directly against the existing DTLZ/MaF problem definitions, same Cui et
al. (2025) protocol (SBX eta=20 pc=1.0, PM eta=20 pm=1/D, same Das-Dennis
partition table per `n_obj`, same fixed 30000-evaluation budget). No
archive (matches NSGA-III's current no-archive production default in the
benchmark suite).

## Testing

- `Solvers/MOEAD/test_main.py` — end-to-end smoke test on a tiny instance,
  mirroring `Solvers/NSGA3/test_main.py`'s / `Solvers/QINSGA3/test_main.py`'s
  own `repair_final_front` flag test.
- `Validation/Benchmarking/algorithms/moead/test_runner.py` — mirrors
  `algorithms/nsga3/test_runner.py` (ref_dirs count, population size
  =165 not 200, output shape, `run_experiment` returns a list,
  `n_obj=5` unsupported).
- No changes needed to `Solvers/QINSGA3/test_repair.py` or
  `test_algorithm.py` — MOEA/D only reaches the repair layer through the
  already-shared, already-tested `_evaluate_pareto`/`_repair_route_result`
  path.
- Full project test suite re-run at the end (151 passing before this
  work) to confirm zero regressions plus the new tests passing.

## Open items for the implementation plan

- Exact Flask route names/URLs for MOEA/D (follow existing NSGA3/QINSGA3
  naming convention).
- Exact CLI flags for `compare_qinsga3_vs_moead.py` (how much of
  `compare_2opt_fairness.py`'s ablation-flag surface, if any, is relevant
  to a QI-NSGA-III vs MOEA/D comparison specifically).
- Whether MOEA/D's own report/HTML output needs any MOEA/D-specific
  wording changes in `Solvers/NSGA3/report.py`'s template, or whether the
  existing algorithm-name-parameterized template already covers it
  (verify during implementation).
