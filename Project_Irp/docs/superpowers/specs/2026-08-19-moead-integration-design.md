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
- No reimplementation of MOEA/D's core algorithm — reuse pymoo's own
  `pymoo.algorithms.moo.moead.MOEAD` structure (neighborhoods, weight
  vectors, crossover/mutation, decomposition scalarization) unchanged.
  The one exception, forced by a constraint-support gap described below,
  is a subclass overriding only the replacement step's comparison rule
  — not a rewrite of the algorithm.
- No changes to `IRPProblem` (`Solvers/NSGA3/problem.py`) — its
  constraint declaration (`n_ieq_constr`, `out["G"]`) stays exactly as
  NSGA-III/QI-NSGA-III already use it.
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
- `_constrained_moead.py` — `ConstrainedMOEAD(MOEAD)`, see Constraint
  handling below.
- `_normalized_decomposition.py` — `NormalizedTchebycheff(Decomposition)`,
  see Objective normalization below.
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

## Constraint handling (added after spec review — blocking issue found)

`IRPProblem` declares real hard inequality constraints
(`n_ieq_constr = 2·|T| + |clients|·|T| + 2·|T|`: tau_return window,
per-client delivery-deadline coverage, depot stock ceiling safety net),
consumed today by pymoo's own feasibility-first constraint-domination in
NSGA-III's environmental selection. Verified directly against pymoo
0.6.1.6 source: `MOEAD._setup()` contains `assert not
problem.has_constraints()` — running pymoo's vanilla `MOEAD` against
`IRPProblem` unchanged raises `AssertionError` immediately at setup, not
a soft warning. This was missed in the original Architecture section,
which assumed `IRPProblem` could be reused completely unmodified (true
for NSGA-III/QI-NSGA-III, false for MOEA/D specifically).

**Fix (Deb's feasibility rule, chosen over a static penalty-weight
reformulation to avoid introducing a tuning parameter with no
counterpart on the NSGA-III/QI-NSGA-III side)**: a subclass,
`Solvers/MOEAD/_constrained_moead.py::ConstrainedMOEAD(MOEAD)`,
overriding only `_replace()` — everything else (neighbor structure,
weight vectors, `_setup`, crossover/mutation, `_infill`) stays pymoo's
own `MOEAD` unchanged. The override reads each individual's constraint
violation (`pop.get("CV")`, pymoo's own aggregate violation, already
computed automatically from `out["G"]` since `IRPProblem` sets
`n_ieq_constr` > 0) and applies Deb (2000)'s parameter-free feasibility
comparison in place of the raw decomposition value whenever either side
is infeasible:

```python
class ConstrainedMOEAD(MOEAD):
    def _setup(self, problem, **kwargs):
        # Bypass MOEAD's own assert -- constraints are handled below via
        # Deb's feasibility rule instead of being rejected outright.
        if self.ref_dirs is None:
            from pymoo.util.ref_dirs import default_ref_dirs
            self.ref_dirs = default_ref_dirs(problem.n_obj)
        self.pop_size = len(self.ref_dirs)
        from scipy.spatial.distance import cdist
        import numpy as np
        self.neighbors = np.argsort(
            cdist(self.ref_dirs, self.ref_dirs), axis=1, kind='quicksort'
        )[:, :self.n_neighbors]
        if self.decomposition is None:
            from pymoo.decomposition.tchebicheff import Tchebicheff
            self.decomposition = Tchebicheff()

    def _replace(self, k, off):
        import numpy as np
        pop = self.pop
        N = self.neighbors[k]

        neighbor_F = pop[N].get("F")
        # Freeze the normalization span for this replacement step -- see
        # NormalizedTchebycheff's own update_nadir docstring for why this
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

        # Deb (2000) feasibility rule, no tuning parameter:
        #  - both feasible  -> decomposition value decides (unchanged MOEAD)
        #  - one feasible   -> the feasible one always wins
        #  - both infeasible -> smaller total violation wins
        off_wins = np.where(
            off_CV <= 0,
            np.where(CV <= 0, off_FV < FV, True),
            np.where(CV <= 0, False, off_CV < CV),
        )
        I = np.where(off_wins)[0]
        pop[N[I]] = off
```

`Solvers/MOEAD/main.py` imports `ConstrainedMOEAD` in place of pymoo's
own `MOEAD` — same constructor signature, same `ref_dirs`/`crossover`/
`mutation`/`decomposition` kwargs, so the rest of the wiring below is
unaffected. `Solvers/MOEAD/test_main.py` includes a unit test
constructing two synthetic populations (feasible vs. infeasible,
infeasible-vs-infeasible with different CV) and asserting `_replace`
picks the expected winner in each of the three cases.

This same fix applies to the DTLZ/MaF benchmark runner
(`Validation/Benchmarking/algorithms/moead/runner.py`) for uniformity,
even though the standard DTLZ/MaF problems used there are themselves
unconstrained (`n_ieq_constr=0`) — `ConstrainedMOEAD` behaves identically
to vanilla `MOEAD` whenever every individual is feasible (`CV <= 0`
throughout, so `off_wins` reduces to the original `off_FV < FV` decomposition
comparison), so reusing one class in both places avoids two parallel
MOEA/D wirings.

## MOEA/D algorithm wiring

`Solvers.MOEAD._constrained_moead.ConstrainedMOEAD(ref_dirs,
n_neighbors=20, decomposition=None, prob_neighbor_mating=0.9,
sampling=..., crossover=..., mutation=...)` — same constructor as
pymoo's own `MOEAD` (see Constraint handling above for why the subclass
is needed). Wiring:
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
  algorithms — decided against. Re-confirmed during spec review: exact
  200 is not reachable via Das-Dennis for 4 objectives at any partition
  count (`n_partitions=8` → 165, `n_partitions=9` → 220, nothing lands on
  200), so keeping 165 is not just the original but the only option that
  preserves a shared reference-direction scheme across all three
  algorithms.

### Objective normalization in decomposition (added after spec review)

The IRP's 4 objectives have heterogeneous raw scales (cost ~15000-25000
DNT, CO2 ~800-1500 kg, time ~70-110 h, BFR ~3500-7000 DNT). NSGA-III and
QI-NSGA-III both already normalize by objective range before using
objectives for niching/guide selection: NSGA-III's environmental
selection (`ReferenceDirectionSurvival`) tracks both `ideal_point` and
`nadir_point` internally, updated every generation, and pymoo's own
NSGA-III implementation normalizes by that range before associating
individuals to reference directions. QI-NSGA-III's `_normalise_F`
function in `Solvers/QINSGA3/algorithm.py` does `(F - ideal) / (nadir -
ideal)` using that *same* tracked ideal/nadir pair (shared with the
survival step it runs alongside), falling back to a from-scratch
per-call estimate only on generation 0 before `survival.norm` is
populated.

Investigation of pymoo's `MOEAD` (`pymoo.algorithms.moo.moead.MOEAD`)
found no equivalent mechanism: `_replace()` passes only `ideal_point` to
`self.decomposition.do(...)`, never `nadir_point`. The default
decomposition for `n_obj > 2` (our case, 4 objectives) is PBI, not
Tchebycheff — and none of pymoo's built-in decomposition variants
(`Tchebicheff`, `PBI`, `ASF`) normalize by objective range; all operate
on raw `F` magnitudes after only subtracting the ideal/utopian point.
Left as-is, raw cost (scale ~20000) would dominate every replacement
decision regardless of the reference-direction weights, effectively
collapsing MOEA/D's decomposition to a cost-only comparison.

**Fix**: a small custom decomposition class,
`Solvers/MOEAD/_normalized_decomposition.py::NormalizedTchebycheff`,
subclassing pymoo's `Decomposition`, tracking its own running nadir
estimate (max `F` seen so far, monotonically expanding, mirroring how
`ReferenceDirectionSurvival.norm.nadir_point` behaves for the other two
algorithms) and normalizing `F` by `(nadir - ideal)` range before the
standard weighted-Tchebycheff formula. The update is a separate method
(`update_nadir`), not folded into `_do()` itself: `ConstrainedMOEAD._replace`
(below) calls `self.decomposition.do(...)` twice per replacement step
(once for the neighborhood, once for the offspring) — updating the
running estimate inside `_do()` would let the second call silently use a
different, more-informed span than the first, biasing the comparison
between them. `_replace` instead calls `update_nadir()` once, on the
union of both F sets, before either `do()` call, so both use the same
frozen snapshot:

```python
class NormalizedTchebycheff(Decomposition):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._nadir_running = None

    def update_nadir(self, F):
        batch_max = np.asarray(F).max(axis=0)
        self._nadir_running = batch_max if self._nadir_running is None \
            else np.maximum(self._nadir_running, batch_max)

    def _do(self, F, weights, **kwargs):
        span = (np.maximum(self._nadir_running - self.utopian_point, 1e-9)
                if self._nadir_running is not None else np.ones(F.shape[1]))
        F_norm = (F - self.utopian_point) / span
        return (np.abs(F_norm) * weights).max(axis=1)
```

This is passed as `decomposition=NormalizedTchebycheff()` to
`ConstrainedMOEAD(...)` in `Solvers/MOEAD/main.py`, replacing pymoo's
default PBI. Tchebycheff
(not PBI) is used specifically because it is the variant already
mentioned in the advisor's own feedback, and because PBI's penalty term
introduces a second hyperparameter (`theta`) with no existing precedent
in this project's other two algorithms — Tchebycheff keeps the
comparison operator-minimal, matching how NSGA-III/QI-NSGA-III's own
niching has no extra tunables either.
`Solvers/MOEAD/test_main.py` includes a unit test on
`NormalizedTchebycheff` directly (synthetic `F`/`weights`/`ideal_point`
inputs, checked against a hand-computed expected value) in addition to
the end-to-end smoke test.

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
  own `repair_final_front` flag test; plus a unit test on
  `NormalizedTchebycheff._do` (synthetic F/weights/ideal_point, hand-
  computed expected value) and a unit test on `ConstrainedMOEAD._replace`
  covering all three Deb's-rule branches (feasible beats infeasible,
  feasible-vs-feasible falls back to the plain decomposition comparison,
  infeasible-vs-infeasible picks the smaller violation) with synthetic
  populations.
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
