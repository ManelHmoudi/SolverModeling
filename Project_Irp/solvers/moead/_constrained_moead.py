"""ConstrainedMOEAD -- pymoo's own MOEAD, unmodified except for _replace(),
which applies Deb (2000)'s parameter-free feasibility rule instead of a
raw decomposition-value comparison. Needed because IRPProblem
(solvers/nsga3/problem.py) declares real hard inequality constraints
(n_ieq_constr = solvers.nsga3.problem.n_constraints: shift window, lateness
per class, depot stock bounds, closed cycle), and pymoo's own MOEAD._setup() contains
`assert not problem.has_constraints()` -- running vanilla MOEAD against
IRPProblem crashes immediately at setup. Everything else (neighbor
structure, weight vectors, _infill, crossover/mutation) is inherited from
pymoo's MOEAD unchanged. See
the earlier design notes (git history),
"Constraint handling", for the full rationale, including why a static
penalty-weight reformulation was rejected (it would add a tuning
parameter with no counterpart on the NSGA-III/QI-NSGA-III side).

Deb's rule (no tuning parameter, matches the "feasibility-first"
principle NSGA-III/QI-NSGA-III already apply via pymoo's own
constraint-domination):
  - one feasible, one infeasible -> the feasible one always wins
  - both feasible                -> normal decomposition value decides
  - both infeasible               -> the smaller total constraint violation wins

Also used, unchanged, for the DTLZ/MaF benchmark runner
(former `Validation/Benchmarking/algorithms/moead/runner.py`, removed — see git history) even though those
problems are themselves unconstrained: when every individual is feasible
(CV <= 0 throughout), the rule below reduces exactly to `off_FV < FV`,
i.e. vanilla MOEAD's own comparison -- reusing one class in both places
avoids two parallel MOEA/D wirings.
"""
import numpy as np
from scipy.spatial.distance import cdist

from pymoo.algorithms.moo.moead import MOEAD
from pymoo.util.reference_direction import default_ref_dirs


class ConstrainedMOEAD(MOEAD):

    def _setup(self, problem, **kwargs):
        if self.ref_dirs is None:
            self.ref_dirs = default_ref_dirs(problem.n_obj)
        self.pop_size = len(self.ref_dirs)
        self.neighbors = np.argsort(
            cdist(self.ref_dirs, self.ref_dirs), axis=1, kind="quicksort"
        )[:, : self.n_neighbors]
        if self.decomposition is None:
            # NOT pymoo's own default_decomp(problem) -- that returns PBI for
            # n_obj > 2 (our case, 4 objectives), which the project's own
            # spec explicitly argues against as a default: PBI's penalty
            # term introduces a second hyperparameter (theta) with no
            # existing precedent on the NSGA-III/QI-NSGA-III side. See
            # the earlier design notes (git history),
            # "Objective normalization in decomposition". Every real call
            # site in this codebase passes decomposition=NormalizedTchebycheff()
            # explicitly, so this fallback is a safety net, not a live path --
            # it must still match the spec's own choice if it ever executes.
            from pymoo.decomposition.tchebicheff import Tchebicheff
            self.decomposition = Tchebicheff()

    def _replace(self, k, off):
        pop = self.pop
        N = self.neighbors[k]

        neighbor_F = pop[N].get("F")
        CV = pop[N].get("CV")[:, 0]
        off_CV = float(off.CV[0])

        # Ideal and nadir used for the normalisation are tracked on FEASIBLE
        # solutions only (as NSGA-III normalises on feasible members first):
        # an infeasible solution that skips demand looks cheap and would
        # distort the scale. Until a feasible solution exists, all are used.
        cand_F = np.vstack([neighbor_F, off.F[None, :]])
        cand_feas = np.concatenate([CV <= 0, [off_CV <= 0]])
        if cand_feas.any():
            feas_F = cand_F[cand_feas]
            batch_min = feas_F.min(axis=0)
            self._feas_ideal = batch_min if getattr(self, "_feas_ideal", None) is None \
                else np.minimum(self._feas_ideal, batch_min)
        else:
            feas_F = cand_F if getattr(self, "_feas_ideal", None) is None else None
        ideal = self._feas_ideal if getattr(self, "_feas_ideal", None) is not None else self.ideal
        # Freeze the normalization span for this replacement step -- see
        # NormalizedTchebycheff.update_nadir's own docstring for why this
        # must happen once, before either do() call below, not inside _do().
        if hasattr(self.decomposition, "update_nadir") and feas_F is not None:
            self.decomposition.update_nadir(feas_F)

        FV = self.decomposition.do(
            neighbor_F, weights=self.ref_dirs[N, :], ideal_point=ideal,
        )
        off_FV = self.decomposition.do(
            off.F[None, :], weights=self.ref_dirs[N, :], ideal_point=ideal,
        )

        # Deb (2000) feasibility rule:
        #  - both feasible    -> decomposition value decides (unchanged MOEAD)
        #  - one feasible     -> the feasible one always wins
        #  - both infeasible  -> smaller total violation wins
        off_wins = np.where(
            off_CV <= 0,
            np.where(CV <= 0, off_FV < FV, True),
            np.where(CV <= 0, False, off_CV < CV),
        )
        I = np.where(off_wins)[0]
        pop[N[I]] = off
