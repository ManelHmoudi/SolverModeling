"""ConstrainedMOEAD -- pymoo's own MOEAD, unmodified except for _replace(),
which applies Deb (2000)'s parameter-free feasibility rule instead of a
raw decomposition-value comparison. Needed because IRPProblem
(Solvers/NSGA3/problem.py) declares real hard inequality constraints
(n_ieq_constr = 2*|T| + |clients|*|T| + 2*|T|: tau_return window, delivery
deadlines, depot stock ceiling), and pymoo's own MOEAD._setup() contains
`assert not problem.has_constraints()` -- running vanilla MOEAD against
IRPProblem crashes immediately at setup. Everything else (neighbor
structure, weight vectors, _infill, crossover/mutation) is inherited from
pymoo's MOEAD unchanged. See
docs/superpowers/specs/2026-08-19-moead-integration-design.md,
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
(Validation/Benchmarking/algorithms/moead/runner.py) even though those
problems are themselves unconstrained: when every individual is feasible
(CV <= 0 throughout), the rule below reduces exactly to `off_FV < FV`,
i.e. vanilla MOEAD's own comparison -- reusing one class in both places
avoids two parallel MOEA/D wirings.
"""
import numpy as np
from scipy.spatial.distance import cdist

from pymoo.algorithms.moo.moead import MOEAD, default_decomp
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
            self.decomposition = default_decomp(problem)

    def _replace(self, k, off):
        pop = self.pop
        N = self.neighbors[k]

        neighbor_F = pop[N].get("F")
        # Freeze the normalization span for this replacement step -- see
        # NormalizedTchebycheff.update_nadir's own docstring for why this
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
