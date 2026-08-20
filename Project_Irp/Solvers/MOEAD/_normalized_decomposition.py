"""Tchebycheff decomposition normalized by a running nadir estimate --
closes a gap in pymoo's own MOEAD found while integrating it into this
project (see docs/superpowers/specs/2026-08-19-moead-integration-design.md,
"Objective normalization in decomposition"): MOEAD._replace() only ever
passes ideal_point to the decomposition, never nadir_point, and none of
pymoo's built-in decomposition variants (Tchebicheff, PBI, ASF) normalize
by objective range -- all operate on raw F magnitudes after only
subtracting the ideal/utopian point. Left unfixed, the IRP's raw cost
scale (~15000-25000 DNT) would dominate every replacement decision
regardless of the reference-direction weights, collapsing MOEA/D's
decomposition to a cost-only comparison.

Mirrors how NSGA-III's own ReferenceDirectionSurvival tracks nadir_point
automatically across generations for the same normalization purpose, and
how QI-NSGA-III's _normalise_F (Solvers/QINSGA3/algorithm.py) reuses that
same tracked ideal/nadir pair -- MOEA/D has no equivalent built-in
mechanism, so this class provides one specific to it.

update_nadir() is a separate method from _do(), not folded into it:
ConstrainedMOEAD._replace() (see _constrained_moead.py) calls do() twice
per replacement step -- once for the neighborhood's F, once for the
offspring's F. Updating the running estimate inside _do() would let the
second call silently use a different, more-informed span than the first,
biasing the comparison between them. Callers must call update_nadir()
once, on the union of both F sets, before either do() call, so both use
the same frozen snapshot for that replacement step.
"""
import numpy as np
from pymoo.core.decomposition import Decomposition


class NormalizedTchebycheff(Decomposition):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._nadir_running = None

    def update_nadir(self, F):
        batch_max = np.asarray(F, dtype=float).max(axis=0)
        self._nadir_running = batch_max if self._nadir_running is None \
            else np.maximum(self._nadir_running, batch_max)

    def _do(self, F, weights, **kwargs):
        if self._nadir_running is None:
            span = np.ones(F.shape[1])
        else:
            span = np.maximum(self._nadir_running - self.utopian_point, 1e-9)
        F_norm = (F - self.utopian_point) / span
        return (np.abs(F_norm) * weights).max(axis=1)
