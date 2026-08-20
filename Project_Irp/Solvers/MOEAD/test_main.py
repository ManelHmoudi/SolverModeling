"""Tests for the MOEA/D IRP integration: end-to-end run_moead smoke tests
(mirroring Solvers/NSGA3/test_main.py / Solvers/QINSGA3/test_main.py), plus
unit tests for the two MOEA/D-specific helper classes this project adds on
top of pymoo's own MOEAD: NormalizedTchebycheff (objective-range
normalization, closing a gap in pymoo's own decomposition variants) and
ConstrainedMOEAD (Deb's feasibility rule, since IRPProblem has hard
constraints pymoo's own MOEAD rejects outright). See
docs/superpowers/specs/2026-08-19-moead-integration-design.md for the
full rationale behind both."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Solvers.MOEAD._normalized_decomposition import NormalizedTchebycheff

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TINY_INSTANCE = os.path.join(PROJECT_DIR, "data", "instance_3_clients.json")


# ── NormalizedTchebycheff ───────────────────────────────────────────────

def test_normalized_tchebycheff_matches_hand_computed_value():
    dec = NormalizedTchebycheff()
    F = np.array([[10.0, 100.0], [20.0, 50.0]])
    weights = np.array([[0.5, 0.5], [0.5, 0.5]])
    ideal = np.array([0.0, 0.0])

    result = dec.do(F, weights=weights, ideal_point=ideal)

    # update_nadir has not been called yet -> _nadir_running is None ->
    # _do falls back to span=ones(n_obj), i.e. no normalization this call.
    # F_norm = F (unnormalized). |F_norm| * weights = [[5,50],[10,25]].
    # max(axis=1) = [50, 25].
    np.testing.assert_allclose(result, [50.0, 25.0])


def test_normalized_tchebycheff_normalizes_after_update_nadir():
    dec = NormalizedTchebycheff()
    F = np.array([[10.0, 100.0], [20.0, 50.0]])
    weights = np.array([[0.5, 0.5], [0.5, 0.5]])
    ideal = np.array([0.0, 0.0])

    dec.update_nadir(F)
    result = dec.do(F, weights=weights, ideal_point=ideal)

    # nadir_running = batch max = [20, 100]. span = [20, 100].
    # F_norm = F / span = [[0.5, 1.0], [1.0, 0.5]].
    # |F_norm| * weights = [[0.25, 0.5], [0.5, 0.25]]. max(axis=1) = [0.5, 0.5].
    np.testing.assert_allclose(result, [0.5, 0.5])


def test_normalized_tchebycheff_nadir_estimate_is_monotonic():
    """The running nadir estimate only ever grows -- a later call with a
    smaller batch max must not shrink the normalization span."""
    dec = NormalizedTchebycheff()

    dec.update_nadir(np.array([[100.0, 100.0]]))
    assert dec._nadir_running.tolist() == [100.0, 100.0]

    dec.update_nadir(np.array([[10.0, 10.0]]))
    assert dec._nadir_running.tolist() == [100.0, 100.0]

    dec.update_nadir(np.array([[500.0, 1.0]]))
    assert dec._nadir_running.tolist() == [500.0, 100.0]
