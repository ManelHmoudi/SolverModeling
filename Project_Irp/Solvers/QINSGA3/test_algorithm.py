"""Unit tests for QINSGA3.algorithm's pure helper functions."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Solvers.QINSGA3.algorithm import _normalise_F, _compute_nadir

# ── _normalise_F ──────────────────────────────────────────────────────────

def test_normalise_F_default_computes_ideal_and_nadir_from_F_alone():
    """Backward-compatible path (ideal/nadir not given): same behaviour as
    before the stateful-normalisation change -- used by _crowding_trim."""
    F = np.array([
        [0.0, 4.0],
        [2.0, 2.0],
        [4.0, 0.0],
    ])
    F_norm = _normalise_F(F)
    ideal = F.min(axis=0)
    nadir = _compute_nadir(F, ideal)
    expected = (F - ideal) / np.where(nadir - ideal > 1e-9, nadir - ideal, 1.0)
    assert np.allclose(F_norm, expected)


def test_normalise_F_with_explicit_ideal_nadir_ignores_F_extremes():
    """When ideal/nadir are given, F is normalised directly against them --
    NOT recomputed from F's own min/max. This is the whole point of sharing
    pymoo's running ideal/nadir: F's own instantaneous extremes must not
    override the run-wide estimate."""
    F = np.array([
        [5.0, 5.0],
        [7.0, 7.0],
    ])
    ideal = np.array([0.0, 0.0])   # far better than anything in F
    nadir = np.array([10.0, 10.0])  # far worse than anything in F

    F_norm = _normalise_F(F, ideal, nadir)

    assert np.allclose(F_norm, F / 10.0)
    # sanity: this is NOT what the from-scratch (ideal=None/nadir=None) path
    # would have produced, since F's own min/max differ from the given ideal/nadir
    assert not np.allclose(F_norm, _normalise_F(F))


def test_normalise_F_explicit_ideal_nadir_handles_degenerate_denominator():
    """If nadir == ideal on some objective (degenerate spread), fall back to
    dividing by 1.0 on that objective instead of producing inf/nan."""
    F = np.array([[3.0, 3.0], [3.0, 5.0]])
    ideal = np.array([3.0, 1.0])
    nadir = np.array([3.0, 5.0])   # same as ideal on objective 0

    F_norm = _normalise_F(F, ideal, nadir)

    assert np.all(np.isfinite(F_norm))
    assert np.allclose(F_norm[:, 0], F[:, 0] - ideal[0])  # denom falls back to 1.0
