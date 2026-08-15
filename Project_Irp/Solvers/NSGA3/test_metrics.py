"""Unit tests for Solvers.NSGA3.metrics -- empirical reference-front build and
its use in GD/IGD, replacing the Das-Dennis reference-direction grid.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Solvers.NSGA3.metrics import build_empirical_reference_front, compute_pareto_metrics


def test_build_empirical_reference_front_keeps_only_non_dominated_hand_verified():
    """5-point Pareto staircase (minimise both objectives, all mutually
    non-dominated) plus one duplicate of a staircase point and one point
    dominated by everything else. Only the 5 unique staircase points must
    survive."""
    staircase = np.array([
        [1.0, 5.0],
        [2.0, 4.0],
        [3.0, 3.0],
        [4.0, 2.0],
        [5.0, 1.0],
    ])
    F_pool = np.vstack([staircase, [3.0, 3.0], [10.0, 10.0]])  # duplicate + dominated

    ref = build_empirical_reference_front(F_pool)

    assert len(ref) == 5
    ref_sorted = ref[np.argsort(ref[:, 0])]
    np.testing.assert_array_equal(ref_sorted, staircase)


def test_build_empirical_reference_front_dominated_point_excluded():
    """A point dominated by another in the pool (worse or equal on every
    objective, strictly worse on at least one) must not appear in the
    result."""
    F_pool = np.array([
        [1.0, 1.0],   # dominates the point below
        [2.0, 2.0],   # dominated by [1,1]
    ])
    ref = build_empirical_reference_front(F_pool)
    assert len(ref) == 1
    np.testing.assert_array_equal(ref[0], [1.0, 1.0])


def test_compute_pareto_metrics_gd_igd_zero_when_front_equals_reference():
    """When F IS the reference front (best possible case), GD and IGD must
    both be exactly 0 -- every solution has a reference point at distance 0
    and vice versa."""
    F = np.array([
        [1.0, 5.0],
        [3.0, 3.0],
        [5.0, 1.0],
    ])
    q = compute_pareto_metrics(F, reference_front=F.copy())
    assert q["GD"] == 0.0
    assert q["IGD"] == 0.0


def test_compute_pareto_metrics_reference_front_uses_same_normalisation_as_F():
    """The reference front must be normalised with the SAME ideal/rng as F
    itself (here via global_ideal/global_nadir) -- verified by checking GD
    is 0 when F is a subset of a (differently-scaled) empirical reference
    front built from a larger pool, using shared global bounds."""
    pool = np.array([
        [0.0, 10.0],
        [5.0, 5.0],
        [10.0, 0.0],
        [20.0, 20.0],   # dominated, excluded from the reference front
    ])
    ref_front = build_empirical_reference_front(pool)  # the 3 non-dominated points
    g_ideal = pool.min(axis=0)
    g_nadir = pool.max(axis=0)

    # F is exactly the reference front -- GD/IGD must be 0 regardless of the
    # shared global_ideal/global_nadir scale used to normalise both sides.
    q = compute_pareto_metrics(ref_front, global_ideal=g_ideal, global_nadir=g_nadir,
                                reference_front=ref_front)
    assert q["GD"] == 0.0
    assert q["IGD"] == 0.0


def test_compute_pareto_metrics_defaults_to_das_dennis_when_no_reference_front():
    """Backward compatibility: omitting reference_front must not raise and
    must still return numeric GD/IGD (falls back to the Das-Dennis
    reference-direction grid, matching pre-fix behaviour)."""
    F = np.array([[1.0, 5.0], [3.0, 3.0], [5.0, 1.0]])
    q = compute_pareto_metrics(F)
    assert isinstance(q["GD"], float)
    assert isinstance(q["IGD"], float)
