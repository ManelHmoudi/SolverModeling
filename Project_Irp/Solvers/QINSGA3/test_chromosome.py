"""Unit tests for QINSGA3.chromosome.QuantumPopulation, focused on the
compensate_dx_dtheta rotation-step modifier -- see QuantumPopulation.rotate()'s
own docstring for the fairness-audit hypothesis this tests (the theta->X
measurement's non-uniform dx/dtheta mapping may explain part of QI-NSGA-III's
residual HV/IGD gap vs NSGA-III)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Solvers.QINSGA3.chromosome import QuantumPopulation, _DX_DTHETA_FLOOR


def _make_pop(theta_value: float, compensate: bool) -> QuantumPopulation:
    """1 individual, 1 gene, theta fixed at theta_value (bypassing the
    random pi/4-centred init)."""
    pop = QuantumPopulation(
        pop_size=1, n_genes=1,
        xl=np.array([0.0]), xu=np.array([1.0]),
        rng=np.random.default_rng(0),
        compensate_dx_dtheta=compensate,
    )
    pop.theta = np.array([[theta_value]])
    return pop


def test_compensate_disabled_by_default():
    pop = QuantumPopulation(pop_size=1, n_genes=1, xl=np.array([0.0]), xu=np.array([1.0]))
    assert pop.compensate_dx_dtheta is False


def test_compensate_matches_uncompensated_at_theta_pi_4():
    """sin(2*pi/4) = sin(pi/2) = 1 -> scale = 1/clip(1,...) = 1, no change."""
    theta0 = np.pi / 4.0
    guide  = np.array([[np.pi / 2.0]])

    pop_off = _make_pop(theta0, compensate=False)
    pop_on  = _make_pop(theta0, compensate=True)

    pop_off.rotate(guide, alpha=0.1)
    pop_on.rotate(guide, alpha=0.1)

    assert np.allclose(pop_off.theta, pop_on.theta)


def test_compensate_amplifies_step_near_theta_zero():
    """Near theta=0, sin(2*theta) -> 0, so compensation should produce a
    LARGER angular step than the uncompensated rotation (moving further
    toward the guide each generation, counteracting the vanishing real
    X-space displacement there)."""
    theta0 = 0.01   # close to the theta=0 boundary
    guide  = np.array([[np.pi / 2.0]])  # pulls upward, away from 0

    pop_off = _make_pop(theta0, compensate=False)
    pop_on  = _make_pop(theta0, compensate=True)

    pop_off.rotate(guide, alpha=0.05)
    pop_on.rotate(guide, alpha=0.05)

    step_off = pop_off.theta[0, 0] - theta0
    step_on  = pop_on.theta[0, 0] - theta0
    assert step_on > step_off > 0


def test_compensate_floor_caps_amplification_at_theta_exactly_zero():
    """At theta=0 exactly, sin(2*0)=0 -- must be floored to _DX_DTHETA_FLOOR
    (max 1/_DX_DTHETA_FLOOR = 10x amplification), not divide-by-zero /
    diverge to infinity."""
    theta0 = 0.0
    guide  = np.array([[np.pi / 2.0]])
    alpha  = 0.05

    pop_off = _make_pop(theta0, compensate=False)
    pop_on  = _make_pop(theta0, compensate=True)

    pop_off.rotate(guide, alpha=alpha)
    pop_on.rotate(guide, alpha=alpha)

    step_off = pop_off.theta[0, 0] - theta0
    step_on  = pop_on.theta[0, 0] - theta0
    max_amplification = 1.0 / _DX_DTHETA_FLOOR
    # tanh saturates below 1, so the ratio is bounded by, not equal to, the
    # raw alpha amplification factor -- assert it's finite and within bound.
    assert np.isfinite(step_on)
    assert step_on <= step_off * max_amplification + 1e-9


def test_compensate_theta_stays_within_bounds():
    """Even with the amplified step, theta must still clip to [0, pi/2]."""
    pop_on = _make_pop(0.001, compensate=True)
    guide  = np.array([[np.pi / 2.0]])
    pop_on.rotate(guide, alpha=0.5)   # large alpha, amplified up to 10x more
    assert 0.0 <= pop_on.theta[0, 0] <= np.pi / 2.0
