"""Unit tests for QINSGA3.algorithm's pure helper functions."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Solvers.QINSGA3.algorithm import (
    _normalise_F, _compute_nadir, _recentring_reset_mask,
    _update_pbest, _rqpso_rotate, _select_guides_ring,
    _max_min_density, _domination_counts, _adaptive_inertia, _pso_rotate,
    _elite_rms_distance, _chaotic_lambda_seed, _chaotic_lambda_step, _chaotic_rotate,
)

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


# ── _recentring_reset_mask ───────────────────────────────────────────────────
# Corrected trigger (direct theta clustering on every occupied niche, every
# generation -- see design doc's "Amendment" for why both the paper's eq. 11
# boundary-convergence gate AND its eq. 13 stagnation gate are structural
# mismatches for QINSGA3, and were dropped rather than reused verbatim) and
# corrected reset target (left to the caller: this function only returns WHO
# to reset, not the reset value -- design doc uses a fresh U(0, pi/2) draw
# instead of the paper's pi/4).

def _two_niche_fixture():
    # ref_dirs: niche 0 along objective 1, niche 1 along objective 2
    ref_dirs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assoc = np.array([0, 0, 0, 1])
    # niche 0: individual 0 sits exactly on the ray (best); 1 is theta-close to
    # it; 2 is theta-far from it. niche 1: single member (nothing to cluster).
    F_norm = np.array([
        [1.00, 0.00],   # niche 0, best (d_perp2 = 0)
        [1.00, 0.10],   # niche 0
        [1.00, 0.05],   # niche 0
        [0.00, 1.00],   # niche 1, alone
    ])
    theta = np.array([
        [0.50, 0.50],   # best
        [0.51, 0.49],   # theta-close to best
        [1.00, 1.00],   # theta-far from best
        [0.20, 0.20],
    ])
    return assoc, theta, F_norm, ref_dirs


def test_recentring_reset_mask_resets_only_the_theta_close_non_best_member():
    assoc, theta, F_norm, ref_dirs = _two_niche_fixture()
    mask = _recentring_reset_mask(assoc, theta, F_norm, ref_dirs, delta=0.05)
    assert mask.tolist() == [False, True, False, False]


def test_recentring_reset_mask_leaves_theta_far_member_untouched():
    assoc, theta, F_norm, ref_dirs = _two_niche_fixture()
    # tighten delta so even the theta-close member (dist ~0.0064) no longer
    # qualifies, and confirm the theta-far member (dist ~0.318) never does
    mask = _recentring_reset_mask(assoc, theta, F_norm, ref_dirs, delta=0.001)
    assert mask.tolist() == [False, False, False, False]


def test_recentring_reset_mask_single_member_niche_never_resets():
    assoc, theta, F_norm, ref_dirs = _two_niche_fixture()
    # niche 1 has only individual 3 -- nothing to compare it against, even
    # with a huge delta that catches everything else (niche 0's members)
    mask = _recentring_reset_mask(assoc, theta, F_norm, ref_dirs, delta=10.0)
    assert not mask[3]


# ── _update_pbest ─────────────────────────────────────────────────────────
# Per-slot running best (positional approximation of RQPSO's personal best --
# see design note above _rqpso_rotate in algorithm.py for why a faithful
# per-individual pbest doesn't transfer to QINSGA3's recombining population).

def test_update_pbest_improves_when_quality_lower():
    pbest_theta  = np.array([[0.5, 0.5], [0.5, 0.5]])
    pbest_scalar = np.array([1.0, 1.0])
    theta        = np.array([[0.9, 0.9], [0.9, 0.9]])
    F_norm       = np.array([[0.2, 0.2], [0.2, 0.2]])  # mean=0.2 < pbest_scalar=1.0

    new_theta, new_scalar = _update_pbest(pbest_theta, pbest_scalar, theta, F_norm)

    assert np.allclose(new_theta, theta)
    assert np.allclose(new_scalar, [0.2, 0.2])


def test_update_pbest_keeps_previous_when_quality_not_better():
    pbest_theta  = np.array([[0.5, 0.5]])
    pbest_scalar = np.array([0.1])
    theta        = np.array([[0.9, 0.9]])
    F_norm       = np.array([[0.8, 0.8]])  # mean=0.8 > pbest_scalar=0.1, worse

    new_theta, new_scalar = _update_pbest(pbest_theta, pbest_scalar, theta, F_norm)

    assert np.allclose(new_theta, pbest_theta)
    assert np.allclose(new_scalar, pbest_scalar)


def test_update_pbest_mixed_slots_update_independently():
    pbest_theta  = np.array([[0.5, 0.5], [0.5, 0.5]])
    pbest_scalar = np.array([0.5, 0.5])
    theta        = np.array([[0.1, 0.1], [0.9, 0.9]])
    F_norm       = np.array([[0.2, 0.2], [0.9, 0.9]])  # slot 0 improves, slot 1 doesn't

    new_theta, new_scalar = _update_pbest(pbest_theta, pbest_scalar, theta, F_norm)

    assert np.allclose(new_theta[0], theta[0])
    assert np.allclose(new_theta[1], pbest_theta[1])
    assert np.allclose(new_scalar, [0.2, 0.5])


# ── _rqpso_rotate ─────────────────────────────────────────────────────────
# [Bodha, Arun, Awasthi, Mahato & Fotis 2025], domain-adapted dual-attractor
# rotation -- see algorithm.py's module note above _rqpso_rotate for the
# two documented adaptations (bounded non-cyclic domain, theta_step scale).

def test_rqpso_rotate_no_movement_when_theta_matches_both_attractors():
    theta = np.array([[0.5, 0.5]])
    rng = np.random.default_rng(0)
    result = _rqpso_rotate(theta, pbest_theta=theta, guide_theta=theta,
                            theta_step=0.3, rng=rng)
    assert np.allclose(result, theta)


def test_rqpso_rotate_moves_toward_attractors_when_theta_is_below_both():
    # diff_p = diff_g = (pi/2 - 0)/(pi/2) = 1 >= 0 regardless of u1, u2 draw
    # (both non-negative), so the step is deterministically non-negative.
    theta = np.zeros((5, 3))
    attractor = np.full((5, 3), np.pi / 2.0)
    rng = np.random.default_rng(1)
    result = _rqpso_rotate(theta, pbest_theta=attractor, guide_theta=attractor,
                            theta_step=0.1, rng=rng)
    assert np.all(result >= theta)


def test_rqpso_rotate_clips_to_domain_bounds():
    theta = np.array([[1.5, 1.5]])
    attractor = np.array([[np.pi / 2.0, np.pi / 2.0]])
    rng = np.random.default_rng(2)
    # theta_step deliberately huge to force overshoot past pi/2
    result = _rqpso_rotate(theta, pbest_theta=attractor, guide_theta=attractor,
                            theta_step=10.0, rng=rng)
    assert np.all(result <= np.pi / 2.0)
    assert np.all(result >= 0.0)


# ── _select_guides_ring ──────────────────────────────────────────────────
# [Tayarani-N & Akbarzadeh-T 2014, §3] Ring structure -- 8th remedy. Each
# niche member rotates toward the best of {self, ring-prev, ring-next}
# instead of the whole niche's single best (see the module docstring above
# _select_guides_ring for why this is a purely per-generation grouping, no
# individual-identity persistence needed).

def test_select_guides_ring_single_member_niche_guides_toward_self():
    assoc    = np.array([0])
    F_norm   = np.array([[1.0, 0.0]])
    ref_dirs = np.array([[1.0, 0.0]])
    theta    = np.array([[0.3, 0.3]])

    guides = _select_guides_ring(assoc, F_norm, ref_dirs, theta)

    assert np.allclose(guides[0], theta[0])


def test_select_guides_ring_two_member_niche_pulls_worse_toward_better():
    assoc    = np.array([0, 0])
    ref_dirs = np.array([[1.0, 0.0]])
    # individual 0 sits exactly on the ray (d_perp2=0, best); individual 1 is off it
    F_norm   = np.array([[1.0, 0.0], [1.0, 5.0]])
    theta    = np.array([[0.2, 0.2], [0.8, 0.8]])

    guides = _select_guides_ring(assoc, F_norm, ref_dirs, theta)

    assert np.allclose(guides[0], theta[0])   # already the best -> guides toward self
    assert np.allclose(guides[1], theta[0])   # worse -> guides toward the better neighbour


def test_select_guides_ring_local_best_differs_from_global_best():
    """5-member ring where the global champion (index 0) is NOT within
    reach of every member -- proves the ring restricts comparison to local
    neighbours rather than collapsing the whole niche onto one point."""
    assoc    = np.array([0, 0, 0, 0, 0])
    ref_dirs = np.array([[1.0, 0.0]])
    # d_perp2 by ring position (0-1-2-3-4-0): [0.0, 5.0, 10.0, 5.0, 0.1]
    F_norm = np.array([
        [1.0, 0.0],                 # pos 0: d_perp2 = 0.0   (global best)
        [1.0, np.sqrt(5.0)],        # pos 1: d_perp2 = 5.0
        [1.0, np.sqrt(10.0)],       # pos 2: d_perp2 = 10.0
        [1.0, np.sqrt(5.0)],        # pos 3: d_perp2 = 5.0
        [1.0, np.sqrt(0.1)],        # pos 4: d_perp2 = 0.1
    ])
    theta = np.arange(5).reshape(5, 1).astype(float)  # theta[k] = k, so guide index is readable

    guides = _select_guides_ring(assoc, F_norm, ref_dirs, theta)

    # position 2's neighbours are 1 and 3 (d=5.0 each, tie) -- picks position 1,
    # NEVER sees position 0 (the global best) even though it's better overall
    assert guides[2, 0] == 1.0
    # position 3's neighbours are 2 (d=10.0) and 4 (d=0.1) -- picks 4, not the global best either
    assert guides[3, 0] == 4.0
    # position 0 is already the global best among its own neighbours -> self
    assert guides[0, 0] == 0.0


# ── _max_min_density / _domination_counts / _adaptive_inertia / _pso_rotate ──
# [Li, Xu, Liu & Li 2008] -- 9th remedy: PSO-style momentum rotation.

def test_max_min_density_matches_hand_computed_example():
    # theta = [0, 1, 10] (1-D): d(0,1)=1, d(0,2)=10, d(1,2)=9
    # d_min = [1, 1, 9] -> d_max_min = 9
    # D(0) = sign(9-1) + sign(9-10) = 1 - 1 = 0
    # D(1) = sign(9-1) + sign(9-9)  = 1 + 0 = 1
    # D(2) = sign(9-10) + sign(9-9) = -1 + 0 = -1
    theta = np.array([[0.0], [1.0], [10.0]])
    D = _max_min_density(theta)
    assert np.allclose(D, [0.0, 1.0, -1.0])


def test_domination_counts_counts_pareto_dominance_excluding_self():
    F = np.array([[1.0, 1.0], [2.0, 2.0], [0.5, 0.5]])
    m = _domination_counts(F)
    # individual 0 dominates only 1; individual 1 dominates none;
    # individual 2 (best on both objectives) dominates 0 and 1
    assert np.array_equal(m, [1, 0, 2])


def test_adaptive_inertia_clips_to_0_2_range():
    D = np.array([10.0, -10.0])
    m = np.array([10.0, 0.0])
    w = _adaptive_inertia(D, m, pop_size=5)
    # raw values would be 4.0 and -2.0 -- both outside [0, 2]
    assert np.allclose(w, [2.0, 0.0])


def test_pso_rotate_no_movement_when_at_rest_and_at_attractors():
    theta = np.array([[0.5, 0.5]])
    velocity = np.zeros_like(theta)
    w = np.array([0.5])
    rng = np.random.default_rng(0)
    new_theta, new_v = _pso_rotate(theta, velocity, pbest_theta=theta, guide_theta=theta,
                                    w=w, v_clip=0.3, rng=rng)
    assert np.allclose(new_theta, theta)
    assert np.allclose(new_v, 0.0)


def test_pso_rotate_preserves_momentum_when_no_pull():
    # pbest == guide == theta (no attractor pull), but a pre-existing velocity
    # must persist (scaled by w) -- this is the whole point of this remedy.
    theta = np.array([[0.5, 0.5]])
    velocity = np.array([[0.05, 0.05]])
    w = np.array([1.0])
    rng = np.random.default_rng(0)
    new_theta, new_v = _pso_rotate(theta, velocity, pbest_theta=theta, guide_theta=theta,
                                    w=w, v_clip=0.3, rng=rng)
    assert np.allclose(new_v, velocity)
    assert np.allclose(new_theta, theta + velocity)


def test_pso_rotate_clips_velocity_to_bounds():
    theta = np.array([[0.1, 0.1]])
    attractor = np.array([[np.pi / 2.0, np.pi / 2.0]])
    velocity = np.zeros_like(theta)
    w = np.array([0.5])
    rng = np.random.default_rng(1)
    new_theta, new_v = _pso_rotate(theta, velocity, pbest_theta=attractor, guide_theta=attractor,
                                    w=w, v_clip=0.05, rng=rng)
    assert np.all(np.abs(new_v) <= 0.05 + 1e-12)


# ── _elite_rms_distance / _chaotic_lambda_seed / _chaotic_lambda_step /
#    _chaotic_rotate ──────────────────────────────────────────────────────
# [Hu Feng-jun & Wu Bin 2009, "Quantum Evolutionary Algorithm for Vehicle
# Routing Problem with Simultaneous Delivery and Pickup", Joint 48th IEEE
# CDC / 28th CCC] -- 10th remedy: chaos-modulated rotation magnitude, RMS
# distance to the niche's archive elites instead of a single guide. See the
# module note above _chaotic_rotate in algorithm.py for the documented
# adaptations (chi direction, archive as B(t), positional lambda state).

def test_elite_rms_distance_matches_hand_computed_example():
    # niche 0: pop members theta=[0,0] and [1,1]; archive elites [2,2],[4,4]
    # member 0: per-gene diffs to elites = [2,4] -> RMS = sqrt(mean([4,16])) = sqrt(10)
    # member 1: per-gene diffs to elites = [1,3] -> RMS = sqrt(mean([1,9]))  = sqrt(5)
    theta        = np.array([[0.0, 0.0], [1.0, 1.0]])
    assoc        = np.array([0, 0])
    guides_theta = np.array([[9.0, 9.0], [9.0, 9.0]])  # unused when archive elites exist
    arch_theta   = np.array([[2.0, 2.0], [4.0, 4.0]])
    arch_assoc   = np.array([0, 0])

    rms = _elite_rms_distance(theta, assoc, guides_theta, arch_theta, arch_assoc)

    assert np.allclose(rms[0], np.sqrt(10.0))
    assert np.allclose(rms[1], np.sqrt(5.0))


def test_elite_rms_distance_falls_back_to_guide_when_niche_has_no_archive_elites():
    theta        = np.array([[0.3, 0.3]])
    assoc        = np.array([0])
    guides_theta = np.array([[0.5, 0.5]])
    arch_theta   = np.array([[9.0, 9.0]])
    arch_assoc   = np.array([1])  # niche 1, not niche 0 -- no elite covers niche 0

    rms = _elite_rms_distance(theta, assoc, guides_theta, arch_theta, arch_assoc)

    assert np.allclose(rms, [[0.2, 0.2]])


def test_elite_rms_distance_no_archive_at_all_falls_back_to_guide():
    theta        = np.array([[0.3, 0.3]])
    assoc        = np.array([0])
    guides_theta = np.array([[0.5, 0.5]])

    rms = _elite_rms_distance(theta, assoc, guides_theta, arch_theta=None, arch_assoc=None)

    assert np.allclose(rms, [[0.2, 0.2]])


def test_chaotic_lambda_seed_clips_to_open_unit_interval():
    F_norm = np.array([[0.0, 0.0], [1.0, 1.0], [0.4, 0.6]])
    lam = _chaotic_lambda_seed(F_norm)
    assert lam[0] > 0.0    # mean=0.0 -> clipped above the lower bound
    assert lam[1] < 1.0    # mean=1.0 -> clipped below the upper bound
    assert np.isclose(lam[2], 0.5)


def test_chaotic_lambda_step_matches_logistic_map():
    lam = np.array([0.2])
    nxt = _chaotic_lambda_step(lam, mu=4.0)
    assert np.isclose(nxt[0], 4.0 * 0.2 * 0.8)


def test_chaotic_lambda_step_clips_fixed_point_to_open_interval():
    # lam=0.5 -> 4*0.5*0.5 = 1.0 exactly, a fixed point the map would then
    # freeze at (1 -> 0 -> 0 -> ...) -- must be nudged strictly below 1.0
    lam = np.array([0.5])
    nxt = _chaotic_lambda_step(lam, mu=4.0)
    assert nxt[0] < 1.0


def test_chaotic_rotate_no_movement_when_theta_matches_guide():
    theta = np.array([[0.5, 0.5]])
    result = _chaotic_rotate(theta, guide_theta=theta, elite_rms=np.array([[0.3, 0.3]]),
                              lam=np.array([0.5]))
    assert np.allclose(result, theta)


def test_chaotic_rotate_moves_toward_guide_by_rms_magnitude_when_lambda_zero():
    theta = np.array([[0.5, 0.5]])
    guide = np.array([[1.0, 1.0]])
    result = _chaotic_rotate(theta, guide_theta=guide, elite_rms=np.array([[0.1, 0.1]]),
                              lam=np.array([0.0]))
    assert np.allclose(result, theta + 0.1)


def test_chaotic_rotate_clips_to_domain_bounds():
    theta = np.array([[1.5, 1.5]])
    guide = np.array([[np.pi / 2.0, np.pi / 2.0]])
    result = _chaotic_rotate(theta, guide_theta=guide, elite_rms=np.array([[10.0, 10.0]]),
                              lam=np.array([0.9]))
    assert np.all(result <= np.pi / 2.0)
    assert np.all(result >= 0.0)
