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
    _select_guides_crowding, _supplement_from_archive_crowding, _select_guides,
    _crowding_saturation_stats, _build_g_constraints,
    _evaluate_with_repair, _repair_pareto_front,
    _generate_offspring_batch, _eliminate_duplicates_refill,
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


# ── _select_guides_crowding ─────────────────────────────────────────────
# Remedy F: niche champion chosen by HIGHEST crowding distance instead of
# LOWEST perpendicular distance to the reference ray. See
# docs/superpowers/specs/2026-08-03-qinsga3-crowding-distance-guide-design.md.

def test_select_guides_crowding_single_pareto_member_guides_toward_itself():
    assoc      = np.array([0])
    pareto_idx = np.array([0])
    F_norm     = np.array([[1.0, 0.0]])
    theta      = np.array([[0.3, 0.3]])

    guides = _select_guides_crowding(assoc, pareto_idx, F_norm, theta)

    assert np.allclose(guides[0], theta[0])


def test_select_guides_crowding_two_member_niche_picks_deterministically_without_crash():
    """With exactly 2 Pareto members and 2 objectives, _crowding_distance
    assigns inf to both (both are boundary points on every objective) --
    argmax's first-index tie-break must still return a valid, non-crashing
    result (the degenerate case flagged in the design doc's Risk section)."""
    assoc      = np.array([0, 0])
    pareto_idx = np.array([0, 1])
    F_norm     = np.array([[0.0, 1.0], [1.0, 0.0]])
    theta      = np.array([[0.2, 0.2], [0.8, 0.8]])

    guides = _select_guides_crowding(assoc, pareto_idx, F_norm, theta)

    assert np.allclose(guides[0], theta[0])
    assert np.allclose(guides[1], theta[0])


def test_select_guides_crowding_differs_from_ray_closest_champion():
    """4-member niche where the crowding-distance winner and the
    reference-ray-closest member (what _select_guides would pick) are
    different, known individuals -- proves the criterion swap actually
    changes which chromosome becomes the guide.

    Dataset (verified by hand against both formulas):
      P0=[.5,.5]  P1=[.1,.6]  P2=[.9,.4]  P3=[.3,.55]
      d_perp2 to ref_dir [1,1]: P0=0.0 (ray-closest) < P3=.0313 < P1=.125 ~= P2=.1251
      crowding distance (2 objectives): P0=1.5 (finite), P1=inf, P2=inf, P3=1.0
      -> argmax picks P1 (first index at the max/inf value) -- NOT P0.
    """
    assoc      = np.array([0, 0, 0, 0])
    pareto_idx = np.array([0, 1, 2, 3])
    F_norm     = np.array([
        [0.5, 0.5],
        [0.1, 0.6],
        [0.9, 0.4],
        [0.3, 0.55],
    ])
    theta = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ])

    guides = _select_guides_crowding(assoc, pareto_idx, F_norm, theta)

    ray_dir = np.array([[1.0, 1.0]])
    baseline_guides = _select_guides(assoc, pareto_idx, F_norm, ray_dir, theta)

    assert np.allclose(guides[0], theta[1])   # P1's theta, not P0's
    assert not np.allclose(guides[0], baseline_guides[0])   # differs from what _select_guides actually picks


# ── _supplement_from_archive_crowding ───────────────────────────────────

def test_supplement_from_archive_crowding_fills_uncovered_niche_by_crowding():
    """Niche 1 has no Pareto representative (pareto_assoc only covers niche
    0); 4 archive candidates are all associated to niche 1. Reuses the exact
    dataset from test_select_guides_crowding_differs_from_ray_closest_champion
    (verified: crowding winner = index 1, ray-closest winner = index 0) to
    show the archive-fallback path picks the same, different champion the
    base _supplement_from_archive would not."""
    guides_theta = np.array([[9.0, 9.0]])   # placeholder, must be overwritten
    assoc        = np.array([1])
    pareto_assoc = np.array([0])            # niche 0 covered, niche 1 is not
    ref_dirs     = np.array([[1.0, 0.0], [1.0, 1.0]])
    arch_F_norm  = np.array([
        [0.5, 0.5],
        [0.1, 0.6],
        [0.9, 0.4],
        [0.3, 0.55],
    ])
    arch_theta = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ])

    result = _supplement_from_archive_crowding(
        guides_theta, assoc, pareto_assoc, arch_theta, arch_F_norm, ref_dirs,
    )

    assert np.allclose(result[0], arch_theta[1])   # crowding winner
    assert not np.allclose(result[0], arch_theta[0])  # not the ray-closest winner


def test_supplement_from_archive_crowding_leaves_covered_niches_untouched():
    guides_theta = np.array([[7.0, 7.0]])
    expected     = guides_theta.copy()
    assoc        = np.array([0])
    pareto_assoc = np.array([0])   # niche 0 IS covered -> no supplementation
    ref_dirs     = np.array([[1.0, 0.0]])
    arch_F_norm  = np.array([[0.2, 0.3]])
    arch_theta   = np.array([[9.0, 9.0]])

    result = _supplement_from_archive_crowding(
        guides_theta, assoc, pareto_assoc, arch_theta, arch_F_norm, ref_dirs,
    )

    assert np.allclose(result[0], expected[0])
    assert not np.allclose(result[0], arch_theta[0])   # archive value was NOT pulled in


# ── _crowding_saturation_stats ──────────────────────────────────────────
# Diagnostic-only helper (not used by _select_guides_crowding itself) added
# after the final review of docs/superpowers/plans/
# 2026-08-03-qinsga3-crowding-distance-guide.md flagged that with M=4
# objectives and small niches, _crowding_distance can assign inf to every
# member of a niche, degenerating _select_guides_crowding's argmax into an
# arbitrary positional tie-break. This measures how often that happens.

def test_crowding_saturation_stats_ignores_single_member_niches():
    assoc      = np.array([0])
    pareto_idx = np.array([0])
    F_norm     = np.array([[1.0, 0.0]])

    n_multi, n_saturated = _crowding_saturation_stats(assoc, pareto_idx, F_norm)

    assert n_multi == 0
    assert n_saturated == 0


def test_crowding_saturation_stats_counts_fully_saturated_niche():
    """2-member niche where _crowding_distance gives inf to both (each is
    simultaneously the min and max on its own objective) -- the exact
    dataset from test_select_guides_crowding_two_member_niche_..., known to
    be fully saturated."""
    assoc      = np.array([0, 0])
    pareto_idx = np.array([0, 1])
    F_norm     = np.array([[0.0, 1.0], [1.0, 0.0]])

    n_multi, n_saturated = _crowding_saturation_stats(assoc, pareto_idx, F_norm)

    assert n_multi == 1
    assert n_saturated == 1


def test_crowding_saturation_stats_does_not_count_partially_saturated_niche():
    """4-member niche from test_select_guides_crowding_differs_from_ray_closest_champion:
    cd = [1.5, inf, inf, 1.0] -- 2 of 4 members are inf, but NOT all of them,
    so the argmax pick (P1) is still meaningfully driven by the criterion,
    not an arbitrary tie-break across the whole niche."""
    assoc      = np.array([0, 0, 0, 0])
    pareto_idx = np.array([0, 1, 2, 3])
    F_norm     = np.array([
        [0.5, 0.5],
        [0.1, 0.6],
        [0.9, 0.4],
        [0.3, 0.55],
    ])

    n_multi, n_saturated = _crowding_saturation_stats(assoc, pareto_idx, F_norm)

    assert n_multi == 1
    assert n_saturated == 0


def test_crowding_saturation_stats_aggregates_across_niches():
    """Two niches: niche 0 (2 members, fully saturated, same data as the
    fully-saturated test above) and niche 1 (4 members, partially saturated,
    same data as the partially-saturated test above) -- confirms per-niche
    results are summed correctly, not just correct for a single niche."""
    assoc      = np.array([0, 0, 1, 1, 1, 1])
    pareto_idx = np.array([0, 1, 2, 3, 4, 5])
    F_norm     = np.array([
        [0.0, 1.0],   # niche 0
        [1.0, 0.0],   # niche 0
        [0.5, 0.5],   # niche 1
        [0.1, 0.6],   # niche 1
        [0.9, 0.4],   # niche 1
        [0.3, 0.55],  # niche 1
    ])

    n_multi, n_saturated = _crowding_saturation_stats(assoc, pareto_idx, F_norm)

    assert n_multi == 2
    assert n_saturated == 1


# ── _build_g_constraints ─────────────────────────────────────────────────
# Remedy G: duplicates IRPProblem._evaluate's G-list construction
# (Solvers/NSGA3/problem.py:67-85) so _evaluate_with_repair can compute G
# for a REPAIRED route_result without calling IRPProblem._evaluate itself
# (which decodes and builds routes internally, with no repair hook). This
# test hand-verifies the duplicate against the same published formula, to
# catch transcription drift.

def test_build_g_constraints_matches_hand_verified_formula():
    """2 periods, 1 client. Hand-computed (mirrors problem.py:67-85 exactly):
      t=1: ret=20 -> [20-100=-80, 0-20=-20]
      t=2: ret=30 -> [30-100=-70, 0-30=-30]
      client 1: t=1: cum_del=8,  cum_dem=10 -> [10-8=2]
                t=2: cum_del=15, cum_dem=15 -> [15-15=0]
      t=1: [depot_stock[1].frigo-50=10-50=-40, depot_stock[1].nonfrigo-50=5-50=-45]
      t=2: [depot_stock[2].frigo-50=60-50=10, depot_stock[2].nonfrigo-50=2-50=-48]
    """
    sets_ = {"clients": [1], "T": [1, 2]}
    params_ = {
        "q_lt": {(1, 1): 10, (1, 2): 5},
        "tau_min": 0.0,
        "tau_max": 100.0,
        "I_O_max_frigo": 50.0,
        "I_O_max_nonfrigo": 50.0,
    }
    route_result = {
        "tau_return": {1: 20.0, 2: 30.0},
        "actual_qty": {(1, 1): 8, (1, 2): 7},
        "depot_stock": {
            1: {"frigo": 10.0, "nonfrigo": 5.0},
            2: {"frigo": 60.0, "nonfrigo": 2.0},
        },
    }

    G = _build_g_constraints(route_result, sets_, params_)

    assert G == [-80, -20, -70, -30, 2, 0, -40, -45, 10, -48]


def test_build_g_constraints_matches_real_irpproblem_evaluate():
    """Live drift guard: _build_g_constraints is a deliberate duplication of
    IRPProblem._evaluate's G-list construction (problem.py:67-85, see
    docs/superpowers/specs/2026-08-04-qinsga3-route-repair-design.md) --
    this compares it against the REAL IRPProblem on a real decoded
    chromosome, catching drift if problem.py's G-list formula ever changes
    (the hand-verified test above catches transcription bugs in this file,
    but can't catch drift in the other file it's meant to stay in sync
    with)."""
    from models.parametres import load_instance
    from Solvers.NSGA3.problem import IRPProblem
    from Solvers.NSGA3.decoder import decode_chromosome, build_routes

    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sets_, params_ = load_instance(os.path.join(project_dir, "data", "instance_3_clients.json"))
    problem = IRPProblem(sets_, params_)

    rng = np.random.default_rng(0)
    x = rng.uniform(problem.xl, problem.xu)

    out = {}
    problem._evaluate(x, out)
    expected_G = out["G"]

    quantities, priorities = decode_chromosome(x, sets_)
    route_result = build_routes(quantities, sets_, params_, priorities)
    actual_G = _build_g_constraints(route_result, sets_, params_)

    assert actual_G == expected_G


# ── _repair_pareto_front ─────────────────────────────────────────────────
# Practical counterpart to use_route_repair (run_qinsga3's repair_final_front
# parameter): repairs only the returned Pareto front, once, after the
# generational loop -- never touches the loop's own runtime.

def test_repair_pareto_front_matches_per_chromosome_evaluate_with_repair():
    """3 distinct chromosomes (real instance_3_clients.json) -- the batch
    helper's output must match calling _evaluate_with_repair on each
    chromosome individually, in the same order, and pareto_X itself must
    be returned unmodified by the caller's own reference (this function
    never touches its pareto_X argument -- it only ever reads from it)."""
    from models.parametres import load_instance
    from Solvers.NSGA3.problem import IRPProblem

    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sets_, params_ = load_instance(os.path.join(project_dir, "data", "instance_3_clients.json"))
    problem = IRPProblem(sets_, params_)

    pareto_X = np.array([
        np.random.default_rng(seed).uniform(problem.xl, problem.xu)
        for seed in (0, 1, 2)
    ])
    pareto_X_snapshot = pareto_X.copy()

    F_arr, G_arr = _repair_pareto_front(pareto_X, sets_, params_)

    assert F_arr.shape[0] == 3
    assert G_arr.shape[0] == 3
    for idx, x in enumerate(pareto_X):
        expected_F, expected_G = _evaluate_with_repair(x, sets_, params_)
        assert np.allclose(F_arr[idx], expected_F)
        assert (G_arr[idx] == expected_G).all()
    assert np.array_equal(pareto_X, pareto_X_snapshot)   # Baldwinian: pareto_X untouched


# ── _generate_offspring_batch / _eliminate_duplicates_refill ───────────────
# Duplicate elimination matching pymoo's NSGA-III default (eliminate_
# duplicates=True): a fairness audit found QI-NSGA-III's hand-rolled
# offspring generation had no equivalent check anywhere in the main
# generational loop -- see _eliminate_duplicates_refill's own docstring.

def _dup_test_fixture():
    """Real 3-client instance + real pymoo SBX/PM ops -- the same operators
    run_qinsga3's own generation loop uses, at production eta/prob values."""
    from models.parametres import load_instance
    from Solvers.NSGA3.problem import IRPProblem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM

    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sets_, params_ = load_instance(os.path.join(project_dir, "data", "instance_3_clients.json"))
    problem = IRPProblem(sets_, params_)
    sbx_op  = SBX(prob=0.9, eta=20.0)
    pm_op   = PM(prob_var=1.0 / problem.n_var, eta=20.0)
    return problem, sbx_op, pm_op


def test_generate_offspring_batch_returns_exactly_n_individuals_for_various_n():
    problem, sbx_op, pm_op = _dup_test_fixture()
    pop_size = 20
    rng      = np.random.default_rng(0)
    X_rotated = rng.uniform(problem.xl, problem.xu, (pop_size, problem.n_var))
    G_parent  = np.zeros((pop_size, 4))   # all feasible -> tournament is a coin flip

    for n in (1, 2, 3, 7, pop_size):
        batch = _generate_offspring_batch(
            n, X_rotated, G_parent, pop_size, 0.9, sbx_op, pm_op, problem,
            problem.xl, problem.xu, rng,
        )
        assert batch.shape == (n, problem.n_var)


def test_generate_offspring_batch_handles_odd_pop_size_without_dropping_individual():
    """Regression test: pop_size can be bumped to an odd n_ref_dirs count
    (e.g. 165 Das-Dennis directions for 4 objectives) when the requested
    pop_size is smaller. n_pairs*2 must never exceed the pop_size winners
    _tournament_select_parents actually returns -- this used to raise
    ValueError: cannot reshape array of size N into shape (n_pairs, 2)."""
    problem, sbx_op, pm_op = _dup_test_fixture()
    pop_size = 7   # odd
    rng      = np.random.default_rng(0)
    X_rotated = rng.uniform(problem.xl, problem.xu, (pop_size, problem.n_var))
    G_parent  = np.zeros((pop_size, 4))

    batch = _generate_offspring_batch(
        pop_size, X_rotated, G_parent, pop_size, 0.9, sbx_op, pm_op, problem,
        problem.xl, problem.xu, rng,
    )
    assert batch.shape == (pop_size, problem.n_var)


def test_eliminate_duplicates_refill_removes_self_duplicate_within_batch():
    problem, sbx_op, pm_op = _dup_test_fixture()
    pop_size = 10
    rng      = np.random.default_rng(1)
    X_parent  = rng.uniform(problem.xl, problem.xu, (pop_size, problem.n_var))
    X_rotated = X_parent.copy()
    G_parent  = np.zeros((pop_size, 4))

    X_offspring = rng.uniform(problem.xl, problem.xu, (pop_size, problem.n_var))
    X_offspring[1] = X_offspring[0]   # force an exact within-batch duplicate

    result = _eliminate_duplicates_refill(
        X_offspring, X_parent, X_rotated, G_parent, pop_size, 0.9,
        sbx_op, pm_op, problem, problem.xl, problem.xu, rng,
    )

    assert result.shape == X_offspring.shape
    D = np.linalg.norm(result[:, None, :] - result[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    assert (D > 1e-16).all(), "duplicate survived elimination"


def test_eliminate_duplicates_refill_removes_duplicate_against_parent():
    problem, sbx_op, pm_op = _dup_test_fixture()
    pop_size = 10
    rng      = np.random.default_rng(2)
    X_parent  = rng.uniform(problem.xl, problem.xu, (pop_size, problem.n_var))
    X_rotated = X_parent.copy()
    G_parent  = np.zeros((pop_size, 4))

    X_offspring    = rng.uniform(problem.xl, problem.xu, (pop_size, problem.n_var))
    X_offspring[3] = X_parent[5]   # force an exact duplicate of a parent

    result = _eliminate_duplicates_refill(
        X_offspring, X_parent, X_rotated, G_parent, pop_size, 0.9,
        sbx_op, pm_op, problem, problem.xl, problem.xu, rng,
    )

    assert result.shape == X_offspring.shape
    D = np.linalg.norm(result[:, None, :] - X_parent[None, :, :], axis=-1)
    assert (D > 1e-16).all(), "offspring duplicating a parent survived elimination"


def test_eliminate_duplicates_refill_no_duplicates_is_a_noop():
    """When the batch already has no duplicates, the function must return
    it unchanged (same values, not just same shape) -- no wasted retries."""
    problem, sbx_op, pm_op = _dup_test_fixture()
    pop_size = 10
    rng      = np.random.default_rng(3)
    X_parent  = rng.uniform(problem.xl, problem.xu, (pop_size, problem.n_var))
    X_rotated = X_parent.copy()
    G_parent  = np.zeros((pop_size, 4))

    X_offspring = rng.uniform(problem.xl, problem.xu, (pop_size, problem.n_var))
    X_offspring_snapshot = X_offspring.copy()

    result = _eliminate_duplicates_refill(
        X_offspring, X_parent, X_rotated, G_parent, pop_size, 0.9,
        sbx_op, pm_op, problem, problem.xl, problem.xu, rng,
    )

    assert np.array_equal(result, X_offspring_snapshot)
