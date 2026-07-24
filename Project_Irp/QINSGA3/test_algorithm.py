import numpy as np

from QINSGA3.algorithm import _normalise_F, _normalise_stats, _select_guides


def test_normalise_F_matches_normalise_stats_composition():
    """_normalise_F(F) must equal (F - ideal) / denom for the ideal/denom
    _normalise_stats(F) returns -- guards the refactor that splits them apart."""
    F = np.array([[0.0, 2.0], [1.0, 0.0], [2.0, 1.0]])

    ideal, denom = _normalise_stats(F)
    expected = (F - ideal) / denom

    np.testing.assert_allclose(_normalise_F(F), expected)


REF_DIRS_2 = np.array([[1.0, 0.0], [0.0, 1.0]])


def test_select_guides_archive_better_than_front_wins():
    """Archive holds a strictly closer representative for niche 0 than the
    current Pareto front -> the guide for niche 0 comes from the archive."""
    assoc      = np.array([0, 0, 1, 1])
    pareto_idx = np.array([0, 2])
    qpop_theta = np.array([[0.1], [0.2], [0.3], [0.4]])
    F_norm     = np.array([
        [0.5, 0.9],   # individual 0: front's niche-0 pick, d_perp^2 to ray0 = 0.81
        [0.0, 0.0],
        [0.9, 0.5],   # individual 2: front's niche-1 pick, d_perp^2 to ray1 = 0.81
        [0.0, 0.0],
    ])
    arch_theta  = np.array([[0.9]])
    arch_F_norm = np.array([[0.5, 0.1]])   # d_perp^2 to ray0 = 0.01 < front's 0.81

    guides = _select_guides(
        assoc, pareto_idx, F_norm, REF_DIRS_2, qpop_theta,
        arch_theta=arch_theta, arch_F_norm=arch_F_norm,
    )

    assert guides[0, 0] == 0.9   # niche 0 -> archive wins
    assert guides[1, 0] == 0.9
    assert guides[2, 0] == 0.3   # niche 1 -> unaffected, front's own pick
    assert guides[3, 0] == 0.3


def test_select_guides_front_beats_worse_archive():
    """Archive's representative for niche 0 is farther than the front's ->
    guide stays on the front's pick (no regression from today's behaviour)."""
    assoc      = np.array([0, 0])
    pareto_idx = np.array([0])
    qpop_theta = np.array([[0.1], [0.2]])
    F_norm     = np.array([
        [0.5, 0.9],   # front pick, d_perp^2 to ray0 = 0.81
        [0.0, 0.0],
    ])
    arch_theta  = np.array([[0.9]])
    arch_F_norm = np.array([[1.5, 0.95]])   # d_perp^2 to ray0 = 0.9025 > front's 0.81

    guides = _select_guides(
        assoc, pareto_idx, F_norm, REF_DIRS_2, qpop_theta,
        arch_theta=arch_theta, arch_F_norm=arch_F_norm,
    )

    assert guides[0, 0] == 0.1
    assert guides[1, 0] == 0.1


def test_select_guides_uncovered_niche_uses_archive():
    """A niche with no Pareto-front representative but an archive one ->
    guide comes from the archive (matches old _supplement_from_archive)."""
    ref_dirs   = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assoc      = np.array([0, 2, 2])
    pareto_idx = np.array([0])       # only niche 0 covered by the front
    qpop_theta = np.array([[0.1], [0.2], [0.3]])
    F_norm     = np.array([
        [0.5, 0.9],
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    arch_theta  = np.array([[0.7]])
    arch_F_norm = np.array([[1.0, 1.0]])   # exactly on ray2 = (1,1)

    guides = _select_guides(
        assoc, pareto_idx, F_norm, ref_dirs, qpop_theta,
        arch_theta=arch_theta, arch_F_norm=arch_F_norm,
    )

    assert guides[1, 0] == 0.7
    assert guides[2, 0] == 0.7


def test_select_guides_without_archive_matches_front_only_behaviour():
    """No archive passed (None) -> identical output to front-only selection,
    i.e. the behaviour used before the archive reaches its 4-entry threshold."""
    assoc      = np.array([0, 0, 1, 1])
    pareto_idx = np.array([0, 2])
    qpop_theta = np.array([[0.1], [0.2], [0.3], [0.4]])
    F_norm     = np.array([
        [0.5, 0.9],
        [0.0, 0.0],
        [0.9, 0.5],
        [0.0, 0.0],
    ])

    guides = _select_guides(assoc, pareto_idx, F_norm, REF_DIRS_2, qpop_theta)

    assert guides[0, 0] == 0.1
    assert guides[2, 0] == 0.3


def test_select_guides_niche_with_neither_front_nor_archive_uses_global_fallback():
    """A niche with no Pareto-front representative and no archive representative
    either (even though the archive itself is available) -> falls back to the
    global fallback (closest-to-origin Pareto member), same as when there's no
    archive at all."""
    assoc      = np.array([0, 1])
    pareto_idx = np.array([0])       # only niche 0 covered by the front
    qpop_theta = np.array([[0.1], [0.2]])
    F_norm     = np.array([
        [0.5, 0.1],
        [0.0, 0.0],
    ])
    arch_theta  = np.array([[0.9]])
    arch_F_norm = np.array([[0.5, 0.05]])   # assigned to niche 0, not niche 1

    guides = _select_guides(
        assoc, pareto_idx, F_norm, REF_DIRS_2, qpop_theta,
        arch_theta=arch_theta, arch_F_norm=arch_F_norm,
    )

    assert guides[1, 0] == 0.1   # niche 1: neither front nor archive -> global fallback


from QINSGA3.algorithm import _update_niche_stagnation


def test_update_niche_stagnation_triggers_after_t_generations():
    """A niche's guide unchanged for exactly t_stagnation consecutive calls
    ends up in the returned stagnant set; one call short does not."""
    assoc = np.array([0, 0])
    guide_theta = np.array([[0.5], [0.5]])
    history, counters = {}, {}
    t_stagnation = 3

    history, counters, stagnant = _update_niche_stagnation(
        assoc, guide_theta, history, counters, t_stagnation)
    assert counters[0] == 0
    assert 0 not in stagnant

    for _ in range(2):
        history, counters, stagnant = _update_niche_stagnation(
            assoc, guide_theta, history, counters, t_stagnation)
    assert counters[0] == 2
    assert 0 not in stagnant   # t_stagnation=3 not yet reached

    history, counters, stagnant = _update_niche_stagnation(
        assoc, guide_theta, history, counters, t_stagnation)
    assert counters[0] == 3
    assert 0 in stagnant


def test_update_niche_stagnation_resets_when_guide_changes():
    """The counter resets to 0 the moment a niche's guide theta changes."""
    assoc = np.array([0, 0])
    history, counters = {}, {}
    t_stagnation = 2

    guide_a = np.array([[0.5], [0.5]])
    guide_b = np.array([[0.7], [0.7]])

    history, counters, _ = _update_niche_stagnation(
        assoc, guide_a, history, counters, t_stagnation)
    history, counters, stagnant = _update_niche_stagnation(
        assoc, guide_a, history, counters, t_stagnation)
    assert counters[0] == 1
    assert 0 not in stagnant

    history, counters, stagnant = _update_niche_stagnation(
        assoc, guide_b, history, counters, t_stagnation)
    assert counters[0] == 0
    assert 0 not in stagnant


from QINSGA3.algorithm import _diversity_preserve_mask


def test_diversity_preserve_mask_resets_similar_converged_neighbour():
    """Within a stagnant niche: the champion (closest to the reference ray)
    is kept; a converged individual similar to it (eq. 12) is reset; a
    converged individual NOT similar to it is left alone; an unconverged
    individual (eq. 11) is left alone regardless of proximity."""
    assoc = np.array([0, 0, 0, 0])
    ref_dirs = np.array([[1.0, 0.0], [0.0, 1.0]])
    qpop_theta = np.array([
        [0.05],   # champion: converged, closest to ray0
        [0.06],   # converged AND similar to champion -> reset
        [1.52],   # converged but NOT similar to champion -> kept
        [0.78],   # NOT converged (near pi/4) -> kept regardless of distance
    ])
    F_norm = np.array([
        [0.9, 0.05],
        [0.5, 0.5],
        [0.5, 0.6],
        [0.0, 0.0],
    ])

    mask = _diversity_preserve_mask(
        assoc, qpop_theta, F_norm, ref_dirs, stagnant_niches={0},
        gamma=0.99, delta=0.1,
    )

    assert mask.tolist() == [False, True, False, False]


def test_diversity_preserve_mask_ignores_non_stagnant_niches():
    """A niche not in stagnant_niches is never touched, even if its
    individuals would otherwise satisfy the converged+similar criteria."""
    assoc = np.array([0, 0])
    ref_dirs = np.array([[1.0, 0.0], [0.0, 1.0]])
    qpop_theta = np.array([[0.05], [0.06]])
    F_norm = np.array([[0.9, 0.05], [0.5, 0.5]])

    mask = _diversity_preserve_mask(
        assoc, qpop_theta, F_norm, ref_dirs, stagnant_niches=set(),
        gamma=0.99, delta=0.1,
    )

    assert mask.tolist() == [False, False]


def test_diversity_preserve_mask_skips_niche_with_fewer_than_two_converged():
    """A stagnant niche with fewer than 2 converged individuals has nothing
    to compare, so nothing is reset."""
    assoc = np.array([0, 0])
    ref_dirs = np.array([[1.0, 0.0], [0.0, 1.0]])
    qpop_theta = np.array([[0.05], [0.78]])   # only one converged (0.78 is not)
    F_norm = np.array([[0.9, 0.05], [0.0, 0.0]])

    mask = _diversity_preserve_mask(
        assoc, qpop_theta, F_norm, ref_dirs, stagnant_niches={0},
        gamma=0.99, delta=0.1,
    )

    assert mask.tolist() == [False, False]
