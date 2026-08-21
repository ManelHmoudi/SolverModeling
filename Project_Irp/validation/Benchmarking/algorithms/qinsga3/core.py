"""Lean QINSGA-III generational loop for synthetic benchmark problems (DTLZ/MaF).

Mirrors Solvers/QINSGA3/algorithm.py::run_qinsga3()'s algorithm exactly (same
generation order: measure parent -> evaluate -> penalise -> non-dominated
sort -> archive update -> normalise (sharing survival.norm's running
ideal/nadir, same as the production loop) -> assign ref dirs -> select
guides (+ supplement from archive) -> rotate (theta-space) -> measure ->
SBX+PM variation (X-space) -> re-encode -> evaluate offspring -> archive
update -> elitist survival (merge parent+offspring, keep best pop_size via
pymoo's own ReferenceDirectionSurvival.do()) -> migrate), but:

  - evaluates the whole population in one vectorised pymoo
    Problem.evaluate() call per generation instead of a multiprocessing
    pool — DTLZ/MaF evaluation is cheap, so a process pool would only add
    overhead (unlike the IRP's expensive simulation-based evaluation).
  - takes any pymoo Problem instance instead of the IRP-specific
    IRPProblem.

All non-dominated-sorting, normalisation, niching, archive, crowding and
theta<->X encoding helpers are imported directly from
Solvers/QINSGA3/algorithm.py — not duplicated — so a future fix to that
shared math applies to both the IRP path and this benchmark path
automatically. Solvers/QINSGA3/algorithm.py itself is never modified by
this module. The generational LOOP ITSELF, however, is a separate copy, not
shared code -- a fix to algorithm.py's loop (e.g. its eval-caching, see
below) does NOT apply here automatically and must be ported by hand, as
done here.

Eval-caching (ported from Solvers/QINSGA3/algorithm.py, kept in sync with
it): the parent population entering a generation is exactly the theta
survival.do() selected last generation, whose F/G were already computed as
part of that generation's merged_pop. With noise_scale == 0.0, measure() is
deterministic, so re-running _eval_batch on this same population every
generation just recomputes already-known numbers -- doubling this
benchmark's real evaluation budget relative to NSGA-III's for no
algorithmic reason. F/G are now cached from survival.do()'s result instead,
cutting real per-generation evaluation cost from ~2x NSGA-III's down to
~1.1x at equal generation counts (one real re-evaluation every
migration_period generations, when migration overwrites qpop.theta
post-survival). Requires survival.do() to be given the REAL (unpenalised)
F/G, not F_pen -- for these unconstrained DTLZ/MaF problems (G always
empty) _penalised_F(F, G) == F exactly, so this is numerically a no-op
versus the previous F_pen-based version. noise_scale defaults to 0.02 here
(see below), so most default benchmark runs do NOT get this speedup unless
called with noise_scale=0.0 explicitly -- matching algorithm.py, where the
production IRP default noise_scale=0.0 is what makes its own cache apply on
almost every generation.

Benchmark-only addition: `escape_prob`. QuantumPopulation always starts
every individual at theta=pi/4, i.e. x ~= midpoint of [xl, xu] for every
gene at generation 0 (confirmed better for the IRP — see
sensitivity/compare_init_spread.py — since it lands closer to feasibility
on a heavily constrained problem). DTLZ4 and MaF5 raise their "position"
variables to a bias exponent (alpha=100): 0.5**100 ~= 8e-31, so 2 of 3(4)
objectives collapse to 0 for the ENTIRE initial population before any
search dynamics run — confirmed by instrumentation (population diversity
logged every 30 generations showed the collapse present at generation 0
already). The X-space PM mutation is a purely local perturbation and can
never climb from x~0.5 to the x~1 region these two problems need. The old
theta-space "strong" mutation (reset to Uniform(0, pi/2), removed when
crossover/mutation moved to X-space — see Solvers/QINSGA3/algorithm.py's
docstring) used to provide that escape by accident. `escape_prob` restores
just that one capability, applied directly on theta after the X-space
variation step, without reverting the X-space fix itself: a 5-run ablation
(escape_prob = (2/n_var)*0.15, matching the old total per-gene reset rate)
took DTLZ4 M3 mean IGD from 0.945922 (collapsed, std=0.000000 across 5
seeds) to 0.071878 (std=0.004177) and MaF5 M3 from 4.888431 to 0.330269,
while DTLZ2 M3 (unbiased control) only moved from 0.067973 to 0.072974 —
not a meaningful regression. Not applied to Solvers/QINSGA3/algorithm.py:
the IRP has no such bias-exponent transform, so this escape mechanism has
no analogous problem to fix there.
"""

from __future__ import annotations

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.core.population import Population
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

from Solvers.QINSGA3.algorithm import (
    _archive_update,
    _assign_ref_dirs,
    _crowding_trim,
    _encode_theta,
    _migrate,
    _normalise_F,
    _penalised_F,
    _select_guides,
    _supplement_from_archive,
)
from Solvers.QINSGA3.chromosome import QuantumPopulation


def run_qinsga3_generic(
    problem,
    ref_dirs: np.ndarray,
    pop_size: int,
    max_gen: int,
    alpha_max: float,
    alpha_min: float,
    p_cross: float,
    eta_cross: float,
    p_mut: float,
    eta_mut: float,
    migration_period: int,
    n_migrate: int,
    seed: int,
    rotation_type: str = "tanh",
    noise_scale: float = 0.02,
    escape_prob: float = 0.0,
    callback=None,
) -> np.ndarray:
    """Run one QINSGA-III instance on a pymoo Problem. Returns the final Pareto front's F.

    callback (default None): if given, called once per generation as
    callback(gen, n_eval, F_survived) -- gen is 0-indexed, n_eval is the
    REAL cumulative evaluation count so far (counted directly, mirroring
    Solvers/QINSGA3/algorithm.py::run_qinsga3's own _n_eval_used counter,
    not a formula -- respects this loop's own eval-caching, see below),
    F_survived is the current generation's elitist-survived population's
    objective matrix (pop_size x n_obj), the same population qpop.theta
    holds after this generation's survival step. Exists so a convergence
    trajectory (e.g. IGD vs evaluations, matching Cui et al. 2025's Fig. 3)
    can be tracked without pymoo's own save_history= (which this loop,
    being a separate copy of algorithm.py's generational structure rather
    than a pymoo Algorithm subclass, has no access to).

    noise_scale controls QuantumPopulation's measurement diversity noise
    (default 0.02, the IRP-tuned value — see Solvers/QINSGA3/chromosome.py).
    That noise exists to stop nearby theta values collapsing to identical
    integer routes after the IRP decoder rounds to integers; DTLZ/MaF have
    no such rounding step, so it can be set to 0 here without affecting the
    IRP path (whose QuantumPopulation calls never pass this argument).

    eta_cross/eta_mut are X-space SBX/PM distribution indices — crossover
    and mutation happen in decision-variable space (matching NSGA-III's own
    operators), while the rotation gate stays in theta-space.

    escape_prob: per-gene probability of resetting theta to a fresh
    Uniform(0, pi/2) draw after the X-space variation step (default 0 =
    no-op, identical to not having this mechanism at all). See the module
    docstring — fixes the DTLZ4/MaF5 bias-transform collapse.
    """
    rng = np.random.default_rng(seed)

    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)
    n_genes = problem.n_var

    qpop     = QuantumPopulation(
        pop_size, n_genes, xl, xu, rng=rng,
        rotation_type=rotation_type, noise_scale=noise_scale,
    )
    sorter   = NonDominatedSorting()
    survival = ReferenceDirectionSurvival(ref_dirs)
    sbx_op   = SBX(prob=p_cross, eta=eta_cross)
    pm_op    = PM(prob=p_mut,    eta=eta_mut)

    arch_X: list[np.ndarray] = []
    arch_F: list[np.ndarray] = []
    arch_theta: list[np.ndarray] = []
    _MAX_ARCHIVE = 500

    _n_eval_used = 0

    def _eval_batch(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nonlocal _n_eval_used
        F, G = problem.evaluate(X, return_values_of=["F", "G"])
        if G is None or (hasattr(G, "size") and G.size == 0):
            G = np.zeros((X.shape[0], 0))
        _n_eval_used += len(X)
        return np.asarray(F), np.asarray(G)

    # Cache of the previous generation's survived F/G, mirroring
    # Solvers/QINSGA3/algorithm.py's own cache (see that module's docstring
    # "Performance" section): when noise_scale == 0.0, measure() is a
    # deterministic function of theta, so the parent entering a generation
    # -- if unchanged since it was last measured+evaluated as part of the
    # PREVIOUS generation's merged_pop -- is bit-identical to what
    # _eval_batch already returned F/G for last generation. Re-evaluating it
    # here recomputes already-known numbers, silently doubling this
    # benchmark's real evaluation budget relative to NSGA-III for no reason
    # (this loop is a separate copy of algorithm.py's generation loop, not
    # shared code, so algorithm.py's own fix does not apply here
    # automatically). Invalidated after generation 0 and after migration
    # (which overwrites qpop.theta post-survival, same as production).
    _prev_survived_F: np.ndarray | None = None
    _prev_survived_G: np.ndarray | None = None
    _prev_cache_valid = False

    for gen in range(max_gen):
        theta_parent = qpop.theta.copy()
        X_parent     = qpop.measure()
        if _prev_cache_valid and noise_scale == 0.0:
            F_parent, G_parent = _prev_survived_F, _prev_survived_G
        else:
            F_parent, G_parent = _eval_batch(X_parent)
        F_pen_parent = _penalised_F(F_parent, G_parent)

        pareto_idx = sorter.do(F_pen_parent)[0]
        _archive_update(
            X_parent[pareto_idx], F_parent[pareto_idx], G_parent[pareto_idx],
            theta_parent[pareto_idx], arch_X, arch_F, arch_theta, _MAX_ARCHIVE,
        )

        # Share the SAME ideal/nadir as the elitist survival step below
        # (pymoo's own ReferenceDirectionSurvival.norm, monotonic across
        # generations) for guide selection and niching -- see
        # Solvers/QINSGA3/algorithm.py::_normalise_F's docstring. Not yet
        # populated on generation 0 (before survival.do() has run once), so
        # that first generation falls back to a from-scratch estimate
        # exactly as before.
        if survival.norm.nadir_point is None:
            F_norm = _normalise_F(F_pen_parent)
        else:
            F_norm = _normalise_F(F_pen_parent, survival.norm.ideal_point, survival.norm.nadir_point)
        assoc  = _assign_ref_dirs(F_norm, ref_dirs)
        guides_theta = _select_guides(assoc, pareto_idx, F_norm, ref_dirs, qpop.theta)

        arch_theta_arr = None
        arch_F_norm    = None
        if len(arch_X) >= 4:
            arch_theta_arr = np.array(arch_theta)
            if survival.norm.nadir_point is None:
                arch_F_norm = _normalise_F(np.array(arch_F))
            else:
                arch_F_norm = _normalise_F(np.array(arch_F), survival.norm.ideal_point, survival.norm.nadir_point)
            pareto_assoc   = assoc[pareto_idx]
            guides_theta   = _supplement_from_archive(
                guides_theta, assoc, pareto_assoc,
                arch_theta_arr, arch_F_norm, ref_dirs,
            )

        alpha = alpha_min + (alpha_max - alpha_min) * (1.0 - gen / max_gen)

        # --- Quantum step: rotation stays in theta-space -----------------
        qpop.rotate(guides_theta, alpha)
        X_rotated = qpop.measure()

        # --- Variation step: SBX + PM in X-SPACE (matches NSGA-III) -----
        if p_cross > 0.0:
            idx     = rng.permutation(pop_size)
            n_pairs = pop_size // 2
            pairs   = idx[: n_pairs * 2].reshape(n_pairs, 2)
            X_pairs = np.transpose(X_rotated[pairs], (1, 0, 2))
            Q       = sbx_op._do(problem, X_pairs, random_state=rng)
            X_rotated[pairs[:, 0]] = Q[0]
            X_rotated[pairs[:, 1]] = Q[1]
        X_varied = np.clip(pm_op._do(problem, X_rotated, random_state=rng), xl, xu)

        theta_offspring = _encode_theta(X_varied, xl, xu)

        # --- Escape mutation: fixes the DTLZ4/MaF5 bias-transform collapse,
        # see module docstring. No-op when escape_prob=0.0 (default). -----
        if escape_prob > 0.0:
            escape_mask = rng.random(theta_offspring.shape) < escape_prob
            if escape_mask.any():
                theta_offspring[escape_mask] = rng.uniform(0.0, np.pi / 2.0, int(escape_mask.sum()))

        qpop.theta      = theta_offspring
        X_offspring     = qpop.measure()
        F_offspring, G_offspring = _eval_batch(X_offspring)
        F_pen_offspring = _penalised_F(F_offspring, G_offspring)

        off_pareto_idx = sorter.do(F_pen_offspring)[0]
        _archive_update(
            X_offspring[off_pareto_idx], F_offspring[off_pareto_idx], G_offspring[off_pareto_idx],
            theta_offspring[off_pareto_idx], arch_X, arch_F, arch_theta, _MAX_ARCHIVE,
        )

        # --- Elitist survival: merge parent + offspring, keep best pop_size
        # via pymoo's own NSGA-III niching survival — the same elitist
        # replacement NSGA-III (pymoo) itself uses every generation.
        #
        # survival.do() (pymoo's public entry point), not survival._do()
        # directly — matches Solvers/QINSGA3/algorithm.py's production loop.
        # Uses the REAL F/G (not F_pen) here, also matching algorithm.py —
        # for these unconstrained DTLZ/MaF problems (n_ieq_constr=0, G always
        # empty) _penalised_F(F, G) == F exactly (zero rows summed = zero
        # penalty), so this is numerically a no-op versus the previous
        # F_pen-based version; it's needed to let survived.get("F")/("G")
        # below feed the eval-cache above with the true (unpenalised)
        # objectives _archive_update expects, and keeps this benchmark copy
        # an honest mirror of the production loop rather than a second
        # implementation that only happens to agree on unconstrained
        # problems.
        theta_pool  = np.vstack([theta_parent, theta_offspring])
        F_true_pool = np.vstack([F_parent, F_offspring])
        G_pool      = np.vstack([G_parent, G_offspring])
        merged_pop  = Population.new(X=theta_pool, F=F_true_pool, G=G_pool)
        survived    = survival.do(problem, merged_pop, n_survive=pop_size, random_state=rng)
        qpop.theta  = np.clip(np.asarray(survived.get("X"), dtype=float), 0.0, np.pi / 2.0)
        _prev_survived_F  = np.asarray(survived.get("F"), dtype=float)
        _prev_survived_G  = np.asarray(survived.get("G"), dtype=float)
        _prev_cache_valid = True

        if callback is not None:
            # Non-dominated subset only, matching pymoo's own res.history[i].opt
            # convention (NSGA-III/MOEA-D's own save_history= snapshots) so a
            # cross-algorithm convergence comparison is apples-to-apples.
            gen_nd_idx = sorter.do(_prev_survived_F, only_non_dominated_front=True)
            callback(gen, _n_eval_used, _prev_survived_F[gen_nd_idx])

        if (arch_F_norm is not None
                and migration_period > 0
                and gen % migration_period == 0):
            _migrate(
                qpop, arch_theta_arr, arch_F_norm,
                assoc, ref_dirs, rng, n_migrate=n_migrate,
            )
            _prev_cache_valid = False

    X_final = qpop.measure()
    F_final, G_final = _eval_batch(X_final)
    F_pen_final = _penalised_F(F_final, G_final)
    final_pareto_idx = sorter.do(F_pen_final)[0]
    _archive_update(
        X_final[final_pareto_idx], F_final[final_pareto_idx],
        G_final[final_pareto_idx], qpop.theta[final_pareto_idx],
        arch_X, arch_F, arch_theta, _MAX_ARCHIVE,
    )

    if arch_X:
        _, arch_F_arr, _ = _crowding_trim(
            np.array(arch_X), np.array(arch_F), np.array(arch_theta), pop_size,
        )
        return arch_F_arr

    return F_final[final_pareto_idx]
