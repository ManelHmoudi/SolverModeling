"""Lean QINSGA-III generational loop for synthetic benchmark problems (DTLZ/MaF).

Mirrors QINSGA3/algorithm.py::run_qinsga3()'s algorithm exactly (same
generation order: measure -> evaluate -> penalise -> non-dominated sort ->
archive update -> normalise -> assign ref dirs -> select guides -> supplement
from archive -> rotate -> crossover -> mutate -> migrate), but:

  - evaluates the whole population in one vectorised pymoo
    Problem.evaluate() call per generation instead of a multiprocessing
    pool — DTLZ/MaF evaluation is cheap, so a process pool would only add
    overhead (unlike the IRP's expensive simulation-based evaluation).
  - takes any pymoo Problem instance instead of the IRP-specific
    IRPProblem.

All non-dominated-sorting, normalisation, niching, archive and crowding
helpers are imported directly from QINSGA3/algorithm.py — not duplicated —
so a future fix to that shared math applies to both the IRP path and this
benchmark path automatically. QINSGA3/algorithm.py itself is never modified
by this module.
"""

from __future__ import annotations

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from QINSGA3.algorithm import (
    _archive_update,
    _assign_ref_dirs,
    _crowding_trim,
    _migrate,
    _normalise_F,
    _penalised_F,
    _select_guides,
    _supplement_from_archive,
)
from QINSGA3.chromosome import QuantumPopulation


def run_qinsga3_generic(
    problem,
    ref_dirs: np.ndarray,
    pop_size: int,
    max_gen: int,
    alpha_max: float,
    alpha_min: float,
    p_mut: float,
    p_mut_strong: float,
    mut_sigma: float,
    p_cross: float,
    eta_cross: float,
    migration_period: int,
    n_migrate: int,
    seed: int,
    rotation_type: str = "tanh",
    noise_scale: float = 0.02,
) -> np.ndarray:
    """Run one QINSGA-III instance on a pymoo Problem. Returns the final Pareto front's F.

    noise_scale controls QuantumPopulation's measurement diversity noise
    (default 0.02, the IRP-tuned value — see QINSGA3/chromosome.py). That
    noise exists to stop nearby theta values collapsing to identical
    integer routes after the IRP decoder rounds to integers; DTLZ/MaF have
    no such rounding step, so it can be set to 0 here without affecting the
    IRP path (whose QuantumPopulation calls never pass this argument).
    """
    rng = np.random.default_rng(seed)

    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)
    n_genes = problem.n_var

    qpop = QuantumPopulation(
        pop_size, n_genes, xl, xu, rng=rng,
        rotation_type=rotation_type, noise_scale=noise_scale,
    )
    sorter = NonDominatedSorting()

    arch_X: list[np.ndarray] = []
    arch_F: list[np.ndarray] = []
    arch_theta: list[np.ndarray] = []
    _MAX_ARCHIVE = 500

    def _eval_batch(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        F, G = problem.evaluate(X, return_values_of=["F", "G"])
        if G is None or (hasattr(G, "size") and G.size == 0):
            G = np.zeros((X.shape[0], 0))
        return np.asarray(F), np.asarray(G)

    for gen in range(max_gen):
        X = qpop.measure()
        F, G = _eval_batch(X)

        F_pen = _penalised_F(F, G)
        fronts = sorter.do(F_pen)
        pareto_idx = fronts[0]

        _archive_update(
            X[pareto_idx], F[pareto_idx], G[pareto_idx], qpop.theta[pareto_idx],
            arch_X, arch_F, arch_theta, _MAX_ARCHIVE,
        )

        F_norm = _normalise_F(F_pen)
        assoc = _assign_ref_dirs(F_norm, ref_dirs)

        guides_theta = _select_guides(assoc, pareto_idx, F_norm, ref_dirs, qpop.theta)

        arch_theta_arr = None
        arch_F_norm = None
        if len(arch_X) >= 4:
            arch_theta_arr = np.array(arch_theta)
            arch_F_norm = _normalise_F(np.array(arch_F))
            pareto_assoc = assoc[pareto_idx]
            guides_theta = _supplement_from_archive(
                guides_theta, assoc, pareto_assoc,
                arch_theta_arr, arch_F_norm, ref_dirs,
            )

        alpha = alpha_min + (alpha_max - alpha_min) * (1.0 - gen / max_gen)
        qpop.rotate(guides_theta, alpha)

        if p_cross > 0.0:
            qpop.crossover(p_cross, eta_cross)

        qpop.mutate(p_mut, p_mut_strong, mut_sigma)

        if (arch_F_norm is not None
                and migration_period > 0
                and gen % migration_period == 0):
            _migrate(
                qpop, arch_theta_arr, arch_F_norm,
                assoc, ref_dirs, rng, n_migrate=n_migrate,
            )

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
