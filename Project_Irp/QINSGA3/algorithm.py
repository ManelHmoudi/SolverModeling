"""QINSGA-III generational loop for the many-objective IRP.

Algorithm per generation  [Li & Wang 2007; Deb & Jain 2014]:
  1. Measure quantum population → classical X matrix (+ small diversity noise)
  2. Evaluate X via IRPProblem → F (4 objectives), G (constraints)
  3. Penalise infeasible solutions (feasibility-first)
  4. NSGA-III non-dominated sort on penalised F
  5. Normalise F via ideal + nadir hyperplane  [Deb & Jain 2014 §IV-A]
  6. Assign each solution to nearest reference direction
  7. Guide selection: best Pareto member in same niche; archive fills empty niches
  8. Adaptive rotation: Δθ = α(g) × tanh((θ_guide − θ) / (π/8)),  α linear decay
  9. SBX crossover + quantum mutation
"""

from __future__ import annotations

import numpy as np

from .chromosome import QuantumPopulation


# ---------------------------------------------------------------------------
# Objective-space helpers
# ---------------------------------------------------------------------------

def _penalised_F(F: np.ndarray, G: np.ndarray) -> np.ndarray:
    cv         = np.maximum(G, 0.0)
    n_violated = (cv > 0).sum(axis=1, keepdims=True)
    total_cv   = cv.sum(axis=1, keepdims=True)
    penalty    = 1e9 * n_violated + 1e6 * total_cv
    return F + penalty


def _compute_nadir(F: np.ndarray, ideal: np.ndarray) -> np.ndarray:
    """Nadir via extreme points + hyperplane intercepts (Deb & Jain 2014, §IV-A).

    For each objective i find the solution minimising ASF(x, e_i) = max_j(x'_j / w_j)
    where x' = F − ideal, w = ε·1 + e_i.  Solve A·a = 1 for intercepts 1/a.
    Falls back to F.max() if the hyperplane is degenerate.
    """
    M            = F.shape[1]
    F_translated = F - ideal
    eps          = 1e-6

    extreme_idx = []
    for i in range(M):
        w = np.full(M, eps)
        w[i] = 1.0
        extreme_idx.append(int((F_translated / w).max(axis=1).argmin()))

    A = F_translated[extreme_idx]
    try:
        a          = np.linalg.solve(A, np.ones(M))
        intercepts = 1.0 / np.where(np.abs(a) > 1e-9, a, 1e-9)
        if np.all(intercepts > 0):
            return ideal + intercepts
    except np.linalg.LinAlgError:
        pass

    return F.max(axis=0)


def _normalise_F(F: np.ndarray) -> np.ndarray:
    """Normalise F: ideal point + nadir from hyperplane (Deb & Jain 2014)."""
    ideal = F.min(axis=0)
    nadir = _compute_nadir(F, ideal)
    denom = np.where(nadir - ideal > 1e-9, nadir - ideal, 1.0)
    return (F - ideal) / denom


def _assign_ref_dirs(F_norm: np.ndarray, ref_dirs: np.ndarray) -> np.ndarray:
    """Assign each solution to the nearest reference direction (perpendicular distance)."""
    norms    = np.linalg.norm(ref_dirs, axis=1, keepdims=True)
    ref_unit = ref_dirs / np.where(norms > 1e-9, norms, 1.0)
    proj     = F_norm @ ref_unit.T
    F_sq     = (F_norm ** 2).sum(axis=1, keepdims=True)
    dist2    = np.maximum(F_sq - proj ** 2, 0.0)
    return dist2.argmin(axis=1)


# ---------------------------------------------------------------------------
# Guide selection
# ---------------------------------------------------------------------------

def _select_guides(
    assoc:      np.ndarray,
    pareto_idx: np.ndarray,
    F_norm:     np.ndarray,
    ref_dirs:   np.ndarray,
    qpop_theta: np.ndarray,
) -> np.ndarray:
    """Return guide θ angles for each individual from the current Pareto front.

    For each individual i: find the Pareto member in the same reference-direction
    niche with smallest perpendicular distance, using the global F_norm so the
    scale is consistent with the niche assignment [Deb & Jain 2014, §IV-B].
    Falls back to the Pareto member closest to the origin when the niche is empty.
    """
    N       = len(assoc)
    n_genes = qpop_theta.shape[1]

    F_par_n         = F_norm[pareto_idx]
    global_fb_theta = qpop_theta[pareto_idx[np.linalg.norm(F_par_n, axis=1).argmin()]]
    pareto_assoc    = assoc[pareto_idx]

    guides_theta = np.empty((N, n_genes))
    for i in range(N):
        rd        = assoc[i]
        same_mask = (pareto_assoc == rd)
        same_idx  = pareto_idx[same_mask]

        if len(same_idx) == 0:
            guides_theta[i] = global_fb_theta
        elif len(same_idx) == 1:
            guides_theta[i] = qpop_theta[same_idx[0]]
        else:
            r_norm   = ref_dirs[rd] / max(np.linalg.norm(ref_dirs[rd]), 1e-9)
            F_same_n = F_norm[same_idx]
            proj     = F_same_n @ r_norm
            d_perp2  = np.maximum((F_same_n ** 2).sum(axis=1) - proj ** 2, 0.0)
            guides_theta[i] = qpop_theta[same_idx[d_perp2.argmin()]]

    return guides_theta


def _supplement_from_archive(
    guides_theta: np.ndarray,
    assoc:        np.ndarray,
    pareto_assoc: np.ndarray,
    arch_theta:   np.ndarray,
    arch_F:       np.ndarray,
    ref_dirs:     np.ndarray,
) -> np.ndarray:
    """Fill empty-niche guides from archive (Li & Wang 2007, §III-C)."""
    arch_F_norm = _normalise_F(arch_F)
    arch_assoc  = _assign_ref_dirs(arch_F_norm, ref_dirs)
    covered     = set(pareto_assoc.tolist())

    for i in range(len(assoc)):
        rd = assoc[i]
        if rd in covered:
            continue
        in_niche = np.where(arch_assoc == rd)[0]
        if len(in_niche) == 0:
            continue
        if len(in_niche) == 1:
            guides_theta[i] = arch_theta[in_niche[0]]
        else:
            r_norm   = ref_dirs[rd] / max(np.linalg.norm(ref_dirs[rd]), 1e-9)
            F_cand_n = arch_F_norm[in_niche]
            proj     = F_cand_n @ r_norm
            d_perp2  = np.maximum((F_cand_n ** 2).sum(axis=1) - proj ** 2, 0.0)
            guides_theta[i] = arch_theta[in_niche[d_perp2.argmin()]]

    return guides_theta


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def _migrate(
    qpop:        QuantumPopulation,
    arch_theta:  np.ndarray,
    arch_F_norm: np.ndarray,
    assoc:       np.ndarray,
    ref_dirs:    np.ndarray,
    rng:         np.random.Generator,
    n_migrate:   int = 10,
) -> None:
    """Local migration: inject best archive θ per niche into n_migrate individuals (Han & Kim 2002)."""
    arch_assoc = _assign_ref_dirs(arch_F_norm, ref_dirs)

    targets = rng.choice(qpop.pop_size, size=min(n_migrate, qpop.pop_size), replace=False)
    for idx in targets:
        rd       = assoc[idx]
        in_niche = np.where(arch_assoc == rd)[0]
        if len(in_niche) > 0:
            r_norm  = ref_dirs[rd] / max(np.linalg.norm(ref_dirs[rd]), 1e-9)
            F_cand  = arch_F_norm[in_niche]
            proj    = F_cand @ r_norm
            d_perp2 = np.maximum((F_cand ** 2).sum(axis=1) - proj ** 2, 0.0)
            qpop.theta[idx] = arch_theta[in_niche[d_perp2.argmin()]]
        else:
            qpop.theta[idx] = arch_theta[np.linalg.norm(arch_F_norm, axis=1).argmin()]

    qpop.theta = np.clip(qpop.theta, 0.0, np.pi / 2.0)


# ---------------------------------------------------------------------------
# Crowding distance and archive trimming
# ---------------------------------------------------------------------------

def _crowding_distance(F: np.ndarray) -> np.ndarray:
    """NSGA-II crowding distance (Deb et al. 2002, §III-B)."""
    N, M = F.shape
    cd   = np.zeros(N)
    for m in range(M):
        order        = np.argsort(F[:, m])
        f_min, f_max = F[order[0], m], F[order[-1], m]
        cd[order[0]]  = np.inf
        cd[order[-1]] = np.inf
        span = f_max - f_min if f_max - f_min > 1e-9 else 1.0
        for k in range(1, N - 1):
            cd[order[k]] += (F[order[k + 1], m] - F[order[k - 1], m]) / span
    return cd


def _crowding_trim(
    X:        np.ndarray,
    F:        np.ndarray,
    theta:    np.ndarray,
    max_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep the max_size most spread solutions by NSGA-II crowding distance."""
    if len(X) <= max_size:
        return X, F, theta
    keep = np.argsort(_crowding_distance(_normalise_F(F)))[-max_size:]
    return X[keep], F[keep], theta[keep]


# ---------------------------------------------------------------------------
# Main QINSGA-III loop
# ---------------------------------------------------------------------------

def run_qinsga3(
    sets_,
    params_,
    ref_dirs:         np.ndarray,
    pop_size:         int   = 200,
    max_gen:          int   = 300,
    alpha_max:        float = 0.05  * np.pi,
    alpha_min:        float = 0.001 * np.pi,
    p_mut:            float | None = None,
    p_cross:          float = 0.9,
    eta_cross:        float = 5.0,
    migration_period: int   = 10,
    n_migrate:        int   = 10,
    seed:             int   = 42,
    callback          = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One QINSGA-III run. Returns (pareto_X, pareto_F, pareto_G)."""
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    from NSGA3.problem import IRPProblem

    rng     = np.random.default_rng(seed)
    problem = IRPProblem(sets_, params_)

    xl       = np.asarray(problem.xl, dtype=float)
    xu       = np.asarray(problem.xu, dtype=float)
    n_genes  = problem.n_var
    n_obj    = problem.n_obj
    n_constr = problem.n_ieq_constr

    if p_mut is None:
        p_mut = 2.0 / n_genes

    qpop   = QuantumPopulation(pop_size, n_genes, xl, xu, rng=rng)
    sorter = NonDominatedSorting()

    def _evaluate_batch(X: np.ndarray):
        F = np.empty((len(X), n_obj))
        G = np.empty((len(X), n_constr))
        for i, x in enumerate(X):
            out = {}
            problem._evaluate(x, out)
            F[i] = out["F"]
            G[i] = out["G"]
        return F, G

    arch_X:     list[np.ndarray] = []
    arch_F:     list[np.ndarray] = []
    arch_theta: list[np.ndarray] = []
    _MAX_ARCHIVE = 500

    def _archive_update(new_X, new_F, new_G, new_theta):
        for x, f, g, th in zip(new_X, new_F, new_G, new_theta):
            if np.maximum(g, 0.0).sum() > 0:
                continue
            dominated = False
            to_remove = []
            for k, af in enumerate(arch_F):
                if np.all(af <= f) and np.any(af < f):
                    dominated = True
                    break
                if np.all(f <= af) and np.any(f < af):
                    to_remove.append(k)
            if dominated:
                continue
            # Skip F vectors already in archive (different X → same decoded routes → same F)
            if any(np.max(np.abs(af - f)) < 1e-6 for af in arch_F):
                continue
            for k in sorted(to_remove, reverse=True):
                arch_X.pop(k)
                arch_F.pop(k)
                arch_theta.pop(k)
            arch_X.append(x.copy())
            arch_F.append(f.copy())
            arch_theta.append(th.copy())

        if len(arch_F) > _MAX_ARCHIVE:
            arr_X, arr_F, arr_theta = _crowding_trim(
                np.array(arch_X), np.array(arch_F), np.array(arch_theta), _MAX_ARCHIVE
            )
            arch_X[:]     = list(arr_X)
            arch_F[:]     = list(arr_F)
            arch_theta[:] = list(arr_theta)

    for gen in range(max_gen):
        X    = qpop.measure()
        F, G = _evaluate_batch(X)

        F_pen      = _penalised_F(F, G)
        fronts     = sorter.do(F_pen)
        pareto_idx = fronts[0]

        _archive_update(X[pareto_idx], F[pareto_idx], G[pareto_idx], qpop.theta[pareto_idx])

        F_norm = _normalise_F(F_pen)
        assoc  = _assign_ref_dirs(F_norm, ref_dirs)

        guides_theta = _select_guides(assoc, pareto_idx, F_norm, ref_dirs, qpop.theta)

        arch_theta_arr = None
        arch_F_arr     = None
        if len(arch_X) >= 4:
            arch_theta_arr = np.array(arch_theta)
            arch_F_arr     = np.array(arch_F)
            pareto_assoc   = assoc[pareto_idx]
            guides_theta   = _supplement_from_archive(
                guides_theta, assoc, pareto_assoc,
                arch_theta_arr, arch_F_arr, ref_dirs,
            )

        alpha = alpha_min + (alpha_max - alpha_min) * (1.0 - gen / max_gen)

        qpop.rotate(guides_theta, alpha)

        if p_cross > 0.0:
            qpop.crossover(p_cross, eta_cross)

        qpop.mutate(p_mut)

        if (arch_theta_arr is not None
                and migration_period > 0
                and gen % migration_period == 0):
            _migrate(qpop, arch_theta_arr, _normalise_F(arch_F_arr),
                     assoc, ref_dirs, rng, n_migrate=n_migrate)

        if callback is not None and (gen % 10 == 0 or gen == max_gen - 1):
            callback(gen, F, G, pareto_idx)

    X_final          = qpop.measure()
    F_final, G_final = _evaluate_batch(X_final)
    _archive_update(X_final, F_final, G_final, qpop.theta)

    if arch_X:
        arch_X_arr, arch_F_arr, _ = _crowding_trim(
            np.array(arch_X), np.array(arch_F), np.array(arch_theta), pop_size
        )
        return arch_X_arr, arch_F_arr, np.zeros((len(arch_X_arr), n_constr))

    F_pen_final = _penalised_F(F_final, G_final)
    idx_final   = sorter.do(F_pen_final)[0]
    return X_final[idx_final], F_final[idx_final], G_final[idx_final]
