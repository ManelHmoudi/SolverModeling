"""QINSGA-III generational loop for the many-objective IRP.

Algorithm per generation:
  1. Measure quantum population → classical X matrix
  2. Evaluate X via IRPProblem → F (4 objectives), G (constraints)
  3. Penalise infeasible solutions (feasibility-first)
  4. NSGA-III non-dominated sort on penalised F
  5. Assign each solution to nearest reference direction (Deb & Jain 2014)
  6. Guide selection: Pareto-front member in same niche (Li & Wang 2007)
  7. Adaptive rotation: Δθ = α(g) × sign(θ_guide − θ) × |cos(2θ)|, α linear decay
  8. Quantum mutation: gene reset to π/4 with prob p_mut
"""

from __future__ import annotations

import numpy as np

from .chromosome import QuantumPopulation


def _penalised_F(F: np.ndarray, G: np.ndarray) -> np.ndarray:
    cv         = np.maximum(G, 0.0)
    n_violated = (cv > 0).sum(axis=1, keepdims=True)
    total_cv   = cv.sum(axis=1, keepdims=True)
    penalty    = 1e9 * n_violated + 1e6 * total_cv
    return F + penalty


def _normalise_F(F: np.ndarray) -> np.ndarray:
    ideal = F.min(axis=0)
    nadir = F.max(axis=0)
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


def _select_guides(
    assoc:      np.ndarray,
    pareto_idx: np.ndarray,
    F_pen:      np.ndarray,
    ref_dirs:   np.ndarray,
) -> np.ndarray:
    """Return guide index for each chromosome from the Pareto front."""
    N = len(assoc)

    F_par   = F_pen[pareto_idx]
    ideal_p = F_par.min(axis=0)
    nadir_p = F_par.max(axis=0)
    denom_p = np.where(nadir_p - ideal_p > 1e-9, nadir_p - ideal_p, 1.0)
    F_par_n = (F_par - ideal_p) / denom_p

    global_fb    = pareto_idx[np.linalg.norm(F_par_n, axis=1).argmin()]
    pareto_assoc = assoc[pareto_idx]

    guides = np.empty(N, dtype=int)
    for i in range(N):
        rd        = assoc[i]
        same_mask = (pareto_assoc == rd)
        same_idx  = pareto_idx[same_mask]

        if len(same_idx) == 0:
            guides[i] = global_fb
        elif len(same_idx) == 1:
            guides[i] = same_idx[0]
        else:
            r_norm   = ref_dirs[rd] / max(np.linalg.norm(ref_dirs[rd]), 1e-9)
            F_same   = F_pen[same_idx]
            ideal_s  = F_same.min(axis=0)
            denom_s  = np.where(F_same.max(axis=0) - ideal_s > 1e-9,
                                F_same.max(axis=0) - ideal_s, 1.0)
            F_same_n = (F_same - ideal_s) / denom_s
            proj     = F_same_n @ r_norm
            F_sq_s   = (F_same_n ** 2).sum(axis=1)
            d_perp2  = np.maximum(F_sq_s - proj ** 2, 0.0)
            guides[i] = same_idx[d_perp2.argmin()]

    return guides


def run_qinsga3(
    sets_,
    params_,
    ref_dirs:   np.ndarray,
    pop_size:   int   = 200,
    max_gen:    int   = 200,
    alpha_max:  float = 0.05  * np.pi,
    alpha_min:  float = 0.001 * np.pi,
    p_mut:      float | None = None,
    seed:       int   = 42,
    callback    = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One QINSGA-III run. Returns (pareto_X, pareto_F, pareto_G)."""
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    from NSGA3.problem import IRPProblem

    rng     = np.random.default_rng(seed)
    problem = IRPProblem(sets_, params_)

    xl      = np.asarray(problem.xl, dtype=float)
    xu      = np.asarray(problem.xu, dtype=float)
    n_genes  = problem.n_var
    n_obj    = problem.n_obj
    n_constr = problem.n_ieq_constr

    if p_mut is None:
        p_mut = 1.0 / n_genes

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

    arch_X: list[np.ndarray] = []
    arch_F: list[np.ndarray] = []

    def _archive_update(new_X, new_F, new_G):
        for x, f, g in zip(new_X, new_F, new_G):
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
            if not dominated:
                for k in sorted(to_remove, reverse=True):
                    arch_X.pop(k)
                    arch_F.pop(k)
                arch_X.append(x.copy())
                arch_F.append(f.copy())

    for gen in range(max_gen):
        X    = qpop.measure()
        F, G = _evaluate_batch(X)

        F_pen      = _penalised_F(F, G)
        fronts     = sorter.do(F_pen)
        pareto_idx = fronts[0]
        _archive_update(X[pareto_idx], F[pareto_idx], G[pareto_idx])

        F_norm     = _normalise_F(F_pen)
        assoc      = _assign_ref_dirs(F_norm, ref_dirs)
        guides_idx = _select_guides(assoc, pareto_idx, F_pen, ref_dirs)
        guides_X   = X[guides_idx]

        alpha = alpha_min + (alpha_max - alpha_min) * (1.0 - gen / max_gen)
        qpop.rotate(X, guides_X, alpha)
        qpop.mutate(p_mut)

        if callback is not None and (gen % 10 == 0 or gen == max_gen - 1):
            callback(gen, F, G, pareto_idx)

    X_final        = qpop.measure()
    F_final, G_final = _evaluate_batch(X_final)
    _archive_update(X_final, F_final, G_final)

    if arch_X:
        return np.array(arch_X), np.array(arch_F), np.zeros((len(arch_X), n_constr))

    F_pen_final = _penalised_F(F_final, G_final)
    idx_final   = sorter.do(F_pen_final)[0]
    return X_final[idx_final], F_final[idx_final], G_final[idx_final]
