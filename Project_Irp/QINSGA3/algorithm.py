"""QINSGA-III generational loop for the many-objective IRP.

Algorithm per generation  [Li et al. ICNC 2008; Deb & Jain 2014]:
  1. Measure quantum population → classical X matrix (+ small diversity noise)
  2. Evaluate X via IRPProblem → F (4 objectives), G (constraints)
     (parallelised: one IRPProblem per worker process, created once via initializer)
  3. Penalise infeasible solutions (feasibility-first)
  4. NSGA-III non-dominated sort on penalised F
  5. Normalise F via ideal + nadir hyperplane  [Deb & Jain 2014 §IV-A]
  6. Assign each solution to nearest reference direction
  7. Guide selection: elitist per niche — best of (current Pareto front,
     archive), archive normalised in the same ideal/nadir frame as the
     population [Han & Kim 2002 elitism principle]
  8. Adaptive rotation: Δθ = α(g) × tanh((θ_guide − θ) / (π/8)),  α linear decay
  9. SBX crossover + quantum mutation

Performance:
  - Population evaluation is parallelised via ProcessPoolExecutor.  Each worker
    process holds one IRPProblem singleton (created once in _worker_init, not
    recreated per evaluation call) — eliminates repeated construction overhead.
  - _crowding_distance, _select_guides, and _archive_update are fully
    vectorised with NumPy broadcasting — no Python inner loops over
    population or archive members.
  - arch_F_norm is computed once per generation (projected into the
    population's ideal/nadir frame via _normalise_stats) and shared by both
    _select_guides and _migrate.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .chromosome import QuantumPopulation


# ---------------------------------------------------------------------------
# Multiprocessing workers
# Must be at module level so they are picklable on Windows "spawn" processes.
# ---------------------------------------------------------------------------

_g_problem = None  # per-process singleton; populated by _worker_init


def _worker_init(sets_: dict, params_: dict) -> None:
    """Create a per-process IRPProblem (called once per worker at pool startup).

    Using ProcessPoolExecutor's initializer= parameter avoids re-constructing
    IRPProblem and re-importing NSGA3.problem on every evaluation call.
    """
    global _g_problem
    from NSGA3.problem import IRPProblem
    _g_problem = IRPProblem(sets_, params_)


def _worker_eval(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one solution in a worker process. Returns (F, G)."""
    out: dict = {}
    _g_problem._evaluate(x, out)
    return out["F"], out["G"]


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
        w    = np.full(M, eps)
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


def _normalise_stats(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ideal point + nadir-minus-ideal span for F (Deb & Jain 2014, §IV-A).

    Split out of _normalise_F so callers that need to project a SECOND array
    (e.g. an external archive) into the exact same normalised frame as F can
    reuse (ideal, denom) instead of recomputing their own — required for a
    valid distance comparison between the two arrays.
    """
    ideal = F.min(axis=0)
    nadir = _compute_nadir(F, ideal)
    denom = np.where(nadir - ideal > 1e-9, nadir - ideal, 1.0)
    return ideal, denom


def _normalise_F(F: np.ndarray) -> np.ndarray:
    """Normalise F: ideal point + nadir from hyperplane (Deb & Jain 2014, §IV-A)."""
    ideal, denom = _normalise_stats(F)
    return (F - ideal) / denom


def _assign_ref_dirs(F_norm: np.ndarray, ref_dirs: np.ndarray) -> np.ndarray:
    """Assign each solution to the nearest reference direction via perpendicular distance.

    d_perp²(f, r̂) = ||f||² − (f · r̂)²   [Deb & Jain 2014, §IV-B, eq. 4]
    """
    norms    = np.linalg.norm(ref_dirs, axis=1, keepdims=True)
    ref_unit = ref_dirs / np.where(norms > 1e-9, norms, 1.0)
    proj     = F_norm @ ref_unit.T
    F_sq     = (F_norm ** 2).sum(axis=1, keepdims=True)
    dist2    = np.maximum(F_sq - proj ** 2, 0.0)
    return dist2.argmin(axis=1)


# ---------------------------------------------------------------------------
# Guide selection (vectorised over niches, not over individuals)
# ---------------------------------------------------------------------------

def _select_guides(
    assoc:       np.ndarray,
    pareto_idx:  np.ndarray,
    F_norm:      np.ndarray,
    ref_dirs:    np.ndarray,
    qpop_theta:  np.ndarray,
    arch_theta:  np.ndarray | None = None,
    arch_F_norm: np.ndarray | None = None,
) -> np.ndarray:
    """Return elitist guide theta angles for each individual, one per niche.

    For every niche present in `assoc`, compares the current Pareto front's
    best representative (smallest perpendicular distance to the reference
    ray) against the archive's best representative in the same niche, and
    keeps whichever is closer. A niche with a representative on only one
    side uses that side; a niche with neither uses the global fallback
    (closest-to-origin Pareto member) — same fallback as before.

    `arch_theta`/`arch_F_norm` are optional (default None, meaning "no
    archive yet" — same output as before this function had archive support).
    When provided, `arch_F_norm` MUST already be normalised in the SAME
    ideal/nadir frame as `F_norm` — the caller's responsibility — otherwise
    the two perpendicular distances are not on a comparable scale.

    This generalises the old split between front-only `_select_guides` and
    archive-only-for-empty-niches `_supplement_from_archive`: the archive is
    elitist by construction (`_archive_update` never lets a dominated
    solution survive), so consulting it for every niche — not just
    uncovered ones — stops a niche's guide from regressing across
    generations [Han & Kim 2002 elitism principle; Zhang 2011 survey,
    "attractor replaced only if better"].
    """
    N         = len(assoc)
    F_par_n   = F_norm[pareto_idx]
    global_fb = qpop_theta[pareto_idx[np.linalg.norm(F_par_n, axis=1).argmin()]]

    ref_norms = np.linalg.norm(ref_dirs, axis=1, keepdims=True)
    ref_unit  = ref_dirs / np.where(ref_norms > 1e-9, ref_norms, 1.0)

    have_archive = arch_theta is not None and arch_F_norm is not None and len(arch_theta) > 0
    arch_assoc   = _assign_ref_dirs(arch_F_norm, ref_dirs) if have_archive else None

    def _best_in_niche(cand_theta: np.ndarray, cand_F: np.ndarray, rd: int):
        proj    = cand_F @ ref_unit[rd]
        d_perp2 = np.maximum((cand_F ** 2).sum(axis=1) - proj ** 2, 0.0)
        best    = d_perp2.argmin()
        return cand_theta[best], d_perp2[best]

    guides_theta = np.tile(global_fb, (N, 1))   # default: global fallback

    for rd in np.unique(assoc):
        front_idx = pareto_idx[assoc[pareto_idx] == rd]
        front_theta, front_dist = (
            _best_in_niche(qpop_theta[front_idx], F_norm[front_idx], rd)
            if len(front_idx) > 0 else (None, None)
        )

        arch_theta_best, arch_dist = (None, None)
        if have_archive:
            in_niche = np.where(arch_assoc == rd)[0]
            if len(in_niche) > 0:
                arch_theta_best, arch_dist = _best_in_niche(
                    arch_theta[in_niche], arch_F_norm[in_niche], rd
                )

        if front_theta is None and arch_theta_best is None:
            continue   # keep the global fallback already in guides_theta
        if front_theta is None:
            best_theta = arch_theta_best
        elif arch_theta_best is None:
            best_theta = front_theta
        else:
            best_theta = front_theta if front_dist <= arch_dist else arch_theta_best

        guides_theta[assoc == rd] = best_theta   # broadcast to whole niche at once

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
    """Inject best archive θ per niche into n_migrate individuals [Han & Kim 2002].

    arch_F_norm is pre-normalised by the caller (same array as passed to
    _select_guides) — no additional _normalise_F call needed here.
    """
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
# Crowding distance and archive trimming (vectorised)
# ---------------------------------------------------------------------------

def _crowding_distance(F: np.ndarray) -> np.ndarray:
    """NSGA-II crowding distance, fully vectorised [Deb et al. 2002, §III-B].

    Inner per-individual accumulation replaced by a single NumPy slice per
    objective:
        cd[order[1:-1]] += (F[order[2:], m] − F[order[:-2], m]) / span
    """
    N, M = F.shape
    cd   = np.zeros(N)
    for m in range(M):
        order        = np.argsort(F[:, m])
        f_min, f_max = F[order[0], m], F[order[-1], m]
        cd[order[0]]  = np.inf
        cd[order[-1]] = np.inf
        span = f_max - f_min if f_max - f_min > 1e-9 else 1.0
        cd[order[1:-1]] += (F[order[2:], m] - F[order[:-2], m]) / span
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
# Archive update (vectorised dominance and duplicate checks)
# ---------------------------------------------------------------------------

def _archive_update(
    new_X:     np.ndarray,
    new_F:     np.ndarray,
    new_G:     np.ndarray,
    new_theta: np.ndarray,
    arch_X:    list,
    arch_F:    list,
    arch_theta: list,
    max_size:  int = 500,
) -> None:
    """Update the external archive with non-dominated feasible solutions.

    Dominance check vectorised [Zitzler 1999, Def. 2]:
        arr dominates f  ⟺  (arr ≤ f).all(axis=1) & (arr < f).any(axis=1)

    Performance fix: arr = np.array(arch_F) is built ONCE per call (not once per
    candidate). Deletions are tracked via a boolean mask and applied in a single
    O(n) list rebuild instead of repeated O(n) list.pop calls.
    """
    feasible = np.maximum(new_G, 0.0).sum(axis=1) == 0
    feas_idx = np.where(feasible)[0]
    if len(feas_idx) == 0:
        return

    cand_X     = new_X[feas_idx]
    cand_F     = new_F[feas_idx]
    cand_theta = new_theta[feas_idx]

    n_arch = len(arch_F)
    arr    = np.array(arch_F) if n_arch > 0 else None
    alive  = np.ones(n_arch, dtype=bool)   # tracks which archive entries survive

    add_X, add_F, add_theta = [], [], []

    for c in range(len(cand_F)):
        f = cand_F[c]

        if arr is not None:
            arr_live = arr[alive]
            if len(arr_live):
                # Skip if dominated by any surviving archive entry
                if ((arr_live <= f).all(axis=1) & (arr_live < f).any(axis=1)).any():
                    continue
                # Skip duplicate F vectors
                if (np.abs(arr_live - f).max(axis=1) < 1e-6).any():
                    continue
                # Mark archive entries dominated by f (boolean mask, no pop yet)
                live_idx = np.where(alive)[0]
                dom = (f <= arr_live).all(axis=1) & (f < arr_live).any(axis=1)
                alive[live_idx[dom]] = False

        # Avoid within-batch duplicates (candidates from same Pareto front)
        if add_F:
            add_arr = np.array(add_F)
            if (np.abs(add_arr - f).max(axis=1) < 1e-6).any():
                continue

        add_X.append(cand_X[c].copy())
        add_F.append(f.copy())
        add_theta.append(cand_theta[c].copy())

    # Apply all deletions in one pass — O(n) list rebuild vs O(n²) repeated pops
    if n_arch > 0 and not alive.all():
        keep = np.where(alive)[0].tolist()
        arch_X[:]     = [arch_X[i]     for i in keep]
        arch_F[:]     = [arch_F[i]     for i in keep]
        arch_theta[:] = [arch_theta[i] for i in keep]

    arch_X.extend(add_X)
    arch_F.extend(add_F)
    arch_theta.extend(add_theta)

    if len(arch_F) > max_size:
        arr_X, arr_F, arr_theta = _crowding_trim(
            np.array(arch_X), np.array(arch_F), np.array(arch_theta), max_size,
        )
        arch_X[:]     = list(arr_X)
        arch_F[:]     = list(arr_F)
        arch_theta[:] = list(arr_theta)


# ---------------------------------------------------------------------------
# Main QINSGA-III loop
# ---------------------------------------------------------------------------

def run_qinsga3(
    sets_,
    params_,
    ref_dirs:         np.ndarray,
    pop_size:         int   = 200,
    max_gen:          int   = 300,
    alpha_max:        float = 0.10  * np.pi,
    alpha_min:        float = 0.001 * np.pi,
    p_mut:            float | None = None,
    p_mut_strong:     float = 0.15,
    mut_sigma:        float = 0.05 * np.pi,
    p_cross:          float = 0.9,
    eta_cross:        float = 5.0,
    migration_period: int   = 10,
    n_migrate:        int   = 10,
    seed:             int   = 42,
    rotation_type:    str   = "tanh",
    callback          = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one QINSGA-III instance. Returns (pareto_X, pareto_F, pareto_G).

    Population evaluation is parallelised via ProcessPoolExecutor: the pool is
    created once per call; each worker process is initialised with a persistent
    IRPProblem singleton (_worker_init) so construction cost is paid once, not
    once per evaluation.  Assumes IRPProblem._evaluate is deterministic — results
    are identical to the sequential version for the same seed.
    """
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    from NSGA3.problem import IRPProblem

    rng     = np.random.default_rng(seed)
    problem = IRPProblem(sets_, params_)   # main-process copy — used only for metadata

    xl       = np.asarray(problem.xl, dtype=float)
    xu       = np.asarray(problem.xu, dtype=float)
    n_genes  = problem.n_var
    n_constr = problem.n_ieq_constr

    if p_mut is None:
        p_mut = 2.0 / n_genes

    qpop   = QuantumPopulation(pop_size, n_genes, xl, xu, rng=rng, rotation_type=rotation_type)
    sorter = NonDominatedSorting()

    arch_X:    list[np.ndarray] = []
    arch_F:    list[np.ndarray] = []
    arch_theta: list[np.ndarray] = []
    _MAX_ARCHIVE = 500

    n_workers = min(os.cpu_count() or 1, pop_size)
    chunksize = max(1, pop_size // (2 * n_workers))

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(sets_, params_),
    ) as pool:

        def _eval_batch(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Evaluate all individuals in X in parallel across worker processes."""
            results = list(pool.map(_worker_eval, list(X), chunksize=chunksize))
            return (
                np.array([r[0] for r in results]),
                np.array([r[1] for r in results]),
            )

        for gen in range(max_gen):
            X    = qpop.measure()
            F, G = _eval_batch(X)

            F_pen      = _penalised_F(F, G)
            fronts     = sorter.do(F_pen)
            pareto_idx = fronts[0]

            _archive_update(
                X[pareto_idx], F[pareto_idx], G[pareto_idx], qpop.theta[pareto_idx],
                arch_X, arch_F, arch_theta, _MAX_ARCHIVE,
            )

            ideal, denom = _normalise_stats(F_pen)
            F_norm = (F_pen - ideal) / denom
            assoc  = _assign_ref_dirs(F_norm, ref_dirs)

            # arch_F_norm projected into the SAME ideal/denom as F_norm (not its
            # own) so _select_guides can validly compare front vs archive
            # distances; also reused by _migrate below.
            arch_theta_arr = None
            arch_F_norm    = None
            if len(arch_X) >= 4:
                arch_theta_arr = np.array(arch_theta)
                arch_F_norm    = (np.array(arch_F) - ideal) / denom

            guides_theta = _select_guides(
                assoc, pareto_idx, F_norm, ref_dirs, qpop.theta,
                arch_theta=arch_theta_arr, arch_F_norm=arch_F_norm,
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

            if callback is not None and (gen % 10 == 0 or gen == max_gen - 1):
                callback(gen, F, G, pareto_idx)

        # Final measurement after all generations. Archive update must only see
        # this generation's own Pareto front (rank 0), not the whole population —
        # otherwise mutually-dominated individuals from the same batch can both
        # enter the archive, since _archive_update only screens new candidates
        # against each other for exact duplicates, never for dominance.
        X_final      = qpop.measure()
        F_final, G_final = _eval_batch(X_final)
        F_pen_final      = _penalised_F(F_final, G_final)
        final_pareto_idx = sorter.do(F_pen_final)[0]
        _archive_update(
            X_final[final_pareto_idx], F_final[final_pareto_idx],
            G_final[final_pareto_idx], qpop.theta[final_pareto_idx],
            arch_X, arch_F, arch_theta, _MAX_ARCHIVE,
        )

    if arch_X:
        arch_X_arr, arch_F_arr, _ = _crowding_trim(
            np.array(arch_X), np.array(arch_F), np.array(arch_theta), pop_size
        )
        return arch_X_arr, arch_F_arr, np.zeros((len(arch_X_arr), n_constr))

    return X_final[final_pareto_idx], F_final[final_pareto_idx], G_final[final_pareto_idx]
