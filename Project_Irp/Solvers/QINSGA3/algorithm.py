"""QINSGA-III generational loop for the many-objective IRP.

Algorithm per generation  [Li et al. ICNC 2008; Deb & Jain 2014]:
  1. Measure quantum population (parent) → classical X matrix
  2. Evaluate parent X via IRPProblem → F (4 objectives), G (constraints)
     (parallelised: one IRPProblem per worker process, created once via initializer)
  3. Penalise infeasible solutions (feasibility-first); archive-update from the
     parent's own Pareto front
  4. Normalise F via ideal + nadir hyperplane  [Deb & Jain 2014 §IV-A]; assign
     each solution to its nearest reference direction; select guides (best
     Pareto member per niche, archive fills empty niches)
  5. Quantum rotation (theta-space only): Δθ = α(g) × tanh((θ_guide − θ) / (π/8)),
     α linear decay — this is the one step that stays in the angle
     representation, since it is the genuinely "quantum" part of the algorithm
  6. Measure the rotated population, then vary it in X-SPACE with pymoo's own
     SBX(eta) + PM(eta) — identical operators to NSGA-III (pymoo) — instead of
     doing crossover/mutation in theta-space. Re-encode the result back to
     theta (θ = arccos(sqrt(p)), p = (x−xl)/(xu−xl)) for next generation's
     rotation step. Evaluate the offspring; archive-update from its own front.
  7. Elitist survivor selection: merge parent + offspring populations and keep
     only the best pop_size via pymoo's actual ReferenceDirectionSurvival
     (rank + niching) — the same elitist replacement NSGA-III itself uses.

Rationale for steps 6-7 (A/B-validated on the IRP's 100-client instance,
NSGA-III reference, shared ideal/nadir; ~2x then ~3x HV improvement
respectively — see git history for the full evidence):
  - Step 7 (elitist survival): previously nothing compared "population
    before variation" vs "after" to discard individuals that got worse —
    the whole population was unconditionally rotated + crossed + mutated
    every generation, with only the external archive providing elitism.
  - Step 6 (X-space variation): the decode x = xl + cos²(θ)(xu−xl) has
    dx/dθ = −(xu−xl)sin(2θ), vanishing at θ=0/π/2 and peaking at θ=π/4 — a
    highly non-uniform mapping. SBX/PM applied directly in θ-space meant a
    fixed-width move in θ produced wildly different moves in the real
    decision variables depending on where θ currently sat.
  - A real (smaller) quality gap vs NSGA-III remains on the 100-client
    instance. The external archive was confirmed to help, not hurt, and is
    kept unconditionally.

Performance:
  - Population evaluation is parallelised via ProcessPoolExecutor.  Each worker
    process holds one IRPProblem singleton (created once in _worker_init, not
    recreated per evaluation call) — eliminates repeated construction overhead.
  - _crowding_distance, _select_guides, _supplement_from_archive, and
    _archive_update are fully vectorised with NumPy broadcasting — no Python
    inner loops over population or archive members.
  - arch_F_norm is computed once per generation and shared by both
    _supplement_from_archive and _migrate, removing a redundant _normalise_F call.
  - The elitist survival step doubles per-generation evaluation cost (parent
    + offspring), so a run now takes roughly 2x as long as before.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.core.population import Population
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

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
    from Solvers.NSGA3.problem import IRPProblem
    _g_problem = IRPProblem(sets_, params_)


def _worker_eval(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one solution in a worker process. Returns (F, G)."""
    out: dict = {}
    _g_problem._evaluate(x, out)
    return out["F"], out["G"]


def _build_g_constraints(route_result: dict, sets_: dict, params_: dict) -> list:
    """Mirrors IRPProblem._evaluate's G-list construction exactly
    (Solvers/NSGA3/problem.py:67-85) -- duplicated here, not imported,
    since calling IRPProblem._evaluate directly would decode and build
    routes itself with no repair hook. Kept in sync manually; see
    docs/superpowers/specs/2026-08-04-qinsga3-route-repair-design.md.
    """
    clients  = sets_["clients"]
    T        = sets_["T"]
    q_lt     = params_["q_lt"]
    tau_min  = params_["tau_min"]
    tau_max  = params_["tau_max"]
    I_max_f  = params_["I_O_max_frigo"]
    I_max_nf = params_["I_O_max_nonfrigo"]
    actual      = route_result["actual_qty"]
    depot_stock = route_result["depot_stock"]

    G = []
    for t in T:
        ret = route_result["tau_return"].get(t, 0.0)
        G.append(ret - tau_max)
        G.append(tau_min - ret)

    for l in clients:
        cum_del = cum_dem = 0
        for t in T:
            cum_del += actual.get((l, t), 0)
            cum_dem += q_lt[l, t]
            G.append(cum_dem - cum_del)

    for t in T:
        G.append(depot_stock[t]["frigo"]    - I_max_f)
        G.append(depot_stock[t]["nonfrigo"] - I_max_nf)

    return G


def _evaluate_with_repair(x: np.ndarray, sets_: dict, params_: dict) -> tuple[np.ndarray, np.ndarray]:
    """Remedy G: decode + repair (2-opt, Baldwinian -- see repair.py) +
    evaluate, replacing IRPProblem._evaluate for QINSGA3 only, when
    use_route_repair=True. See
    docs/superpowers/specs/2026-08-04-qinsga3-route-repair-design.md.
    """
    from Solvers.NSGA3.decoder import decode_chromosome, build_routes
    from Solvers.NSGA3.evaluator import compute_f1, compute_f2, compute_f3, compute_f4
    from Solvers.QINSGA3.repair import _repair_route_result

    quantities, priorities = decode_chromosome(x, sets_)
    route_result = build_routes(quantities, sets_, params_, priorities)
    route_result = _repair_route_result(route_result, sets_, params_)

    F = np.array([
        compute_f1(route_result, sets_, params_),
        compute_f2(route_result, sets_, params_),
        compute_f3(route_result, sets_, params_),
        compute_f4(route_result, sets_, params_),
    ])
    G = np.array(_build_g_constraints(route_result, sets_, params_))
    return F, G


def _worker_eval_repaired(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same per-process pattern as _worker_eval, but routes through
    _evaluate_with_repair (remedy G) instead of _g_problem._evaluate."""
    return _evaluate_with_repair(x, _g_problem.sets_, _g_problem.params_)


def _repair_pareto_front(
    pareto_X: np.ndarray, sets_: dict, params_: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Post-processing variant of remedy G, for run_qinsga3's
    repair_final_front parameter: repairs each front chromosome's decoded
    route ONCE, sequentially, after the generational search loop has
    already finished (the loop's own process pool is closed by this
    point, and the front is small -- tens of individuals, not
    pop_size x max_gen -- so a sequential pass here is not the cost driver
    use_route_repair's every-generation repair was). Reuses
    _evaluate_with_repair unchanged (same decode -> repair -> evaluate ->
    G-list chain use_route_repair's worker path already uses) for each
    chromosome. Baldwinian: pareto_X itself is never modified, only the
    returned F/G arrays.
    """
    F_list = []
    G_list = []
    for x in pareto_X:
        F, G = _evaluate_with_repair(x, sets_, params_)
        F_list.append(F)
        G_list.append(G)
    return np.array(F_list), np.array(G_list)


# ---------------------------------------------------------------------------
# Objective-space helpers
# ---------------------------------------------------------------------------

def _penalised_F(F: np.ndarray, G: np.ndarray) -> np.ndarray:
    cv         = np.maximum(G, 0.0)
    n_violated = (cv > 0).sum(axis=1, keepdims=True)
    total_cv   = cv.sum(axis=1, keepdims=True)
    penalty    = 1e9 * n_violated + 1e6 * total_cv
    return F + penalty


def _tournament_select_parents(G: np.ndarray, pop_size: int, rng: np.random.Generator) -> np.ndarray:
    """CV-based binary tournament mating selection, matching pymoo's own
    NSGA-III default (TournamentSelection(func_comp=comp_by_cv_then_random),
    pymoo/algorithms/moo/nsga3.py) -- QI-NSGA-III previously had no mating
    selection at all (pure random.permutation pairing), a real asymmetry a
    fairness audit found: on a heavily constrained problem where a sizeable
    fraction of the population is infeasible each generation, NSGA-III's
    mating gets real per-generation pressure toward feasible parents that
    QI-NSGA-III's pairing lacked.

    Draws pop_size winners, one per parent slot (pop_size//2 pairs of 2),
    via binary tournaments: two candidates per slot, drawn from
    concatenated random permutations of the population (matches pymoo's own
    random_permutations use -- avoids the slight non-uniformity of plain
    sampling-with-replacement, e.g. an individual facing itself). Whichever
    candidate has strictly lower total constraint violation wins; if either
    is infeasible and both have equal CV (including the case where both are
    feasible, CV=0), the winner is random -- exactly comp_by_cv_then_random's
    own rule, just vectorised instead of pymoo's per-slot Python loop.
    """
    cv = np.maximum(G, 0.0).sum(axis=1)

    n_random = pop_size * 2
    n_perms  = -(-n_random // pop_size)  # ceil division
    perms    = np.concatenate([rng.permutation(pop_size) for _ in range(n_perms)])[:n_random]
    a, b     = perms[0::2], perms[1::2]

    cv_a, cv_b = cv[a], cv[b]
    prefer_a   = cv_a < cv_b
    prefer_b   = cv_b < cv_a
    coin       = rng.random(pop_size) < 0.5

    return np.where(prefer_a, a, np.where(prefer_b, b, np.where(coin, a, b)))


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


def _normalise_F(
    F:     np.ndarray,
    ideal: np.ndarray | None = None,
    nadir: np.ndarray | None = None,
) -> np.ndarray:
    """Normalise F: ideal point + nadir from hyperplane (Deb & Jain 2014, §IV-A).

    If ideal/nadir are not given, both are computed fresh from F alone (used
    by _crowding_trim's one-off archive trim, where there is no notion of
    "run so far" to track). If given, F is normalised directly against them
    instead -- run_qinsga3's main loop passes pymoo's own running
    ideal_point/nadir_point (ReferenceDirectionSurvival.norm, updated every
    generation from the merged parent+offspring pool, monotonic across the
    whole run) here, instead of recomputing an unstable from-scratch
    ideal/nadir from only the current generation's population every call --
    see "Design history" in Solvers/QINSGA3/README.md for why this matters:
    guide selection and rotation used to run on a per-generation-only
    estimate that could drift generation to generation even when nothing
    about the actual search changed, while the elitist survival step
    (pymoo's own ReferenceDirectionSurvival.do()) already used a stable,
    monotonic one for the same ideal/nadir concept.
    """
    if ideal is None:
        ideal = F.min(axis=0)
    if nadir is None:
        nadir = _compute_nadir(F, ideal)
    denom = np.where(nadir - ideal > 1e-9, nadir - ideal, 1.0)
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
    assoc:      np.ndarray,
    pareto_idx: np.ndarray,
    F_norm:     np.ndarray,
    ref_dirs:   np.ndarray,
    qpop_theta: np.ndarray,
) -> np.ndarray:
    """Return guide θ angles for each individual from the current Pareto front.

    Vectorised variant: iterates over unique occupied niches (≤ n_ref_dirs ≪ N)
    instead of all N individuals.  For each occupied niche the best Pareto member
    (minimum perpendicular distance to its reference ray) is broadcast to every
    population member in that niche [Deb & Jain 2014, §IV-B].
    Population members whose niche has no Pareto representative receive the
    Pareto member closest to the origin as a global fallback.
    """
    N            = len(assoc)
    F_par_n      = F_norm[pareto_idx]
    global_fb    = qpop_theta[pareto_idx[np.linalg.norm(F_par_n, axis=1).argmin()]]
    pareto_assoc = assoc[pareto_idx]

    # Pre-normalise all reference directions once — reused for every niche iteration
    ref_norms = np.linalg.norm(ref_dirs, axis=1, keepdims=True)
    ref_unit  = ref_dirs / np.where(ref_norms > 1e-9, ref_norms, 1.0)  # (n_dirs, M)

    guides_theta = np.tile(global_fb, (N, 1))   # default: global fallback

    for rd in np.unique(pareto_assoc):
        same_mask = pareto_assoc == rd
        same_idx  = pareto_idx[same_mask]

        if len(same_idx) == 1:
            best_theta = qpop_theta[same_idx[0]]
        else:
            F_same  = F_norm[same_idx]
            proj    = F_same @ ref_unit[rd]
            d_perp2 = np.maximum((F_same ** 2).sum(axis=1) - proj ** 2, 0.0)
            best_theta = qpop_theta[same_idx[d_perp2.argmin()]]

        guides_theta[assoc == rd] = best_theta  # broadcast to whole niche at once

    return guides_theta


def _select_guides_ring(
    assoc:      np.ndarray,
    F_norm:     np.ndarray,
    ref_dirs:   np.ndarray,
    qpop_theta: np.ndarray,
) -> np.ndarray:
    """8th remedy: Ring-structured guide selection [Tayarani-N &
    Akbarzadeh-T 2014, Evol. Intel. 7:219-239, §3] -- Ring is the structure
    the paper itself found best for combinatorial problems (Table 2:
    "the best structure for the algorithm when solving the Knapsack problem
    is the Ring structure").

    Unlike _select_guides (every population member in a niche is broadcast
    the SAME single niche champion -- the mechanism the diagnostic chapter
    identified as the root cause of QI-NSGA-III's chromosome-diversity
    collapse on the IRP, see Solvers/IRP_results_summary.md), each niche's
    population members are arranged in a ring (fixed order = current array
    order, no cross-generation persistence needed -- unlike pbest/eq. 13,
    this is a purely per-generation grouping, so the individual-identity
    mismatch that broke those two ports does not apply here) and each member
    rotates toward the BEST of itself and its two ring neighbours only, not
    the whole niche's single best. A niche can therefore end up pulling
    toward several different local points instead of collapsing onto one.

    Niches with 1 member trivially guide toward themselves (no rotation).
    Niches with 2 members are each other's only neighbour on both sides
    (harmless duplication, not a special case).

    No archive-fallback for uncovered niches here -- the caller applies
    _supplement_from_archive afterward exactly as with _select_guides, since
    that mechanism is unrelated to the ring topology.
    """
    ref_norms = np.linalg.norm(ref_dirs, axis=1, keepdims=True)
    ref_unit  = ref_dirs / np.where(ref_norms > 1e-9, ref_norms, 1.0)

    guides_theta = qpop_theta.copy()   # default: self (updated per niche below)

    for rd in np.unique(assoc):
        pop_idx = np.where(assoc == rd)[0]
        n = len(pop_idx)

        F_niche = F_norm[pop_idx]
        proj    = F_niche @ ref_unit[rd]
        d_self  = np.maximum((F_niche ** 2).sum(axis=1) - proj ** 2, 0.0)

        idx_prev = np.roll(pop_idx, 1)
        idx_next = np.roll(pop_idx, -1)
        d_prev   = np.roll(d_self, 1)
        d_next   = np.roll(d_self, -1)

        stacked_d   = np.stack([d_self, d_prev, d_next], axis=1)          # (n, 3)
        stacked_idx = np.stack([pop_idx, idx_prev, idx_next], axis=1)     # (n, 3)
        best_local  = stacked_idx[np.arange(n), stacked_d.argmin(axis=1)]

        guides_theta[pop_idx] = qpop_theta[best_local]

    return guides_theta


def _select_guides_crowding(
    assoc:      np.ndarray,
    pareto_idx: np.ndarray,
    F_norm:     np.ndarray,
    qpop_theta: np.ndarray,
) -> np.ndarray:
    """Remedy F: same niche-champion broadcast as _select_guides, but the
    champion is chosen by HIGHEST crowding distance within the niche
    [Deb et al. 2002, §III-B -- _crowding_distance, already used elsewhere
    in this module for archive trimming] instead of LOWEST perpendicular
    distance to the reference ray. See
    docs/superpowers/specs/2026-08-03-qinsga3-crowding-distance-guide-design.md.

    ref_dirs is not needed here -- crowding distance doesn't reference the
    niche's ray, and assoc/pareto_idx already encode niche membership. The
    global fallback (closest-to-origin Pareto member, for niches with no
    Pareto representative) is unchanged from _select_guides -- it is not the
    criterion under test.
    """
    N            = len(assoc)
    F_par_n      = F_norm[pareto_idx]
    global_fb    = qpop_theta[pareto_idx[np.linalg.norm(F_par_n, axis=1).argmin()]]
    pareto_assoc = assoc[pareto_idx]

    guides_theta = np.tile(global_fb, (N, 1))

    for rd in np.unique(pareto_assoc):
        same_mask = pareto_assoc == rd
        same_idx  = pareto_idx[same_mask]

        if len(same_idx) == 1:
            best_theta = qpop_theta[same_idx[0]]
        else:
            F_same     = F_norm[same_idx]
            cd         = _crowding_distance(F_same)
            best_theta = qpop_theta[same_idx[cd.argmax()]]

        guides_theta[assoc == rd] = best_theta

    return guides_theta


def _crowding_saturation_stats(
    assoc:      np.ndarray,
    pareto_idx: np.ndarray,
    F_norm:     np.ndarray,
) -> tuple:
    """Diagnostic only -- not called by _select_guides_crowding itself; wired
    optionally via run_qinsga3's crowding_saturation_log parameter. Added
    after the final review of docs/superpowers/plans/
    2026-08-03-qinsga3-crowding-distance-guide.md flagged (and simulated,
    but never measured) that with several objectives and small niches,
    _crowding_distance can assign inf to every Pareto member of a niche,
    degenerating _select_guides_crowding's argmax over crowding distance
    into an arbitrary positional (first-index) tie-break rather than a
    genuine diversity-driven pick. See the design doc's "Risk" section.

    Returns (n_multi_member_niches, n_fully_saturated_niches): among
    occupied niches with >=2 Pareto members (the only case where the
    argmax can be ambiguous -- a 1-member niche always guides toward
    itself, no crowding distance involved), how many have EVERY member at
    crowding distance = inf.
    """
    pareto_assoc = assoc[pareto_idx]
    n_multi     = 0
    n_saturated = 0

    for rd in np.unique(pareto_assoc):
        same_idx = pareto_idx[pareto_assoc == rd]
        if len(same_idx) < 2:
            continue
        n_multi += 1
        cd = _crowding_distance(F_norm[same_idx])
        if np.all(np.isinf(cd)):
            n_saturated += 1

    return n_multi, n_saturated


def _supplement_from_archive(
    guides_theta: np.ndarray,
    assoc:        np.ndarray,
    pareto_assoc: np.ndarray,
    arch_theta:   np.ndarray,
    arch_F_norm:  np.ndarray,
    ref_dirs:     np.ndarray,
) -> np.ndarray:
    """Fill empty-niche guides from the external archive [Li & Wang 2007, §III-C].

    arch_F_norm is pre-normalised by the caller — the same array is reused by
    _migrate in the same generation, so _normalise_F is called only once total.

    Vectorised: iterates only over niches that are (a) not covered by the current
    Pareto front and (b) present in the population — typically a small subset of
    n_ref_dirs.
    """
    arch_assoc = _assign_ref_dirs(arch_F_norm, ref_dirs)
    covered    = set(pareto_assoc.tolist())

    ref_norms = np.linalg.norm(ref_dirs, axis=1, keepdims=True)
    ref_unit  = ref_dirs / np.where(ref_norms > 1e-9, ref_norms, 1.0)

    pop_rds           = np.unique(assoc)
    uncovered_pop_rds = pop_rds[~np.isin(pop_rds, list(covered))]

    for rd in uncovered_pop_rds:
        in_niche = np.where(arch_assoc == rd)[0]
        if len(in_niche) == 0:
            continue
        if len(in_niche) == 1:
            best_theta = arch_theta[in_niche[0]]
        else:
            F_cand  = arch_F_norm[in_niche]
            proj    = F_cand @ ref_unit[rd]
            d_perp2 = np.maximum((F_cand ** 2).sum(axis=1) - proj ** 2, 0.0)
            best_theta = arch_theta[in_niche[d_perp2.argmin()]]

        guides_theta[assoc == rd] = best_theta

    return guides_theta


def _supplement_from_archive_crowding(
    guides_theta: np.ndarray,
    assoc:        np.ndarray,
    pareto_assoc: np.ndarray,
    arch_theta:   np.ndarray,
    arch_F_norm:  np.ndarray,
    ref_dirs:     np.ndarray,
) -> np.ndarray:
    """Remedy F counterpart to _supplement_from_archive: fills the same
    uncovered-niche guides from the external archive, but the archive
    candidate is chosen by highest crowding distance within the niche
    instead of lowest perpendicular distance to the reference ray -- kept
    consistent with _select_guides_crowding so no single generation mixes
    the two criteria across niches. See
    docs/superpowers/specs/2026-08-03-qinsga3-crowding-distance-guide-design.md.

    ref_dirs is still needed here (only) to compute arch_assoc via
    _assign_ref_dirs -- niche MEMBERSHIP is still by reference-ray
    association; only the in-niche tie-break criterion changes.
    """
    arch_assoc = _assign_ref_dirs(arch_F_norm, ref_dirs)
    covered    = set(pareto_assoc.tolist())

    pop_rds           = np.unique(assoc)
    uncovered_pop_rds = pop_rds[~np.isin(pop_rds, list(covered))]

    for rd in uncovered_pop_rds:
        in_niche = np.where(arch_assoc == rd)[0]
        if len(in_niche) == 0:
            continue
        if len(in_niche) == 1:
            best_theta = arch_theta[in_niche[0]]
        else:
            F_cand     = arch_F_norm[in_niche]
            cd         = _crowding_distance(F_cand)
            best_theta = arch_theta[in_niche[cd.argmax()]]

        guides_theta[assoc == rd] = best_theta

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
    _supplement_from_archive) — no additional _normalise_F call needed here.
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
# Niche-recentring reset operator
# [Tayarani-N & Akbarzadeh-T 2014, Evol. Intel. 7:219-239, §5], adapted --
# see docs/superpowers/specs/2026-08-01-qinsga3-niche-recentring-reset-design.md
# for why. Two of the paper's mechanisms don't transfer to QINSGA3 and were
# dropped rather than reused verbatim:
#   - eq. 11 (convergence = theta near 0/pi/2): QINSGA3 initialises and
#     converges around theta=pi/4 ("maximum superposition"), not toward the
#     classical-bit boundaries the paper's formula was written for -- measured
#     0% trigger on the real IRP at every gamma from 0.99 to 0.50.
#   - eq. 13 (stagnation gate = guide unchanged for T generations): QINSGA3's
#     elitist survival re-selects the whole population from a freshly merged
#     parent+offspring pool every generation, so the per-niche "champion" has
#     no structural reason to stay identical across generations even when the
#     niche is otherwise tightly converged -- measured 0/60 generations where
#     any niche satisfied this on the real IRP (median guide shift 0.033 rad
#     generation to generation, comparable to the clustering threshold itself).
# What's kept: eq. 12 (mutual theta-distance clustering) as the sole trigger,
# checked on every occupied niche every generation; keep-best-by-reference-ray
# selection; the reset value is a fresh U(0, pi/2) draw rather than the
# paper's theta<-pi/4 (pi/4 is where the population already sits, so
# resetting there would be close to a no-op -- see design doc).
# ---------------------------------------------------------------------------

def _recentring_reset_mask(
    assoc:      np.ndarray,
    qpop_theta: np.ndarray,
    F_norm:     np.ndarray,
    ref_dirs:   np.ndarray,
    delta:      float,
) -> np.ndarray:
    """Detect converged clusters in every occupied niche, by direct theta
    distance (eq. 12, unmodified):

        (1/n_genes) * sum_k |theta_ik - theta_jk| / (pi/2) < delta

    Within each niche with >=2 members, "best" (kept) is the member closest to
    the niche's reference ray -- the same criterion _select_guides already
    uses. Every other member within delta of the best is marked for reset (the
    caller decides the reset value).

    Returns a boolean mask of shape (pop_size,): True for individuals to reset.
    """
    pop_size = len(assoc)
    reset    = np.zeros(pop_size, dtype=bool)

    ref_norms = np.linalg.norm(ref_dirs, axis=1, keepdims=True)
    ref_unit  = ref_dirs / np.where(ref_norms > 1e-9, ref_norms, 1.0)

    for rd in np.unique(assoc):
        niche_idx = np.where(assoc == rd)[0]
        if len(niche_idx) < 2:
            continue

        proj    = F_norm[niche_idx] @ ref_unit[rd]
        d_perp2 = np.maximum((F_norm[niche_idx] ** 2).sum(axis=1) - proj ** 2, 0.0)
        best_local = niche_idx[d_perp2.argmin()]

        best_theta   = qpop_theta[best_local]
        dist         = np.abs(qpop_theta[niche_idx] - best_theta).mean(axis=1) / (np.pi / 2.0)
        similar_mask = dist < delta

        losers = niche_idx[similar_mask & (niche_idx != best_local)]
        reset[losers] = True

    return reset


# ---------------------------------------------------------------------------
# RQPSO-style dual-attractor rotation (7th remedy)
# [Bodha, Arun, Awasthi, Mahato & Fotis 2025, "Rotational gate based quantum
# particle swarm optimization for benchmark suites and combined economic
# emission dispatch", Engineering Research Express 7(4):045345], adapted --
# see the docstring of sensitivity/compare_rqpso_rotation.py for the full
# derivation. Two adaptations, both necessary (domain change), not invented:
#   - The paper's theta lives on a full circle [0, 2*pi) and uses a
#     shortest-arc WRAP(.) distance; QINSGA3's theta is a bounded, non-cyclic
#     parameter in [0, pi/2] (Li & Wang 2007's cos^2 mixing angle, not a
#     phase) -- there is no wraparound to exploit, so the plain difference
#     (guide - theta) is used instead, normalised by the domain width pi/2
#     (matches the eq. 12 port's own convention elsewhere in this file).
#   - The paper's Theta_t = pi*(1 - t/T) rotation budget is specific to its
#     [0, 2*pi) domain's scale; reusing it verbatim would apply a step an
#     order of magnitude too large for QINSGA3's pi/2-wide domain. Replaced
#     with QINSGA3's own already-calibrated alpha_max as the magnitude scale,
#     keeping the paper's linear decay-to-zero SHAPE unchanged:
#     theta_step = alpha_max * (1 - gen/max_gen).
# c1 = c2 = 2.05 is the paper's own value, kept as-is (not re-tuned).
#
# pbest ("personal best") has no clean equivalent in QINSGA3: unlike RQPSO's
# non-recombining particles (persistent identity across iterations), QINSGA3
# has SBX crossover, and elitist survival re-selects the whole population
# from a freshly merged parent+offspring pool every generation -- there is no
# stable per-individual identity to track a personal history against (the
# same architectural mismatch already found for the niche-recentring
# operator's stagnation gate, see docs/superpowers/specs/
# 2026-08-01-qinsga3-niche-recentring-reset-design.md). Approximated here by
# tracking a best-ever theta/quality PER POPULATION SLOT INDEX rather than
# per individual identity -- an explicit, documented approximation, not a
# faithful port, using the same positional convention _migrate already uses
# elsewhere in this file.
# ---------------------------------------------------------------------------

_RQPSO_C1 = 2.05
_RQPSO_C2 = 2.05


def _update_pbest(
    pbest_theta: np.ndarray,
    pbest_scalar: np.ndarray,
    theta: np.ndarray,
    F_norm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-slot running best (see module note above): scalar quality is the
    mean of the normalised objectives (equal-weight scalarisation, matching
    the E(v, w, B) = sum_l w_l*f_l(v) construction with w_l=1/h used in
    Demidova & Maslennikov 2025 [same issue as ref. [65], "QI-NSGA-III"] --
    lower is better since F_norm is oriented for minimisation). Updates
    pbest_theta/pbest_scalar in place where this generation's slot quality
    improves on its own recorded best.
    """
    quality = F_norm.mean(axis=1)
    improved = quality < pbest_scalar
    new_pbest_theta = np.where(improved[:, None], theta, pbest_theta)
    new_pbest_scalar = np.where(improved, quality, pbest_scalar)
    return new_pbest_theta, new_pbest_scalar


def _rqpso_rotate(
    theta:       np.ndarray,
    pbest_theta: np.ndarray,
    guide_theta: np.ndarray,
    theta_step:  float,
    rng:         np.random.Generator,
) -> np.ndarray:
    """Dual-attractor rotation update (eq. Δθ in the paper, domain-adapted):

        Δθ = c1*u1*((pbest - theta)/(pi/2))*theta_step
           + c2*u2*((guide - theta)/(pi/2))*theta_step

    u1, u2 ~ U(0,1), sampled independently per gene per individual (matches
    the paper's per-qubit sampling). Returns the new, clipped theta.
    """
    u1 = rng.uniform(0.0, 1.0, size=theta.shape)
    u2 = rng.uniform(0.0, 1.0, size=theta.shape)
    diff_p = (pbest_theta - theta) / (np.pi / 2.0)
    diff_g = (guide_theta - theta) / (np.pi / 2.0)
    delta = _RQPSO_C1 * u1 * diff_p * theta_step + _RQPSO_C2 * u2 * diff_g * theta_step
    return np.clip(theta + delta, 0.0, np.pi / 2.0)


# ---------------------------------------------------------------------------
# PSO-style momentum rotation (9th remedy)
# [Li, Xu, Liu & Li 2008, "Quantum Multi-objective Evolutionary Algorithm
# with Particle Swarm Optimization Method", ICNC 2008]. Same dual-attractor
# targets as _rqpso_rotate (pbest, niche guide as gbest) but with a genuinely
# new ingredient never tested before: a VELOCITY that PERSISTS across
# generations (inertia), instead of every remedy so far recomputing the step
# from scratch each generation. c1=c2=2 and the alpha_max/alpha_min bounds
# already in this file (Vmax=0.10*pi, Vmin=0.001*pi in the paper's own
# notation) are this same paper's own values -- already the cited source of
# QINSGA3's alpha_max/alpha_min (see README), just never used with momentum
# before.
#
# Two adaptations, both necessary and documented:
#   - The paper reports Vmax/Vmin as fixed bounds without a decay schedule;
#     reusing QINSGA3's own already-adopted linear decay (alpha_max ->
#     alpha_min over the run, same convention as the existing tanh rotation
#     and _rqpso_rotate's theta_step) rather than inventing a new schedule.
#   - The adaptive inertia weight w_i = D(i)/N + m(i)/N (paper's eq. 3.1) is
#     ported verbatim (D(i) = per-individual "max-min distance" density in
#     theta-space, eq. in §3.1; m(i) = number of individuals i dominates,
#     via the same Pareto dominance check _archive_update already uses) but
#     clipped to [0, 2]: the paper's own formula has no clip and was only
#     validated on their own single-archive knapsack setup, not established
#     to stay non-negative in general -- w_i<0 would invert the velocity
#     term and cause divergence, an instability outside the paper's own
#     tested regime rather than part of its actual mechanism.
# ---------------------------------------------------------------------------

_PSO_C1 = 2.0
_PSO_C2 = 2.0


def _max_min_density(theta: np.ndarray) -> np.ndarray:
    """Per-individual "max-min distance" density D(i) [Li, Xu, Liu & Li 2008,
    §3.1]: d_ji = Euclidean distance between i and j in theta-space;
    d_min(i) = min over j!=i of d_ji; d_max_min = max over i of d_min(i) (the
    single most-isolated individual's nearest-neighbour distance, used as a
    population-wide density threshold); D(i) = sum over j!=i of
    sign(d_max_min - d_ji) -- positive when many other individuals are
    closer to i than that threshold (i is in a crowded region), negative
    when i is relatively isolated.
    """
    sq = (theta ** 2).sum(axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * theta @ theta.T, 0.0)
    d = np.sqrt(d2)

    d_for_min = d.copy()
    np.fill_diagonal(d_for_min, np.inf)
    d_min = d_for_min.min(axis=1)
    d_max_min = d_min.max()

    sign_matrix = np.sign(d_max_min - d)
    np.fill_diagonal(sign_matrix, 0.0)  # exclude j == i from the sum
    return sign_matrix.sum(axis=1)


def _domination_counts(F: np.ndarray) -> np.ndarray:
    """m(i) = number of other individuals that i Pareto-dominates (F assumed
    oriented for minimisation, same convention as the rest of this file's
    dominance checks, e.g. _archive_update)."""
    dominates = (F[:, None, :] <= F[None, :, :]).all(axis=2) & \
                (F[:, None, :] < F[None, :, :]).any(axis=2)
    np.fill_diagonal(dominates, False)
    return dominates.sum(axis=1)


def _adaptive_inertia(D: np.ndarray, m: np.ndarray, pop_size: int) -> np.ndarray:
    """w_i = D(i)/N + m(i)/N [eq. 3.1], clipped to [0, 2] for stability (see
    module note above -- not part of the paper's own formula, a necessary
    safeguard against the untested w_i < 0 regime)."""
    w = D / pop_size + m / pop_size
    return np.clip(w, 0.0, 2.0)


def _pso_rotate(
    theta:       np.ndarray,
    velocity:    np.ndarray,
    pbest_theta: np.ndarray,
    guide_theta: np.ndarray,
    w:           np.ndarray,
    v_clip:      float,
    rng:         np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """v(t+1) = w*v(t) + c1*u1*(pbest-theta) + c2*u2*(guide-theta)
       theta(t+1) = theta(t) + v(t+1)
    [Li, Xu, Liu & Li 2008, §2.3/3.4]. w is per-individual (shape (pop_size,)).
    v is clipped to [-v_clip, v_clip] (see module note re: Vmax/Vmin decay).
    Returns (new_theta, new_velocity) -- velocity must be threaded back into
    the caller's state for the next generation (this is the actual novelty:
    a persistent momentum term no prior remedy in this project has used).
    """
    u1 = rng.uniform(0.0, 1.0, size=theta.shape)
    u2 = rng.uniform(0.0, 1.0, size=theta.shape)
    new_v = (w[:, None] * velocity
             + _PSO_C1 * u1 * (pbest_theta - theta)
             + _PSO_C2 * u2 * (guide_theta - theta))
    new_v = np.clip(new_v, -v_clip, v_clip)
    new_theta = np.clip(theta + new_v, 0.0, np.pi / 2.0)
    return new_theta, new_v


# ---------------------------------------------------------------------------
# Chaos-modulated rotation (10th remedy)
# [Hu Feng-jun & Wu Bin 2009, "Quantum Evolutionary Algorithm for Vehicle
# Routing Problem with Simultaneous Delivery and Pickup", Joint 48th IEEE
# CDC / 28th Chinese Control Conference, eq. 15-19]. Unlike every prior
# remedy (tanh, RQPSO, PSO-momentum), the rotation MAGNITUDE here is not a
# smooth function of generation index or fitness rank -- it is modulated by
# a per-individual CHAOTIC sequence (logistic map, mu>=4), whose stationary
# distribution is U-shaped (more time spent near 0 and near 1 than near
# 0.5), producing bursty step sizes instead of a monotonically decaying
# schedule. The paper's own magnitude (eq. 16) is an RMS distance to
# "several best individuals" rather than a single guide.
#
# Three adaptations, all necessary (binary/VRP -> continuous IRP domain
# mismatch, or numerical-stability guards), not invented:
#   - eq. 17's direction chi = sgn(alpha_i*beta_i*(b_i-0.5)) depends on a
#     BINARY bit b_i of the best solution (grey-binary VRP encoding in the
#     source paper) -- meaningless for QINSGA3's continuous theta. Replaced
#     with chi = sign(guide_theta - theta), the same "which side is the
#     target on" direction logic already used by every other rotation
#     variant in this file (tanh, RQPSO, PSO).
#   - eq. 16's "K best individuals" (the paper's own external best-solution
#     archive B(t), Fig. 1) maps directly onto QINSGA3's own external
#     archive (_archive_update) -- no new elitism machinery invented, reused
#     as-is, restricted per niche via the archive's reference-direction
#     association. Niches not yet covered by the archive fall back to a
#     single-point RMS (K=1: the niche's own guide, itself already resolved
#     through _select_guides/_select_guides_ring + _supplement_from_archive)
#     rather than being left undefined.
#   - lambda must persist as a genuine chaotic TIME SERIES across
#     generations (eq. 19 evolves lambda_t -> lambda_{t+1}, only seeded once
#     from the individual's own fitness at generation 0) -- the same
#     individual-identity mismatch already documented for pbest/velocity
#     (QINSGA3's elitist survival re-selects the whole population from a
#     freshly merged pool every generation). Approximated the same way:
#     lambda tracked PER POPULATION SLOT, not per individual identity.
#     Additionally clipped away from the exact fixed points 0.0/1.0 (not in
#     the paper) -- floating-point rounding collapses the logistic map to
#     the absorbing fixed point 0 after enough iterations otherwise, which
#     would silently turn the chaotic term into a constant and defeat the
#     entire mechanism being tested.
# ---------------------------------------------------------------------------

_CHAOS_MU = 4.0


def _elite_rms_distance(
    theta:        np.ndarray,
    assoc:        np.ndarray,
    guides_theta: np.ndarray,
    arch_theta:   np.ndarray | None,
    arch_assoc:   np.ndarray | None,
) -> np.ndarray:
    """Per-gene RMS distance from each individual to its niche's archive
    elites [eq. 16, adapted -- see module note above]:

        Delta_theta_i = sqrt( mean_k( (elite_k - theta_i)^2 ) )

    Niches with no archive elites yet fall back to a K=1 "RMS" against the
    niche's own already-resolved guide (reduces to |guide - theta|).
    """
    result = np.empty_like(theta)

    for rd in np.unique(assoc):
        pop_idx = np.where(assoc == rd)[0]
        theta_n = theta[pop_idx]

        elite_idx = (np.where(arch_assoc == rd)[0]
                     if arch_assoc is not None else np.array([], dtype=int))

        if len(elite_idx) > 0:
            elites = arch_theta[elite_idx]                          # (K, D)
            diff   = theta_n[:, None, :] - elites[None, :, :]        # (n, K, D)
            rms    = np.sqrt((diff ** 2).mean(axis=1))               # (n, D)
        else:
            rms = np.abs(theta_n - guides_theta[pop_idx])

        result[pop_idx] = rms

    return result


def _chaotic_lambda_seed(F_norm: np.ndarray) -> np.ndarray:
    """lambda_0 = f(X_i) [eq. 18], using the same equal-weight normalised
    quality scalar as _update_pbest, clipped away from the logistic map's
    absorbing fixed points 0.0/1.0 (see module note)."""
    quality = F_norm.mean(axis=1)
    return np.clip(quality, 1e-4, 1.0 - 1e-4)


def _chaotic_lambda_step(lam: np.ndarray, mu: float = _CHAOS_MU) -> np.ndarray:
    """Logistic map [eq. 19]: lambda_{t+1} = mu*lambda_t*(1-lambda_t), mu>=4
    for full chaos. Clipped away from 0.0/1.0 for the same reason as the seed."""
    return np.clip(mu * lam * (1.0 - lam), 1e-6, 1.0 - 1e-6)


def _chaotic_rotate(
    theta:       np.ndarray,
    guide_theta: np.ndarray,
    elite_rms:   np.ndarray,
    lam:         np.ndarray,
) -> np.ndarray:
    """theta_i = chi * Delta_theta_i * (1 + chi*lambda_i)  [eq. 15, adapted
    direction -- see module note]. Deterministic given (elite_rms, lam): no
    stochastic sampling, unlike RQPSO/PSO -- chaos comes from the logistic
    map's own trajectory, not injected randomness.
    """
    chi  = np.sign(guide_theta - theta)
    step = chi * elite_rms * (1.0 + chi * lam[:, None])
    return np.clip(theta + step, 0.0, np.pi / 2.0)


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

    Standard Pareto dominance check, vectorised:
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

def _encode_theta(X: np.ndarray, xl: np.ndarray, xu: np.ndarray) -> np.ndarray:
    """Inverse of QuantumPopulation.measure()'s mean map: p = (x−xl)/(xu−xl),
    θ = arccos(sqrt(p)) so cos²(θ) reproduces p (up to measurement noise)."""
    p = np.clip((X - xl) / np.where(xu - xl > 1e-12, xu - xl, 1.0), 0.0, 1.0)
    return np.clip(np.arccos(np.sqrt(p)), 0.0, np.pi / 2.0)


def run_qinsga3(
    sets_,
    params_,
    ref_dirs:         np.ndarray,
    pop_size:         int   = 200,
    max_gen:          int   = 300,
    alpha_max:        float = 0.10  * np.pi,
    alpha_min:        float = 0.001 * np.pi,
    p_cross:          float = 0.9,
    eta_cross:        float = 20.0,
    p_mut:            float | None = None,
    eta_mut:          float = 20.0,
    migration_period: int   = 10,
    n_migrate:        int   = 10,
    delta_similar:    float = 0.0,
    noise_scale:      float = 0.02,
    use_rqpso_rotation: bool = False,
    use_ring_guides:  bool   = False,
    use_crowding_guides: bool = False,
    use_pso_rotation: bool   = False,
    use_chaotic_rotation: bool = False,
    use_route_repair: bool = False,
    repair_final_front: bool = True,
    seed:             int   = 42,
    rotation_type:    str   = "tanh",
    callback          = None,
    crowding_saturation_log: list | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one QINSGA-III instance. Returns (pareto_X, pareto_F, pareto_G).

    Population evaluation is parallelised via ProcessPoolExecutor: the pool is
    created once per call; each worker process is initialised with a persistent
    IRPProblem singleton (_worker_init) so construction cost is paid once, not
    once per evaluation.  Assumes IRPProblem._evaluate is deterministic — results
    are identical to the sequential version for the same seed.

    eta_cross/eta_mut default to 20 to match NSGA-III's own SBX/PM operators
    (NSGA3/main.py) — crossover and mutation now happen in X-space (see module
    docstring), so these parameters mean the same thing as NSGA-III's.

    delta_similar controls the niche-recentring reset operator (disabled by
    default, delta_similar=0.0) -- see
    docs/superpowers/specs/2026-08-01-qinsga3-niche-recentring-reset-design.md.
    Ablation-only until validated against Solvers/NSGA3 with the project's
    shared ideal/nadir + Mann-Whitney protocol.

    noise_scale is exposed here (default 0.02, unchanged) so ablations can
    test strengthening QuantumPopulation.measure()'s diversity noise -- see
    Guzel et al. (2022), "QNSGA-II: A Quantum Computing-Inspired Approach to
    Multi-Objective Optimization" (IEEE ISNCC): QNSGA-II's whole population
    starts from IDENTICAL quantum chromosomes and relies entirely on
    probabilistic measurement (not chromosome-level separation) for
    diversity. QINSGA3 already has a measurement-noise term
    (chromosome.py::measure, Platel et al. 2009) but it defaults to a small
    value tuned only to break integer-rounding ties, not to carry the
    diversity load the way QNSGA-II's measurement does.

    use_rqpso_rotation (disabled by default) replaces the tanh rotation gate
    with the dual-attractor update from Bodha, Arun, Awasthi, Mahato & Fotis
    (2025) -- see the module note above _rqpso_rotate for the full formula
    and the two documented domain adaptations. alpha_min/rotation_type are
    unused when this is enabled (alpha_max still sets the step magnitude).

    use_ring_guides (disabled by default) replaces _select_guides's single
    niche-wide champion with the Ring-structured local guide from
    Tayarani-N & Akbarzadeh-T (2014) §3 -- see _select_guides_ring's
    docstring. Combinable with use_rqpso_rotation (gbest becomes the ring
    guide instead of the niche champion) but validated independently first.

    use_crowding_guides (disabled by default) replaces _select_guides's
    reference-ray-closest niche champion (and _supplement_from_archive's
    archive fallback) with the HIGHEST-crowding-distance member instead --
    see _select_guides_crowding's docstring for the full rationale and
    docs/superpowers/specs/2026-08-03-qinsga3-crowding-distance-guide-design.md
    for the design. Mutually exclusive with use_ring_guides (both replace
    the same guide-selection step) -- combinable in principle with the
    rotation-rule variants (use_rqpso_rotation/use_pso_rotation/
    use_chaotic_rotation) but validated alone first, same as every other
    remedy in this module.

    crowding_saturation_log (disabled by default, None) -- if a list is
    passed and use_crowding_guides is True, one (n_multi_member_niches,
    n_fully_saturated_niches) tuple from _crowding_saturation_stats is
    appended to it every generation. Diagnostic only, added after the
    final review of docs/superpowers/plans/
    2026-08-03-qinsga3-crowding-distance-guide.md found the design doc's
    own "Risk" section (crowding distance saturating to inf for small
    niches) had been simulated but never actually measured on a real run.

    use_pso_rotation (disabled by default) replaces the tanh rotation gate
    with the momentum-based update from Li, Xu, Liu & Li (2008) -- see the
    module note above _pso_rotate for the full formula and the two
    documented adaptations (velocity-bound decay schedule, inertia clip).
    Mutually exclusive with use_rqpso_rotation in practice (both claim the
    rotation step; only one should be True at a time).

    use_chaotic_rotation (disabled by default) replaces the tanh rotation
    gate with the chaos-modulated update from Hu Feng-jun & Wu Bin (2009) --
    see the module note above _chaotic_rotate for the full formula and the
    three documented adaptations (direction, archive-as-B(t), positional
    lambda state). Mutually exclusive with the other rotation variants.

    use_route_repair (disabled by default) replaces each worker's call to
    _worker_eval with _worker_eval_repaired, applying a 2-opt local-search
    repair (see Solvers/QINSGA3/repair.py) to every individual's decoded
    route before scoring it, for both parent and offspring populations
    every generation. Baldwinian: the repair never changes the chromosome,
    only the fitness it is scored with. See
    docs/superpowers/specs/2026-08-04-qinsga3-route-repair-design.md.
    Consequence: the returned Pareto front's pareto_F reflects repaired
    fitness, but pareto_X (the chromosomes) will NOT reproduce those exact
    objective values if decoded through the normal
    decode_chromosome/build_routes/compute_f1..f4 path without also
    re-running _repair_route_result -- the repair is Baldwinian
    (fitness-only), never re-encoded into the chromosome, matching this
    module's other design notes on the same topic.

    repair_final_front (ENABLED by default -- the one adopted correction
    remedy G produced; see Solvers/QINSGA3/README.md's "Design history")
    is the practical counterpart to use_route_repair: instead of repairing
    every individual every generation (which is what makes use_route_repair
    slow -- ~3.5x to ~15x baseline depending on scale, even after the
    delta-cost/merged-pass/windowed-search optimisations in repair.py),
    this repairs ONLY the returned Pareto front, ONCE, after the
    generational loop has already finished. The search loop's own runtime
    is completely unaffected -- this trades the "does repairing the
    search's fitness signal help guide selection/survival throughout the
    run" research question (what use_route_repair tests, still disabled
    by default -- an ablation-only research variant, not adopted) for a
    purely practical one: does polishing the final reported front improve
    it, at effectively zero added cost to the algorithm's own runtime.
    Validated at the project's own 20-seed gold-standard protocol (shared
    ideal/nadir vs Solvers/NSGA3's cache, Mann-Whitney U): HV +29.2%
    (p=0.000059), GD -15.3% (p=0.001116), IGD -10.4% (p=0.000179), all
    significant, runtime unchanged vs the pre-remedy-G baseline -- see
    Solvers/IRP_results_summary.md's "Remède G — variante pratique"
    section for the full numbers. Still significantly worse than NSGA-III
    on the same instance (the gap is not closed, only narrowed). Set
    repair_final_front=False to reproduce the pre-remedy-G behaviour (e.g.
    for ablation comparisons against this new default -- see
    sensitivity/compare_route_repair_final.py). Mutually exclusive in
    practice with use_route_repair (combining both would repair the front
    twice, redundantly) -- not asserted against, since nothing currently
    calls them together, but do not combine them. Same Baldwinian
    consequence as use_route_repair: pareto_X is unchanged, only pareto_F/
    pareto_G are repaired.
    """
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    from Solvers.NSGA3.problem import IRPProblem

    rng     = np.random.default_rng(seed)
    problem = IRPProblem(sets_, params_)   # main-process copy — used for metadata + SBX/PM bounds

    xl       = np.asarray(problem.xl, dtype=float)
    xu       = np.asarray(problem.xu, dtype=float)
    n_genes  = problem.n_var
    n_constr = problem.n_ieq_constr

    if p_mut is None:
        p_mut = 1.0 / n_genes   # matches NSGA-III's pm = 1/D convention

    assert not (use_ring_guides and use_crowding_guides), (
        "use_ring_guides and use_crowding_guides both replace the same "
        "guide-selection step and are mutually exclusive"
    )
    assert sum([use_rqpso_rotation, use_pso_rotation, use_chaotic_rotation]) <= 1, (
        "use_rqpso_rotation, use_pso_rotation and use_chaotic_rotation all "
        "replace the same rotation-gate step and are mutually exclusive "
        "(the per-generation dispatch below silently applies only the "
        "first True flag, in that priority order, if more than one is set)"
    )

    qpop     = QuantumPopulation(pop_size, n_genes, xl, xu, rng=rng, rotation_type=rotation_type,
                                  noise_scale=noise_scale)
    sorter   = NonDominatedSorting()
    survival = ReferenceDirectionSurvival(ref_dirs)
    sbx_op   = SBX(prob=p_cross, eta=eta_cross)
    # prob_var (per-gene rate), NOT prob (pymoo's per-INDIVIDUAL mutation
    # gate): pm_op is invoked below via _do() directly (not pymoo's own
    # Mutation.do() wrapper, which is what actually reads self.prob), so
    # self.prob is never consulted here -- only self.prob_var controls the
    # per-gene rate _do() uses (PolynomialMutation._do -> get_prob_var()).
    # Passing p_mut as `prob` (as before) silently discarded it: pymoo's own
    # get_prob_var() fallback (min(0.5, 1/problem.n_var)) took over instead,
    # making p_mut a dead parameter -- see sensitivity/compare_pmut.py, which
    # varies p_mut expecting a real effect. At p_mut's own default (1/n_genes
    # == 1/problem.n_var), this fallback already matched p_mut numerically,
    # so this fix is a no-op for every default-parameter run; it only
    # restores effect for callers that pass a non-default p_mut.
    pm_op    = PM(prob_var=p_mut, eta=eta_mut)

    arch_X:    list[np.ndarray] = []
    arch_F:    list[np.ndarray] = []
    arch_theta: list[np.ndarray] = []
    _MAX_ARCHIVE = 500

    if use_rqpso_rotation or use_pso_rotation:
        pbest_theta  = qpop.theta.copy()
        pbest_scalar = np.full(pop_size, np.inf)
    if use_pso_rotation:
        velocity = np.zeros_like(qpop.theta)
    if use_chaotic_rotation:
        lam = None   # seeded from generation 0's own F_norm (eq. 18), then evolves per-slot

    n_workers = min(os.cpu_count() or 1, pop_size)
    chunksize = max(1, pop_size // (2 * n_workers))

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(sets_, params_),
    ) as pool:

        def _eval_batch(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Evaluate all individuals in X in parallel across worker processes."""
            worker_fn = _worker_eval_repaired if use_route_repair else _worker_eval
            results = list(pool.map(worker_fn, list(X), chunksize=chunksize))
            return (
                np.array([r[0] for r in results]),
                np.array([r[1] for r in results]),
            )

        for gen in range(max_gen):
            theta_parent = qpop.theta.copy()
            X_parent     = qpop.measure()
            F_parent, G_parent = _eval_batch(X_parent)
            F_pen_parent = _penalised_F(F_parent, G_parent)

            pareto_idx = sorter.do(F_pen_parent)[0]
            _archive_update(
                X_parent[pareto_idx], F_parent[pareto_idx], G_parent[pareto_idx],
                theta_parent[pareto_idx], arch_X, arch_F, arch_theta, _MAX_ARCHIVE,
            )

            # Share the SAME ideal/nadir as the elitist survival step below
            # (pymoo's own ReferenceDirectionSurvival.norm, monotonic across
            # generations) for guide selection and niching -- see _normalise_F's
            # docstring. Not yet populated on generation 0 (before survival.do()
            # has run once), so that first generation falls back to a
            # from-scratch estimate exactly as before.
            #
            # Uses F_parent (the REAL objectives), NOT F_pen_parent, for the
            # exact same reason the elitist survival step below already uses
            # F_true_pool instead of a penalised F (see that step's own
            # comment): _penalised_F adds the SAME scalar penalty to every
            # objective, which for an infeasible individual is orders of
            # magnitude larger than the objectives themselves -- normalising
            # that collapses the individual's direction in objective space
            # onto a near-constant ray (proportional to 1/(nadir-ideal) per
            # objective), independent of its actual F. Every infeasible
            # individual in the population was therefore being assigned to
            # (effectively) the SAME reference direction / niche, regardless
            # of where it actually sat in objective space -- corrupting guide
            # selection for exactly the individuals furthest from feasibility.
            # F_pen_parent is still the right choice for pareto_idx above
            # (ranking: penalised F correctly makes any infeasible individual
            # dominated by any feasible one) -- only the DIRECTION computation
            # here needs the real objectives, matching the survival step's own
            # already-established "real F for niching, penalty for ranking"
            # split.
            if survival.norm.nadir_point is None:
                F_norm = _normalise_F(F_parent)
            else:
                F_norm = _normalise_F(F_parent, survival.norm.ideal_point, survival.norm.nadir_point)
            assoc  = _assign_ref_dirs(F_norm, ref_dirs)

            if use_chaotic_rotation and lam is None:
                lam = _chaotic_lambda_seed(F_norm)

            if use_ring_guides:
                guides_theta = _select_guides_ring(assoc, F_norm, ref_dirs, qpop.theta)
            elif use_crowding_guides:
                guides_theta = _select_guides_crowding(assoc, pareto_idx, F_norm, qpop.theta)
                if crowding_saturation_log is not None:
                    crowding_saturation_log.append(
                        _crowding_saturation_stats(assoc, pareto_idx, F_norm)
                    )
            else:
                guides_theta = _select_guides(assoc, pareto_idx, F_norm, ref_dirs, qpop.theta)

            # arch_F_norm computed once and shared by both _supplement_from_archive
            # and _migrate — avoids a redundant _normalise_F call per generation
            arch_theta_arr = None
            arch_F_norm    = None
            if len(arch_X) >= 4:
                arch_theta_arr = np.array(arch_theta)
                if survival.norm.nadir_point is None:
                    arch_F_norm = _normalise_F(np.array(arch_F))
                else:
                    arch_F_norm = _normalise_F(np.array(arch_F), survival.norm.ideal_point, survival.norm.nadir_point)
                pareto_assoc   = assoc[pareto_idx]
                supplement_fn  = (_supplement_from_archive_crowding if use_crowding_guides
                                   else _supplement_from_archive)
                guides_theta   = supplement_fn(
                    guides_theta, assoc, pareto_assoc,
                    arch_theta_arr, arch_F_norm, ref_dirs,
                )

            alpha = alpha_min + (alpha_max - alpha_min) * (1.0 - gen / max_gen)

            # --- Quantum step: rotation stays in theta-space -----------------
            if use_rqpso_rotation:
                theta_step = alpha_max * (1.0 - gen / max_gen)
                qpop.theta = _rqpso_rotate(qpop.theta, pbest_theta, guides_theta, theta_step, rng)
            elif use_pso_rotation:
                v_clip = alpha_min + (alpha_max - alpha_min) * (1.0 - gen / max_gen)
                D_density = _max_min_density(qpop.theta)
                m_dom     = _domination_counts(F_pen_parent)
                w_inertia = _adaptive_inertia(D_density, m_dom, pop_size)
                qpop.theta, velocity = _pso_rotate(
                    qpop.theta, velocity, pbest_theta, guides_theta, w_inertia, v_clip, rng,
                )
            elif use_chaotic_rotation:
                arch_assoc_arr = (_assign_ref_dirs(arch_F_norm, ref_dirs)
                                   if arch_F_norm is not None else None)
                elite_rms = _elite_rms_distance(
                    qpop.theta, assoc, guides_theta, arch_theta_arr, arch_assoc_arr,
                )
                qpop.theta = _chaotic_rotate(qpop.theta, guides_theta, elite_rms, lam)
                lam = _chaotic_lambda_step(lam)
            else:
                qpop.rotate(guides_theta, alpha)
            X_rotated = qpop.measure()

            # --- Variation step: SBX + PM in X-SPACE (matches NSGA-III) -----
            if p_cross > 0.0:
                # CV-based tournament (matches pymoo's NSGA-III default
                # mating selection, see _tournament_select_parents's
                # docstring) -- winners paired consecutively into pop_size//2
                # mating pairs, same shape the old pure-random permutation
                # pairing produced.
                winners = _tournament_select_parents(G_parent, pop_size, rng)
                n_pairs = pop_size // 2
                pairs   = winners[: n_pairs * 2].reshape(n_pairs, 2)
                X_pairs = np.transpose(X_rotated[pairs], (1, 0, 2))  # (2, n_matings, n_var)
                Q       = sbx_op._do(problem, X_pairs, random_state=rng)
                # Per-pair crossover probability p_cross, applied explicitly:
                # sbx_op is called via _do() directly (not pymoo's own
                # Crossover.do() wrapper, core/crossover.py), so self.prob
                # (set to p_cross at construction) is never consulted --
                # _do() always produces crossed offspring for every pair.
                # do()'s own semantics (compute Q for all pairs, then keep it
                # only for pairs selected by rng.random(n) < prob, copying
                # the parents through unchanged otherwise) are reproduced
                # here since do() itself is bypassed.
                cross            = rng.random(n_pairs) < p_cross
                Q[:, ~cross]     = X_pairs[:, ~cross]
                X_rotated[pairs[:, 0]] = Q[0]
                X_rotated[pairs[:, 1]] = Q[1]
            X_varied = np.clip(pm_op._do(problem, X_rotated, random_state=rng), xl, xu)

            theta_offspring = _encode_theta(X_varied, xl, xu)
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
            # IMPORTANT: use the REAL objectives (F_parent/F_offspring) and G
            # here, and call survival.do() (pymoo's public entry point) —
            # NOT the pre-penalised F_pen_* with survival._do() directly.
            # do() first splits feasible/infeasible via CV (derived from G)
            # and only fills remaining slots with infeasible individuals
            # (sorted by constraint violation) — this is the exact
            # "feasibility first" mechanism NSGA-III (pymoo) itself gets for
            # free every generation via its own Survival.do() wrapper.
            # Calling _do() directly on the manually penalised F_pen (as
            # before) bypassed that wrapper entirely, mixing the penalty into
            # the non-dominated sort instead — a real asymmetry that
            # disadvantaged QI-NSGA-III relative to NSGA-III in head-to-head
            # comparisons (confirmed by reading pymoo's Survival.do() /
            # ReferenceDirectionSurvival source: filter_infeasible=True is
            # only applied by the do() wrapper, never by _do()).
            theta_pool  = np.vstack([theta_parent, theta_offspring])
            F_true_pool = np.vstack([F_parent, F_offspring])
            G_pool      = np.vstack([G_parent, G_offspring])
            merged_pop  = Population.new(X=theta_pool, F=F_true_pool, G=G_pool)
            survived    = survival.do(problem, merged_pop, n_survive=pop_size, random_state=rng)
            qpop.theta  = np.clip(np.asarray(survived.get("X"), dtype=float), 0.0, np.pi / 2.0)

            if use_rqpso_rotation or use_pso_rotation:
                survived_F_norm = _normalise_F(
                    np.asarray(survived.get("F"), dtype=float),
                    survival.norm.ideal_point, survival.norm.nadir_point,
                )
                pbest_theta, pbest_scalar = _update_pbest(
                    pbest_theta, pbest_scalar, qpop.theta, survived_F_norm,
                )

            # --- Niche-recentring reset (ablation-only, delta_similar=0.0
            # disables it) -- same positional approximation _migrate below
            # already makes: assoc/F_norm were computed from the PARENT
            # population earlier this generation, applied here against the
            # post-survival qpop.theta. See docs/superpowers/specs/
            # 2026-08-01-qinsga3-niche-recentring-reset-design.md.
            if delta_similar > 0.0:
                reset_mask = _recentring_reset_mask(
                    assoc, qpop.theta, F_norm, ref_dirs, delta_similar,
                )
                n_reset = int(reset_mask.sum())
                if n_reset:
                    qpop.theta[reset_mask] = rng.uniform(
                        0.0, np.pi / 2.0, size=(n_reset, n_genes)
                    )

            if (arch_F_norm is not None
                    and migration_period > 0
                    and gen % migration_period == 0):
                _migrate(
                    qpop, arch_theta_arr, arch_F_norm,
                    assoc, ref_dirs, rng, n_migrate=n_migrate,
                )

            if callback is not None and (gen % 10 == 0 or gen == max_gen - 1):
                callback(gen, F_offspring, G_offspring, off_pareto_idx)

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
        pareto_X, pareto_F, pareto_G = (
            arch_X_arr, arch_F_arr, np.zeros((len(arch_X_arr), n_constr))
        )
    else:
        pareto_X, pareto_F, pareto_G = (
            X_final[final_pareto_idx], F_final[final_pareto_idx], G_final[final_pareto_idx]
        )

    if repair_final_front:
        pareto_F, pareto_G = _repair_pareto_front(pareto_X, sets_, params_)
        # Repair optimises f1 only, per individual -- it can newly dominate
        # another front member (confirmed on a real front: 40/40
        # non-dominated before repair, only 26/40 after). Re-filter to
        # non-dominated so the returned "Pareto front" still is one.
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
        nd_idx = NonDominatedSorting().do(pareto_F)[0]
        if len(nd_idx) < len(pareto_X):
            pareto_X, pareto_F, pareto_G = pareto_X[nd_idx], pareto_F[nd_idx], pareto_G[nd_idx]

    return pareto_X, pareto_F, pareto_G
