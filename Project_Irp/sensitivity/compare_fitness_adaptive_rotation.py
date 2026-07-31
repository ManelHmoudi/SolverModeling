"""Fitness-adaptive rotation-angle test for QINSGA-III.

Context: Kumar, Solanki, Jhariya, Shrivastava & Gupta (2026), "Quantum-inspired
NSGA-II for multi-objective optimization of electric vehicle charging
stations", Scientific Reports, propose an adaptive quantum rotation gate for
their EAH-QNSGA-II algorithm:

    delta_theta_i = eta * tanh((f(x_best) - f(x)) / (|f(x)| + eps))
    eta = 0.05 (Table 3)

This replaces QINSGA-III's current generation-dependent envelope
    alpha(g) = alpha_min + (alpha_max - alpha_min) * (1 - g / max_gen)
with a fitness-gap-dependent one. Their algorithm is single/aggregate-
objective NSGA-II; QINSGA-III is 4-objective and niche-based, so "f(x)" has
no direct equivalent here.

Formula mapping (paper -> QINSGA-III), same discipline as
docs/superpowers/specs/2026-07-24-qinsga3-diversity-preserving-design.md's
"Formula mapping" table -- every deviation justified, nothing invented:

  - "f(x)" (single fitness value) -> per-individual perpendicular distance to
    its niche's reference ray, d_perp(x) -- the same many-objective "how good
    is this solution in its niche" proxy _select_guides() already uses to
    pick a niche's champion (lower = better, mirrors a minimization f(x)).
  - "f(x_best)" -> d_perp of the niche's champion (the guide), i.e. the same
    value _select_guides() used to pick guides_theta. Necessary adaptation:
    the paper's x_best is a single global best; QINSGA-III already tracks a
    *per-niche* best (the guide) as its closest equivalent.
  - eta, eps -> ported as-is (eta=0.05, eps=1e-6 -- paper doesn't give an
    exact eps, "small constant to avoid division by zero" is all Table 3
    specifies).
  - Necessary adaptation the paper's context doesn't need: alpha is clipped
    to >= 0. In QINSGA-III's rotation gate, alpha multiplies
    tanh((theta_guide - theta) / (pi/8)) -- a per-gene term that already
    encodes "move toward the guide's angle". A negative alpha there would
    invert that into "move away from the guide", which has no equivalent
    meaning in the paper's own formula (their Delta_theta directly IS the
    rotation, with no separate "guide angle" target to invert away from).
    Clipping preserves the paper's magnitude idea without inventing a new
    "move away" mechanic QINSGA-III has no other use for.

This is deliberately unlike the already-tried-and-rejected per-niche
"1/5-success-rule" adaptive step (sensitivity/compare_adaptive_rotation.py,
found non-significant, HV 0.161 vs 0.168, p=1.0): that scaled a *decaying*
alpha(g) up/down by a discrete niche-improvement signal. This one is
per-individual (not per-niche), continuous (tanh of a normalised gap, not a
+/-10% step), and eta is FIXED across the whole run (no generation decay at
all) -- matching the paper's own Table 3 setup exactly rather than combining
it with QINSGA-III's existing schedule.

Usage:
    python -m sensitivity.compare_fitness_adaptive_rotation
    python -m sensitivity.compare_fitness_adaptive_rotation --seeds 42 137 271 --gen 300
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(MODULE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.core.population import Population
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from scipy.stats import mannwhitneyu

from models.parametres          import load_instance
from Solvers.NSGA3.decoder       import decode_chromosome, build_routes
from Solvers.NSGA3.evaluator     import compute_f1, compute_f2, compute_f3, compute_f4
from Solvers.NSGA3.metrics       import compute_pareto_metrics
from Solvers.NSGA3.problem       import IRPProblem
from Solvers.QINSGA3.algorithm   import (
    run_qinsga3, _encode_theta, _worker_init, _worker_eval, _penalised_F,
    _normalise_F, _assign_ref_dirs, _select_guides, _supplement_from_archive,
    _migrate, _crowding_trim, _archive_update,
)
from Solvers.QINSGA3.chromosome  import QuantumPopulation

N_PARTITIONS = 8
N_OBJ        = 4
POP_SIZE     = 200

DEFAULT_SEEDS = [42, 137, 271]

# Kumar et al. (2026), Table 3.
ETA = 0.05
EPS = 1e-6

_NSGA3_CACHE = os.path.join(PROJECT_DIR, "Solvers", "NSGA3", "nsga3_chromosomes.json")


def _ref_unit(ref_dirs: np.ndarray) -> np.ndarray:
    ref_norms = np.linalg.norm(ref_dirs, axis=1, keepdims=True)
    return ref_dirs / np.where(ref_norms > 1e-9, ref_norms, 1.0)


def _perp_dist_per_individual(F_norm: np.ndarray, ref_dirs: np.ndarray, assoc: np.ndarray) -> np.ndarray:
    """Perpendicular distance of every individual to its own niche's reference
    ray -- the per-individual "how good is this solution in its niche" proxy,
    same geometry _select_guides() uses to rank niche members."""
    unit = _ref_unit(ref_dirs)
    rays = unit[assoc]
    proj = np.einsum("ij,ij->i", F_norm, rays)
    d2   = np.maximum((F_norm ** 2).sum(axis=1) - proj ** 2, 0.0)
    return np.sqrt(d2)


def _guide_quality(assoc: np.ndarray, pareto_idx: np.ndarray, F_norm: np.ndarray, ref_dirs: np.ndarray) -> np.ndarray:
    """Per-niche champion's perpendicular distance (the guide's own quality),
    broadcast to every population member of that niche -- mirrors
    _select_guides()'s champion pick exactly, but returns the distance value
    instead of theta. Global fallback = Pareto member closest to the origin,
    same convention as _select_guides()."""
    N            = len(assoc)
    unit         = _ref_unit(ref_dirs)
    F_par_n      = F_norm[pareto_idx]
    pareto_assoc = assoc[pareto_idx]

    fb_local  = np.linalg.norm(F_par_n, axis=1).argmin()
    fb_F      = F_par_n[fb_local]
    fb_proj   = fb_F @ unit[pareto_assoc[fb_local]]
    global_fb = float(np.sqrt(max((fb_F ** 2).sum() - fb_proj ** 2, 0.0)))
    quality   = np.full(N, global_fb)

    for rd in np.unique(pareto_assoc):
        same_mask = pareto_assoc == rd
        same_idx  = pareto_idx[same_mask]
        F_same    = F_norm[same_idx]
        proj      = F_same @ unit[rd]
        d_perp2   = np.maximum((F_same ** 2).sum(axis=1) - proj ** 2, 0.0)
        quality[assoc == rd] = np.sqrt(d_perp2.min())

    return quality


def _fitness_adaptive_alpha(assoc: np.ndarray, pareto_idx: np.ndarray, F_norm: np.ndarray, ref_dirs: np.ndarray) -> np.ndarray:
    """[Kumar et al. 2026, Table 3] delta_theta = eta * tanh((q_guide - q_own) / (q_own + eps)),
    q = perpendicular distance to reference ray (lower = better, mirrors a
    minimization f(x)). Clipped to >= 0 -- see module docstring."""
    own_q   = _perp_dist_per_individual(F_norm, ref_dirs, assoc)
    guide_q = _guide_quality(assoc, pareto_idx, F_norm, ref_dirs)
    gap     = (own_q - guide_q) / (own_q + EPS)
    alpha   = ETA * np.tanh(gap)
    return np.clip(alpha, 0.0, None)[:, None]  # (N, 1), broadcasts over genes


def run_qinsga3_fitness_adaptive(
    sets_, params_, ref_dirs,
    pop_size=200, max_gen=300,
    p_cross=0.9, eta_cross=20.0,
    p_mut=None, eta_mut=20.0,
    migration_period=10, n_migrate=10,
    seed=42, rotation_type="tanh", callback=None,
):
    """Identical to production QINSGA3.algorithm.run_qinsga3 (elitist
    survival + X-space SBX/PM, rotation in theta-space), except alpha is
    computed per-individual from the fitness-gap formula above instead of
    QINSGA-III's own generation-dependent envelope."""
    rng     = np.random.default_rng(seed)
    problem = IRPProblem(sets_, params_)

    xl       = np.asarray(problem.xl, dtype=float)
    xu       = np.asarray(problem.xu, dtype=float)
    n_genes  = problem.n_var
    n_constr = problem.n_ieq_constr

    if p_mut is None:
        p_mut = 1.0 / n_genes

    qpop     = QuantumPopulation(pop_size, n_genes, xl, xu, rng=rng, rotation_type=rotation_type)
    sorter   = NonDominatedSorting()
    survival = ReferenceDirectionSurvival(ref_dirs)
    sbx_op   = SBX(prob=p_cross, eta=eta_cross)
    pm_op    = PM(prob=p_mut,    eta=eta_mut)

    arch_X, arch_F, arch_theta = [], [], []
    _MAX_ARCHIVE = 500

    n_workers = min(os.cpu_count() or 1, pop_size)
    chunksize = max(1, pop_size // (2 * n_workers))

    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_worker_init, initargs=(sets_, params_),
    ) as pool:

        def _eval_batch(X):
            results = list(pool.map(_worker_eval, list(X), chunksize=chunksize))
            return (np.array([r[0] for r in results]), np.array([r[1] for r in results]))

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

            F_norm = _normalise_F(F_pen_parent)
            assoc  = _assign_ref_dirs(F_norm, ref_dirs)
            guides_theta = _select_guides(assoc, pareto_idx, F_norm, ref_dirs, qpop.theta)

            arch_theta_arr = None
            arch_F_norm    = None
            if len(arch_X) >= 4:
                arch_theta_arr = np.array(arch_theta)
                arch_F_norm    = _normalise_F(np.array(arch_F))
                pareto_assoc   = assoc[pareto_idx]
                guides_theta   = _supplement_from_archive(
                    guides_theta, assoc, pareto_assoc, arch_theta_arr, arch_F_norm, ref_dirs,
                )

            # --- Fitness-adaptive rotation step [Kumar et al. 2026] --------
            alpha_vec = _fitness_adaptive_alpha(assoc, pareto_idx, F_norm, ref_dirs)

            # --- Quantum step: rotation stays in theta-space -----------------
            qpop.rotate(guides_theta, alpha_vec)
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
            qpop.theta      = theta_offspring
            X_offspring     = qpop.measure()
            F_offspring, G_offspring = _eval_batch(X_offspring)
            F_pen_offspring = _penalised_F(F_offspring, G_offspring)

            off_pareto_idx = sorter.do(F_pen_offspring)[0]
            _archive_update(
                X_offspring[off_pareto_idx], F_offspring[off_pareto_idx], G_offspring[off_pareto_idx],
                theta_offspring[off_pareto_idx], arch_X, arch_F, arch_theta, _MAX_ARCHIVE,
            )

            theta_pool = np.vstack([theta_parent, theta_offspring])
            F_pool     = np.vstack([F_pen_parent, F_pen_offspring])
            merged_pop = Population.new(X=theta_pool, F=F_pool)
            survived   = survival._do(None, merged_pop, pop_size, random_state=rng)
            qpop.theta = np.clip(np.asarray(survived.get("X"), dtype=float), 0.0, np.pi / 2.0)

            if (arch_F_norm is not None and migration_period > 0 and gen % migration_period == 0):
                _migrate(qpop, arch_theta_arr, arch_F_norm, assoc, ref_dirs, rng, n_migrate=n_migrate)

            if callback is not None and (gen % 10 == 0 or gen == max_gen - 1):
                callback(gen, F_offspring, G_offspring, off_pareto_idx)

        X_final = qpop.measure()
        F_final, G_final = _eval_batch(X_final)
        F_pen_final      = _penalised_F(F_final, G_final)
        final_pareto_idx = sorter.do(F_pen_final)[0]
        _archive_update(
            X_final[final_pareto_idx], F_final[final_pareto_idx], G_final[final_pareto_idx],
            qpop.theta[final_pareto_idx], arch_X, arch_F, arch_theta, _MAX_ARCHIVE,
        )

    if arch_X:
        arch_X_arr, arch_F_arr, _ = _crowding_trim(
            np.array(arch_X), np.array(arch_F), np.array(arch_theta), pop_size,
        )
        return arch_X_arr, arch_F_arr, np.zeros((len(arch_X_arr), n_constr))
    return X_final[final_pareto_idx], F_final[final_pareto_idx], G_final[final_pareto_idx]


# ---------------------------------------------------------------------------
# Comparison harness
# ---------------------------------------------------------------------------

def _decode_to_F(chromosomes, sets_, params_) -> np.ndarray:
    F = []
    for chrom in chromosomes:
        quantities, priorities = decode_chromosome(np.array(chrom), sets_)
        route_result = build_routes(quantities, sets_, params_, priorities)
        F.append([
            compute_f1(route_result, sets_, params_),
            compute_f2(route_result, sets_, params_),
            compute_f3(route_result, sets_, params_),
            compute_f4(route_result, sets_, params_),
        ])
    return np.array(F)


def _load_nsga3_reference(sets_, params_, instance: str) -> list[np.ndarray]:
    with open(_NSGA3_CACHE, encoding="utf-8") as f:
        cache = json.load(f)
    expected = f"{instance}_clients"
    cached   = cache["runs"][0]["meta_base"]["instance"]
    if cached != expected:
        raise ValueError(f"NSGA-III cache is for '{cached}', not '{expected}'")
    return [_decode_to_F(r["chromosomes"], sets_, params_) for r in cache["runs"]]


def _stats(vals):
    arr = np.array([v for v in vals if v is not None], dtype=float)
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(arr.mean()), "std": float(arr.std())}


def run_comparison(instance: str, seeds: list[int], max_gen: int, pop_size: int) -> None:
    data_path = os.path.join(PROJECT_DIR, "data", f"instance_{instance}_clients.json")
    sets_, params_ = load_instance(data_path)
    ref_dirs = get_reference_directions("das-dennis", N_OBJ, n_partitions=N_PARTITIONS)
    effective_pop = max(pop_size, len(ref_dirs))

    print("=" * 92)
    print("  FITNESS-ADAPTIVE ROTATION-ANGLE TEST — QINSGA-III (Kumar et al. 2026 vs NSGA-III cache)")
    print("=" * 92)
    print(f"  Instance : {instance} clients | pop={effective_pop} | gen={max_gen} | eta={ETA} | eps={EPS}")
    print(f"  Seeds    : {seeds}")
    print("=" * 92)

    print("\nDecodage du front NSGA-III (cache, reference fixe)...")
    nsga3_F_runs = _load_nsga3_reference(sets_, params_, instance)
    print(f"  NSGA-III : {len(nsga3_F_runs)} runs, "
          f"{sum(len(f) for f in nsga3_F_runs)} solutions")

    raw: dict[str, list] = {"baseline (prod. actuelle)": [], "fitness_adaptive (test)": []}

    print("\n>>> baseline (run_qinsga3, production actuelle, inchange)")
    for seed in seeds:
        print(f"  seed={seed} ... ", end="", flush=True)
        t0 = time.time()
        _, pareto_F, _ = run_qinsga3(
            sets_=sets_, params_=params_, ref_dirs=ref_dirs,
            pop_size=effective_pop, max_gen=max_gen, seed=seed,
        )
        elapsed = round(time.time() - t0, 1)
        raw["baseline (prod. actuelle)"].append((seed, pareto_F, elapsed))
        print(f"front={len(pareto_F) if pareto_F is not None else 0}  time={elapsed}s", flush=True)

    print("\n>>> fitness_adaptive (alpha = eta * tanh(gap), Kumar et al. 2026)")
    for seed in seeds:
        print(f"  seed={seed} ... ", end="", flush=True)
        t0 = time.time()
        _, pareto_F, _ = run_qinsga3_fitness_adaptive(
            sets_=sets_, params_=params_, ref_dirs=ref_dirs,
            pop_size=effective_pop, max_gen=max_gen, seed=seed,
        )
        elapsed = round(time.time() - t0, 1)
        raw["fitness_adaptive (test)"].append((seed, pareto_F, elapsed))
        print(f"front={len(pareto_F) if pareto_F is not None else 0}  time={elapsed}s", flush=True)

    all_F = list(nsga3_F_runs)
    for lbl in raw:
        all_F.extend(F for (_, F, _) in raw[lbl] if F is not None and len(F) > 0)
    all_F_stack  = np.vstack(all_F)
    global_ideal = all_F_stack.min(axis=0)
    global_nadir = all_F_stack.max(axis=0)

    print(f"\nIdeal global partage : {global_ideal}")
    print(f"Nadir global partage  : {global_nadir}")

    groups: dict[str, dict[str, list]] = {}

    nsga3_vals = {"HV": [], "GD": [], "IGD": [], "Spacing": []}
    for F_run in nsga3_F_runs:
        q = compute_pareto_metrics(F_run, global_ideal, global_nadir)
        for k in nsga3_vals:
            nsga3_vals[k].append(q[k])
    groups["NSGA-III (reference)"] = nsga3_vals

    for lbl in ("baseline (prod. actuelle)", "fitness_adaptive (test)"):
        vals = {"HV": [], "GD": [], "IGD": [], "Spacing": [], "elapsed_s": [], "front_size": []}
        for seed, F, elapsed in raw[lbl]:
            if F is None or len(F) == 0:
                continue
            q = compute_pareto_metrics(F, global_ideal, global_nadir)
            for k in ("HV", "GD", "IGD", "Spacing"):
                vals[k].append(q[k])
            vals["elapsed_s"].append(elapsed)
            vals["front_size"].append(len(F))
        groups[lbl] = vals

    print(f"\n{'-'*92}")
    print("  RESULTATS (ideal/nadir global partage NSGA-III + baseline + fitness_adaptive)")
    print(f"{'-'*92}")
    for metric, higher in (("HV", True), ("GD", False), ("IGD", False), ("Spacing", False)):
        arrow = "^" if higher else "v"
        print(f"\n  {metric} ({arrow})")
        for name, vals in groups.items():
            s = _stats(vals[metric])
            print(f"    {name:<28} mean={s['mean']:.6f}  std={s['std']:.6f}")

    print(f"\n{'-'*92}")
    print("  Mann-Whitney U : baseline vs fitness_adaptive")
    print(f"{'-'*92}")
    for metric in ("HV", "GD", "IGD", "Spacing"):
        a, b = groups["baseline (prod. actuelle)"][metric], groups["fitness_adaptive (test)"][metric]
        if len(a) < 2 or len(b) < 2:
            print(f"  {metric:<8} : pas assez de runs valides pour un test")
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        sig = "significatif (p<0.05)" if p < 0.05 else "non significatif"
        print(f"  {metric:<8} U={u:.1f}  p={p:.6f}  -> {sig}")

    print(f"\n{'-'*92}")
    print("  Mann-Whitney U : fitness_adaptive vs NSGA-III (l'objectif final)")
    print(f"{'-'*92}")
    for metric in ("HV", "GD", "IGD", "Spacing"):
        a, b = groups["fitness_adaptive (test)"][metric], groups["NSGA-III (reference)"][metric]
        if len(a) < 2 or len(b) < 2:
            print(f"  {metric:<8} : pas assez de runs valides pour un test")
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        sig = "significatif (p<0.05)" if p < 0.05 else "non significatif"
        print(f"  {metric:<8} U={u:.1f}  p={p:.6f}  -> {sig}")

    print(f"\n{'-'*92}")
    print("  Temps moyen par run")
    print(f"{'-'*92}")
    for lbl in ("baseline (prod. actuelle)", "fitness_adaptive (test)"):
        s = _stats(groups[lbl]["elapsed_s"])
        print(f"    {lbl:<28} mean={s['mean']:.1f}s")

    print(f"\n{'='*92}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test de rotation adaptative par fitness (Kumar et al. 2026) — QINSGA-III")
    parser.add_argument("--instance", default="100",
                        choices=["3", "5", "15", "25", "30", "40", "100"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--gen",   type=int, default=300)
    parser.add_argument("--pop",   type=int, default=POP_SIZE)
    args = parser.parse_args()
    run_comparison(args.instance, args.seeds, args.gen, args.pop)
