"""Partial, in-search route repair test for QINSGA-III (archive-only, every
generation) -- the untested middle ground between the two existing repair
variants.

Context: Remedy G (`Solvers/QINSGA3/repair.py`, `use_route_repair`) proved
that 2-opt-repairing the decoded routes before scoring is the ONLY mechanism
in this project's whole campaign (A-I) that produced a real quality gain --
but repairing EVERY individual EVERY generation costs ~15x baseline runtime,
disqualifying it in practice. Its adopted practical variant
(`repair_final_front`, production default) repairs only the returned front,
ONCE, after the search loop ends -- cheap, but the repaired signal never
reaches guide selection or the archive DURING the search.

This script tests a third point: repair only the individuals that actually
enter the archive each generation (the current generation's own rank-0
front, both parent and offspring -- NOT the whole population). Measured on a
real 100-generation run (`Solvers/QINSGA3/algorithm.py`'s own `off_pareto_idx`
via the `callback` hook): this front is typically 10-20 individuals out of a
population of 200 (5-10%), not the 41-65 the FINAL accumulated archive front
reaches by generation 300 -- an order of magnitude cheaper than
`use_route_repair`'s full-population repair.

Mechanism: wherever the production loop calls `_archive_update` with raw
(X, F, G) for the current generation's Pareto-front candidates, this variant
first repairs those candidates' F/G via `_evaluate_with_repair` (same
Baldwinian 2-opt repair remedy G already uses) before they enter the
archive. Nothing else changes: `pareto_idx`/`off_pareto_idx` selection,
guide selection, survival, SBX/PM, rotation, and the chromosomes themselves
all stay driven by the RAW (unrepaired) fitness, exactly like production --
matching this project's established Baldwinian convention (repair never
changes which individuals are "current best", only the fitness assigned to
archive entries). The only place this can change downstream behaviour is
`_supplement_from_archive` (already in production, unmodified) reading a
now-more-accurate `arch_F` when filling in niche guides that lack a
population-native champion -- so a repaired archive can only make guides
MORE reliable, never touches the rest of the search.

Both baseline and test keep `repair_final_front=True` (production default)
at the end, so the final front is repaired identically in both arms -- the
only isolated variable is whether archive-feeding candidates were ALSO
repaired throughout the search, not just at the very end.

Usage:
    python -m sensitivity.compare_archive_repair
    python -m sensitivity.compare_archive_repair --seeds 42 137 271 --gen 300
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
    _migrate, _crowding_trim, _archive_update, _worker_eval_repaired,
    _repair_pareto_front,
)
from Solvers.QINSGA3.chromosome  import QuantumPopulation

N_PARTITIONS = 8
N_OBJ        = 4
POP_SIZE     = 200

DEFAULT_SEEDS = [42, 137, 271]

_NSGA3_CACHE = os.path.join(PROJECT_DIR, "Solvers", "NSGA3", "nsga3_chromosomes.json")


def _repair_candidates(X: np.ndarray, pool: ProcessPoolExecutor, n_workers: int) -> tuple[np.ndarray, np.ndarray]:
    """Repair each candidate's decoded route (Baldwinian, same mechanism as
    Remedy G) and return its repaired (F, G) -- X itself is never touched.

    Uses the SAME worker pool as the main evaluation batches, via
    `_worker_eval_repaired` (already defined in algorithm.py for
    `use_route_repair`'s worker path) -- parallel across n_workers processes.
    The first version of this script called `_evaluate_with_repair` in a
    plain sequential Python loop in the main process, which turned out to be
    the dominant cost (measured ~2.8x-4.8x baseline instead of the ~1.7x
    the front-size-based estimate predicted) -- repair itself was already
    using the optimised delta-cost 2-opt from repair.py, the loop just never
    used the idle pool sitting right there."""
    if len(X) == 0:
        return np.empty((0, 4)), np.empty((0, 0))
    chunksize = max(1, len(X) // (2 * n_workers))
    results = list(pool.map(_worker_eval_repaired, list(X), chunksize=chunksize))
    return np.array([r[0] for r in results]), np.array([r[1] for r in results])


def run_qinsga3_archive_repair(
    sets_, params_, ref_dirs,
    pop_size=200, max_gen=300,
    alpha_max=0.10 * np.pi, alpha_min=0.001 * np.pi,
    p_cross=0.9, eta_cross=20.0,
    p_mut=None, eta_mut=20.0,
    migration_period=10, n_migrate=10,
    repair_final_front=True,
    seed=42, rotation_type="tanh", callback=None,
):
    """Identical to production Solvers/QINSGA3/algorithm.py::run_qinsga3 in
    every respect -- same search loop, same rank-0 selection, guide
    selection, survival, SBX/PM, rotation, all driven by RAW fitness --
    except every candidate handed to `_archive_update` (this generation's
    own Pareto front, parent AND offspring) is first repaired via
    `_evaluate_with_repair`, so the archive (and anything reading it, i.e.
    `_supplement_from_archive`'s niche-guide fallback) sees repaired F/G
    instead of the raw greedy-decoder values."""
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
            F_arch, G_arch = _repair_candidates(X_parent[pareto_idx], pool, n_workers)
            _archive_update(
                X_parent[pareto_idx], F_arch, G_arch,
                theta_parent[pareto_idx], arch_X, arch_F, arch_theta, _MAX_ARCHIVE,
            )

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
                    guides_theta, assoc, pareto_assoc, arch_theta_arr, arch_F_norm, ref_dirs,
                )

            alpha = alpha_min + (alpha_max - alpha_min) * (1.0 - gen / max_gen)

            qpop.rotate(guides_theta, alpha)
            X_rotated = qpop.measure()

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
            F_arch_off, G_arch_off = _repair_candidates(X_offspring[off_pareto_idx], pool, n_workers)
            _archive_update(
                X_offspring[off_pareto_idx], F_arch_off, G_arch_off,
                theta_offspring[off_pareto_idx], arch_X, arch_F, arch_theta, _MAX_ARCHIVE,
            )

            theta_pool  = np.vstack([theta_parent, theta_offspring])
            F_true_pool = np.vstack([F_parent, F_offspring])
            G_pool      = np.vstack([G_parent, G_offspring])
            merged_pop  = Population.new(X=theta_pool, F=F_true_pool, G=G_pool)
            survived    = survival.do(problem, merged_pop, n_survive=pop_size, random_state=rng)
            qpop.theta  = np.clip(np.asarray(survived.get("X"), dtype=float), 0.0, np.pi / 2.0)

            if (arch_F_norm is not None and migration_period > 0 and gen % migration_period == 0):
                _migrate(qpop, arch_theta_arr, arch_F_norm, assoc, ref_dirs, rng, n_migrate=n_migrate)

            if callback is not None and (gen % 10 == 0 or gen == max_gen - 1):
                callback(gen, F_offspring, G_offspring, off_pareto_idx)

        X_final = qpop.measure()
        F_final, G_final = _eval_batch(X_final)
        F_pen_final      = _penalised_F(F_final, G_final)
        final_pareto_idx = sorter.do(F_pen_final)[0]
        F_arch_final, G_arch_final = _repair_candidates(X_final[final_pareto_idx], pool, n_workers)
        _archive_update(
            X_final[final_pareto_idx], F_arch_final, G_arch_final,
            qpop.theta[final_pareto_idx], arch_X, arch_F, arch_theta, _MAX_ARCHIVE,
        )

    if arch_X:
        arch_X_arr, arch_F_arr, _ = _crowding_trim(
            np.array(arch_X), np.array(arch_F), np.array(arch_theta), pop_size,
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

    return pareto_X, pareto_F, pareto_G


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
    print("  REPARATION PARTIELLE EN COURS DE RECHERCHE (archive seule) — QINSGA-III")
    print("=" * 92)
    print(f"  Instance : {instance} clients | pop={effective_pop} | gen={max_gen}")
    print(f"  Seeds    : {seeds}")
    print("=" * 92)

    print("\nDecodage du front NSGA-III (cache, reference fixe)...")
    nsga3_F_runs = _load_nsga3_reference(sets_, params_, instance)
    print(f"  NSGA-III : {len(nsga3_F_runs)} runs, "
          f"{sum(len(f) for f in nsga3_F_runs)} solutions")

    raw: dict[str, list] = {"baseline (repair_final_front seul)": [], "archive repair (test)": []}

    print("\n>>> baseline (repair_final_front seul, prod. actuelle)")
    for seed in seeds:
        print(f"  seed={seed} ... ", end="", flush=True)
        t0 = time.time()
        X, pareto_F, _ = run_qinsga3(
            sets_=sets_, params_=params_, ref_dirs=ref_dirs,
            pop_size=effective_pop, max_gen=max_gen, seed=seed,
        )
        elapsed = round(time.time() - t0, 1)
        raw["baseline (repair_final_front seul)"].append((seed, pareto_F, elapsed))
        print(f"front={len(pareto_F) if pareto_F is not None else 0}  time={elapsed}s", flush=True)

    print("\n>>> archive repair (test, reparation du front courant a chaque generation)")
    for seed in seeds:
        print(f"  seed={seed} ... ", end="", flush=True)
        t0 = time.time()
        X, pareto_F, _ = run_qinsga3_archive_repair(
            sets_=sets_, params_=params_, ref_dirs=ref_dirs,
            pop_size=effective_pop, max_gen=max_gen, seed=seed,
        )
        elapsed = round(time.time() - t0, 1)
        raw["archive repair (test)"].append((seed, pareto_F, elapsed))
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

    for lbl in ("baseline (repair_final_front seul)", "archive repair (test)"):
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
    print("  RESULTATS (ideal/nadir global partage NSGA-III + baseline + archive repair)")
    print(f"{'-'*92}")
    for metric, higher in (("HV", True), ("GD", False), ("IGD", False), ("Spacing", False)):
        arrow = "^" if higher else "v"
        print(f"\n  {metric} ({arrow})")
        for name, vals in groups.items():
            s = _stats(vals[metric])
            print(f"    {name:<32} mean={s['mean']:.6f}  std={s['std']:.6f}")

    print(f"\n{'-'*92}")
    print(f"  Mann-Whitney U : baseline vs archive repair")
    print(f"{'-'*92}")
    for metric in ("HV", "GD", "IGD", "Spacing"):
        a, b = groups["baseline (repair_final_front seul)"][metric], groups["archive repair (test)"][metric]
        if len(a) < 2 or len(b) < 2:
            print(f"  {metric:<8} : pas assez de runs valides pour un test")
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        sig = "significatif (p<0.05)" if p < 0.05 else "non significatif"
        print(f"  {metric:<8} U={u:.1f}  p={p:.6f}  -> {sig}")

    print(f"\n{'-'*92}")
    print(f"  Mann-Whitney U : archive repair vs NSGA-III (l'objectif final)")
    print(f"{'-'*92}")
    for metric in ("HV", "GD", "IGD", "Spacing"):
        a, b = groups["archive repair (test)"][metric], groups["NSGA-III (reference)"][metric]
        if len(a) < 2 or len(b) < 2:
            print(f"  {metric:<8} : pas assez de runs valides pour un test")
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        sig = "significatif (p<0.05)" if p < 0.05 else "non significatif"
        print(f"  {metric:<8} U={u:.1f}  p={p:.6f}  -> {sig}")

    print(f"\n{'-'*92}")
    print("  Temps moyen par run")
    print(f"{'-'*92}")
    for lbl in ("baseline (repair_final_front seul)", "archive repair (test)"):
        s = _stats(groups[lbl]["elapsed_s"])
        print(f"    {lbl:<32} mean={s['mean']:.1f}s")

    print(f"\n{'='*92}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test de reparation partielle de l'archive en cours de recherche — QINSGA-III")
    parser.add_argument("--instance", default="100",
                        choices=["3", "5", "15", "25", "30", "40", "100"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--gen",   type=int, default=300)
    parser.add_argument("--pop",   type=int, default=POP_SIZE)
    args = parser.parse_args()
    run_comparison(args.instance, args.seeds, args.gen, args.pop)
