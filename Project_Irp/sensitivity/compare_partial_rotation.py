"""Stability-targeted partial rotation test for QINSGA-III.

Context: `sensitivity/compare_rotation_prob.py` already tested masking
rotation PER INDIVIDUAL, at random (Bernoulli per individual, independent
every generation) -- non-significant, chromosome diversity essentially
unchanged (0.003984 -> 0.003727, see Solvers/IRP_results_summary.md's
"Premieres tentatives" table). This script tests a different axis: masking
PER GENE, per individual, selected by a stability criterion instead of at
random.

Fix tested here: each generation, for every individual, only the `frac`
(default 0.15) fraction of genes whose theta is already CLOSEST to the
niche's archive elites (lowest _elite_rms_distance -- reused unmodified from
Solvers/QINSGA3/algorithm.py) receive this generation's rotation update; the
rest keep their pre-rotation theta value, shaped only by SBX/PM afterward
(same "un-rotated this generation" semantics as compare_rotation_prob.py's
un-rotated individuals, just at gene granularity with a targeted rather than
random selection). Ablation-only -- see docs/superpowers/specs/
2026-08-06-qinsga3-partial-rotation-by-stability-design.md for the full
design and why this is worth checking despite rotation_prob's null result.

--select least_stable inverts the criterion: rotate the genes FURTHEST from
the elites (the ones the rotation gate would correct the most) instead of
the closest ones. Added after --select most_stable (the default) showed
diversity x5.1 but quality regressed on HV/GD/IGD -- the hypothesis being
that "most_stable" protects the genes that need no help and abandons the
genes that most need correcting to unguided SBX/PM, i.e. it targets rotation
where it does the least good instead of the most.

Usage:
    python -m sensitivity.compare_partial_rotation
    python -m sensitivity.compare_partial_rotation --seeds 42 137 271 --gen 300 --frac 0.15 --select least_stable
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
    _migrate, _crowding_trim, _archive_update, _elite_rms_distance,
)
from Solvers.QINSGA3.chromosome  import QuantumPopulation

N_PARTITIONS = 8
N_OBJ        = 4
POP_SIZE     = 200

DEFAULT_SEEDS = [42, 137, 271]
DEFAULT_FRAC  = 0.15

_NSGA3_CACHE = os.path.join(PROJECT_DIR, "Solvers", "NSGA3", "nsga3_chromosomes.json")


def _partial_rotation_mask(elite_rms: np.ndarray, frac: float, select: str = "most_stable") -> np.ndarray:
    """Per individual (row), True for the `frac` fraction of genes selected
    by `select`:
      "most_stable"  -- LOWEST elite_rms (already closest to the elites).
      "least_stable" -- HIGHEST elite_rms (furthest from the elites, i.e.
                        the genes the rotation gate would correct the most
                        if left unmasked)."""
    pop_size, n_genes = elite_rms.shape
    k = max(1, int(round(frac * n_genes)))
    if select == "most_stable":
        idx = np.argpartition(elite_rms, k - 1, axis=1)[:, :k]
    elif select == "least_stable":
        idx = np.argpartition(elite_rms, -k, axis=1)[:, -k:]
    else:
        raise ValueError(f"select must be 'most_stable' or 'least_stable', got '{select}'")
    mask = np.zeros_like(elite_rms, dtype=bool)
    np.put_along_axis(mask, idx, True, axis=1)
    return mask


def run_qinsga3_partial_rotation(
    sets_, params_, ref_dirs,
    pop_size=200, max_gen=300,
    alpha_max=0.10 * np.pi, alpha_min=0.001 * np.pi,
    p_cross=0.9, eta_cross=20.0,
    p_mut=None, eta_mut=20.0,
    migration_period=10, n_migrate=10,
    frac=0.15, select="most_stable",
    seed=42, rotation_type="tanh", callback=None,
):
    """Identical to production Solvers/QINSGA3/algorithm.py::run_qinsga3
    (elitist survival via survival.do(), X-space SBX/PM, shared ideal/nadir
    normalisation via survival.norm), except only the `frac` fraction of
    genes selected by `select` (per-gene elite_rms, "most_stable" = lowest /
    closest to elites, "least_stable" = highest / furthest from elites) are
    rotated each generation -- the rest keep their pre-rotation theta,
    shaped only by SBX/PM like NSGA-III."""
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

            if survival.norm.nadir_point is None:
                F_norm = _normalise_F(F_pen_parent)
            else:
                F_norm = _normalise_F(F_pen_parent, survival.norm.ideal_point, survival.norm.nadir_point)
            assoc  = _assign_ref_dirs(F_norm, ref_dirs)
            guides_theta = _select_guides(assoc, pareto_idx, F_norm, ref_dirs, qpop.theta)

            arch_theta_arr = None
            arch_F_norm    = None
            arch_assoc_arr = None
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
                arch_assoc_arr = _assign_ref_dirs(arch_F_norm, ref_dirs)

            alpha = alpha_min + (alpha_max - alpha_min) * (1.0 - gen / max_gen)

            # --- Stability-targeted partial rotation: only the frac of genes
            # (per individual) selected by `select` get this generation's
            # rotation update ---
            elite_rms = _elite_rms_distance(
                qpop.theta, assoc, guides_theta, arch_theta_arr, arch_assoc_arr,
            )
            qpop.rotate(guides_theta, alpha)
            if frac < 1.0:
                mask       = _partial_rotation_mask(elite_rms, frac, select)
                qpop.theta = np.where(mask, qpop.theta, theta_parent)
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
            _archive_update(
                X_offspring[off_pareto_idx], F_offspring[off_pareto_idx], G_offspring[off_pareto_idx],
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


def _chrom_diversity(X_list, xl, xu):
    X = np.array(X_list, dtype=float)
    denom = np.where(xu - xl > 1e-9, xu - xl, 1.0)
    P = (X - xl) / denom
    return float(P.var(axis=0).mean())


def run_comparison(instance: str, seeds: list[int], max_gen: int, pop_size: int, frac: float, select: str) -> None:
    data_path = os.path.join(PROJECT_DIR, "data", f"instance_{instance}_clients.json")
    sets_, params_ = load_instance(data_path)
    problem_ref = IRPProblem(sets_, params_)
    xl = np.asarray(problem_ref.xl, dtype=float)
    xu = np.asarray(problem_ref.xu, dtype=float)
    ref_dirs = get_reference_directions("das-dennis", N_OBJ, n_partitions=N_PARTITIONS)
    effective_pop = max(pop_size, len(ref_dirs))

    print("=" * 92)
    print(f"  ROTATION PARTIELLE PAR STABILITE (frac={frac}, select={select}) — QINSGA-III (baseline vs NSGA-III cache)")
    print("=" * 92)
    print(f"  Instance : {instance} clients | pop={effective_pop} | gen={max_gen}")
    print(f"  Seeds    : {seeds}")
    print("=" * 92)

    print("\nDecodage du front NSGA-III (cache, reference fixe)...")
    nsga3_F_runs = _load_nsga3_reference(sets_, params_, instance)
    print(f"  NSGA-III : {len(nsga3_F_runs)} runs, "
          f"{sum(len(f) for f in nsga3_F_runs)} solutions")

    raw: dict[str, list] = {"baseline (frac=1.0)": [], f"frac={frac} (test)": []}
    chrom_X: dict[str, list] = {"baseline (frac=1.0)": [], f"frac={frac} (test)": []}

    print("\n>>> baseline (frac=1.0, prod. actuelle)")
    for seed in seeds:
        print(f"  seed={seed} ... ", end="", flush=True)
        t0 = time.time()
        X, pareto_F, _ = run_qinsga3(
            sets_=sets_, params_=params_, ref_dirs=ref_dirs,
            pop_size=effective_pop, max_gen=max_gen, seed=seed,
        )
        elapsed = round(time.time() - t0, 1)
        raw["baseline (frac=1.0)"].append((seed, pareto_F, elapsed))
        chrom_X["baseline (frac=1.0)"].extend(X.tolist())
        print(f"front={len(pareto_F) if pareto_F is not None else 0}  time={elapsed}s", flush=True)

    print(f"\n>>> frac={frac} (test)")
    for seed in seeds:
        print(f"  seed={seed} ... ", end="", flush=True)
        t0 = time.time()
        X, pareto_F, _ = run_qinsga3_partial_rotation(
            sets_=sets_, params_=params_, ref_dirs=ref_dirs,
            pop_size=effective_pop, max_gen=max_gen, seed=seed, frac=frac, select=select,
        )
        elapsed = round(time.time() - t0, 1)
        raw[f"frac={frac} (test)"].append((seed, pareto_F, elapsed))
        chrom_X[f"frac={frac} (test)"].extend(X.tolist())
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

    for lbl in ("baseline (frac=1.0)", f"frac={frac} (test)"):
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
    print("  RESULTATS (ideal/nadir global partage NSGA-III + baseline + rotation partielle)")
    print(f"{'-'*92}")
    for metric, higher in (("HV", True), ("GD", False), ("IGD", False), ("Spacing", False)):
        arrow = "^" if higher else "v"
        print(f"\n  {metric} ({arrow})")
        for name, vals in groups.items():
            s = _stats(vals[metric])
            print(f"    {name:<32} mean={s['mean']:.6f}  std={s['std']:.6f}")

    print(f"\n{'-'*92}")
    print("  Diversite chromosome  Var(X_norm)  (pooled sur tous les runs testes)")
    print(f"{'-'*92}")
    for lbl in ("baseline (frac=1.0)", f"frac={frac} (test)"):
        d = _chrom_diversity(chrom_X[lbl], xl, xu)
        print(f"    {lbl:<32} {d:.6f}")

    print(f"\n{'-'*92}")
    print(f"  Mann-Whitney U : baseline vs frac={frac}")
    print(f"{'-'*92}")
    for metric in ("HV", "GD", "IGD", "Spacing"):
        a, b = groups["baseline (frac=1.0)"][metric], groups[f"frac={frac} (test)"][metric]
        if len(a) < 2 or len(b) < 2:
            print(f"  {metric:<8} : pas assez de runs valides pour un test")
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        sig = "significatif (p<0.05)" if p < 0.05 else "non significatif"
        print(f"  {metric:<8} U={u:.1f}  p={p:.6f}  -> {sig}")

    print(f"\n{'-'*92}")
    print(f"  Mann-Whitney U : frac={frac} vs NSGA-III (l'objectif final)")
    print(f"{'-'*92}")
    for metric in ("HV", "GD", "IGD", "Spacing"):
        a, b = groups[f"frac={frac} (test)"][metric], groups["NSGA-III (reference)"][metric]
        if len(a) < 2 or len(b) < 2:
            print(f"  {metric:<8} : pas assez de runs valides pour un test")
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        sig = "significatif (p<0.05)" if p < 0.05 else "non significatif"
        print(f"  {metric:<8} U={u:.1f}  p={p:.6f}  -> {sig}")

    print(f"\n{'-'*92}")
    print("  Temps moyen par run")
    print(f"{'-'*92}")
    for lbl in ("baseline (frac=1.0)", f"frac={frac} (test)"):
        s = _stats(groups[lbl]["elapsed_s"])
        print(f"    {lbl:<32} mean={s['mean']:.1f}s")

    print(f"\n{'='*92}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test de rotation partielle par stabilite — QINSGA-III")
    parser.add_argument("--instance", default="100",
                        choices=["3", "5", "15", "25", "30", "40", "100"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--gen",   type=int, default=300)
    parser.add_argument("--pop",   type=int, default=POP_SIZE)
    parser.add_argument("--frac",  type=float, default=DEFAULT_FRAC)
    parser.add_argument("--select", choices=["most_stable", "least_stable"], default="most_stable")
    args = parser.parse_args()
    run_comparison(args.instance, args.seeds, args.gen, args.pop, args.frac, args.select)
