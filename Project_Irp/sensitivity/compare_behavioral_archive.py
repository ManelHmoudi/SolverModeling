"""Behavioral (route-structural) archive trim test for QINSGA-III.

Context: the production final-front trim (`Solvers/QINSGA3/algorithm.py::
_crowding_trim`) picks the max_size most-spread archive members using ONLY
crowding distance in OBJECTIVE (F) space -- X/theta are carried along as
payload, never used for the diversity decision. The user's hypothesis: since
f1-f4 are scalar sums over arcs/flows (`Solvers/NSGA3/evaluator.py`), two
structurally very different route sets can score near-identically in F,
while two nearly-identical route sets can occupy "spread" positions in F --
so F-only trimming may keep behaviorally redundant solutions and discard
structurally distinct ones, purely by accident of objective-space geometry.

Prior diagnostic context (`Solvers/IRP_results_summary.md`'s H3): the
POPULATION-wide decoded-route diversity (Jaccard distance on
(period,vehicle,arc) sets, `sensitivity/test_route_diversity.py`) is only
mildly lower for QI-NSGA-III than NSGA-III (ratio 0.846) -- nowhere near the
13x chromosome-diversity gap. That refuted H3 at the population level. This
script asks a narrower, different question: even if population-wide route
diversity isn't badly collapsed, does the ARCHIVE's F-only trim (which
reduces to at most pop_size entries for the final report) throw away
structurally distinct solutions in favour of F-space spread alone?

Unlike remedies A-H (all of which changed search DYNAMICS, and all
regressed quality when they increased diversity), this only changes which
of the ALREADY non-dominated final solutions get reported -- the search
loop itself is untouched. Lower risk: this can only change WHICH equally
Pareto-valid solutions are kept, not how well the search converges. Applied
only to the final trim (`repair_final_front`'s minimal-risk pattern), not to
the live archive used for niche guides/migration during search.

Mechanism: combine two per-candidate spread scores when trimming the final
archive to max_size:
  - F-crowding (existing `_crowding_distance`, normalised, inf capped).
  - Structural crowding: mean Jaccard distance (reusing `_route_arcset`/
    `_jaccard_distance` from `sensitivity/test_route_diversity.py`,
    unmodified) from each candidate to every other candidate in the pool
    (all-pairs -- the final archive is small, no sampling needed).
Combined score = w_structural * structural_norm + (1 - w_structural) * f_norm,
keep the max_size highest. See docs/superpowers/specs/
2026-08-06-qinsga3-partial-rotation-by-stability-design.md's sibling design
note for how this fits alongside the partial-rotation remedies (H).

Reports the usual HV/GD/IGD/Spacing (expected to barely move -- same
non-dominated set, different representatives) PLUS the actual target
metric: mean pairwise Jaccard route-diversity of the reported final front
(baseline vs test).

Calibration note (found before running the full campaign): the production
final trim only engages when the non-dominated archive exceeds `pop_size`
(200) -- but full-run archives typically converge to 41-65 members by
generation 300 (continuous dominance filtering shrinks it well below 200
over a 300-generation run), so a trim at `pop_size` is almost always a
no-op and would show zero difference regardless of the criterion used. This
script instead trims BOTH baseline and test down to a smaller `report_size`
(default 30, well below the typical 41-65 unconstrained front) -- reframing
the question as "which N solutions to hand to a decision-maker" (a real
practical need: nobody reviews 60 Pareto solutions by hand) rather than
"which non-dominated solutions survive", and actually exercising the
criterion being tested. Baseline uses `_crowding_trim` (production, F-only)
at `report_size`; test uses `_crowding_trim_behavioral` at the same
`report_size` -- apples to apples.

Usage:
    python -m sensitivity.compare_behavioral_archive
    python -m sensitivity.compare_behavioral_archive --seeds 42 137 271 --gen 300 --w_structural 0.5 --report_size 30
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
    _migrate, _crowding_distance, _crowding_trim, _archive_update,
)
from Solvers.QINSGA3.chromosome  import QuantumPopulation
from sensitivity.test_route_diversity import _route_arcset, _jaccard_distance

N_PARTITIONS = 8
N_OBJ        = 4
POP_SIZE     = 200

DEFAULT_SEEDS = [42, 137, 271]
DEFAULT_W     = 0.5
DEFAULT_REPORT_SIZE = 30

_NSGA3_CACHE = os.path.join(PROJECT_DIR, "Solvers", "NSGA3", "nsga3_chromosomes.json")


def _decode_arcsets(X: np.ndarray, sets_: dict, params_: dict) -> list:
    out = []
    for chrom in X:
        quantities, priorities = decode_chromosome(np.array(chrom), sets_)
        rr = build_routes(quantities, sets_, params_, priorities)
        out.append(_route_arcset(rr))
    return out


def _structural_crowding(arcsets: list) -> np.ndarray:
    """Per candidate, mean Jaccard distance to every other candidate in the
    pool -- the route-structure analogue of F-space crowding distance."""
    n = len(arcsets)
    if n <= 1:
        return np.zeros(n)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = _jaccard_distance(arcsets[i], arcsets[j])
            d[i, j] = d[j, i] = dist
    return d.mean(axis=1)


def _crowding_trim_behavioral(
    X: np.ndarray, F: np.ndarray, theta: np.ndarray,
    sets_: dict, params_: dict, max_size: int, w_structural: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Like Solvers/QINSGA3/algorithm.py::_crowding_trim, but the spread
    score combines F-crowding with route-structural (Jaccard) crowding
    instead of using F-crowding alone."""
    if len(X) <= max_size:
        return X, F, theta

    f_cd = _crowding_distance(_normalise_F(F))
    finite = f_cd[np.isfinite(f_cd)]
    cap = (finite.max() if len(finite) else 1.0) * 2.0
    f_cd_capped = np.where(np.isinf(f_cd), cap, f_cd)
    f_norm = f_cd_capped / (f_cd_capped.max() + 1e-12)

    arcsets  = _decode_arcsets(X, sets_, params_)
    struct   = _structural_crowding(arcsets)
    struct_norm = struct / (struct.max() + 1e-12) if struct.max() > 0 else struct

    combined = w_structural * struct_norm + (1.0 - w_structural) * f_norm
    keep = np.argsort(combined)[-max_size:]
    return X[keep], F[keep], theta[keep]


def run_qinsga3_behavioral_archive(
    sets_, params_, ref_dirs,
    pop_size=200, max_gen=300,
    alpha_max=0.10 * np.pi, alpha_min=0.001 * np.pi,
    p_cross=0.9, eta_cross=20.0,
    p_mut=None, eta_mut=20.0,
    migration_period=10, n_migrate=10,
    w_structural=0.5, report_size=DEFAULT_REPORT_SIZE,
    seed=42, rotation_type="tanh", callback=None,
):
    """Identical to production Solvers/QINSGA3/algorithm.py::run_qinsga3 in
    every respect -- same search loop, same live archive feeding niche
    guides/migration (unaffected: it still trims to `pop_size` internally
    via the unmodified `_archive_update`/`_crowding_trim`) -- except the
    FINAL trim (only, once, at the end) uses `_crowding_trim_behavioral`
    at `report_size` instead of `_crowding_trim` at `pop_size`."""
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
        arch_X_arr, arch_F_arr, _ = _crowding_trim_behavioral(
            np.array(arch_X), np.array(arch_F), np.array(arch_theta),
            sets_, params_, report_size, w_structural,
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


def _front_route_diversity(X: np.ndarray, sets_: dict, params_: dict) -> float:
    """Mean ALL-PAIRS Jaccard route distance of a single reported front --
    the target metric this remedy tries to increase. No sampling: final
    fronts here are small (tens of solutions)."""
    arcsets = _decode_arcsets(X, sets_, params_)
    n = len(arcsets)
    if n < 2:
        return float("nan")
    dists = [
        _jaccard_distance(arcsets[i], arcsets[j])
        for i in range(n) for j in range(i + 1, n)
    ]
    return float(np.mean(dists))


def run_comparison(instance: str, seeds: list[int], max_gen: int, pop_size: int,
                    w_structural: float, report_size: int) -> None:
    data_path = os.path.join(PROJECT_DIR, "data", f"instance_{instance}_clients.json")
    sets_, params_ = load_instance(data_path)
    problem_ref = IRPProblem(sets_, params_)
    ref_dirs = get_reference_directions("das-dennis", N_OBJ, n_partitions=N_PARTITIONS)
    effective_pop = max(pop_size, len(ref_dirs))

    print("=" * 92)
    print(f"  ARCHIVE COMPORTEMENTALE (w_structural={w_structural}, report_size={report_size}) — QINSGA-III")
    print("=" * 92)
    print(f"  Instance : {instance} clients | pop={effective_pop} | gen={max_gen}")
    print(f"  Seeds    : {seeds}")
    print("=" * 92)

    print("\nDecodage du front NSGA-III (cache, reference fixe)...")
    nsga3_F_runs = _load_nsga3_reference(sets_, params_, instance)
    print(f"  NSGA-III : {len(nsga3_F_runs)} runs, "
          f"{sum(len(f) for f in nsga3_F_runs)} solutions")

    raw: dict[str, list] = {"baseline (F-crowding seul)": [], f"w_structural={w_structural} (test)": []}
    route_div: dict[str, list] = {"baseline (F-crowding seul)": [], f"w_structural={w_structural} (test)": []}

    print(f"\n>>> baseline (F-crowding seul, trim a report_size={report_size})")
    for seed in seeds:
        print(f"  seed={seed} ... ", end="", flush=True)
        t0 = time.time()
        X_full, F_full, _ = run_qinsga3(
            sets_=sets_, params_=params_, ref_dirs=ref_dirs,
            pop_size=effective_pop, max_gen=max_gen, seed=seed,
        )
        # Same report_size cap as the test arm, but F-crowding only (production
        # criterion) -- apples-to-apples comparison at the size that actually
        # engages the trim (see module docstring's calibration note).
        X, pareto_F, _ = _crowding_trim(X_full, F_full, X_full, report_size)
        elapsed = round(time.time() - t0, 1)
        raw["baseline (F-crowding seul)"].append((seed, pareto_F, elapsed))
        rd = _front_route_diversity(X, sets_, params_)
        route_div["baseline (F-crowding seul)"].append(rd)
        print(f"front_full={len(F_full)}  front_report={len(pareto_F)}  route_div={rd:.4f}  time={elapsed}s", flush=True)

    print(f"\n>>> w_structural={w_structural} (test, trim a report_size={report_size})")
    for seed in seeds:
        print(f"  seed={seed} ... ", end="", flush=True)
        t0 = time.time()
        X, pareto_F, _ = run_qinsga3_behavioral_archive(
            sets_=sets_, params_=params_, ref_dirs=ref_dirs,
            pop_size=effective_pop, max_gen=max_gen, seed=seed,
            w_structural=w_structural, report_size=report_size,
        )
        elapsed = round(time.time() - t0, 1)
        raw[f"w_structural={w_structural} (test)"].append((seed, pareto_F, elapsed))
        rd = _front_route_diversity(X, sets_, params_)
        route_div[f"w_structural={w_structural} (test)"].append(rd)
        print(f"front_report={len(pareto_F)}  route_div={rd:.4f}  time={elapsed}s", flush=True)

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

    for lbl in ("baseline (F-crowding seul)", f"w_structural={w_structural} (test)"):
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
    print("  RESULTATS (ideal/nadir global partage NSGA-III + baseline + archive comportementale)")
    print(f"{'-'*92}")
    for metric, higher in (("HV", True), ("GD", False), ("IGD", False), ("Spacing", False)):
        arrow = "^" if higher else "v"
        print(f"\n  {metric} ({arrow})")
        for name, vals in groups.items():
            s = _stats(vals[metric])
            print(f"    {name:<32} mean={s['mean']:.6f}  std={s['std']:.6f}")

    print(f"\n{'-'*92}")
    print("  Diversite de tournees du front rapporte  (Jaccard, all-pairs par run) -- METRIQUE CIBLE")
    print(f"{'-'*92}")
    for lbl in ("baseline (F-crowding seul)", f"w_structural={w_structural} (test)"):
        s = _stats(route_div[lbl])
        print(f"    {lbl:<32} mean={s['mean']:.6f}  std={s['std']:.6f}")

    print(f"\n{'-'*92}")
    print(f"  Mann-Whitney U : baseline vs w_structural={w_structural}")
    print(f"{'-'*92}")
    for metric in ("HV", "GD", "IGD", "Spacing"):
        a, b = groups["baseline (F-crowding seul)"][metric], groups[f"w_structural={w_structural} (test)"][metric]
        if len(a) < 2 or len(b) < 2:
            print(f"  {metric:<8} : pas assez de runs valides pour un test")
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        sig = "significatif (p<0.05)" if p < 0.05 else "non significatif"
        print(f"  {metric:<8} U={u:.1f}  p={p:.6f}  -> {sig}")
    a, b = route_div["baseline (F-crowding seul)"], route_div[f"w_structural={w_structural} (test)"]
    if len(a) >= 2 and len(b) >= 2:
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        sig = "significatif (p<0.05)" if p < 0.05 else "non significatif"
        print(f"  {'route_div':<8} U={u:.1f}  p={p:.6f}  -> {sig}")

    print(f"\n{'-'*92}")
    print(f"  Mann-Whitney U : w_structural={w_structural} vs NSGA-III (l'objectif final)")
    print(f"{'-'*92}")
    for metric in ("HV", "GD", "IGD", "Spacing"):
        a, b = groups[f"w_structural={w_structural} (test)"][metric], groups["NSGA-III (reference)"][metric]
        if len(a) < 2 or len(b) < 2:
            print(f"  {metric:<8} : pas assez de runs valides pour un test")
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        sig = "significatif (p<0.05)" if p < 0.05 else "non significatif"
        print(f"  {metric:<8} U={u:.1f}  p={p:.6f}  -> {sig}")

    print(f"\n{'-'*92}")
    print("  Temps moyen par run")
    print(f"{'-'*92}")
    for lbl in ("baseline (F-crowding seul)", f"w_structural={w_structural} (test)"):
        s = _stats(groups[lbl]["elapsed_s"])
        print(f"    {lbl:<32} mean={s['mean']:.1f}s")

    print(f"\n{'='*92}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test d'archive comportementale (diversite structurelle des tournees) — QINSGA-III")
    parser.add_argument("--instance", default="100",
                        choices=["3", "5", "15", "25", "30", "40", "100"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--gen",   type=int, default=300)
    parser.add_argument("--pop",   type=int, default=POP_SIZE)
    parser.add_argument("--w_structural", type=float, default=DEFAULT_W)
    parser.add_argument("--report_size", type=int, default=DEFAULT_REPORT_SIZE)
    args = parser.parse_args()
    run_comparison(args.instance, args.seeds, args.gen, args.pop, args.w_structural, args.report_size)
