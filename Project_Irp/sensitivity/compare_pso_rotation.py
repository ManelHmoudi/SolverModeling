"""PSO momentum rotation test for QINSGA-III (9th remedy attempt, after
RQPSO dual-attractor rotation -- the 7th -- confirmed that adding a second
global attractor without any counter-force only compounds the contraction
problem; see Solvers/IRP_results_summary.md's "Remede C" section).

Context: every remedy tried so far (rotation frequency/target/magnitude
tuning, external reset, measurement noise, dual-attractor RQPSO, ring
guides) recomputes the rotation step FROM SCRATCH every generation -- there
is no memory of past movement. This 9th remedy adds a genuinely new
ingredient: a VELOCITY that PERSISTS across generations (momentum/inertia),
so an individual that has built up speed in one direction does not
immediately snap toward a new guide -- it changes direction gradually. In
principle this could resist the deterministic per-generation contraction
already diagnosed as the root cause, without the disruptive side effects of
random resets (remedies A/B) or the compounding effect of a second
memoryless attractor (remedy C).

Grounded in a single article read in full: Li, Xu, Liu & Li (2008),
"Quantum Multi-objective Evolutionary Algorithm with Particle Swarm
Optimization Method", ICNC 2008 -- notably the SAME paper already cited as
the source of QINSGA3's alpha_max/alpha_min values (Vmax=0.10*pi,
Vmin=0.001*pi match exactly), just never used with its own momentum
mechanism before. Full formula and the two documented adaptations (velocity
clamp decay schedule reusing alpha_max/alpha_min; inertia weight clipped to
[0,2] for stability, since the paper's own w_i = D(i)/N + m(i)/N has no
clip and was only validated in a regime that apparently never produced
negative values) are in the module note above
Solvers/QINSGA3/algorithm.py::_pso_rotate.

Pre-campaign instrumented check (60 generations, real 100-client instance)
confirmed the mechanism is not degenerate: adaptive inertia w varies
meaningfully across individuals every single generation checked (range
0.0-1.95, median 0.83), and mean |velocity| stays substantial throughout
(median 0.112 rad) rather than collapsing to zero or saturating.

Already wired into production run_qinsga3 as use_pso_rotation (default
False) -- no separate reimplementation needed.

Usage:
    python -m sensitivity.compare_pso_rotation
    python -m sensitivity.compare_pso_rotation --seeds 42 137 271
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(MODULE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np
from pymoo.util.ref_dirs import get_reference_directions
from scipy.stats import mannwhitneyu

from models.parametres         import load_instance
from Solvers.NSGA3.decoder      import decode_chromosome, build_routes
from Solvers.NSGA3.evaluator    import compute_f1, compute_f2, compute_f3, compute_f4
from Solvers.NSGA3.metrics      import compute_pareto_metrics
from Solvers.NSGA3.problem      import IRPProblem
from Solvers.QINSGA3.algorithm  import run_qinsga3

N_PARTITIONS = 8
N_OBJ        = 4
POP_SIZE     = 200
DEFAULT_SEEDS = [42, 137, 271]

_NSGA3_CACHE = os.path.join(PROJECT_DIR, "Solvers", "NSGA3", "nsga3_chromosomes.json")


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


def run_comparison(instance: str, seeds: list[int], max_gen: int, pop_size: int) -> None:
    data_path = os.path.join(PROJECT_DIR, "data", f"instance_{instance}_clients.json")
    sets_, params_ = load_instance(data_path)
    problem_ref = IRPProblem(sets_, params_)
    xl = np.asarray(problem_ref.xl, dtype=float)
    xu = np.asarray(problem_ref.xu, dtype=float)
    ref_dirs = get_reference_directions("das-dennis", N_OBJ, n_partitions=N_PARTITIONS)
    effective_pop = max(pop_size, len(ref_dirs))

    print("=" * 92)
    print("  PSO MOMENTUM ROTATION — QINSGA-III (baseline tanh vs NSGA-III cache)")
    print("=" * 92)
    print(f"  Instance : {instance} clients | pop={effective_pop} | gen={max_gen}")
    print(f"  Seeds    : {seeds}")
    print("=" * 92)

    print("\nDecodage du front NSGA-III (cache, reference fixe)...")
    nsga3_F_runs = _load_nsga3_reference(sets_, params_, instance)
    print(f"  NSGA-III : {len(nsga3_F_runs)} runs, "
          f"{sum(len(f) for f in nsga3_F_runs)} solutions")

    baseline_lbl = "baseline (tanh)"
    test_lbl     = "PSO momentum (test)"
    raw:     dict[str, list] = {baseline_lbl: [], test_lbl: []}
    chrom_X: dict[str, list] = {baseline_lbl: [], test_lbl: []}

    print(f"\n>>> {baseline_lbl} (prod. actuelle)")
    for seed in seeds:
        print(f"  seed={seed} ... ", end="", flush=True)
        t0 = time.time()
        X, pareto_F, _ = run_qinsga3(
            sets_=sets_, params_=params_, ref_dirs=ref_dirs,
            pop_size=effective_pop, max_gen=max_gen, seed=seed,
        )
        elapsed = round(time.time() - t0, 1)
        raw[baseline_lbl].append((seed, pareto_F, elapsed))
        chrom_X[baseline_lbl].extend(X.tolist())
        print(f"front={len(pareto_F) if pareto_F is not None else 0}  time={elapsed}s", flush=True)

    print(f"\n>>> {test_lbl}")
    for seed in seeds:
        print(f"  seed={seed} ... ", end="", flush=True)
        t0 = time.time()
        X, pareto_F, _ = run_qinsga3(
            sets_=sets_, params_=params_, ref_dirs=ref_dirs,
            pop_size=effective_pop, max_gen=max_gen, seed=seed, use_pso_rotation=True,
        )
        elapsed = round(time.time() - t0, 1)
        raw[test_lbl].append((seed, pareto_F, elapsed))
        chrom_X[test_lbl].extend(X.tolist())
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

    for lbl in (baseline_lbl, test_lbl):
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
    print("  RESULTATS (ideal/nadir global partage NSGA-III + baseline + test)")
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
    for lbl in (baseline_lbl, test_lbl):
        d = _chrom_diversity(chrom_X[lbl], xl, xu)
        print(f"    {lbl:<32} {d:.6f}")

    print(f"\n{'-'*92}")
    print(f"  Mann-Whitney U : baseline vs {test_lbl}")
    print(f"{'-'*92}")
    for metric in ("HV", "GD", "IGD", "Spacing"):
        a, b = groups[baseline_lbl][metric], groups[test_lbl][metric]
        if len(a) < 2 or len(b) < 2:
            print(f"  {metric:<8} : pas assez de runs valides pour un test")
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        sig = "significatif (p<0.05)" if p < 0.05 else "non significatif"
        print(f"  {metric:<8} U={u:.1f}  p={p:.6f}  -> {sig}")

    print(f"\n{'-'*92}")
    print(f"  Mann-Whitney U : {test_lbl} vs NSGA-III (l'objectif final)")
    print(f"{'-'*92}")
    for metric in ("HV", "GD", "IGD", "Spacing"):
        a, b = groups[test_lbl][metric], groups["NSGA-III (reference)"][metric]
        if len(a) < 2 or len(b) < 2:
            print(f"  {metric:<8} : pas assez de runs valides pour un test")
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        sig = "significatif (p<0.05)" if p < 0.05 else "non significatif"
        print(f"  {metric:<8} U={u:.1f}  p={p:.6f}  -> {sig}")

    print(f"\n{'-'*92}")
    print("  Temps moyen par run")
    print(f"{'-'*92}")
    for lbl in (baseline_lbl, test_lbl):
        s = _stats(groups[lbl]["elapsed_s"])
        print(f"    {lbl:<32} mean={s['mean']:.1f}s")

    print(f"\n{'='*92}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test de la rotation avec momentum PSO — QINSGA-III")
    parser.add_argument("--instance", default="100",
                        choices=["3", "5", "15", "25", "30", "40", "100"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--gen", type=int, default=300)
    parser.add_argument("--pop", type=int, default=POP_SIZE)
    args = parser.parse_args()
    run_comparison(args.instance, args.seeds, args.gen, args.pop)
