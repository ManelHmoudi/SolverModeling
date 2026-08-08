"""QI-NSGA-III vs NSGA-III on SMALLER IRP instances, run fresh (no cache).

Context: every other comparison in this project's campaign uses the 100-
client instance, where QI-NSGA-III loses to NSGA-III -- diagnosed as decoder
fragility (`_nearest_neighbour`'s greedy, irrevocable choices cascade into
disproportionate route changes, worse as routes get longer/more complex).
On the continuous DTLZ/MaF benchmarks (no combinatorial decoder at all),
QI-NSGA-III wins 17/25 cases. The natural, honest question in between: does
QI-NSGA-III's disadvantage on the real IRP shrink on SMALLER instances
(shorter routes, presumably less butterfly-effect amplification), or is it
present even there? This is a characterisation question (where is each
algorithm competitive), not an attempt to cherry-pick a favourable instance
-- results are reported whichever way they go.

Deliberately does NOT touch Solvers/NSGA3/nsga3_chromosomes.json (the shared
100-client "gold standard" cache every other script in this project reads)
-- `Solvers/NSGA3/main.py::run_nsga3` OVERWRITES that file as a side effect,
which would corrupt every other comparison relying on it. This script runs
NSGA-III fresh via pymoo directly (same config as main.py: SBX/PM eta=20,
das-dennis ref dirs, pop/gen matched to QINSGA3) and holds results only in
memory -- no cache read or write for either algorithm.

Usage:
    python -m sensitivity.compare_small_instances --instance 15 --seeds 42 137 271
    python -m sensitivity.compare_small_instances --instance 25 --seeds 42 137 271 --gen 300
"""
from __future__ import annotations

import argparse
import os
import sys
import time

MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(MODULE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np
from pymoo.algorithms.moo.nsga3    import NSGA3
from pymoo.util.ref_dirs           import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm   import PM
from pymoo.operators.sampling.rnd  import FloatRandomSampling
from pymoo.optimize                import minimize
from pymoo.termination             import get_termination
from scipy.stats import mannwhitneyu

from models.parametres          import load_instance
from Solvers.NSGA3.decoder       import decode_chromosome, build_routes
from Solvers.NSGA3.evaluator     import compute_f1, compute_f2, compute_f3, compute_f4
from Solvers.NSGA3.metrics       import compute_pareto_metrics
from Solvers.NSGA3.problem       import IRPProblem
from Solvers.QINSGA3.algorithm   import run_qinsga3

N_PARTITIONS = 8
N_OBJ        = 4
POP_SIZE     = 200

DEFAULT_SEEDS = [42, 137, 271]


def _decode_to_F(X: np.ndarray, sets_: dict, params_: dict) -> np.ndarray:
    F = []
    for chrom in X:
        quantities, priorities = decode_chromosome(np.array(chrom), sets_)
        route_result = build_routes(quantities, sets_, params_, priorities)
        F.append([
            compute_f1(route_result, sets_, params_),
            compute_f2(route_result, sets_, params_),
            compute_f3(route_result, sets_, params_),
            compute_f4(route_result, sets_, params_),
        ])
    return np.array(F)


def _run_nsga3_fresh(problem, ref_dirs, pop_size, max_gen, seed):
    n_genes = problem.n_var
    algorithm = NSGA3(
        pop_size  = pop_size,
        ref_dirs  = ref_dirs,
        sampling  = FloatRandomSampling(),
        crossover = SBX(prob=0.9, eta=20),
        mutation  = PM(prob=1.0 / n_genes, eta=20),
    )
    result = minimize(problem, algorithm, get_termination("n_gen", max_gen),
                       verbose=False, seed=seed)
    return result.X if result.X is not None else np.empty((0, n_genes))


def _stats(vals):
    arr = np.array([v for v in vals if v is not None], dtype=float)
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(arr.mean()), "std": float(arr.std())}


def run_comparison(instance: str, seeds: list[int], max_gen: int, pop_size: int) -> None:
    data_path = os.path.join(PROJECT_DIR, "data", f"instance_{instance}_clients.json")
    sets_, params_ = load_instance(data_path)
    problem  = IRPProblem(sets_, params_)
    ref_dirs = get_reference_directions("das-dennis", N_OBJ, n_partitions=N_PARTITIONS)
    effective_pop = max(pop_size, len(ref_dirs))

    print("=" * 92)
    print(f"  QI-NSGA-III vs NSGA-III SUR PETITE INSTANCE ({instance} clients, runs frais, pas de cache)")
    print("=" * 92)
    print(f"  n_var={problem.n_var} | pop={effective_pop} | gen={max_gen} | seeds={seeds}")
    print("=" * 92)

    raw: dict[str, list] = {"NSGA-III": [], "QI-NSGA-III": []}

    print("\n>>> NSGA-III (frais, pymoo direct, aucun cache touche)")
    for seed in seeds:
        print(f"  seed={seed} ... ", end="", flush=True)
        t0 = time.time()
        X = _run_nsga3_fresh(problem, ref_dirs, effective_pop, max_gen, seed)
        F = _decode_to_F(X, sets_, params_) if len(X) else np.empty((0, N_OBJ))
        elapsed = round(time.time() - t0, 1)
        raw["NSGA-III"].append((seed, F, elapsed))
        print(f"front={len(F)}  time={elapsed}s", flush=True)

    print("\n>>> QI-NSGA-III (frais, repair_final_front=True, prod. actuelle)")
    for seed in seeds:
        print(f"  seed={seed} ... ", end="", flush=True)
        t0 = time.time()
        _, F, _ = run_qinsga3(
            sets_=sets_, params_=params_, ref_dirs=ref_dirs,
            pop_size=effective_pop, max_gen=max_gen, seed=seed,
        )
        elapsed = round(time.time() - t0, 1)
        raw["QI-NSGA-III"].append((seed, F, elapsed))
        print(f"front={len(F) if F is not None else 0}  time={elapsed}s", flush=True)

    all_F = []
    for lbl in raw:
        all_F.extend(F for (_, F, _) in raw[lbl] if F is not None and len(F) > 0)
    all_F_stack  = np.vstack(all_F)
    global_ideal = all_F_stack.min(axis=0)
    global_nadir = all_F_stack.max(axis=0)

    print(f"\nIdeal global partage : {global_ideal}")
    print(f"Nadir global partage  : {global_nadir}")

    groups: dict[str, dict[str, list]] = {}
    for lbl in ("NSGA-III", "QI-NSGA-III"):
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
    print(f"  RESULTATS ({instance} clients, ideal/nadir global partage)")
    print(f"{'-'*92}")
    for metric, higher in (("HV", True), ("GD", False), ("IGD", False), ("Spacing", False)):
        arrow = "^" if higher else "v"
        print(f"\n  {metric} ({arrow})")
        for name, vals in groups.items():
            s = _stats(vals[metric])
            print(f"    {name:<14} mean={s['mean']:.6f}  std={s['std']:.6f}")

    print(f"\n{'-'*92}")
    print(f"  Mann-Whitney U : NSGA-III vs QI-NSGA-III  ({instance} clients)")
    print(f"{'-'*92}")
    for metric in ("HV", "GD", "IGD", "Spacing"):
        a, b = groups["NSGA-III"][metric], groups["QI-NSGA-III"][metric]
        if len(a) < 2 or len(b) < 2:
            print(f"  {metric:<8} : pas assez de runs valides pour un test")
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        sig = "significatif (p<0.05)" if p < 0.05 else "non significatif"
        print(f"  {metric:<8} U={u:.1f}  p={p:.6f}  -> {sig}")

    print(f"\n{'-'*92}")
    print("  Temps moyen par run")
    print(f"{'-'*92}")
    for lbl in ("NSGA-III", "QI-NSGA-III"):
        s = _stats(groups[lbl]["elapsed_s"])
        print(f"    {lbl:<14} mean={s['mean']:.1f}s")

    print(f"\n{'='*92}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QI-NSGA-III vs NSGA-III sur une petite instance, runs frais")
    parser.add_argument("--instance", default="15",
                        choices=["3", "5", "15", "25", "30", "40", "100"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--gen",   type=int, default=300)
    parser.add_argument("--pop",   type=int, default=POP_SIZE)
    args = parser.parse_args()
    run_comparison(args.instance, args.seeds, args.gen, args.pop)
