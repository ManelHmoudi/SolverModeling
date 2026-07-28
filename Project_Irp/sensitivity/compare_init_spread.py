"""Initial-population spread test for QINSGA-III.

Hypothesis: QuantumPopulation.__init__ (Solvers/QINSGA3/chromosome.py) starts
EVERY individual at theta = pi/4 +- 0.05. Since the decode is
x = xl + cos^2(theta) * (xu - xl) and cos^2(pi/4) = 0.5 EXACTLY, every gene of
every individual starts at the MIDPOINT of its [xl, xu] range, with only
~noise_scale=0.02 (~2%) of extra spread. NSGA-III (pymoo) instead uses
FloatRandomSampling(): every gene is drawn ~ Uniform(xl, xu) independently,
covering the whole decision box from generation 0. On a 600-variable problem
(100 clients) this looked like a plausible handicap for QI-NSGA-III.

Result: FALSIFIED, and reversed — the concentrated init outperforms a
dispersed one on this instance (see git history for the full A/B numbers),
likely because it lands closer to feasibility on a heavily constrained
problem. Kept as documentation of a ruled-out avenue.

This script does NOT touch chromosome.py or algorithm.py. It defines a
QuantumPopulation subclass whose initial theta is drawn so the DECODED x for
generation 0 is distributed ~ Uniform(xl, xu) exactly like NSGA-III's
sampling (p = cos^2(theta) ~ U(0,1) <=> theta = arccos(sqrt(p))), then runs
the unmodified run_qinsga3 loop by temporarily monkeypatching the
QuantumPopulation reference Solvers.QINSGA3.algorithm uses internally.

Usage:
    python -m sensitivity.compare_init_spread
    python -m sensitivity.compare_init_spread --seeds 42 137 271 --gen 300
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

from models.parametres          import load_instance
from Solvers.NSGA3.decoder       import decode_chromosome, build_routes
from Solvers.NSGA3.evaluator     import compute_f1, compute_f2, compute_f3, compute_f4
from Solvers.NSGA3.metrics       import compute_pareto_metrics
from Solvers.QINSGA3.chromosome  import QuantumPopulation
import Solvers.QINSGA3.algorithm as qalgo

N_PARTITIONS = 8
N_OBJ        = 4
POP_SIZE     = 200

DEFAULT_SEEDS = [42, 137, 271]

_NSGA3_CACHE = os.path.join(PROJECT_DIR, "Solvers", "NSGA3", "nsga3_chromosomes.json")


class BroadInitQuantumPopulation(QuantumPopulation):
    """QuantumPopulation whose initial theta decodes to x ~ Uniform(xl, xu)
    per gene (matching pymoo's FloatRandomSampling spread) instead of the
    default theta = pi/4 +- 0.05 (x collapsed to the midpoint of each gene's
    range). Nothing else changes: same rotate/crossover/mutate/measure.
    """

    def __init__(self, pop_size, n_genes, xl, xu, rng=None,
                 rotation_type="tanh", noise_scale=0.02):
        super().__init__(pop_size, n_genes, xl, xu, rng=rng,
                          rotation_type=rotation_type, noise_scale=noise_scale)
        p = self.rng.uniform(0.0, 1.0, size=(pop_size, n_genes))
        self.theta = np.clip(np.arccos(np.sqrt(p)), 0.0, np.pi / 2.0)


def _run_qinsga3_broad_init(**kwargs):
    """Run the UNMODIFIED QINSGA3.algorithm.run_qinsga3 loop, but with the
    working population's theta initialised broadly (see class above) instead
    of concentrated at pi/4. Monkeypatches the QuantumPopulation symbol
    QINSGA3.algorithm uses internally for the duration of the call only.
    """
    original = qalgo.QuantumPopulation
    qalgo.QuantumPopulation = BroadInitQuantumPopulation
    try:
        return qalgo.run_qinsga3(**kwargs)
    finally:
        qalgo.QuantumPopulation = original


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
    print("  INITIAL-POPULATION SPREAD TEST — QINSGA-III (vs NSGA-III cache)")
    print("=" * 92)
    print(f"  Instance : {instance} clients | pop={effective_pop} | gen={max_gen}")
    print(f"  Seeds    : {seeds}")
    print("=" * 92)

    print("\nDecodage du front NSGA-III (cache, reference fixe)...")
    nsga3_F_runs = _load_nsga3_reference(sets_, params_, instance)
    print(f"  NSGA-III : {len(nsga3_F_runs)} runs, "
          f"{sum(len(f) for f in nsga3_F_runs)} solutions")

    raw: dict[str, list] = {"concentre (actuel)": [], "disperse (test)": []}

    print("\n>>> init concentree (actuel, theta=pi/4+-0.05, x au milieu de la plage)")
    for seed in seeds:
        print(f"  seed={seed} ... ", end="", flush=True)
        t0 = time.time()
        _, pareto_F, _ = qalgo.run_qinsga3(
            sets_=sets_, params_=params_, ref_dirs=ref_dirs,
            pop_size=effective_pop, max_gen=max_gen, seed=seed,
        )
        elapsed = round(time.time() - t0, 1)
        raw["concentre (actuel)"].append((seed, pareto_F, elapsed))
        print(f"front={len(pareto_F) if pareto_F is not None else 0}  time={elapsed}s", flush=True)

    print("\n>>> init dispersee (test, x ~ Uniform(xl,xu) comme NSGA-III)")
    for seed in seeds:
        print(f"  seed={seed} ... ", end="", flush=True)
        t0 = time.time()
        _, pareto_F, _ = _run_qinsga3_broad_init(
            sets_=sets_, params_=params_, ref_dirs=ref_dirs,
            pop_size=effective_pop, max_gen=max_gen, seed=seed,
        )
        elapsed = round(time.time() - t0, 1)
        raw["disperse (test)"].append((seed, pareto_F, elapsed))
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

    for lbl in ("concentre (actuel)", "disperse (test)"):
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
    print("  RESULTATS (ideal/nadir global partage NSGA-III + les 2 inits)")
    print(f"{'-'*92}")
    for metric, higher in (("HV", True), ("GD", False), ("IGD", False), ("Spacing", False)):
        arrow = "^" if higher else "v"
        print(f"\n  {metric} ({arrow})")
        for name, vals in groups.items():
            s = _stats(vals[metric])
            print(f"    {name:<22} mean={s['mean']:.6f}  std={s['std']:.6f}")

    print(f"\n{'-'*92}")
    print("  Mann-Whitney U : concentre vs disperse")
    print(f"{'-'*92}")
    for metric in ("HV", "GD", "IGD", "Spacing"):
        a, b = groups["concentre (actuel)"][metric], groups["disperse (test)"][metric]
        if len(a) < 2 or len(b) < 2:
            print(f"  {metric:<8} : pas assez de runs valides pour un test")
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        sig = "significatif (p<0.05)" if p < 0.05 else "non significatif"
        print(f"  {metric:<8} U={u:.1f}  p={p:.6f}  -> {sig}")

    print(f"\n{'-'*92}")
    print("  Temps moyen par run")
    print(f"{'-'*92}")
    for lbl in ("concentre (actuel)", "disperse (test)"):
        s = _stats(groups[lbl]["elapsed_s"])
        print(f"    {lbl:<22} mean={s['mean']:.1f}s")

    print(f"\n{'='*92}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test de dispersion de la population initiale QINSGA-III")
    parser.add_argument("--instance", default="100",
                        choices=["3", "5", "15", "25", "30", "40", "100"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--gen",   type=int, default=300)
    parser.add_argument("--pop",   type=int, default=POP_SIZE)
    args = parser.parse_args()
    run_comparison(args.instance, args.seeds, args.gen, args.pop)
