"""Component ablation of QI-NSGA-III (answers "the component analysis does not
fully isolate its mechanisms").

Every variant switches off ONE mechanism, with the SAME evaluation budget,
seeds, instance, population and serial execution as NSGA-III:

  full            QI-NSGA-III as published
  no_rotation     quantum rotation off (alpha_max = alpha_min = 0)
  no_migration    archive migration off (migration_period = 0)
  no_arch_guides  archive supplementation of empty-niche guides off
  none            all three off  (≈ NSGA-III operators in theta-space)
  nsga3           pymoo NSGA-III reference

Hypervolume is computed on objectives normalised with the ideal/nadir of the
union of all final fronts of the instance (same reference for every variant),
reference point 1.1 in each objective. Wilcoxon signed-rank tests compare each
variant with "full", paired by seed.

Usage:
    python -m scripts.run_ablation --instance data/instance_100_clients.json --seeds 10 --gen 300
Writes results/ablation_<instance>.csv and prints a summary table.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from pymoo.algorithms.moo.nsga3 import NSGA3                     # noqa: E402
from pymoo.indicators.hv import HV                               # noqa: E402
from pymoo.operators.crossover.sbx import SBX                    # noqa: E402
from pymoo.operators.mutation.pm import PM                       # noqa: E402
from pymoo.optimize import minimize                              # noqa: E402
from pymoo.termination import get_termination                    # noqa: E402
from pymoo.util.ref_dirs import get_reference_directions         # noqa: E402
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting  # noqa: E402
from scipy.stats import wilcoxon                                 # noqa: E402

from models.parameters import load_instance                      # noqa: E402
from solvers.nsga3.decoder import n_genes                        # noqa: E402
from solvers.nsga3.problem import IRPProblem                     # noqa: E402
from solvers.qinsga3.algorithm import run_qinsga3                # noqa: E402
from solvers.qinsga3.main import ALPHA_MAX, ALPHA_MIN, SEEDS     # noqa: E402

VARIANTS = {
    "full":           {},
    "no_rotation":    {"alpha_max": 0.0, "alpha_min": 0.0},
    "no_migration":   {"migration_period": 0},
    "no_arch_guides": {"use_archive_guides": False},
    "none":           {"alpha_max": 0.0, "alpha_min": 0.0, "migration_period": 0,
                       "use_archive_guides": False},
}


def run_variant(name, sets_, params_, ref_dirs, pop, n_gen, seed):
    D = n_genes(sets_)
    if name == "nsga3":
        problem = IRPProblem(sets_, params_)
        algo = NSGA3(pop_size=pop, ref_dirs=ref_dirs, crossover=SBX(prob=0.9, eta=20),
                     mutation=PM(prob=1.0, prob_var=1.0 / D, eta=20))
        res = minimize(problem, algo, get_termination("n_eval", pop * n_gen), seed=seed, verbose=False)
        # same final-front rule as every QI variant: feasible non-dominated set of the
        # last population (pymoo's res.F keeps only one representative per niche)
        F_pop, G_pop = res.pop.get("F"), res.pop.get("G")
        feasible = np.maximum(G_pop, 0).sum(axis=1) <= 1e-9
        if not feasible.any():
            return np.empty((0, 4)), int(res.algorithm.evaluator.n_eval)
        F_feas = F_pop[feasible]
        front = NonDominatedSorting().do(F_feas, only_non_dominated_front=True)
        return np.atleast_2d(F_feas[front]), int(res.algorithm.evaluator.n_eval)
    kw = dict(alpha_max=ALPHA_MAX, alpha_min=ALPHA_MIN)
    kw.update(VARIANTS[name])
    stats = {}
    _, F, G = run_qinsga3(sets_, params_, ref_dirs, pop_size=pop, max_gen=n_gen, seed=seed,
                          max_evals=pop * n_gen, archive_final_front=False, eliminate_duplicates=True,
                          repair_final_front=False, n_workers=1, stats=stats, **kw)
    feasible = np.maximum(G, 0).sum(axis=1) <= 1e-9 if len(G) else np.zeros(0, bool)
    return np.atleast_2d(F[feasible]) if feasible.any() else np.empty((0, 4)), stats["n_evals"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--gen", type=int, default=300)
    ap.add_argument("--pop", type=int, default=200)
    args = ap.parse_args()

    sets_, params_ = load_instance(args.instance)
    ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=8)
    pop = max(args.pop, len(ref_dirs))
    names = list(VARIANTS) + ["nsga3"]
    fronts = {}
    for seed in SEEDS[:args.seeds]:
        for name in names:
            t0 = time.time()
            F, n_evals = run_variant(name, sets_, params_, ref_dirs, pop, args.gen, seed)
            fronts[name, seed] = (F, n_evals, time.time() - t0)
            print(f"seed {seed:5d} {name:15s} front={len(F):3d} evals={n_evals} {time.time() - t0:6.1f}s",
                  flush=True)

    allF = np.vstack([F for F, _, _ in fronts.values() if len(F)])
    ideal, nadir = allF.min(axis=0), allF.max(axis=0)
    scale = np.where(nadir - ideal > 1e-12, nadir - ideal, 1.0)
    hv = HV(ref_point=np.full(4, 1.1))
    rows = []
    for (name, seed), (F, n_evals, sec) in fronts.items():
        value = float(hv((F - ideal) / scale)) if len(F) else 0.0
        rows.append({"variant": name, "seed": seed, "hv": value, "front": len(F),
                     "n_evals": n_evals, "seconds": round(sec, 1)})

    out_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(args.instance))[0]
    path = os.path.join(out_dir, f"ablation_{tag}.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    full = {r["seed"]: r["hv"] for r in rows if r["variant"] == "full"}
    print(f"\n{'variant':15s} {'HV mean':>9s} {'HV std':>8s} {'evals':>7s}  p (vs full)")
    for name in names:
        hv_v = {r["seed"]: r["hv"] for r in rows if r["variant"] == name}
        seeds = sorted(set(full) & set(hv_v))
        vals = np.array([hv_v[s] for s in seeds])
        evals = np.mean([r["n_evals"] for r in rows if r["variant"] == name])
        p = "-"
        if name != "full" and len(seeds) >= 5 and np.any(vals != [full[s] for s in seeds]):
            p = f"{wilcoxon([full[s] for s in seeds], vals).pvalue:.4f}"
        print(f"{name:15s} {vals.mean():9.4f} {vals.std():8.4f} {evals:7.0f}  {p}")
    print(f"\nwritten {os.path.relpath(path, BASE_DIR)}")


if __name__ == "__main__":
    main()
