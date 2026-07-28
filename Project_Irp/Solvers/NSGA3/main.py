"""
NSGA3 — Many-Objective IRP solved with NSGA-III (pymoo).

Usage (standalone):
    python -m NSGA3.main                          # instance_25_clients.json
    python -m NSGA3.main --instance 30
    python -m NSGA3.main --instance 25 --pop 200 --gen 200 --runs 5

Called from app.py via run_nsga3_report().
"""

import argparse
import json
import os
import random as _random
import sys
import time
import threading

import numpy as np

MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(MODULE_DIR))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from pymoo.algorithms.moo.nsga3    import NSGA3
from pymoo.util.ref_dirs           import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm   import PM
from pymoo.operators.sampling.rnd  import FloatRandomSampling
from pymoo.optimize                import minimize
from pymoo.termination             import get_termination

from models.parametres  import load_instance
from .problem           import IRPProblem
from .report            import write_report, generate_and_open
from .report_builder    import _evaluate_pareto, _build_report_data

DEFAULT_REPORT_PATH = os.path.join(MODULE_DIR, "nsga3_report.html")
_CHROM_CACHE_PATH   = os.path.join(MODULE_DIR, "nsga3_chromosomes.json")

# Algorithm hyper-parameters
POP_SIZE       = 200
N_GEN          = 300  # matches QINSGA3.N_GEN — equal search budget for a fair comparison
CROSSOVER_PROB = 0.9
# pm = 1/D (D = number of decision variables) as recommended by Deb & Jain (2014).
# Computed dynamically in run_nsga3() once the instance is loaded.
MUTATION_PROB  = None
N_PARTITIONS   = 8   # Das-Dennis: C(4+8-1,8) = 165 reference points for 4 objectives

# Distinct seeds for up to 20 consecutive runs
SEEDS = [42, 137, 271, 491, 613, 733, 857, 977, 1009, 1123,
         1249, 1373, 1499, 1609, 1733, 1871, 1997, 2113, 2237, 2351]

# Prevent concurrent NSGA-III runs from corrupting numpy's global RNG
_run_lock = threading.Lock()



def render_from_instance(data_path):
    """Re-evaluate cached NSGA-III chromosomes with the current instance JSON.
    Supports both old single-run cache and new multi-run cache formats.
    """
    if not os.path.exists(_CHROM_CACHE_PATH):
        raise FileNotFoundError(
            "No cached chromosomes. Run the algorithm at least once first."
        )
    with open(_CHROM_CACHE_PATH, encoding="utf-8") as f:
        cache = json.load(f)

    sets_, params_ = load_instance(data_path)
    n_genes_expected = len(sets_["clients"]) * len(sets_["T"]) + len(sets_["clients"])

    if "runs" in cache:
        runs_cache = cache["runs"]
    else:
        runs_cache = [{"chromosomes": cache["chromosomes"], "meta_base": cache["meta_base"]}]

    runs_data = []
    for i, run_cache in enumerate(runs_cache):
        pareto_X = np.array(run_cache["chromosomes"])
        if pareto_X.shape[1] != n_genes_expected:
            raise ValueError(
                f"Run {i+1}: cached chromosomes have {pareto_X.shape[1]} genes but the "
                f"current instance requires {n_genes_expected}. Re-run the algorithm."
            )
        runs_data.append(_evaluate_pareto(pareto_X, sets_, params_, run_cache["meta_base"]))

    return _build_report_data(runs_data)


def run_nsga3(data_path=None, pop_size=POP_SIZE, n_gen=N_GEN,
              crossover_prob=CROSSOVER_PROB, mutation_prob=MUTATION_PROB,
              n_runs=1):
    """Run NSGA-III n_runs times with distinct seeds, cache all Pareto fronts, return report data."""
    n_runs = max(1, min(20, int(n_runs)))

    if data_path is None:
        data_path = os.path.join(PROJECT_DIR, "data", "instance_25_clients.json")

    sets_, params_ = load_instance(data_path)
    n_clients      = len(sets_["clients"])

    # pm = 1/D where D = total number of decision variables in the chromosome
    # D = n_clients * n_periods (quantity genes) + n_clients (priority genes)
    if mutation_prob is None:
        n_genes       = n_clients * len(sets_["T"]) + n_clients
        mutation_prob = 1.0 / n_genes
        print(f"[NSGA3] mutation_prob = 1/D = 1/{n_genes} = {mutation_prob:.6f}", flush=True)

    problem  = IRPProblem(sets_, params_)
    ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=N_PARTITIONS)

    effective_pop = max(pop_size, len(ref_dirs))
    if effective_pop != pop_size:
        print(f"[NSGA3] pop_size bumped {pop_size} → {effective_pop} "
              f"(must be >= n_ref_dirs={len(ref_dirs)})", flush=True)

    print(f"[NSGA3] {n_clients} clients | {len(sets_['T'])} periods | "
          f"{len(sets_['M'])} vehicles | {n_clients * len(sets_['T'])} genes | "
          f"pop={effective_pop} gen={n_gen} | runs={n_runs}", flush=True)

    if not _run_lock.acquire(blocking=False):
        raise RuntimeError(
            "[NSGA3] Another run is already in progress. "
            "Wait for it to finish before launching a new one."
        )

    raw_runs = []
    try:
        for run_idx in range(n_runs):
            seed = SEEDS[run_idx % len(SEEDS)]
            print(f"\n[NSGA3] === Run {run_idx + 1}/{n_runs}  seed={seed} ===", flush=True)

            np.random.seed(seed)
            _random.seed(seed)

            algorithm = NSGA3(
                pop_size  = effective_pop,
                ref_dirs  = ref_dirs,
                sampling  = FloatRandomSampling(),
                crossover = SBX(prob=crossover_prob, eta=20),
                mutation  = PM(prob=mutation_prob,   eta=20),
            )

            t_start = time.time()
            result  = minimize(
                problem,
                algorithm,
                get_termination("n_gen", n_gen),
                verbose = True,
                seed    = seed,
            )
            elapsed = time.time() - t_start

            pareto_X = result.X
            if pareto_X is None or len(pareto_X) == 0:
                print(f"[NSGA3] Run {run_idx + 1}: no feasible solutions — skipping.", flush=True)
                continue

            print(f"[NSGA3] Run {run_idx + 1} done in {elapsed:.1f}s | "
                  f"Pareto front: {len(pareto_X)} solutions", flush=True)
            raw_runs.append({"seed": seed, "elapsed": elapsed, "pareto_X": pareto_X})
    finally:
        _run_lock.release()

    if not raw_runs:
        raise RuntimeError(
            "[NSGA3] No feasible solutions found in any run — all solutions violate "
            f"the hard constraint (tau_return > tau_max={params_['tau_max']}). "
            "Check tau_max in the instance JSON or increase pop/gen."
        )

    base_meta = {
        "instance":       f"{n_clients}_clients",
        "pop_size":       effective_pop,
        "n_gen":          n_gen,
        "crossover_prob": crossover_prob,
        "mutation_prob":  mutation_prob,
        "n_runs":         n_runs,
        "n_completed":    len(raw_runs),
    }

    runs_data = []
    for i, raw in enumerate(raw_runs):
        per_run_meta = {
            **base_meta,
            "run_id":    i + 1,
            "seed":      raw["seed"],
            "elapsed_s": round(raw["elapsed"], 1),
        }
        runs_data.append(_evaluate_pareto(raw["pareto_X"], sets_, params_, per_run_meta))

    # Cache all runs (supports multi-run refresh via render_from_instance)
    cache = {
        "n_runs": len(raw_runs),
        "runs": [
            {
                "seed":        raw["seed"],
                "chromosomes": raw["pareto_X"].tolist(),
                "meta_base":   runs_data[i]["meta"],
            }
            for i, raw in enumerate(raw_runs)
        ],
    }
    with open(_CHROM_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)

    return _build_report_data(runs_data)


def run_nsga3_report(output_path=DEFAULT_REPORT_PATH, data_path=None,
                     pop_size=POP_SIZE, n_gen=N_GEN,
                     crossover_prob=CROSSOVER_PROB, mutation_prob=MUTATION_PROB,
                     n_runs=1):
    data = run_nsga3(data_path, pop_size, n_gen, crossover_prob, mutation_prob, n_runs)
    return write_report(data, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSGA-III solver for the many-objective IRP")
    parser.add_argument("--instance", default="25", choices=["3", "5", "15", "25", "30", "40", "100"])
    parser.add_argument("--pop",  type=int,   default=POP_SIZE)
    parser.add_argument("--gen",  type=int,   default=N_GEN)
    parser.add_argument("--cx",   type=float, default=CROSSOVER_PROB)
    parser.add_argument("--mut",  type=float, default=None,
                        help="Mutation probability (default: 1/D, D=number of decision variables)")
    parser.add_argument("--runs", type=int,   default=1)
    args = parser.parse_args()

    dp   = os.path.join(PROJECT_DIR, "data", f"instance_{args.instance}_clients.json")
    out  = os.path.join(MODULE_DIR,  f"nsga3_report_{args.instance}clients.html")
    data = run_nsga3(dp, args.pop, args.gen, args.cx, args.mut, args.runs)
    path = write_report(data, out)
    print(f"[NSGA3] Report → {path}", flush=True)
    generate_and_open(path)
