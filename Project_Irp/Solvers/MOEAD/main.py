"""MOEAD -- Many-Objective IRP solved with MOEA/D (pymoo), constraint-
handled via Deb's feasibility rule (see _constrained_moead.py) and
objective-range-normalized via a custom Tchebycheff decomposition (see
_normalized_decomposition.py) -- both needed because vanilla pymoo MOEAD
cannot run against IRPProblem unmodified. See
docs/superpowers/specs/2026-08-19-moead-integration-design.md for the
full rationale.

Usage (standalone):
    python -m Solvers.MOEAD.main                          # instance_25_clients.json
    python -m Solvers.MOEAD.main --instance 30
    python -m Solvers.MOEAD.main --instance 25 --gen 300 --runs 5

Called from app.py via run_moead_report().
Report format is identical to NSGA3 -- NSGA3/report.py is reused unchanged.
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

from pymoo.util.ref_dirs           import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm   import PM
from pymoo.operators.sampling.rnd  import FloatRandomSampling
from pymoo.optimize                import minimize
from pymoo.termination             import get_termination

from models.parametres            import load_instance
from Solvers.NSGA3.problem        import IRPProblem
from Solvers.NSGA3.report         import write_report, generate_and_open
from Solvers.NSGA3.report_builder import _evaluate_pareto, _build_report_data
from ._constrained_moead          import ConstrainedMOEAD
from ._normalized_decomposition   import NormalizedTchebycheff

DEFAULT_REPORT_PATH = os.path.join(MODULE_DIR, "moead_report.html")
_CHROM_CACHE_PATH   = os.path.join(MODULE_DIR, "moead_chromosomes.json")

N_GEN          = 300  # matches NSGA3/QINSGA3.N_GEN -- equal search budget for a fair comparison
CROSSOVER_PROB = 0.9
MUTATION_PROB  = None  # pm = 1/D, computed dynamically once the instance is loaded
# N_PARTITIONS=8 -> 165 reference directions (das-dennis, M=4), shared with
# NSGA3/QINSGA3's own N_PARTITIONS. Unlike those two, MOEA/D's population
# size is NOT independently settable (pymoo's own MOEAD._setup(): one
# individual per decomposed subproblem) -- this fixes it at 165, not the
# 200 NSGA3/QINSGA3 use. See docs/superpowers/specs/
# 2026-08-19-moead-integration-design.md's population-size discussion for
# why 200 is not reachable for MOEA/D at any Das-Dennis partition count.
N_PARTITIONS = 8

SEEDS = [42, 137, 271, 491, 613, 733, 857, 977, 1009, 1123,
         1249, 1373, 1499, 1609, 1733, 1871, 1997, 2113, 2237, 2351]

_run_lock = threading.Lock()


def render_from_instance(data_path):
    """Re-evaluate cached MOEA/D chromosomes with the current instance JSON."""
    if not os.path.exists(_CHROM_CACHE_PATH):
        raise FileNotFoundError(
            "No cached MOEA/D chromosomes. Run the algorithm at least once first."
        )
    with open(_CHROM_CACHE_PATH, encoding="utf-8") as f:
        cache = json.load(f)

    sets_, params_ = load_instance(data_path)
    n_genes_expected = len(sets_["clients"]) * len(sets_["T"]) + len(sets_["clients"])

    runs_data = []
    for i, run_cache in enumerate(cache.get("runs", [])):
        pareto_X = np.array(run_cache["chromosomes"])
        if pareto_X.shape[1] != n_genes_expected:
            raise ValueError(
                f"Run {i+1}: cached chromosomes have {pareto_X.shape[1]} genes but the "
                f"current instance requires {n_genes_expected}. Re-run the algorithm."
            )
        repair = run_cache["meta_base"].get("repair_final_front", False)
        runs_data.append(_evaluate_pareto(pareto_X, sets_, params_, run_cache["meta_base"], repair=repair))

    return _build_report_data(runs_data)


def run_moead(data_path=None, n_gen=N_GEN,
              crossover_prob=CROSSOVER_PROB, mutation_prob=MUTATION_PROB,
              n_runs=1, repair_final_front: bool = True):
    """Run MOEA/D n_runs times with distinct seeds, cache all Pareto fronts,
    return report data.

    No pop_size parameter (unlike run_nsga3/run_qinsga3_solver): MOEA/D's
    population size is fixed at len(ref_dirs) by pymoo's own MOEAD._setup()
    (one individual per decomposed subproblem) -- it cannot be set
    independently. With the project's shared Das-Dennis N_PARTITIONS=8
    scheme (4 objectives), this is 165, not the 200 NSGA-III/QI-NSGA-III
    use -- see docs/superpowers/specs/2026-08-19-moead-integration-design.md.

    repair_final_front (default True, matches NSGA-III/QI-NSGA-III): see
    run_nsga3's own docstring -- identical meaning here, same shared
    _evaluate_pareto path (current production defaults use_two_opt=False,
    use_delivery_shift=True apply automatically).
    """
    n_runs = max(1, min(20, int(n_runs)))

    if data_path is None:
        data_path = os.path.join(PROJECT_DIR, "data", "instance_25_clients.json")

    sets_, params_ = load_instance(data_path)
    n_clients      = len(sets_["clients"])

    if mutation_prob is None:
        n_genes       = n_clients * len(sets_["T"]) + n_clients
        mutation_prob = 1.0 / n_genes
        print(f"[MOEAD] mutation_prob = 1/D = 1/{n_genes} = {mutation_prob:.6f}", flush=True)

    problem  = IRPProblem(sets_, params_)
    ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=N_PARTITIONS)
    effective_pop = len(ref_dirs)

    print(f"[MOEAD] {n_clients} clients | {len(sets_['T'])} periods | "
          f"{len(sets_['M'])} vehicles | {n_clients * len(sets_['T'])} genes | "
          f"pop={effective_pop} (=len(ref_dirs), not independently settable) "
          f"gen={n_gen} | runs={n_runs}", flush=True)

    if not _run_lock.acquire(blocking=False):
        raise RuntimeError(
            "[MOEAD] Another run is already in progress. "
            "Wait for it to finish before launching a new one."
        )

    raw_runs = []
    try:
        for run_idx in range(n_runs):
            seed = SEEDS[run_idx % len(SEEDS)]
            print(f"\n[MOEAD] === Run {run_idx + 1}/{n_runs}  seed={seed} ===", flush=True)

            np.random.seed(seed)
            _random.seed(seed)

            algorithm = ConstrainedMOEAD(
                ref_dirs       = ref_dirs,
                decomposition  = NormalizedTchebycheff(),
                sampling       = FloatRandomSampling(),
                crossover      = SBX(prob=crossover_prob, eta=20),
                # prob=1.0, prob_var=mutation_prob -- same per-gene-rate
                # convention as Solvers/NSGA3/main.py::run_nsga3 (see its
                # own long comment on prob vs prob_var).
                mutation       = PM(prob=1.0, prob_var=mutation_prob, eta=20),
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
                print(f"[MOEAD] Run {run_idx + 1}: no feasible solutions — skipping.", flush=True)
                continue

            print(f"[MOEAD] Run {run_idx + 1} done in {elapsed:.1f}s | "
                  f"Pareto front: {len(pareto_X)} solutions", flush=True)
            raw_runs.append({"seed": seed, "elapsed": elapsed, "pareto_X": pareto_X})
    finally:
        _run_lock.release()

    if not raw_runs:
        raise RuntimeError(
            "[MOEAD] No feasible solutions found in any run — all solutions violate "
            f"the hard constraint (tau_return > tau_max={params_['tau_max']}). "
            "Check tau_max in the instance JSON or increase gen."
        )

    base_meta = {
        "instance":           f"{n_clients}_clients",
        "pop_size":           effective_pop,
        "n_gen":              n_gen,
        "crossover_prob":     crossover_prob,
        "mutation_prob":      mutation_prob,
        "n_runs":             n_runs,
        "n_completed":        len(raw_runs),
        "algorithm":          "MOEAD",
        "repair_final_front": repair_final_front,
    }

    runs_data = []
    for i, raw in enumerate(raw_runs):
        per_run_meta = {
            **base_meta,
            "run_id":    i + 1,
            "seed":      raw["seed"],
            "elapsed_s": round(raw["elapsed"], 1),
        }
        runs_data.append(_evaluate_pareto(
            raw["pareto_X"], sets_, params_, per_run_meta, repair=repair_final_front,
        ))

    cache = {
        "algorithm": "MOEAD",
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


def run_moead_report(output_path=DEFAULT_REPORT_PATH, data_path=None,
                     n_gen=N_GEN, crossover_prob=CROSSOVER_PROB,
                     mutation_prob=MUTATION_PROB, n_runs=1,
                     repair_final_front: bool = True):
    data = run_moead(
        data_path, n_gen, crossover_prob, mutation_prob, n_runs,
        repair_final_front=repair_final_front,
    )
    return write_report(data, output_path, algo_label="MOEA/D")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MOEA/D solver for the many-objective IRP")
    parser.add_argument("--instance", default="25", choices=["3", "5", "15", "25", "30", "40", "100"])
    parser.add_argument("--gen",  type=int,   default=N_GEN)
    parser.add_argument("--cx",   type=float, default=CROSSOVER_PROB)
    parser.add_argument("--mut",  type=float, default=None,
                        help="Mutation probability (default: 1/D, D=number of decision variables)")
    parser.add_argument("--runs", type=int,   default=1)
    args = parser.parse_args()

    dp   = os.path.join(PROJECT_DIR, "data", f"instance_{args.instance}_clients.json")
    out  = os.path.join(MODULE_DIR,  f"moead_report_{args.instance}clients.html")
    data = run_moead(dp, args.gen, args.cx, args.mut, args.runs)
    path = write_report(data, out, algo_label="MOEA/D")
    print(f"[MOEAD] Report → {path}", flush=True)
    generate_and_open(path)
