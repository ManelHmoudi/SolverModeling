"""
NSGA3 — Many-Objective IRP solved with NSGA-III (pymoo).

Usage (standalone):
    python -m NSGA3.main                     # instance_25_clients.json
    python -m NSGA3.main --instance 30
    python -m NSGA3.main --instance 25 --pop 200 --gen 200

Called from app.py via run_nsga3_report().
"""

import argparse
import os
import sys
import time

MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(MODULE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from pymoo.algorithms.moo.nsga3    import NSGA3
from pymoo.util.ref_dirs           import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm   import PM
from pymoo.operators.sampling.rnd  import FloatRandomSampling
from pymoo.optimize                import minimize
from pymoo.termination             import get_termination

from models.parametres import load_instance
from .problem          import IRPProblem
from .decoder          import decode_chromosome, build_routes
from .evaluator        import compute_f1, compute_f2, compute_f3, compute_f4_detail
from .report           import write_report, generate_and_open, compute_node_positions

DEFAULT_REPORT_PATH = os.path.join(MODULE_DIR, "nsga3_report.html")

# Algorithm hyper-parameters
POP_SIZE       = 100
N_GEN          = 100
CROSSOVER_PROB = 0.9
MUTATION_PROB  = 0.2
N_PARTITIONS   = 6   # Das-Dennis: C(4+6-1,6) = 84 reference points for 4 objectives


def _build_delivery_rows(route_result, sets_, params_):
    """Per-(client, period) delivery rows with cumulative coverage for the report."""
    deliveries = []
    for l in sets_["clients"]:
        cum_recu = cum_dem = 0
        last_k   = -1
        for t in sets_["T"]:
            recu = int(route_result["actual_qty"].get((l, t), 0))
            dem  = int(params_["q_lt"].get((l, t), 0))
            if recu > 0:
                last_k = route_result["truck_assign"].get((l, t), -1)
            cum_recu += recu
            cum_dem  += dem
            deliveries.append({
                "l":        l,
                "t":        t,
                "k":        route_result["truck_assign"].get((l, t), last_k),
                "recu":     recu,
                "dem":      dem,
                "cum_recu": cum_recu,
                "cum_dem":  cum_dem,
                "balance":  cum_recu - cum_dem,
            })
    return deliveries


def run_nsga3(data_path=None, pop_size=POP_SIZE, n_gen=N_GEN,
              crossover_prob=CROSSOVER_PROB, mutation_prob=MUTATION_PROB):
    """Run NSGA-III and return the full report-ready data dict."""
    if data_path is None:
        data_path = os.path.join(PROJECT_DIR, "data", "instance_25_clients.json")

    sets_, params_ = load_instance(data_path)
    n_clients      = len(sets_["clients"])

    print(f"[NSGA3] {n_clients} clients | {len(sets_['T'])} periods | "
          f"{len(sets_['M'])} vehicles | {n_clients * len(sets_['T'])} genes | "
          f"pop={pop_size} gen={n_gen}", flush=True)

    problem  = IRPProblem(sets_, params_)
    ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=N_PARTITIONS)

    algorithm = NSGA3(
        pop_size  = pop_size,
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
        seed    = 42,
    )
    elapsed  = time.time() - t_start
    pareto_X = result.X  # (n_solutions, n_var)

    print(f"[NSGA3] Done in {elapsed:.1f}s | Pareto front: {len(pareto_X)} solutions",
          flush=True)

    # Re-evaluate each Pareto solution to collect full route detail for the report.
    # pymoo only stores objective values, not intermediate route dicts.
    solutions = []
    for i, chromosome in enumerate(pareto_X):
        quantities   = decode_chromosome(chromosome, sets_)
        route_result = build_routes(quantities, sets_, params_)

        f1         = compute_f1(route_result, sets_, params_)
        f2         = compute_f2(route_result, sets_, params_)
        f3         = compute_f3(route_result, sets_, params_)
        f4, f4_sub = compute_f4_detail(route_result, sets_, params_)

        routes_report = {}
        for t in sets_["T"]:
            trucks_t = [
                {"k": k, "path": info["path"], "qty": info["qty"]}
                for k, info in route_result["routes_data"].get(t, {}).items()
            ]
            # tau_return already computed by the decoder — no need to recompute
            tau_ret = route_result["tau_return"].get(t, 0.0)
            routes_report[str(t)] = {
                "trucks":     trucks_t,
                "R_frigo":    params_["R_frigo"].get(t, 0.0),
                "R_nonfrigo": params_["R_nonfrigo"].get(t, 0.0),
                "tau_return": round(tau_ret, 4),
                "shipped":    sum(route_result["actual_qty"].get((l, t), 0)
                                  for l in sets_["clients"]),
            }

        solutions.append({
            "id":         i,
            "objectives": {
                "f1": round(f1, 4),
                "f2": round(f2, 4),
                "f3": round(f3, 4),
                "f4": round(f4, 4),
            },
            "routes":      routes_report,
            "depot_stock": {str(t): route_result["depot_stock"][t] for t in sets_["T"]},
            "deliveries":  _build_delivery_rows(route_result, sets_, params_),
            "bfr_sub":     f4_sub,
        })

    return {
        "meta": {
            "instance":          f"{n_clients}_clients",
            "n_nodes":           len(sets_["N"]),
            "n_clients":         n_clients,
            "n_periods":         len(sets_["T"]),
            "n_vehicles":        len(sets_["M"]),
            "pop_size":          pop_size,
            "n_gen":             n_gen,
            "crossover_prob":    crossover_prob,
            "mutation_prob":     mutation_prob,
            "n_pareto":          len(solutions),
            "elapsed_s":         round(elapsed, 1),
            "node_positions":    compute_node_positions(sets_["N"]),
            "I_O_init_frigo":    params_["I_O_init_frigo"],
            "I_O_init_nonfrigo": params_["I_O_init_nonfrigo"],
        },
        "solutions": solutions,
    }


def run_nsga3_report(output_path=DEFAULT_REPORT_PATH, data_path=None,
                     pop_size=POP_SIZE, n_gen=N_GEN,
                     crossover_prob=CROSSOVER_PROB, mutation_prob=MUTATION_PROB):
    data = run_nsga3(data_path, pop_size, n_gen, crossover_prob, mutation_prob)
    return write_report(data, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSGA-III solver for the many-objective IRP")
    parser.add_argument("--instance", default="25", choices=["25", "30"])
    parser.add_argument("--pop",  type=int,   default=POP_SIZE)
    parser.add_argument("--gen",  type=int,   default=N_GEN)
    parser.add_argument("--cx",   type=float, default=CROSSOVER_PROB)
    parser.add_argument("--mut",  type=float, default=MUTATION_PROB)
    args = parser.parse_args()

    dp   = os.path.join(PROJECT_DIR, "data", f"instance_{args.instance}_clients.json")
    out  = os.path.join(MODULE_DIR,  f"nsga3_report_{args.instance}clients.html")
    data = run_nsga3(dp, args.pop, args.gen, args.cx, args.mut)
    path = write_report(data, out)
    print(f"[NSGA3] Report → {path}", flush=True)
    generate_and_open(path)