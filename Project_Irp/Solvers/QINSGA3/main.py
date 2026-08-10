"""QINSGA3 — Many-objective IRP solved with Quantum-Inspired NSGA-III.

Usage (standalone):
    python -m Solvers.QINSGA3.main                          # instance_25_clients.json
    python -m Solvers.QINSGA3.main --instance 30
    python -m Solvers.QINSGA3.main --instance 25 --pop 200 --gen 200 --runs 3

Called from app.py via run_qinsga3_report().
Report format is identical to NSGA3 — NSGA3/report.py is reused unchanged.

On Windows, multiprocessing uses the "spawn" start method, which requires all
worker-callable code to be importable at module level (see algorithm.py).
The _run_lock prevents concurrent GUI-triggered runs from corrupting shared state.
"""

import argparse
import json
import os
import sys
import time
import threading

MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(MODULE_DIR))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np
from pymoo.util.ref_dirs import get_reference_directions

from models.parametres              import load_instance
from Solvers.NSGA3.report_builder    import _evaluate_pareto, _build_report_data
from Solvers.NSGA3.report            import write_report, generate_and_open
from .algorithm                      import run_qinsga3

DEFAULT_REPORT_PATH = os.path.join(MODULE_DIR, "qinsga3_report.html")
_CHROM_CACHE_PATH   = os.path.join(MODULE_DIR, "qinsga3_chromosomes.json")

POP_SIZE     = 200
N_GEN        = 300
ALPHA_MAX    = 0.10  * np.pi   # Vmax = ±0.1π   [Li et al. ICNC 2008]
ALPHA_MIN    = 0.001 * np.pi   # Vmin = ±0.001π [Li et al. ICNC 2008]
# N_PARTITIONS=8 → 165 reference directions (das-dennis, M=4).
# Matches NSGA3 for a fair algorithmic comparison.
# Reducing to 6 (84 dirs) improves per-niche coverage but biases Spacing metrics.
N_PARTITIONS = 8
N_OBJ        = 4   # IRP objectives: f1 cost, f2 CO2, f3 time, f4 BFR

SEEDS = [42, 137, 271, 491, 613, 733, 857, 977, 1009, 1123,
         1249, 1373, 1499, 1609, 1733, 1871, 1997, 2113, 2237, 2351]

_run_lock = threading.Lock()


def run_qinsga3_solver(
    data_path:        str | None = None,
    pop_size:         int        = POP_SIZE,
    n_gen:            int        = N_GEN,
    alpha_max:        float      = ALPHA_MAX,
    alpha_min:        float      = ALPHA_MIN,
    p_cross:          float      = 0.9,
    eta_cross:        float      = 20.0,
    p_mut:            float | None = None,
    eta_mut:          float      = 20.0,
    migration_period: int        = 10,
    n_migrate:        int        = 10,
    n_runs:           int        = 1,
    rotation_type:    str        = "tanh",
    repair_final_front: bool     = True,
) -> dict:
    """Run QINSGA-III and return structured report data.

    Args:
        data_path:        Path to instance JSON (default: instance_25_clients.json).
        pop_size:         Population size; auto-bumped to ≥ n_ref_dirs if needed.
        n_gen:            Number of generations per run.
        alpha_max:        Max quantum rotation step (radians) — decays linearly to alpha_min.
        alpha_min:        Min quantum rotation step at last generation.
        p_cross:          SBX crossover probability (X-space, matches NSGA-III).
        eta_cross:        SBX distribution index (X-space; default 20 matches NSGA-III's own SBX).
        p_mut:            Per-gene mutation probability (default: 1/n_genes, matches NSGA-III).
        eta_mut:          PM distribution index (X-space; default 20 matches NSGA-III's own PM).
        migration_period: Every N generations, inject archive solutions into population.
        n_migrate:        Number of individuals updated per migration event.
        n_runs:           Independent runs (different seeds); results are merged.
        repair_final_front: Post-decode 2-opt repair on the returned Pareto
                             front only (Baldwinian -- see
                             Solvers/QINSGA3/repair.py). Default True.
    """
    n_runs = max(1, min(20, int(n_runs)))

    if data_path is None:
        data_path = os.path.join(PROJECT_DIR, "data", "instance_25_clients.json")

    sets_, params_ = load_instance(data_path)
    n_clients      = len(sets_["clients"])
    n_genes        = n_clients * len(sets_["T"]) + n_clients

    if p_mut is None:
        p_mut = 1.0 / n_genes
        print(f"[QINSGA3] p_mut = 1/D = 1/{n_genes} = {p_mut:.6f}", flush=True)

    ref_dirs = get_reference_directions("das-dennis", N_OBJ, n_partitions=N_PARTITIONS)

    effective_pop = max(pop_size, len(ref_dirs))
    if effective_pop != pop_size:
        print(f"[QINSGA3] pop_size bumped {pop_size} → {effective_pop} "
              f"(must be >= n_ref_dirs={len(ref_dirs)})", flush=True)

    print(
        f"[QINSGA3] {n_clients} clients | {len(sets_['T'])} periods | "
        f"{len(sets_['M'])} vehicles | {n_genes} genes | "
        f"pop={effective_pop} gen={n_gen} | α_max={alpha_max:.4f} | "
        f"p_cross={p_cross:.2f} | runs={n_runs}",
        flush=True,
    )

    if not _run_lock.acquire(blocking=False):
        raise RuntimeError(
            "[QINSGA3] Another run is already in progress. "
            "Wait for it to finish before launching a new one."
        )

    raw_runs = []
    try:
        for run_idx in range(n_runs):
            seed = SEEDS[run_idx % len(SEEDS)]
            print(f"\n[QINSGA3] === Run {run_idx + 1}/{n_runs}  seed={seed} ===", flush=True)

            def _progress(gen, F, G, pareto_idx):
                feasible = (np.maximum(G[pareto_idx], 0).sum(axis=1) == 0).sum()
                print(
                    f"[QINSGA3]  gen {gen:4d} | Pareto={len(pareto_idx)} "
                    f"| feasible={feasible}",
                    flush=True,
                )

            t_start = time.time()
            pareto_X, pareto_F, pareto_G = run_qinsga3(
                sets_               = sets_,
                params_             = params_,
                ref_dirs            = ref_dirs,
                pop_size            = effective_pop,
                max_gen             = n_gen,
                alpha_max           = alpha_max,
                alpha_min           = alpha_min,
                p_cross             = p_cross,
                eta_cross           = eta_cross,
                p_mut               = p_mut,
                eta_mut             = eta_mut,
                migration_period    = migration_period,
                n_migrate           = n_migrate,
                seed                = seed,
                rotation_type       = rotation_type,
                callback            = _progress,
                # run_qinsga3_solver does its own repair later via
                # _evaluate_pareto (which decodes pareto_X from scratch), so
                # the returned pareto_F here is never used -- always skip the
                # internal repair to avoid repairing the final front twice.
                repair_final_front  = False,
            )
            elapsed = time.time() - t_start

            if pareto_X is None or len(pareto_X) == 0:
                print(f"[QINSGA3] Run {run_idx + 1}: no feasible solutions — skipping.", flush=True)
                continue

            print(
                f"[QINSGA3] Run {run_idx + 1} done in {elapsed:.1f}s | "
                f"Pareto front: {len(pareto_X)} solutions",
                flush=True,
            )
            raw_runs.append({"seed": seed, "elapsed": elapsed,
                              "pareto_X": pareto_X})
    finally:
        _run_lock.release()

    if not raw_runs:
        raise RuntimeError(
            "[QINSGA3] No feasible solutions found in any run. "
            "Try increasing pop_size / n_gen or relaxing tau_max."
        )

    base_meta = {
        "instance":           f"{n_clients}_clients",
        "pop_size":           effective_pop,
        "n_gen":              n_gen,
        "alpha_max":          round(alpha_max, 6),
        "alpha_min":          round(alpha_min, 6),
        "crossover_prob":     round(p_cross, 6),
        "eta_cross":          round(float(eta_cross), 6),
        "mutation_prob":      round(p_mut, 6),
        "eta_mut":            round(float(eta_mut), 6),
        "n_runs":             n_runs,
        "n_completed":        len(raw_runs),
        "algorithm":          "QINSGA3",
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
        "algorithm": "QINSGA3",
        "n_runs":    len(raw_runs),
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


def render_from_instance(data_path: str) -> dict:
    """Re-evaluate cached QINSGA-III chromosomes with the current instance JSON."""
    if not os.path.exists(_CHROM_CACHE_PATH):
        raise FileNotFoundError(
            "No cached QINSGA-III chromosomes. Run the algorithm at least once first."
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
                f"Run {i+1}: cached chromosomes have {pareto_X.shape[1]} genes but "
                f"current instance requires {n_genes_expected}. Re-run the algorithm."
            )
        repair = run_cache["meta_base"].get("repair_final_front", False)
        runs_data.append(_evaluate_pareto(pareto_X, sets_, params_, run_cache["meta_base"], repair=repair))

    return _build_report_data(runs_data)


def run_qinsga3_report(
    output_path:      str        = DEFAULT_REPORT_PATH,
    data_path:        str | None = None,
    pop_size:         int        = POP_SIZE,
    n_gen:            int        = N_GEN,
    alpha_max:        float      = ALPHA_MAX,
    alpha_min:        float      = ALPHA_MIN,
    p_cross:          float      = 0.9,
    eta_cross:        float      = 20.0,
    p_mut:            float | None = None,
    eta_mut:          float      = 20.0,
    migration_period: int        = 10,
    n_migrate:        int        = 10,
    n_runs:           int        = 1,
    rotation_type:    str        = "tanh",
    repair_final_front: bool     = True,
) -> str:
    data = run_qinsga3_solver(
        data_path, pop_size, n_gen, alpha_max, alpha_min,
        p_cross, eta_cross, p_mut, eta_mut,
        migration_period=migration_period,
        n_migrate=n_migrate,
        n_runs=n_runs,
        rotation_type=rotation_type,
        repair_final_front=repair_final_front,
    )
    return write_report(data, output_path, algo_label="QI-NSGA-III")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QINSGA-III solver for the many-objective IRP")
    parser.add_argument("--instance", default="25", choices=["3", "5", "15", "25", "30", "40", "100"])
    parser.add_argument("--pop",       type=int,   default=POP_SIZE)
    parser.add_argument("--gen",       type=int,   default=N_GEN)
    parser.add_argument("--alpha-max", type=float, default=ALPHA_MAX)
    parser.add_argument("--alpha-min", type=float, default=ALPHA_MIN)
    parser.add_argument("--cross",     type=float, default=0.9,
                        help="SBX crossover probability, X-space (default: 0.9)")
    parser.add_argument("--eta",       type=float, default=20.0,
                        help="SBX distribution index η, X-space (default: 20, matches NSGA-III)")
    parser.add_argument("--mut",       type=float, default=None,
                        help="Per-gene mutation probability, X-space (default: 1/D)")
    parser.add_argument("--eta-mut",   type=float, default=20.0,
                        help="PM distribution index η, X-space (default: 20, matches NSGA-III)")
    parser.add_argument("--runs",      type=int,   default=1)
    args = parser.parse_args()

    dp   = os.path.join(PROJECT_DIR, "data", f"instance_{args.instance}_clients.json")
    out  = os.path.join(MODULE_DIR,  f"qinsga3_report_{args.instance}clients.html")
    data = run_qinsga3_solver(
        dp, args.pop, args.gen, args.alpha_max, args.alpha_min,
        p_cross=args.cross, eta_cross=args.eta,
        p_mut=args.mut, eta_mut=args.eta_mut, n_runs=args.runs,
    )
    path = write_report(data, out, algo_label="QI-NSGA-III")
    print(f"[QINSGA3] Report → {path}", flush=True)
    generate_and_open(path)
