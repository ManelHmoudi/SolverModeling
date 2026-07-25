"""Rotation gate comparison: tanh / tanh_soft / linear.

Runs QINSGA-III on the IRP instance with all rotation variants over N_RUNS
independent seeds. Hypervolume is computed with a FIXED global reference point
(worst nadir across all runs × 1.1) so values are directly comparable.

Metrics:
  - Hypervolume (HV, fixed ref)  — higher is better
  - Pareto front size            — higher is better
  - Runtime (seconds)

Usage:
    python -m sensitivity.compare_rotation --instance 15 --runs 10 --gen 200
    python -m sensitivity.compare_rotation --instance 5  --runs 5  --gen 150
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
from pymoo.indicators.hv import HV
from pymoo.util.ref_dirs import get_reference_directions

from models.parametres import load_instance
from QINSGA3.algorithm import run_qinsga3

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEEDS = [42, 137, 271, 491, 613, 733, 857, 977, 1009, 1123]

ROTATION_TYPES = {
    "tanh":      "Δθ = α·tanh((θ_guide−θ)/(π/8))  [agressif]",
    "tanh_soft": "Δθ = α·tanh((θ_guide−θ)/(π/4))  [doux]",
    "linear":    "Δθ = α·(θ_guide−θ)/(π/2)         [linéaire, littérature]",
}

N_PARTITIONS = 8


# ---------------------------------------------------------------------------
# Step behaviour summary (printed for reference)
# ---------------------------------------------------------------------------
def _print_step_table() -> None:
    print("\n  Comportement du pas Δθ (en % de α) selon la distance au guide :")
    print(f"  {'diff':>8} | {'tanh(π/8)':>10} | {'tanh_soft(π/4)':>14} | {'linear':>8}")
    print(f"  {'─'*8}-+-{'─'*10}-+-{'─'*14}-+-{'─'*8}")
    diffs = [np.pi/16, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2]
    labels = ["π/16", "π/8", "π/4", "3π/8", "π/2"]
    for d, lbl in zip(diffs, labels):
        t1 = np.tanh(d / (np.pi/8)) * 100
        t2 = np.tanh(d / (np.pi/4)) * 100
        li = (d / (np.pi/2)) * 100
        print(f"  {lbl:>8} | {t1:>9.1f}% | {t2:>13.1f}% | {li:>7.1f}%")
    print()


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def _run_one(
    sets_, params_, ref_dirs, rotation_type, pop_size, max_gen, seed
) -> tuple[np.ndarray | None, float]:
    """Return (pareto_F, elapsed_s). pareto_F is None if no solution found."""
    t0 = time.time()
    try:
        _, pareto_F, _ = run_qinsga3(
            sets_         = sets_,
            params_       = params_,
            ref_dirs      = ref_dirs,
            pop_size      = pop_size,
            max_gen       = max_gen,
            rotation_type = rotation_type,
            seed          = seed,
        )
    except RuntimeError:
        pareto_F = None
    elapsed = time.time() - t0
    return pareto_F, round(elapsed, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _stats(values: list[float]) -> dict:
    arr = np.array([v for v in values if not np.isnan(v)])
    if len(arr) == 0:
        return {"mean": float("nan"), "best": float("nan"), "worst": float("nan"), "std": float("nan")}
    return {
        "mean":  float(np.mean(arr)),
        "best":  float(np.max(arr)),
        "worst": float(np.min(arr)),
        "std":   float(np.std(arr)),
    }


def _print_table(results: list[dict], metric: str, label: str) -> None:
    print(f"\n{'─'*70}")
    print(f"  {label}")
    print(f"{'─'*70}")
    print(f"  {'Rotation':<12} {'Moyenne':>14} {'Meilleur':>14} {'Pire':>14} {'Écart-type':>12}")
    print(f"  {'─'*12} {'─'*14} {'─'*14} {'─'*14} {'─'*12}")

    row_means = {}
    for rtype in ROTATION_TYPES:
        vals  = [r[metric] for r in results if r["rotation"] == rtype]
        stats = _stats(vals)
        row_means[rtype] = stats["mean"]
        print(
            f"  {rtype:<12} {stats['mean']:>14.2f} {stats['best']:>14.2f} "
            f"{stats['worst']:>14.2f} {stats['std']:>12.2f}"
        )

    best_rtype = max(row_means, key=lambda k: row_means[k])
    print(f"\n  ★ Meilleur : {best_rtype.upper()}")
    print(f"{'─'*70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_comparison(
    instance:  str = "5",
    n_runs:    int = 5,
    max_gen:   int = 150,
    pop_size:  int = 200,
) -> None:
    data_path = os.path.join(PROJECT_DIR, "data", f"instance_{instance}_clients.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Instance non trouvée : {data_path}")

    sets_, params_ = load_instance(data_path)
    ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=N_PARTITIONS)
    effective_pop = max(pop_size, len(ref_dirs))

    n_clients = len(sets_["clients"])
    n_genes   = n_clients * len(sets_["T"]) + n_clients

    print("=" * 70)
    print("  COMPARAISON PORTES DE ROTATION — QINSGA-III")
    print("=" * 70)
    print(f"  Instance   : {instance} clients  |  Variables : {n_genes}")
    print(f"  pop_size   : {effective_pop}  |  max_gen : {max_gen}  |  runs : {n_runs}")
    print(f"  Seeds      : {SEEDS[:n_runs]}")
    print()
    for rtype, desc in ROTATION_TYPES.items():
        print(f"  [{rtype:<10}]  {desc}")
    print("=" * 70)

    _print_step_table()

    # ── Phase 1 : collecte de tous les fronts ───────────────────────────────
    raw: dict[str, list] = {rt: [] for rt in ROTATION_TYPES}   # {rtype: [(pareto_F, elapsed)]}

    for rtype in ROTATION_TYPES:
        print(f"\n>>> Rotation : {rtype.upper()}")
        for run_idx in range(n_runs):
            seed = SEEDS[run_idx % len(SEEDS)]
            print(f"  Run {run_idx+1}/{n_runs}  seed={seed} ... ", end="", flush=True)
            pareto_F, elapsed = _run_one(
                sets_, params_, ref_dirs, rtype, effective_pop, max_gen, seed
            )
            raw[rtype].append((pareto_F, elapsed))
            size = len(pareto_F) if pareto_F is not None else 0
            print(f"front={size}  time={elapsed}s")

    # ── Phase 2 : point de référence HV global ──────────────────────────────
    # Construit à partir du nadir le plus pessimiste sur TOUS les runs et configs.
    all_F = [F for runs in raw.values() for (F, _) in runs if F is not None]
    if not all_F:
        print("\nAucune solution faisable trouvée. Arrêt.")
        return

    global_nadir = np.max(np.vstack([F.max(axis=0) for F in all_F]), axis=0) * 1.1 + 1e-6
    hv_indicator = HV(ref_point=global_nadir)

    print(f"\n  Point de référence HV global (nadir×1.1) : {np.round(global_nadir, 2)}")

    # ── Phase 3 : calcul des métriques ──────────────────────────────────────
    all_results: list[dict] = []
    for rtype in ROTATION_TYPES:
        for run_idx, (pareto_F, elapsed) in enumerate(raw[rtype]):
            seed = SEEDS[run_idx % len(SEEDS)]
            hv   = float(hv_indicator(pareto_F)) if pareto_F is not None else float("nan")
            size = len(pareto_F) if pareto_F is not None else 0
            all_results.append({
                "rotation":   rtype,
                "seed":       seed,
                "hv":         hv,
                "front_size": size,
                "elapsed_s":  elapsed,
            })

    # ── Tableaux de résultats ────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  RÉSULTATS AGRÉGÉS  (HV calculé avec point de référence FIXE global)")
    print("=" * 70)

    _print_table(all_results, "hv",         "Hypervolume (↑ meilleur)")
    _print_table(all_results, "front_size", "Taille front Pareto (↑ meilleur)")

    # Runtime
    print(f"\n{'─'*70}")
    print("  Temps moyen par run")
    print(f"{'─'*70}")
    for rtype in ROTATION_TYPES:
        times = [r["elapsed_s"] for r in all_results if r["rotation"] == rtype]
        print(f"  {rtype:<12}  moy={np.mean(times):.1f}s  "
              f"min={min(times):.1f}s  max={max(times):.1f}s")

    # Détail par run
    print(f"\n{'─'*70}")
    print("  DÉTAIL PAR RUN")
    print(f"{'─'*70}")
    print(f"  {'Rotation':<12} {'Seed':>6} {'HV':>16} {'Front':>8} {'Time':>8}")
    print(f"  {'─'*12} {'─'*6} {'─'*16} {'─'*8} {'─'*8}")
    for r in all_results:
        hv_str = f"{r['hv']:.2f}" if not np.isnan(r["hv"]) else "N/A"
        print(
            f"  {r['rotation']:<12} {r['seed']:>6} {hv_str:>16} "
            f"{r['front_size']:>8} {r['elapsed_s']:>7.1f}s"
        )

    # Classement final
    print(f"\n{'─'*70}")
    print("  CLASSEMENT FINAL (HV moyen)")
    print(f"{'─'*70}")
    ranking = sorted(
        ROTATION_TYPES.keys(),
        key=lambda rt: np.nanmean([r["hv"] for r in all_results if r["rotation"] == rt]),
        reverse=True,
    )
    hv_best = np.nanmean([r["hv"] for r in all_results if r["rotation"] == ranking[0]])
    for pos, rtype in enumerate(ranking, 1):
        hv_mean = np.nanmean([r["hv"] for r in all_results if r["rotation"] == rtype])
        diff    = (hv_best - hv_mean) / hv_best * 100 if hv_best > 0 else 0
        marker  = "★ MEILLEUR" if pos == 1 else f"  −{diff:.1f}% vs meilleur"
        print(f"  {pos}. {rtype:<12}  HV moy = {hv_mean:.2f}   {marker}")
    print(f"{'─'*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparaison portes de rotation QINSGA-III")
    parser.add_argument("--instance", default="5",
                        choices=["3", "5", "15", "25", "30", "40", "100"])
    parser.add_argument("--runs",    type=int, default=5)
    parser.add_argument("--gen",     type=int, default=150)
    parser.add_argument("--pop",     type=int, default=200)
    args = parser.parse_args()
    run_comparison(args.instance, args.runs, args.gen, args.pop)
