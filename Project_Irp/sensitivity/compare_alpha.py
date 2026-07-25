"""Alpha (rotation step) comparison: alpha_max and alpha_min tuning.

Phase 1 — alpha_max comparison (alpha_min fixed at 0.001*pi):
  Tests three alpha_max values while keeping alpha_min constant.
  Determines the optimal upper rotation bound.

Phase 2 — alpha_min comparison (alpha_max fixed at best from phase 1):
  Tests three alpha_min values to determine the optimal lower bound.

The linear decay schedule is:
    alpha(g) = alpha_min + (alpha_max - alpha_min) * (1 - g / max_gen)

Reference: Li et al. (ICNC 2008) use Vmax = ±0.1*pi, Vmin = ±0.001*pi
for a PSO-based quantum algorithm. Values adapted here for NSGA-III context.

Metrics (fixed global HV reference point):
  - Hypervolume (HV)     — higher is better
  - Pareto front size    — higher is better

Usage:
    python -m sensitivity.compare_alpha --instance 15 --runs 10 --gen 200
    python -m sensitivity.compare_alpha --instance 15 --runs 10 --gen 200 --phase 2
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
SEEDS        = [42, 137, 271, 491, 613, 733, 857, 977, 1009, 1123]
N_PARTITIONS = 8

# Phase 1 : vary alpha_max, fix alpha_min
ALPHA_MIN_FIXED = 0.001 * np.pi

ALPHA_MAX_VARIANTS = {
    "0.01*pi (petit)":   0.01  * np.pi,
    "0.05*pi (actuel)":  0.05  * np.pi,
    "0.10*pi (Li 2008)": 0.10  * np.pi,
}

# Phase 2 : fix alpha_max at best from phase 1, vary alpha_min
# (alpha_max will be set dynamically after phase 1)
ALPHA_MIN_VARIANTS = {
    "0.0005*pi": 0.0005 * np.pi,
    "0.001*pi (actuel)": 0.001 * np.pi,
    "0.005*pi": 0.005  * np.pi,
}


# ---------------------------------------------------------------------------
# Decay schedule display
# ---------------------------------------------------------------------------
def _print_decay(alpha_max: float, alpha_min: float, max_gen: int) -> None:
    print(f"\n  Decroissance lineaire : alpha(g) = alpha_min + (alpha_max - alpha_min)*(1 - g/G)")
    gens = [0, max_gen // 4, max_gen // 2, 3 * max_gen // 4, max_gen - 1]
    print(f"  {'Generation':>12} {'alpha':>12} {'% de alpha_max':>16}")
    print(f"  {'─'*12} {'─'*12} {'─'*16}")
    for g in gens:
        a = alpha_min + (alpha_max - alpha_min) * (1 - g / max_gen)
        print(f"  {g:>12} {a:>12.5f} {a/alpha_max*100:>15.1f}%")
    print()


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def _run_one(
    sets_, params_, ref_dirs, alpha_max, alpha_min, pop_size, max_gen, seed
) -> tuple[np.ndarray | None, float]:
    t0 = time.time()
    try:
        _, pareto_F, _ = run_qinsga3(
            sets_      = sets_,
            params_    = params_,
            ref_dirs   = ref_dirs,
            pop_size   = pop_size,
            max_gen    = max_gen,
            alpha_max  = alpha_max,
            alpha_min  = alpha_min,
            seed       = seed,
        )
    except RuntimeError:
        pareto_F = None
    return pareto_F, round(time.time() - t0, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _stats(values: list[float]) -> dict:
    arr = np.array([v for v in values if not np.isnan(v)])
    if len(arr) == 0:
        return {"mean": float("nan"), "best": float("nan"),
                "worst": float("nan"), "std": float("nan")}
    return {"mean": float(np.mean(arr)), "best": float(np.max(arr)),
            "worst": float(np.min(arr)), "std": float(np.std(arr))}


def _run_phase(
    label: str,
    variants: dict[str, tuple[float, float]],  # name -> (alpha_max, alpha_min)
    sets_, params_, ref_dirs, pop_size, max_gen, n_runs,
) -> list[dict]:
    """Run all variants and return result list."""
    raw: dict[str, list] = {name: [] for name in variants}

    for name, (amax, amin) in variants.items():
        print(f"\n>>> {label} = {name}  (alpha_max={amax:.5f}, alpha_min={amin:.5f})")
        for run_idx in range(n_runs):
            seed = SEEDS[run_idx % len(SEEDS)]
            print(f"  Run {run_idx+1}/{n_runs}  seed={seed} ... ", end="", flush=True)
            pareto_F, elapsed = _run_one(
                sets_, params_, ref_dirs, amax, amin, pop_size, max_gen, seed
            )
            raw[name].append((pareto_F, elapsed))
            size = len(pareto_F) if pareto_F is not None else 0
            print(f"front={size}  time={elapsed}s")

    # Fixed global reference point
    all_F = [F for runs in raw.values() for (F, _) in runs if F is not None]
    if not all_F:
        print("Aucune solution. Arret.")
        return []

    global_nadir = np.max(np.vstack([F.max(axis=0) for F in all_F]), axis=0) * 1.1 + 1e-6
    hv_ind       = HV(ref_point=global_nadir)
    print(f"\n  Point de reference HV global : {np.round(global_nadir, 2)}")

    results = []
    for name in variants:
        for run_idx, (pareto_F, elapsed) in enumerate(raw[name]):
            seed = SEEDS[run_idx % len(SEEDS)]
            hv   = float(hv_ind(pareto_F)) if pareto_F is not None else float("nan")
            size = len(pareto_F) if pareto_F is not None else 0
            results.append({
                "name": name, "seed": seed,
                "hv": hv, "front_size": size, "elapsed_s": elapsed,
            })
    return results


def _print_results(results: list[dict], variant_names: list[str]) -> str:
    """Print summary tables. Returns name of best variant by HV."""
    print("\n" + "=" * 74)
    print("  RESULTATS AGREGES  (HV avec reference fixe globale)")
    print("=" * 74)

    for metric, label in [("hv", "Hypervolume (superieur = meilleur)"),
                           ("front_size", "Taille front Pareto (superieur = meilleur)")]:
        print(f"\n{'─'*74}")
        print(f"  {label}")
        print(f"{'─'*74}")
        print(f"  {'Variante':<22} {'Moyenne':>14} {'Meilleur':>12} {'Pire':>12} {'Ecart-type':>12}")
        print(f"  {'─'*22} {'─'*14} {'─'*12} {'─'*12} {'─'*12}")
        row_means = {}
        for name in variant_names:
            vals  = [r[metric] for r in results if r["name"] == name]
            s     = _stats(vals)
            row_means[name] = s["mean"]
            print(f"  {name:<22} {s['mean']:>14.2f} {s['best']:>12.2f} "
                  f"{s['worst']:>12.2f} {s['std']:>12.2f}")
        best = max(row_means, key=lambda k: row_means[k])
        print(f"\n  Meilleur : {best}")
        print(f"{'─'*74}")

    # Per-run detail
    print(f"\n{'─'*74}  DETAIL PAR RUN")
    print(f"  {'Variante':<22} {'Seed':>6} {'HV':>16} {'Front':>7} {'Temps':>7}")
    print(f"  {'─'*22} {'─'*6} {'─'*16} {'─'*7} {'─'*7}")
    for r in results:
        hv_str = f"{r['hv']:.2f}" if not np.isnan(r["hv"]) else "N/A"
        print(f"  {r['name']:<22} {r['seed']:>6} {hv_str:>16} "
              f"{r['front_size']:>7} {r['elapsed_s']:>6.1f}s")

    # Ranking
    print(f"\n{'─'*74}  CLASSEMENT FINAL")
    hv_by_name = {
        name: np.nanmean([r["hv"] for r in results if r["name"] == name])
        for name in variant_names
    }
    ranking  = sorted(variant_names, key=lambda n: hv_by_name[n], reverse=True)
    hv_best  = hv_by_name[ranking[0]]
    for pos, name in enumerate(ranking, 1):
        diff   = (hv_best - hv_by_name[name]) / hv_best * 100
        marker = "MEILLEUR" if pos == 1 else f"-{diff:.1f}% vs meilleur"
        print(f"  {pos}. {name:<22}  HV = {hv_by_name[name]:.2f}   {marker}")
    print(f"{'─'*74}\n")
    return ranking[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_comparison(
    instance: str = "15",
    n_runs:   int = 10,
    max_gen:  int = 200,
    pop_size: int = 200,
    phase:    int = 1,
    best_alpha_max: float | None = None,
) -> None:
    data_path = os.path.join(PROJECT_DIR, "data", f"instance_{instance}_clients.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Instance non trouvee : {data_path}")

    sets_, params_ = load_instance(data_path)
    ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=N_PARTITIONS)
    effective_pop = max(pop_size, len(ref_dirs))

    print("=" * 74)
    print("  COMPARAISON BORNES DE ROTATION alpha_max / alpha_min — QINSGA-III")
    print("=" * 74)
    print(f"  Instance : {instance} clients  |  pop={effective_pop}  gen={max_gen}  runs={n_runs}")
    print(f"  Reference litterature : Li et al. (ICNC 2008) Vmax=0.1*pi, Vmin=0.001*pi")
    print("=" * 74)

    if phase == 1:
        print("\n  PHASE 1 — Comparaison alpha_max  (alpha_min fixe = 0.001*pi)\n")
        for name, amax in ALPHA_MAX_VARIANTS.items():
            _print_decay(amax, ALPHA_MIN_FIXED, max_gen)

        variants = {
            name: (amax, ALPHA_MIN_FIXED)
            for name, amax in ALPHA_MAX_VARIANTS.items()
        }
        results  = _run_phase("alpha_max", variants, sets_, params_,
                               ref_dirs, effective_pop, max_gen, n_runs)
        best_name = _print_results(results, list(variants))
        best_amax = ALPHA_MAX_VARIANTS[best_name]
        print(f"  => Meilleur alpha_max : {best_name}  ({best_amax:.5f} rad)")
        print(f"  => Relancez avec --phase 2 --best-alpha-max {best_amax:.5f}")
        print()

    else:  # phase 2
        if best_alpha_max is None:
            best_alpha_max = 0.05 * np.pi
            print(f"  [INFO] alpha_max non specifie, utilisation de 0.05*pi")

        print(f"\n  PHASE 2 — Comparaison alpha_min  (alpha_max fixe = {best_alpha_max:.5f})\n")
        for name, amin in ALPHA_MIN_VARIANTS.items():
            _print_decay(best_alpha_max, amin, max_gen)

        variants = {
            name: (best_alpha_max, amin)
            for name, amin in ALPHA_MIN_VARIANTS.items()
        }
        results   = _run_phase("alpha_min", variants, sets_, params_,
                                ref_dirs, effective_pop, max_gen, n_runs)
        best_name = _print_results(results, list(variants))
        best_amin = ALPHA_MIN_VARIANTS[best_name]
        print(f"  => Meilleur alpha_min : {best_name}  ({best_amin:.6f} rad)")
        print(f"  => Parametres finaux : alpha_max={best_alpha_max:.5f}, alpha_min={best_amin:.6f}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparaison alpha QINSGA-III")
    parser.add_argument("--instance", default="15",
                        choices=["3", "5", "15", "25", "30", "40", "100"])
    parser.add_argument("--runs",  type=int,   default=10)
    parser.add_argument("--gen",   type=int,   default=200)
    parser.add_argument("--pop",   type=int,   default=200)
    parser.add_argument("--phase", type=int,   default=1, choices=[1, 2])
    parser.add_argument("--best-alpha-max", type=float, default=None,
                        dest="best_alpha_max",
                        help="alpha_max optimal de la phase 1 (pour la phase 2)")
    args = parser.parse_args()

    run_comparison(
        args.instance, args.runs, args.gen, args.pop,
        args.phase, args.best_alpha_max,
    )
