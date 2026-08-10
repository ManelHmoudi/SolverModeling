"""Best-improvement vs first-improvement 2-opt repair test for QINSGA-III's
`repair_final_front`.

Context: `sensitivity/compare_repair_thoroughness.py` (this session) tested
widening `_repair_route_result`'s 2-opt candidate window from 8 to
unbounded, expecting a widened neighbourhood could only help. It didn't --
mixed, mostly non-significant, GD even trended worse in 5/5 paired seeds.
Root cause identified: `_repair_route_result` accepts the FIRST candidate
that improves `_route_f1_contribution` (first-improvement local search), not
the best one found. Widening the candidate window does not monotonically
help under first-improvement -- it can change WHICH improving swap is found
first, sending the local search down an entirely different (not necessarily
better) trajectory of subsequent iterations. Only best-improvement
(evaluate every candidate in the neighbourhood each iteration, apply
strictly the best one) is mathematically guaranteed to reach an
equal-or-better local optimum than first-improvement, for the same
neighbourhood.

This script isolates exactly that one variable -- acceptance rule
(first-improvement vs best-improvement) -- keeping the window at the
production value (8) in both arms, so window size is not a confound (that
axis was already tested separately, see compare_repair_thoroughness.py).

Same paired design as compare_repair_thoroughness.py: `run_qinsga3(...,
repair_final_front=False)` run ONCE per seed to get `pareto_X` (repair is
Baldwinian, never touches X), then BOTH repair variants are applied to that
SAME `pareto_X` -- no search-run-to-run variance, half the compute cost of
two independent campaigns.

Usage:
    python -m sensitivity.compare_repair_best_improvement
    python -m sensitivity.compare_repair_best_improvement --seeds 42 137 271 491 613 --gen 300
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
from scipy.stats import mannwhitneyu, wilcoxon

from models.parametres          import load_instance
from Solvers.NSGA3.decoder       import decode_chromosome, build_routes
from Solvers.NSGA3.evaluator     import compute_f1, compute_f2, compute_f3, compute_f4
from Solvers.NSGA3.metrics       import compute_pareto_metrics
from Solvers.QINSGA3.algorithm   import run_qinsga3, _repair_pareto_front, _build_g_constraints
from Solvers.QINSGA3.repair      import (
    _two_opt_candidates, _evaluate_candidate, _replace_route_arcs,
    _route_traversal_time, _MAX_REPAIR_ITER, _TWO_OPT_WINDOW,
)

N_PARTITIONS = 8
N_OBJ        = 4
POP_SIZE     = 200

DEFAULT_SEEDS = [42, 137, 271]

_NSGA3_CACHE = os.path.join(PROJECT_DIR, "Solvers", "NSGA3", "nsga3_chromosomes.json")


def _route_f1_contribution(path, f_vars, arrival_times, t, k, params_) -> float:
    """Copy of repair.py's private helper (not exported)."""
    c_ijk = params_["c_ijk"]
    d     = params_["d"]
    c1    = params_["c1"]
    c2    = params_["c2"]
    ET    = params_["ET"]
    LT    = params_["LT"]
    y1 = 0.0
    for idx in range(len(path) - 1):
        i, j = path[idx], path[idx + 1]
        y1 += c_ijk[i, j, k] * d[i, j] * f_vars[i, j, t, k]
    y3 = 0.0
    for l in path[1:-1]:
        arr = arrival_times.get((l, t), 0.0)
        if arr > 0.0:
            y3 += c1 * max(0.0, ET[l, t] - arr)
            y3 += c2 * max(0.0, arr - LT[l, t])
    return y1 + y3


def _repair_route_result_best(route_result: dict, sets_: dict, params_: dict, window: int | None) -> dict:
    """Best-improvement variant of repair.py's `_repair_route_result`: each
    iteration evaluates EVERY candidate in the 2-opt neighbourhood (not just
    until the first improving one) and applies strictly the best one --
    mathematically guaranteed to reach an equal-or-better local optimum than
    first-improvement for the same neighbourhood, unlike widening the window
    alone under first-improvement (see module docstring)."""
    working = dict(route_result)
    working["x"] = dict(route_result["x"])
    working["f"] = dict(route_result["f"])
    working["arrival_times"] = dict(route_result["arrival_times"])
    working["tau_return"] = dict(route_result["tau_return"])
    working["routes_data"] = {
        t: dict(routes) for t, routes in route_result["routes_data"].items()
    }

    for t, routes in route_result["routes_data"].items():
        tau_return_before = route_result["tau_return"].get(t, 0.0)

        for k, info in routes.items():
            path = list(info["path"])
            if len(path) <= 3:
                continue
            qty_on_route = {int(l): q for l, q in info["qty"].items()}

            current_contrib = _route_f1_contribution(
                path, working["f"], working["arrival_times"], t, k, params_
            )

            for _ in range(_MAX_REPAIR_ITER):
                best = None   # (candidate, trial_x, trial_f, trial_arrivals, trial_contrib)
                best_contrib = current_contrib
                for i, j, candidate in _two_opt_candidates(path, max_window=window):
                    evaluated = _evaluate_candidate(
                        candidate, qty_on_route, t, k, tau_return_before, params_
                    )
                    if evaluated is None:
                        continue
                    trial_x, trial_f, trial_arrivals, trial_contrib = evaluated
                    if trial_contrib < best_contrib:
                        best_contrib = trial_contrib
                        best = (candidate, trial_x, trial_f, trial_arrivals, trial_contrib)

                if best is None:
                    break
                candidate, trial_x, trial_f, trial_arrivals, trial_contrib = best
                working["x"] = _replace_route_arcs(working["x"], path, t, k, trial_x)
                working["f"] = _replace_route_arcs(working["f"], path, t, k, trial_f)
                working["arrival_times"] = {**working["arrival_times"], **trial_arrivals}
                working["routes_data"][t][k] = {"path": candidate, "qty": info["qty"]}
                working["tau_return"][t] = max(
                    _route_traversal_time(r["path"], k2, params_)
                    for k2, r in working["routes_data"][t].items()
                )
                path = candidate
                current_contrib = trial_contrib

    return working


def _evaluate_with_repair_best(x: np.ndarray, sets_: dict, params_: dict, window: int | None):
    quantities, priorities = decode_chromosome(x, sets_)
    route_result = build_routes(quantities, sets_, params_, priorities)
    route_result = _repair_route_result_best(route_result, sets_, params_, window)

    F = np.array([
        compute_f1(route_result, sets_, params_),
        compute_f2(route_result, sets_, params_),
        compute_f3(route_result, sets_, params_),
        compute_f4(route_result, sets_, params_),
    ])
    G = np.array(_build_g_constraints(route_result, sets_, params_))
    return F, G


def _repair_pareto_front_best(pareto_X: np.ndarray, sets_: dict, params_: dict, window: int | None):
    F_list, G_list = [], []
    for x in pareto_X:
        F, G = _evaluate_with_repair_best(x, sets_, params_, window)
        F_list.append(F)
        G_list.append(G)
    return np.array(F_list), np.array(G_list)


# ---------------------------------------------------------------------------
# Comparison harness
# ---------------------------------------------------------------------------

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
    print("  REPARATION 2-OPT BEST-IMPROVEMENT vs FIRST-IMPROVEMENT (fenetre=8) — QINSGA-III")
    print("=" * 92)
    print(f"  Instance : {instance} clients | pop={effective_pop} | gen={max_gen}")
    print(f"  Seeds    : {seeds}")
    print("=" * 92)

    print("\nDecodage du front NSGA-III (cache, reference fixe)...")
    nsga3_F_runs = _load_nsga3_reference(sets_, params_, instance)
    print(f"  NSGA-III : {len(nsga3_F_runs)} runs, "
          f"{sum(len(f) for f in nsga3_F_runs)} solutions")

    raw: dict[str, list] = {"baseline (first-improvement, prod)": [], "best-improvement (test)": []}

    print("\n>>> un seul run de recherche par seed (repair_final_front=False), puis les deux reparations")
    for seed in seeds:
        print(f"  seed={seed} ... recherche ... ", end="", flush=True)
        t0 = time.time()
        pareto_X, _, _ = run_qinsga3(
            sets_=sets_, params_=params_, ref_dirs=ref_dirs,
            pop_size=effective_pop, max_gen=max_gen, seed=seed,
            repair_final_front=False,
        )
        t_search = time.time() - t0

        t0 = time.time()
        F_baseline, _ = _repair_pareto_front(pareto_X, sets_, params_)
        t_baseline_repair = time.time() - t0
        raw["baseline (first-improvement, prod)"].append((seed, F_baseline, t_search + t_baseline_repair))

        t0 = time.time()
        F_test, _ = _repair_pareto_front_best(pareto_X, sets_, params_, window=_TWO_OPT_WINDOW)
        t_test_repair = time.time() - t0
        raw["best-improvement (test)"].append((seed, F_test, t_search + t_test_repair))

        print(f"front={len(pareto_X)}  t_search={t_search:.1f}s  "
              f"t_repair_first={t_baseline_repair:.1f}s  t_repair_best={t_test_repair:.1f}s", flush=True)

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

    for lbl in ("baseline (first-improvement, prod)", "best-improvement (test)"):
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
    print("  RESULTATS (ideal/nadir global partage NSGA-III + les deux variantes)")
    print(f"{'-'*92}")
    for metric, higher in (("HV", True), ("GD", False), ("IGD", False), ("Spacing", False)):
        arrow = "^" if higher else "v"
        print(f"\n  {metric} ({arrow})")
        for name, vals in groups.items():
            s = _stats(vals[metric])
            print(f"    {name:<32} mean={s['mean']:.6f}  std={s['std']:.6f}")

    print(f"\n{'-'*92}")
    print("  Comparaison APPARIEE (meme front pareto_X, meme seed)")
    print(f"{'-'*92}")
    for metric, higher in (("HV", True), ("GD", False), ("IGD", False), ("Spacing", False)):
        b_list = [compute_pareto_metrics(F, global_ideal, global_nadir)[metric]
                  for (_, F, _) in raw["baseline (first-improvement, prod)"] if F is not None and len(F) > 0]
        t_list = [compute_pareto_metrics(F, global_ideal, global_nadir)[metric]
                  for (_, F, _) in raw["best-improvement (test)"] if F is not None and len(F) > 0]
        diffs = [ (t - b) if higher else (b - t) for b, t in zip(b_list, t_list) ]
        n_better = sum(1 for d in diffs if d > 1e-9)
        n_worse  = sum(1 for d in diffs if d < -1e-9)
        n_same   = len(diffs) - n_better - n_worse
        print(f"  {metric:<8} best-improvement meilleur: {n_better}/{len(diffs)}  pire: {n_worse}/{len(diffs)}  egal: {n_same}/{len(diffs)}  (diffs={['%.4f' % d for d in diffs]})")

    print(f"\n{'-'*92}")
    print(f"  Mann-Whitney U (non apparie) : first-improvement vs best-improvement")
    print(f"{'-'*92}")
    for metric in ("HV", "GD", "IGD", "Spacing"):
        a, b = groups["baseline (first-improvement, prod)"][metric], groups["best-improvement (test)"][metric]
        if len(a) < 2 or len(b) < 2:
            print(f"  {metric:<8} : pas assez de runs valides pour un test")
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        sig = "significatif (p<0.05)" if p < 0.05 else "non significatif"
        print(f"  {metric:<8} U={u:.1f}  p={p:.6f}  -> {sig}")

    print(f"\n{'-'*92}")
    print("  Wilcoxon signed-rank APPARIE (meme front de depart par seed -- test correct pour ce design)")
    print(f"{'-'*92}")
    for metric, higher in (("HV", True), ("GD", False), ("IGD", False), ("Spacing", False)):
        b_list = [compute_pareto_metrics(F, global_ideal, global_nadir)[metric]
                  for (_, F, _) in raw["baseline (first-improvement, prod)"] if F is not None and len(F) > 0]
        t_list = [compute_pareto_metrics(F, global_ideal, global_nadir)[metric]
                  for (_, F, _) in raw["best-improvement (test)"] if F is not None and len(F) > 0]
        diffs = [ (t - b) if higher else (b - t) for b, t in zip(b_list, t_list) ]
        if len(diffs) < 6 or all(d == 0 for d in diffs):
            print(f"  {metric:<8} : pas assez de paires non-nulles pour un test")
            continue
        stat, p = wilcoxon(diffs, alternative="greater")
        sig = "significatif (p<0.05)" if p < 0.05 else "non significatif"
        print(f"  {metric:<8} stat={stat:.1f}  p={p:.6f}  -> best-improvement meilleur, {sig}")

    print(f"\n{'-'*92}")
    print(f"  Mann-Whitney U : best-improvement vs NSGA-III (l'objectif final)")
    print(f"{'-'*92}")
    for metric in ("HV", "GD", "IGD", "Spacing"):
        a, b = groups["best-improvement (test)"][metric], groups["NSGA-III (reference)"][metric]
        if len(a) < 2 or len(b) < 2:
            print(f"  {metric:<8} : pas assez de runs valides pour un test")
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        sig = "significatif (p<0.05)" if p < 0.05 else "non significatif"
        print(f"  {metric:<8} U={u:.1f}  p={p:.6f}  -> {sig}")

    print(f"\n{'-'*92}")
    print("  Temps moyen (recherche + reparation)")
    print(f"{'-'*92}")
    for lbl in ("baseline (first-improvement, prod)", "best-improvement (test)"):
        s = _stats(groups[lbl]["elapsed_s"])
        print(f"    {lbl:<32} mean={s['mean']:.1f}s")

    print(f"\n{'='*92}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test de reparation 2-opt best-improvement vs first-improvement — QINSGA-III")
    parser.add_argument("--instance", default="100",
                        choices=["3", "5", "15", "25", "30", "40", "100"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--gen",   type=int, default=300)
    parser.add_argument("--pop",   type=int, default=POP_SIZE)
    args = parser.parse_args()
    run_comparison(args.instance, args.seeds, args.gen, args.pop)
