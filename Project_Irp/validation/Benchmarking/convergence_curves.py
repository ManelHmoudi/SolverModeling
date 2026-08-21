"""IGD convergence curves (IGD vs. evaluations), matching Cui et al. (2025)
Fig. 3's exact style: one subplot per (problem, M) combination, one line per
algorithm, x-axis in evaluations (FE, thousands), y-axis IGD, mean +/- std
shaded band across independent seeds.

Unlike the final-value comparison this project's other benchmark tooling
reports (Best/Median/Worst/Mean over 30 runs, one IGD number per run), this
tracks IGD *during* the search -- at every generation, not just at the end
-- to show HOW each algorithm gets there, not just where it ends up.

NSGA-III and MOEA/D (ConstrainedMOEAD) both run via pymoo's own minimize(),
which supports save_history=True natively: res.history[i] is a per-
generation Algorithm snapshot with .opt (current non-dominated front) and
.evaluator.n_eval (real cumulative evaluation count) already tracked by
pymoo itself. QI-NSGA-III has no pymoo Algorithm object (a separate custom
generational loop, Validation/Benchmarking/algorithms/qinsga3/core.py) --
it uses that module's own callback= parameter instead (added alongside
this script specifically to make this tracking possible; see that
function's own docstring), invoked once per generation with an equivalent
(gen, n_eval, F_non_dominated) tuple.

Each seed's trajectory is interpolated onto a common evaluation-count grid
(np.interp, step-function hold of the last known IGD) before averaging
across seeds -- individual runs land on different exact n_eval checkpoints
(QI-NSGA-III's own eval-caching makes its per-generation n_eval increment
data-dependent, not a fixed multiple of generation number), so averaging
raw per-generation values across seeds without first aligning them onto a
shared x-axis would silently mix IGD-at-generation-g from one seed with
IGD-at-a-different-real-evaluation-count from another.

Usage:
    python -m Validation.Benchmarking.convergence_curves --gen 20 --seeds 42          # smoke
    python -m Validation.Benchmarking.convergence_curves --seeds 42 137 271 491 613 733 857 --gen 326
"""
from __future__ import annotations

import argparse
import os
import sys

MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(MODULE_DIR))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymoo.algorithms.moo.nsga3    import NSGA3
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm   import PM
from pymoo.operators.sampling.rnd  import FloatRandomSampling
from pymoo.optimize                import minimize
from pymoo.termination             import get_termination
from pymoo.util.ref_dirs           import get_reference_directions
from pymoo.indicators.igd          import IGD

from Validation.Benchmarking.dtlz.dtlz_problems import get_problem
from Validation.Benchmarking.algorithms.nsga3.runner import get_run_config, _N_OBJ_TO_P
from Validation.Benchmarking.algorithms.qinsga3.core import run_qinsga3_generic
from Validation.Benchmarking.metrics.igd_metric import _get_true_front, _P_STAR_PARTITIONS
from Solvers.MOEAD._constrained_moead        import ConstrainedMOEAD
from Solvers.MOEAD._normalized_decomposition import NormalizedTchebycheff

PROBLEMS      = ["DTLZ7", "MaF5"]
M_VALUES      = [3, 4]
DEFAULT_SEEDS = [42, 137, 271, 491, 613, 733, 857]
N_GRID_POINTS = 200

QI_ALPHA_MAX = 0.10 * np.pi
QI_ALPHA_MIN = 0.001 * np.pi
QI_MIGRATION_PERIOD = 5
QI_N_MIGRATE = 20
QI_NOISE_SCALE = 0.0
QI_ESCAPE_PROB_FACTOR = 2.0 * 0.15


def _get_problem_by_name(name: str, n_obj: int):
    if name.upper().startswith("DTLZ"):
        return get_problem(name, n_obj=n_obj)
    else:
        from Validation.Benchmarking.maf.maf_problems import get_problem as get_maf_problem
        return get_maf_problem(name, n_obj=n_obj)


def _track_nsga3(problem, ref_dirs, pop_size, n_gen, seed, true_front):
    algorithm = NSGA3(
        pop_size=pop_size, ref_dirs=ref_dirs, sampling=FloatRandomSampling(),
        crossover=SBX(prob=1.0, eta=20), mutation=PM(prob=1.0 / problem.n_var, eta=20),
    )
    res = minimize(problem, algorithm, get_termination("n_gen", n_gen),
                    seed=seed, verbose=False, save_history=True)
    ind = IGD(true_front)
    n_evals, igds = [], []
    for snap in res.history:
        F = snap.opt.get("F")
        if F is None or len(F) == 0:
            continue
        n_evals.append(snap.evaluator.n_eval)
        igds.append(float(ind(F)))
    return np.array(n_evals), np.array(igds)


def _track_moead(problem, ref_dirs, seed, true_front):
    algorithm = ConstrainedMOEAD(
        ref_dirs=ref_dirs, decomposition=NormalizedTchebycheff(),
        sampling=FloatRandomSampling(), crossover=SBX(prob=1.0, eta=20),
        mutation=PM(prob=1.0 / problem.n_var, eta=20),
    )
    n_gen = _current_n_gen["value"]
    res = minimize(problem, algorithm, get_termination("n_gen", n_gen),
                    seed=seed, verbose=False, save_history=True)
    ind = IGD(true_front)
    n_evals, igds = [], []
    for snap in res.history:
        F = snap.opt.get("F")
        if F is None or len(F) == 0:
            continue
        n_evals.append(snap.evaluator.n_eval)
        igds.append(float(ind(F)))
    return np.array(n_evals), np.array(igds)


def _track_qinsga3(problem, ref_dirs, pop_size, n_gen, seed, true_front):
    ind = IGD(true_front)
    n_evals, igds = [], []

    def cb(gen, n_eval, F):
        if len(F) == 0:
            return
        n_evals.append(n_eval)
        igds.append(float(ind(F)))

    p_mut = 1.0 / problem.n_var
    escape_prob = QI_ESCAPE_PROB_FACTOR / problem.n_var
    run_qinsga3_generic(
        problem, ref_dirs, pop_size, max_gen=n_gen,
        alpha_max=QI_ALPHA_MAX, alpha_min=QI_ALPHA_MIN,
        p_cross=1.0, eta_cross=20.0, p_mut=p_mut, eta_mut=20.0,
        migration_period=QI_MIGRATION_PERIOD, n_migrate=QI_N_MIGRATE,
        seed=seed, rotation_type="tanh", noise_scale=QI_NOISE_SCALE,
        escape_prob=escape_prob, callback=cb,
    )
    return np.array(n_evals), np.array(igds)


_current_n_gen = {"value": None}  # set per (problem, M) in main loop, read by _track_moead


def _interp_on_grid(n_evals, igds, grid):
    """Step-hold interpolation: IGD at grid point x is the last known IGD at
    or before x (never look ahead), clamped to the first/last observed
    value outside the trajectory's own range."""
    if len(n_evals) == 0:
        return np.full_like(grid, np.nan, dtype=float)
    order = np.argsort(n_evals)
    ne, ig = n_evals[order], igds[order]
    idx = np.searchsorted(ne, grid, side="right") - 1
    idx = np.clip(idx, 0, len(ne) - 1)
    out = ig[idx]
    out[grid < ne[0]] = ig[0]
    return out


def run_convergence_campaign(seeds: list[int], n_gen_m3: int, n_gen_m4: int, output_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    panel_i = 0

    for m in M_VALUES:
        n_gen = n_gen_m3 if m == 3 else n_gen_m4
        _current_n_gen["value"] = n_gen
        ref_dirs, pop_size = get_run_config(m)
        tmax_grid = np.linspace(0, 30000, N_GRID_POINTS)

        for prob_name in PROBLEMS:
            ax = axes.flat[panel_i]
            problem, _ = _get_problem_by_name(prob_name, m)

            # DTLZ5/6/7 have a degenerate/disconnected true Pareto front; pymoo
            # only ships a precomputed reference front for them at M=3 (raises
            # "Not implemented yet." beyond that) -- the SAME structural limit
            # already documented and skipped in dtlz_problems.py's own
            # DEGENERATE_PROBLEMS handling for the rest of this project's
            # benchmark suite. Applied here identically rather than crashing.
            try:
                true_front = _get_true_front(problem)
            except Exception as e:
                print(f"=== {prob_name} M={m}: SKIPPED (no true Pareto front available: {e}) ===", flush=True)
                ax.text(0.5, 0.5, f"{prob_name}, M={m}\npas de front de reference\ndisponible (pymoo)",
                        ha="center", va="center", transform=ax.transAxes, fontsize=10, color="#888")
                ax.set_title(f"{panel_labels[panel_i]} {prob_name}, M={m}", fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
                panel_i += 1
                continue

            print(f"=== {prob_name} M={m} (n_gen={n_gen}, pop={pop_size}) ===", flush=True)

            curves = {"NSGA-III": [], "QI-NSGA-III": [], "MOEA/D": []}
            for seed in seeds:
                print(f"  seed={seed}", flush=True)
                ne, ig = _track_nsga3(problem, ref_dirs, pop_size, n_gen, seed, true_front)
                curves["NSGA-III"].append(_interp_on_grid(ne, ig, tmax_grid))

                ne, ig = _track_qinsga3(problem, ref_dirs, pop_size, n_gen, seed, true_front)
                curves["QI-NSGA-III"].append(_interp_on_grid(ne, ig, tmax_grid))

                ne, ig = _track_moead(problem, ref_dirs, seed, true_front)
                curves["MOEA/D"].append(_interp_on_grid(ne, ig, tmax_grid))

            colors = {"NSGA-III": "#1f77b4", "QI-NSGA-III": "#d62728", "MOEA/D": "#2ca02c"}
            for algo, series_list in curves.items():
                arr = np.array(series_list)
                mean = np.nanmean(arr, axis=0)
                std  = np.nanstd(arr, axis=0)
                ax.plot(tmax_grid / 1000, mean, label=algo, color=colors[algo], linewidth=1.6)
                ax.fill_between(tmax_grid / 1000, mean - std, mean + std, color=colors[algo], alpha=0.15)

            ax.set_xlabel("FE*1000")
            ax.set_ylabel("IGD")
            ax.set_title(f"{panel_labels[panel_i]} {prob_name}, M={m}", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)
            panel_i += 1

    fig.suptitle(
        f"Convergence de l'IGD (moyenne +/- ecart-type, {len(seeds)} seeds) "
        f"-- NSGA-III vs QI-NSGA-III vs MOEA/D",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    print(f"\nSaved -> {output_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IGD convergence curves (DTLZ7/MaF5, M=3/4)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--gen", type=int, default=None,
                        help="Override BOTH M=3 and M=4 generation counts (smoke testing). "
                             "Default: Cui et al. (2025) Table 2 values (326 for M=3, 250 for M=4).")
    args = parser.parse_args()

    n_gen_m3 = args.gen if args.gen is not None else 326
    n_gen_m4 = args.gen if args.gen is not None else 250

    out = os.path.join(MODULE_DIR, "convergence_curves.png")
    run_convergence_campaign(args.seeds, n_gen_m3, n_gen_m4, out)
