"""
Shared validation engine — runs NSGA-III on a benchmark suite (DTLZ, MaF, ...)
and saves per-run / summary IGD CSVs.

This module is suite-agnostic: it only needs a "problems module" exposing
PROBLEM_NAMES, get_problem(name, n_obj) -> (problem, n_gen), and
DEGENERATE_PROBLEMS (a set of problem names whose IGD cannot be scored at
some n_obj values, skipped rather than crashing). validation/dtlz/ and
validation/maf/ each have their own thin main_validation.py CLI script that
calls validate() here with their own problems module and results directory —
this keeps the run/save/print logic in one place instead of duplicated per
suite.
"""
import csv
import os
import time

from validation.algorithms.nsga3.runner import run_experiment as _default_run_experiment
from validation.algorithms.seeds import SEEDS as _SEEDS
from validation.metrics.igd_metric import compute_igd, igd_statistics


def _save_run_csv(results_dir: str, problem_name: str, igd_values: list, n_obj: int) -> str:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"igd_{problem_name}_M{n_obj}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "seed", "igd"])
        for i, igd in enumerate(igd_values):
            writer.writerow([i + 1, _SEEDS[i], f"{igd:.6f}"])
    return path


def _save_summary_csv(results_dir: str, summary_rows: list, n_obj: int) -> str:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"summary_M{n_obj}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Problem", "IGD_Best", "IGD_Median", "IGD_Worst", "IGD_Mean", "IGD_Std"])
        for row in summary_rows:
            writer.writerow([
                row["problem"],
                f"{row['best']:.6f}",
                f"{row['median']:.6f}",
                f"{row['worst']:.6f}",
                f"{row['mean']:.6f}",
                f"{row['std']:.6f}",
            ])
    return path


def _print_table(suite_label: str, summary_rows: list, n_obj: int, algorithm_label: str):
    header = f"{'Problem':<10} {'IGD Best':>12} {'IGD Median':>12} {'IGD Worst':>12} {'IGD Mean':>12} {'IGD Std':>12}"
    sep = "-" * len(header)
    print("\n" + sep)
    print(f"  {algorithm_label} on {suite_label} ({n_obj} objectives) - Cui et al. 2025 protocol")
    print(sep)
    print(header)
    print(sep)
    for row in summary_rows:
        print(
            f"{row['problem']:<10} {row['best']:>12.6f} {row['median']:>12.6f} "
            f"{row['worst']:>12.6f} {row['mean']:>12.6f} {row['std']:>12.6f}"
        )
    print(sep + "\n")


def validate(problems_module, results_dir: str, suite_label: str, n_runs: int = 30, n_obj: int = 4,
             run_experiment_fn=_default_run_experiment, algorithm_label: str = "NSGA-III"):
    """Run many-objective validation on every problem in problems_module.PROBLEM_NAMES.

    Args:
        problems_module : module exposing PROBLEM_NAMES, get_problem(name, n_obj),
                           and DEGENERATE_PROBLEMS (set of names to skip when their
                           IGD can't be scored at this n_obj).
        results_dir      : directory to write igd_*.csv / summary_M{n}.csv into.
        suite_label       : display name for progress printing (e.g. "DTLZ1-7").
        n_runs            : number of independent runs per problem (1-30).
        n_obj             : number of objectives.
        run_experiment_fn : callable(problem_name, problem, n_gen, n_runs) -> list of fronts;
                             defaults to the classic NSGA-III runner.
        algorithm_label   : display name for the printed results table (e.g. "NSGA-III", "QINSGA3").

    Returns:
        List of summary dicts with keys: problem, best, median, worst, mean, std.
    """
    summary_rows = []

    for name in problems_module.PROBLEM_NAMES:
        print(f"\n{'='*50}")
        print(f"Problem: {name}  ({n_runs} runs, {n_obj} objectives)")
        print(f"{'='*50}")

        if name in problems_module.DEGENERATE_PROBLEMS and n_obj != 3:
            print(
                f"  Skipping {name} at M={n_obj}: no analytical Pareto front "
                f"available for this problem beyond M=3."
            )
            continue

        problem, n_gen = problems_module.get_problem(name, n_obj=n_obj)
        print(f"  n_var={problem.n_var}, n_obj={problem.n_obj}, n_gen={n_gen}")

        t0 = time.time()
        fronts = run_experiment_fn(name, problem, n_gen, n_runs)
        elapsed = time.time() - t0
        print(f"  {name}: {n_runs} runs completed in {elapsed:.1f}s", flush=True)

        igd_values = [compute_igd(problem, front) for front in fronts]
        stats = igd_statistics(igd_values)

        csv_path = _save_run_csv(results_dir, name, igd_values, n_obj)
        print(f"  Saved per-run IGD -> {csv_path}")

        summary_rows.append({
            "problem": name,
            "best":    stats["best"],
            "median":  stats["median"],
            "worst":   stats["worst"],
            "mean":    stats["mean"],
            "std":     stats["std"],
        })

    summary_path = _save_summary_csv(results_dir, summary_rows, n_obj)
    print(f"\nSummary CSV -> {summary_path}")

    _print_table(suite_label, summary_rows, n_obj, algorithm_label)
    return summary_rows
