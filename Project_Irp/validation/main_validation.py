"""
Main validation script: runs NSGA-III on DTLZ1-4 (Deb & Jain 2014 protocol).

Usage:
    python -m validation.main_validation                      # 20 runs, 4 objectives
    python -m validation.main_validation --runs 3             # 3 runs, 4 objectives
    python -m validation.main_validation --n_obj 3            # 20 runs, 3 objectives
    python -m validation.main_validation --n_obj 3 --runs 20  # full 3-obj validation
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation.benchmarks.dtlz_problems import PROBLEM_NAMES, get_problem
from validation.algorithms.nsga3_runner import run_experiment, _SEEDS
from validation.metrics.igd_metric import compute_igd, igd_statistics

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _save_run_csv(problem_name: str, igd_values: list, n_obj: int) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"igd_{problem_name}_M{n_obj}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "seed", "igd"])
        for i, igd in enumerate(igd_values):
            writer.writerow([i + 1, _SEEDS[i], f"{igd:.6f}"])
    return path


def _save_summary_csv(summary_rows: list, n_obj: int) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"summary_M{n_obj}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Problem", "IGD_Best", "IGD_Median", "IGD_Worst"])
        for row in summary_rows:
            writer.writerow([
                row["problem"],
                f"{row['best']:.6f}",
                f"{row['median']:.6f}",
                f"{row['worst']:.6f}",
            ])
    return path


def _print_table(summary_rows: list, n_obj: int):
    header = f"{'Problem':<10} {'IGD Best':>12} {'IGD Median':>12} {'IGD Worst':>12}"
    sep = "-" * len(header)
    print("\n" + sep)
    print(f"  NSGA-III on DTLZ1-4 ({n_obj} objectives) - Deb & Jain 2014")
    print(sep)
    print(header)
    print(sep)
    for row in summary_rows:
        print(
            f"{row['problem']:<10} {row['best']:>12.6f} {row['median']:>12.6f} {row['worst']:>12.6f}"
        )
    print(sep + "\n")


def validate(n_runs: int = 20, n_obj: int = 4):
    """Run NSGA-III validation on DTLZ1-4.

    Args:
        n_runs : number of independent runs per problem (1-20).
        n_obj  : number of objectives (3 or 4).

    Returns:
        List of summary dicts with keys: problem, best, median, worst.
    """
    summary_rows = []

    for name in PROBLEM_NAMES:
        print(f"\n{'='*50}")
        print(f"Problem: {name}  ({n_runs} runs, {n_obj} objectives)")
        print(f"{'='*50}")

        problem, n_gen = get_problem(name, n_obj=n_obj)
        print(f"  n_var={problem.n_var}, n_obj={problem.n_obj}, n_gen={n_gen}")

        t0 = time.time()
        fronts = run_experiment(name, problem, n_gen, n_runs)
        elapsed = time.time() - t0
        print(f"  {name}: {n_runs} runs completed in {elapsed:.1f}s", flush=True)

        igd_values = [compute_igd(problem, front) for front in fronts]
        stats = igd_statistics(igd_values)

        csv_path = _save_run_csv(name, igd_values, n_obj)
        print(f"  Saved per-run IGD -> {csv_path}")

        summary_rows.append({
            "problem": name,
            "best":    stats["best"],
            "median":  stats["median"],
            "worst":   stats["worst"],
        })

    summary_path = _save_summary_csv(summary_rows, n_obj)
    print(f"\nSummary CSV -> {summary_path}")

    _print_table(summary_rows, n_obj)
    return summary_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate NSGA-III on DTLZ1-4 (Deb & Jain 2014)"
    )
    parser.add_argument(
        "--runs", type=int, default=20,
        help="Number of independent runs per problem (default: 20)"
    )
    parser.add_argument(
        "--n_obj", type=int, default=4, choices=[3, 4],
        help="Number of objectives (default: 4)"
    )
    args = parser.parse_args()

    if args.runs < 1 or args.runs > 20:
        print("--runs must be between 1 and 20")
        sys.exit(1)

    validate(n_runs=args.runs, n_obj=args.n_obj)
