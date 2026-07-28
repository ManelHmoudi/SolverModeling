"""
MaF validation CLI — runs NSGA-III/QINSGA3 on MaF1-7 (Cui et al. 2025 protocol, M=3/M=4).

Usage:
    python -m Validation.Benchmarking.maf.main_validation                        # NSGA-III, 30 runs, 4 objectives
    python -m Validation.Benchmarking.maf.main_validation --algorithm qinsga3    # QINSGA3, 30 runs, 4 objectives
    python -m Validation.Benchmarking.maf.main_validation --runs 3               # 3 runs, 4 objectives
    python -m Validation.Benchmarking.maf.main_validation --n_obj 3              # 30 runs, 3 objectives
    python -m Validation.Benchmarking.maf.main_validation --n_obj 3 --runs 30    # full 3-obj validation
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from Validation.Benchmarking.maf import maf_problems
from Validation.Benchmarking.engine import validate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate NSGA-III/QINSGA3 on MaF1-7 (Cui et al. 2025 protocol, M=3/M=4)"
    )
    parser.add_argument(
        "--runs", type=int, default=30,
        help="Number of independent runs per problem (default: 30)"
    )
    parser.add_argument(
        "--n_obj", type=int, default=4, choices=[3, 4],
        help="Number of objectives (default: 4)"
    )
    parser.add_argument(
        "--algorithm", default="nsga3", choices=["nsga3", "qinsga3"],
        help="Which algorithm to validate (default: nsga3)"
    )
    args = parser.parse_args()

    if args.runs < 1 or args.runs > 30:
        print("--runs must be between 1 and 30")
        sys.exit(1)

    if args.algorithm == "qinsga3":
        from Validation.Benchmarking.algorithms.qinsga3.runner import run_experiment
        algorithm_label = "QINSGA3"
    else:
        from Validation.Benchmarking.algorithms.nsga3.runner import run_experiment
        algorithm_label = "NSGA-III"

    RESULTS_DIR_ALGO = os.path.join(os.path.dirname(__file__), "results", args.algorithm)

    validate(
        problems_module=maf_problems,
        results_dir=RESULTS_DIR_ALGO,
        suite_label="MaF1-7",
        n_runs=args.runs,
        n_obj=args.n_obj,
        run_experiment_fn=run_experiment,
        algorithm_label=algorithm_label,
    )
