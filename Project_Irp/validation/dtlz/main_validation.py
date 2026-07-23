"""
DTLZ validation CLI — runs NSGA-III on DTLZ1-7 (Cui et al. 2025 protocol, M=3/M=4).

Usage:
    python -m validation.dtlz.main_validation                      # 30 runs, 4 objectives
    python -m validation.dtlz.main_validation --runs 3             # 3 runs, 4 objectives
    python -m validation.dtlz.main_validation --n_obj 3            # 30 runs, 3 objectives
    python -m validation.dtlz.main_validation --n_obj 3 --runs 30  # full 3-obj validation
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from validation.dtlz import dtlz_problems
from validation.engine import validate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "nsga3")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate NSGA-III on DTLZ1-7 (Cui et al. 2025 protocol, M=3/M=4)"
    )
    parser.add_argument(
        "--runs", type=int, default=30,
        help="Number of independent runs per problem (default: 30)"
    )
    parser.add_argument(
        "--n_obj", type=int, default=4, choices=[3, 4],
        help="Number of objectives (default: 4)"
    )
    args = parser.parse_args()

    if args.runs < 1 or args.runs > 30:
        print("--runs must be between 1 and 30")
        sys.exit(1)

    validate(
        problems_module=dtlz_problems,
        results_dir=RESULTS_DIR,
        suite_label="DTLZ1-7",
        n_runs=args.runs,
        n_obj=args.n_obj,
    )
