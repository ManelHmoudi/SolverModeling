"""
IGD (Inverted Generational Distance) calculator using pymoo.

compute_igd: calculates IGD for a single run's approximation front.
igd_statistics: aggregates (best, median, worst) over multiple runs.
"""
import numpy as np
from pymoo.indicators.igd import IGD
from pymoo.util.ref_dirs import get_reference_directions


def _get_true_front(problem):
    """
    Retrieve or approximate the true Pareto front for a DTLZ problem.
    pymoo's DTLZ pareto_front() accepts ref_dirs for 4-objective problems.
    """
    ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)
    try:
        pf = problem.pareto_front(ref_dirs=ref_dirs)
    except TypeError:
        pf = problem.pareto_front()
    if pf is None:
        raise ValueError(
            f"Could not retrieve true Pareto front for {type(problem).__name__}. "
            "Ensure pymoo version supports this problem's pareto_front()."
        )
    return pf


def compute_igd(problem, approx_front: np.ndarray) -> float:
    """
    Compute IGD between approx_front and the problem's true Pareto front.

    Args:
        problem: pymoo Problem instance (must have pareto_front() method).
        approx_front: np.ndarray of shape (n_solutions, n_obj) — objective values.

    Returns:
        IGD value (float). Lower is better.
    """
    true_front = _get_true_front(problem)
    indicator = IGD(true_front)
    return float(indicator(approx_front))


def igd_statistics(igd_values: list) -> dict:
    """
    Compute best, median, worst IGD over multiple independent runs.

    Args:
        igd_values: list of IGD floats, one per run.

    Returns:
        {"best": float, "median": float, "worst": float}
    """
    arr = np.array(igd_values, dtype=float)
    return {
        "best":   float(np.min(arr)),
        "median": float(np.median(arr)),
        "worst":  float(np.max(arr)),
    }
