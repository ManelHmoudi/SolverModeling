"""
Pareto-front quality indicators for the NSGA-III IRP solver.

Metrics computed (all on normalised objectives in [0,1]^4):

  HV      – Hypervolume dominated by the front.
             Reference point = (1.1, 1.1, 1.1, 1.1).  Higher is better.

  GD      – Generational Distance: mean distance from each obtained solution
             to the nearest point on the Das-Dennis reference set (uniform
             hyperplane in normalised space).  Lower is better.

  IGD     – Inverted GD: mean distance from each reference-set point to the
             nearest obtained solution.  Combines convergence + diversity.
             Lower is better.

  Spacing – Standard deviation of nearest-neighbour distances between
             obtained solutions (normalised space).  Lower = more uniform.
"""

import numpy as np
from pymoo.indicators.hv  import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.gd  import GD
from pymoo.util.ref_dirs  import get_reference_directions


def _normalise(F: np.ndarray):
    """Normalise objective matrix to [0,1]^n_obj using ideal/nadir of F."""
    ideal = F.min(axis=0)
    nadir = F.max(axis=0)
    rng   = nadir - ideal
    rng[rng < 1e-10] = 1.0          # avoid /0 when all solutions tie on an objective
    return (F - ideal) / rng, ideal, nadir


def _reference_set(n_obj: int, n_partitions: int = 12) -> np.ndarray:
    """
    Uniform Das-Dennis reference directions on the unit simplex in R^n_obj.
    Used as a surrogate 'true' Pareto front in normalised space.
    """
    return get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)


def _spacing(F_norm: np.ndarray) -> float:
    """
    Spacing metric S: std-dev of nearest-neighbour distances.
    Measures uniformity of the distribution along the front.
    S = 0 means perfectly equidistant solutions.
    """
    n = len(F_norm)
    if n < 2:
        return 0.0
    d_min = []
    for i in range(n):
        dists    = np.linalg.norm(F_norm - F_norm[i], axis=1)
        dists[i] = np.inf
        d_min.append(dists.min())
    d_min  = np.array(d_min)
    d_bar  = d_min.mean()
    return float(np.sqrt(np.sum((d_min - d_bar) ** 2) / max(n - 1, 1)))


def compute_pareto_metrics(F: np.ndarray) -> dict:
    """
    Parameters
    ----------
    F : np.ndarray, shape (n_solutions, 4)
        Raw (un-normalised) objective values from the NSGA-III Pareto front.

    Returns
    -------
    dict with keys: HV, GD, IGD, Spacing  (all floats, rounded to 6 dp)
    and   ideal / nadir arrays for display.
    """
    if F is None or len(F) == 0:
        return {"HV": None, "GD": None, "IGD": None, "Spacing": None,
                "ideal": None, "nadir": None}

    F_norm, ideal, nadir = _normalise(np.array(F, dtype=float))
    n_obj = F_norm.shape[1]

    # Reference set on unit simplex (Das-Dennis, uniform)
    ref_set = _reference_set(n_obj, n_partitions=12)

    # HV – reference point slightly beyond normalised nadir
    ref_point = np.ones(n_obj) * 1.1
    hv  = float(HV(ref_point=ref_point)(F_norm))

    # GD / IGD relative to Das-Dennis reference set
    gd  = float(GD (pf=ref_set)(F_norm))
    igd = float(IGD(pf=ref_set)(F_norm))

    # Spacing
    sp = _spacing(F_norm)

    return {
        "HV":      round(hv,  6),
        "GD":      round(gd,  6),
        "IGD":     round(igd, 6),
        "Spacing": round(sp,  6),
        "ideal": {f"f{i+1}": round(float(ideal[i]), 4) for i in range(n_obj)},
        "nadir":  {f"f{i+1}": round(float(nadir[i]), 4) for i in range(n_obj)},
    }