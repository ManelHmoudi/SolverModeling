"""Pareto-front quality indicators (HV, GD, IGD, Spacing) on normalised objectives."""

import numpy as np
from pymoo.indicators.hv  import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.gd  import GD
from pymoo.util.ref_dirs  import get_reference_directions

# Denser grid than the optimizer's N_PARTITIONS=8 (165 dirs) — this gives 364 reference
# points which provides a finer approximation of the true Pareto front for GD/IGD.
# Using a denser measurement grid than the search grid is standard practice.
_METRIC_N_PARTITIONS = 12


def _normalise(F: np.ndarray):
    ideal = F.min(axis=0)
    nadir = F.max(axis=0)
    rng   = nadir - ideal
    rng[rng < 1e-10] = 1.0
    return (F - ideal) / rng, ideal, nadir


def _reference_set(n_obj: int, n_partitions: int = _METRIC_N_PARTITIONS) -> np.ndarray:
    return get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)


def _spacing(F_norm: np.ndarray) -> float:
    n = len(F_norm)
    if n < 2:
        return 0.0
    d_min = []
    for i in range(n):
        dists    = np.linalg.norm(F_norm - F_norm[i], axis=1)
        dists[i] = np.inf
        d_min.append(dists.min())
    d_min = np.array(d_min)
    d_bar = d_min.mean()
    return float(np.sqrt(np.sum((d_min - d_bar) ** 2) / max(n - 1, 1)))


def compute_pareto_metrics(
    F: np.ndarray,
    global_ideal: np.ndarray | None = None,
    global_nadir: np.ndarray | None = None,
) -> dict:
    """Compute HV, GD, IGD, Spacing for a Pareto front F.

    When global_ideal / global_nadir are supplied (global scale shared across
    all runs), HV/GD/IGD/Spacing are comparable between runs.
    Without them, each run is normalised to its own [0,1]^M — values are NOT
    cross-run comparable, though HV is still reported in [0, 1] (fraction of
    the ideal-to-nadir volume dominated; see ref_point normalisation below).
    The ideal/nadir reported in the dict always reflect the local run range
    (for informational display in the report).
    """
    if F is None or len(F) == 0:
        return {"HV": None, "GD": None, "IGD": None, "Spacing": None,
                "ideal": None, "nadir": None}

    F_arr = np.array(F, dtype=float)
    _, local_ideal, local_nadir = _normalise(F_arr)

    if global_ideal is not None and global_nadir is not None:
        g_ideal = np.asarray(global_ideal, dtype=float)
        g_nadir = np.asarray(global_nadir, dtype=float)
        rng = g_nadir - g_ideal
        rng[rng < 1e-10] = 1.0
        F_norm = (F_arr - g_ideal) / rng
    else:
        F_norm = (F_arr - local_ideal) / np.where(
            local_nadir - local_ideal > 1e-10, local_nadir - local_ideal, 1.0
        )

    n_obj   = F_norm.shape[1]
    ref_set = _reference_set(n_obj, n_partitions=_METRIC_N_PARTITIONS)

    # ref_point sits 10% beyond the nadir (standard HV margin), so the raw
    # pymoo HV is bounded by ref_point.prod() = 1.1**n_obj (~1.46 for 4
    # objectives), NOT by 1. Dividing by that bound keeps HV in [0, 1] so
    # it reads as "fraction of the ideal-to-nadir volume dominated" and
    # stays comparable across n_obj values.
    ref_point   = np.ones(n_obj) * 1.1
    hv_raw      = float(HV(ref_point=ref_point)(F_norm))
    hv          = hv_raw / float(ref_point.prod())
    gd  = float(GD(pf=ref_set)(F_norm))
    igd = float(IGD(pf=ref_set)(F_norm))
    sp  = _spacing(F_norm)

    return {
        "HV":      round(hv,  6),
        "GD":      round(gd,  6),
        "IGD":     round(igd, 6),
        "Spacing": round(sp,  6),
        "ideal": {f"f{i+1}": round(float(local_ideal[i]), 4) for i in range(n_obj)},
        "nadir":  {f"f{i+1}": round(float(local_nadir[i]), 4) for i in range(n_obj)},
    }
