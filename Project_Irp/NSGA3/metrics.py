"""Pareto-front quality indicators (HV, GD, IGD, Spacing) on normalised objectives."""

import numpy as np
from pymoo.indicators.hv  import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.gd  import GD
from pymoo.util.ref_dirs  import get_reference_directions


def _normalise(F: np.ndarray):
    ideal = F.min(axis=0)
    nadir = F.max(axis=0)
    rng   = nadir - ideal
    rng[rng < 1e-10] = 1.0
    return (F - ideal) / rng, ideal, nadir


def _reference_set(n_obj: int, n_partitions: int = 12) -> np.ndarray:
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


def compute_pareto_metrics(F: np.ndarray) -> dict:
    if F is None or len(F) == 0:
        return {"HV": None, "GD": None, "IGD": None, "Spacing": None,
                "ideal": None, "nadir": None}

    F_norm, ideal, nadir = _normalise(np.array(F, dtype=float))
    n_obj   = F_norm.shape[1]
    ref_set = _reference_set(n_obj, n_partitions=12)

    ref_point = np.ones(n_obj) * 1.1
    hv  = float(HV(ref_point=ref_point)(F_norm))
    gd  = float(GD(pf=ref_set)(F_norm))
    igd = float(IGD(pf=ref_set)(F_norm))
    sp  = _spacing(F_norm)

    return {
        "HV":      round(hv,  6),
        "GD":      round(gd,  6),
        "IGD":     round(igd, 6),
        "Spacing": round(sp,  6),
        "ideal": {f"f{i+1}": round(float(ideal[i]), 4) for i in range(n_obj)},
        "nadir":  {f"f{i+1}": round(float(nadir[i]), 4) for i in range(n_obj)},
    }
