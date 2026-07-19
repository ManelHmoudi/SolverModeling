"""
MaF1-3 problem definitions for NSGA-III validation.

Cheng, R., Li, M., Tian, Y., Zhang, X., Yang, S., Jin, Y., & Yao, X. (2017).
"A benchmark test suite for evolutionary many-objective optimization."
Complex & Intelligent Systems, 3(1), 67-81.

pymoo has no built-in MaF support, so MaF1-3 are implemented here as direct,
line-by-line translations of the reference MATLAB implementation in PlatEMO
(BIMK/PlatEMO, Problems/Multi-objective optimization/MaF/MaF{1,2,3}.m) — the
platform Cui et al. (2025) themselves used. Each class mirrors pymoo's own
DTLZ convention: _evaluate() computes the objectives, _calc_pareto_front(ref_dirs)
returns a reference point set for IGD (so validation/metrics/igd_metric.py
needs no changes to support them).

n_var = n_obj + 9 for all three (k=10 distance variables, same convention as
DTLZ2-6), so n_var values coincide numerically with DTLZ2-6's.

Termination — same protocol as DTLZ (validation/dtlz/dtlz_problems.py):
Tmax = 30000 evaluations (Cui et al. 2025, Table 2 — identical p/H/N/Tmax row
for "DTLZ 1-7" and "MaF 1-7"), n_gen = Tmax // pop_size.

MaF2's true front (see MaF2._calc_pareto_front) needs an internal, denser
Das-Dennis sample than the caller-supplied ref_dirs: PlatEMO's own GetOptimum
generates a uniform point set and then REJECTS most of it (only angles within
a narrow band survive), so reusing ref_dirs as-is would leave far too few
points to be a useful IGD reference set.
"""
import numpy as np
from pymoo.core.problem import Problem
from pymoo.util.ref_dirs import get_reference_directions

from validation.algorithms.nsga3_runner import get_run_config

PROBLEM_NAMES = ["MaF1", "MaF2", "MaF3"]

DEGENERATE_PROBLEMS = set()  # MaF1-3 all have an analytical front at any M

TMAX = 30000  # evaluation budget (Cui et al. 2025), shared with DTLZ


def _uniform_simplex_points(n_obj, n_partitions=12):
    """Points on the unit simplex (sum=1), i.e. PlatEMO's UniformPoint(N,M)."""
    return get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)


class MaF1(Problem):
    """Inverted DTLZ1 — linear front, oriented away from the origin."""

    def __init__(self, n_var, n_obj):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        M = self.n_obj
        n = x.shape[0]
        X_, X_M = x[:, :M - 1], x[:, M - 1:]
        g = np.sum((X_M - 0.5) ** 2, axis=1)
        ones_col = np.ones((n, 1))
        cp = np.fliplr(np.cumprod(np.hstack([ones_col, X_]), axis=1))
        rev = np.hstack([ones_col, 1 - X_[:, ::-1]])
        factor = cp * rev
        out["F"] = (1 + g)[:, None] - (1 + g)[:, None] * factor

    def _calc_pareto_front(self, ref_dirs=None):
        if ref_dirs is None:
            ref_dirs = _uniform_simplex_points(self.n_obj)
        return 1 - ref_dirs


class MaF2(Problem):
    """DTLZ2BZ — spherical front restricted to a narrow angular band, per-objective g."""

    def __init__(self, n_var, n_obj):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float)

    def _group_slices(self):
        M, D = self.n_obj, self.n_var
        k = D - M + 1
        nk = k // M
        for m in range(M):
            if m < M - 1:
                yield slice(m * nk, (m + 1) * nk)
            else:
                yield slice((M - 1) * nk, k)

    def _evaluate(self, x, out, *args, **kwargs):
        M = self.n_obj
        n = x.shape[0]
        X_, X_M = x[:, :M - 1], x[:, M - 1:]
        g = np.zeros((n, M))
        for m, seg in enumerate(self._group_slices()):
            g[:, m] = np.sum((X_M[:, seg] / 2 + 0.25 - 0.5) ** 2, axis=1)
        theta = X_ / 2 + 0.25
        ones_col = np.ones((n, 1))
        cp = np.fliplr(np.cumprod(np.hstack([ones_col, np.cos(theta * np.pi / 2)]), axis=1))
        rev = np.hstack([ones_col, np.sin(theta[:, ::-1] * np.pi / 2)])
        out["F"] = (1 + g) * cp * rev

    @staticmethod
    def _reconstruct_c(R):
        """Recover the per-point cos(theta) coordinates from simplex points R
        (PlatEMO MaF2.GetOptimum's recursive loop, direct translation).

        R(:,1)==0 at simplex boundary points causes an intentional 0-division
        here (mirrors the MATLAB original) — those rows come out NaN and are
        dropped by the angular-band filter in _calc_pareto_front, so it's
        silenced rather than left to print a RuntimeWarning on every run.
        """
        n, M = R.shape
        c = np.zeros((n, M - 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            for j in range(2, M + 1):
                t = M - j
                prod_term = np.prod(c[:, t + 1:M - 1], axis=1)
                temp = R[:, j - 1] / R[:, 0] * prod_term
                c[:, t] = np.sqrt(1.0 / (1.0 + temp ** 2))
        return c

    def _calc_pareto_front(self, ref_dirs=None):
        # Oversample: most simplex points get rejected by the angular-band filter below,
        # so reusing a caller-supplied low-density ref_dirs would starve the reference set.
        R = _uniform_simplex_points(self.n_obj, n_partitions=34)
        c = self._reconstruct_c(R)
        lo, hi = np.cos(3 * np.pi / 8), np.cos(np.pi / 8)
        valid = np.all((c >= lo) & (c <= hi), axis=1)
        c = c[valid]
        n = c.shape[0]
        ones_col = np.ones((n, 1))
        cp = np.fliplr(np.cumprod(np.hstack([ones_col, c]), axis=1))
        rev = np.hstack([ones_col, np.sqrt(1 - c[:, ::-1] ** 2)])
        return cp * rev


class MaF3(Problem):
    """Convex DTLZ3 — multimodal distance function (like DTLZ3), convex front."""

    def __init__(self, n_var, n_obj):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        M, D = self.n_obj, self.n_var
        n = x.shape[0]
        X_, X_M = x[:, :M - 1], x[:, M - 1:]
        k = D - M + 1
        g = 100 * (k + np.sum((X_M - 0.5) ** 2 - np.cos(20 * np.pi * (X_M - 0.5)), axis=1))
        ones_col = np.ones((n, 1))
        cp = np.fliplr(np.cumprod(np.hstack([ones_col, np.cos(X_ * np.pi / 2)]), axis=1))
        rev = np.hstack([ones_col, np.sin(X_[:, ::-1] * np.pi / 2)])
        f = (1 + g)[:, None] * cp * rev
        out["F"] = np.hstack([f[:, :M - 1] ** 4, f[:, M - 1:M] ** 2])

    def _calc_pareto_front(self, ref_dirs=None):
        if ref_dirs is None:
            ref_dirs = _uniform_simplex_points(self.n_obj)
        R = ref_dirs ** 2
        temp = np.sum(np.sqrt(R[:, :-1]), axis=1) + R[:, -1]
        R = R.copy()
        R[:, :-1] = R[:, :-1] / (temp ** 2)[:, None]
        R[:, -1] = R[:, -1] / temp
        return R


_CLASSES = {"MAF1": MaF1, "MAF2": MaF2, "MAF3": MaF3}


def get_problem(name: str, n_obj: int = 4):
    """Return (pymoo Problem instance, n_generations) for the given MaF name.

    Args:
        name  : one of "MaF1", "MaF2", "MaF3" (case-insensitive).
        n_obj : number of objectives (default 4). Supported: 3, 4.

    Returns:
        (problem, n_gen) where n_gen = TMAX // pop_size for this n_obj.
    """
    key = name.upper()
    if key not in _CLASSES:
        raise ValueError(f"Unknown problem '{name}'. Choose from {PROBLEM_NAMES}")
    n_var = n_obj + 9
    problem = _CLASSES[key](n_var=n_var, n_obj=n_obj)
    _, pop_size = get_run_config(n_obj)  # raises ValueError for unsupported n_obj
    n_gen = TMAX // pop_size
    return problem, n_gen
