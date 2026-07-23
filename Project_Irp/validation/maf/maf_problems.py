"""
MaF1-7 problem definitions for NSGA-III validation.

Cheng, R., Li, M., Tian, Y., Zhang, X., Yang, S., Jin, Y., & Yao, X. (2017).
"A benchmark test suite for evolutionary many-objective optimization."
Complex & Intelligent Systems, 3(1), 67-81.

pymoo has no built-in MaF support, so MaF1-7 are implemented here as direct,
line-by-line translations of the reference MATLAB implementation in PlatEMO
(BIMK/PlatEMO, Problems/Multi-objective optimization/MaF/MaF{1..7}.m) — the
platform Cui et al. (2025) themselves used. Each class mirrors pymoo's own
DTLZ convention: _evaluate() computes the objectives, _calc_pareto_front(ref_dirs)
returns a reference point set for IGD (so validation/metrics/igd_metric.py
needs no changes to support them).

n_var = n_obj + k - 1, k=10 (MaF1-6) or k=20 (MaF7, same convention as DTLZ7).

Termination — same protocol as DTLZ (validation/dtlz/dtlz_problems.py):
Tmax = 30000 evaluations (Cui et al. 2025, Table 2 — identical p/H/N/Tmax row
for "DTLZ 1-7" and "MaF 1-7"), n_gen = Tmax // pop_size.

Unlike this project's DTLZ5-7 (whose true front is only computable at M=3
because pymoo's own reference-front code is capped there), every MaF1-7 true
front below is a closed-form function of M — PlatEMO's GetOptimum for MaF6/7
is parameterised by M throughout — so all seven are scored at both M=3 and
M=4 with no DEGENERATE_PROBLEMS exclusion needed.

MaF2's true front (see MaF2._calc_pareto_front) needs an internal, denser
Das-Dennis sample than the caller-supplied ref_dirs: PlatEMO's own GetOptimum
generates a uniform point set and then REJECTS most of it (only angles within
a narrow band survive), so reusing ref_dirs as-is would leave far too few
points to be a useful IGD reference set. MaF6 and MaF7 similarly ignore the
caller-supplied (M-dimensional) ref_dirs and sample their own front-intrinsic
parameter space instead (MaF6: a fixed 2D simplex regardless of M; MaF7: a
dense (M-1)-D grid) — see each class for details.
"""
import numpy as np
from pymoo.core.problem import Problem
from pymoo.util.ref_dirs import get_reference_directions

from validation.algorithms.nsga3.runner import get_run_config

PROBLEM_NAMES = ["MaF1", "MaF2", "MaF3", "MaF4", "MaF5", "MaF6", "MaF7"]

DEGENERATE_PROBLEMS = set()  # MaF1-7 all have an analytical front at any M

TMAX = 30000  # evaluation budget (Cui et al. 2025), shared with DTLZ

_K = {  # distance-variable count k, n_var = n_obj + k - 1 (PlatEMO Setting())
    "MAF1": 10, "MAF2": 10, "MAF3": 10, "MAF4": 10, "MAF5": 10, "MAF6": 10,
    "MAF7": 20,
}


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

    # Oversample partition counts tuned so the post-filter point count lands
    # close to Cui et al. (2025)'s P*=10,000 IGD reference budget:
    #   n_obj=3 -> p=324 -> ~9,980 valid points; n_obj=4 -> p=89 -> ~9,874 valid points
    # (~19% and ~8% of oversampled points survive the angular-band filter, respectively).
    _OVERSAMPLE_PARTITIONS = {3: 324, 4: 89}

    def _calc_pareto_front(self, ref_dirs=None):
        # Oversample: most simplex points get rejected by the angular-band filter below,
        # so reusing a caller-supplied low-density ref_dirs would starve the reference set.
        p = self._OVERSAMPLE_PARTITIONS.get(self.n_obj, 34)
        R = _uniform_simplex_points(self.n_obj, n_partitions=p)
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


class MaF4(Problem):
    """Inverted and scaled DTLZ3 — same multimodal g as MaF3, objectives scaled 2^1..2^M."""

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
        f = (1 + g)[:, None] - (1 + g)[:, None] * cp * rev
        scale = 2.0 ** np.arange(1, M + 1)
        out["F"] = f * scale

    def _calc_pareto_front(self, ref_dirs=None):
        if ref_dirs is None:
            ref_dirs = _uniform_simplex_points(self.n_obj)
        R = ref_dirs / np.linalg.norm(ref_dirs, axis=1, keepdims=True)
        scale = 2.0 ** np.arange(1, self.n_obj + 1)
        return (1 - R) * scale


class MaF5(Problem):
    """Scaled DTLZ4 — DTLZ4's bias (alpha=100) plus objectives scaled 2^M..2^1."""

    def __init__(self, n_var, n_obj):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        M = self.n_obj
        n = x.shape[0]
        X_ = x[:, :M - 1] ** 100
        X_M = x[:, M - 1:]
        g = np.sum((X_M - 0.5) ** 2, axis=1)
        ones_col = np.ones((n, 1))
        cp = np.fliplr(np.cumprod(np.hstack([ones_col, np.cos(X_ * np.pi / 2)]), axis=1))
        rev = np.hstack([ones_col, np.sin(X_[:, ::-1] * np.pi / 2)])
        f = (1 + g)[:, None] * cp * rev
        scale = 2.0 ** np.arange(M, 0, -1)
        out["F"] = f * scale

    def _calc_pareto_front(self, ref_dirs=None):
        if ref_dirs is None:
            ref_dirs = _uniform_simplex_points(self.n_obj)
        R = ref_dirs / np.linalg.norm(ref_dirs, axis=1, keepdims=True)
        scale = 2.0 ** np.arange(self.n_obj, 0, -1)
        return R * scale


class MaF6(Problem):
    """DTLZ5(I=2,M) — degenerate front: a 1-parameter curve embedded in M dimensions,
    for any M (PlatEMO's generalisation of DTLZ5's fixed-M=3 degenerate curve)."""

    _I = 2  # PlatEMO fixes I=2: exactly one free angle regardless of M

    def __init__(self, n_var, n_obj):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        M, I = self.n_obj, self._I
        n = x.shape[0]
        X_, X_M = x[:, :M - 1].copy(), x[:, M - 1:]
        g = np.sum((X_M - 0.5) ** 2, axis=1)
        if M - 1 >= I:
            temp = g[:, None]
            X_[:, I - 1:M - 1] = (1 + 2 * temp * X_[:, I - 1:M - 1]) / (2 + 2 * temp)
        ones_col = np.ones((n, 1))
        cp = np.fliplr(np.cumprod(np.hstack([ones_col, np.cos(X_ * np.pi / 2)]), axis=1))
        rev = np.hstack([ones_col, np.sin(X_[:, ::-1] * np.pi / 2)])
        out["F"] = (1 + 100 * g)[:, None] * cp * rev

    def _calc_pareto_front(self, ref_dirs=None):
        # Front-intrinsic dimensionality is fixed at I=2 regardless of M (a curve, not
        # an (M-1)-manifold), so the caller-supplied M-dim ref_dirs don't apply here —
        # sample a dense 1-parameter line segment directly instead.
        M, I = self.n_obj, self._I
        R2 = get_reference_directions("das-dennis", I, n_partitions=10000)  # 10,001 pts
        R2 = R2 / np.linalg.norm(R2, axis=1, keepdims=True)
        pad = np.repeat(R2[:, :1], M - I, axis=1)
        R = np.hstack([pad, R2])
        exponents = np.concatenate([[M - I], np.arange(M - I, 2 - I - 1, -1)])
        exponents = np.maximum(exponents, 0)
        return R / (np.sqrt(2.0) ** exponents)[None, :]


class MaF7(Problem):
    """DTLZ7 — disconnected Pareto front (2^(M-1) separate regions)."""

    def __init__(self, n_var, n_obj):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        M = self.n_obj
        n = x.shape[0]
        X_, X_M = x[:, :M - 1], x[:, M - 1:]
        g = 1 + 9 * np.mean(X_M, axis=1)
        f = np.empty((n, M))
        f[:, :M - 1] = X_
        f[:, M - 1] = (1 + g) * (M - np.sum(X_ / (1 + g)[:, None] * (1 + np.sin(3 * np.pi * X_)), axis=1))
        out["F"] = f

    # Points per axis of the (M-1)-D sampling grid below, chosen so the total count
    # (pts_per_dim ** (M-1)) lands close to the P*~=10,000 IGD reference budget.
    _GRID_PTS = {2: 100, 3: 22}

    def _calc_pareto_front(self, ref_dirs=None):
        # PlatEMO's GetOptimum samples a GRID over [0,1]^(M-1) (not a simplex — DTLZ7's
        # first M-1 objectives ARE the free decision variables), then bends it through a
        # piecewise map that concentrates points into the 2^(M-1) valid disconnected
        # regions. Front-intrinsic, so — like MaF6 — the caller-supplied ref_dirs is unused.
        M = self.n_obj
        interval = np.array([0.0, 0.251412, 0.631627, 0.859401])
        median = (interval[1] - interval[0]) / (interval[3] - interval[2] + interval[1] - interval[0])
        pts = self._GRID_PTS.get(M - 1, max(2, round(10000 ** (1.0 / (M - 1)))))
        axes = [np.linspace(0.0, 1.0, pts) for _ in range(M - 1)]
        mesh = np.meshgrid(*axes, indexing="ij")
        X = np.stack([m.ravel() for m in mesh], axis=1)
        lo, hi = X <= median, X > median
        X = X.copy()
        X[lo] = X[lo] * (interval[1] - interval[0]) / median + interval[0]
        X[hi] = (X[hi] - median) * (interval[3] - interval[2]) / (1 - median) + interval[2]
        last = 2 * (M - np.sum(X / 2 * (1 + np.sin(3 * np.pi * X)), axis=1))
        return np.hstack([X, last[:, None]])


_CLASSES = {
    "MAF1": MaF1, "MAF2": MaF2, "MAF3": MaF3, "MAF4": MaF4,
    "MAF5": MaF5, "MAF6": MaF6, "MAF7": MaF7,
}


def get_problem(name: str, n_obj: int = 4):
    """Return (pymoo Problem instance, n_generations) for the given MaF name.

    Args:
        name  : one of "MaF1".."MaF7" (case-insensitive).
        n_obj : number of objectives (default 4). Supported: 3, 4.

    Returns:
        (problem, n_gen) where n_gen = TMAX // pop_size for this n_obj.
    """
    key = name.upper()
    if key not in _CLASSES:
        raise ValueError(f"Unknown problem '{name}'. Choose from {PROBLEM_NAMES}")
    n_var = n_obj + _K[key] - 1
    problem = _CLASSES[key](n_var=n_var, n_obj=n_obj)
    _, pop_size = get_run_config(n_obj)  # raises ValueError for unsupported n_obj
    n_gen = TMAX // pop_size
    return problem, n_gen
