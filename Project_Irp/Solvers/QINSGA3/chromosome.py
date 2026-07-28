"""Quantum chromosome population for QINSGA-III.

Encoding: each gene j is represented by angle θ_j ∈ [0, π/2].

Measurement [Li & Wang 2007, eq. 3]:
    x_j = xl_j + cos²(θ_j) × (xu_j − xl_j)

Initial state [Han & Kim 2002]:
    All individuals start at π/4 (maximum superposition) with a small
    uniform perturbation ±0.05 rad to break symmetry so the first
    measurement does not collapse all solutions to the same point.

Rotation gate variants (see rotate()):
    "tanh"      [Li et al. ICNC 2008, aggressive]
    "tanh_soft" [empirical soft variant]
    "linear"    [Han & Kim 2002, literature baseline]

Crossover and mutation are NOT methods of this class — Solvers/QINSGA3/algorithm.py
applies pymoo's own SBX/PM directly in decision-variable (X) space instead
(see that module's docstring for why: the cos² measurement map above is
highly non-uniform, which was found to hurt variation operators applied in
θ-space directly).
"""

import numpy as np


_ROTATION_TYPES = ("tanh", "tanh_soft", "linear")


class QuantumPopulation:
    """Population of quantum chromosomes stored as angle matrix θ (pop_size, n_genes).

    rotation_type controls the quantum rotation gate used in rotate():
      "tanh"      — Δθ = α × tanh((θ_guide − θ) / (π/8))
                    Aggressive: saturates at 76 % of α when |diff| = π/8.
                    Near-constant step for most of [0, π/2].
                    [Li et al. ICNC 2008]

      "tanh_soft" — Δθ = α × tanh((θ_guide − θ) / (π/4))
                    Soft: saturates at 76 % of α when |diff| = π/4.
                    More proportional for moderate angular distances.

      "linear"    — Δθ = α × (θ_guide − θ) / (π/2)
                    Strictly proportional; max step = α.
                    Standard adaptation of Han & Kim 2002 for continuous domain.
    """

    def __init__(
        self,
        pop_size:      int,
        n_genes:       int,
        xl:            np.ndarray,
        xu:            np.ndarray,
        rng:           np.random.Generator | None = None,
        rotation_type: str = "tanh",
        noise_scale:   float = 0.02,
    ) -> None:
        if rotation_type not in _ROTATION_TYPES:
            raise ValueError(f"rotation_type must be one of {_ROTATION_TYPES}, got '{rotation_type}'")
        self.pop_size      = pop_size
        self.n_genes       = n_genes
        self.xl            = np.asarray(xl, dtype=float)
        self.xu            = np.asarray(xu, dtype=float)
        self.rng           = rng if rng is not None else np.random.default_rng()
        self.rotation_type = rotation_type
        self.noise_scale   = noise_scale

        # Maximum superposition θ = π/4  [Han & Kim 2002, §II-A]
        # ±0.05 rad perturbation breaks symmetry so the first measurement
        # does not collapse all N solutions to the same decoded point.
        self.theta = np.full((pop_size, n_genes), np.pi / 4.0)
        self.theta += self.rng.uniform(-0.05, 0.05, (pop_size, n_genes))
        self.theta  = np.clip(self.theta, 0.0, np.pi / 2.0)

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def measure(self) -> np.ndarray:
        """Collapse quantum state → classical decision-variable matrix (pop_size, n_genes).

        Deterministic component [Li & Wang 2007, eq. 3]:
            x_j = xl_j + cos²(θ_j) × (xu_j − xl_j)

        Diversity noise [Platel et al. 2009, §4.2]:
            σ_j = noise_scale × |sin(2θ_j)| × (xu_j − xl_j)
            noise ~ N(0, σ_j)

        The noise amplitude is proportional to |sin(2θ)|, which peaks at
        superposition (θ ≈ π/4) and vanishes at convergence (θ → 0 or π/2).
        This prevents nearby θ values from collapsing to identical integer
        routes after the IRP decoder rounds to integers — noise_scale
        defaults to 0.02 (the value this was originally tuned at for the
        IRP) so every existing caller is unaffected; it exists as a
        parameter so continuous-domain callers (no integer rounding to
        protect against) can set it to 0 without touching this file again.
        """
        p     = np.cos(self.theta) ** 2
        mu    = self.xl + p * (self.xu - self.xl)
        sigma = self.noise_scale * np.abs(np.sin(2.0 * self.theta)) * (self.xu - self.xl)
        noise = self.rng.standard_normal(self.theta.shape) * sigma
        return np.clip(mu + noise, self.xl, self.xu)

    # ------------------------------------------------------------------
    # Rotation gate
    # ------------------------------------------------------------------

    def rotate(
        self,
        guides_theta: np.ndarray,
        alpha:        float,
    ) -> None:
        """Apply quantum rotation gate toward the guide angles.

        All three variants share the same maximum step bound α and are
        clipped to [0, π/2] after the update.

        "tanh" (aggressive) [Li et al. ICNC 2008]:
            Δθ = α × tanh(diff / (π/8))
            Saturates quickly — nearly constant step ≈ α across most of
            the domain.  Preferred for fast convergence.

        "tanh_soft":
            Δθ = α × tanh(diff / (π/4))
            Gentler saturation — more proportional for moderate distances.

        "linear" [Han & Kim 2002, §II-B]:
            Δθ = α × diff / (π/2)
            Strictly proportional; step is zero when already at the guide.
            Literature baseline — slowest but most stable near convergence.
        """
        diff = guides_theta - self.theta
        if self.rotation_type == "tanh":
            self.theta += alpha * np.tanh(diff / (np.pi / 8.0))
        elif self.rotation_type == "tanh_soft":
            self.theta += alpha * np.tanh(diff / (np.pi / 4.0))
        else:  # "linear"
            self.theta += alpha * diff / (np.pi / 2.0)
        self.theta = np.clip(self.theta, 0.0, np.pi / 2.0)
