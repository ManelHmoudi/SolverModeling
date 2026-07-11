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

Crossover [Deb 2001, §2.3]:
    Bounded SBX in θ-space.

Mutation (two-tier, see mutate()):
    "strong" [Han & Kim 2002]: continuous generalisation of the quantum NOT
        gate — reset selected genes to Uniform(0, π/2). Maximum diversification,
        but can destroy structure the rotation gate has already converged on.
    "weak": local perturbation θ += N(0, sigma), clipped to [0, π/2] —
        explores around the current angle instead of discarding it, so the
        gate's progress is not wiped out. Balances exploration/exploitation.
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
    ) -> None:
        if rotation_type not in _ROTATION_TYPES:
            raise ValueError(f"rotation_type must be one of {_ROTATION_TYPES}, got '{rotation_type}'")
        self.pop_size      = pop_size
        self.n_genes       = n_genes
        self.xl            = np.asarray(xl, dtype=float)
        self.xu            = np.asarray(xu, dtype=float)
        self.rng           = rng if rng is not None else np.random.default_rng()
        self.rotation_type = rotation_type

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
            σ_j = 0.02 × |sin(2θ_j)| × (xu_j − xl_j)
            noise ~ N(0, σ_j)

        The noise amplitude is proportional to |sin(2θ)|, which peaks at
        superposition (θ ≈ π/4) and vanishes at convergence (θ → 0 or π/2).
        This prevents nearby θ values from collapsing to identical integer
        routes after the IRP decoder rounds to integers.
        """
        p     = np.cos(self.theta) ** 2
        mu    = self.xl + p * (self.xu - self.xl)
        sigma = 0.02 * np.abs(np.sin(2.0 * self.theta)) * (self.xu - self.xl)
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

    # ------------------------------------------------------------------
    # Crossover
    # ------------------------------------------------------------------

    def crossover(self, p_cross: float, eta: float) -> None:
        """Quantum SBX crossover in θ-space — bounded variant [Deb 2001, §2.3].

        Randomly pairs individuals; each pair crosses with probability p_cross.
        The bounded SBX spread factor β_q accounts for the distance from each
        parent to the domain boundary [0, π/2], ensuring offspring never leave
        the feasible angle range and that the distribution is correctly shaped
        near the bounds [Deb 2001, eq. 2.13].

        Generation order (rotate → crossover → mutate) follows NSGA-III
        [Deb & Jain 2014] adapted to the quantum domain.

        Recommended: p_cross = 0.9, eta = 5.
        """
        lo, hi = 0.0, np.pi / 2.0
        idx    = self.rng.permutation(self.pop_size)

        for k in range(0, self.pop_size - 1, 2):
            i, j = idx[k], idx[k + 1]
            if self.rng.random() > p_cross:
                continue

            # p1 ≤ p2 per gene (vectorised)
            p1   = np.minimum(self.theta[i], self.theta[j])
            p2   = np.maximum(self.theta[i], self.theta[j])
            diff = p2 - p1

            active = diff > 1e-14   # skip genes where parents are identical

            # Boundary-aware spread factor α [Deb 2001, eq. 2.13]:
            # min(p1 − lo, hi − p2) ≥ 0 because θ ∈ [lo, hi]
            beta_a = np.where(
                active,
                np.maximum(
                    1.0 + 2.0 * np.minimum(p1 - lo, hi - p2) / np.where(active, diff, 1.0),
                    1e-6,
                ),
                2.0,
            )
            alpha = 2.0 - np.power(beta_a, -(eta + 1.0))

            u = self.rng.random(self.n_genes)
            beta_q = np.where(
                u <= 1.0 / alpha,
                np.power(alpha * u, 1.0 / (eta + 1.0)),
                np.power(
                    1.0 / np.maximum(2.0 - alpha * u, 1e-12),
                    1.0 / (eta + 1.0),
                ),
            )

            mid       = 0.5 * (p1 + p2)
            half_diff = 0.5 * diff

            c1 = np.clip(mid - beta_q * half_diff, lo, hi)
            c2 = np.clip(mid + beta_q * half_diff, lo, hi)

            c1 = np.where(active, c1, p1)
            c2 = np.where(active, c2, p2)

            # Random offspring assignment avoids directional bias
            swap          = self.rng.random(self.n_genes) < 0.5
            self.theta[i] = np.where(swap, c1, c2)
            self.theta[j] = np.where(swap, c2, c1)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def mutate(self, prob: float, p_strong: float = 0.15, sigma: float = 0.05 * np.pi) -> None:
        """Two-tier quantum mutation on genes selected with probability `prob`.

        Each selected gene is mutated:
          - "strong" (probability p_strong) [Han & Kim 2002, §II-C]: reset to
            Uniform(0, π/2) — continuous generalisation of the quantum NOT
            gate. Maximum diversification, but can undo structure the
            rotation gate already converged on.
          - "weak" (probability 1 − p_strong): θ += N(0, sigma), clipped to
            [0, π/2] — local perturbation that explores around the current
            angle instead of discarding it.

        Recommended: prob = 2 / n_genes (two genes mutated per individual on
        average); p_strong = 0.15 keeps most mutations local while still
        allowing occasional full resets to escape stagnation. Empirically,
        p_strong=0.3 caused GD instability across seeds on the 100-client IRP
        instance (see validation/compare_pstrong.py) — 0.15 improved mean GD,
        IGD, HV and Spacing over 10 seeds.
        """
        mask = self.rng.random(self.theta.shape) < prob
        if not mask.any():
            return

        strong_mask = mask & (self.rng.random(self.theta.shape) < p_strong)
        weak_mask   = mask & ~strong_mask

        n_strong = int(strong_mask.sum())
        if n_strong > 0:
            self.theta[strong_mask] = self.rng.uniform(0.0, np.pi / 2.0, n_strong)

        n_weak = int(weak_mask.sum())
        if n_weak > 0:
            perturbed = self.theta[weak_mask] + self.rng.standard_normal(n_weak) * sigma
            self.theta[weak_mask] = np.clip(perturbed, 0.0, np.pi / 2.0)
