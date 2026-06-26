"""Quantum chromosome population for QINSGA-III.

Encoding: each gene j is represented by angle θ_j ∈ [0, π/2].
Measurement: x_j = xl_j + cos²(θ_j) × (xu_j − xl_j)   [Li & Wang 2007, eq. 3]
Initial state: all individuals at π/4 (maximum superposition) with small
uniform perturbation ±0.05 to break symmetry  [Han & Kim 2002].
"""

import numpy as np


class QuantumPopulation:
    """Population of quantum chromosomes stored as angle matrix θ (pop_size, n_genes)."""

    def __init__(
        self,
        pop_size: int,
        n_genes:  int,
        xl:       np.ndarray,
        xu:       np.ndarray,
        rng:      np.random.Generator | None = None,
    ) -> None:
        self.pop_size = pop_size
        self.n_genes  = n_genes
        self.xl       = np.asarray(xl, dtype=float)
        self.xu       = np.asarray(xu, dtype=float)
        self.rng      = rng if rng is not None else np.random.default_rng()

        # All individuals start at π/4 (maximum superposition) — Han & Kim 2002.
        # A small uniform perturbation ±0.05 rad breaks symmetry so that the
        # first measurement does not collapse all solutions to the same point.
        self.theta = np.full((pop_size, n_genes), np.pi / 4.0)
        self.theta += self.rng.uniform(-0.05, 0.05, (pop_size, n_genes))
        self.theta  = np.clip(self.theta, 0.0, np.pi / 2.0)

    def measure(self) -> np.ndarray:
        """Collapse quantum state → classical decision-variable matrix (pop_size, n_genes).

        x_j = xl_j + cos²(θ_j) × (xu_j − xl_j)   [Li & Wang 2007, eq. 3]

        Small diversity noise σ = 0.02 × |sin(2θ)| × (xu−xl) is added so nearby
        θ values yield different decoded routes after integer rounding (IRP decoder).
        Noise vanishes at convergence (θ → 0 or π/2) and peaks at superposition
        (θ ≈ π/4)  [Platel et al. 2009, §4.2].
        """
        p      = np.cos(self.theta) ** 2
        mu     = self.xl + p * (self.xu - self.xl)
        sigma  = 0.02 * np.abs(np.sin(2.0 * self.theta)) * (self.xu - self.xl)
        noise  = self.rng.standard_normal(self.theta.shape) * sigma
        return np.clip(mu + noise, self.xl, self.xu)

    def rotate(
        self,
        guides_theta: np.ndarray,
        alpha:        float,
    ) -> None:
        """Adaptive quantum rotation gate (Li & Wang 2007; Platel et al. 2009).

        Δθ_ij = α × tanh((θ_guide_ij − θ_ij) / (π/8))

        tanh scaling gives large steps far from the guide and small steps near it,
        preventing oscillation at convergence. The |cos(2θ)| damping factor from
        binary QIEAs is omitted: it vanishes at superposition (θ=π/4), blocking
        rotation where exploration is most needed.
        """
        diff       = guides_theta - self.theta
        self.theta += alpha * np.tanh(diff / (np.pi / 8.0))
        self.theta  = np.clip(self.theta, 0.0, np.pi / 2.0)

    def crossover(self, p_cross: float, eta: float) -> None:
        """Quantum SBX crossover in θ-space — bounded variant (Deb 2001, §2.3).

        Randomly pairs individuals; each pair crosses with probability p_cross.
        The bounded SBX computes β_q taking the distance to the domain boundaries
        [0, π/2] into account, so offspring are never generated outside the
        feasible angle range and the distribution is correctly shaped near bounds.

        Placing crossover after rotate() and before mutate() follows the standard
        NSGA-III generation order (Deb & Jain 2014) adapted to the quantum domain.

        Recommended: p_cross = 0.9, eta = 5.
        """
        lo, hi = 0.0, np.pi / 2.0
        idx    = self.rng.permutation(self.pop_size)

        for k in range(0, self.pop_size - 1, 2):
            i, j = idx[k], idx[k + 1]
            if self.rng.random() > p_cross:
                continue

            # p1 ≤ p2 per gene (vectorised)
            p1 = np.minimum(self.theta[i], self.theta[j])
            p2 = np.maximum(self.theta[i], self.theta[j])
            diff = p2 - p1

            active = diff > 1e-14   # skip genes where parents are identical

            # Boundary-aware spread factor α (Deb 2001, eq. 2.13)
            # min(p1-lo, hi-p2) ≥ 0 because θ ∈ [lo, hi]
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

            # Where parents were identical keep them unchanged
            c1 = np.where(active, c1, p1)
            c2 = np.where(active, c2, p2)

            # Randomly assign offspring to avoid directional bias
            swap          = self.rng.random(self.n_genes) < 0.5
            self.theta[i] = np.where(swap, c1, c2)
            self.theta[j] = np.where(swap, c2, c1)

    def mutate(self, prob: float) -> None:
        """Quantum mutation: reset selected genes to Uniform(0, π/2).

        Generalises the quantum NOT gate to continuous angles (Han & Kim 2002).
        Resets across the full angle range [0, π/2] for genuine diversification.
        Recommended prob = 2/n_genes (two genes mutated per individual on average).
        """
        mask  = self.rng.random(self.theta.shape) < prob
        n_mut = int(mask.sum())
        if n_mut > 0:
            self.theta[mask] = self.rng.uniform(0.0, np.pi / 2.0, n_mut)
