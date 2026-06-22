"""Quantum chromosome population for QINSGA-III.

Encoding: each gene j is represented by angle θ_j ∈ [0, π/2].
Measurement: x_j = xl_j + cos²(θ_j) × (xu_j − xl_j)
Initial state: θ_j = π/4 ± ε (superposition with small diversity noise).
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

        self.theta = np.full((pop_size, n_genes), np.pi / 4.0)
        self.theta += self.rng.uniform(-0.05, 0.05, self.theta.shape)
        self.theta  = np.clip(self.theta, 0.0, np.pi / 2.0)

    def measure(self) -> np.ndarray:
        """Collapse quantum state → classical decision-variable matrix (pop_size, n_genes)."""
        p      = np.cos(self.theta) ** 2
        mu     = self.xl + p * (self.xu - self.xl)
        spread = np.abs(np.sin(2.0 * self.theta))
        sigma  = 0.05 * spread * (self.xu - self.xl)
        noise  = self.rng.standard_normal(self.theta.shape) * sigma
        return np.clip(mu + noise, self.xl, self.xu)

    def rotate(
        self,
        X:        np.ndarray,
        guides_X: np.ndarray,
        alpha:    float,
    ) -> None:
        """Apply adaptive quantum rotation gate (Han & Kim 2002).

        Δθ = alpha × sign(θ_guide − θ) × |cos(2θ)|
        The |cos(2θ)| factor stops rotation at boundaries, enforcing natural convergence.
        """
        safe_range  = np.where(self.xu - self.xl > 1e-9, self.xu - self.xl, 1.0)
        p_guide     = np.clip((guides_X - self.xl) / safe_range, 0.0, 1.0)
        theta_guide = np.arccos(np.sqrt(p_guide))

        magnitude   = np.abs(np.cos(2.0 * self.theta))
        direction   = np.sign(theta_guide - self.theta)

        self.theta += alpha * direction * magnitude
        self.theta  = np.clip(self.theta, 0.0, np.pi / 2.0)

    def mutate(self, prob: float) -> None:
        """Reset selected genes to superposition (π/4 ± noise). Recommended prob = 1/n_genes."""
        mask  = self.rng.random(self.theta.shape) < prob
        n_mut = int(mask.sum())
        if n_mut > 0:
            noise = self.rng.uniform(-0.1, 0.1, n_mut)
            self.theta[mask] = np.pi / 4.0 + noise
            self.theta = np.clip(self.theta, 0.0, np.pi / 2.0)
