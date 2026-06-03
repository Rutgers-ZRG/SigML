from __future__ import annotations

import numpy as np


def sample_delta_vec(grid, rng: np.random.Generator, alpha: float) -> np.ndarray:
    """Sample a causal one-orbital hybridization vector on a Valenti mesh.

    This follows the scalar-orbital specialization of Valenti Eq. B2/B3:
    nonnegative random DLR weights with an exponential envelope. Positive
    spectral weights give Im Delta(iw) <= 0 on positive Matsubara frequencies.
    In this DLR representation the imaginary-time kernel is
    -exp(-omega*tau)/(1 + exp(-beta*omega)), so nonnegative coefficients make
    Delta(tau) nonpositive and each positive pole contributes a causal scalar
    term whose positive-frequency imaginary part is nonpositive.
    """
    alpha = float(alpha)
    if alpha < 0.0:
        raise ValueError("alpha must be nonnegative")

    k = np.arange(1, grid.n_tau + 1, dtype=float)
    u1 = float(rng.random())
    u2 = rng.random(grid.n_tau)
    weights = np.exp(-alpha * k) * u1 * u2
    delta_tau = grid.gtau_from_coeffs(weights.astype(complex))
    delta_vec = grid.gtau_to_vec(delta_tau).real
    return np.asarray(delta_vec, dtype=np.float64)
