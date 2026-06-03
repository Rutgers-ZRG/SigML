from __future__ import annotations

import numpy as np

from sigml.solver.dyson import positive_frequency_imaginary_parts


def g_mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error between predicted and target G vectors."""
    diff = np.asarray(pred) - np.asarray(target)
    return float(np.mean(np.square(np.abs(diff))))


def positive_freq_causality_rate(
    sigma_iw: np.ndarray,
    iw: np.ndarray,
    tol: float = 1e-6,
) -> float:
    """Fraction of samples with nonpositive Im values/eigenvalues at positive frequencies."""
    pos_imag = positive_frequency_imaginary_parts(sigma_iw, iw)
    sigma = np.asarray(sigma_iw)
    if sigma.ndim >= 3 and sigma.shape[-1] == sigma.shape[-2]:
        causal = np.all(pos_imag <= tol, axis=(-2, -1))
    else:
        causal = np.all(pos_imag <= tol, axis=-1)
    return float(np.mean(causal))


def quasiparticle_proxy(g_tau: np.ndarray, grid, beta: float) -> float | np.ndarray:
    """Use -G(beta/2), evaluated through the grid's DLR representation."""
    value = -grid.eval_at_tau(g_tau, float(beta) / 2.0).real
    if np.asarray(value).shape == ():
        return float(value)
    return value


def orbital_occupation(g_tau: np.ndarray) -> float | np.ndarray:
    """Estimate orbital occupation matrix from the final imaginary-time node."""
    occ = -np.asarray(g_tau)[..., -1].real
    if np.asarray(occ).shape == ():
        return float(occ)
    return occ
