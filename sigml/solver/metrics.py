from __future__ import annotations

import numpy as np

from sigml.solver.dyson import positive_matsubara_mask


def g_mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error between predicted and target G vectors."""
    diff = np.asarray(pred) - np.asarray(target)
    return float(np.mean(np.square(np.abs(diff))))


def positive_freq_causality_rate(
    sigma_iw: np.ndarray,
    iw: np.ndarray,
    tol: float = 1e-6,
) -> float:
    """Fraction of samples with Im Sigma(iw) <= 0 on all positive frequencies."""
    sigma = np.asarray(sigma_iw)
    mask = positive_matsubara_mask(iw)
    pos_imag = sigma[..., mask].imag
    causal = np.all(pos_imag <= tol, axis=-1)
    return float(np.mean(causal))


def quasiparticle_proxy(g_tau: np.ndarray, grid, beta: float) -> float:
    """Use -G(beta/2), evaluated through the grid's DLR representation."""
    return float(-grid.eval_at_tau(g_tau, float(beta) / 2.0).real)
