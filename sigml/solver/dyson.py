from __future__ import annotations

import numpy as np


def sigma_from_g(
    g_iw: np.ndarray,
    delta_iw: np.ndarray,
    mu: float,
    eps_d: float,
    iw: np.ndarray,
) -> np.ndarray:
    """Compute impurity self-energy from Dyson's equation."""
    return iw + mu - eps_d - delta_iw - 1.0 / g_iw


def positive_matsubara_mask(iw: np.ndarray) -> np.ndarray:
    """Return the mask for positive Matsubara frequencies."""
    return np.asarray(iw).imag > 0


def is_causal(sigma_iw: np.ndarray, iw: np.ndarray, tol: float = 1e-6) -> bool:
    """Check Im Sigma(iw) <= 0 on positive Matsubara frequencies only."""
    mask = positive_matsubara_mask(iw)
    return bool(np.all(np.asarray(sigma_iw)[mask].imag <= tol))
