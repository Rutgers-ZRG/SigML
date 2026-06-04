from __future__ import annotations

import numpy as np


def sigma_from_g(
    g_iw: np.ndarray,
    delta_iw: np.ndarray,
    mu: float,
    eps_d: float | np.ndarray,
    iw: np.ndarray,
) -> np.ndarray:
    """Compute impurity self-energy from Dyson's equation."""
    g = np.asarray(g_iw, dtype=complex)
    delta = np.asarray(delta_iw, dtype=complex)
    iw_arr = np.asarray(iw, dtype=complex)
    eps = np.asarray(eps_d, dtype=complex)
    if _is_matrix_block(g):
        if delta.shape != g.shape:
            raise ValueError(f"delta_iw must have shape {g.shape}, got {delta.shape}")
        if iw_arr.shape != g.shape[:-2]:
            raise ValueError(f"iw must have shape {g.shape[:-2]}, got {iw_arr.shape}")
        eye = np.eye(g.shape[-1], dtype=complex)
        one_body = (iw_arr + mu)[..., None, None] * eye - eps
        return one_body - delta - np.linalg.inv(g)
    return iw_arr + mu - eps - delta - 1.0 / g


def positive_matsubara_mask(iw: np.ndarray) -> np.ndarray:
    """Return the mask for positive Matsubara frequencies."""
    return np.asarray(iw).imag > 0


def _is_matrix_block(values: np.ndarray) -> bool:
    return values.ndim >= 3 and values.shape[-1] == values.shape[-2]


def positive_frequency_imaginary_parts(values_iw: np.ndarray, iw: np.ndarray) -> np.ndarray:
    """Imaginary scalar values or eigenvalues on positive Matsubara frequencies."""
    values = np.asarray(values_iw, dtype=complex)
    mask = positive_matsubara_mask(iw)
    if _is_matrix_block(values):
        selected = values[..., mask, :, :] if values.shape[-3] == mask.shape[0] else values[mask]
        return np.linalg.eigvals(selected).imag
    return values[..., mask].imag


def is_causal(sigma_iw: np.ndarray, iw: np.ndarray, tol: float = 1e-6) -> bool:
    """Check Im Sigma eigenvalues <= 0 on positive Matsubara frequencies."""
    return bool(np.all(positive_frequency_imaginary_parts(sigma_iw, iw) <= tol))
