from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


T2G_BROAD_RANGES = {
    "U": (2.0, 6.0),
    "J": (0.25, 0.9),
    "beta": (40.0, 40.0),
    "mu": (0.8, 3.2),
}
"""Documented smoke-set ranges for Stage-2 SrVO3 t2g Kanamori baths."""


@dataclass(frozen=True)
class T2GBathSample:
    """Synthetic t2g Kanamori bath sample on block DLR tau nodes."""

    delta: np.ndarray
    eps_d: np.ndarray
    U: float
    J: float
    beta: float
    mu: float
    mode: str


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


def sample_t2g_bath(
    grid,
    rng: np.random.Generator,
    *,
    mode: str = "broad",
    trajectory: list[T2GBathSample | dict[str, Any]] | None = None,
    neighborhood: float = 0.1,
    alpha: float | None = None,
) -> T2GBathSample:
    """Sample a causal three-orbital t2g Kanamori hybridization bath.

    ``broad`` samples a material-agnostic smoke set over ``T2G_BROAD_RANGES``.
    ``warm`` samples convex causal neighborhoods around supplied DMFT-like
    trajectory iterates. Phase 9 will provide real SrVO3 iterates; the sampler
    accepts the same shape now so tests and downstream plumbing can use stubs.
    """

    if mode == "broad":
        params = _sample_broad_scalars(rng, beta=float(grid.beta))
        delta = _sample_causal_t2g_delta(grid, rng, alpha=alpha)
        eps_d = _sample_crystal_field(rng)
        return T2GBathSample(delta=delta, eps_d=eps_d, mode="broad", **params)

    if mode == "warm":
        if not trajectory:
            raise ValueError("trajectory must contain at least one sample in warm mode")
        if neighborhood < 0.0 or neighborhood > 1.0:
            raise ValueError("neighborhood must be in [0, 1]")
        center = _coerce_t2g_sample(trajectory[int(rng.integers(0, len(trajectory)))])
        params = _sample_warm_scalars(center, rng, neighborhood, beta=float(grid.beta))
        fresh = _sample_causal_t2g_delta(grid, rng, alpha=alpha)
        eps_fresh = _sample_crystal_field(rng)
        mix = float(neighborhood)
        delta = (1.0 - mix) * center.delta + mix * fresh
        eps_d = _hermitian_part((1.0 - mix) * center.eps_d + mix * eps_fresh)
        return T2GBathSample(delta=delta, eps_d=eps_d, mode="warm", **params)

    raise ValueError(f"Unsupported t2g bath mode {mode!r}; expected 'broad' or 'warm'")


def _sample_broad_scalars(rng: np.random.Generator, *, beta: float) -> dict[str, float]:
    u = _uniform_range(rng, "U")
    j_hi = min(T2G_BROAD_RANGES["J"][1], 0.22 * u)
    j_lo = min(T2G_BROAD_RANGES["J"][0], j_hi)
    j = float(rng.uniform(j_lo, j_hi))
    mu = _uniform_range(rng, "mu")
    return {"U": u, "J": j, "beta": beta, "mu": mu}


def _uniform_range(rng: np.random.Generator, name: str) -> float:
    lo, hi = T2G_BROAD_RANGES[name]
    return float(rng.uniform(lo, hi))


def _sample_warm_scalars(
    center: T2GBathSample, rng: np.random.Generator, neighborhood: float, *, beta: float
) -> dict[str, float]:
    jitter = max(float(neighborhood), 1e-12)
    u = _clip_range("U", center.U + rng.normal(0.0, 0.25 * jitter))
    j = _clip_range("J", center.J + rng.normal(0.0, 0.12 * jitter))
    mu = _clip_range("mu", center.mu + rng.normal(0.0, 0.25 * jitter))
    return {"U": u, "J": min(j, 0.25 * u), "beta": beta, "mu": mu}


def _clip_range(name: str, value: float) -> float:
    lo, hi = T2G_BROAD_RANGES[name]
    return float(np.clip(value, lo, hi))


def _sample_causal_t2g_delta(
    grid,
    rng: np.random.Generator,
    *,
    alpha: float | None,
) -> np.ndarray:
    alpha = float(rng.uniform(0.02, 0.28) if alpha is None else alpha)
    if alpha < 0.0:
        raise ValueError("alpha must be nonnegative")

    k = np.arange(1, grid.n_tau + 1, dtype=float)
    envelope = np.exp(-alpha * k)
    scale = float(rng.uniform(0.15, 1.2))
    causal_sign = _causal_coeff_sign(grid)
    coeffs = np.empty((3, 3, grid.n_tau), dtype=complex)
    for idx, weight in enumerate(envelope * scale * rng.random(grid.n_tau)):
        coeffs[:, :, idx] = causal_sign * weight * _random_density_matrix(rng, 3)
    delta = grid.gtau_from_coeffs(coeffs)
    return _hermitian_tau(delta)


def _causal_coeff_sign(grid) -> float:
    coeffs = np.zeros((1, 1, grid.n_tau), dtype=complex)
    coeffs[0, 0, 0] = 1.0
    giw = grid.giw_from_coeffs(coeffs)
    pos = grid.iw_nodes.imag > 0
    imag = np.asarray(giw[0, 0, pos]).imag
    return 1.0 if np.max(imag) <= 0.0 else -1.0


def _random_density_matrix(rng: np.random.Generator, dim: int) -> np.ndarray:
    raw = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    psd = raw @ raw.conj().T
    trace = float(np.trace(psd).real)
    if trace <= 0.0:
        return np.eye(dim, dtype=complex) / dim
    return psd / trace


def _sample_crystal_field(rng: np.random.Generator) -> np.ndarray:
    diag = rng.normal(0.0, 0.12, size=3)
    diag -= np.mean(diag)
    offdiag = 0.03 * (rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
    return _hermitian_part(np.diag(diag) + offdiag)


def _hermitian_tau(blocks: np.ndarray) -> np.ndarray:
    return 0.5 * (blocks + np.swapaxes(blocks.conj(), 0, 1))


def _hermitian_part(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(matrix, dtype=complex) + np.asarray(matrix, dtype=complex).conj().T)


def _coerce_t2g_sample(sample: T2GBathSample | dict[str, Any]) -> T2GBathSample:
    if isinstance(sample, T2GBathSample):
        return sample
    if isinstance(sample, dict):
        return T2GBathSample(
            delta=np.asarray(sample["delta"], dtype=complex),
            eps_d=np.asarray(sample.get("eps_d", np.zeros((3, 3), dtype=complex)), dtype=complex),
            U=float(sample["U"]),
            J=float(sample["J"]),
            beta=float(sample["beta"]),
            mu=float(sample["mu"]),
            mode=str(sample.get("mode", "trajectory")),
        )
    raise TypeError("trajectory entries must be T2GBathSample instances or dicts")
