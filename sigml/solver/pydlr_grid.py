from __future__ import annotations

import numpy as np
import pydlr


class PydlrGrid:
    """General-beta pydlr-backed DLR grid using the solver's last-axis API."""

    def __init__(self, beta: float = 70.0, lamb: float = 700.0, eps: float = 1e-13):
        self.beta = float(beta)
        self.lamb = float(lamb)
        self.eps = float(eps)
        self._dlr = pydlr.dlr(lamb=self.lamb, eps=self.eps)
        self.rank = int(self._dlr.rank)
        self.n_tau = self.rank
        self.feature_dim = 2 * self.n_tau
        self.tau_nodes = np.asarray(self._dlr.get_tau(self.beta), dtype=float)
        self.iw_nodes = np.asarray(self._dlr.get_matsubara_frequencies(self.beta))
        self.real_frequency_nodes = np.asarray(self._dlr.get_dlr_frequencies(), dtype=float) / self.beta

    def coeffs_from_gtau(self, gtau: np.ndarray) -> np.ndarray:
        g = np.asarray(gtau, dtype=complex)
        if g.shape[-1] != self.rank:
            raise ValueError(f"Expected last dimension {self.rank}, got {g.shape[-1]}")
        rhs = np.moveaxis(g, -1, 0).reshape(self.rank, -1)
        coeffs = self._dlr.dlr_from_tau(rhs)
        coeffs = coeffs.reshape((self.rank,) + g.shape[:-1])
        return np.moveaxis(coeffs, 0, -1)

    def gtau_from_coeffs(self, coeffs: np.ndarray) -> np.ndarray:
        c = self._coeffs_first_axis(coeffs)
        gtau = self._dlr.tau_from_dlr(c)
        return self._restore_last_axis(gtau, np.asarray(coeffs).shape[:-1])

    def giw_from_coeffs(self, coeffs: np.ndarray) -> np.ndarray:
        c = self._coeffs_first_axis(coeffs)
        giw = self._dlr.matsubara_from_dlr(c, self.beta)
        return self._restore_last_axis(giw, np.asarray(coeffs).shape[:-1])

    def gtau_to_giw(self, gtau: np.ndarray) -> np.ndarray:
        return self.giw_from_coeffs(self.coeffs_from_gtau(gtau))

    def giw_to_gtau(self, giw: np.ndarray) -> np.ndarray:
        g = np.asarray(giw, dtype=complex)
        if g.shape[-1] != self.rank:
            raise ValueError(f"Expected last dimension {self.rank}, got {g.shape[-1]}")
        rhs = np.moveaxis(g, -1, 0).reshape(self.rank, -1)
        coeffs = self._dlr.dlr_from_matsubara(rhs, self.beta)
        coeffs = coeffs.reshape((self.rank,) + g.shape[:-1])
        return self.gtau_from_coeffs(np.moveaxis(coeffs, 0, -1))

    def coeffs_from_giw_lstsq(self, omega: np.ndarray, giw: np.ndarray) -> np.ndarray:
        """Fit DLR coefficients from an arbitrary positive Matsubara mesh.

        Haule eDMFT text files are sampled on a dense positive fermionic
        Matsubara grid rather than pydlr's compressed frequency nodes.  pydlr's
        least-squares fitter expects frequency as ``i*omega`` and matrix blocks
        with frequency on the first axis.
        """

        omega_arr = np.asarray(omega, dtype=float)
        g = np.asarray(giw, dtype=complex)
        if omega_arr.ndim != 1:
            raise ValueError(f"omega must be one-dimensional, got {omega_arr.shape}")
        if g.shape[-1] != omega_arr.shape[0]:
            raise ValueError(
                f"giw last dimension must match omega length {omega_arr.shape[0]}, got {g.shape}"
            )
        if g.ndim < 3 or g.shape[-3] != g.shape[-2]:
            raise ValueError(f"giw must end with a square matrix block and frequency, got {g.shape}")

        block_shape = g.shape[-3:-1]
        leading_shape = g.shape[:-3]
        g_qaa = np.moveaxis(g.reshape((-1,) + block_shape + (g.shape[-1],)), -1, 1)
        fitted = [
            self._dlr.lstsq_dlr_from_matsubara(1j * omega_arr, block, self.beta)
            for block in g_qaa
        ]
        coeffs = np.asarray(fitted).reshape(leading_shape + (self.rank,) + block_shape)
        return np.moveaxis(coeffs, -3, -1)

    def gtau_from_giw_lstsq(self, omega: np.ndarray, giw: np.ndarray) -> np.ndarray:
        return self.gtau_from_coeffs(self.coeffs_from_giw_lstsq(omega, giw))

    def giw_from_coeffs_at_omega(self, coeffs: np.ndarray, omega: np.ndarray) -> np.ndarray:
        omega_arr = np.asarray(omega, dtype=float)
        if omega_arr.ndim != 1:
            raise ValueError(f"omega must be one-dimensional, got {omega_arr.shape}")
        c = np.asarray(coeffs, dtype=complex)
        if c.shape[-1] != self.rank:
            raise ValueError(f"Expected last dimension {self.rank}, got {c.shape[-1]}")
        kernel = -1.0 / (1j * omega_arr[:, None] - self.real_frequency_nodes[None, :])
        values = np.tensordot(c, kernel.T, axes=([-1], [0]))
        return values

    def vec_to_gtau(self, vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec)
        if arr.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected last dimension {self.feature_dim}, got {arr.shape[-1]}")
        paired = np.reshape(arr, arr.shape[:-1] + (self.n_tau, 2))
        return paired[..., 0] + 1j * paired[..., 1]

    def gtau_to_vec(self, gtau: np.ndarray) -> np.ndarray:
        g = np.asarray(gtau)
        if g.shape[-1] != self.n_tau:
            raise ValueError(f"Expected last dimension {self.n_tau}, got {g.shape[-1]}")
        paired = np.stack((g.real, g.imag), axis=-1)
        return np.reshape(paired, g.shape[:-1] + (self.feature_dim,))

    def eval_at_tau(self, gtau: np.ndarray, tau: float):
        coeffs = self.coeffs_from_gtau(gtau)
        omega = self.real_frequency_nodes
        tau = float(tau)
        log_kernel_abs = -omega * tau - np.logaddexp(0.0, -self.beta * omega)
        kernel = np.exp(log_kernel_abs)
        values = np.tensordot(kernel, coeffs, axes=(0, -1))
        if values.shape == ():
            return complex(values)
        return values

    def giw_positive(self, coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        giw = self.giw_from_coeffs(coeffs)
        mask = self.iw_nodes.imag > 0
        return self.iw_nodes[mask], giw[..., mask]

    def _coeffs_first_axis(self, coeffs: np.ndarray) -> np.ndarray:
        c = np.asarray(coeffs, dtype=complex)
        if c.shape[-1] != self.rank:
            raise ValueError(f"Expected last dimension {self.rank}, got {c.shape[-1]}")
        return np.moveaxis(c, -1, 0).reshape(self.rank, -1)

    def _restore_last_axis(self, values: np.ndarray, leading_shape: tuple[int, ...]) -> np.ndarray:
        arr = np.asarray(values).reshape((self.rank,) + leading_shape)
        return np.moveaxis(arr, 0, -1)
