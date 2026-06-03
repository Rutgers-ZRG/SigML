from __future__ import annotations

import numpy as np
import pydlr


class PydlrGrid:
    """Independent pydlr-backed analytic DLR grid for tests."""

    def __init__(self, beta: float = 70.0, lamb: float = 700.0, eps: float = 1e-13):
        self.beta = float(beta)
        self.lamb = float(lamb)
        self.eps = float(eps)
        self._dlr = pydlr.dlr(lamb=self.lamb, eps=self.eps)
        self.rank = int(self._dlr.rank)
        self.tau_nodes = np.asarray(self._dlr.get_tau(self.beta), dtype=float)
        self.iw_nodes = np.asarray(self._dlr.get_matsubara_frequencies(self.beta))

    def coeffs_from_gtau(self, gtau: np.ndarray) -> np.ndarray:
        g = np.asarray(gtau, dtype=complex)
        if g.shape[0] != self.rank:
            raise ValueError(f"Expected first dimension {self.rank}, got {g.shape[0]}")
        return self._dlr.dlr_from_tau(g)

    def gtau_from_coeffs(self, coeffs: np.ndarray) -> np.ndarray:
        return self._dlr.tau_from_dlr(np.asarray(coeffs, dtype=complex))

    def giw_from_coeffs(self, coeffs: np.ndarray) -> np.ndarray:
        return self._dlr.matsubara_from_dlr(np.asarray(coeffs, dtype=complex), self.beta)

    def giw_positive(self, coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        giw = self.giw_from_coeffs(coeffs)
        mask = self.iw_nodes.imag > 0
        return self.iw_nodes[mask], giw[mask]
