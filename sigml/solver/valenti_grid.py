from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


DEFAULT_MESH_PATH = (
    "/Users/li/dev/RA/dmft/ref/mlDMFT/mldmft/models/orb1/mesh_beta70.h5"
)


class ValentiOrb1Grid:
    """Load Valenti's orb1 DLR mesh and transforms from mesh_beta70.h5."""

    def __init__(self, mesh_path: str | Path = DEFAULT_MESH_PATH, beta: float = 70.0):
        if not np.isclose(float(beta), 70.0):
            raise ValueError(
                "ValentiOrb1Grid supports beta=70.0 only for the orb1 mesh; "
                "general beta support requires porting Valenti switch_mesh."
            )
        self.mesh_path = str(mesh_path)
        self.beta = float(beta)

        with h5py.File(self.mesh_path, "r") as h5:
            root = h5["mesh_dlr_imtime"]
            it = root["dlr_it/it"][()]
            self.tau_nodes = self._mesh_saved_tau(it, self.beta)
            self.real_frequency_nodes = root["dlr_it/rf"][()] / self.beta
            self._cf2it = root["dlr_it/cf2it"][()]
            self._cf2if = self._read_complex(root["dlr_if/cf2if"][()])
            self._if_indices = root["dlr_if/if"][()].astype(np.int64)

        self.n_tau = int(self.tau_nodes.shape[0])
        self.feature_dim = 2 * self.n_tau
        self.iw_nodes = 1j * np.pi * (2 * self._if_indices + 1) / self.beta

    @staticmethod
    def _read_complex(arr: np.ndarray) -> np.ndarray:
        return arr[..., 0] + 1j * arr[..., 1]

    @staticmethod
    def _mesh_saved_tau(it: np.ndarray, beta: float) -> np.ndarray:
        it = np.asarray(it, dtype=float)
        return np.where(it >= 0.0, beta * it, beta * (1.0 + it))

    def vec_to_gtau(self, vec118: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec118)
        if vec.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected last dimension {self.feature_dim}, got {vec.shape[-1]}")
        paired = np.reshape(vec, vec.shape[:-1] + (self.n_tau, 2))
        return paired[..., 0] + 1j * paired[..., 1]

    def gtau_to_vec(self, g59c: np.ndarray) -> np.ndarray:
        g = np.asarray(g59c)
        if g.shape[-1] != self.n_tau:
            raise ValueError(f"Expected last dimension {self.n_tau}, got {g.shape[-1]}")
        paired = np.stack((g.real, g.imag), axis=-1)
        return np.reshape(paired, g.shape[:-1] + (self.feature_dim,))

    def coeffs_from_gtau(self, g59c: np.ndarray) -> np.ndarray:
        g = np.asarray(g59c, dtype=complex)
        if g.shape != (self.n_tau,):
            raise ValueError(f"Expected shape ({self.n_tau},), got {g.shape}")
        return np.linalg.solve(self._cf2it.astype(complex), g)

    def gtau_from_coeffs(self, coeffs: np.ndarray) -> np.ndarray:
        c = np.asarray(coeffs, dtype=complex)
        if c.shape != (self.n_tau,):
            raise ValueError(f"Expected shape ({self.n_tau},), got {c.shape}")
        return self._cf2it @ c

    def gtau_to_giw(self, g59c: np.ndarray) -> np.ndarray:
        return self._cf2if @ self.coeffs_from_gtau(g59c)

    def giw_to_gtau(self, giw: np.ndarray) -> np.ndarray:
        g = np.asarray(giw, dtype=complex)
        if g.shape != (self.n_tau,):
            raise ValueError(f"Expected shape ({self.n_tau},), got {g.shape}")
        coeffs = np.linalg.solve(self._cf2if, g)
        return self.gtau_from_coeffs(coeffs)

    def eval_at_tau(self, g59c: np.ndarray, tau: float) -> complex:
        coeffs = self.coeffs_from_gtau(g59c)
        omega = self.real_frequency_nodes
        tau = float(tau)
        log_kernel_abs = -omega * tau - np.logaddexp(0.0, -self.beta * omega)
        kernel = -np.exp(log_kernel_abs)
        return complex(kernel @ coeffs)
