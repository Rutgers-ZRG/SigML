from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SolverDataset(Dataset):
    """NPZ-backed solver dataset for Delta/scalar inputs and G-vector labels."""

    def __init__(self, path: str | Path):
        data = np.load(path)
        self.delta = np.asarray(data["delta"])
        self.g = np.asarray(data["g"])
        self.U = np.asarray(data["U"], dtype=np.float32)
        self.mu = np.asarray(data["mu"], dtype=np.float32)
        self.beta = np.asarray(data["beta"], dtype=np.float32)
        self.J = np.asarray(data["J"], dtype=np.float32) if "J" in data else None
        self.eps_d = (
            np.asarray(data["eps_d"], dtype=np.float32)
            if "eps_d" in data
            else np.zeros_like(self.mu, dtype=np.float32)
        )
        self._stored_x = np.asarray(data["x"], dtype=np.float32) if "x" in data else None
        self._stored_y = np.asarray(data["y"], dtype=np.float32) if "y" in data else None
        self._stored_scalar_names = tuple(str(x) for x in data["scalar_names"]) if "scalar_names" in data else None

        self._validate()
        self.delta_tau = self._tau_view(self.delta)
        self.g_tau = self._tau_view(self.g)
        self.delta_dlr = self._dlr_features(self.delta)
        self.g_dlr = self._dlr_features(self.g)

        if self._stored_x is not None or self._stored_y is not None:
            if self._stored_x is None or self._stored_y is None:
                raise ValueError("stored x and y must either both be present or both be omitted")
            if self._stored_y.shape != self.g_dlr.shape:
                raise ValueError(f"stored y must have shape {self.g_dlr.shape}, got {self._stored_y.shape}")
            if self._stored_x.shape[0] != self.delta_dlr.shape[0]:
                raise ValueError(
                    f"stored x first dimension must be {self.delta_dlr.shape[0]}, got {self._stored_x.shape}"
                )
            if self._stored_x.shape[1] <= self.delta_dlr.shape[1]:
                raise ValueError("stored x must include block features followed by scalar channels")
            np.testing.assert_allclose(
                self._stored_x[:, : self.delta_dlr.shape[1]],
                self.delta_dlr,
                rtol=1e-5,
                atol=1e-6,
            )
            np.testing.assert_allclose(self._stored_y, self.g_dlr, rtol=1e-5, atol=1e-6)
            self.x = self._stored_x.astype(np.float32)
            self.y = self._stored_y.astype(np.float32)
            scalar_dim = self.x.shape[1] - self.delta_dlr.shape[1]
            self.scalar_names = (
                self._stored_scalar_names
                if self._stored_scalar_names is not None
                else tuple(f"scalar_{idx}" for idx in range(scalar_dim))
            )
            if len(self.scalar_names) != scalar_dim:
                raise ValueError(
                    f"scalar_names must have length {scalar_dim}, got {len(self.scalar_names)}"
                )
        else:
            eps_scalar = self._scalar_eps_d()
            mu_nn_over_u = (self.mu - eps_scalar) / self.U
            if self.J is None and self.orbital_dim == 1:
                scalars = np.stack((self.U, mu_nn_over_u, self.beta), axis=1)
                self.scalar_names = ("U", "mu_minus_eps_d_over_U", "beta")
            else:
                j = self.J if self.J is not None else np.zeros_like(self.U, dtype=np.float32)
                scalars = np.stack((self.U, mu_nn_over_u, self.beta, j), axis=1)
                self.scalar_names = ("U", "mu_minus_eps_d_over_U", "beta", "J")
            self.x = np.concatenate((self.delta_dlr, scalars), axis=1).astype(np.float32)
            self.y = self.g_dlr.astype(np.float32)

    def _validate(self) -> None:
        n = self.delta.shape[0]
        self.orbital_dim = self._validate_block_array("delta", self.delta)
        g_orbital_dim = self._validate_block_array("g", self.g)
        if g_orbital_dim != self.orbital_dim:
            raise ValueError(
                f"g orbital dimension {g_orbital_dim} does not match delta {self.orbital_dim}"
            )
        if self.g.shape != self.delta.shape:
            raise ValueError(f"g must have shape {self.delta.shape}, got {self.g.shape}")
        for name, arr in {
            "U": self.U,
            "mu": self.mu,
            "beta": self.beta,
        }.items():
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
        if self.eps_d.shape not in ((n,), (n, self.orbital_dim), (n, self.orbital_dim, self.orbital_dim)):
            raise ValueError(
                f"eps_d must have shape ({n},), ({n}, {self.orbital_dim}), "
                f"or ({n}, {self.orbital_dim}, {self.orbital_dim}), got {self.eps_d.shape}"
            )
        if self.J is not None and self.J.shape != (n,):
            raise ValueError(f"J must have shape ({n},), got {self.J.shape}")
        if np.any(self.U == 0.0):
            raise ValueError("U must be nonzero because the input uses (mu - eps_d) / U")

    def _scalar_eps_d(self) -> np.ndarray:
        eps = np.asarray(self.eps_d, dtype=np.float32)
        if eps.ndim == 1:
            return eps
        if eps.ndim == 2:
            return np.mean(eps, axis=1).astype(np.float32)
        return (np.trace(eps, axis1=1, axis2=2).real / eps.shape[1]).astype(np.float32)

    @staticmethod
    def _validate_block_array(name: str, arr: np.ndarray) -> int:
        if arr.ndim == 2 and arr.shape[1] == 118:
            return 1
        if arr.ndim == 4 and arr.shape[1] == arr.shape[2] and arr.shape[3] > 0:
            if not np.allclose(arr, np.swapaxes(arr.conj(), 1, 2)):
                raise ValueError(f"{name} must be Hermitian over its orbital block axes")
            return int(arr.shape[1])
        raise ValueError(
            f"{name} must have shape (n, 118) or (n, M, M, n_tau), got {arr.shape}"
        )

    @staticmethod
    def _dlr_features(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            return np.asarray(arr, dtype=np.float32)
        paired = np.stack((arr.real, arr.imag), axis=-1)
        return np.reshape(paired, (arr.shape[0], -1)).astype(np.float32)

    @staticmethod
    def _tau_view(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 4:
            return np.asarray(arr)
        paired = np.reshape(arr, (arr.shape[0], 1, 1, arr.shape[1] // 2, 2))
        return paired[..., 0] + 1j * paired[..., 1]

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.x[index]),
            "y": torch.from_numpy(self.y[index]),
        }
