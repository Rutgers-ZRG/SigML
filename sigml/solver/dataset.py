from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SolverDataset(Dataset):
    """NPZ-backed solver dataset for Delta/scalar inputs and G-vector labels."""

    def __init__(self, path: str | Path):
        data = np.load(path)
        self.delta = np.asarray(data["delta"], dtype=np.float32)
        self.g = np.asarray(data["g"], dtype=np.float32)
        self.U = np.asarray(data["U"], dtype=np.float32)
        self.mu = np.asarray(data["mu"], dtype=np.float32)
        self.beta = np.asarray(data["beta"], dtype=np.float32)
        self.eps_d = np.asarray(data["eps_d"], dtype=np.float32)

        self._validate()
        mu_nn_over_u = (self.mu - self.eps_d) / self.U
        scalars = np.stack((self.U, mu_nn_over_u, self.beta), axis=1)
        self.x = np.concatenate((self.delta, scalars), axis=1).astype(np.float32)
        self.y = self.g.astype(np.float32)

    def _validate(self) -> None:
        if self.delta.ndim != 2 or self.delta.shape[1] != 118:
            raise ValueError(f"delta must have shape (n, 118), got {self.delta.shape}")
        if self.g.shape != self.delta.shape:
            raise ValueError(f"g must have shape {self.delta.shape}, got {self.g.shape}")
        n = self.delta.shape[0]
        for name, arr in {
            "U": self.U,
            "mu": self.mu,
            "beta": self.beta,
            "eps_d": self.eps_d,
        }.items():
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
        if np.any(self.U == 0.0):
            raise ValueError("U must be nonzero because the input uses (mu - eps_d) / U")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.x[index]),
            "y": torch.from_numpy(self.y[index]),
        }
