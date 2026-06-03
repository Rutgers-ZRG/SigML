from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from sigml.solver.net import FeedforwardNet
from sigml.solver.valenti_grid import DEFAULT_MESH_PATH, ValentiOrb1Grid


DEFAULT_CKPT_PATH = (
    "/Users/li/dev/RA/dmft/ref/mlDMFT/mldmft/models/orb1/save_3000.pth"
)


class Orb1Oracle:
    """Valenti orb1 neural-network impurity-solver oracle."""

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CKPT_PATH,
        mesh_path: str | Path = DEFAULT_MESH_PATH,
        device: str | torch.device = "cpu",
    ):
        self.checkpoint_path = str(checkpoint_path)
        self.device = torch.device(device)
        self.grid = ValentiOrb1Grid(mesh_path, beta=70.0)
        self.net = FeedforwardNet().to(self.device)
        state = torch.load(self.checkpoint_path, map_location=self.device)
        self.net.load_state_dict(state, strict=True)
        self.net.eval()

    def solve(
        self,
        delta_vec: np.ndarray,
        U: float,
        mu: float,
        beta: float,
        eps_d: float = 0.0,
    ) -> np.ndarray:
        delta = np.asarray(delta_vec, dtype=np.float32)
        if delta.shape != (self.grid.feature_dim,):
            raise ValueError(f"Expected delta_vec shape ({self.grid.feature_dim},), got {delta.shape}")
        if not np.isclose(float(beta), 70.0):
            raise ValueError(
                "Orb1Oracle.solve supports beta=70.0 only for the orb1 mesh; "
                "general beta support requires porting Valenti switch_mesh."
            )
        if U == 0:
            raise ValueError("U must be nonzero because the Valenti input uses (mu - eps_d) / U")
        scalars = np.array([U, (mu - eps_d) / U, beta], dtype=np.float32)
        x = np.concatenate([delta, scalars], axis=0)[None, :]
        with torch.no_grad():
            y = self.net(torch.from_numpy(x).to(self.device)).detach().cpu().numpy()
        return y[0]
