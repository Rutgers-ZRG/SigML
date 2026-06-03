import os
import tempfile

import numpy as np
import torch

from sigml.solver.dataset import SolverDataset


def test_dataset_item():
    n = 5
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "data.npz")
        U = np.linspace(1.0, 5.0, n)
        np.savez(
            path,
            delta=np.random.randn(n, 118),
            g=np.random.randn(n, 118),
            U=U,
            mu=0.5 * U,
            beta=np.full(n, 70.0),
            eps_d=np.zeros(n),
        )
        ds = SolverDataset(path)
        item = ds[0]
        assert len(ds) == n
        assert item["x"].shape == (121,)
        assert item["y"].shape == (118,)
        assert torch.is_tensor(item["x"])
        assert torch.is_tensor(item["y"])
        assert item["x"][-3:].tolist() == [1.0, 0.5, 70.0]
