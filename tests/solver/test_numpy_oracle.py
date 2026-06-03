from pathlib import Path

import numpy as np
import torch

from sigml.solver.net import FeedforwardNet
from sigml.solver.numpy_oracle import NumpyOrb1Oracle


CKPT = "/Users/li/dev/RA/dmft/ref/mlDMFT/mldmft/models/orb1/save_3000.pth"
MESH = "/Users/li/dev/RA/dmft/ref/mlDMFT/mldmft/models/orb1/mesh_beta70.h5"


def test_numpy_oracle_matches_torch(tmp_path: Path):
    state = torch.load(CKPT, map_location="cpu")
    weights = tmp_path / "orb1_weights.npz"
    np.savez(weights, **{key: value.detach().cpu().numpy() for key, value in state.items()})

    rng = np.random.default_rng(0)
    x = rng.normal(size=121).astype(np.float32)
    x[-3:] = np.array([3.0, 0.5, 70.0], dtype=np.float32)

    net = FeedforwardNet()
    net.load_state_dict(state, strict=True)
    net.eval()
    with torch.no_grad():
        torch_out = net(torch.from_numpy(x[None, :])).numpy()[0]

    oracle = NumpyOrb1Oracle(weights, mesh_path=MESH)
    numpy_out = oracle.solve(x[:-3], U=3.0, mu=1.5, beta=70.0)
    np.testing.assert_allclose(numpy_out, torch_out, rtol=3e-5, atol=3e-5)
