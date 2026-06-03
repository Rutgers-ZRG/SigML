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


def test_dataset_m3_tau_blocks_include_j_and_preserve_hermiticity():
    n = 2
    m = 3
    n_tau = 59
    rng = np.random.default_rng(0)
    delta = rng.normal(size=(n, m, m, n_tau)) + 1j * rng.normal(size=(n, m, m, n_tau))
    g = rng.normal(size=(n, m, m, n_tau)) + 1j * rng.normal(size=(n, m, m, n_tau))
    delta = 0.5 * (delta + np.swapaxes(delta.conj(), 1, 2))
    g = 0.5 * (g + np.swapaxes(g.conj(), 1, 2))

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m3.npz")
        U = np.array([4.0, 5.0], dtype=np.float32)
        mu = np.array([2.0, 2.5], dtype=np.float32)
        beta = np.full(n, 70.0, dtype=np.float32)
        J = np.array([0.6, 0.7], dtype=np.float32)
        np.savez(path, delta=delta, g=g, U=U, mu=mu, beta=beta, J=J)

        ds = SolverDataset(path)
        item = ds[0]

        assert ds.orbital_dim == 3
        assert ds.delta_tau.shape == (n, m, m, n_tau)
        assert ds.g_tau.shape == (n, m, m, n_tau)
        np.testing.assert_allclose(ds.delta_tau, np.swapaxes(ds.delta_tau.conj(), 1, 2))
        np.testing.assert_allclose(ds.g_tau, np.swapaxes(ds.g_tau.conj(), 1, 2))
        assert item["x"].shape == (m * m * 2 * n_tau + 4,)
        assert item["y"].shape == (m * m * 2 * n_tau,)
        assert item["x"][-4:].tolist() == [4.0, 0.5, 70.0, 0.6000000238418579]
