import h5py
import numpy as np

from sigml.solver.valenti_grid import ValentiOrb1Grid


MESH = "/Users/li/dev/RA/dmft/ref/mlDMFT/mldmft/models/orb1/mesh_beta70.h5"


def test_tau_nodes_match_saved_mesh():
    grid = ValentiOrb1Grid(MESH, beta=70.0)
    assert grid.n_tau == 59 and grid.feature_dim == 118
    assert np.all(grid.tau_nodes >= 0) and np.all(grid.tau_nodes <= 70.0 + 1e-9)


def test_vec_roundtrip():
    grid = ValentiOrb1Grid(MESH, beta=70.0)
    g = np.random.randn(59) + 1j * np.random.randn(59)
    np.testing.assert_allclose(grid.vec_to_gtau(grid.gtau_to_vec(g)), g, atol=1e-12)


def test_tau_to_iw_roundtrip():
    grid = ValentiOrb1Grid(MESH, beta=70.0)
    w0 = 1.1
    gtau = -np.exp(-w0 * grid.tau_nodes) / (1 + np.exp(-70.0 * w0))
    giw = grid.gtau_to_giw(gtau.astype(complex))
    gtau_rt = grid.giw_to_gtau(giw)
    np.testing.assert_allclose(gtau_rt, gtau, atol=1e-6)
