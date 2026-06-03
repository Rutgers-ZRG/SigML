import numpy as np

from sigml.solver.pydlr_grid import PydlrGrid
from sigml.solver.valenti_grid import ValentiOrb1Grid


MESH = "/Users/li/dev/RA/dmft/ref/mlDMFT/mldmft/models/orb1/mesh_beta70.h5"


def test_pydlr_rank_and_single_pole_sign():
    grid = PydlrGrid(beta=70.0, lamb=700.0, eps=1e-13)
    assert grid.rank == 59
    w0 = 1.3
    gtau = -np.exp(-w0 * grid.tau_nodes) / (1 + np.exp(-70.0 * w0))
    iw, giw = grid.giw_positive(grid.coeffs_from_gtau(gtau))
    np.testing.assert_allclose(giw, 1.0 / (iw - w0), atol=1e-6)
    assert np.all(giw.imag <= 1e-9)


def test_pydlr_mesh_differs_from_valenti_guard():
    p = PydlrGrid(beta=70.0, lamb=700.0, eps=1e-13)
    v = ValentiOrb1Grid(MESH, beta=70.0)
    assert p.tau_nodes.shape == v.tau_nodes.shape
    assert np.max(np.abs(np.sort(p.tau_nodes) - np.sort(v.tau_nodes))) > 1e-3
