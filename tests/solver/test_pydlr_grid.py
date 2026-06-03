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


def test_pydlr_beta40_round_trips_matrix_blocks_with_valenti_like_interface():
    grid = PydlrGrid(beta=40.0, lamb=600.0, eps=1e-10)
    assert grid.beta == 40.0
    assert grid.n_tau == grid.rank
    assert grid.feature_dim == 2 * grid.n_tau

    gtau = np.empty((3, 3, grid.n_tau), dtype=complex)
    for i in range(3):
        for j in range(3):
            weight = (i + 1) - 0.2j * (j + 1)
            pole = 0.25 + 0.08 * i + 0.03 * j
            gtau[i, j] = weight * np.exp(-pole * grid.tau_nodes)

    coeffs = grid.coeffs_from_gtau(gtau)
    roundtrip = grid.gtau_from_coeffs(coeffs)
    giw = grid.gtau_to_giw(gtau)
    gtau_from_iw = grid.giw_to_gtau(giw)
    vec = grid.gtau_to_vec(gtau)

    assert coeffs.shape == gtau.shape
    assert giw.shape == gtau.shape
    assert vec.shape == (3, 3, grid.feature_dim)
    np.testing.assert_allclose(grid.vec_to_gtau(vec), gtau, atol=1e-12)
    np.testing.assert_allclose(roundtrip, gtau, atol=1e-9)
    np.testing.assert_allclose(gtau_from_iw, gtau, atol=1e-8)
    mid = grid.eval_at_tau(gtau, 20.0)
    assert mid.shape == (3, 3)
    assert np.all(np.isfinite(mid))
    np.testing.assert_allclose(mid[1, 2], grid.eval_at_tau(gtau[1, 2], 20.0), atol=1e-12)


def test_pydlr_mesh_differs_from_valenti_guard():
    p = PydlrGrid(beta=70.0, lamb=700.0, eps=1e-13)
    v = ValentiOrb1Grid(MESH, beta=70.0)
    assert p.tau_nodes.shape == v.tau_nodes.shape
    assert np.max(np.abs(np.sort(p.tau_nodes) - np.sort(v.tau_nodes))) > 1e-3


def test_valenti_grid_handles_matrix_blocks_per_component():
    grid = ValentiOrb1Grid(MESH, beta=70.0)
    gtau = np.empty((3, 3, grid.n_tau), dtype=complex)
    for i in range(3):
        for j in range(3):
            weight = (i + 1) + 0.1j * (j + 1)
            gtau[i, j] = weight * np.exp(-(0.5 + 0.1 * i + 0.03 * j) * grid.tau_nodes)

    coeffs = grid.coeffs_from_gtau(gtau)
    roundtrip = grid.gtau_from_coeffs(coeffs)
    giw = grid.gtau_to_giw(gtau)
    gtau_from_iw = grid.giw_to_gtau(giw)
    mid = grid.eval_at_tau(gtau, 35.0)

    assert coeffs.shape == gtau.shape
    assert giw.shape == gtau.shape
    assert mid.shape == (3, 3)
    np.testing.assert_allclose(roundtrip, gtau, atol=1e-10)
    np.testing.assert_allclose(gtau_from_iw, gtau, atol=1e-8)
    np.testing.assert_allclose(mid[1, 2], grid.eval_at_tau(gtau[1, 2], 35.0), atol=1e-12)
