import numpy as np

from sigml.solver.metrics import (
    g_mse,
    orbital_occupation,
    positive_freq_causality_rate,
    quasiparticle_proxy,
)
from sigml.solver.valenti_grid import ValentiOrb1Grid


MESH = "/Users/li/dev/RA/dmft/ref/mlDMFT/mldmft/models/orb1/mesh_beta70.h5"


def test_g_mse():
    pred = np.array([1.0, 2.0, 3.0])
    target = np.array([1.0, 0.0, 5.0])
    assert g_mse(pred, target) == np.mean((pred - target) ** 2)


def test_positive_freq_causality_rate():
    iw = 1j * np.array([-3.0, -1.0, 1.0, 3.0])
    sigma = np.array(
        [
            [1.0j, 1.0j, -0.1j, -0.2j],
            [1.0j, 1.0j, -0.1j, 0.2j],
        ]
    )
    assert positive_freq_causality_rate(sigma, iw) == 0.5


def test_matrix_positive_freq_causality_rate_uses_eigenvalues():
    iw = 1j * np.array([-3.0, -1.0, 1.0, 3.0])
    sigma = np.zeros((2, 4, 3, 3), dtype=complex)
    sigma[0, 2:] = np.array([np.diag([-0.1j, -0.2j, -0.3j]), np.diag([-0.2j, -0.3j, -0.4j])])
    sigma[1, 2:] = sigma[0, 2:]
    sigma[1, 3] = np.diag([-0.2j, 0.01j, -0.4j])
    assert positive_freq_causality_rate(sigma, iw) == 0.5


def test_quasiparticle_proxy_uses_dlr_eval():
    grid = ValentiOrb1Grid(MESH, beta=70.0)
    gtau = -np.exp(-1.0 * grid.tau_nodes) / (1 + np.exp(-70.0))
    assert quasiparticle_proxy(gtau, grid, beta=70.0) == -grid.eval_at_tau(gtau, 35.0).real


def test_matrix_quasiparticle_proxy_and_occupation_are_blockwise():
    grid = ValentiOrb1Grid(MESH, beta=70.0)
    gtau = np.zeros((3, 3, grid.n_tau), dtype=complex)
    for i in range(3):
        gtau[i, i] = -(i + 1) * np.exp(-grid.tau_nodes) / (1 + np.exp(-70.0))
    z = quasiparticle_proxy(gtau, grid, beta=70.0)
    occ = orbital_occupation(gtau)

    assert z.shape == (3, 3)
    assert occ.shape == (3, 3)
    np.testing.assert_allclose(z[2, 2], -grid.eval_at_tau(gtau[2, 2], 35.0).real, atol=1e-14)
    np.testing.assert_allclose(occ, -gtau[..., -1].real)
