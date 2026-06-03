import numpy as np

from sigml.solver.metrics import (
    g_mse,
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


def test_quasiparticle_proxy_uses_dlr_eval():
    grid = ValentiOrb1Grid(MESH, beta=70.0)
    gtau = -np.exp(-1.0 * grid.tau_nodes) / (1 + np.exp(-70.0))
    assert quasiparticle_proxy(gtau, grid, beta=70.0) == -grid.eval_at_tau(gtau, 35.0).real
