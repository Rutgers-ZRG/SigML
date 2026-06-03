import numpy as np

from sigml.solver.hybridization import sample_delta_vec
from sigml.solver.valenti_grid import ValentiOrb1Grid


MESH = "/Users/li/dev/RA/dmft/ref/mlDMFT/mldmft/models/orb1/mesh_beta70.h5"


def test_sampled_delta_causal_positive_freq():
    grid = ValentiOrb1Grid(MESH, beta=70.0)
    rng = np.random.default_rng(0)
    vec = sample_delta_vec(grid, rng, alpha=0.2)
    assert vec.shape == (118,)
    assert np.all(np.isfinite(vec))

    giw = grid.gtau_to_giw(grid.vec_to_gtau(vec))
    iw = grid.iw_nodes
    assert np.all(np.isfinite(giw))
    assert np.all(giw[iw.imag > 0].imag <= 1e-9)


def test_sampled_delta_many_sample_causality_and_tau_sign():
    grid = ValentiOrb1Grid(MESH, beta=70.0)
    rng = np.random.default_rng(20260603)
    pos_iw = grid.iw_nodes.imag > 0

    for _ in range(200):
        vec = sample_delta_vec(grid, rng, alpha=0.2)
        delta_tau = grid.vec_to_gtau(vec)
        delta_iw = grid.gtau_to_giw(delta_tau)

        assert np.all(delta_iw[pos_iw].imag <= 1e-9)
        assert np.all((-delta_tau.real) >= -1e-10)
        assert np.max(np.abs(delta_tau.imag)) <= 1e-10


def test_sampled_delta_distribution_is_not_degenerate():
    grid = ValentiOrb1Grid(MESH, beta=70.0)
    rng = np.random.default_rng(20260604)

    curves = np.array(
        [grid.vec_to_gtau(sample_delta_vec(grid, rng, alpha=0.2)).real for _ in range(200)]
    )
    coeffs = np.array([grid.coeffs_from_gtau(curve.astype(complex)).real for curve in curves])
    weight_sums = coeffs.sum(axis=1)
    weighted_indices = coeffs @ np.arange(1, grid.n_tau + 1) / weight_sums

    assert np.ptp(np.linalg.norm(coeffs, axis=1)) > 1e-3
    assert np.ptp(weight_sums) > 1e-3
    assert np.ptp(weighted_indices) > 1.0
    assert np.unique(np.round(weighted_indices, decimals=4)).size > 100
