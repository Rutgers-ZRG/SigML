import numpy as np

from sigml.solver.hybridization import (
    T2G_BROAD_RANGES,
    T2GBathSample,
    sample_t2g_bath,
)
from sigml.solver.pydlr_grid import PydlrGrid
from sigml.solver.valenti_grid import ValentiOrb1Grid


MESH = "/Users/li/dev/RA/dmft/ref/mlDMFT/mldmft/models/orb1/mesh_beta70.h5"


def _assert_hermitian_causal_roundtrip(grid, sample, tol=1e-8):
    assert sample.delta.shape == (3, 3, grid.n_tau)
    assert sample.eps_d.shape == (3, 3)
    np.testing.assert_allclose(sample.delta, np.swapaxes(sample.delta.conj(), 0, 1), atol=tol)
    np.testing.assert_allclose(sample.eps_d, sample.eps_d.conj().T, atol=tol)

    coeffs = grid.coeffs_from_gtau(sample.delta)
    np.testing.assert_allclose(grid.gtau_from_coeffs(coeffs), sample.delta, atol=tol)

    delta_iw = grid.gtau_to_giw(sample.delta)
    pos = grid.iw_nodes.imag > 0
    for block in np.moveaxis(delta_iw[..., pos], -1, 0):
        eigvals = np.linalg.eigvals(block)
        assert np.max(eigvals.imag) <= 1e-7


def test_sample_t2g_bath_broad_covers_documented_ranges_and_is_causal():
    grid = PydlrGrid(beta=40.0, lamb=600.0, eps=1e-10)
    rng = np.random.default_rng(20260603)

    samples = [sample_t2g_bath(grid, rng, mode="broad") for _ in range(96)]

    for sample in samples:
        assert sample.mode == "broad"
        _assert_hermitian_causal_roundtrip(grid, sample)

    for name in ("U", "J", "beta", "mu"):
        values = np.asarray([getattr(sample, name) for sample in samples])
        lo, hi = T2G_BROAD_RANGES[name]
        assert np.min(values) >= lo
        assert np.max(values) <= hi
        assert np.min(values) <= lo + 0.25 * (hi - lo)
        assert np.max(values) >= hi - 0.25 * (hi - lo)


def test_sample_t2g_bath_defaults_stage2_beta_to_grid_beta40():
    grid = PydlrGrid(beta=40.0, lamb=600.0, eps=1e-10)
    rng = np.random.default_rng(11)
    samples = [sample_t2g_bath(grid, rng, mode="broad") for _ in range(12)]
    assert {sample.beta for sample in samples} == {40.0}


def test_sample_t2g_bath_warm_mode_centers_on_stub_trajectory():
    grid = PydlrGrid(beta=40.0, lamb=600.0, eps=1e-10)
    rng = np.random.default_rng(7)
    center = sample_t2g_bath(grid, rng, mode="broad")
    trajectory = [
        T2GBathSample(
            delta=center.delta,
            eps_d=center.eps_d,
            U=4.2,
            J=0.58,
            beta=40.0,
            mu=2.1,
            mode="stub",
        )
    ]

    warm = [
        sample_t2g_bath(
            grid,
            rng,
            mode="warm",
            trajectory=trajectory,
            neighborhood=0.05,
        )
        for _ in range(24)
    ]

    for sample in warm:
        assert sample.mode == "warm"
        _assert_hermitian_causal_roundtrip(grid, sample)

    center_norm = np.linalg.norm(center.delta)
    broad = [sample_t2g_bath(grid, rng, mode="broad") for _ in range(24)]
    warm_distance = np.mean([np.linalg.norm(sample.delta - center.delta) for sample in warm])
    broad_distance = np.mean([np.linalg.norm(sample.delta - center.delta) for sample in broad])

    assert warm_distance < 0.12 * max(center_norm, 1e-12)
    assert warm_distance < 0.25 * broad_distance
    assert abs(np.mean([sample.U for sample in warm]) - 4.2) < 0.08
    assert abs(np.mean([sample.J for sample in warm]) - 0.58) < 0.03
    assert abs(np.mean([sample.mu for sample in warm]) - 2.1) < 0.08
    assert {sample.beta for sample in warm} == {40.0}
