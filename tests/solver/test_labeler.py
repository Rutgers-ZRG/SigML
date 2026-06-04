import numpy as np
import pytest

from sigml.solver.hybridization import sample_delta_vec
from sigml.solver.labeler import CthybT2GLabeler, CtsegLabeler, OracleLabeler
from sigml.solver.pydlr_grid import PydlrGrid
from sigml.solver.valenti_grid import ValentiOrb1Grid


MESH = "/Users/li/dev/RA/dmft/ref/mlDMFT/mldmft/models/orb1/mesh_beta70.h5"


def test_oracle_labeler_returns_causal_g():
    lab = OracleLabeler()
    grid = ValentiOrb1Grid(MESH, beta=70.0)
    delta_vec = sample_delta_vec(grid, np.random.default_rng(0), alpha=0.05)
    g_vec = lab.solve(delta_vec=delta_vec, U=3.0, mu=1.5, beta=70.0, eps_d=0.0)
    giw = grid.gtau_to_giw(grid.vec_to_gtau(g_vec))
    iw = grid.iw_nodes
    assert g_vec.shape == (118,)
    assert np.all(np.isfinite(g_vec))
    assert np.all(giw[iw.imag > 0].imag <= 1e-6)


def test_ctseg_labeler_documents_phase_c_setup():
    lab = CtsegLabeler(n_tau=16, n_cycles=10, n_warmup_cycles=0)
    with pytest.raises(ImportError, match="triqs_ctseg"):
        lab.solve(delta_vec=np.zeros(118), U=3.0, mu=1.5, beta=70.0, eps_d=0.0)


def test_ctseg_labeler_near_atomic_smoke_if_triqs_available():
    pytest.importorskip("triqs_ctseg")
    grid = ValentiOrb1Grid(MESH, beta=70.0)
    tau = grid.tau_nodes
    beta = 70.0
    bath_v = 0.10
    delta_tau = -(bath_v**2) * np.exp(-0.0 * tau) / (1.0 + np.exp(-beta * 0.0))
    delta_vec = grid.gtau_to_vec(delta_tau.astype(complex)).real

    lab = CtsegLabeler(
        grid=grid,
        n_tau=2001,
        n_cycles=2_000,
        n_warmup_cycles=100,
        projection="direct",
    )
    g_vec = lab.solve(delta_vec=delta_vec, U=2.0, mu=1.0, beta=beta, eps_d=0.0)
    g_tau = grid.vec_to_gtau(g_vec)
    g_iw = grid.gtau_to_giw(g_tau)
    assert g_vec.shape == (grid.feature_dim,)
    assert np.all(np.isfinite(g_vec))
    assert np.all(g_tau.real <= 1e-8)
    assert np.all(g_iw[grid.iw_nodes.imag > 0].imag <= 1e-4)


def test_cthyb_t2g_labeler_documents_solid_dmft_setup():
    grid = PydlrGrid(beta=40.0, lamb=600.0, eps=1e-10)
    lab = CthybT2GLabeler(n_tau=128, n_iw=32, n_cycles=10, n_warmup_cycles=0)
    assert lab.grid.beta == 40.0
    assert lab.grid.n_tau == grid.n_tau
    with pytest.raises(ImportError, match="triqs_cthyb"):
        lab.solve(
            delta_dlr=np.zeros((3, 3, lab.grid.n_tau), dtype=complex),
            U=4.0,
            J=0.6,
            mu=2.0,
            beta=40.0,
            eps_d=np.zeros((3, 3), dtype=complex),
        )


def test_cthyb_t2g_labeler_rejects_non_hermitian_blocks():
    lab = CthybT2GLabeler()
    delta = np.zeros((3, 3, lab.grid.n_tau), dtype=complex)
    delta[0, 1, :] = 0.1
    with pytest.raises(ValueError, match="Hermitian"):
        lab._validate_block_tau("delta_dlr", delta)


def test_cthyb_t2g_labeler_rejects_complex_crystal_field():
    lab = CthybT2GLabeler()
    eps_d = np.zeros((3, 3), dtype=complex)
    eps_d[0, 1] = 0.1j
    eps_d[1, 0] = -0.1j
    with pytest.raises(ValueError, match="real-valued"):
        lab._validate_eps_d(eps_d)


def test_cthyb_t2g_projector_fits_beta40_solver_gtau_to_beta40_grid():
    grid = PydlrGrid(beta=40.0, lamb=600.0, eps=1e-10)
    lab = CthybT2GLabeler(grid=grid, projection="dlr")
    tau_mesh = np.linspace(0.0, 40.0, 2001)
    gtau = np.zeros((3, 3, tau_mesh.size), dtype=complex)
    for orb, pole in enumerate((0.35, 0.5, 0.75)):
        gtau[orb, orb] = -np.exp(-pole * tau_mesh) / (1.0 + np.exp(-grid.beta * pole))

    class _Block:
        def __init__(self, mesh, data):
            self.mesh = mesh
            self.data = data

    solver_g_tau = {
        spin: _Block(tau_mesh, np.moveaxis(gtau, -1, 0).copy())
        for spin in CthybT2GLabeler.spin_blocks
    }

    projected = lab._project_g_tau_to_block_dlr(solver_g_tau)

    assert projected.shape == (3, 3, grid.n_tau)
    np.testing.assert_allclose(projected, np.swapaxes(projected.conj(), 0, 1), atol=1e-10)
    for orb, pole in enumerate((0.35, 0.5, 0.75)):
        expected = -np.exp(-pole * grid.tau_nodes) / (1.0 + np.exp(-grid.beta * pole))
        np.testing.assert_allclose(projected[orb, orb].real, expected, atol=2e-3)
    assert lab.last_direct_g_dlr is not None
    assert lab.last_dlr_g_dlr is not None


def test_cthyb_t2g_near_atomic_smoke_if_triqs_available():
    pytest.importorskip("triqs_cthyb")
    grid = PydlrGrid(beta=40.0, lamb=600.0, eps=1e-10)
    tau = grid.tau_nodes
    beta = 40.0
    delta = np.zeros((3, 3, grid.n_tau), dtype=complex)
    for orb, (bath_v, bath_eps) in enumerate([(0.08, -0.2), (0.10, 0.0), (0.12, 0.25)]):
        delta[orb, orb] = -(bath_v**2) * np.exp(-bath_eps * tau) / (1.0 + np.exp(-beta * bath_eps))
    eps_d = np.diag([-0.05, 0.0, 0.05]).astype(complex)

    lab = CthybT2GLabeler(
        grid=grid,
        n_tau=1201,
        n_iw=200,
        n_cycles=300,
        n_warmup_cycles=20,
        projection="direct",
        solve_kwargs={"verbosity": 0, "random_seed": 20260603},
    )
    g_dlr = lab.solve(delta_dlr=delta, U=2.0, J=0.25, mu=1.0, beta=beta, eps_d=eps_d)
    g_iw = grid.gtau_to_giw(g_dlr)
    pos = grid.iw_nodes.imag > 0

    assert g_dlr.shape == (3, 3, grid.n_tau)
    assert np.all(np.isfinite(g_dlr))
    np.testing.assert_allclose(g_dlr, np.swapaxes(g_dlr.conj(), 0, 1), atol=1e-10)
    for block in np.moveaxis(g_iw[..., pos], -1, 0):
        assert np.max(np.linalg.eigvals(block).imag) <= 1e-3
    assert lab.last_info is not None
    assert lab.last_info.average_sign is not None
