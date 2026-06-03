import numpy as np
import pytest

from sigml.solver.hybridization import sample_delta_vec
from sigml.solver.labeler import CtsegLabeler, OracleLabeler
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
