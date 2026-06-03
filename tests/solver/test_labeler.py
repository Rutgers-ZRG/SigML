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
    lab = CtsegLabeler()
    with pytest.raises(NotImplementedError, match="amarel3-triqs-setup.md"):
        lab.solve(delta_vec=np.zeros(118), U=3.0, mu=1.5, beta=70.0, eps_d=0.0)
