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
