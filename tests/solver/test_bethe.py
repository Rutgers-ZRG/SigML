import numpy as np

from sigml.solver.bethe import dmft_loop, make_analytic_noninteracting_solver
from sigml.solver.valenti_grid import ValentiOrb1Grid


MESH = "/Users/li/dev/RA/dmft/ref/mlDMFT/mldmft/models/orb1/mesh_beta70.h5"


def test_bethe_loop_analytic_noninteracting():
    grid = ValentiOrb1Grid(MESH, beta=70.0)
    solver = make_analytic_noninteracting_solver(grid)
    res = dmft_loop(
        solver,
        U=0.0,
        mu=0.0,
        beta=70.0,
        t=1.0,
        grid=grid,
        mix=0.5,
        tol=1e-5,
        max_iter=200,
    )
    assert res.converged
    giw = grid.gtau_to_giw(grid.vec_to_gtau(res.g_vec))
    iw = grid.iw_nodes
    m = iw.imag > 0
    resid = giw[m] - 1.0 / (iw[m] - 1.0 * giw[m])
    assert np.max(np.abs(resid)) < 1e-3
