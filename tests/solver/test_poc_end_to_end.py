import numpy as np

from sigml.solver.bethe import dmft_loop
from sigml.solver.metrics import quasiparticle_proxy
from sigml.solver.oracle import Orb1Oracle


SEED = (
    "/Users/li/dev/RA/dmft/ref/mlDMFT/mldmft/examples/1orbital/NN/"
    "example_inputs/inputs.npy"
)


def _run_mott_point(oracle, U):
    initial_delta_vec = np.load(SEED).reshape(-1)
    res = dmft_loop(
        oracle.solve,
        U=U,
        mu=U / 2,
        beta=70.0,
        t=1.0,
        grid=oracle.grid,
        mix=0.6,
        tol=1e-4,
        max_iter=300,
        initial_delta_vec=initial_delta_vec,
    )
    g59 = oracle.grid.vec_to_gtau(res.g_vec)
    return quasiparticle_proxy(g59, oracle.grid, beta=70.0), res


def test_mott_collapse():
    oracle = Orb1Oracle()
    metal, metal_res = _run_mott_point(oracle, 1.0)
    ins, ins_res = _run_mott_point(oracle, 6.0)
    assert metal_res.converged
    assert ins_res.converged
    assert 0.01 < metal < 0.02
    assert 0.0 < ins < 0.003
    assert metal > ins
