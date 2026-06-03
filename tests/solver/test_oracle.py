import importlib.util
from pathlib import Path

import numpy as np
import torch

from sigml.solver.net import FeedforwardNet
from sigml.solver.oracle import Orb1Oracle


REF_NET = Path("/Users/li/dev/RA/dmft/ref/mlDMFT/mldmft/models/orb1/NNet.py")
CKPT = "/Users/li/dev/RA/dmft/ref/mlDMFT/mldmft/models/orb1/save_3000.pth"
EXAMPLE_INPUT = (
    "/Users/li/dev/RA/dmft/ref/mlDMFT/mldmft/examples/1orbital/NN/"
    "example_inputs/inputs.npy"
)


def _reference_net_class():
    spec = importlib.util.spec_from_file_location("valenti_orb1_nnet", REF_NET)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.FeedforwardNet


def _loaded_pair():
    state = torch.load(CKPT, map_location="cpu")
    ref_net = _reference_net_class()()
    port_net = FeedforwardNet()
    ref_net.load_state_dict(state, strict=True)
    port_net.load_state_dict(state, strict=True)
    ref_net.eval()
    port_net.eval()
    return ref_net, port_net


def test_orb1_port_matches_reference_code_on_random_input():
    ref_net, port_net = _loaded_pair()
    torch.manual_seed(20260603)
    x = torch.randn(3, 121)
    with torch.no_grad():
        ref_out = ref_net(x)
        port_out = port_net(x)
    torch.testing.assert_close(port_out, ref_out, rtol=0.0, atol=0.0)


def test_orb1_port_matches_reference_code_on_example_input():
    ref_net, port_net = _loaded_pair()
    delta = np.load(EXAMPLE_INPUT)
    x = delta.reshape(delta.shape[0], delta.shape[1] * delta.shape[2])
    x = np.concatenate(
        [x, np.array([[7.0, 3.5 / 7.0, 50.0]], dtype=x.dtype)],
        axis=1,
    )
    with torch.no_grad():
        ref_out = ref_net(torch.tensor(x, dtype=torch.float32))
        port_out = port_net(torch.tensor(x, dtype=torch.float32))
    torch.testing.assert_close(port_out, ref_out, rtol=0.0, atol=0.0)


def test_oracle_solve_runs_and_is_causal_positive_freq():
    oracle = Orb1Oracle()
    delta = np.load(EXAMPLE_INPUT).reshape(-1)
    g_vec = oracle.solve(delta_vec=delta, U=7.0, mu=3.5, beta=50.0, eps_d=0.0)
    g59 = oracle.grid.vec_to_gtau(g_vec)
    giw = oracle.grid.gtau_to_giw(g59)
    pos = giw[oracle.grid.iw_nodes.imag > 0]
    assert np.all(pos.imag <= 1e-6)
