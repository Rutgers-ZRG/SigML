import torch

from sigml.solver.net import FeedforwardNet


CKPT = "/Users/li/dev/RA/dmft/ref/mlDMFT/mldmft/models/orb1/save_3000.pth"


def test_io_shapes():
    net = FeedforwardNet()
    y = net(torch.randn(4, 121))
    assert y.shape == (4, 118)


def test_strict_load_orb1():
    net = FeedforwardNet()
    state = torch.load(CKPT, map_location="cpu")
    net.load_state_dict(state, strict=True)
