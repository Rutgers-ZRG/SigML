import torch

from sigml.solver.net import BlockResNet, FeedforwardNet, block_features_to_matrix


CKPT = "/Users/li/dev/RA/dmft/ref/mlDMFT/mldmft/models/orb1/save_3000.pth"


def test_io_shapes():
    net = FeedforwardNet()
    y = net(torch.randn(4, 121))
    assert y.shape == (4, 118)


def test_strict_load_orb1():
    net = FeedforwardNet()
    state = torch.load(CKPT, map_location="cpu")
    net.load_state_dict(state, strict=True)


def test_block_resnet_m3_shapes_backward_and_hermitian_output():
    m = 3
    n_tau = 5
    net = BlockResNet(orbital_dim=m, n_tau=n_tau, hidden_dim=32, num_layers=2)
    x = torch.randn(4, m * m * n_tau * 2 + 4)
    y = net(x)
    assert y.shape == (4, m * m * n_tau * 2)

    blocks = block_features_to_matrix(y, orbital_dim=m, n_tau=n_tau)
    assert torch.allclose(blocks, blocks.transpose(1, 2).conj(), atol=1e-6)
    assert torch.max(torch.abs(blocks.diagonal(dim1=1, dim2=2).imag)) <= 1e-6

    loss = y.square().mean()
    loss.backward()
    assert all(param.grad is not None for param in net.parameters() if param.requires_grad)
