import pytest
import torch

from sigml.solver.nn_solver_cli import load_model
from sigml.solver.net import (
    BlockResNet,
    FeedforwardNet,
    OrbitalIrrepNet,
    block_features_to_matrix,
    matrix_to_block_features,
    positive_matsubara_causality_penalty,
)


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


def _random_symmetric_block_features(batch: int, orbital_dim: int, n_tau: int) -> torch.Tensor:
    raw = torch.randn(batch, orbital_dim, orbital_dim, n_tau)
    blocks = 0.5 * (raw + raw.transpose(1, 2))
    return matrix_to_block_features(blocks.to(torch.complex64))


def _random_rotation(orbital_dim: int) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(orbital_dim, orbital_dim))
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = q * signs
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_orbital_irrep_net_shapes_backward_and_hermitian_output():
    m = 3
    n_tau = 4
    net = OrbitalIrrepNet(orbital_dim=m, n_tau=n_tau, hidden_channels=4, num_layers=2)
    x = torch.cat((_random_symmetric_block_features(3, m, n_tau), torch.randn(3, 4)), dim=1)
    y = net(x)
    assert y.shape == (3, m * m * n_tau * 2)

    blocks = block_features_to_matrix(y, orbital_dim=m, n_tau=n_tau)
    assert torch.allclose(blocks, blocks.transpose(1, 2).conj(), atol=1e-6)
    assert torch.max(torch.abs(blocks.imag)) <= 1e-6

    loss = y.square().mean()
    loss.backward()
    assert all(param.grad is not None for param in net.parameters() if param.requires_grad)


def test_orbital_irrep_net_is_numerically_equivariant_under_so3_conjugation():
    m = 3
    n_tau = 5
    net = OrbitalIrrepNet(orbital_dim=m, n_tau=n_tau, hidden_channels=5, num_layers=3)
    net.eval()
    features = _random_symmetric_block_features(2, m, n_tau)
    scalars = torch.tensor([[4.0, 0.45, 40.0, 0.65], [3.5, 0.50, 40.0, 0.55]])
    x = torch.cat((features, scalars), dim=1)
    q = _random_rotation(m)
    q_complex = q.to(torch.complex64)

    delta = block_features_to_matrix(features, orbital_dim=m, n_tau=n_tau)
    rotated_delta = torch.einsum("ab,nbct,dc->nadt", q_complex, delta, q_complex)
    rotated_x = torch.cat((matrix_to_block_features(rotated_delta), scalars), dim=1)

    with torch.no_grad():
        y = net(x)
        y_rotated_input = net(rotated_x)

    g = block_features_to_matrix(y, orbital_dim=m, n_tau=n_tau)
    expected_rotated_g = torch.einsum("ab,nbct,dc->nadt", q_complex, g, q_complex)
    actual_rotated_g = block_features_to_matrix(y_rotated_input, orbital_dim=m, n_tau=n_tau)
    assert torch.allclose(actual_rotated_g, expected_rotated_g, atol=2e-5, rtol=2e-5)


def test_positive_matsubara_causality_penalty_uses_positive_imaginary_eigenvalues():
    iw = 1j * torch.tensor([-3.0, -1.0, 1.0, 3.0])
    good = torch.diag_embed(
        torch.tensor(
            [
                [0.2 + 0.1j, 0.3 + 0.2j],
                [0.2 + 0.1j, 0.3 + 0.2j],
                [0.2 - 0.1j, 0.3 - 0.2j],
                [0.2 - 0.1j, 0.3 - 0.2j],
            ]
        )
    )
    bad = good.clone()
    bad[2, 0, 0] = 0.2 + 0.4j
    assert positive_matsubara_causality_penalty(good, iw) == pytest.approx(0.0)
    assert positive_matsubara_causality_penalty(bad, iw) > 0.0


def test_orbital_irrep_checkpoint_loader_round_trips(tmp_path):
    net = OrbitalIrrepNet(orbital_dim=3, n_tau=3, hidden_channels=4, num_layers=1)
    checkpoint = {
        "architecture": "e3nn-irrep",
        "model_state_dict": net.state_dict(),
        "input_dim": net.input_dim,
        "output_dim": net.output_dim,
        "orbital_dim": 3,
        "n_tau": 3,
        "scalar_dim": 4,
        "hidden_dim": 4,
        "num_layers": 1,
    }
    path = tmp_path / "e3nn_irrep.pth"
    torch.save(checkpoint, path)
    loaded = load_model(path, input_dim=net.input_dim, output_dim=net.output_dim)
    x = torch.cat((_random_symmetric_block_features(2, 3, 3), torch.randn(2, 4)), dim=1)
    with torch.no_grad():
        assert torch.allclose(loaded(x), net(x), atol=1e-6)
