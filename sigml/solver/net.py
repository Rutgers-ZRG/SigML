import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedforwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(121, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 118)

        self.alpha1 = nn.Linear(3, 10)
        self.alpha2 = nn.Linear(10, 256)
        self.alpha3 = nn.Linear(10, 512)

        self.eps1 = nn.Parameter(torch.tensor(0.5))
        self.eps2 = nn.Parameter(torch.tensor(0.5))
        self.eps3 = nn.Parameter(torch.tensor(0.5))

    def forward(self, input):
        params = input[:, -3:]
        x_params = F.relu(self.alpha1(params))
        x = F.relu(self.fc1(input))
        x = F.relu((1 - self.eps1) * self.fc2(x) + self.eps1 * (self.alpha2(x_params)))
        x = F.gelu((1 - self.eps2) * self.fc3(x) + self.eps2 * (self.alpha3(x_params)))
        x = F.gelu((1 - self.eps3) * self.fc4(x) + self.eps3 * (self.alpha3(x_params)))

        return self.fc5(x)


def Gfuncloss(outputs, labels, omegas, alpha=0.0001):
    bs = outputs.size(0)
    return (torch.norm(outputs - labels)) / float(bs)


def block_features_to_matrix(
    features: torch.Tensor, *, orbital_dim: int, n_tau: int
) -> torch.Tensor:
    """Convert flat real/imag block features to complex (N, M, M, N_tau)."""
    expected = orbital_dim * orbital_dim * n_tau * 2
    if features.ndim != 2 or features.shape[1] != expected:
        raise ValueError(f"features must have shape (batch, {expected}), got {tuple(features.shape)}")
    paired = features.reshape(features.shape[0], orbital_dim, orbital_dim, n_tau, 2)
    return torch.complex(paired[..., 0], paired[..., 1])


def matrix_to_block_features(blocks: torch.Tensor) -> torch.Tensor:
    """Convert complex (N, M, M, N_tau) blocks to flat real/imag features."""
    if blocks.ndim != 4 or blocks.shape[1] != blocks.shape[2]:
        raise ValueError(f"blocks must have shape (batch, M, M, N_tau), got {tuple(blocks.shape)}")
    paired = torch.stack((blocks.real, blocks.imag), dim=-1)
    return paired.reshape(blocks.shape[0], -1)


def hermitianize_block_features(
    features: torch.Tensor, *, orbital_dim: int, n_tau: int
) -> torch.Tensor:
    blocks = block_features_to_matrix(features, orbital_dim=orbital_dim, n_tau=n_tau)
    hermitian = 0.5 * (blocks + blocks.transpose(1, 2).conj())
    return matrix_to_block_features(hermitian)


class ScalarConditionedResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, scalar_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.scalar = nn.Linear(scalar_dim, hidden_dim)
        self.gate = nn.Linear(scalar_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        conditioned = self.linear(x) + self.scalar(scalars)
        gated = torch.sigmoid(self.gate(scalars))
        update = F.gelu(conditioned) * gated
        return self.norm(x + update)


class BlockResNet(nn.Module):
    """Flatten-ResNet baseline for MxM Hermitian block solver I/O.

    The network consumes flattened complex Delta blocks followed by scalar
    conditioning channels and returns flattened complex G blocks. The final
    symmetrization enforces Hermiticity by construction.
    """

    def __init__(
        self,
        *,
        orbital_dim: int = 3,
        n_tau: int = 59,
        scalar_dim: int = 4,
        hidden_dim: int = 512,
        num_layers: int = 4,
    ):
        super().__init__()
        self.orbital_dim = int(orbital_dim)
        self.n_tau = int(n_tau)
        self.scalar_dim = int(scalar_dim)
        self.block_dim = self.orbital_dim * self.orbital_dim * self.n_tau * 2
        self.input_dim = self.block_dim + self.scalar_dim
        self.output_dim = self.block_dim

        self.input = nn.Linear(self.input_dim, hidden_dim)
        self.scalar_embed = nn.Linear(self.scalar_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ScalarConditionedResidualBlock(hidden_dim, self.scalar_dim) for _ in range(num_layers)]
        )
        self.output = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim != 2 or input.shape[1] != self.input_dim:
            raise ValueError(f"input must have shape (batch, {self.input_dim}), got {tuple(input.shape)}")
        scalars = input[:, -self.scalar_dim :]
        x = F.gelu(self.input(input) + self.scalar_embed(scalars))
        for block in self.blocks:
            x = block(x, scalars)
        raw = self.output(x)
        return hermitianize_block_features(raw, orbital_dim=self.orbital_dim, n_tau=self.n_tau)
