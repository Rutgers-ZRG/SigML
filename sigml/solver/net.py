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


class BlockMLP(nn.Module):
    """Plain MLP baseline for arbitrary MxM block solver I/O."""

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

        layers: list[nn.Module] = []
        left_dim = self.input_dim
        for _ in range(int(num_layers)):
            layers.append(nn.Linear(left_dim, hidden_dim))
            layers.append(nn.GELU())
            left_dim = hidden_dim
        layers.append(nn.Linear(left_dim, self.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim != 2 or input.shape[1] != self.input_dim:
            raise ValueError(f"input must have shape (batch, {self.input_dim}), got {tuple(input.shape)}")
        raw = self.net(input)
        return hermitianize_block_features(raw, orbital_dim=self.orbital_dim, n_tau=self.n_tau)


class InputNormalizedBlockMLP(nn.Module):
    """BlockMLP with fixed input normalization and raw-output contract."""

    def __init__(
        self,
        *,
        orbital_dim: int = 3,
        n_tau: int = 59,
        scalar_dim: int = 4,
        hidden_dim: int = 512,
        num_layers: int = 4,
        x_mean: torch.Tensor,
        x_scale: torch.Tensor,
    ):
        super().__init__()
        self.base = BlockMLP(
            orbital_dim=orbital_dim,
            n_tau=n_tau,
            scalar_dim=scalar_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        if x_mean.shape != (self.base.input_dim,) or x_scale.shape != (self.base.input_dim,):
            raise ValueError(
                f"x_mean/x_scale must have shape ({self.base.input_dim},), "
                f"got {tuple(x_mean.shape)} and {tuple(x_scale.shape)}"
            )
        self.register_buffer("x_mean", x_mean.detach().clone().float())
        self.register_buffer("x_scale", x_scale.detach().clone().float())

    @property
    def orbital_dim(self) -> int:
        return self.base.orbital_dim

    @property
    def n_tau(self) -> int:
        return self.base.n_tau

    @property
    def scalar_dim(self) -> int:
        return self.base.scalar_dim

    @property
    def input_dim(self) -> int:
        return self.base.input_dim

    @property
    def output_dim(self) -> int:
        return self.base.output_dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = (input - self.x_mean.to(input.dtype)) / self.x_scale.clamp_min(1e-12).to(input.dtype)
        return self.base(x)


class ScalarConditionedEquivariantBlock(nn.Module):
    def __init__(self, irreps, *, scalar_dim: int, condition_dim: int):
        super().__init__()
        from e3nn import o3
        from e3nn.nn import Gate

        self.condition = nn.Sequential(
            nn.Linear(scalar_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )
        condition_irreps = o3.Irreps(f"{condition_dim}x0e")
        self.tensor_product = o3.FullyConnectedTensorProduct(irreps, condition_irreps, irreps)

        hidden_channels = irreps.count("0e")
        self.pre_gate = o3.Linear(irreps, o3.Irreps(f"{hidden_channels}x0e+{hidden_channels}x0e+{hidden_channels}x2e"))
        self.gate = Gate(
            o3.Irreps(f"{hidden_channels}x0e"),
            [F.silu],
            o3.Irreps(f"{hidden_channels}x0e"),
            [torch.sigmoid],
            o3.Irreps(f"{hidden_channels}x2e"),
        )

    def forward(self, x: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        conditioned = self.tensor_product(x, self.condition(scalars))
        return x + self.gate(self.pre_gate(conditioned))


class OrbitalIrrepNet(nn.Module):
    """e3nn equivariant t2g block solver with symmetric-real ``0e + 2e`` nodes.

    Input and output match :class:`BlockResNet`: flattened complex Delta block
    features followed by scalar channels, returning flattened complex G block
    features. The optional Hermitian antisymmetric ``1e`` channel is omitted for
    the block-diagonal SrVO3 path, so outputs are real symmetric Hermitian.
    """

    def __init__(
        self,
        *,
        orbital_dim: int = 3,
        n_tau: int = 59,
        scalar_dim: int = 4,
        hidden_channels: int = 8,
        num_layers: int = 3,
        condition_dim: int | None = None,
    ):
        super().__init__()
        if int(orbital_dim) != 3:
            raise ValueError("OrbitalIrrepNet currently implements the t2g orbital_dim=3 case")
        from e3nn import o3

        self.orbital_dim = int(orbital_dim)
        self.n_tau = int(n_tau)
        self.scalar_dim = int(scalar_dim)
        self.hidden_channels = int(hidden_channels)
        self.num_layers = int(num_layers)
        self.block_dim = self.orbital_dim * self.orbital_dim * self.n_tau * 2
        self.input_dim = self.block_dim + self.scalar_dim
        self.output_dim = self.block_dim

        rtp = o3.ReducedTensorProducts("ij=ji", i=o3.Irreps("1e"))
        self.register_buffer("symmetric_basis", rtp.change_of_basis.detach().clone())
        self.register_buffer("scalar_scale", self._default_scalar_scale(self.scalar_dim))

        self.input_irreps = o3.Irreps(f"{self.n_tau}x0e+{self.n_tau}x2e")
        self.hidden_irreps = o3.Irreps(f"{self.hidden_channels}x0e+{self.hidden_channels}x2e")
        self.input_linear = o3.Linear(self.input_irreps, self.hidden_irreps)
        condition_dim = int(condition_dim if condition_dim is not None else max(4, self.hidden_channels))
        self.layers = nn.ModuleList(
            [
                ScalarConditionedEquivariantBlock(
                    self.hidden_irreps,
                    scalar_dim=self.scalar_dim,
                    condition_dim=condition_dim,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.output_linear = o3.Linear(self.hidden_irreps, self.input_irreps)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim != 2 or input.shape[1] != self.input_dim:
            raise ValueError(f"input must have shape (batch, {self.input_dim}), got {tuple(input.shape)}")
        scalars = input[:, -self.scalar_dim :] / self.scalar_scale.to(input.dtype)
        blocks = block_features_to_matrix(
            input[:, : self.block_dim],
            orbital_dim=self.orbital_dim,
            n_tau=self.n_tau,
        )
        x = self._matrix_to_irreps(blocks.real)
        x = self.input_linear(x)
        for layer in self.layers:
            x = layer(x, scalars)
        return matrix_to_block_features(self._irreps_to_matrix(self.output_linear(x)).to(torch.complex64))

    def _matrix_to_irreps(self, blocks: torch.Tensor) -> torch.Tensor:
        coeffs = torch.einsum("aij,bijt->bta", self.symmetric_basis.to(blocks.dtype), blocks)
        scalars = coeffs[..., 0]
        quadrupoles = coeffs[..., 1:].reshape(blocks.shape[0], self.n_tau * 5)
        return torch.cat((scalars, quadrupoles), dim=1)

    def _irreps_to_matrix(self, irreps: torch.Tensor) -> torch.Tensor:
        scalars = irreps[:, : self.n_tau].reshape(irreps.shape[0], self.n_tau, 1)
        quadrupoles = irreps[:, self.n_tau :].reshape(irreps.shape[0], self.n_tau, 5)
        coeffs = torch.cat((scalars, quadrupoles), dim=-1)
        return torch.einsum("bta,aij->bijt", coeffs, self.symmetric_basis.to(irreps.dtype))

    @staticmethod
    def _default_scalar_scale(scalar_dim: int) -> torch.Tensor:
        base = torch.ones(int(scalar_dim), dtype=torch.float32)
        if scalar_dim >= 1:
            base[0] = 10.0
        if scalar_dim >= 3:
            base[2] = 100.0
        if scalar_dim >= 4:
            base[3] = 2.0
        return base


def positive_matsubara_causality_penalty(values_iw: torch.Tensor, iw: torch.Tensor) -> torch.Tensor:
    """Penalty for positive Im eigenvalues on positive Matsubara nodes.

    This mirrors ``positive_freq_causality_rate``/``is_causal``: scalar values or
    matrix eigenvalues should have nonpositive imaginary parts for ``Im iw > 0``.
    """

    mask = iw.imag > 0
    values = values_iw[..., mask, :, :] if values_iw.ndim >= 3 and values_iw.shape[-1] == values_iw.shape[-2] else values_iw[..., mask]
    if values.ndim >= 3 and values.shape[-1] == values.shape[-2]:
        imag_parts = torch.linalg.eigvals(values).imag
    else:
        imag_parts = values.imag
    return F.relu(imag_parts).square().mean()
