from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from sigml.solver.dataset import SolverDataset
from sigml.solver.net import (
    BlockResNet,
    FeedforwardNet,
    block_features_to_matrix,
    matrix_to_block_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SigML NN impurity solver.")
    parser.add_argument("--dataset", required=True, help="Path to solver NPZ dataset.")
    parser.add_argument("--output", required=True, help="Path for output checkpoint.")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--architecture",
        choices=("auto", "feedforward", "block-resnet"),
        default="auto",
        help="Model architecture. auto keeps the legacy orb1 net for M=1 and uses block-resnet for M>1.",
    )
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--augment", action="store_true", help="Apply orbital Q Delta Q^T augmentation.")
    parser.add_argument(
        "--augment-mode",
        choices=("rotation", "permutation", "mixed"),
        default="mixed",
        help="Orbital transform family used when --augment is set.",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    dataset = SolverDataset(args.dataset)
    n_val = int(round(len(dataset) * args.val_fraction))
    n_val = min(max(n_val, 0), max(len(dataset) - 1, 0))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(args.seed)
    if n_val:
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    else:
        train_ds, val_ds = dataset, None

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False) if val_ds is not None else None
    )
    device = torch.device(args.device)
    architecture = _resolve_architecture(args.architecture, dataset)
    model = _build_model(args, dataset=dataset, architecture=architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    loss_history = []

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        n_seen = 0
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            if args.augment:
                if dataset.orbital_dim == 1:
                    raise ValueError("--augment requires a block dataset with orbital_dim > 1")
                x, y = augment_block_batch(
                    x,
                    y,
                    orbital_dim=dataset.orbital_dim,
                    n_tau=dataset.delta_tau.shape[-1],
                    scalar_dim=len(dataset.scalar_names),
                    mode=args.augment_mode,
                )
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu()) * x.shape[0]
            n_seen += x.shape[0]

        item = {"epoch": epoch + 1, "train_loss": train_loss / max(n_seen, 1)}
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            n_val_seen = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["x"].to(device)
                    y = batch["y"].to(device)
                    loss = criterion(model(x), y)
                    val_loss += float(loss.detach().cpu()) * x.shape[0]
                    n_val_seen += x.shape[0]
            item["val_loss"] = val_loss / max(n_val_seen, 1)
        loss_history.append(item)

    return {
        "architecture": architecture,
        "model_state_dict": model.state_dict(),
        "input_dim": int(dataset.x.shape[1]),
        "output_dim": int(dataset.y.shape[1]),
        "orbital_dim": int(dataset.orbital_dim),
        "n_tau": int(dataset.delta_tau.shape[-1]),
        "scalar_dim": int(len(dataset.scalar_names)),
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "config": vars(args).copy(),
        "loss_history": loss_history,
    }


def _resolve_architecture(requested: str, dataset: SolverDataset) -> str:
    if requested != "auto":
        return requested
    return "feedforward" if dataset.orbital_dim == 1 else "block-resnet"


def _build_model(args: argparse.Namespace, *, dataset: SolverDataset, architecture: str) -> torch.nn.Module:
    if architecture == "feedforward":
        return FeedforwardNet()
    if architecture == "block-resnet":
        return BlockResNet(
            orbital_dim=dataset.orbital_dim,
            n_tau=dataset.delta_tau.shape[-1],
            scalar_dim=len(dataset.scalar_names),
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
        )
    raise ValueError(f"unsupported architecture {architecture!r}")


def random_orbital_transform(
    orbital_dim: int,
    *,
    mode: str = "mixed",
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample an orthogonal orbital transform for Q block Q^T augmentation."""
    if mode == "mixed":
        mode = "permutation" if torch.rand((), generator=generator).item() < 0.5 else "rotation"
    if mode == "permutation":
        perm = torch.randperm(orbital_dim, generator=generator, device=device)
        return torch.eye(orbital_dim, device=device, dtype=dtype)[perm]
    if mode == "rotation":
        sample = torch.randn(
            orbital_dim,
            orbital_dim,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        q, r = torch.linalg.qr(sample)
        signs = torch.sign(torch.diagonal(r))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        q = q * signs
        if torch.linalg.det(q) < 0:
            q[:, 0] = -q[:, 0]
        return q
    raise ValueError(f"unsupported augmentation mode {mode!r}")


def augment_block_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    orbital_dim: int,
    n_tau: int,
    scalar_dim: int,
    mode: str = "mixed",
    q: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply (Delta, G) -> (Q Delta Q^T, Q G Q^T) to a batch."""
    if q is None:
        q = random_orbital_transform(
            orbital_dim,
            mode=mode,
            device=x.device,
            dtype=x.dtype,
        )
    else:
        q = q.to(device=x.device, dtype=x.dtype)
    delta = block_features_to_matrix(x[:, :-scalar_dim], orbital_dim=orbital_dim, n_tau=n_tau)
    target = block_features_to_matrix(y, orbital_dim=orbital_dim, n_tau=n_tau)
    q_complex = q.to(dtype=delta.real.dtype).to(delta.dtype)
    aug_delta = torch.einsum("ab,nbct,dc->nadt", q_complex, delta, q_complex)
    aug_target = torch.einsum("ab,nbct,dc->nadt", q_complex, target, q_complex)
    aug_x = torch.cat((matrix_to_block_features(aug_delta), x[:, -scalar_dim:]), dim=1)
    return aug_x, matrix_to_block_features(aug_target)


def main() -> None:
    args = parse_args()
    checkpoint = train(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)


if __name__ == "__main__":
    main()
