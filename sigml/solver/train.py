from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from sigml.solver.dataset import SolverDataset
from sigml.solver.net import FeedforwardNet


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
    model = FeedforwardNet().to(device)
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
        "model_state_dict": model.state_dict(),
        "config": vars(args).copy(),
        "loss_history": loss_history,
    }


def main() -> None:
    args = parse_args()
    checkpoint = train(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)


if __name__ == "__main__":
    main()
