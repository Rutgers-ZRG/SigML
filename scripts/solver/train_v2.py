#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from sigml.solver.dataset import SolverDataset
from sigml.solver.net import FeedforwardNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small on-distribution CTSEG net and export NumPy weights.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights-npz", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    dataset = SolverDataset(args.dataset)
    n_val = max(1, int(round(len(dataset) * args.val_fraction))) if len(dataset) > 1 else 0
    n_val = min(n_val, max(len(dataset) - 1, 0))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = (
        random_split(dataset, [n_train, n_val], generator=generator) if n_val else (dataset, None)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False) if val_ds else None
    device = torch.device(args.device)
    model = FeedforwardNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    history = []
    best_state = None
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * x.shape[0]
            seen += x.shape[0]
        row = {"epoch": epoch, "train_loss": total / max(seen, 1)}
        if val_loader:
            model.eval()
            val_total = 0.0
            val_seen = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["x"].to(device)
                    y = batch["y"].to(device)
                    loss = criterion(model(x), y)
                    val_total += float(loss.detach().cpu()) * x.shape[0]
                    val_seen += x.shape[0]
            row["val_loss"] = val_total / max(val_seen, 1)
            if row["val_loss"] < best_val:
                best_val = row["val_loss"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(row)

    state = best_state if best_state is not None else {k: v.detach().cpu() for k, v in model.state_dict().items()}
    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": state, "config": vars(args), "loss_history": history}, ckpt_path)

    weights_path = Path(args.weights_npz)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(weights_path, **{k: v.numpy() for k, v in state.items()})

    summary = {
        "dataset": args.dataset,
        "n_total": len(dataset),
        "n_train": n_train,
        "n_val": n_val,
        "best_val_loss": best_val if np.isfinite(best_val) else None,
        "final": history[-1],
        "history_tail": history[-10:],
        "checkpoint": str(ckpt_path),
        "weights_npz": str(weights_path),
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
