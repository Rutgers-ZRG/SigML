#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sigml.solver.dataset import SolverDataset
from sigml.solver.metrics import g_mse, positive_freq_causality_rate
from sigml.solver.net import BlockResNet, OrbitalIrrepNet, block_features_to_matrix
from sigml.solver.pydlr_grid import PydlrGrid
from sigml.solver.train import augment_block_batch


@dataclass(frozen=True)
class ArchitectureConfig:
    name: str
    hidden_dim: int
    num_layers: int
    augment: bool
    augment_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare Stage-2 t2g nets on real SrVO3-warm CTHYB labels."
    )
    parser.add_argument("--dataset", default="data/bootstrap_t2g_b40.npz")
    parser.add_argument("--out-dir", default="SAVED_MODELS/stage2_real_b40")
    parser.add_argument("--results-json", default=None)
    parser.add_argument("--folds", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--block-hidden-dim", type=int, default=128)
    parser.add_argument("--block-layers", type=int, default=3)
    parser.add_argument("--e3nn-hidden-dim", type=int, default=8)
    parser.add_argument("--e3nn-layers", type=int, default=2)
    parser.add_argument("--grid-beta", type=float, default=40.0)
    parser.add_argument("--grid-lamb", type=float, default=600.0)
    parser.add_argument("--grid-eps", type=float, default=1e-10)
    parser.add_argument(
        "--train-fractions",
        type=float,
        nargs="+",
        default=[0.5, 1.0],
        help="Fractions of each fold's training rows used for data-efficiency comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ds = SolverDataset(args.dataset)
    if ds.orbital_dim != 3:
        raise ValueError("Phase-12 real-material comparison expects an M=3 t2g dataset")

    grid = PydlrGrid(beta=args.grid_beta, lamb=args.grid_lamb, eps=args.grid_eps)
    if grid.rank != ds.g.shape[-1]:
        raise ValueError(f"grid rank {grid.rank} does not match dataset rank {ds.g.shape[-1]}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    architectures = [
        ArchitectureConfig(
            name="block-resnet-aug",
            hidden_dim=args.block_hidden_dim,
            num_layers=args.block_layers,
            augment=True,
            augment_mode="mixed",
        ),
        ArchitectureConfig(
            name="e3nn-irrep",
            hidden_dim=args.e3nn_hidden_dim,
            num_layers=args.e3nn_layers,
            augment=False,
            augment_mode="none",
        ),
    ]

    fold_pairs = _kfold_indices(len(ds), args.folds, args.seed)
    rows = []
    for arch in architectures:
        for fraction in args.train_fractions:
            for fold_id, (train_idx, val_idx) in enumerate(fold_pairs):
                use_idx = _fractional_train_indices(train_idx, fraction, args.seed + fold_id)
                model = _build_model(ds, arch).to(args.device)
                history = _fit(
                    model,
                    ds,
                    use_idx,
                    arch=arch,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    device=args.device,
                    seed=args.seed + 1000 * fold_id + int(100 * fraction),
                )
                metrics = _evaluate(model, ds, val_idx, grid=grid, device=args.device)
                ckpt_path = out_dir / f"{arch.name}_frac{fraction:g}_fold{fold_id}.pt"
                torch.save(
                    {
                        "architecture": arch.name,
                        "model_state_dict": model.state_dict(),
                        "input_dim": int(ds.x.shape[1]),
                        "output_dim": int(ds.y.shape[1]),
                        "orbital_dim": int(ds.orbital_dim),
                        "n_tau": int(ds.g.shape[-1]),
                        "scalar_dim": int(len(ds.scalar_names)),
                        "hidden_dim": int(arch.hidden_dim),
                        "num_layers": int(arch.num_layers),
                        "config": vars(args).copy() | {
                            "augment": arch.augment,
                            "augment_mode": arch.augment_mode,
                            "train_fraction": float(fraction),
                            "fold": int(fold_id),
                            "train_indices": use_idx.tolist(),
                            "val_indices": val_idx.tolist(),
                            "scalar_names": list(ds.scalar_names),
                        },
                        "loss_history": history,
                        "metrics": metrics,
                    },
                    ckpt_path,
                )
                row = {
                    "architecture": arch.name,
                    "train_fraction": float(fraction),
                    "fold": int(fold_id),
                    "n_train": int(len(use_idx)),
                    "n_val": int(len(val_idx)),
                    "checkpoint": str(ckpt_path),
                    **metrics,
                }
                rows.append(row)
                print(
                    f"{arch.name} frac={fraction:g} fold={fold_id} "
                    f"gtau_mse={metrics['gtau_mse']:.6g} "
                    f"causality={metrics['causality_rate']:.3f}"
                )

        final_model = _build_model(ds, arch).to(args.device)
        final_history = _fit(
            final_model,
            ds,
            np.arange(len(ds)),
            arch=arch,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=args.device,
            seed=args.seed + 999,
        )
        final_path = out_dir / f"{arch.name}_all_rows.pt"
        torch.save(
            {
                "architecture": arch.name,
                "model_state_dict": final_model.state_dict(),
                "input_dim": int(ds.x.shape[1]),
                "output_dim": int(ds.y.shape[1]),
                "orbital_dim": int(ds.orbital_dim),
                "n_tau": int(ds.g.shape[-1]),
                "scalar_dim": int(len(ds.scalar_names)),
                "hidden_dim": int(arch.hidden_dim),
                "num_layers": int(arch.num_layers),
                "config": vars(args).copy() | {
                    "augment": arch.augment,
                    "augment_mode": arch.augment_mode,
                    "train_indices": list(range(len(ds))),
                    "scalar_names": list(ds.scalar_names),
                },
                "loss_history": final_history,
            },
            final_path,
        )

    summary = _summarize(rows, final_paths={arch.name: str(out_dir / f"{arch.name}_all_rows.pt") for arch in architectures})
    payload = {
        "dataset": str(args.dataset),
        "n_rows": int(len(ds)),
        "folds": int(args.folds),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "grid": {"beta": grid.beta, "lamb": grid.lamb, "eps": grid.eps, "rank": grid.rank},
        "scalar_names": list(ds.scalar_names),
        "rows": rows,
        "summary": summary,
    }
    results_json = Path(args.results_json) if args.results_json else out_dir / "results.json"
    results_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {results_json}")


def _build_model(ds: SolverDataset, arch: ArchitectureConfig) -> torch.nn.Module:
    kwargs = {
        "orbital_dim": ds.orbital_dim,
        "n_tau": ds.g.shape[-1],
        "scalar_dim": len(ds.scalar_names),
        "num_layers": arch.num_layers,
    }
    if arch.name == "block-resnet-aug":
        return BlockResNet(hidden_dim=arch.hidden_dim, **kwargs)
    if arch.name == "e3nn-irrep":
        return OrbitalIrrepNet(hidden_channels=arch.hidden_dim, **kwargs)
    raise ValueError(f"unsupported architecture {arch.name!r}")


def _fit(
    model: torch.nn.Module,
    ds: SolverDataset,
    train_idx: np.ndarray,
    *,
    arch: ArchitectureConfig,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    seed: int,
) -> list[dict[str, float]]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.from_numpy(ds.x[train_idx])
    y = torch.from_numpy(ds.y[train_idx])
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()
    history = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        n_seen = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if arch.augment:
                xb, yb = augment_block_batch(
                    xb,
                    yb,
                    orbital_dim=ds.orbital_dim,
                    n_tau=ds.g.shape[-1],
                    scalar_dim=len(ds.scalar_names),
                    mode=arch.augment_mode,
                )
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * xb.shape[0]
            n_seen += xb.shape[0]
        history.append({"epoch": float(epoch + 1), "train_loss": total / max(n_seen, 1)})
    return history


def _evaluate(
    model: torch.nn.Module,
    ds: SolverDataset,
    val_idx: np.ndarray,
    *,
    grid: PydlrGrid,
    device: str,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        pred_y = model(torch.from_numpy(ds.x[val_idx]).to(device)).cpu()
    pred_coeffs = block_features_to_matrix(
        pred_y,
        orbital_dim=ds.orbital_dim,
        n_tau=ds.g.shape[-1],
    ).numpy()
    target_coeffs = ds.g[val_idx]
    pred_tau = grid.gtau_from_coeffs(pred_coeffs)
    target_tau = grid.gtau_from_coeffs(target_coeffs)
    pred_iw = np.moveaxis(grid.giw_from_coeffs(pred_coeffs), -1, 1)
    return {
        "gtau_mse": g_mse(pred_tau, target_tau),
        "dlr_mse": g_mse(pred_coeffs, target_coeffs),
        "causality_rate": positive_freq_causality_rate(pred_iw, grid.iw_nodes, tol=1e-6),
    }


def _kfold_indices(n_rows: int, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if folds < 2 or folds > n_rows:
        raise ValueError(f"folds must be in [2, {n_rows}], got {folds}")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_rows)
    val_splits = np.array_split(indices, folds)
    pairs = []
    for split in val_splits:
        val = np.sort(split)
        train = np.sort(np.setdiff1d(indices, val, assume_unique=True))
        pairs.append((train, val))
    return pairs


def _fractional_train_indices(train_idx: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"train fraction must be in (0, 1], got {fraction}")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(train_idx)
    n_use = max(1, int(round(len(shuffled) * fraction)))
    return np.sort(shuffled[:n_use])


def _summarize(rows: list[dict], *, final_paths: dict[str, str]) -> dict:
    summary: dict[str, dict] = {}
    for arch in sorted({row["architecture"] for row in rows}):
        summary[arch] = {"final_all_rows_checkpoint": final_paths[arch], "fractions": {}}
        arch_rows = [row for row in rows if row["architecture"] == arch]
        for fraction in sorted({row["train_fraction"] for row in arch_rows}):
            selected = [row for row in arch_rows if row["train_fraction"] == fraction]
            for metric in ("gtau_mse", "dlr_mse", "causality_rate"):
                values = [float(row[metric]) for row in selected]
                summary[arch]["fractions"][str(fraction)] = summary[arch]["fractions"].get(str(fraction), {})
                summary[arch]["fractions"][str(fraction)][metric] = {
                    "mean": mean(values),
                    "std": pstdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                }
    for arch, item in summary.items():
        fractions = item["fractions"]
        if "0.5" in fractions and "1.0" in fractions:
            half = fractions["0.5"]["gtau_mse"]["mean"]
            full = fractions["1.0"]["gtau_mse"]["mean"]
            item["half_over_full_gtau_mse"] = None if full == 0.0 else half / full
    return summary


if __name__ == "__main__":
    main()
