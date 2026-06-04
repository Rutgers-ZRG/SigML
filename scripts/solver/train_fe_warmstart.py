#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from sigml.solver.dataset import SolverDataset
from sigml.solver.dyson import sigma_from_g
from sigml.solver.metrics import g_mse, orbital_occupation, positive_freq_causality_rate
from sigml.solver.net import InputNormalizedBlockMLP, block_features_to_matrix
from sigml.solver.pydlr_grid import PydlrGrid
from sigml.solver.train import augment_block_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate the Fe M=5 warm-start solver net.")
    parser.add_argument("--dataset", default="data/fe_tscan5000k_b232.npz")
    parser.add_argument("--checkpoint", default="checkpoints/fe_m5_block_mlp_b232.pt")
    parser.add_argument("--summary", default="results/fe_m5_warmstart_summary.json")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=900)
    parser.add_argument("--final-epochs", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--augment", choices=("none", "d-shell-permutation"), default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, checkpoint = run(args)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.checkpoint)
    print(json.dumps(summary, indent=2, sort_keys=True))


def run(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object]]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = SolverDataset(args.dataset)
    raw = np.load(args.dataset, allow_pickle=True)
    groups = (
        np.asarray(raw["source_iteration_group"]).astype(str)
        if "source_iteration_group" in raw
        else np.char.add(np.asarray(raw["source"]).astype(str), np.char.add(":", np.asarray(raw["iteration_suffix"]).astype(str)))
    )
    folds = grouped_fold_indices(groups, args.folds)
    device = torch.device(args.device)

    fold_summaries = []
    oof = np.zeros_like(dataset.y, dtype=np.float32)
    for fold_id, (train_idx, val_idx) in enumerate(folds):
        model = build_fe_model(args, dataset, train_idx).to(device)
        best_state, history = fit_model(
            model,
            dataset,
            train_idx,
            val_idx,
            args=args,
            device=device,
            epochs=args.epochs,
        )
        model.load_state_dict(best_state)
        pred = predict(model, dataset, val_idx, device=device)
        oof[val_idx] = pred
        metrics = evaluate_predictions(
            pred,
            dataset.y[val_idx],
            raw,
            val_idx,
        )
        metrics.update(
            {
                "fold": fold_id,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "heldout_iteration_suffixes": sorted(set(groups[val_idx]), key=suffix_sort_key),
                "best_val_loss": float(min(row["val_loss"] for row in history)),
                "final_train_loss": float(history[-1]["train_loss"]),
            }
        )
        fold_summaries.append(metrics)

    cv_metrics = evaluate_predictions(oof, dataset.y, raw, np.arange(len(dataset)))
    final_model = build_fe_model(args, dataset, np.arange(len(dataset))).to(device)
    final_state, final_history = fit_model(
        final_model,
        dataset,
        np.arange(len(dataset)),
        np.array([], dtype=int),
        args=args,
        device=device,
        epochs=args.final_epochs,
    )

    checkpoint = {
        "architecture": "input-normalized-block-mlp",
        "model_state_dict": final_state,
        "x_mean": final_model.x_mean.detach().cpu(),
        "x_scale": final_model.x_scale.detach().cpu(),
        "input_dim": int(dataset.x.shape[1]),
        "output_dim": int(dataset.y.shape[1]),
        "orbital_dim": int(dataset.orbital_dim),
        "n_tau": int(dataset.delta_tau.shape[-1]),
        "scalar_dim": int(len(dataset.scalar_names)),
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "config": vars(args).copy(),
        "loss_history": final_history,
    }
    summary = {
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "architecture": "input-normalized-block-mlp",
        "augmentation": args.augment,
        "folds": fold_summaries,
        "cross_validation": cv_metrics,
        "final_train_loss": float(final_history[-1]["train_loss"]),
        "n_labels": int(len(dataset)),
        "orbital_dim": int(dataset.orbital_dim),
        "n_tau": int(dataset.delta_tau.shape[-1]),
        "scalar_names": list(dataset.scalar_names),
    }
    return summary, checkpoint


def grouped_fold_indices(groups: np.ndarray, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.asarray(sorted(set(groups), key=suffix_sort_key))
    n_folds = min(max(int(n_folds), 2), len(unique))
    folds = np.array_split(unique, n_folds)
    out = []
    all_idx = np.arange(groups.shape[0])
    for heldout in folds:
        val_mask = np.isin(groups, heldout)
        out.append((all_idx[~val_mask], all_idx[val_mask]))
    return out


def build_fe_model(
    args: argparse.Namespace,
    dataset: SolverDataset,
    train_idx: np.ndarray,
) -> InputNormalizedBlockMLP:
    x_train = torch.from_numpy(dataset.x[np.asarray(train_idx, dtype=int)])
    x_mean = x_train.mean(dim=0)
    x_scale = x_train.std(dim=0)
    x_scale = torch.where(x_scale < 1e-6, torch.ones_like(x_scale), x_scale)
    return InputNormalizedBlockMLP(
        orbital_dim=dataset.orbital_dim,
        n_tau=dataset.delta_tau.shape[-1],
        scalar_dim=len(dataset.scalar_names),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        x_mean=x_mean,
        x_scale=x_scale,
    )


def fit_model(
    model: torch.nn.Module,
    dataset: SolverDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    args: argparse.Namespace,
    device: torch.device,
    epochs: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]]]:
    train_loader = DataLoader(
        Subset(dataset, [int(x) for x in train_idx]),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = (
        DataLoader(Subset(dataset, [int(x) for x in val_idx]), batch_size=args.batch_size)
        if len(val_idx)
        else None
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()
    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_total = 0.0
        train_seen = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            if args.augment == "d-shell-permutation":
                x, y = augment_block_batch(
                    x,
                    y,
                    orbital_dim=dataset.orbital_dim,
                    n_tau=dataset.delta_tau.shape[-1],
                    scalar_dim=len(dataset.scalar_names),
                    mode="d-shell-permutation",
                )
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            train_total += float(loss.detach().cpu()) * x.shape[0]
            train_seen += x.shape[0]

        row = {"epoch": float(epoch), "train_loss": train_total / max(train_seen, 1)}
        if val_loader is not None:
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
                best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        history.append(row)

    if val_loader is None:
        best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    return best_state, history


def predict(
    model: torch.nn.Module,
    dataset: SolverDataset,
    indices: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(Subset(dataset, [int(x) for x in indices]), batch_size=64)
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            rows.append(model(batch["x"].to(device)).detach().cpu().numpy())
    return np.concatenate(rows, axis=0).astype(np.float32)


def evaluate_predictions(
    pred_y: np.ndarray,
    target_y: np.ndarray,
    raw: np.lib.npyio.NpzFile,
    indices: np.ndarray,
) -> dict[str, object]:
    m = int(raw["g"].shape[1])
    n_tau = int(raw["g"].shape[-1])
    beta = float(np.asarray(raw["beta"])[0])
    lamb = float(np.asarray(raw["dlr_lamb"]))
    eps = float(np.asarray(raw["dlr_eps"]))
    grid = PydlrGrid(beta=beta, lamb=lamb, eps=eps)
    idx = np.asarray(indices, dtype=int)

    pred_tau = _features_to_numpy_blocks(pred_y, orbital_dim=m, n_tau=n_tau)
    target_tau = _features_to_numpy_blocks(target_y, orbital_dim=m, n_tau=n_tau)
    diag = np.arange(m)
    pred_diag = pred_tau[:, diag, diag, :]
    target_diag = target_tau[:, diag, diag, :]
    pred_coeffs = grid.coeffs_from_gtau(pred_tau)
    target_coeffs = np.asarray(raw["g_dlr_coeffs"])[idx]
    pred_iw = grid.giw_from_coeffs(pred_coeffs)
    target_iw = grid.giw_from_coeffs(target_coeffs)
    iw = grid.iw_nodes

    occ_pred = orbital_occupation(pred_tau)[:, diag, diag]
    occ_ref = orbital_occupation(target_tau)[:, diag, diag]
    sigma_metrics = _sigma_metrics(pred_iw, target_iw, raw, idx, grid)

    return {
        "g_tau_mse_diag": g_mse(pred_diag, target_diag),
        "g_tau_mse_all_block": g_mse(pred_tau, target_tau),
        "g_iw_mse_diag": g_mse(pred_iw[:, diag, diag, :], target_iw[:, diag, diag, :]),
        "g_causality_rate": positive_freq_causality_rate(np.moveaxis(pred_iw, -1, -3), iw),
        "g_max_positive_imag_eig": _max_positive_imag_eig(np.moveaxis(pred_iw, -1, -3), iw),
        "occupation_mae": float(np.mean(np.abs(occ_pred - occ_ref))),
        "occupation_ref_mean": float(np.mean(occ_ref)),
        "occupation_pred_mean": float(np.mean(occ_pred)),
        **sigma_metrics,
    }


def _features_to_numpy_blocks(features: np.ndarray, *, orbital_dim: int, n_tau: int) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
    return block_features_to_matrix(tensor, orbital_dim=orbital_dim, n_tau=n_tau).numpy()


def _sigma_metrics(
    pred_g_iw: np.ndarray,
    target_g_iw: np.ndarray,
    raw: np.lib.npyio.NpzFile,
    indices: np.ndarray,
    grid: PydlrGrid,
) -> dict[str, object]:
    idx = np.asarray(indices, dtype=int)
    delta_iw = grid.giw_from_coeffs(np.asarray(raw["delta_dlr_coeffs"])[idx])
    iw = np.broadcast_to(grid.iw_nodes[None, :], (idx.shape[0], grid.rank))
    eps_d = np.asarray(raw["eps_d"])[idx]
    eps_matrix = np.zeros((idx.shape[0], 1, 5, 5), dtype=complex)
    diag = np.arange(5)
    eps_matrix[:, 0, diag, diag] = eps_d
    mu = np.asarray(raw["mu"])[idx]

    pred_sigma = sigma_from_g(
        np.moveaxis(pred_g_iw, -1, -3),
        np.moveaxis(delta_iw, -1, -3),
        mu[:, None],
        eps_matrix,
        iw,
    )
    target_sigma = sigma_from_g(
        np.moveaxis(target_g_iw, -1, -3),
        np.moveaxis(delta_iw, -1, -3),
        mu[:, None],
        eps_matrix,
        iw,
    )
    sigout = np.moveaxis(grid.giw_from_coeffs(np.asarray(raw["sigma_dlr_coeffs"])[idx]), -1, -3)
    siginp_dynamic = np.moveaxis(
        grid.giw_from_coeffs(np.asarray(raw["siginp_dynamic_dlr_coeffs"])[idx]),
        -1,
        -3,
    )
    s_oo = np.asarray(raw["siginp_s_oo"])[idx]
    siginp_abs = siginp_dynamic.copy()
    for sample in range(idx.shape[0]):
        for orbital in diag:
            siginp_abs[sample, :, orbital, orbital] += s_oo[sample, orbital]

    return {
        "sigma_implied_vs_reference_mse_diag": _diag_freq_mse(pred_sigma, target_sigma),
        "sigma_implied_vs_sigout_mse_diag": _diag_freq_mse(pred_sigma, sigout),
        "sigma_implied_vs_siginp_plus_soo_mse_diag": _diag_freq_mse(pred_sigma, siginp_abs),
        "sigma_causality_rate": positive_freq_causality_rate(pred_sigma, grid.iw_nodes),
        "sigma_max_positive_imag_eig": _max_positive_imag_eig(pred_sigma, grid.iw_nodes),
    }


def _diag_freq_mse(left: np.ndarray, right: np.ndarray) -> float:
    diag = np.arange(left.shape[-1])
    return g_mse(left[..., diag, diag], right[..., diag, diag])


def _max_positive_imag_eig(values: np.ndarray, iw: np.ndarray) -> float:
    mask = np.asarray(iw).imag > 0
    selected = np.asarray(values)[..., mask, :, :]
    eig = np.linalg.eigvals(selected).imag
    return float(np.max(eig))


def suffix_sort_key(value: str) -> tuple[int, tuple[int, ...] | tuple[str, ...]]:
    if value == "current":
        return (1, ())
    body = value[1:] if value.startswith(".") else value
    parts = body.split(".")
    if all(part.isdigit() for part in parts):
        return (0, tuple(int(part) for part in parts))
    return (0, tuple(parts))


if __name__ == "__main__":
    main()
