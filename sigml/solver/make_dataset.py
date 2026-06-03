from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from sigml.solver.dataset import SolverDataset
from sigml.solver.dyson import sigma_from_g
from sigml.solver.hybridization import sample_delta_vec
from sigml.solver.labeler import OracleLabeler
from sigml.solver.metrics import g_mse, positive_freq_causality_rate
from sigml.solver.net import FeedforwardNet
from sigml.solver.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and train the Phase-B solver PoC.")
    parser.add_argument("--dataset", default="data/solver_poc_distill.npz")
    parser.add_argument("--checkpoint", default="SAVED_MODELS/solver_poc_distill.pth")
    parser.add_argument("--doc", default="docs/solver_poc_phaseB.md")
    parser.add_argument("--n-base", type=int, default=1000)
    parser.add_argument("--target-total", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--beta", type=float, default=70.0)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--mix", type=float, default=0.7)
    parser.add_argument("--max-augment-iter", type=int, default=2)
    parser.add_argument("--jitter-sigma", type=float, default=0.15)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _is_causal_g(grid, vec: np.ndarray, tol: float = 1e-6) -> bool:
    giw = grid.gtau_to_giw(grid.vec_to_gtau(vec))
    return bool(np.all(np.isfinite(giw)) and np.all(giw[grid.iw_nodes.imag > 0].imag <= tol))


def _label_is_valid(grid, g_vec: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(g_vec)) and _is_causal_g(grid, g_vec))


def _append_sample(store: dict[str, list], delta_vec, g_vec, U, mu, beta, eps_d, source: str) -> None:
    store["delta"].append(np.asarray(delta_vec, dtype=np.float32))
    store["g"].append(np.asarray(g_vec, dtype=np.float32))
    store["U"].append(float(U))
    store["mu"].append(float(mu))
    store["beta"].append(float(beta))
    store["eps_d"].append(float(eps_d))
    store["source"].append(source)


def build_dataset(args: argparse.Namespace) -> dict[str, np.ndarray | dict]:
    rng = np.random.default_rng(args.seed)
    labeler = OracleLabeler()
    grid = labeler.grid
    store: dict[str, list] = {k: [] for k in ["delta", "g", "U", "mu", "beta", "eps_d", "source"]}
    stats = {
        "base_attempts": 0,
        "base_kept": 0,
        "base_skipped": 0,
        "augment_attempts": 0,
        "augment_kept": 0,
        "augment_skipped": 0,
    }

    base_records: list[tuple[np.ndarray, np.ndarray, float]] = []
    max_base_attempts = max(args.n_base * 20, args.n_base)
    while stats["base_kept"] < args.n_base and stats["base_attempts"] < max_base_attempts:
        stats["base_attempts"] += 1
        U = float(rng.uniform(0.5, 7.0))
        mu = 0.5 * U
        alpha = float(rng.uniform(0.0, 0.3))
        delta_vec = sample_delta_vec(grid, rng, alpha=alpha)
        if not _is_causal_g(grid, delta_vec, tol=1e-9):
            stats["base_skipped"] += 1
            continue
        g_vec = labeler.solve(delta_vec, U=U, mu=mu, beta=args.beta, eps_d=0.0)
        if not _label_is_valid(grid, g_vec):
            stats["base_skipped"] += 1
            continue
        _append_sample(store, delta_vec, g_vec, U, mu, args.beta, 0.0, "base")
        base_records.append((delta_vec, g_vec, U))
        stats["base_kept"] += 1

    if stats["base_kept"] < args.n_base:
        raise RuntimeError(f"Only kept {stats['base_kept']} base samples after {stats['base_attempts']} attempts")

    while len(store["delta"]) < args.target_total and base_records:
        delta_vec, g_vec, U0 = base_records[int(rng.integers(0, len(base_records)))]
        current_delta = np.asarray(delta_vec, dtype=np.float64)
        current_g = np.asarray(g_vec, dtype=np.float64)
        n_iter = int(rng.integers(1, args.max_augment_iter + 1))
        for _ in range(n_iter):
            if len(store["delta"]) >= args.target_total:
                break
            stats["augment_attempts"] += 1
            U = float(np.clip(U0 * np.exp(rng.normal(0.0, args.jitter_sigma)), 0.5, 7.0))
            mu = 0.5 * U
            delta_new = (args.t**2) * current_g
            mixed_delta = args.mix * current_delta + (1.0 - args.mix) * delta_new
            if not np.all(np.isfinite(mixed_delta)) or not _is_causal_g(grid, mixed_delta):
                stats["augment_skipped"] += 1
                break
            next_g = labeler.solve(mixed_delta, U=U, mu=mu, beta=args.beta, eps_d=0.0)
            if not _label_is_valid(grid, next_g):
                stats["augment_skipped"] += 1
                break
            _append_sample(store, mixed_delta, next_g, U, mu, args.beta, 0.0, "augment")
            stats["augment_kept"] += 1
            current_delta = mixed_delta
            current_g = next_g

    arrays = {
        "delta": np.stack(store["delta"]).astype(np.float32),
        "g": np.stack(store["g"]).astype(np.float32),
        "U": np.asarray(store["U"], dtype=np.float32),
        "mu": np.asarray(store["mu"], dtype=np.float32),
        "beta": np.asarray(store["beta"], dtype=np.float32),
        "eps_d": np.asarray(store["eps_d"], dtype=np.float32),
        "source": np.asarray(store["source"]),
    }
    return {"arrays": arrays, "stats": stats}


def _predict(model: FeedforwardNet, x: torch.Tensor, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            y = model(x[start : start + batch_size].to(device)).detach().cpu().numpy()
            out.append(y)
    return np.concatenate(out, axis=0)


def evaluate(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    batch_size: int,
    device: str,
    val_fraction: float,
    seed: int,
) -> dict:
    ds = SolverDataset(dataset_path)
    n_val = int(round(len(ds) * val_fraction))
    n_val = min(max(n_val, 1), len(ds) - 1)
    n_train = len(ds) - n_val
    _, val_subset = torch.utils.data.random_split(
        ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    val_indices = np.asarray(val_subset.indices, dtype=int)
    x = torch.from_numpy(ds.x[val_indices]).float()
    y = ds.y[val_indices]
    device_obj = torch.device(device)
    model = FeedforwardNet().to(device_obj)
    ckpt = torch.load(checkpoint_path, map_location=device_obj)
    model.load_state_dict(ckpt["model_state_dict"])
    pred = _predict(model, x, device_obj, batch_size)

    grid = OracleLabeler().grid
    sigmas = []
    for row, pred_g in zip(ds.x[val_indices], pred):
        delta_vec = row[: grid.feature_dim]
        U, mu_over_u, beta = row[-3:]
        mu = float(mu_over_u * U)
        delta_iw = grid.gtau_to_giw(grid.vec_to_gtau(delta_vec))
        g_iw = grid.gtau_to_giw(grid.vec_to_gtau(pred_g))
        sigmas.append(sigma_from_g(g_iw, delta_iw, mu=mu, eps_d=0.0, iw=grid.iw_nodes))
    sigma = np.stack(sigmas)
    return {
        "n_validation": int(n_val),
        "g_mse": g_mse(pred, y),
        "causality_rate": positive_freq_causality_rate(sigma, grid.iw_nodes),
    }


def write_doc(args, dataset_stats, train_history, eval_stats) -> None:
    doc = Path(args.doc)
    doc.parent.mkdir(parents=True, exist_ok=True)
    last = train_history[-1]
    lines = [
        "# Solver PoC Phase B",
        "",
        "Local oracle-distillation plumbing run for Tasks 6-9.",
        "",
        "## Artifacts",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Checkpoint: `{args.checkpoint}`",
        "",
        "## Data",
        "",
        f"- Base kept: {dataset_stats['base_kept']} / {dataset_stats['base_attempts']} attempts",
        f"- Base skipped: {dataset_stats['base_skipped']}",
        f"- Augmentation kept: {dataset_stats['augment_kept']} / {dataset_stats['augment_attempts']} attempts",
        f"- Augmentation skipped: {dataset_stats['augment_skipped']}",
        f"- Total samples: {dataset_stats['base_kept'] + dataset_stats['augment_kept']}",
        "",
        "## Training",
        "",
        f"- Epochs: {args.epochs}",
        f"- Batch size: {args.batch_size}",
        f"- Learning rate: {args.lr}",
        f"- Device: {args.device}",
        f"- Final train loss: {last['train_loss']:.8g}",
        f"- Final validation loss: {last.get('val_loss', float('nan')):.8g}",
        "",
        "## Held-Out Validation",
        "",
        f"- Held-out samples: {eval_stats['n_validation']}",
        f"- Student-vs-oracle `g_mse`: {eval_stats['g_mse']:.8g}",
        f"- Predicted-Sigma causality rate: {eval_stats['causality_rate']:.6f}",
        "",
        "## Notes",
        "",
        "Labels are from the released Valenti `orb1` oracle, not CTSEG. This validates the local data and training pipeline only.",
        "Augmentation uses damped one-to-two-step Bethe updates and skips any non-finite or acausal sample.",
    ]
    doc.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.device is None:
        args.device = "mps" if torch.backends.mps.is_available() else "cpu"

    built = build_dataset(args)
    arrays = built["arrays"]
    dataset_path = Path(args.dataset)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dataset_path, **arrays)

    train_args = SimpleNamespace(
        dataset=args.dataset,
        output=args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device=args.device,
    )
    checkpoint = train(train_args)
    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, ckpt_path)

    eval_stats = evaluate(
        args.checkpoint,
        args.dataset,
        args.batch_size,
        args.device,
        args.val_fraction,
        args.seed,
    )
    write_doc(args, built["stats"], checkpoint["loss_history"], eval_stats)
    print(f"dataset={args.dataset}")
    print(f"checkpoint={args.checkpoint}")
    print(f"g_mse={eval_stats['g_mse']:.8g}")
    print(f"causality_rate={eval_stats['causality_rate']:.6f}")


if __name__ == "__main__":
    main()
