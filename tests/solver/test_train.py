import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

from sigml.solver.dataset import SolverDataset
from sigml.solver.net import block_features_to_matrix
from sigml.solver.train import augment_block_batch, random_orbital_transform, train


def test_train_cli_smoke_writes_checkpoint():
    n = 8
    with tempfile.TemporaryDirectory() as d:
        dataset = os.path.join(d, "data.npz")
        output = os.path.join(d, "ckpt.pth")
        U = np.linspace(1.0, 5.0, n)
        np.savez(
            dataset,
            delta=np.random.randn(n, 118).astype(np.float32),
            g=np.random.randn(n, 118).astype(np.float32),
            U=U.astype(np.float32),
            mu=(0.5 * U).astype(np.float32),
            beta=np.full(n, 70.0, dtype=np.float32),
            eps_d=np.zeros(n, dtype=np.float32),
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "sigml.solver.train",
                "--dataset",
                dataset,
                "--output",
                output,
                "--epochs",
                "1",
                "--batch-size",
                "4",
            ],
            check=True,
            cwd=os.getcwd(),
        )
        ckpt = torch.load(output, map_location="cpu")
        assert {"model_state_dict", "config", "loss_history"}.issubset(ckpt)
        assert len(ckpt["loss_history"]) == 1


def _hermitian_blocks(n: int, m: int, n_tau: int, rng: np.random.Generator) -> np.ndarray:
    real = rng.normal(size=(n, m, m, n_tau))
    imag = rng.normal(size=(n, m, m, n_tau))
    blocks = real + 1j * imag
    return 0.5 * (blocks + np.swapaxes(blocks.conj(), 1, 2))


def _causal_blocks(n: int, m: int, n_tau: int, rng: np.random.Generator) -> np.ndarray:
    blocks = np.empty((n, m, m, n_tau), dtype=np.complex128)
    for sample in range(n):
        for tau in range(n_tau):
            a = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
            blocks[sample, :, :, tau] = -(a @ a.conj().T)
    return blocks


def _synthetic_block_dataset(
    path: str, *, n: int = 24, m: int = 3, n_tau: int = 4, causal: bool = False
) -> None:
    rng = np.random.default_rng(123)
    delta = _causal_blocks(n, m, n_tau, rng) if causal else _hermitian_blocks(n, m, n_tau, rng)
    eye = np.eye(m)[:, :, None]
    U = np.linspace(3.0, 5.0, n, dtype=np.float32)
    mu = (0.45 * U).astype(np.float32)
    beta = np.full(n, 40.0, dtype=np.float32)
    J = np.linspace(0.5, 0.7, n, dtype=np.float32)
    scalar_shift = (0.03 * U + 0.02 * J)[:, None, None, None] * eye[None, ...]
    g = 0.35 * delta - scalar_shift
    np.savez(path, delta=delta, g=g, U=U, mu=mu, beta=beta, J=J)


def test_random_orbital_transform_and_batch_augmentation_preserve_target_conjugation(tmp_path):
    path = tmp_path / "m3.npz"
    _synthetic_block_dataset(str(path), n=3, causal=True)
    ds = SolverDataset(path)
    batch = {
        "x": torch.stack([ds[i]["x"] for i in range(3)]),
        "y": torch.stack([ds[i]["y"] for i in range(3)]),
    }
    q = random_orbital_transform(3, mode="permutation", device=torch.device("cpu"))
    aug_x, aug_y = augment_block_batch(
        batch["x"],
        batch["y"],
        orbital_dim=3,
        n_tau=4,
        scalar_dim=4,
        mode="permutation",
        q=q,
    )

    delta = block_features_to_matrix(batch["x"][:, :-4], orbital_dim=3, n_tau=4)
    target = block_features_to_matrix(batch["y"], orbital_dim=3, n_tau=4)
    aug_delta = block_features_to_matrix(aug_x[:, :-4], orbital_dim=3, n_tau=4)
    aug_target = block_features_to_matrix(aug_y, orbital_dim=3, n_tau=4)
    q_complex = q.to(delta.dtype)
    expected_delta = torch.einsum("ab,nbct,dc->nadt", q_complex, delta, q_complex)
    expected_target = torch.einsum("ab,nbct,dc->nadt", q_complex, target, q_complex)

    assert torch.allclose(aug_x[:, -4:], batch["x"][:, -4:])
    assert torch.allclose(aug_delta, expected_delta, atol=1e-6)
    assert torch.allclose(aug_target, expected_target, atol=1e-6)
    assert torch.allclose(aug_delta, aug_delta.transpose(1, 2).conj(), atol=1e-6)
    assert torch.allclose(aug_target, aug_target.transpose(1, 2).conj(), atol=1e-6)
    for tau in range(aug_delta.shape[-1]):
        assert torch.linalg.eigvalsh(aug_delta[:, :, :, tau]).max() <= 1e-5
        assert torch.linalg.eigvalsh(aug_target[:, :, :, tau]).max() <= 1e-5


def test_block_training_smoke_loss_decreases_on_synthetic_data(tmp_path):
    dataset = tmp_path / "m3.npz"
    output = tmp_path / "ckpt.pth"
    _synthetic_block_dataset(str(dataset), n=32)
    args = type(
        "Args",
        (),
        {
            "dataset": str(dataset),
            "output": str(output),
            "epochs": 40,
            "batch_size": 16,
            "lr": 5e-3,
            "val_fraction": 0.0,
            "seed": 0,
            "device": "cpu",
            "architecture": "block-resnet",
            "hidden_dim": 64,
            "num_layers": 2,
            "augment": True,
            "augment_mode": "permutation",
        },
    )()
    ckpt = train(args)
    history = ckpt["loss_history"]
    assert ckpt["architecture"] == "block-resnet"
    assert history[-1]["train_loss"] < history[0]["train_loss"]
