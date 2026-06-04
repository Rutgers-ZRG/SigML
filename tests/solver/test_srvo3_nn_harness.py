from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from sigml.solver.nn_solver_client import run_nn_solver
from sigml.solver.pydlr_grid import PydlrGrid
from sigml.solver.srvo3_nn_harness import (
    Srvo3NNSidecarSettings,
    _mu_minus_eps_over_u,
    make_nn_solver_config,
    regularize_sigma_iw,
    stabilize_g_dlr,
    write_stub_block_resnet_checkpoint,
)


def test_stub_block_resnet_checkpoint_runs_through_sidecar(tmp_path: Path):
    grid = PydlrGrid(beta=40.0, lamb=600.0, eps=1e-8)
    ckpt = write_stub_block_resnet_checkpoint(
        tmp_path / "stub.pt",
        orbital_dim=3,
        n_tau=grid.n_tau,
        hidden_dim=8,
        num_layers=1,
        seed=7,
    )

    checkpoint = torch.load(ckpt, map_location="cpu")
    assert checkpoint["architecture"] == "block-resnet"
    assert checkpoint["stub_untrained"] is True
    assert checkpoint["n_tau"] == grid.n_tau

    delta = np.zeros((3, 3, grid.n_tau), dtype=np.complex64)
    for orb in range(3):
        delta[orb, orb] = -0.05 * np.exp(-grid.tau_nodes / grid.beta)

    result = run_nn_solver(
        make_nn_solver_config(
            Srvo3NNSidecarSettings(
                checkpoint_path=ckpt,
                repo_root=Path.cwd(),
                cli_path=Path("sigml/solver/nn_solver_cli.py"),
                python_command=(sys.executable,),
                timeout_seconds=60.0,
            )
        ),
        delta_dlr=delta,
        U=2.0,
        mu_over_U=0.5,
        beta=40.0,
        J=0.65,
        work_dir=tmp_path,
    )

    assert result.g_dlr.shape == delta.shape
    np.testing.assert_allclose(result.g_dlr, np.swapaxes(result.g_dlr.conj(), 0, 1), atol=1e-6)


def test_stabilize_g_dlr_preserves_hermiticity_and_adds_diagonal():
    grid = PydlrGrid(beta=40.0, lamb=600.0, eps=1e-8)
    g = np.zeros((3, 3, grid.n_tau), dtype=complex)
    g[0, 1] = 1.0 + 0.2j

    stabilized = stabilize_g_dlr(g, grid=grid, eps=1e-3)

    np.testing.assert_allclose(stabilized, np.swapaxes(stabilized.conj(), 0, 1))
    assert np.all(np.diagonal(stabilized, axis1=0, axis2=1).real < 0.0)


def test_mu_minus_eps_over_u_matches_dataset_scalar_convention():
    eps_d = np.diag([0.30, 0.33, 0.36]).astype(complex)
    assert _mu_minus_eps_over_u(12.3, eps_d, U=2.0) == (12.3 - 0.33) / 2.0


def test_regularize_sigma_iw_clamps_positive_frequency_diagonal():
    mesh = [-(2 * n + 1) * 1j for n in range(4, -1, -1)] + [(2 * n + 1) * 1j for n in range(5)]
    sigma = _FakeBlockGf(
        "up_0",
        np.stack([np.eye(3, dtype=complex) * (1.0 + 100.0j) for _ in mesh]),
        mesh,
    )

    diag = regularize_sigma_iw(sigma, tail_fraction=0.4, max_abs=50.0, causality_eps=1e-6)

    data = sigma.blocks[0].data
    pos = np.flatnonzero(np.asarray(mesh).imag > 0.0)
    assert diag["causality_violations"] > 0
    assert np.all(np.diagonal(data[pos], axis1=1, axis2=2).imag <= 0.0)
    for idx in pos:
        neg_idx = int(np.argmin(np.abs(np.asarray(mesh) + mesh[idx])))
        np.testing.assert_allclose(data[neg_idx], data[idx].conj())


def test_regularize_sigma_iw_uses_reference_tail_anchor():
    mesh = [-(2 * n + 1) * 1j for n in range(4, -1, -1)] + [(2 * n + 1) * 1j for n in range(5)]
    sigma = _FakeBlockGf(
        "up_0",
        np.stack([np.eye(3, dtype=complex) * (2.0 + 200.0j) for _ in mesh]),
        mesh,
    )
    reference = _FakeBlockGf(
        "up_0",
        np.stack([np.eye(3, dtype=complex) * (0.5 - 0.1j) for _ in mesh]),
        mesh,
    )

    regularize_sigma_iw(sigma, reference=reference, tail_fraction=0.2, max_abs=50.0)

    data = sigma.blocks[0].data
    ref = reference.blocks[0].data
    pos = np.flatnonzero(np.asarray(mesh).imag > 0.0)
    np.testing.assert_allclose(data[pos[-1]], ref[pos[-1]])


class _FakeBlock:
    def __init__(self, data: np.ndarray, mesh: list[complex]):
        self.data = np.asarray(data, dtype=complex)
        self.mesh = mesh


class _FakeBlockGf:
    def __init__(self, name: str, data: np.ndarray, mesh: list[complex]):
        self.blocks = [_FakeBlock(data, mesh)]
        self.names = [name]

    def __iter__(self):
        return iter(zip(self.names, self.blocks))
