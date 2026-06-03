from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from sigml.solver.nn_solver_client import (
    NNSolverConfig,
    NNSolverSubprocessError,
    run_nn_solver,
    write_solver_input,
)
from sigml.solver.nn_solver_schema import (
    flatten_block_features,
    read_solver_output,
    solver_input_features,
    unflatten_block_features,
)


def _hermitian_blocks(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.normal(size=(3, 3, 5)) + 1j * rng.normal(size=(3, 3, 5))
    return 0.5 * (arr + np.swapaxes(arr.conj(), 0, 1))


def test_nn_solver_npz_schema_round_trips_block_dlr_and_scalars(tmp_path: Path):
    delta = _hermitian_blocks()
    in_path = tmp_path / "in.npz"

    write_solver_input(
        in_path,
        delta_dlr=delta,
        U=4.2,
        mu_over_U=0.48,
        beta=70.0,
        J=0.6,
    )

    features, metadata = solver_input_features(in_path)
    restored = unflatten_block_features(
        flatten_block_features(delta), orbital_dim=3, n_tau=5
    )

    np.testing.assert_allclose(metadata.delta_dlr, delta)
    np.testing.assert_allclose(restored, delta)
    np.testing.assert_allclose(features[:-4], flatten_block_features(delta))
    np.testing.assert_allclose(features[-4:], np.array([4.2, 0.48, 70.0, 0.6], dtype=np.float32))
    assert features.dtype == np.float32


def test_nn_solver_cli_round_trip_matches_in_process_stub(tmp_path: Path):
    delta = _hermitian_blocks(seed=1).astype(np.complex64)
    in_path = tmp_path / "in.npz"
    out_path = tmp_path / "out.npz"
    ckpt_path = tmp_path / "stub.pt"

    write_solver_input(
        in_path,
        delta_dlr=delta,
        U=3.5,
        mu_over_U=0.5,
        beta=40.0,
        J=0.4,
    )
    features, _ = solver_input_features(in_path)
    output_dim = features.size - 4

    stub = torch.nn.Linear(features.size, output_dim, bias=False)
    with torch.no_grad():
        stub.weight.zero_()
        stub.weight[:, :output_dim] = torch.eye(output_dim)
    torch.save(
        {
            "architecture": "linear",
            "input_dim": features.size,
            "output_dim": output_dim,
            "state_dict": stub.state_dict(),
        },
        ckpt_path,
    )

    x = torch.from_numpy(features[None, :])
    with torch.no_grad():
        expected = unflatten_block_features(
            stub(x).numpy()[0], orbital_dim=3, n_tau=5
        )

    result = run_nn_solver(
        NNSolverConfig(
            cli_path=Path("sigml/solver/nn_solver_cli.py"),
            checkpoint_path=ckpt_path,
            python_executable=sys.executable,
        ),
        delta_dlr=delta,
        U=3.5,
        mu_over_U=0.5,
        beta=40.0,
        J=0.4,
        work_dir=tmp_path,
    )

    np.testing.assert_allclose(result.g_dlr, expected, atol=1e-7, rtol=0.0)

    direct = subprocess.run(
        [
            sys.executable,
            "sigml/solver/nn_solver_cli.py",
            "--in",
            str(in_path),
            "--out",
            str(out_path),
            "--ckpt",
            str(ckpt_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert direct.stderr == ""
    np.testing.assert_allclose(read_solver_output(out_path).g_dlr, expected, atol=1e-7, rtol=0.0)


def test_nn_solver_client_propagates_subprocess_failure(tmp_path: Path):
    delta = _hermitian_blocks(seed=2)

    with pytest.raises(NNSolverSubprocessError) as excinfo:
        run_nn_solver(
            NNSolverConfig(
                cli_path=Path("sigml/solver/nn_solver_cli.py"),
                checkpoint_path=tmp_path / "missing.pt",
                python_executable=sys.executable,
            ),
            delta_dlr=delta,
            U=4.0,
            mu_over_U=0.5,
            beta=70.0,
            J=0.6,
            work_dir=tmp_path,
        )

    message = str(excinfo.value)
    assert "NN solver sidecar failed with exit code" in message
    assert "missing.pt" in message
    assert "stderr:" in message
