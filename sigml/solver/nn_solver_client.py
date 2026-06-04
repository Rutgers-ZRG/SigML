from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from sigml.solver.nn_solver_schema import (
    SolverOutput,
    read_solver_output,
    write_solver_input,
)


@dataclass(frozen=True)
class NNSolverConfig:
    """Subprocess configuration for the torch sidecar.

    This module intentionally has no torch dependency so it can be imported in
    the TRIQS/solid_dmft Python environment. Use python_command for env wrappers,
    for example ("conda", "run", "-n", "nequip", "python").
    """

    checkpoint_path: str | Path
    cli_path: str | Path = Path(__file__).with_name("nn_solver_cli.py")
    python_executable: str | Path = sys.executable
    python_command: Sequence[str] | None = None
    env: Mapping[str, str] | None = None
    timeout_seconds: float | None = None

    def command_prefix(self) -> list[str]:
        if self.python_command is not None:
            return [str(part) for part in self.python_command]
        return [str(self.python_executable)]


class NNSolverSubprocessError(RuntimeError):
    pass


def run_nn_solver(
    config: NNSolverConfig,
    *,
    delta_dlr: np.ndarray,
    U: float,
    mu_over_U: float,
    beta: float,
    J: float,
    work_dir: str | Path | None = None,
) -> SolverOutput:
    if work_dir is None:
        with tempfile.TemporaryDirectory(prefix="sigml-nn-solver-") as tmp:
            return _run_with_paths(config, Path(tmp), delta_dlr, U, mu_over_U, beta, J)
    path = Path(work_dir)
    path.mkdir(parents=True, exist_ok=True)
    return _run_with_paths(config, path, delta_dlr, U, mu_over_U, beta, J)


def _run_with_paths(
    config: NNSolverConfig,
    work_dir: Path,
    delta_dlr: np.ndarray,
    U: float,
    mu_over_U: float,
    beta: float,
    J: float,
) -> SolverOutput:
    in_path = work_dir / "nn_solver_input.npz"
    out_path = work_dir / "nn_solver_output.npz"
    write_solver_input(
        in_path,
        delta_dlr=delta_dlr,
        U=U,
        mu_over_U=mu_over_U,
        beta=beta,
        J=J,
    )
    run_nn_solver_files(config, input_path=in_path, output_path=out_path)
    return read_solver_output(out_path)


def run_nn_solver_files(
    config: NNSolverConfig,
    *,
    input_path: str | Path,
    output_path: str | Path,
) -> None:
    cmd = [
        *config.command_prefix(),
        str(config.cli_path),
        "--in",
        str(input_path),
        "--out",
        str(output_path),
        "--ckpt",
        str(config.checkpoint_path),
    ]
    env = os.environ.copy()
    if config.env is not None:
        env.update({str(key): str(value) for key, value in config.env.items()})

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=config.timeout_seconds,
    )
    if completed.returncode != 0:
        raise NNSolverSubprocessError(
            "NN solver sidecar failed with exit code "
            f"{completed.returncode}\ncommand: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
