#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from sigml.solver.pydlr_grid import PydlrGrid
from sigml.solver.srvo3_nn_harness import (
    Srvo3NNSidecarSettings,
    default_base_python_command,
    result_to_json,
    run_srvo3_nn_harness,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SrVO3 Stage-2 NN sidecar DMFT harness with a stub net."
    )
    parser.add_argument("--h5", required=True, help="Input public/tutorial svo.h5 archive.")
    parser.add_argument("--out-dir", required=True, help="Scratch output directory.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--ckpt", default=None, help="Checkpoint path; created as an untrained stub if absent.")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--beta", type=float, default=40.0)
    parser.add_argument("--n-iw", type=int, default=1025)
    parser.add_argument("--omega-max", type=float, default=15.0)
    parser.add_argument("--eps", type=float, default=1e-10)
    parser.add_argument("--mix-sigma", type=float, default=0.5)
    parser.add_argument("--mu-precision", type=float, default=0.01)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument(
        "--convergence-tol",
        type=float,
        default=None,
        help="Optional stop tolerance on consecutive NN G(tau) MSE.",
    )
    parser.add_argument("--min-iterations", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--python-command",
        nargs="+",
        default=None,
        help='Torch sidecar command prefix, e.g. "conda run -n base python".',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = PydlrGrid(beta=args.beta, lamb=args.beta * args.omega_max, eps=args.eps)
    ckpt = Path(args.ckpt).resolve() if args.ckpt else out_dir / "stub_block_resnet.pt"
    python_command = tuple(args.python_command or default_base_python_command())
    if not ckpt.exists():
        _write_stub_checkpoint_with_sidecar_python(
            ckpt,
            repo_root=repo_root,
            python_command=python_command,
            n_tau=grid.n_tau,
        )

    settings = Srvo3NNSidecarSettings(
        checkpoint_path=ckpt,
        repo_root=repo_root,
        cli_path=repo_root / "sigml" / "solver" / "nn_solver_cli.py",
        python_command=python_command,
        timeout_seconds=args.timeout_seconds,
    )
    result = run_srvo3_nn_harness(
        h5_path=args.h5,
        output_dir=out_dir,
        settings=settings,
        n_iterations=args.iterations,
        beta=args.beta,
        n_iw=args.n_iw,
        omega_max=args.omega_max,
        eps=args.eps,
        mix_sigma=args.mix_sigma,
        mu_precision=args.mu_precision,
        regularization=args.regularization,
        convergence_tol=args.convergence_tol,
        min_iterations=args.min_iterations,
    )
    print(json.dumps(result_to_json(result), indent=2, sort_keys=True))


def _write_stub_checkpoint_with_sidecar_python(
    ckpt: Path,
    *,
    repo_root: Path,
    python_command: tuple[str, ...],
    n_tau: int,
) -> None:
    code = (
        "from sigml.solver.srvo3_nn_harness import write_stub_block_resnet_checkpoint; "
        f"write_stub_block_resnet_checkpoint({str(ckpt)!r}, "
        f"orbital_dim=3, n_tau={int(n_tau)}, hidden_dim=32, num_layers=1)"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    completed = subprocess.run(
        [*python_command, "-c", code],
        text=True,
        capture_output=True,
        env=env,
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Failed to create stub checkpoint with sidecar Python\n"
            f"command: {' '.join([*python_command, '-c', code])}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


if __name__ == "__main__":
    main()
