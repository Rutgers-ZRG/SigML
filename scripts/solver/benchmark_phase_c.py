#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sigml.solver.bethe import dmft_loop
from sigml.solver.dyson import positive_matsubara_mask, sigma_from_g
from sigml.solver.labeler import CtsegLabeler
from sigml.solver.metrics import quasiparticle_proxy
from sigml.solver.numpy_oracle import NumpyOrb1Oracle
from sigml.solver.valenti_grid import ValentiOrb1Grid


def physical_seed(grid: ValentiOrb1Grid, beta: float, t: float, scale: float = 0.25) -> np.ndarray:
    tau = grid.tau_nodes
    gtau = -scale * np.ones_like(tau, dtype=complex)
    return (t**2) * grid.gtau_to_vec(gtau).real


def ctseg_dmft_loop(
    labeler: CtsegLabeler,
    grid: ValentiOrb1Grid,
    U: float,
    beta: float,
    t: float,
    mix: float,
    tol: float,
    max_iter: int,
    initial_delta_vec: np.ndarray,
) -> dict:
    mu = U / 2.0
    delta_vec = np.asarray(initial_delta_vec, dtype=float).copy()
    last_g = np.zeros(grid.feature_dim)
    solve_delta_vec = delta_vec.copy()
    errors = []
    for n_iter in range(1, max_iter + 1):
        solve_delta_vec = delta_vec.copy()
        last_g = labeler.solve(delta_vec, U=U, mu=mu, beta=beta, eps_d=0.0)
        delta_new = (t**2) * last_g
        delta_next = (1.0 - mix) * delta_new + mix * delta_vec
        err = float(np.linalg.norm(delta_next - delta_vec) / np.sqrt(grid.feature_dim))
        errors.append(err)
        delta_vec = delta_next
        if err < tol:
            break
    return {
        "converged": errors[-1] < tol if errors else False,
        "n_iter": n_iter,
        "errors": errors,
        "g_vec": last_g,
        "delta_vec": delta_vec,
        "solve_delta_vec": solve_delta_vec,
    }


def sigma_low(grid: ValentiOrb1Grid, g_vec: np.ndarray, delta_vec: np.ndarray, U: float, n: int) -> np.ndarray:
    g_iw = grid.gtau_to_giw(grid.vec_to_gtau(g_vec))
    delta_iw = grid.gtau_to_giw(grid.vec_to_gtau(delta_vec))
    sigma = sigma_from_g(g_iw, delta_iw, mu=U / 2.0, eps_d=0.0, iw=grid.iw_nodes)
    pos_idx = np.flatnonzero(positive_matsubara_mask(grid.iw_nodes))
    order = np.argsort(np.abs(grid.iw_nodes[pos_idx].imag))
    return sigma[pos_idx[order[:n]]]


def run_mott_curve(
    oracle: NumpyOrb1Oracle,
    grid: ValentiOrb1Grid,
    beta: float,
    t: float,
    u_values: list[float],
    mix: float,
    tol: float,
    max_iter: int,
) -> dict:
    up = []
    down = []
    seed = physical_seed(grid, beta, t)
    for U in u_values:
        res = dmft_loop(
            oracle.solve,
            U=U,
            mu=U / 2.0,
            beta=beta,
            t=t,
            grid=grid,
            mix=mix,
            tol=tol,
            max_iter=max_iter,
            initial_delta_vec=seed,
        )
        seed = res.delta_vec
        up.append(
            {
                "U": U,
                "converged": res.converged,
                "n_iter": res.n_iter,
                "minus_g_beta_over_2": quasiparticle_proxy(grid.vec_to_gtau(res.g_vec), grid, beta),
            }
        )
    seed = np.zeros(grid.feature_dim)
    for U in reversed(u_values):
        res = dmft_loop(
            oracle.solve,
            U=U,
            mu=U / 2.0,
            beta=beta,
            t=t,
            grid=grid,
            mix=mix,
            tol=tol,
            max_iter=max_iter,
            initial_delta_vec=seed,
        )
        seed = res.delta_vec
        down.append(
            {
                "U": U,
                "converged": res.converged,
                "n_iter": res.n_iter,
                "minus_g_beta_over_2": quasiparticle_proxy(grid.vec_to_gtau(res.g_vec), grid, beta),
            }
        )
    return {"up": up, "down": list(reversed(down))}


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# SigML NN Solver PoC Results",
        "",
        f"Generated on Amarel run directory: `{summary['run_dir']}`",
        "",
        "## NN + 1 CTQMC Refinement",
        "",
        "| U | full CTSEG iters | NN iters | refine CTSEG iters | low-frequency Sigma max abs diff |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary["comparison"]:
        lines.append(
            f"| {row['U']:.3g} | {row['full_ctseg_n_iter']} | {row['nn_n_iter']} | "
            f"1 | {row['sigma_low_max_abs_diff']:.6g} |"
        )
    lines.extend(["", "## Mott / Hysteresis", ""])
    lines.append("| sweep | U | -G(beta/2) | converged | iterations |")
    lines.append("|---|---:|---:|---|---:|")
    for sweep in ("up", "down"):
        for row in summary["mott_curve"][sweep]:
            lines.append(
                f"| {sweep} | {row['U']:.3g} | {row['minus_g_beta_over_2']:.6g} | "
                f"{row['converged']} | {row['n_iter']} |"
            )
    lines.extend(
        [
            "",
            "## Real-CTSEG Training Set",
            "",
            summary["training_set_note"],
            "",
            "## Run Metadata",
            "",
            "```json",
            json.dumps(summary["metadata"], indent=2, sort_keys=True),
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-C full CTSEG vs NN+1 benchmark.")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--u-values", nargs="+", type=float, default=[2.0, 4.0])
    parser.add_argument("--mott-u-values", nargs="+", type=float, default=[2.0, 3.0, 4.0, 5.0, 6.0])
    parser.add_argument("--beta", type=float, default=70.0)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--mix", type=float, default=0.5)
    parser.add_argument("--tol", type=float, default=5e-3)
    parser.add_argument("--max-ctseg-iter", type=int, default=12)
    parser.add_argument("--max-nn-iter", type=int, default=80)
    parser.add_argument("--n-low", type=int, default=8)
    parser.add_argument("--n-tau", type=int, default=10001)
    parser.add_argument("--n-cycles", type=int, default=100000)
    parser.add_argument("--n-warmup-cycles", type=int, default=5000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = ValentiOrb1Grid(args.mesh, beta=args.beta)
    oracle = NumpyOrb1Oracle(args.weights, mesh_path=args.mesh)
    ctseg = CtsegLabeler(
        grid=grid,
        n_tau=args.n_tau,
        n_cycles=args.n_cycles,
        n_warmup_cycles=args.n_warmup_cycles,
        projection="dlr",
    )

    comparison = []
    for U in args.u_values:
        seed = physical_seed(grid, args.beta, args.t)
        full = ctseg_dmft_loop(
            ctseg,
            grid,
            U=U,
            beta=args.beta,
            t=args.t,
            mix=args.mix,
            tol=args.tol,
            max_iter=args.max_ctseg_iter,
            initial_delta_vec=seed,
        )
        nn = dmft_loop(
            oracle.solve,
            U=U,
            mu=U / 2.0,
            beta=args.beta,
            t=args.t,
            grid=grid,
            mix=args.mix,
            tol=args.tol,
            max_iter=args.max_nn_iter,
            initial_delta_vec=seed,
        )
        refined_g = ctseg.solve(nn.delta_vec, U=U, mu=U / 2.0, beta=args.beta, eps_d=0.0)
        full_sigma = sigma_low(grid, full["g_vec"], full["solve_delta_vec"], U, args.n_low)
        refined_sigma = sigma_low(grid, refined_g, nn.delta_vec, U, args.n_low)
        comparison.append(
            {
                "U": U,
                "full_ctseg_converged": full["converged"],
                "full_ctseg_n_iter": full["n_iter"],
                "full_ctseg_errors": full["errors"],
                "nn_converged": nn.converged,
                "nn_n_iter": nn.n_iter,
                "sigma_low_max_abs_diff": float(np.max(np.abs(full_sigma - refined_sigma))),
                "sigma_low_mean_abs_diff": float(np.mean(np.abs(full_sigma - refined_sigma))),
                "full_minus_g_beta_over_2": quasiparticle_proxy(grid.vec_to_gtau(full["g_vec"]), grid, args.beta),
                "refined_minus_g_beta_over_2": quasiparticle_proxy(grid.vec_to_gtau(refined_g), grid, args.beta),
            }
        )

    summary = {
        "run_dir": str(Path.cwd()),
        "comparison": comparison,
        "mott_curve": run_mott_curve(
            oracle,
            grid,
            beta=args.beta,
            t=args.t,
            u_values=args.mott_u_values,
            mix=args.mix,
            tol=args.tol,
            max_iter=args.max_nn_iter,
        ),
        "training_set_note": (
            "Scale-out real-CTSEG training was not launched by this benchmark script. "
            "Use scripts/solver/gen_labels.slurm after measuring the per-solve cost from this run."
        ),
        "metadata": vars(args),
    }
    (out_dir / "phase_c_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    write_markdown(out_dir / "solver_poc_results.md", summary)
    print(json.dumps(summary["comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
