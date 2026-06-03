#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from scripts.solver.v2_common import (
    compare_state,
    ctseg_dmft_loop,
    low_freq,
    metric_summary,
    nn_dmft_state,
    physical_seed,
    sigma_iw,
    write_json,
)
from sigml.solver.labeler import CtsegLabeler
from sigml.solver.numpy_oracle import NumpyOrb1Oracle
from sigml.solver.valenti_grid import ValentiOrb1Grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose U=4 Sigma mismatch for the SigML solver PoC.")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--beta", type=float, default=70.0)
    parser.add_argument("--U", type=float, default=4.0)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--mix", type=float, default=0.5)
    parser.add_argument("--tol", type=float, default=5e-3)
    parser.add_argument("--max-ctseg-iter", type=int, default=8)
    parser.add_argument("--max-nn-iter", type=int, default=80)
    parser.add_argument("--n-low", type=int, default=8)
    parser.add_argument("--n-tau", type=int, default=10001)
    parser.add_argument("--n-cycles", type=int, default=100000)
    parser.add_argument("--n-warmup-cycles", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    grid = ValentiOrb1Grid(args.mesh, beta=args.beta)
    oracle = NumpyOrb1Oracle(args.weights, mesh_path=args.mesh)
    ctseg = CtsegLabeler(
        grid=grid,
        n_tau=args.n_tau,
        n_cycles=args.n_cycles,
        n_warmup_cycles=args.n_warmup_cycles,
        projection="dlr",
    )
    seed = physical_seed(grid, args.t)

    full = ctseg_dmft_loop(
        ctseg, grid, args.U, args.beta, args.t, args.mix, args.tol, args.max_ctseg_iter, seed
    )
    nn = nn_dmft_state(
        oracle.solve, grid, args.U, args.beta, args.t, args.mix, args.tol, args.max_nn_iter, seed
    )
    refined_g = ctseg.solve(nn["delta_vec"], U=args.U, mu=args.U / 2.0, beta=args.beta, eps_d=0.0)
    refined = {
        "converged": True,
        "n_iter": 1,
        "g_vec": refined_g,
        "delta_vec": nn["delta_vec"],
        "solve_delta_vec": nn["delta_vec"],
        "projection_error": (
            metric_summary(ctseg.last_direct_g_vec, ctseg.last_dlr_g_vec)
            if ctseg.last_direct_g_vec is not None and ctseg.last_dlr_g_vec is not None
            else None
        ),
        "info": ctseg.last_info.__dict__ if ctseg.last_info is not None else {},
    }

    full_sigma = low_freq(grid, sigma_iw(grid, full["g_vec"], full["solve_delta_vec"], args.U), args.n_low)
    refined_sigma = low_freq(
        grid, sigma_iw(grid, refined["g_vec"], refined["solve_delta_vec"], args.U), args.n_low
    )
    nn_sigma = low_freq(grid, sigma_iw(grid, nn["g_vec"], nn["solve_delta_vec"], args.U), args.n_low)

    summary = {
        "metadata": vars(args),
        "elapsed_seconds_python": time.time() - start,
        "states": {
            "full_ctseg": {
                "converged": full["converged"],
                "n_iter": full["n_iter"],
                "errors": full["errors"],
                "projection_errors": full["projection_errors"],
                "info": full["info"],
            },
            "nn": {"converged": nn["converged"], "n_iter": nn["n_iter"]},
            "nn_plus_one_ctseg": {
                "projection_error": refined["projection_error"],
                "info": refined["info"],
            },
        },
        "comparisons": {
            "full_vs_nn": compare_state(grid, full, nn, args.U, args.n_low),
            "full_vs_nn_plus_one_ctseg": compare_state(grid, full, refined, args.U, args.n_low),
            "nn_vs_nn_plus_one_ctseg": compare_state(grid, nn, refined, args.U, args.n_low),
            "full_sigma_low": [[float(z.real), float(z.imag)] for z in full_sigma],
            "nn_sigma_low": [[float(z.real), float(z.imag)] for z in nn_sigma],
            "refined_sigma_low": [[float(z.real), float(z.imag)] for z in refined_sigma],
        },
    }
    write_json(args.out, summary)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
