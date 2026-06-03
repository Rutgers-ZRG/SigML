#!/usr/bin/env python
from __future__ import annotations

import argparse
import time

import numpy as np

from scripts.solver.v2_common import compare_state, ctseg_dmft_loop, nn_dmft_state, physical_seed, write_json
from sigml.solver.labeler import CtsegLabeler
from sigml.solver.numpy_oracle import NumpyOrb1Oracle
from sigml.solver.valenti_grid import ValentiOrb1Grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a v2 trained net against full CTSEG.")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--beta", type=float, default=70.0)
    parser.add_argument("--U", type=float, default=4.0)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--mix", type=float, default=0.5)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--max-ctseg-iter", type=int, default=14)
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
    net = NumpyOrb1Oracle(args.weights, mesh_path=args.mesh)
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
        net.solve, grid, args.U, args.beta, args.t, args.mix, args.tol, args.max_nn_iter, seed
    )
    refined_g = ctseg.solve(nn["delta_vec"], U=args.U, mu=args.U / 2.0, beta=args.beta, eps_d=0.0)
    refined = {
        "g_vec": refined_g,
        "delta_vec": nn["delta_vec"],
        "solve_delta_vec": nn["delta_vec"],
    }
    payload = {
        "metadata": vars(args),
        "elapsed_seconds_python": time.time() - start,
        "full_ctseg": {
            "converged": full["converged"],
            "n_iter": full["n_iter"],
            "errors": full["errors"],
            "projection_errors": full["projection_errors"],
            "info": full["info"],
        },
        "nn": {
            "converged": nn["converged"],
            "n_iter": nn["n_iter"],
            "delta_norm": float(np.linalg.norm(nn["delta_vec"]) / np.sqrt(grid.feature_dim)),
        },
        "nn_plus_one_ctseg": {
            "info": ctseg.last_info.__dict__ if ctseg.last_info is not None else {},
            "projection_error": (
                None
                if ctseg.last_direct_g_vec is None or ctseg.last_dlr_g_vec is None
                else {
                    "max_abs": float(np.max(np.abs(ctseg.last_direct_g_vec - ctseg.last_dlr_g_vec))),
                    "mean_abs": float(np.mean(np.abs(ctseg.last_direct_g_vec - ctseg.last_dlr_g_vec))),
                }
            ),
        },
        "comparisons": {
            "full_vs_nn": compare_state(grid, full, nn, args.U, args.n_low),
            "full_vs_nn_plus_one_ctseg": compare_state(grid, full, refined, args.U, args.n_low),
            "nn_vs_nn_plus_one_ctseg": compare_state(grid, nn, refined, args.U, args.n_low),
        },
    }
    write_json(args.out, payload)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
