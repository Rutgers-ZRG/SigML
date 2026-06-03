#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from scripts.solver.v2_common import (
    compare_state,
    ctseg_dmft_loop,
    low_freq,
    nn_dmft_state,
    physical_seed,
    sigma_iw,
    write_json,
)
from sigml.solver.labeler import CtsegLabeler
from sigml.solver.numpy_oracle import NumpyOrb1Oracle
from sigml.solver.valenti_grid import ValentiOrb1Grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a v2 trained net against full CTSEG.")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--beta", type=float, default=70.0)
    parser.add_argument("--U", type=float, action="append", default=None)
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


def minus_g_mid(grid: ValentiOrb1Grid, g_vec: np.ndarray, beta: float) -> float:
    return float(-grid.eval_at_tau(grid.vec_to_gtau(g_vec), beta / 2.0).real)


def compact_full_state(state: dict) -> dict:
    return {
        "converged": state["converged"],
        "n_iter": state["n_iter"],
        "errors": state["errors"],
        "projection_errors": state["projection_errors"],
        "info": state["info"],
    }


def low_sigma_table(grid: ValentiOrb1Grid, state: dict, U: float, n_low: int) -> list[list[float]]:
    values = low_freq(grid, sigma_iw(grid, state["g_vec"], state["solve_delta_vec"], U), n_low)
    return [[float(z.real), float(z.imag)] for z in values]


def run_case(args: argparse.Namespace, grid: ValentiOrb1Grid, net: NumpyOrb1Oracle, U: float) -> dict:
    ctseg = CtsegLabeler(
        grid=grid,
        n_tau=args.n_tau,
        n_cycles=args.n_cycles,
        n_warmup_cycles=args.n_warmup_cycles,
        projection="dlr",
    )
    seed = physical_seed(grid, args.t)
    full = ctseg_dmft_loop(
        ctseg, grid, U, args.beta, args.t, args.mix, args.tol, args.max_ctseg_iter, seed
    )
    nn = nn_dmft_state(
        net.solve, grid, U, args.beta, args.t, args.mix, args.tol, args.max_nn_iter, seed
    )
    refined_g = ctseg.solve(nn["delta_vec"], U=U, mu=U / 2.0, beta=args.beta, eps_d=0.0)
    refined = {
        "converged": True,
        "n_iter": int(nn["n_iter"] + 1),
        "g_vec": refined_g,
        "delta_vec": nn["delta_vec"],
        "solve_delta_vec": nn["delta_vec"],
    }

    projection_error = None
    if ctseg.last_direct_g_vec is not None and ctseg.last_dlr_g_vec is not None:
        diff = ctseg.last_direct_g_vec - ctseg.last_dlr_g_vec
        projection_error = {
            "max_abs": float(np.max(np.abs(diff))),
            "mean_abs": float(np.mean(np.abs(diff))),
            "rms": float(np.sqrt(np.mean(np.abs(diff) ** 2))),
        }

    return {
        "U": float(U),
        "full_ctseg": compact_full_state(full),
        "nn": {
            "converged": nn["converged"],
            "n_iter": nn["n_iter"],
            "delta_norm": float(np.linalg.norm(nn["delta_vec"]) / np.sqrt(grid.feature_dim)),
        },
        "nn_plus_one_ctseg": {
            "n_iter": refined["n_iter"],
            "info": ctseg.last_info.__dict__ if ctseg.last_info is not None else {},
            "projection_error": projection_error,
        },
        "observables": {
            "minus_G_beta_over_2": {
                "full_ctseg": minus_g_mid(grid, full["g_vec"], args.beta),
                "nn": minus_g_mid(grid, nn["g_vec"], args.beta),
                "nn_plus_one_ctseg": minus_g_mid(grid, refined["g_vec"], args.beta),
            }
        },
        "comparisons": {
            "full_vs_nn": compare_state(grid, full, nn, U, args.n_low),
            "full_vs_nn_plus_one_ctseg": compare_state(grid, full, refined, U, args.n_low),
            "nn_vs_nn_plus_one_ctseg": compare_state(grid, nn, refined, U, args.n_low),
        },
        "sigma_iw_low_diagnostic": {
            "full_ctseg": low_sigma_table(grid, full, U, args.n_low),
            "nn": low_sigma_table(grid, nn, U, args.n_low),
            "nn_plus_one_ctseg": low_sigma_table(grid, refined, U, args.n_low),
        },
        "vectors": {
            "full_g_vec": full["g_vec"],
            "full_solve_delta_vec": full["solve_delta_vec"],
            "nn_g_vec": nn["g_vec"],
            "nn_delta_vec": nn["delta_vec"],
            "nn_plus_one_g_vec": refined["g_vec"],
        },
    }


def main() -> None:
    args = parse_args()
    start = time.time()
    args.U = args.U if args.U is not None else [2.0, 4.0]
    grid = ValentiOrb1Grid(args.mesh, beta=args.beta)
    net = NumpyOrb1Oracle(args.weights, mesh_path=args.mesh)

    cases = [run_case(args, grid, net, float(U)) for U in args.U]
    npz_payload: dict[str, np.ndarray] = {}
    serializable_cases = []
    for case in cases:
        vectors = case.pop("vectors")
        u_tag = f"U{case['U']:g}".replace(".", "p")
        for name, vec in vectors.items():
            npz_payload[f"{u_tag}_{name}"] = np.asarray(vec)
        serializable_cases.append(case)

    payload = {
        "metadata": vars(args),
        "elapsed_seconds_python": time.time() - start,
        "cases": serializable_cases,
    }
    write_json(args.out, payload)
    rank = int(os.environ.get("PMI_RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0")))
    if rank == 0 and npz_payload:
        np.savez(Path(args.out).with_suffix(".npz"), **npz_payload)
    if rank == 0:
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
