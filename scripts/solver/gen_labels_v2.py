#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from scripts.solver.v2_common import finite_vec, physical_seed
from sigml.solver.hybridization import sample_delta_vec
from sigml.solver.labeler import CtsegLabeler
from sigml.solver.numpy_oracle import NumpyOrb1Oracle
from sigml.solver.valenti_grid import ValentiOrb1Grid


def oracle_near_sc_delta(
    oracle: NumpyOrb1Oracle,
    grid: ValentiOrb1Grid,
    rng: np.random.Generator,
    U: float,
    beta: float,
    t: float,
    mix: float,
    alpha: float,
) -> np.ndarray:
    if rng.random() < 0.5:
        delta = physical_seed(grid, t, scale=float(rng.uniform(0.15, 0.45)))
    else:
        delta = sample_delta_vec(grid, rng, alpha=alpha)
    n_steps = int(rng.integers(2, 8))
    for _ in range(n_steps):
        g = oracle.solve(delta, U=U, mu=U / 2.0, beta=beta, eps_d=0.0)
        candidate = (1.0 - mix) * (t**2) * g + mix * delta
        if not finite_vec(candidate) or float(np.max(np.abs(candidate))) > 5.0:
            break
        delta = np.asarray(candidate, dtype=np.float64)
    return delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate bounded v2 real-CTSEG labels.")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--beta", type=float, default=70.0)
    parser.add_argument("--u-min", type=float, default=1.0)
    parser.add_argument("--u-max", type=float, default=7.0)
    parser.add_argument("--near-sc-fraction", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--mix", type=float, default=0.5)
    parser.add_argument("--noise-every", type=int, default=10)
    parser.add_argument("--n-tau", type=int, default=10001)
    parser.add_argument("--n-cycles", type=int, default=30000)
    parser.add_argument("--n-warmup-cycles", type=int, default=3000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed + args.task_id)
    grid = ValentiOrb1Grid(args.mesh, beta=args.beta)
    oracle = NumpyOrb1Oracle(args.weights, mesh_path=args.mesh)
    labeler = CtsegLabeler(
        grid=grid,
        n_tau=args.n_tau,
        n_cycles=args.n_cycles,
        n_warmup_cycles=args.n_warmup_cycles,
        projection="dlr",
    )

    delta_rows = []
    g_rows = []
    u_rows = []
    mu_rows = []
    beta_rows = []
    eps_rows = []
    source_rows = []
    replicate_rows = []
    info_rows = []

    for sample_index in range(args.n_samples):
        U = float(rng.uniform(args.u_min, args.u_max))
        if rng.random() < args.near_sc_fraction:
            source = "oracle_near_sc"
            delta = oracle_near_sc_delta(oracle, grid, rng, U, args.beta, args.t, args.mix, args.alpha)
        else:
            source = "random"
            delta = sample_delta_vec(grid, rng, alpha=args.alpha)
        if not finite_vec(delta) or float(np.max(np.abs(delta))) > 5.0:
            source = "fallback_random"
            delta = sample_delta_vec(grid, rng, alpha=args.alpha)
        n_reps = 2 if args.noise_every > 0 and args.task_id % args.noise_every == 0 and sample_index == 0 else 1
        for rep in range(n_reps):
            g_vec = labeler.solve(delta, U=U, mu=U / 2.0, beta=args.beta, eps_d=0.0)
            delta_rows.append(delta)
            g_rows.append(g_vec)
            u_rows.append(U)
            mu_rows.append(U / 2.0)
            beta_rows.append(args.beta)
            eps_rows.append(0.0)
            source_rows.append(source)
            replicate_rows.append(f"{args.task_id}:{sample_index}")
            info = labeler.last_info.__dict__ if labeler.last_info is not None else {}
            info["source"] = source
            info["replicate_key"] = replicate_rows[-1]
            info["replicate_index"] = rep
            info_rows.append(json.dumps(info, sort_keys=True))

    rank = int(os.environ.get("PMI_RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0")))
    if rank == 0:
        output = out_dir / f"labels_v2_{args.task_id:05d}.npz"
        np.savez(
            output,
            delta=np.asarray(delta_rows, dtype=np.float64),
            g=np.asarray(g_rows, dtype=np.float64),
            U=np.asarray(u_rows, dtype=np.float64),
            mu=np.asarray(mu_rows, dtype=np.float64),
            beta=np.asarray(beta_rows, dtype=np.float64),
            eps_d=np.asarray(eps_rows, dtype=np.float64),
            source=np.asarray(source_rows),
            replicate_key=np.asarray(replicate_rows),
            info_json=np.asarray(info_rows),
            elapsed_seconds=np.asarray([time.time() - start], dtype=np.float64),
        )
        print(f"wrote {output} rows={len(g_rows)} elapsed={time.time() - start:.3f}s")


if __name__ == "__main__":
    main()
