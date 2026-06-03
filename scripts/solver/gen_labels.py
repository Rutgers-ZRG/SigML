#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from sigml.solver.hybridization import sample_delta_vec
from sigml.solver.labeler import CtsegLabeler
from sigml.solver.valenti_grid import ValentiOrb1Grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CTSEG labels for sampled Delta vectors.")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--beta", type=float, default=70.0)
    parser.add_argument("--u-min", type=float, default=2.0)
    parser.add_argument("--u-max", type=float, default=6.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-tau", type=int, default=10001)
    parser.add_argument("--n-cycles", type=int, default=100000)
    parser.add_argument("--n-warmup-cycles", type=int, default=5000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed + args.task_id)
    grid = ValentiOrb1Grid(args.mesh, beta=args.beta)
    labeler = CtsegLabeler(
        grid=grid,
        n_tau=args.n_tau,
        n_cycles=args.n_cycles,
        n_warmup_cycles=args.n_warmup_cycles,
        projection="dlr",
    )

    deltas = []
    labels = []
    params = []
    infos = []
    for _ in range(args.n_samples):
        U = float(rng.uniform(args.u_min, args.u_max))
        mu = U / 2.0
        delta_vec = sample_delta_vec(grid, rng, alpha=args.alpha)
        g_vec = labeler.solve(delta_vec, U=U, mu=mu, beta=args.beta, eps_d=0.0)
        deltas.append(delta_vec)
        labels.append(g_vec)
        params.append((U, mu, args.beta))
        infos.append(labeler.last_info.__dict__ if labeler.last_info is not None else {})

    output = out_dir / f"labels_{args.task_id:05d}.npz"
    np.savez(
        output,
        delta_vec=np.asarray(deltas),
        g_vec=np.asarray(labels),
        params=np.asarray(params),
        info_json=np.asarray([json.dumps(info, sort_keys=True) for info in infos]),
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
