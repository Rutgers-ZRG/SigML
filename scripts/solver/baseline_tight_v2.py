#!/usr/bin/env python
from __future__ import annotations

import argparse
import time

from scripts.solver.v2_common import ctseg_dmft_loop, physical_seed, write_json
from sigml.solver.labeler import CtsegLabeler
from sigml.solver.valenti_grid import ValentiOrb1Grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tight-tolerance full CTSEG Bethe baselines.")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--beta", type=float, default=70.0)
    parser.add_argument("--U", type=float, default=4.0)
    parser.add_argument("--tols", nargs="+", type=float, default=[1e-4, 1e-5])
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--mix", type=float, default=0.5)
    parser.add_argument("--max-ctseg-iter", type=int, default=20)
    parser.add_argument("--n-tau", type=int, default=10001)
    parser.add_argument("--n-cycles", type=int, default=100000)
    parser.add_argument("--n-warmup-cycles", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    grid = ValentiOrb1Grid(args.mesh, beta=args.beta)
    labeler = CtsegLabeler(
        grid=grid,
        n_tau=args.n_tau,
        n_cycles=args.n_cycles,
        n_warmup_cycles=args.n_warmup_cycles,
        projection="dlr",
    )
    seed = physical_seed(grid, args.t)
    rows = []
    for tol in args.tols:
        res = ctseg_dmft_loop(
            labeler, grid, args.U, args.beta, args.t, args.mix, tol, args.max_ctseg_iter, seed
        )
        rows.append(
            {
                "tol": tol,
                "converged": res["converged"],
                "n_iter": res["n_iter"],
                "errors": res["errors"],
                "projection_errors": res["projection_errors"],
                "info": res["info"],
            }
        )
    write_json(
        args.out,
        {
            "metadata": vars(args),
            "elapsed_seconds_python": time.time() - start,
            "baselines": rows,
        },
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
