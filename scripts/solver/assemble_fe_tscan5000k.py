#!/usr/bin/env python
from __future__ import annotations

import argparse

from sigml.solver.edmft_fe import assemble_fe_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble Haule eDMFT/Wien2k Fe tscan_5000k_v3 labels into beta=2.32 M=5 block data."
    )
    parser.add_argument("--source", default="/scratch/lz432/tscan_5000k_v3")
    parser.add_argument("--output", default="data/fe_tscan5000k_b232.npz")
    parser.add_argument("--summary", default="data/fe_tscan5000k_b232_summary.json")
    parser.add_argument("--beta", type=float, default=2.32)
    parser.add_argument("--lamb", type=float, default=80.0)
    parser.add_argument("--eps", type=float, default=1e-10)
    parser.add_argument("--exclude-current", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = assemble_fe_dataset(
        args.source,
        args.output,
        summary=args.summary,
        beta=args.beta,
        lamb=args.lamb,
        eps=args.eps,
        include_current=not args.exclude_current,
    )
    print(
        f"wrote {summary['output']} labels={summary['n_labels']} "
        f"delta_shape={summary['delta_shape']} rank={summary['rank']}"
    )


if __name__ == "__main__":
    main()
