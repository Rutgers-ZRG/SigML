#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import triqs.utility.mpi as mpi

from sigml.solver.dyson import positive_matsubara_mask
from sigml.solver.labeler import CtsegLabeler
from sigml.solver.valenti_grid import ValentiOrb1Grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CtsegLabeler on a near-atomic bath.")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--beta", type=float, default=70.0)
    parser.add_argument("--U", type=float, default=2.0)
    parser.add_argument("--bath-v", type=float, default=0.10)
    parser.add_argument("--bath-eps", type=float, default=0.0)
    parser.add_argument("--n-tau", type=int, default=10001)
    parser.add_argument("--n-cycles", type=int, default=100000)
    parser.add_argument("--n-warmup-cycles", type=int, default=5000)
    args = parser.parse_args()

    grid = ValentiOrb1Grid(args.mesh, beta=args.beta)
    tau = grid.tau_nodes
    delta_tau = -(args.bath_v**2) * np.exp(-args.bath_eps * tau) / (
        1.0 + np.exp(-args.beta * args.bath_eps)
    )
    delta_vec = grid.gtau_to_vec(delta_tau.astype(complex)).real
    labeler = CtsegLabeler(
        grid=grid,
        n_tau=args.n_tau,
        n_cycles=args.n_cycles,
        n_warmup_cycles=args.n_warmup_cycles,
        projection="dlr",
    )
    g_vec = labeler.solve(delta_vec, U=args.U, mu=args.U / 2.0, beta=args.beta, eps_d=0.0)
    g_tau = grid.vec_to_gtau(g_vec)
    g_iw = grid.gtau_to_giw(g_tau)
    causal = bool(np.all(g_iw[positive_matsubara_mask(grid.iw_nodes)].imag <= 1e-4))
    sane_tau = bool(np.all(g_tau.real <= 1e-8) and np.all(np.isfinite(g_tau.real)))
    summary = {
        "beta": args.beta,
        "U": args.U,
        "mu": args.U / 2.0,
        "bath_v": args.bath_v,
        "bath_eps": args.bath_eps,
        "n_cycles": args.n_cycles,
        "n_warmup_cycles": args.n_warmup_cycles,
        "g_vec_shape": list(g_vec.shape),
        "g_tau_min": float(np.min(g_tau.real)),
        "g_tau_max": float(np.max(g_tau.real)),
        "causal_positive_iw": causal,
        "sane_tau": sane_tau,
        "ctseg_info": labeler.last_info.__dict__ if labeler.last_info is not None else {},
    }
    if mpi.is_master_node():
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True))
        np.savez(out.with_suffix(".npz"), delta_vec=delta_vec, g_vec=g_vec, g_tau=g_tau, g_iw=g_iw)
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
