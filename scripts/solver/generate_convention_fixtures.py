#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import triqs.utility.mpi as mpi
from h5 import HDFArchive

from sigml.solver.labeler import CtsegLabeler
from sigml.solver.valenti_grid import ValentiOrb1Grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate convention-locking solver fixtures.")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--mldmft-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--beta", type=float, default=70.0)
    parser.add_argument("--U", type=float, default=2.0)
    parser.add_argument("--bath-v", type=float, default=0.10)
    parser.add_argument("--bath-eps", type=float, default=0.0)
    parser.add_argument("--n-tau", type=int, default=10001)
    parser.add_argument("--n-cycles", type=int, default=100000)
    parser.add_argument("--n-warmup-cycles", type=int, default=5000)
    return parser.parse_args()


def known_delta_vec(grid: ValentiOrb1Grid, beta: float, bath_v: float, bath_eps: float) -> np.ndarray:
    tau = grid.tau_nodes
    delta_tau = -(bath_v**2) * np.exp(-bath_eps * tau) / (1.0 + np.exp(-beta * bath_eps))
    return grid.gtau_to_vec(delta_tau.astype(complex)).real


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.mldmft_root).resolve()))
    from mldmft.utils import DLR_to_NNinput, NNoutput_to_DLR
    from triqs.gf import make_gf_dlr_imfreq

    grid = ValentiOrb1Grid(args.mesh, beta=args.beta)
    delta_vec = known_delta_vec(grid, args.beta, args.bath_v, args.bath_eps)
    with HDFArchive(args.mesh, "r") as h5:
        dlr_tau_mesh = h5["mesh_dlr_imtime"]

    delta_dlr = NNoutput_to_DLR(delta_vec[None, :], dlr_tau_mesh, sample_index=0)
    nn_input = DLR_to_NNinput(delta_dlr, dlr_tau_mesh, args.U, args.U / 2.0, args.beta)
    delta_iw_dlr = make_gf_dlr_imfreq(delta_dlr)
    delta_iw = np.asarray([delta_iw_dlr["up"][iw][0, 0] for iw in delta_iw_dlr["up"].mesh], dtype=complex)

    labeler = CtsegLabeler(
        grid=grid,
        n_tau=args.n_tau,
        n_cycles=args.n_cycles,
        n_warmup_cycles=args.n_warmup_cycles,
        projection="dlr",
    )
    g_vec = labeler.solve(delta_vec, U=args.U, mu=args.U / 2.0, beta=args.beta, eps_d=0.0)
    raw_tau_mesh = np.asarray([float(tau) for tau in labeler._last_solver_g_tau.mesh], dtype=float)
    raw_g_tau = np.asarray(
        [labeler._last_solver_g_tau(float(tau))[0, 0] for tau in raw_tau_mesh],
        dtype=complex,
    )

    if mpi.is_master_node():
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(
            out / "valenti_reference_delta_beta70.npz",
            beta=np.asarray(args.beta),
            U=np.asarray(args.U),
            mu=np.asarray(args.U / 2.0),
            delta_vec=delta_vec,
            valenti_nn_input=np.asarray(nn_input[0], dtype=float),
            valenti_delta_iw=delta_iw,
        )
        np.savez(
            out / "ctseg_projection_near_atomic_beta70.npz",
            beta=np.asarray(args.beta),
            U=np.asarray(args.U),
            mu=np.asarray(args.U / 2.0),
            bath_v=np.asarray(args.bath_v),
            bath_eps=np.asarray(args.bath_eps),
            n_cycles=np.asarray(args.n_cycles),
            n_warmup_cycles=np.asarray(args.n_warmup_cycles),
            delta_vec=delta_vec,
            raw_tau_mesh=raw_tau_mesh,
            raw_g_tau=raw_g_tau,
            direct_g_vec=labeler.last_direct_g_vec,
            dlr_g_vec=g_vec,
            ctseg_info=np.asarray(str(labeler.last_info.__dict__ if labeler.last_info else {})),
        )
        print(f"wrote fixtures to {out}")


if __name__ == "__main__":
    main()
