#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sigml.solver.labeler import CthybT2GLabeler
from sigml.solver.metrics import orbital_occupation, quasiparticle_proxy
from sigml.solver.pydlr_grid import PydlrGrid
from sigml.solver.srvo3_nn_harness import _eval_dlr_iw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the single SrVO3 NN-loop CTHYB refinement solve."
    )
    parser.add_argument("--harness-npz", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-cycles", type=int, default=300_000)
    parser.add_argument("--n-warmup-cycles", type=int, default=8_000)
    parser.add_argument("--length-cycle", type=int, default=120)
    parser.add_argument("--n-tau", type=int, default=10_001)
    parser.add_argument("--n-iw", type=int, default=1_025)
    parser.add_argument("--beta", type=float, default=40.0)
    parser.add_argument("--omega-max", type=float, default=15.0)
    parser.add_argument("--eps", type=float, default=1e-10)
    parser.add_argument("--U", type=float, default=2.0)
    parser.add_argument("--J", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=20260603)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = PydlrGrid(beta=args.beta, lamb=args.beta * args.omega_max, eps=args.eps)
    harness = np.load(args.harness_npz)
    delta = np.asarray(harness["delta"][-1], dtype=np.complex128)
    eps_d = np.asarray(harness["eps_d"], dtype=np.complex128)
    mu = float(np.asarray(harness["mu"][-1]))

    labeler = CthybT2GLabeler(
        grid=grid,
        n_tau=args.n_tau,
        n_iw=args.n_iw,
        n_cycles=args.n_cycles,
        n_warmup_cycles=args.n_warmup_cycles,
        length_cycle=args.length_cycle,
        projection="dlr",
        solve_kwargs={"verbosity": 0, "random_seed": int(args.seed)},
    )
    g_refined = labeler.solve(
        delta_dlr=delta,
        U=args.U,
        J=args.J,
        mu=mu,
        beta=args.beta,
        eps_d=eps_d,
    )
    sigma_iw = sigma_iw_from_delta_g(delta=delta, g=g_refined, eps_d=eps_d, grid=grid)
    occ_per_spin = np.diag(orbital_occupation(g_refined)).real
    occ_spin_summed = 2.0 * occ_per_spin
    z_proxy = np.diag(quasiparticle_proxy(g_refined, grid, args.beta)).real
    info = labeler.last_info

    npz_path = out_dir / "srvo3_nn_plus_one_cthyb_observables.npz"
    np.savez(
        npz_path,
        delta=delta,
        eps_d=eps_d,
        mu=np.asarray(mu, dtype=np.float64),
        g=g_refined,
        sigma_iw=sigma_iw,
        iw=grid.iw_nodes,
        beta=np.asarray(args.beta, dtype=np.float64),
        tau_nodes=grid.tau_nodes,
        occupation_per_spin=occ_per_spin,
        occupation_spin_summed=occ_spin_summed,
        z_proxy=z_proxy,
    )
    summary = {
        "harness_npz": str(Path(args.harness_npz).resolve()),
        "npz": str(npz_path),
        "n_cycles": int(args.n_cycles),
        "n_warmup_cycles": int(args.n_warmup_cycles),
        "length_cycle": int(args.length_cycle),
        "mu": mu,
        "occupation_per_spin": [float(x) for x in occ_per_spin],
        "occupation_spin_summed": [float(x) for x in occ_spin_summed],
        "occupation_total_spin_summed": float(np.sum(occ_spin_summed)),
        "z_proxy_minus_gbeta2": [float(x) for x in z_proxy],
        "cthyb": {
            "average_sign": None if info is None or info.average_sign is None else float(info.average_sign),
            "average_order": None if info is None or info.average_order is None else float(info.average_order),
        },
    }
    (out_dir / "srvo3_nn_plus_one_cthyb_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def sigma_iw_from_delta_g(
    *,
    delta: np.ndarray,
    g: np.ndarray,
    eps_d: np.ndarray,
    grid: PydlrGrid,
) -> np.ndarray:
    delta_coeffs = grid.coeffs_from_gtau(delta)
    g_coeffs = grid.coeffs_from_gtau(g)
    eye = np.eye(delta.shape[0], dtype=np.complex128)
    sigma = np.zeros((grid.iw_nodes.size, delta.shape[0], delta.shape[1]), dtype=np.complex128)
    for idx, iw in enumerate(grid.iw_nodes):
        delta_iw = _eval_dlr_iw(delta_coeffs, complex(iw), grid)
        g_iw = _eval_dlr_iw(g_coeffs, complex(iw), grid)
        sigma[idx] = complex(iw) * eye - eps_d - delta_iw - np.linalg.inv(g_iw)
    return sigma


if __name__ == "__main__":
    main()
