#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sigml.solver.hybridization import sample_t2g_bath
from sigml.solver.labeler import CthybT2GLabeler
from sigml.solver.pydlr_grid import PydlrGrid


@dataclass(frozen=True)
class Srvo3Center:
    name: str
    delta: np.ndarray
    eps_d: np.ndarray
    U: float
    J: float
    beta: float
    mu: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SrVO3-warm beta=40 t2g CTHYB labels.")
    parser.add_argument("--reference-h5", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--beta", type=float, default=40.0)
    parser.add_argument("--omega-max", type=float, default=15.0)
    parser.add_argument("--eps", type=float, default=1e-10)
    parser.add_argument("--n-cycles", type=int, default=300_000)
    parser.add_argument("--n-warmup-cycles", type=int, default=8_000)
    parser.add_argument("--length-cycle", type=int, default=120)
    parser.add_argument("--n-tau-solver", type=int, default=10_001)
    parser.add_argument("--n-iw-solver", type=int, default=1_025)
    parser.add_argument("--neighborhood", type=float, default=0.025)
    parser.add_argument("--projection", choices=("direct", "dlr"), default="direct")
    parser.add_argument("--solver-interface", choices=("delta", "g0"), default="g0")
    parser.add_argument("--verbosity", type=int, default=0)
    parser.add_argument("--converged-only", action="store_true")
    return parser.parse_args()


def _complex_from_h5(raw: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw)
    if arr.shape[-1] == 2:
        return arr[..., 0] + 1j * arr[..., 1]
    return arr.astype(complex)


def _project_uniform_tau(delta_tau: np.ndarray, grid: PydlrGrid) -> np.ndarray:
    uniform_tau = np.linspace(0.0, float(grid.beta), delta_tau.shape[1])
    spin_avg = np.mean(delta_tau, axis=0)
    projected = np.zeros((3, 3, grid.n_tau), dtype=complex)
    for i in range(3):
        for j in range(3):
            projected[i, j] = np.interp(grid.tau_nodes, uniform_tau, spin_avg[:, i, j].real)
            projected[i, j] += 1j * np.interp(grid.tau_nodes, uniform_tau, spin_avg[:, i, j].imag)
    diag = np.zeros_like(projected)
    for orb in range(3):
        diag[orb, orb] = projected[orb, orb]
    return diag


def _effective_atomic_levels(reference_h5: str | Path, iteration: str, beta: float) -> tuple[float, np.ndarray]:
    import h5py
    from triqs_dft_tools.sumk_dft import SumkDFT

    sum_k = SumkDFT(hdf_file=str(reference_h5), beta=beta, n_iw=1025)
    with h5py.File(reference_h5, "r") as h5:
        group = h5[f"DMFT_results/{iteration}"]
        mu = float(group["chemical_potential_post"][()])
        dc = {spin: np.asarray(group[f"DC_pot/0/{spin}"][()]) for spin in ("up", "down")}
    sum_k.set_mu(mu)
    sum_k.dc_imp = [dc]
    eal_sumk = sum_k.eff_atomic_levels()[0]
    eal_solver = sum_k.block_structure.convert_matrix(eal_sumk, space_from="sumk", space_to="solver")
    up = np.asarray(eal_solver["up_0"], dtype=complex)
    down = np.asarray(eal_solver["down_0"], dtype=complex)
    eal = 0.5 * (up + down)
    eal = 0.5 * (eal + eal.conj().T)
    return mu, eal


def load_srvo3_centers(reference_h5: str | Path, grid: PydlrGrid) -> list[Srvo3Center]:
    import h5py

    names = ("it_6", "it_8", "it_10", "last_iter")
    centers: list[Srvo3Center] = []
    with h5py.File(reference_h5, "r") as h5:
        root = h5["DMFT_results"]
        for name in names:
            if name not in root:
                continue
            group = root[name]
            blocks = [
                _complex_from_h5(group[f"Delta_time_0/{spin}/data"][()])
                for spin in ("down_0", "up_0")
            ]
            mu, eal = _effective_atomic_levels(reference_h5, name, grid.beta)
            centers.append(
                Srvo3Center(
                    name=name,
                    delta=_project_uniform_tau(np.stack(blocks, axis=0), grid),
                    eps_d=eal,
                    U=2.0,
                    J=0.65,
                    beta=float(grid.beta),
                    mu=mu,
                )
            )
    if not centers:
        raise ValueError(f"No SrVO3 centers found in {reference_h5}")
    return centers


def warm_sample(center: Srvo3Center, grid: PydlrGrid, rng: np.random.Generator, neighborhood: float) -> Srvo3Center:
    fresh = sample_t2g_bath(grid, rng, mode="broad", alpha=0.08)
    mix = float(neighborhood)
    delta = (1.0 - mix) * center.delta + mix * fresh.delta
    diag_delta = np.zeros_like(delta)
    for orb in range(3):
        diag_delta[orb, orb] = delta[orb, orb]

    u = float(np.clip(center.U + rng.normal(0.0, 0.25 * max(mix, 1e-12)), 1.8, 2.2))
    j = float(np.clip(center.J + rng.normal(0.0, 0.12 * max(mix, 1e-12)), 0.55, 0.75))
    mu = float(center.mu + rng.normal(0.0, 0.25 * max(mix, 1e-12)))
    eps_d = center.eps_d - (mu - center.mu) * np.eye(3, dtype=complex)
    return Srvo3Center(
        name=f"warm_{center.name}",
        delta=diag_delta,
        eps_d=eps_d,
        U=u,
        J=j,
        beta=center.beta,
        mu=mu,
    )


def _info_json(info: Any, sample: Srvo3Center, elapsed: float, provenance: dict[str, Any]) -> str:
    payload = {
        "sample": {
            "center": sample.name,
            "U": sample.U,
            "J": sample.J,
            "beta": sample.beta,
            "mu": sample.mu,
            "eps_d_diag": np.diag(sample.eps_d).real.tolist(),
        },
        "elapsed_seconds": elapsed,
        "provenance": provenance,
    }
    if info is not None:
        payload["cthyb"] = {
            "average_sign": info.average_sign,
            "average_order": info.average_order,
            "orbital_occupations": (
                None if info.orbital_occupations is None else np.asarray(info.orbital_occupations).tolist()
            ),
        }
    return json.dumps(payload, sort_keys=True)


def main() -> None:
    args = parse_args()
    start = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = PydlrGrid(beta=args.beta, lamb=args.beta * args.omega_max, eps=args.eps)
    centers = load_srvo3_centers(args.reference_h5, grid)
    rng = np.random.default_rng(args.seed + args.task_id)
    if args.converged_only:
        sample = [c for c in centers if c.name == "last_iter"][-1]
    else:
        sample = warm_sample(centers[int(rng.integers(0, len(centers)))], grid, rng, args.neighborhood)

    labeler = CthybT2GLabeler(
        grid=grid,
        n_tau=args.n_tau_solver,
        n_iw=args.n_iw_solver,
        n_cycles=args.n_cycles,
        n_warmup_cycles=args.n_warmup_cycles,
        length_cycle=args.length_cycle,
        projection=args.projection,
        delta_interface=args.solver_interface == "delta",
        solve_kwargs={"verbosity": args.verbosity, "random_seed": int(args.seed + args.task_id)},
    )

    solve_start = time.time()
    g = labeler.solve(
        delta_dlr=sample.delta,
        U=sample.U,
        J=sample.J,
        mu=sample.mu,
        beta=sample.beta,
        eps_d=sample.eps_d,
    )
    solve_elapsed = time.time() - solve_start

    rank = int(os.environ.get("PMI_RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0")))
    if rank == 0:
        provenance = {
            "reference_h5": str(args.reference_h5),
            "extraction": "Delta_time_0 plus SumkDFT.eff_atomic_levels(), converted to solver basis; diagonal t2g Delta only",
            "neighborhood": args.neighborhood,
            "seed": args.seed,
            "task_id": args.task_id,
            "projection": args.projection,
            "solver_interface": args.solver_interface,
            "cycles": args.n_cycles,
            "warmup_cycles": args.n_warmup_cycles,
            "length_cycle": args.length_cycle,
            "grid": {"beta": grid.beta, "lamb": grid.lamb, "eps": grid.eps, "rank": grid.rank},
        }
        path = out_dir / f"bootstrap_t2g_b40_{args.task_id:05d}.npz"
        np.savez(
            path,
            delta=np.asarray([sample.delta], dtype=np.complex128),
            g=np.asarray([g], dtype=np.complex128),
            U=np.asarray([sample.U], dtype=np.float64),
            mu=np.asarray([sample.mu], dtype=np.float64),
            beta=np.asarray([sample.beta], dtype=np.float64),
            J=np.asarray([sample.J], dtype=np.float64),
            eps_d=np.asarray([np.diag(sample.eps_d).real], dtype=np.float64),
            source=np.asarray([sample.name]),
            info_json=np.asarray([_info_json(labeler.last_info, sample, solve_elapsed, provenance)]),
            elapsed_seconds=np.asarray([time.time() - start], dtype=np.float64),
        )
        print(f"wrote {path} rows=1 solve_elapsed={solve_elapsed:.3f}s total={time.time() - start:.3f}s")


if __name__ == "__main__":
    main()
