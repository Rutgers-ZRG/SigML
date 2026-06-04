#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from sigml.solver.metrics import g_mse, orbital_occupation
from sigml.solver.pydlr_grid import PydlrGrid
from sigml.solver.srvo3_nn_harness import _eval_dlr_iw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Stage-2 SrVO3 NN-loop and NN+1 CTHYB outputs to the reference."
    )
    parser.add_argument("--reference-npz", required=True)
    parser.add_argument("--reference-summary-json", required=True)
    parser.add_argument("--harness-npz", required=True)
    parser.add_argument("--harness-summary-json", required=True)
    parser.add_argument("--refined-npz", default=None)
    parser.add_argument("--refined-summary-json", default=None)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--beta", type=float, default=40.0)
    parser.add_argument("--omega-max", type=float, default=15.0)
    parser.add_argument("--eps", type=float, default=1e-10)
    parser.add_argument("--z-fit-points", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid = PydlrGrid(beta=args.beta, lamb=args.beta * args.omega_max, eps=args.eps)
    reference = load_reference(args.reference_npz, args.reference_summary_json, grid=grid)
    harness = np.load(args.harness_npz)
    harness_summary = json.loads(Path(args.harness_summary_json).read_text())

    nn_loop = summarize_stage(
        name="nn_loop",
        g=np.asarray(harness["g"][-1], dtype=np.complex128),
        delta=np.asarray(harness["delta"][-1], dtype=np.complex128),
        eps_d=np.asarray(harness["eps_d"], dtype=np.complex128),
        reference=reference,
        grid=grid,
        z_fit_points=args.z_fit_points,
    )
    nn_loop["iterations"] = int(harness_summary["n_iterations"])
    nn_loop["converged"] = bool(harness_summary["iterations"][-1].get("converged", False))
    nn_loop["final_g_delta_mse"] = harness_summary["iterations"][-1].get("g_delta_mse")

    payload: dict[str, Any] = {
        "reference": reference["summary"],
        "nn_loop": nn_loop,
    }
    if args.refined_npz:
        refined = np.load(args.refined_npz)
        refined_summary = (
            json.loads(Path(args.refined_summary_json).read_text())
            if args.refined_summary_json
            else {}
        )
        refined_stage = summarize_stage(
            name="nn_plus_one_cthyb",
            g=np.asarray(refined["g"], dtype=np.complex128),
            delta=np.asarray(refined["delta"], dtype=np.complex128),
            eps_d=np.asarray(refined["eps_d"], dtype=np.complex128),
            reference=reference,
            grid=grid,
            z_fit_points=args.z_fit_points,
        )
        refined_stage["cthyb"] = refined_summary.get("cthyb", {})
        payload["nn_plus_one_cthyb"] = refined_stage

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


def load_reference(reference_npz: str | Path, summary_json: str | Path, *, grid: PydlrGrid) -> dict[str, Any]:
    data = np.load(reference_npz)
    summary = json.loads(Path(summary_json).read_text())
    ref_g_dense = np.mean(np.asarray(data["G_tau"], dtype=np.complex128), axis=0)
    ref_tau = np.asarray(data["tau"], dtype=np.float64)
    ref_g = interpolate_block_tau(ref_g_dense, ref_tau, grid.tau_nodes)
    return {
        "g": ref_g,
        "occupation_spin_summed": np.asarray(data["spin_summed_occ"], dtype=np.float64),
        "z": np.asarray(data["spin_avg_Z"], dtype=np.float64),
        "summary": {
            "occupation_spin_summed": [float(x) for x in data["spin_summed_occ"]],
            "occupation_total": float(np.sum(data["spin_summed_occ"])),
            "Z": [float(x) for x in data["spin_avg_Z"]],
            "G_tau_diag_mid": summary.get("spin_avg_gbeta2"),
            "reference_last_iteration": int(summary.get("reference_last_iteration", -1)),
            "source_h5": summary.get("source_h5"),
        },
    }


def summarize_stage(
    *,
    name: str,
    g: np.ndarray,
    delta: np.ndarray,
    eps_d: np.ndarray,
    reference: dict[str, Any],
    grid: PydlrGrid,
    z_fit_points: int,
) -> dict[str, Any]:
    del name
    occupation = 2.0 * np.diag(orbital_occupation(g)).real
    sigma_iw = sigma_iw_from_delta_g(delta=delta, g=g, eps_d=eps_d, grid=grid)
    z = estimate_z_from_sigma(sigma_iw, grid.iw_nodes, n_fit=z_fit_points)
    g_diag = np.diagonal(g, axis1=0, axis2=1).T
    ref_g_diag = np.diagonal(reference["g"], axis1=0, axis2=1).T
    occ_ref = reference["occupation_spin_summed"]
    z_ref = reference["z"]
    return {
        "G_tau_mse_all": g_mse(g, reference["g"]),
        "G_tau_mse_diag": g_mse(g_diag, ref_g_diag),
        "G_tau_mae_diag": float(np.mean(np.abs(g_diag - ref_g_diag))),
        "occupation_spin_summed": [float(x) for x in occupation],
        "occupation_total": float(np.sum(occupation)),
        "occupation_abs_error": [float(x) for x in np.abs(occupation - occ_ref)],
        "occupation_total_error": float(np.sum(occupation) - np.sum(occ_ref)),
        "Z_estimate": [float(x) for x in z],
        "Z_reference": [float(x) for x in z_ref],
        "Z_abs_error": [float(x) for x in np.abs(z - z_ref)],
    }


def interpolate_block_tau(values: np.ndarray, source_tau: np.ndarray, target_tau: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.complex128)
    out = np.zeros((arr.shape[1], arr.shape[2], target_tau.size), dtype=np.complex128)
    for i in range(arr.shape[1]):
        for j in range(arr.shape[2]):
            real = np.interp(target_tau, source_tau, arr[:, i, j].real)
            imag = np.interp(target_tau, source_tau, arr[:, i, j].imag)
            out[i, j] = real + 1j * imag
    return out


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


def estimate_z_from_sigma(sigma_iw: np.ndarray, iw_nodes: np.ndarray, *, n_fit: int) -> np.ndarray:
    iw = np.asarray(iw_nodes)
    sigma = np.asarray(sigma_iw)
    pos = np.flatnonzero(iw.imag > 0.0)
    selected = pos[: max(2, int(n_fit))]
    omega = iw[selected].imag
    z = []
    for orb in range(sigma.shape[1]):
        y = sigma[selected, orb, orb].imag
        slope, _ = np.polyfit(omega, y, deg=1)
        z.append(1.0 / (1.0 - slope))
    return np.asarray(z, dtype=np.float64)


if __name__ == "__main__":
    main()
