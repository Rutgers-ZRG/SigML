from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import numpy as np

from sigml.solver.bethe import dmft_loop
from sigml.solver.dyson import positive_matsubara_mask, sigma_from_g
from sigml.solver.labeler import CtsegLabeler
from sigml.solver.valenti_grid import ValentiOrb1Grid


def physical_seed(grid: ValentiOrb1Grid, t: float, scale: float = 0.25) -> np.ndarray:
    gtau = -scale * np.ones_like(grid.tau_nodes, dtype=complex)
    return (t**2) * grid.gtau_to_vec(gtau).real


def finite_vec(vec: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(np.asarray(vec))))


def metric_summary(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = np.asarray(a) - np.asarray(b)
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rms": float(np.sqrt(np.mean(np.abs(diff) ** 2))),
    }


def gtau(grid: ValentiOrb1Grid, vec: np.ndarray) -> np.ndarray:
    return grid.vec_to_gtau(vec)


def giw(grid: ValentiOrb1Grid, vec: np.ndarray) -> np.ndarray:
    return grid.gtau_to_giw(gtau(grid, vec))


def sigma_iw(grid: ValentiOrb1Grid, g_vec: np.ndarray, delta_vec: np.ndarray, U: float) -> np.ndarray:
    return sigma_from_g(
        giw(grid, g_vec),
        giw(grid, delta_vec),
        mu=U / 2.0,
        eps_d=0.0,
        iw=grid.iw_nodes,
    )


def low_freq(grid: ValentiOrb1Grid, arr: np.ndarray, n_low: int) -> np.ndarray:
    pos_idx = np.flatnonzero(positive_matsubara_mask(grid.iw_nodes))
    order = np.argsort(np.abs(grid.iw_nodes[pos_idx].imag))
    return np.asarray(arr)[pos_idx[order[:n_low]]]


def compare_state(
    grid: ValentiOrb1Grid,
    left: dict,
    right: dict,
    U: float,
    n_low: int,
) -> dict:
    return {
        "G_tau_vec": metric_summary(left["g_vec"], right["g_vec"]),
        "Delta_tau_vec": metric_summary(left["solve_delta_vec"], right["solve_delta_vec"]),
        "Delta_iw_low": metric_summary(
            low_freq(grid, giw(grid, left["solve_delta_vec"]), n_low),
            low_freq(grid, giw(grid, right["solve_delta_vec"]), n_low),
        ),
        "Sigma_iw_low": metric_summary(
            low_freq(grid, sigma_iw(grid, left["g_vec"], left["solve_delta_vec"], U), n_low),
            low_freq(grid, sigma_iw(grid, right["g_vec"], right["solve_delta_vec"], U), n_low),
        ),
    }


def ctseg_dmft_loop(
    labeler: CtsegLabeler,
    grid: ValentiOrb1Grid,
    U: float,
    beta: float,
    t: float,
    mix: float,
    tol: float,
    max_iter: int,
    initial_delta_vec: np.ndarray,
) -> dict:
    mu = U / 2.0
    delta_vec = np.asarray(initial_delta_vec, dtype=float).copy()
    last_g = np.zeros(grid.feature_dim)
    solve_delta_vec = delta_vec.copy()
    errors: list[float] = []
    projection_errors: list[dict[str, float | None]] = []
    infos: list[dict] = []
    for n_iter in range(1, max_iter + 1):
        solve_delta_vec = delta_vec.copy()
        last_g = labeler.solve(delta_vec, U=U, mu=mu, beta=beta, eps_d=0.0)
        if labeler.last_direct_g_vec is not None and labeler.last_dlr_g_vec is not None:
            projection_errors.append(metric_summary(labeler.last_direct_g_vec, labeler.last_dlr_g_vec))
        else:
            projection_errors.append({"max_abs": None, "mean_abs": None, "rms": None})
        infos.append(labeler.last_info.__dict__ if labeler.last_info is not None else {})
        delta_new = (t**2) * last_g
        delta_next = (1.0 - mix) * delta_new + mix * delta_vec
        err = float(np.linalg.norm(delta_next - delta_vec) / np.sqrt(grid.feature_dim))
        errors.append(err)
        delta_vec = delta_next
        if err < tol:
            break
    return {
        "converged": bool(errors[-1] < tol) if errors else False,
        "n_iter": int(n_iter),
        "errors": errors,
        "projection_errors": projection_errors,
        "info": infos,
        "g_vec": last_g,
        "delta_vec": delta_vec,
        "solve_delta_vec": solve_delta_vec,
    }


def nn_dmft_state(
    solver: Callable,
    grid: ValentiOrb1Grid,
    U: float,
    beta: float,
    t: float,
    mix: float,
    tol: float,
    max_iter: int,
    initial_delta_vec: np.ndarray,
) -> dict:
    res = dmft_loop(
        solver,
        U=U,
        mu=U / 2.0,
        beta=beta,
        t=t,
        grid=grid,
        mix=mix,
        tol=tol,
        max_iter=max_iter,
        initial_delta_vec=initial_delta_vec,
    )
    return {
        "converged": bool(res.converged),
        "n_iter": int(res.n_iter),
        "g_vec": res.g_vec,
        "delta_vec": res.delta_vec,
        "solve_delta_vec": res.delta_vec,
    }


def write_json(path: str | Path, payload: dict) -> None:
    rank = int(os.environ.get("PMI_RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0")))
    if rank != 0:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
