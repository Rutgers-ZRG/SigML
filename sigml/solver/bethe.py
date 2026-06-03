from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class DMFTResult:
    converged: bool
    n_iter: int
    g_vec: np.ndarray
    delta_vec: np.ndarray


Solver = Callable[[np.ndarray, float, float, float, float], np.ndarray]


def dmft_loop(
    solver: Solver,
    U: float,
    mu: float,
    beta: float,
    t: float,
    grid,
    mix: float,
    tol: float,
    max_iter: int,
    eps_d: float = 0.0,
    initial_delta_vec: np.ndarray | None = None,
) -> DMFTResult:
    """Run Bethe-lattice DMFT self-consistency on Valenti tau-node vectors."""
    if initial_delta_vec is None:
        delta_vec = np.zeros(grid.feature_dim, dtype=float)
    else:
        delta_vec = np.asarray(initial_delta_vec, dtype=float).copy()
        if delta_vec.shape != (grid.feature_dim,):
            raise ValueError(
                f"initial_delta_vec has shape {delta_vec.shape}, expected ({grid.feature_dim},)"
            )
    g_vec = np.zeros(grid.feature_dim, dtype=float)

    for n_iter in range(1, max_iter + 1):
        g_vec = np.asarray(solver(delta_vec, U, mu, beta, eps_d), dtype=float)
        if g_vec.shape != (grid.feature_dim,):
            raise ValueError(f"Solver returned shape {g_vec.shape}, expected ({grid.feature_dim},)")

        delta_new = (t**2) * g_vec
        delta_next = (1.0 - mix) * delta_new + mix * delta_vec
        err = np.linalg.norm(delta_next - delta_vec) / np.sqrt(grid.feature_dim)
        delta_vec = delta_next
        if err < tol:
            return DMFTResult(True, n_iter, g_vec, delta_vec)

    return DMFTResult(False, max_iter, g_vec, delta_vec)


def make_analytic_noninteracting_solver(grid) -> Solver:
    """Build a U=0 impurity solver: G(iw)=1/(iw+mu-eps_d-Delta(iw))."""

    def solve(
        delta_vec: np.ndarray,
        U: float,
        mu: float,
        beta: float,
        eps_d: float = 0.0,
    ) -> np.ndarray:
        delta_tau = grid.vec_to_gtau(delta_vec)
        delta_iw = grid.gtau_to_giw(delta_tau)
        g_iw = 1.0 / (grid.iw_nodes + mu - eps_d - delta_iw)
        g_tau = grid.giw_to_gtau(g_iw)
        return grid.gtau_to_vec(g_tau).real

    return solve
