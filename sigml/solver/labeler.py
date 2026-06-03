from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from sigml.solver.valenti_grid import DEFAULT_MESH_PATH, ValentiOrb1Grid

if TYPE_CHECKING:
    from sigml.solver.oracle import Orb1Oracle


class OracleLabeler:
    """Local oracle-distillation labeler using Valenti's released orb1 model."""

    def __init__(self, oracle: "Orb1Oracle | None" = None):
        if oracle is None:
            from sigml.solver.oracle import Orb1Oracle

            oracle = Orb1Oracle()
        self.oracle = oracle
        self.grid = self.oracle.grid

    def solve(
        self,
        delta_vec: np.ndarray,
        U: float,
        mu: float,
        beta: float,
        eps_d: float = 0.0,
    ) -> np.ndarray:
        return self.oracle.solve(delta_vec, U=U, mu=mu, beta=beta, eps_d=eps_d)


@dataclass(frozen=True)
class CtsegSolveInfo:
    """Small summary of the most recent CTSEG solve."""

    average_sign: float | None
    density_up: float | None
    density_down: float | None
    perturbation_order_delta: float | None


class CtsegLabeler:
    """TRIQS/CTSEG labeler for the one-orbital SigML PoC.

    The verified TRIQS/CTSEG wiring, Slurm template, and the important
    near-atomic bath note are documented in:
    /Users/li/dev/RA/dmft/docs/amarel3-triqs-setup.md

    CTSEG rejects the strict Delta=0 atomic limit, so validation should use a
    real/near-atomic bath. Imports are intentionally lazy so local Phase A/B
    tests can run without TRIQS installed.
    """

    def __init__(
        self,
        grid: ValentiOrb1Grid | None = None,
        mesh_path: str | Path = DEFAULT_MESH_PATH,
        n_tau: int = 10001,
        n_cycles: int = 100_000,
        n_warmup_cycles: int = 5_000,
        length_cycle: int = 50,
        projection: str = "dlr",
        solve_kwargs: dict[str, Any] | None = None,
    ):
        self.grid = grid if grid is not None else ValentiOrb1Grid(mesh_path, beta=70.0)
        self.n_tau = int(n_tau)
        self.n_cycles = int(n_cycles)
        self.n_warmup_cycles = int(n_warmup_cycles)
        self.length_cycle = int(length_cycle)
        self.projection = projection
        self.solve_kwargs = dict(solve_kwargs or {})
        self.last_info: CtsegSolveInfo | None = None

    def solve(
        self,
        delta_vec: np.ndarray,
        U: float,
        mu: float,
        beta: float,
        eps_d: float = 0.0,
    ) -> np.ndarray:
        Solver, n_op = self._import_ctseg()

        delta = np.asarray(delta_vec, dtype=float)
        if delta.shape != (self.grid.feature_dim,):
            raise ValueError(f"Expected delta_vec shape ({self.grid.feature_dim},), got {delta.shape}")

        delta_gtau = self.grid.vec_to_gtau(delta)
        solver = Solver(
            beta=float(beta),
            gf_struct=[("up", 1), ("down", 1)],
            n_tau=self.n_tau,
        )

        delta_uniform = np.array(
            [self.grid.eval_at_tau(delta_gtau, float(tau)) for tau in solver.Delta_tau["up"].mesh],
            dtype=complex,
        )
        for block in ("up", "down"):
            solver.Delta_tau[block].data[:, 0, 0] = delta_uniform
        solver.Jperp_tau.data[:] = 0.0

        h_int = float(U) * n_op("up", 0) * n_op("down", 0)
        h_loc0 = -float(mu - eps_d) * (n_op("up", 0) + n_op("down", 0))

        solve_kwargs = {
            "h_int": h_int,
            "h_loc0": h_loc0,
            "n_cycles": self.n_cycles,
            "n_warmup_cycles": self.n_warmup_cycles,
            "length_cycle": self.length_cycle,
            "measure_G_tau": True,
            "measure_densities": True,
            "measure_average_sign": True,
            "measure_pert_order": True,
        }
        solve_kwargs.update(self.solve_kwargs)
        solver.solve(**solve_kwargs)

        g_vec = self._project_g_tau_to_valenti_vec(solver.results.G_tau["up"])
        self.last_info = self._extract_info(solver)
        return np.asarray(g_vec, dtype=np.float64)

    @staticmethod
    def _import_ctseg():
        try:
            from triqs_ctseg import Solver
            from triqs.operators import n
        except ImportError as exc:
            raise ImportError(
                "CtsegLabeler requires the TRIQS/triqs_ctseg environment. "
                "On Amarel: ssh amarel3; source "
                "/home/lz432/miniconda3/etc/profile.d/conda.sh; "
                "conda activate triqs_ctseg."
            ) from exc
        return Solver, n

    def _project_g_tau_to_valenti_vec(self, g_tau_up: Any) -> np.ndarray:
        if self.projection == "direct":
            return self._direct_project_g_tau(g_tau_up)

        try:
            from h5 import HDFArchive
            from triqs.gf import BlockGf, make_gf_dlr, make_gf_dlr_imtime

            with HDFArchive(self.grid.mesh_path, "r") as h5:
                dlr_tau_mesh = h5["mesh_dlr_imtime"]
            g_tau_dlr = BlockGf(mesh=dlr_tau_mesh, gf_struct=[("up", 1), ("down", 1)])
            for tau in g_tau_dlr.mesh:
                value = g_tau_up(float(tau))[0, 0]
                g_tau_dlr["up"][tau][0, 0] = value
                g_tau_dlr["down"][tau][0, 0] = value
            g_dlr = make_gf_dlr(g_tau_dlr)
            g_tau_nodes = make_gf_dlr_imtime(g_dlr)
            values = [g_tau_nodes["up"][tau][0, 0] for tau in g_tau_nodes["up"].mesh]
            return self.grid.gtau_to_vec(np.asarray(values, dtype=complex)).real
        except Exception:
            if self.projection == "dlr":
                raise
            return self._direct_project_g_tau(g_tau_up)

    def _direct_project_g_tau(self, g_tau_up: Any) -> np.ndarray:
        values = np.array([g_tau_up(float(tau))[0, 0] for tau in self.grid.tau_nodes], dtype=complex)
        return self.grid.gtau_to_vec(values).real

    @staticmethod
    def _result_value(obj: Any, *names: str) -> float | None:
        for name in names:
            if hasattr(obj, name):
                value = getattr(obj, name)
                try:
                    return float(value)
                except (TypeError, ValueError):
                    try:
                        return float(value[()])
                    except Exception:
                        return None
        return None

    def _extract_info(self, solver: Any) -> CtsegSolveInfo:
        results = solver.results
        density_up = density_down = None
        if hasattr(results, "densities"):
            try:
                density_up = float(results.densities["up"][0, 0])
                density_down = float(results.densities["down"][0, 0])
            except Exception:
                density_up = density_down = None
        return CtsegSolveInfo(
            average_sign=self._result_value(results, "average_sign"),
            density_up=density_up,
            density_down=density_down,
            perturbation_order_delta=self._result_value(
                results, "perturbation_order_Delta", "pert_order_Delta", "pert_order"
            ),
        )
