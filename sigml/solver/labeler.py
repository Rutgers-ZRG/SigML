from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from sigml.solver.pydlr_grid import PydlrGrid
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


@dataclass(frozen=True)
class CthybSolveInfo:
    """Small summary of the most recent CTHYB solve."""

    average_sign: float | None
    average_order: float | None
    orbital_occupations: np.ndarray | None


class CthybT2GLabeler:
    """TRIQS/CTHYB labeler for three-orbital t2g Kanamori baths.

    The interaction is the rotationally-invariant Hubbard-Kanamori form:
    ``Uprime = U - 2J`` with same-spin ``Uprime - J`` terms, plus the
    spin-flip and pair-hopping terms provided by ``h_int_kanamori``.
    Imports are lazy so local tests can skip cleanly outside the TRIQS/CTHYB
    environment.
    """

    spin_blocks = ("up", "down")
    orbital_dim = 3
    default_beta = 40.0
    default_omega_max = 15.0
    default_eps = 1e-10

    def __init__(
        self,
        grid: PydlrGrid | ValentiOrb1Grid | None = None,
        mesh_path: str | Path = DEFAULT_MESH_PATH,
        beta: float = default_beta,
        omega_max: float = default_omega_max,
        eps: float = default_eps,
        n_tau: int = 10_001,
        n_iw: int = 1_025,
        n_cycles: int = 100_000,
        n_warmup_cycles: int = 5_000,
        length_cycle: int = 50,
        projection: str = "dlr",
        delta_interface: bool = True,
        solve_kwargs: dict[str, Any] | None = None,
    ):
        del mesh_path
        self.grid = (
            grid
            if grid is not None
            else PydlrGrid(beta=float(beta), lamb=float(beta) * float(omega_max), eps=float(eps))
        )
        self.n_tau = int(n_tau)
        self.n_iw = int(n_iw)
        self.n_cycles = int(n_cycles)
        self.n_warmup_cycles = int(n_warmup_cycles)
        self.length_cycle = int(length_cycle)
        self.projection = projection
        self.delta_interface = bool(delta_interface)
        self.solve_kwargs = dict(solve_kwargs or {})
        self.last_info: CthybSolveInfo | None = None
        self.last_direct_g_dlr: np.ndarray | None = None
        self.last_dlr_g_dlr: np.ndarray | None = None
        self._last_solver_g_tau: Any | None = None

    def solve(
        self,
        delta_dlr: np.ndarray,
        U: float,
        J: float,
        mu: float,
        beta: float,
        eps_d: np.ndarray | None = None,
    ) -> np.ndarray:
        Solver, c_op, c_dag_op, h_int_kanamori, U_matrix_kanamori = self._import_cthyb()

        delta_tau = self._validate_block_tau("delta_dlr", delta_dlr)
        eps = self._validate_eps_d(eps_d)
        if not np.isclose(float(beta), self.grid.beta):
            raise ValueError(
                f"CTHYB solve beta={float(beta)} must match the DLR grid beta={self.grid.beta}"
            )

        solver = Solver(
            beta=float(beta),
            gf_struct=[(spin, self.orbital_dim) for spin in self.spin_blocks],
            n_tau=self.n_tau,
            n_iw=self.n_iw,
            delta_interface=self.delta_interface,
        )
        if self.delta_interface:
            self._set_delta_tau(solver, delta_tau=delta_tau)
        else:
            self._set_g0_iw(solver, delta_tau=delta_tau, eps_d=eps)

        u_same, u_opp = U_matrix_kanamori(self.orbital_dim, float(U), float(J))
        h_int = h_int_kanamori(
            list(self.spin_blocks),
            self.orbital_dim,
            u_same,
            u_opp,
            float(J),
            off_diag=True,
        )
        h_loc0 = self._local_one_body_hamiltonian(c_op, c_dag_op, eps_d=eps, mu=float(mu))

        solve_kwargs = {
            "h_int": h_int,
            "h_loc0": h_loc0,
            "n_cycles": self.n_cycles,
            "n_warmup_cycles": self.n_warmup_cycles,
            "length_cycle": self.length_cycle,
            "measure_pert_order": True,
        }
        if self.delta_interface:
            solve_kwargs["h_loc0"] = h_loc0
        solve_kwargs["measure_G_tau"] = True
        solve_kwargs.update(self.solve_kwargs)
        solver.solve(**solve_kwargs)

        self._last_solver_g_tau = solver.G_tau
        g_dlr = self._project_g_tau_to_block_dlr(solver.G_tau)
        self.last_info = self._extract_info(solver, g_dlr)
        return np.asarray(g_dlr, dtype=np.complex128)

    @staticmethod
    def _import_cthyb():
        try:
            from triqs_cthyb import Solver
            from triqs.operators import c, c_dag
            from triqs.operators.util import U_matrix_kanamori, h_int_kanamori
        except ImportError as exc:
            raise ImportError(
                "CthybT2GLabeler requires the TRIQS/triqs_cthyb environment. "
                "On Amarel: ssh amarel3; source "
                "/home/lz432/miniconda3/etc/profile.d/conda.sh; "
                "conda activate solid_dmft."
            ) from exc
        return Solver, c, c_dag, h_int_kanamori, U_matrix_kanamori

    def _set_g0_iw(self, solver: Any, *, delta_tau: np.ndarray, eps_d: np.ndarray) -> None:
        coeffs = self.grid.coeffs_from_gtau(delta_tau)
        identity = np.eye(self.orbital_dim, dtype=complex)
        for spin in self.spin_blocks:
            for iw in solver.G0_iw[spin].mesh:
                z = complex(iw)
                delta_iw = self._eval_dlr_iw(coeffs, z)
                inverse_g0 = z * identity - eps_d - delta_iw
                solver.G0_iw[spin][iw] = np.linalg.inv(inverse_g0)

    def _eval_dlr_iw(self, coeffs: np.ndarray, z: complex) -> np.ndarray:
        kernel = 1.0 / (complex(z) - self.grid.real_frequency_nodes)
        return np.tensordot(kernel, coeffs, axes=(0, -1))

    def _set_delta_tau(self, solver: Any, *, delta_tau: np.ndarray) -> None:
        for spin in self.spin_blocks:
            for idx, tau in enumerate(solver.Delta_tau[spin].mesh):
                solver.Delta_tau[spin].data[idx, :, :] = self.grid.eval_at_tau(delta_tau, float(tau))

    def _local_one_body_hamiltonian(
        self,
        c_op: Any,
        c_dag_op: Any,
        *,
        eps_d: np.ndarray,
        mu: float,
    ) -> Any:
        h_loc0 = 0
        del mu
        # In solid_dmft's delta_interface path this is SumkDFT.eff_atomic_levels():
        # the one-body impurity Hamiltonian already shifted by mu and DC.
        one_body = np.asarray(eps_d, dtype=float)
        for spin in self.spin_blocks:
            for i in range(self.orbital_dim):
                for j in range(self.orbital_dim):
                    value = one_body[i, j]
                    if abs(value) > 0.0:
                        h_loc0 += value * c_dag_op(spin, i) * c_op(spin, j)
        return h_loc0

    def _project_g_tau_to_block_dlr(self, solver_g_tau: Any) -> np.ndarray:
        direct = self._direct_project_g_tau(solver_g_tau)
        self.last_direct_g_dlr = direct
        self.last_dlr_g_dlr = None
        if self.projection == "direct":
            return direct

        coeffs = self.grid.coeffs_from_gtau(direct)
        projected = self.grid.gtau_from_coeffs(coeffs)
        projected = self._hermitian_part(projected)
        self.last_dlr_g_dlr = projected
        return projected

    def _direct_project_g_tau(self, solver_g_tau: Any) -> np.ndarray:
        values = np.zeros((self.orbital_dim, self.orbital_dim, self.grid.n_tau), dtype=complex)
        for spin in self.spin_blocks:
            block = solver_g_tau[spin]
            mesh = np.asarray([float(tau) for tau in block.mesh], dtype=float)
            data = np.asarray(block.data, dtype=complex)
            for i in range(self.orbital_dim):
                for j in range(self.orbital_dim):
                    real = np.interp(self.grid.tau_nodes, mesh, data[:, i, j].real)
                    imag = np.interp(self.grid.tau_nodes, mesh, data[:, i, j].imag)
                    values[i, j] += real + 1j * imag
        values /= float(len(self.spin_blocks))
        return self._hermitian_part(values)

    def _extract_info(self, solver: Any, g_dlr: np.ndarray) -> CthybSolveInfo:
        occupations = -np.diagonal(g_dlr[:, :, -1], axis1=0, axis2=1).real
        return CthybSolveInfo(
            average_sign=self._result_value(solver, "average_sign"),
            average_order=self._result_value(solver, "average_order"),
            orbital_occupations=np.asarray(occupations, dtype=np.float64),
        )

    def _validate_block_tau(self, name: str, block: np.ndarray) -> np.ndarray:
        arr = np.asarray(block, dtype=complex)
        expected = (self.orbital_dim, self.orbital_dim, self.grid.n_tau)
        if arr.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {arr.shape}")
        if not np.allclose(arr, np.swapaxes(arr.conj(), 0, 1), atol=1e-8, rtol=1e-8):
            raise ValueError(f"{name} must be Hermitian over orbital block axes")
        return self._hermitian_part(arr)

    def _validate_eps_d(self, eps_d: np.ndarray | None) -> np.ndarray:
        if eps_d is None:
            return np.zeros((self.orbital_dim, self.orbital_dim), dtype=float)
        eps = np.asarray(eps_d, dtype=complex)
        expected = (self.orbital_dim, self.orbital_dim)
        if eps.shape != expected:
            raise ValueError(f"eps_d must have shape {expected}, got {eps.shape}")
        if not np.allclose(eps, eps.conj().T, atol=1e-8, rtol=1e-8):
            raise ValueError("eps_d must be Hermitian")
        if np.max(np.abs(eps.imag)) > 1e-10:
            raise ValueError("eps_d must be real-valued for the current TRIQS/CTHYB h_loc0 interface")
        return (0.5 * (eps + eps.conj().T)).real

    @staticmethod
    def _hermitian_part(blocks: np.ndarray) -> np.ndarray:
        arr = np.asarray(blocks, dtype=complex)
        return 0.5 * (arr + np.swapaxes(arr.conj(), 0, 1))

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
        self.last_direct_g_vec: np.ndarray | None = None
        self.last_dlr_g_vec: np.ndarray | None = None
        self._last_solver_g_tau: Any | None = None

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

        self._last_solver_g_tau = solver.results.G_tau["up"]
        g_vec = self._project_g_tau_to_valenti_vec(self._last_solver_g_tau)
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
        self.last_direct_g_vec = self._direct_project_g_tau(g_tau_up)
        self.last_dlr_g_vec = None
        if self.projection == "direct":
            return self.last_direct_g_vec

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
            self.last_dlr_g_vec = self.grid.gtau_to_vec(np.asarray(values, dtype=complex)).real
            return self.last_dlr_g_vec
        except Exception:
            if self.projection == "dlr":
                raise
            return self.last_direct_g_vec

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
