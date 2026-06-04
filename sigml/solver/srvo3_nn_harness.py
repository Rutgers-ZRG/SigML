from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from sigml.solver.metrics import g_mse, orbital_occupation, quasiparticle_proxy
from sigml.solver.nn_solver_client import NNSolverConfig, run_nn_solver

if TYPE_CHECKING:
    from sigml.solver.pydlr_grid import PydlrGrid


SPIN_BLOCKS = ("up_0", "down_0")


@dataclass(frozen=True)
class Srvo3NNSidecarSettings:
    checkpoint_path: Path
    repo_root: Path
    cli_path: Path
    python_command: tuple[str, ...]
    timeout_seconds: float | None = 300.0


@dataclass(frozen=True)
class Srvo3NNIteration:
    iteration: int
    chemical_potential: float
    occupation_diag: list[float]
    occupation_total: float
    quasiparticle_diag: list[float]
    delta_norm: float
    g_norm: float
    sigma_norm: float
    converged: bool = False
    g_delta_mse: float | None = None


@dataclass(frozen=True)
class Srvo3NNHarnessResult:
    h5_path: str
    checkpoint_path: str
    n_iterations: int
    beta: float
    grid_rank: int
    output_npz: str
    iterations: list[Srvo3NNIteration]


def default_base_python_command() -> tuple[str, ...]:
    conda = shutil.which("conda")
    if conda is not None:
        return (conda, "run", "-n", "base", "python")
    return ("conda", "run", "-n", "base", "python")


def make_nn_solver_config(settings: Srvo3NNSidecarSettings) -> NNSolverConfig:
    env = {
        "PYTHONPATH": str(settings.repo_root),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    return NNSolverConfig(
        checkpoint_path=settings.checkpoint_path,
        cli_path=settings.cli_path,
        python_command=settings.python_command,
        env=env,
        timeout_seconds=settings.timeout_seconds,
    )


def write_stub_block_resnet_checkpoint(
    path: str | Path,
    *,
    orbital_dim: int,
    n_tau: int,
    scalar_dim: int = 4,
    hidden_dim: int = 32,
    num_layers: int = 1,
    seed: int = 20260603,
) -> Path:
    """Write a deterministic untrained Phase-10 style checkpoint for harness runs."""

    import torch

    from sigml.solver.net import BlockResNet

    torch.manual_seed(int(seed))
    model = BlockResNet(
        orbital_dim=int(orbital_dim),
        n_tau=int(n_tau),
        scalar_dim=int(scalar_dim),
        hidden_dim=int(hidden_dim),
        num_layers=int(num_layers),
    )
    ckpt = {
        "architecture": "block-resnet",
        "orbital_dim": int(orbital_dim),
        "n_tau": int(n_tau),
        "scalar_dim": int(scalar_dim),
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
        "model_state_dict": model.state_dict(),
        "stub_untrained": True,
        "seed": int(seed),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    return path


def run_srvo3_nn_harness(
    *,
    h5_path: str | Path,
    output_dir: str | Path,
    settings: Srvo3NNSidecarSettings,
    n_iterations: int = 3,
    beta: float = 40.0,
    n_iw: int = 1025,
    omega_max: float = 15.0,
    eps: float = 1e-10,
    mix_sigma: float = 0.5,
    mu_precision: float = 0.01,
    regularization: float = 1e-3,
    convergence_tol: float | None = None,
    min_iterations: int = 2,
) -> Srvo3NNHarnessResult:
    """Run a small SrVO3 SumkDFT loop whose impurity solve is the NN sidecar.

    This is a mechanism harness. It deliberately uses a stub/untrained network
    checkpoint; the output is not a physics validation target.
    """

    from triqs.gf import inverse
    from triqs_dft_tools.sumk_dft import SumkDFT
    from sigml.solver.pydlr_grid import PydlrGrid

    h5_path = Path(h5_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = PydlrGrid(beta=float(beta), lamb=float(beta) * float(omega_max), eps=float(eps))
    sum_k = SumkDFT(hdf_file=str(h5_path), beta=float(beta), n_iw=int(n_iw))
    sigma = _zero_solver_block_gf(sum_k, beta=float(beta), n_iw=int(n_iw))
    config = make_nn_solver_config(settings)

    iterations: list[Srvo3NNIteration] = []
    saved_delta: list[np.ndarray] = []
    saved_g: list[np.ndarray] = []
    saved_sigma: list[np.ndarray] = []
    saved_mu: list[float] = []
    previous_g_dlr: np.ndarray | None = None

    for it in range(1, int(n_iterations) + 1):
        sum_k.put_Sigma([sigma])
        try:
            mu = float(sum_k.calc_mu(precision=float(mu_precision)))
        except TypeError:
            mu = float(sum_k.calc_mu())
        sum_k.set_mu(mu)

        g_loc = sum_k.extract_G_loc()[0]
        g0 = g_loc.copy()
        g0 << inverse(sigma + inverse(g_loc))
        delta_dlr, eps_d = delta_dlr_from_g0(sum_k, g0, grid)

        nn_out = run_nn_solver(
            config,
            delta_dlr=delta_dlr,
            U=2.0,
            mu_over_U=_mu_minus_eps_over_u(mu, eps_d, U=2.0),
            beta=float(beta),
            J=0.65,
            work_dir=output_dir / f"it_{it:03d}_sidecar",
        )
        g_dlr = stabilize_g_dlr(nn_out.g_dlr, grid=grid, eps=float(regularization))
        g_freq = block_gf_from_dlr(g0, g_dlr, grid)
        sigma_new = g0.copy()
        sigma_new << inverse(g0) - inverse(g_freq)
        if it > 1:
            sigma << (1.0 - float(mix_sigma)) * sigma + float(mix_sigma) * sigma_new
        else:
            sigma << sigma_new

        occ = np.diag(orbital_occupation(g_dlr)).real
        z_proxy = np.diag(quasiparticle_proxy(g_dlr, grid, float(beta))).real
        sigma_dense = block_gf_average(sigma)
        g_delta_mse = None
        converged = False
        if previous_g_dlr is not None:
            g_delta_mse = g_mse(g_dlr, previous_g_dlr)
            converged = (
                convergence_tol is not None
                and it >= int(min_iterations)
                and g_delta_mse <= float(convergence_tol)
            )
        iterations.append(
            Srvo3NNIteration(
                iteration=it,
                chemical_potential=mu,
                occupation_diag=[float(x) for x in occ],
                occupation_total=float(np.sum(occ)),
                quasiparticle_diag=[float(x) for x in z_proxy],
                delta_norm=float(np.linalg.norm(delta_dlr)),
                g_norm=float(np.linalg.norm(g_dlr)),
                sigma_norm=float(np.linalg.norm(sigma_dense)),
                converged=bool(converged),
                g_delta_mse=None if g_delta_mse is None else float(g_delta_mse),
            )
        )
        saved_delta.append(np.asarray(delta_dlr, dtype=np.complex128))
        saved_g.append(np.asarray(g_dlr, dtype=np.complex128))
        saved_sigma.append(np.asarray(sigma_dense, dtype=np.complex128))
        saved_mu.append(mu)
        previous_g_dlr = np.asarray(g_dlr, dtype=np.complex128)
        if converged:
            break

    npz_path = output_dir / "srvo3_nn_harness_observables.npz"
    np.savez(
        npz_path,
        delta=np.asarray(saved_delta, dtype=np.complex128),
        g=np.asarray(saved_g, dtype=np.complex128),
        sigma=np.asarray(saved_sigma, dtype=np.complex128),
        mu=np.asarray(saved_mu, dtype=np.float64),
        beta=np.asarray(float(beta), dtype=np.float64),
        tau_nodes=np.asarray(grid.tau_nodes, dtype=np.float64),
        eps_d=np.asarray(eps_d, dtype=np.complex128),
    )
    result = Srvo3NNHarnessResult(
        h5_path=str(h5_path),
        checkpoint_path=str(settings.checkpoint_path),
        n_iterations=int(n_iterations),
        beta=float(beta),
        grid_rank=int(grid.rank),
        output_npz=str(npz_path),
        iterations=iterations,
    )
    (output_dir / "srvo3_nn_harness_summary.json").write_text(
        json.dumps(result_to_json(result), indent=2, sort_keys=True) + "\n"
    )
    return result


def result_to_json(result: Srvo3NNHarnessResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["iterations"] = [asdict(row) for row in result.iterations]
    return payload


def delta_dlr_from_g0(sum_k: Any, g0: Any, grid: PydlrGrid) -> tuple[np.ndarray, np.ndarray]:
    from triqs.gf import iOmega_n, inverse

    eal_sumk = sum_k.eff_atomic_levels()[0]
    eal_solver = sum_k.block_structure.convert_matrix(
        eal_sumk,
        space_from="sumk",
        space_to="solver",
    )
    delta_iw = g0.copy()
    for name, block in delta_iw:
        block << iOmega_n - inverse(g0[name]) - eal_solver[name]

    delta = _dlr_nodes_from_block_iw(delta_iw, grid)
    eps_d = _spin_average_matrix(eal_solver)
    return _hermitian_part(delta), eps_d


def block_gf_from_dlr(template: Any, g_dlr: np.ndarray, grid: PydlrGrid) -> Any:
    coeffs = grid.coeffs_from_gtau(g_dlr)
    out = template.copy()
    for name, block in out:
        for idx, iw in enumerate(block.mesh):
            block.data[idx, :, :] = _eval_dlr_iw(coeffs, complex(iw), grid)
    return out


def block_gf_average(gf: Any, block_names: Sequence[str] = SPIN_BLOCKS) -> np.ndarray:
    available = set(_block_names(gf))
    arrays = [np.asarray(gf[name].data, dtype=complex) for name in block_names if name in available]
    if not arrays:
        arrays = [np.asarray(block.data, dtype=complex) for _, block in gf]
    return np.mean(np.stack(arrays, axis=0), axis=0)


def stabilize_g_dlr(g_dlr: np.ndarray, *, grid: PydlrGrid, eps: float = 1e-3) -> np.ndarray:
    """Keep the stub output invertible enough for Dyson in the mechanism test."""

    g = _hermitian_part(np.asarray(g_dlr, dtype=complex))
    n_tau = g.shape[-1]
    eye = np.eye(g.shape[0], dtype=complex)
    tau_profile = -float(eps) * np.exp(-grid.tau_nodes / max(float(grid.beta), 1.0))
    g = g + eye[:, :, None] * tau_profile.reshape(1, 1, n_tau)
    return _hermitian_part(g)


def _zero_solver_block_gf(sum_k: Any, *, beta: float, n_iw: int) -> Any:
    from triqs.gf import BlockGf, GfImFreq, MeshImFreq

    mesh = sum_k.mesh if hasattr(sum_k, "mesh") else MeshImFreq(beta=float(beta), S="Fermion", n_max=int(n_iw))
    blocks = []
    names = []
    for name, dim in _iter_gf_struct(sum_k.gf_struct_solver[0]):
        names.append(name)
        size = int(dim) if isinstance(dim, (int, np.integer)) else len(dim)
        blocks.append(GfImFreq(mesh=mesh, target_shape=[size, size]))
    sigma = BlockGf(name_list=names, block_list=blocks, make_copies=False)
    for _, block in sigma:
        block.data[:] = 0.0
    return sigma


def _dlr_nodes_from_block_iw(delta_iw: Any, grid: PydlrGrid) -> np.ndarray:
    values = np.zeros((3, 3, grid.n_tau), dtype=complex)
    count = 0
    available = set(_block_names(delta_iw))
    for name in SPIN_BLOCKS:
        if name not in available:
            continue
        block = delta_iw[name]
        mesh = np.asarray([complex(iw) for iw in block.mesh])
        data = np.asarray(block.data, dtype=complex)
        nearest = np.asarray([int(np.argmin(np.abs(mesh - target))) for target in grid.iw_nodes])
        values += grid.giw_to_gtau(np.moveaxis(data[nearest, :, :], 0, -1))
        count += 1
    if count == 0:
        raise ValueError(f"None of the expected SrVO3 spin blocks {SPIN_BLOCKS!r} are present")
    return values / float(count)


def _spin_average_matrix(blocks: dict[str, np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(blocks[name], dtype=complex) for name in SPIN_BLOCKS if name in blocks]
    if not arrays:
        arrays = [np.asarray(value, dtype=complex) for value in blocks.values()]
    return _hermitian_part(np.mean(np.stack(arrays, axis=0), axis=0))


def _mu_minus_eps_over_u(mu: float, eps_d: np.ndarray, *, U: float) -> float:
    eps = np.asarray(eps_d, dtype=complex)
    if eps.ndim == 0:
        eps_scalar = float(eps.real)
    else:
        eps_scalar = float(np.trace(eps).real / eps.shape[0])
    return (float(mu) - eps_scalar) / float(U)


def _block_names(gf: Any) -> list[str]:
    if hasattr(gf, "indices"):
        return [str(name) for name in gf.indices]
    return [str(name) for name, _ in gf]


def _iter_gf_struct(gf_struct: Any):
    if hasattr(gf_struct, "items"):
        return gf_struct.items()
    return iter(gf_struct)


def _hermitian_part(block: np.ndarray) -> np.ndarray:
    arr = np.asarray(block, dtype=complex)
    return 0.5 * (arr + np.swapaxes(arr.conj(), 0, 1))


def _eval_dlr_iw(coeffs: np.ndarray, z: complex, grid: PydlrGrid) -> np.ndarray:
    kernel = 1.0 / (complex(z) - grid.real_frequency_nodes)
    return np.tensordot(kernel, coeffs, axes=(0, -1))
