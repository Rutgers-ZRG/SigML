from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
SCHEMA_NAME = "sigml_block_dlr_sidecar_v1"


@dataclass(frozen=True)
class SolverInput:
    """Validated torch sidecar input.

    NPZ input schema:
      schema_version: int scalar, currently 1.
      schema_name: string scalar, "sigml_block_dlr_sidecar_v1".
      delta_dlr: complex64/complex128 array, shape (M, M, N_tau).
        Hermiticity convention: delta_dlr[j, i, t] == conj(delta_dlr[i, j, t])
        for every tau node t; diagonal entries are real up to numerical noise.
      U: float scalar.
      mu_over_U: float scalar. This is the NN scalar mu / U for the current bath.
      beta: float scalar.
      J: float scalar.
      orbital_dim: int scalar equal to M.
      n_tau: int scalar equal to N_tau.

    CLI feature layout:
      concat(real/imag interleaved flatten(delta_dlr), [U, mu_over_U, beta, J])
      as float32. The block flatten order is C-order over (M, M, N_tau), with
      the last axis of the intermediate representation holding [real, imag].
    """

    delta_dlr: np.ndarray
    U: float
    mu_over_U: float
    beta: float
    J: float

    @property
    def orbital_dim(self) -> int:
        return int(self.delta_dlr.shape[0])

    @property
    def n_tau(self) -> int:
        return int(self.delta_dlr.shape[2])


@dataclass(frozen=True)
class SolverOutput:
    """Validated torch sidecar output.

    NPZ output schema:
      schema_version: int scalar, currently 1.
      schema_name: string scalar, "sigml_block_dlr_sidecar_v1".
      g_dlr: complex64/complex128 array, shape (M, M, N_tau), using the same
        Hermiticity convention as delta_dlr.
      orbital_dim: int scalar equal to M.
      n_tau: int scalar equal to N_tau.
    """

    g_dlr: np.ndarray

    @property
    def orbital_dim(self) -> int:
        return int(self.g_dlr.shape[0])

    @property
    def n_tau(self) -> int:
        return int(self.g_dlr.shape[2])


def flatten_block_features(block: np.ndarray) -> np.ndarray:
    block = _validate_block("block", block)
    paired = np.stack((block.real, block.imag), axis=-1)
    return np.reshape(paired, (-1,)).astype(np.float32)


def unflatten_block_features(features: np.ndarray, *, orbital_dim: int, n_tau: int) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    expected = orbital_dim * orbital_dim * n_tau * 2
    if features.shape != (expected,):
        raise ValueError(f"features must have shape ({expected},), got {features.shape}")
    paired = np.reshape(features, (orbital_dim, orbital_dim, n_tau, 2))
    return paired[..., 0].astype(np.float32) + 1j * paired[..., 1].astype(np.float32)


def solver_input_features(path: str | Path) -> tuple[np.ndarray, SolverInput]:
    solver_input = read_solver_input(path)
    scalars = np.array(
        [solver_input.U, solver_input.mu_over_U, solver_input.beta, solver_input.J],
        dtype=np.float32,
    )
    return np.concatenate((flatten_block_features(solver_input.delta_dlr), scalars)), solver_input


def read_solver_input(path: str | Path) -> SolverInput:
    data = np.load(path)
    _validate_schema_header(data)
    delta = _validate_block("delta_dlr", _require_key(data, "delta_dlr"))
    orbital_dim = _read_int(data, "orbital_dim")
    n_tau = _read_int(data, "n_tau")
    if delta.shape != (orbital_dim, orbital_dim, n_tau):
        raise ValueError(
            f"delta_dlr shape {delta.shape} does not match orbital_dim={orbital_dim}, n_tau={n_tau}"
        )
    return SolverInput(
        delta_dlr=delta,
        U=_read_float(data, "U"),
        mu_over_U=_read_float(data, "mu_over_U"),
        beta=_read_float(data, "beta"),
        J=_read_float(data, "J"),
    )


def write_solver_input(
    path: str | Path,
    *,
    delta_dlr: np.ndarray,
    U: float,
    mu_over_U: float,
    beta: float,
    J: float,
) -> None:
    delta = _validate_block("delta_dlr", delta_dlr)
    np.savez(
        path,
        schema_version=np.array(SCHEMA_VERSION, dtype=np.int32),
        schema_name=np.array(SCHEMA_NAME),
        delta_dlr=delta,
        U=np.array(U, dtype=np.float32),
        mu_over_U=np.array(mu_over_U, dtype=np.float32),
        beta=np.array(beta, dtype=np.float32),
        J=np.array(J, dtype=np.float32),
        orbital_dim=np.array(delta.shape[0], dtype=np.int32),
        n_tau=np.array(delta.shape[2], dtype=np.int32),
    )


def read_solver_output(path: str | Path) -> SolverOutput:
    data = np.load(path)
    _validate_schema_header(data)
    g = _validate_block("g_dlr", _require_key(data, "g_dlr"))
    orbital_dim = _read_int(data, "orbital_dim")
    n_tau = _read_int(data, "n_tau")
    if g.shape != (orbital_dim, orbital_dim, n_tau):
        raise ValueError(
            f"g_dlr shape {g.shape} does not match orbital_dim={orbital_dim}, n_tau={n_tau}"
        )
    return SolverOutput(g_dlr=g)


def write_solver_output(path: str | Path, *, g_dlr: np.ndarray) -> None:
    g = _validate_block("g_dlr", g_dlr)
    np.savez(
        path,
        schema_version=np.array(SCHEMA_VERSION, dtype=np.int32),
        schema_name=np.array(SCHEMA_NAME),
        g_dlr=g,
        orbital_dim=np.array(g.shape[0], dtype=np.int32),
        n_tau=np.array(g.shape[2], dtype=np.int32),
    )


def _validate_schema_header(data: Any) -> None:
    version = _read_int(data, "schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {version}; expected {SCHEMA_VERSION}")
    name = str(np.asarray(_require_key(data, "schema_name")).item())
    if name != SCHEMA_NAME:
        raise ValueError(f"unsupported schema_name {name!r}; expected {SCHEMA_NAME!r}")


def _validate_block(name: str, block: np.ndarray) -> np.ndarray:
    arr = np.asarray(block)
    if arr.ndim != 3 or arr.shape[0] != arr.shape[1] or arr.shape[2] <= 0:
        raise ValueError(f"{name} must have shape (M, M, N_tau), got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.complexfloating):
        arr = arr.astype(np.complex64)
    if not np.allclose(arr, np.swapaxes(arr.conj(), 0, 1), atol=1e-6, rtol=1e-6):
        raise ValueError(
            f"{name} must be Hermitian: {name}[j, i, t] == conj({name}[i, j, t])"
        )
    return arr


def _require_key(data: Any, key: str) -> np.ndarray:
    if key not in data:
        raise ValueError(f"missing required npz key {key!r}")
    return data[key]


def _read_float(data: Any, key: str) -> float:
    arr = np.asarray(_require_key(data, key))
    if arr.shape != ():
        raise ValueError(f"{key} must be a scalar, got shape {arr.shape}")
    return float(arr.item())


def _read_int(data: Any, key: str) -> int:
    arr = np.asarray(_require_key(data, key))
    if arr.shape != ():
        raise ValueError(f"{key} must be a scalar, got shape {arr.shape}")
    return int(arr.item())
