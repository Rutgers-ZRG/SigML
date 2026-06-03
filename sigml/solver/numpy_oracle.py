from __future__ import annotations

from pathlib import Path

import numpy as np

from sigml.solver.valenti_grid import DEFAULT_MESH_PATH, ValentiOrb1Grid


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _gelu(x: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import erf

        return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))
    except ImportError:
        erf = np.vectorize(__import__("math").erf)
        return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return x @ weight.T + bias


class NumpyOrb1Oracle:
    """NumPy-only inference path for Valenti's orb1 network."""

    def __init__(
        self,
        weights_path: str | Path,
        mesh_path: str | Path = DEFAULT_MESH_PATH,
    ):
        self.weights_path = str(weights_path)
        self.grid = ValentiOrb1Grid(mesh_path, beta=70.0)
        with np.load(self.weights_path) as data:
            self.weights = {key: data[key] for key in data.files}

    def solve(
        self,
        delta_vec: np.ndarray,
        U: float,
        mu: float,
        beta: float,
        eps_d: float = 0.0,
    ) -> np.ndarray:
        delta = np.asarray(delta_vec, dtype=np.float64)
        if delta.shape != (self.grid.feature_dim,):
            raise ValueError(f"Expected delta_vec shape ({self.grid.feature_dim},), got {delta.shape}")
        if U == 0:
            raise ValueError("U must be nonzero because the Valenti input uses (mu - eps_d) / U")

        scalars = np.array([U, (mu - eps_d) / U, beta], dtype=np.float64)
        x = np.concatenate([delta, scalars], axis=0)[None, :]
        params = x[:, -3:]
        w = self.weights

        x_params = _relu(_linear(params, w["alpha1.weight"], w["alpha1.bias"]))
        h = _relu(_linear(x, w["fc1.weight"], w["fc1.bias"]))
        h = _relu(
            (1.0 - float(w["eps1"])) * _linear(h, w["fc2.weight"], w["fc2.bias"])
            + float(w["eps1"]) * _linear(x_params, w["alpha2.weight"], w["alpha2.bias"])
        )
        alpha3 = _linear(x_params, w["alpha3.weight"], w["alpha3.bias"])
        h = _gelu(
            (1.0 - float(w["eps2"])) * _linear(h, w["fc3.weight"], w["fc3.bias"])
            + float(w["eps2"]) * alpha3
        )
        h = _gelu(
            (1.0 - float(w["eps3"])) * _linear(h, w["fc4.weight"], w["fc4.bias"])
            + float(w["eps3"]) * alpha3
        )
        return _linear(h, w["fc5.weight"], w["fc5.bias"])[0]
