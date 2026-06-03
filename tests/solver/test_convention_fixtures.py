from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from sigml.solver.labeler import CtsegLabeler
from sigml.solver.valenti_grid import ValentiOrb1Grid


MESH = "/Users/li/dev/RA/dmft/ref/mlDMFT/mldmft/models/orb1/mesh_beta70.h5"
FIXTURE_DIR = Path(__file__).with_name("fixtures")
VALENTI_FIXTURE = FIXTURE_DIR / "valenti_reference_delta_beta70.npz"
CTSEG_FIXTURE = FIXTURE_DIR / "ctseg_projection_near_atomic_beta70.npz"


def _load_fixture(path: Path):
    if not path.exists():
        pytest.skip(f"missing recorded fixture: {path}")
    return np.load(path)


def test_valenti_grid_matches_recorded_mldmft_utils_reference():
    fixture = _load_fixture(VALENTI_FIXTURE)
    grid = ValentiOrb1Grid(MESH, beta=float(fixture["beta"]))

    delta_vec = fixture["delta_vec"]
    valenti_nn_input = fixture["valenti_nn_input"]
    np.testing.assert_allclose(grid.gtau_to_vec(grid.vec_to_gtau(delta_vec)).real, valenti_nn_input[:-3])
    # Valenti/TRIQS' DLR imfreq object stores beta-scaled values; ValentiOrb1Grid
    # exposes the physical Matsubara convention used by the solver PoC.
    np.testing.assert_allclose(
        grid.gtau_to_giw(grid.vec_to_gtau(delta_vec)),
        fixture["valenti_delta_iw"] / float(fixture["beta"]),
        atol=5e-10,
        rtol=5e-10,
    )


class _Tau:
    def __init__(self, value: float):
        self.value = float(value)

    def __float__(self) -> float:
        return self.value


class _Block:
    def __init__(self, mesh):
        self.mesh = mesh
        self._values = {float(tau): np.zeros((1, 1), dtype=complex) for tau in mesh}

    def __getitem__(self, tau):
        return self._values[float(tau)]


class _BlockGf:
    def __init__(self, mesh, gf_struct):
        self.mesh = mesh
        self.blocks = {name: _Block(mesh) for name, _ in gf_struct}

    def __getitem__(self, name):
        return self.blocks[name]


class _HDFArchive:
    def __init__(self, path, mode):
        self._mesh = [_Tau(tau) for tau in ValentiOrb1Grid(MESH, beta=70.0).tau_nodes]

    def __enter__(self):
        return {"mesh_dlr_imtime": self._mesh}

    def __exit__(self, exc_type, exc, tb):
        return False


class _RecordedGTau:
    def __init__(self, tau_mesh: np.ndarray, g_tau: np.ndarray):
        self.mesh = [_Tau(tau) for tau in tau_mesh]
        self._tau = np.asarray(tau_mesh, dtype=float)
        self._real = np.asarray(g_tau).real
        self._imag = np.asarray(g_tau).imag

    def __call__(self, tau: float):
        value = np.interp(tau, self._tau, self._real) + 1j * np.interp(tau, self._tau, self._imag)
        return np.asarray([[value]], dtype=complex)


def test_ctseg_default_dlr_projection_matches_recorded_fixture(monkeypatch):
    fixture = _load_fixture(CTSEG_FIXTURE)
    grid = ValentiOrb1Grid(MESH, beta=float(fixture["beta"]))

    fake_gf = types.SimpleNamespace(
        BlockGf=_BlockGf,
        make_gf_dlr=lambda gf: gf,
        make_gf_dlr_imtime=lambda gf: gf,
    )
    monkeypatch.setitem(sys.modules, "h5", types.SimpleNamespace(HDFArchive=_HDFArchive))
    monkeypatch.setitem(sys.modules, "triqs", types.SimpleNamespace(gf=fake_gf))
    monkeypatch.setitem(sys.modules, "triqs.gf", fake_gf)

    labeler = CtsegLabeler(grid=grid)
    g_tau = _RecordedGTau(fixture["raw_tau_mesh"], fixture["raw_g_tau"])
    projected = labeler._project_g_tau_to_valenti_vec(g_tau)

    assert labeler.projection == "dlr"
    np.testing.assert_allclose(projected, fixture["dlr_g_vec"], atol=1e-6, rtol=1e-6)
