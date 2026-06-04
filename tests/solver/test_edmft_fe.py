import json

import numpy as np

from sigml.solver.dataset import SolverDataset
from sigml.solver.edmft_fe import (
    ORBITALS_REAL_HARMONIC,
    assemble_fe_dataset,
    components_to_diagonal_block,
    matched_iteration_suffixes,
    parse_complex_table,
    parse_impurity_metadata,
)
from sigml.solver.pydlr_grid import PydlrGrid


def test_pydlr_beta232_lstsq_round_trips_causal_diagonal_block():
    grid = PydlrGrid(beta=2.32, lamb=80.0, eps=1e-10)
    omega = (2 * np.arange(199) + 1) * np.pi / grid.beta
    poles = np.asarray([1.4, 1.6, 2.5, 2.7, 2.8])
    giw = np.zeros((5, 5, omega.size), dtype=complex)
    for orb, pole in enumerate(poles):
        giw[orb, orb] = 1.0 / (1j * omega - pole)

    coeffs = grid.coeffs_from_giw_lstsq(omega, giw)
    gtau = grid.gtau_from_coeffs(coeffs)
    fitted = grid.giw_from_coeffs_at_omega(coeffs, omega)

    assert grid.beta == 2.32
    assert coeffs.shape == (5, 5, grid.rank)
    assert gtau.shape == (5, 5, grid.n_tau)
    np.testing.assert_allclose(fitted, giw, atol=1e-8)
    for iwn in range(fitted.shape[-1]):
        assert np.max(np.linalg.eigvals(fitted[..., iwn]).imag) <= 1e-10


def test_edmft_fe_parser_reads_diagonal_real_harmonic_fixture(tmp_path):
    root = _write_fe_fixture(tmp_path)

    meta = parse_impurity_metadata(root, 0)
    delta = parse_complex_table(root / "imp.0" / "Delta.inp.1.1", components=5)
    block = components_to_diagonal_block(delta.values)

    assert meta.beta == 2.32
    assert meta.U == 5.0
    assert meta.J == 0.93
    assert meta.dc_scheme == "nominal"
    assert meta.orbital_order == ORBITALS_REAL_HARMONIC
    assert meta.sigind.tolist() == np.diag([1, 2, 3, 4, 5]).tolist()
    assert matched_iteration_suffixes(root, 0, include_current=False) == [".1.1"]
    assert delta.values.shape == (199, 5)
    assert block.shape == (5, 5, 199)
    offdiag = block.copy()
    for idx in range(5):
        offdiag[idx, idx, :] = 0.0
    assert np.count_nonzero(offdiag) == 0


def test_assemble_fe_dataset_emits_m5_beta232_warm_labels(tmp_path):
    root = _write_fe_fixture(tmp_path)
    output = tmp_path / "fe_tscan5000k_b232.npz"
    summary = tmp_path / "summary.json"

    result = assemble_fe_dataset(
        root,
        output,
        summary=summary,
        impurities=(0,),
        lamb=80.0,
        eps=1e-10,
        include_current=False,
    )
    ds = SolverDataset(output)

    assert result["n_labels"] == 1
    assert result["delta_shape"] == [1, 5, 5, 29]
    assert ds.orbital_dim == 5
    assert ds.delta_tau.shape == (1, 5, 5, 29)
    assert ds.g_tau.shape == (1, 5, 5, 29)
    assert ds.scalar_names == ("U", "mu_over_U", "beta", "J")
    assert ds[0]["x"].shape == (5 * 5 * 2 * 29 + 4,)
    assert ds[0]["y"].shape == (5 * 5 * 2 * 29,)
    np.testing.assert_allclose(ds[0]["x"][-4:].numpy(), [5.0, 28.3 / 5.0, 2.32, 0.93])
    assert json.loads(summary.read_text())["max_causality_imag_eig"] <= 1e-10
    with np.load(output) as data:
        assert data["sigma_dlr_coeffs"].shape == (1, 5, 5, 29)
        assert data["siginp_dynamic_dlr_coeffs"].shape == (1, 5, 5, 29)
        np.testing.assert_allclose(data["siginp_s_oo"][0], [26.0, 26.1, 26.2, 26.3, 26.4])
        np.testing.assert_allclose(data["siginp_Edc"][0], [25.175] * 5)


def _write_fe_fixture(root):
    (root / "imp.0").mkdir()
    (root / "params.dat").write_text(
        """
solver='CTQMC'
DCs='nominal'
iparams0={"U":[5.0,"#"],"J":[0.93,"#"],"beta":[2.32,"#"],"nf0":[6.0,"#"]}
""".strip()
        + "\n"
    )
    (root / "imp.0" / "PARAMS").write_text(
        """
U 5.0 #
J 0.93 #
beta 2.32 #
nf0 6.0 #
Ed [-28.3, -28.31, -28.47, -28.48, -28.49] # Impurity levels
mu 28.3 # QMC chemical potential
""".strip()
        + "\n"
    )
    (root / "fetest7.indmfl").write_text(
        """
17 40 1 5
1 0.025 0.025 200 -3.000000 1.000000
1
1 1 0
  2 2 1
#================
1 5 5
1     5   5       # cix-num, dimension, num-independent-components
#---------------- # Independent components are --------------
'z^2' 'x^2-y^2' 'xz' 'yz' 'xy'
#---------------- # Sigind follows --------------------------
1 0 0 0 0
0 2 0 0 0
0 0 3 0 0
0 0 0 4 0
0 0 0 0 5
#---------------- # Transformation matrix follows -----------
""".strip()
        + "\n"
    )
    beta = 2.32
    omega = (2 * np.arange(199) + 1) * np.pi / beta
    _write_table(root / "imp.0" / "Delta.inp.1.1", omega, -0.4 * np.ones(5))
    _write_table(root / "imp.0" / "Gf.out.1.1", omega, np.asarray([1.4, 1.6, 2.5, 2.7, 2.8]), header=True)
    _write_table(root / "imp.0" / "Sig.out.1.1", omega, np.asarray([-26.0, -26.1, -26.2, -26.3, -26.4]), header=True)
    (root / "sig.inp.1.1").write_text(
        "# s_oo= [26.0, 26.1, 26.2, 26.3, 26.4]\n"
        "# Edc= [25.175, 25.175, 25.175, 25.175, 25.175]\n"
        + "".join(_table_lines(omega, np.asarray([0.2, 0.3, 0.4, 0.5, 0.6])))
    )
    return root


def _write_table(path, omega, poles, header=False):
    lines = []
    if header:
        lines.append("# nf=6.36 mu=28.3 T=0.4310344827586207 mom=[1,1,1,1,1,]\n")
    lines.extend(_table_lines(omega, poles))
    path.write_text("".join(lines))


def _table_lines(omega, poles):
    lines = []
    for w in omega:
        row = [w]
        for pole in poles:
            value = 1.0 / (1j * w - pole)
            row.extend([value.real, value.imag])
        lines.append(" ".join(f"{x:.16e}" for x in row) + "\n")
    return lines
