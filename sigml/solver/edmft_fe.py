from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sigml.solver.pydlr_grid import PydlrGrid


ORBITALS_REAL_HARMONIC = ("z^2", "x^2-y^2", "xz", "yz", "xy")
SCALAR_NAMES = ("U", "mu_over_U", "beta", "J")


@dataclass(frozen=True)
class EdmftBlock:
    omega: np.ndarray
    values: np.ndarray
    header: dict[str, object]


@dataclass(frozen=True)
class ImpurityMetadata:
    impurity: int
    U: float
    J: float
    beta: float
    mu: float
    Ed: np.ndarray
    nf0: float | None
    dc_scheme: str
    orbital_order: tuple[str, ...]
    sigind: np.ndarray


def parse_key_value_file(path: str | Path) -> dict[str, object]:
    values: dict[str, object] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = _parse_literal(value.strip())
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            values[parts[0]] = _parse_literal(parts[1].strip())
    return values


def parse_params_dat(path: str | Path) -> dict[str, object]:
    params = parse_key_value_file(path)
    for key, value in list(params.items()):
        if key.startswith("iparams") and isinstance(value, dict):
            params[key] = {k: _param_value(v) for k, v in value.items()}
    return params


def parse_indmfl(path: str | Path) -> dict[int, dict[str, object]]:
    lines = Path(path).read_text().splitlines()
    blocks: dict[int, dict[str, object]] = {}
    i = 0
    while i < len(lines):
        match = re.match(r"\s*(\d+)\s+(\d+)\s+(\d+)\s+#\s*cix-num", lines[i])
        if not match:
            i += 1
            continue
        cix = int(match.group(1))
        dim = int(match.group(2))
        i += 1
        while i < len(lines) and "Independent components" not in lines[i]:
            i += 1
        i += 1
        labels = tuple(re.findall(r"'([^']+)'", lines[i]))
        while i < len(lines) and "Sigind follows" not in lines[i]:
            i += 1
        i += 1
        sigind = np.asarray(
            [[int(x) for x in lines[i + row].split("#", 1)[0].split()] for row in range(dim)],
            dtype=int,
        )
        blocks[cix] = {"dim": dim, "orbital_order": labels, "sigind": sigind}
        i += dim
    if not blocks:
        raise ValueError(f"No cix blocks found in {path}")
    return blocks


def parse_complex_table(path: str | Path, components: int | None = None) -> EdmftBlock:
    headers: list[str] = []
    rows: list[list[float]] = []
    for raw in Path(path).read_text().splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            headers.append(stripped[1:].strip())
            continue
        rows.append([float(x) for x in stripped.split()])
    if not rows:
        raise ValueError(f"No numeric rows found in {path}")
    arr = np.asarray(rows, dtype=float)
    if (arr.shape[1] - 1) % 2 != 0:
        raise ValueError(f"Expected frequency plus real/imaginary pairs in {path}, got {arr.shape[1]} columns")
    n_components = (arr.shape[1] - 1) // 2
    if components is not None and n_components != components:
        raise ValueError(f"Expected {components} components in {path}, got {n_components}")
    values = arr[:, 1::2] + 1j * arr[:, 2::2]
    return EdmftBlock(omega=arr[:, 0], values=values, header=_parse_headers(headers))


def components_to_diagonal_block(values: np.ndarray, orbital_dim: int = 5) -> np.ndarray:
    arr = np.asarray(values, dtype=complex)
    if arr.ndim != 2 or arr.shape[1] != orbital_dim:
        raise ValueError(f"Expected component values with shape (n_freq, {orbital_dim}), got {arr.shape}")
    block = np.zeros((orbital_dim, orbital_dim, arr.shape[0]), dtype=complex)
    for idx in range(orbital_dim):
        block[idx, idx, :] = arr[:, idx]
    return block


def parse_impurity_metadata(root: str | Path, impurity: int) -> ImpurityMetadata:
    root_path = Path(root)
    params = parse_params_dat(root_path / "params.dat")
    imp_params = parse_key_value_file(root_path / f"imp.{impurity}" / "PARAMS")
    indmfl_path = next(root_path.glob("*.indmfl"))
    indmfl = parse_indmfl(indmfl_path)
    cix = impurity + 1
    block = indmfl[cix]
    iparams = params.get(f"iparams{impurity}", {})
    merged = {**(iparams if isinstance(iparams, dict) else {}), **imp_params}
    beta = float(merged["beta"])
    return ImpurityMetadata(
        impurity=impurity,
        U=float(merged["U"]),
        J=float(merged["J"]),
        beta=beta,
        mu=float(merged["mu"]),
        Ed=np.asarray(merged["Ed"], dtype=float),
        nf0=None if "nf0" not in merged else float(merged["nf0"]),
        dc_scheme=str(params.get("DCs", "")),
        orbital_order=tuple(str(x) for x in block["orbital_order"]),
        sigind=np.asarray(block["sigind"], dtype=int),
    )


def matched_iteration_suffixes(root: str | Path, impurity: int, include_current: bool = True) -> list[str]:
    imp_dir = Path(root) / f"imp.{impurity}"
    delta_suffixes = {_suffix(path.name, "Delta.inp") for path in imp_dir.glob("Delta.inp*")}
    gf_suffixes = {_suffix(path.name, "Gf.out") for path in imp_dir.glob("Gf.out*")}
    suffixes = sorted(delta_suffixes & gf_suffixes, key=_suffix_sort_key)
    if not include_current:
        suffixes = [x for x in suffixes if x]
    return suffixes


def assemble_fe_dataset(
    root: str | Path,
    output: str | Path,
    *,
    summary: str | Path | None = None,
    impurities: tuple[int, ...] = (0, 1, 2, 3),
    beta: float = 2.32,
    lamb: float = 80.0,
    eps: float = 1e-10,
    include_current: bool = True,
) -> dict[str, object]:
    root_path = Path(root)
    grid = PydlrGrid(beta=beta, lamb=lamb, eps=eps)
    delta_rows = []
    g_rows = []
    delta_coeff_rows = []
    g_coeff_rows = []
    sigma_coeff_rows = []
    siginp_coeff_rows = []
    siginp_s_oo = []
    siginp_Edc = []
    U = []
    J = []
    beta_rows = []
    mu = []
    eps_d = []
    source = []
    info_json = []
    impurity_index = []
    suffix_rows = []
    nf = []
    causality_max_im = []
    roundtrip_delta = []
    roundtrip_g = []

    for impurity in impurities:
        meta = parse_impurity_metadata(root_path, impurity)
        if abs(meta.beta - beta) > 1e-12:
            raise ValueError(f"imp.{impurity} beta {meta.beta} does not match requested beta {beta}")
        for suffix in matched_iteration_suffixes(root_path, impurity, include_current=include_current):
            delta_file = root_path / f"imp.{impurity}" / f"Delta.inp{suffix}"
            gf_file = root_path / f"imp.{impurity}" / f"Gf.out{suffix}"
            sig_file = root_path / f"imp.{impurity}" / f"Sig.out{suffix}"
            siginp_file = root_path / f"sig.inp{suffix}"
            delta_iw = parse_complex_table(delta_file, components=5)
            g_iw = parse_complex_table(gf_file, components=5)
            sigma_iw = parse_complex_table(sig_file, components=5) if sig_file.exists() else None
            siginp_iw = parse_complex_table(siginp_file) if siginp_file.exists() else None
            if not np.allclose(delta_iw.omega, g_iw.omega, rtol=0.0, atol=1e-10):
                raise ValueError(f"Matsubara grid mismatch for imp.{impurity} suffix {suffix or '<current>'}")
            delta_block_iw = components_to_diagonal_block(delta_iw.values)
            g_block_iw = components_to_diagonal_block(g_iw.values)
            delta_coeff = grid.coeffs_from_giw_lstsq(delta_iw.omega, delta_block_iw)
            g_coeff = grid.coeffs_from_giw_lstsq(g_iw.omega, g_block_iw)
            sigma_coeff = (
                grid.coeffs_from_giw_lstsq(sigma_iw.omega, components_to_diagonal_block(sigma_iw.values))
                if sigma_iw is not None
                else np.full((5, 5, grid.rank), np.nan + 1j * np.nan)
            )
            if siginp_iw is not None:
                offset = impurity * 5 if siginp_iw.values.shape[1] >= (impurity + 1) * 5 else 0
                siginp_values = siginp_iw.values[:, offset : offset + 5]
                siginp_coeff = grid.coeffs_from_giw_lstsq(
                    siginp_iw.omega, components_to_diagonal_block(siginp_values)
                )
                s_oo = np.asarray(siginp_iw.header.get("s_oo", np.full(20, np.nan)), dtype=float)[
                    offset : offset + 5
                ]
                edc = np.asarray(siginp_iw.header.get("Edc", np.full(20, np.nan)), dtype=float)[
                    offset : offset + 5
                ]
            else:
                siginp_coeff = np.full((5, 5, grid.rank), np.nan + 1j * np.nan)
                s_oo = np.full(5, np.nan)
                edc = np.full(5, np.nan)
            delta_tau = _real_diagonal_tau(grid.gtau_from_coeffs(delta_coeff))
            g_tau = _real_diagonal_tau(grid.gtau_from_coeffs(g_coeff))
            g_iw_fit = grid.giw_from_coeffs_at_omega(g_coeff, g_iw.omega)
            _, g_iw_pos = grid.giw_positive(g_coeff)

            delta_rows.append(delta_tau)
            g_rows.append(g_tau)
            delta_coeff_rows.append(delta_coeff)
            g_coeff_rows.append(g_coeff)
            sigma_coeff_rows.append(sigma_coeff)
            siginp_coeff_rows.append(siginp_coeff)
            siginp_s_oo.append(s_oo)
            siginp_Edc.append(edc)
            U.append(meta.U)
            J.append(meta.J)
            beta_rows.append(meta.beta)
            mu.append(meta.mu)
            eps_d.append(meta.Ed)
            source.append(str(root_path))
            impurity_index.append(impurity)
            suffix_rows.append(suffix or "current")
            nf.append(float(g_iw.header.get("nf", np.nan)))
            causality_max_im.append(_max_imag_eig(g_iw_pos))
            roundtrip_delta.append(
                float(np.max(np.abs(grid.giw_from_coeffs_at_omega(delta_coeff, delta_iw.omega) - delta_block_iw)))
            )
            roundtrip_g.append(float(np.max(np.abs(g_iw_fit - g_block_iw))))
            info_json.append(
                json.dumps(
                    {
                        "source": str(root_path),
                        "impurity": impurity,
                        "suffix": suffix or "current",
                        "U": meta.U,
                        "J": meta.J,
                        "beta": meta.beta,
                        "mu_qmc": meta.mu,
                        "Ed": meta.Ed.tolist(),
                        "s_oo": s_oo.tolist(),
                        "Edc": edc.tolist(),
                        "nf": nf[-1],
                        "dc_scheme": meta.dc_scheme,
                        "orbital_order": list(meta.orbital_order),
                        "sigind": meta.sigind.tolist(),
                    },
                    sort_keys=True,
                )
            )

    delta_arr = np.asarray(delta_rows)
    g_arr = np.asarray(g_rows)
    U_arr = np.asarray(U, dtype=np.float32)
    mu_arr = np.asarray(mu, dtype=np.float32)
    beta_arr = np.asarray(beta_rows, dtype=np.float32)
    J_arr = np.asarray(J, dtype=np.float32)
    eps_d_arr = np.asarray(eps_d, dtype=np.float32)
    x_block = np.stack((delta_arr.real, delta_arr.imag), axis=-1).reshape(delta_arr.shape[0], -1)
    y = np.stack((g_arr.real, g_arr.imag), axis=-1).reshape(g_arr.shape[0], -1).astype(np.float32)
    scalars = np.stack((U_arr, mu_arr / U_arr, beta_arr, J_arr), axis=1)
    x = np.concatenate((x_block.astype(np.float32), scalars.astype(np.float32)), axis=1)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        delta=delta_arr,
        g=g_arr,
        delta_dlr_coeffs=np.asarray(delta_coeff_rows),
        g_dlr_coeffs=np.asarray(g_coeff_rows),
        sigma_dlr_coeffs=np.asarray(sigma_coeff_rows),
        siginp_dynamic_dlr_coeffs=np.asarray(siginp_coeff_rows),
        siginp_s_oo=np.asarray(siginp_s_oo, dtype=np.float64),
        siginp_Edc=np.asarray(siginp_Edc, dtype=np.float64),
        U=U_arr,
        mu=mu_arr,
        beta=beta_arr,
        J=J_arr,
        eps_d=eps_d_arr,
        x=x,
        y=y,
        scalar_names=np.asarray(SCALAR_NAMES),
        source=np.asarray(source),
        info_json=np.asarray(info_json),
        impurity_index=np.asarray(impurity_index, dtype=np.int64),
        iteration_suffix=np.asarray(suffix_rows),
        nf=np.asarray(nf, dtype=np.float64),
        orbital_order=np.asarray(ORBITALS_REAL_HARMONIC),
        dlr_tau_nodes=grid.tau_nodes,
        dlr_iw_nodes=grid.iw_nodes,
        dlr_lamb=np.asarray(lamb),
        dlr_eps=np.asarray(eps),
        causality_max_imag_eig=np.asarray(causality_max_im, dtype=np.float64),
        dlr_roundtrip_delta_max=np.asarray(roundtrip_delta, dtype=np.float64),
        dlr_roundtrip_g_max=np.asarray(roundtrip_g, dtype=np.float64),
    )

    summary_data = {
        "output": str(output_path),
        "n_labels": int(delta_arr.shape[0]),
        "delta_shape": list(delta_arr.shape),
        "g_shape": list(g_arr.shape),
        "beta": beta,
        "lamb": lamb,
        "eps": eps,
        "rank": int(grid.rank),
        "scalar_names": list(SCALAR_NAMES),
        "impurity_counts": {
            str(i): int(np.sum(np.asarray(impurity_index) == i)) for i in sorted(set(impurity_index))
        },
        "max_causality_imag_eig": float(np.max(causality_max_im)) if causality_max_im else None,
        "max_dlr_roundtrip_delta": float(np.max(roundtrip_delta)) if roundtrip_delta else None,
        "max_dlr_roundtrip_g": float(np.max(roundtrip_g)) if roundtrip_g else None,
        "iteration_suffixes": sorted(set(suffix_rows), key=_suffix_sort_key),
    }
    if summary is not None:
        Path(summary).parent.mkdir(parents=True, exist_ok=True)
        Path(summary).write_text(json.dumps(summary_data, indent=2, sort_keys=True) + "\n")
    return summary_data


def _parse_literal(value: str) -> object:
    cleaned = value.strip().rstrip(",")
    try:
        return ast.literal_eval(cleaned)
    except (ValueError, SyntaxError):
        return cleaned.strip("'\"")


def _param_value(value: object) -> object:
    if isinstance(value, list) and value:
        return value[0]
    return value


def _parse_headers(headers: list[str]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for line in headers:
        for key, raw in re.findall(r"(\w+)=\s*((?:\[[^\]]*\])|(?:\S+))", line):
            parsed[key] = _parse_literal(raw)
    return parsed


def _suffix(name: str, stem: str) -> str:
    if name == stem:
        return ""
    if not name.startswith(stem + "."):
        return ""
    return name[len(stem) :]


def _suffix_sort_key(suffix: str) -> tuple[int, tuple[int, ...] | tuple[str, ...]]:
    if suffix in ("", "current"):
        return (1, ())
    body = suffix[1:] if suffix.startswith(".") else suffix
    parts = body.split(".")
    if all(part.isdigit() for part in parts):
        return (0, tuple(int(part) for part in parts))
    return (0, tuple(parts))


def _max_imag_eig(g_iw_positive: np.ndarray) -> float:
    vals = np.moveaxis(np.asarray(g_iw_positive), -1, 0)
    max_imag = -np.inf
    for mat in vals:
        eig = np.linalg.eigvals(mat)
        max_imag = max(max_imag, float(np.max(eig.imag)))
    return max_imag


def _real_diagonal_tau(block_tau: np.ndarray) -> np.ndarray:
    arr = np.asarray(block_tau, dtype=complex)
    if arr.ndim != 3 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Expected (M, M, n_tau) block, got {arr.shape}")
    projected = np.zeros_like(arr)
    for idx in range(arr.shape[0]):
        projected[idx, idx, :] = arr[idx, idx, :].real
    return projected
