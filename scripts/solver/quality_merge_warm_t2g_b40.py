#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from sigml.solver.pydlr_grid import PydlrGrid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quality-gate and merge SrVO3-warm beta=40 t2g CTHYB labels."
    )
    parser.add_argument("--bootstrap", required=True, help="Existing accepted bootstrap NPZ.")
    parser.add_argument("--new-shards", required=True, help="Directory of new one-row shard NPZs.")
    parser.add_argument("--output", required=True, help="Merged accepted dataset NPZ.")
    parser.add_argument("--summary", required=True, help="JSON quality summary path.")
    parser.add_argument("--reject-log", required=True, help="JSONL reject log path.")
    parser.add_argument("--beta", type=float, default=40.0)
    parser.add_argument("--omega-max", type=float, default=15.0)
    parser.add_argument("--eps", type=float, default=1e-10)
    parser.add_argument("--min-sign", type=float, default=0.98)
    parser.add_argument("--min-spinful-occupation", type=float, default=0.7)
    parser.add_argument("--max-spinful-occupation", type=float, default=1.3)
    parser.add_argument("--max-diagonal-gtau", type=float, default=1e-2)
    parser.add_argument("--max-positive-giw-eig", type=float, default=2e-2)
    return parser.parse_args()


def _rows(path: Path) -> list[dict[str, Any]]:
    with np.load(path, allow_pickle=True) as data:
        n = int(np.asarray(data["g"]).shape[0])
        rows = []
        for idx in range(n):
            rows.append(
                {
                    "origin": str(path),
                    "row": idx,
                    "delta": np.asarray(data["delta"][idx], dtype=np.complex128),
                    "g": np.asarray(data["g"][idx], dtype=np.complex128),
                    "U": float(np.asarray(data["U"])[idx]),
                    "mu": float(np.asarray(data["mu"])[idx]),
                    "beta": float(np.asarray(data["beta"])[idx]),
                    "J": float(np.asarray(data["J"])[idx]),
                    "eps_d": np.asarray(data["eps_d"][idx], dtype=np.float64),
                    "source": str(np.asarray(data["source"])[idx]),
                    "info_json": str(np.asarray(data["info_json"])[idx]),
                }
            )
    return rows


def _cthyb_info(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw).get("cthyb", {})
    except json.JSONDecodeError:
        return {}


def _occupations(row: dict[str, Any], info: dict[str, Any]) -> np.ndarray:
    if info.get("orbital_occupations") is not None:
        return np.asarray(info["orbital_occupations"], dtype=np.float64)
    return -np.diag(row["g"][:, :, -1]).real.astype(np.float64)


def _max_positive_giw_eig(grid: PydlrGrid, g_tau: np.ndarray) -> float:
    giw = grid.gtau_to_giw(g_tau)
    max_eig = -np.inf
    for idx, iw in enumerate(grid.iw_nodes):
        if complex(iw).imag <= 0:
            continue
        spectral = (giw[:, :, idx] - giw[:, :, idx].conj().T) / (2j)
        max_eig = max(max_eig, float(np.max(np.linalg.eigvalsh(spectral).real)))
    return float(max_eig)


def _quality(row: dict[str, Any], grid: PydlrGrid, args: argparse.Namespace) -> dict[str, Any]:
    reasons: list[str] = []
    delta = row["delta"]
    g = row["g"]
    info = _cthyb_info(row["info_json"])
    sign = info.get("average_sign")
    sign = None if sign is None else float(sign)
    occ = _occupations(row, info)

    if delta.shape != (3, 3, grid.n_tau):
        reasons.append(f"delta_shape={delta.shape}")
    if g.shape != (3, 3, grid.n_tau):
        reasons.append(f"g_shape={g.shape}")
    if not np.isclose(row["beta"], args.beta):
        reasons.append(f"beta={row['beta']}")
    if not np.all(np.isfinite(delta)) or not np.all(np.isfinite(g)):
        reasons.append("nonfinite")
    if not np.allclose(delta, np.swapaxes(delta.conj(), 0, 1), atol=1e-8, rtol=1e-8):
        reasons.append("delta_nonhermitian")
    if not np.allclose(g, np.swapaxes(g.conj(), 0, 1), atol=1e-8, rtol=1e-8):
        reasons.append("g_nonhermitian")
    if sign is None or sign < args.min_sign:
        reasons.append(f"sign={sign}")
    if occ.shape != (3,) or not np.all(np.isfinite(occ)):
        reasons.append(f"occupation_shape={occ.shape}")

    diag = np.diagonal(g, axis1=0, axis2=1).T
    max_diag_gtau = float(np.max(diag.real))
    max_diag_imag = float(np.max(np.abs(diag.imag)))
    spinful_occ_total = float(2.0 * np.sum(occ))
    max_positive_giw_eig = _max_positive_giw_eig(grid, g)
    if max_diag_gtau > args.max_diagonal_gtau:
        reasons.append(f"positive_Gtau_diag={max_diag_gtau:.6g}")
    if spinful_occ_total < args.min_spinful_occupation or spinful_occ_total > args.max_spinful_occupation:
        reasons.append(f"spinful_occ_total={spinful_occ_total:.6g}")
    if max_positive_giw_eig > args.max_positive_giw_eig:
        reasons.append(f"positive_Giw_eig={max_positive_giw_eig:.6g}")

    return {
        "accepted": not reasons,
        "reasons": reasons,
        "average_sign": sign,
        "occupation": occ.tolist(),
        "spinful_occupation_total": spinful_occ_total,
        "max_diagonal_gtau": max_diag_gtau,
        "max_diagonal_gtau_imag_abs": max_diag_imag,
        "max_positive_giw_eig": max_positive_giw_eig,
    }


def _stack(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "delta": np.stack([r["delta"] for r in rows]).astype(np.complex128),
        "g": np.stack([r["g"] for r in rows]).astype(np.complex128),
        "U": np.asarray([r["U"] for r in rows], dtype=np.float64),
        "mu": np.asarray([r["mu"] for r in rows], dtype=np.float64),
        "beta": np.asarray([r["beta"] for r in rows], dtype=np.float64),
        "J": np.asarray([r["J"] for r in rows], dtype=np.float64),
        "eps_d": np.stack([r["eps_d"] for r in rows]).astype(np.float64),
        "source": np.asarray([r["source"] for r in rows]),
        "info_json": np.asarray([r["info_json"] for r in rows]),
    }


def _features(arr: np.ndarray) -> np.ndarray:
    paired = np.stack((arr.real, arr.imag), axis=-1)
    return np.reshape(paired, (arr.shape[0], -1)).astype(np.float32)


def _feature_arrays(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta_features = _features(arrays["delta"])
    g_features = _features(arrays["g"])
    eps_scalar = np.mean(arrays["eps_d"], axis=1).astype(np.float32)
    scalars = np.stack(
        (
            arrays["U"].astype(np.float32),
            ((arrays["mu"] - eps_scalar) / arrays["U"]).astype(np.float32),
            arrays["beta"].astype(np.float32),
            arrays["J"].astype(np.float32),
        ),
        axis=1,
    )
    x = np.concatenate((delta_features, scalars), axis=1).astype(np.float32)
    y = g_features.astype(np.float32)
    scalar_names = np.asarray(("U", "mu_minus_eps_d_over_U", "beta", "J"))
    return x, y, scalar_names


def _stats(values: list[float]) -> dict[str, float] | None:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    return {
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    args = parse_args()
    grid = PydlrGrid(beta=args.beta, lamb=args.beta * args.omega_max, eps=args.eps)
    bootstrap_rows = _rows(Path(args.bootstrap))
    shard_paths = sorted(Path(args.new_shards).glob("bootstrap_t2g_b40_*.npz"))
    new_rows = [row for path in shard_paths for row in _rows(path)]

    accepted: list[dict[str, Any]] = []
    reject_records: list[dict[str, Any]] = []
    quality_records: list[dict[str, Any]] = []
    for row in [*bootstrap_rows, *new_rows]:
        quality = _quality(row, grid, args)
        record = {
            "origin": row["origin"],
            "row": row["row"],
            "source": row["source"],
            **quality,
        }
        quality_records.append(record)
        if quality["accepted"]:
            accepted.append(row)
        else:
            reject_records.append(record)

    arrays = _stack(accepted)
    x, y, scalar_names = _feature_arrays(arrays)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays, x=x, y=y, scalar_names=scalar_names)

    reject_log = Path(args.reject_log)
    reject_log.parent.mkdir(parents=True, exist_ok=True)
    reject_log.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in reject_records))

    signs = [r["average_sign"] for r in quality_records if r["accepted"] and r["average_sign"] is not None]
    occ = np.asarray([r["occupation"] for r in quality_records if r["accepted"]], dtype=np.float64)
    spinful_occ = [r["spinful_occupation_total"] for r in quality_records if r["accepted"]]
    gtau_max = [r["max_diagonal_gtau"] for r in quality_records if r["accepted"]]
    giw_max = [r["max_positive_giw_eig"] for r in quality_records if r["accepted"]]
    summary = {
        "bootstrap_path": str(Path(args.bootstrap)),
        "new_shard_dir": str(Path(args.new_shards)),
        "output": str(output),
        "reject_log": str(reject_log),
        "raw_bootstrap_rows": len(bootstrap_rows),
        "raw_new_rows": len(new_rows),
        "accepted_total_rows": len(accepted),
        "accepted_bootstrap_rows": sum(1 for row in accepted if row["origin"] == str(Path(args.bootstrap))),
        "accepted_new_rows": sum(1 for row in accepted if row["origin"] != str(Path(args.bootstrap))),
        "rejected_rows": len(reject_records),
        "reject_reasons": {
            reason: sum(reason in r["reasons"] for r in reject_records)
            for reason in sorted({reason for r in reject_records for reason in r["reasons"]})
        },
        "shape_delta": list(arrays["delta"].shape),
        "shape_g": list(arrays["g"].shape),
        "source_counts": {
            str(source): int(np.sum(arrays["source"] == source))
            for source in sorted(set(arrays["source"]))
        },
        "average_sign": _stats(signs),
        "occupation_spin_averaged_per_orbital": {
            "min": None if occ.size == 0 else np.min(occ, axis=0).tolist(),
            "mean": None if occ.size == 0 else np.mean(occ, axis=0).tolist(),
            "max": None if occ.size == 0 else np.max(occ, axis=0).tolist(),
        },
        "occupation_spinful_total": _stats(spinful_occ),
        "max_diagonal_gtau": _stats(gtau_max),
        "max_positive_giw_eig": _stats(giw_max),
        "feature_shape_x": list(x.shape),
        "feature_shape_y": list(y.shape),
        "scalar_names": [str(name) for name in scalar_names],
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"accepted={summary['accepted_total_rows']} "
        f"bootstrap={summary['accepted_bootstrap_rows']} "
        f"new={summary['accepted_new_rows']} rejected={summary['rejected_rows']}"
    )


if __name__ == "__main__":
    main()
