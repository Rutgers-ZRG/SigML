#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from sigml.solver.edmft_fe import assemble_fe_dataset


DEFAULT_ROOTS = (
    "/scratch/lz432/tscan_5000k_v3",
    "/scratch/lz432/Fe_fcc",
    "/scratch/lz432/Fe_fcc_disp_P1",
    "/scratch/lz432/tscan_5000k_pert1",
    "/scratch/lz432/fetest7",
    "/scratch/lz432/fetest8",
    "/scratch/lz432/fetest_base",
    "/scratch/lz432/fetest_fp",
    "/scratch/lz432/fetest_mat",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble scaled beta=2.32 Fe eDMFT warm labels from compatible existing roots."
    )
    parser.add_argument("--roots", nargs="+", default=list(DEFAULT_ROOTS))
    parser.add_argument("--output", default="data/fe_warm_v2.npz")
    parser.add_argument("--summary", default="data/fe_warm_v2_summary.json")
    parser.add_argument("--reject-log", default="data/fe_warm_v2_rejects.jsonl")
    parser.add_argument("--scratch-copy", default="/scratch/lz432/sigml_fe_warmstart/fe_warm_v2.npz")
    parser.add_argument("--beta", type=float, default=2.32)
    parser.add_argument("--lamb", type=float, default=80.0)
    parser.add_argument("--eps", type=float, default=1e-10)
    parser.add_argument("--include-current", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    summary_path = Path(args.summary)
    reject_log = Path(args.reject_log)
    reject_log.parent.mkdir(parents=True, exist_ok=True)
    reject_log.write_text("")

    with tempfile.TemporaryDirectory(prefix="fe_warm_v2_") as tmp:
        tmp_path = Path(tmp)
        shards = []
        shard_summaries = []
        for idx, root in enumerate(args.roots):
            shard = tmp_path / f"shard_{idx}.npz"
            shard_summary = tmp_path / f"shard_{idx}.json"
            shard_rejects = tmp_path / f"shard_{idx}_rejects.jsonl"
            try:
                result = assemble_fe_dataset(
                    root,
                    shard,
                    summary=shard_summary,
                    reject_log=shard_rejects,
                    impurities=_available_impurities(Path(root)),
                    beta=args.beta,
                    lamb=args.lamb,
                    eps=args.eps,
                    include_current=args.include_current,
                    quality_gate=True,
                    max_roundtrip=2e-1,
                    max_tau_abs=1e5,
                )
            except (FileNotFoundError, KeyError, ValueError) as exc:
                with reject_log.open("a") as handle:
                    handle.write(json.dumps({"source": root, "reasons": [str(exc)]}, sort_keys=True) + "\n")
                continue
            if shard_rejects.exists() and shard_rejects.read_text():
                with reject_log.open("a") as handle:
                    handle.write(shard_rejects.read_text().rstrip() + "\n")
            if result["n_labels"]:
                shards.append(shard)
                shard_summaries.append(result)

        if not shards:
            raise RuntimeError("No Fe labels survived assembly and quality gates")

        merged = _merge_npz(shards)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **merged)
    scratch_path = Path(args.scratch_copy)
    scratch_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(scratch_path, **merged)

    condition_summary = _condition_summary(merged)
    summary = {
        "output": str(output),
        "scratch_copy": str(scratch_path),
        "n_labels": int(merged["delta"].shape[0]),
        "delta_shape": list(merged["delta"].shape),
        "g_shape": list(merged["g"].shape),
        "beta_values": condition_summary["beta_values"],
        "temperature_kelvin_values": condition_summary["temperature_kelvin_values"],
        "U_range": condition_summary["U_range"],
        "J_range": condition_summary["J_range"],
        "dc_schemes": condition_summary["dc_schemes"],
        "source_counts": {
            str(src): int(np.sum(merged["source"].astype(str) == str(src)))
            for src in sorted(set(merged["source"].astype(str)))
        },
        "shards": shard_summaries,
        "reject_log": str(reject_log),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _merge_npz(paths: list[Path]) -> dict[str, np.ndarray]:
    loaded = [np.load(path, allow_pickle=True) for path in paths]
    first = loaded[0]
    merged = {}
    row_count = int(first["delta"].shape[0])
    for key in first.files:
        values = [np.asarray(data[key]) for data in loaded]
        if values[0].shape[:1] == (row_count,):
            merged[key] = np.concatenate(values, axis=0)
        else:
            for other in values[1:]:
                np.testing.assert_array_equal(values[0], other)
            merged[key] = values[0]
    for data in loaded:
        data.close()
    return merged


def _available_impurities(root: Path) -> tuple[int, ...]:
    impurities = []
    for path in root.glob("imp.*"):
        if path.is_dir() and path.name.split(".", 1)[1].isdigit():
            impurities.append(int(path.name.split(".", 1)[1]))
    if not impurities:
        raise ValueError(f"No imp.N directories found in {root}")
    return tuple(sorted(impurities))


def _condition_summary(data: dict[str, np.ndarray]) -> dict[str, object]:
    beta = np.asarray(data["beta"], dtype=float)
    U = np.asarray(data["U"], dtype=float)
    J = np.asarray(data["J"], dtype=float)
    info = [json.loads(str(row)) for row in data["info_json"]]
    return {
        "beta_values": sorted(float(x) for x in np.unique(beta)),
        "temperature_kelvin_values": sorted(float(11604.51812 / x) for x in np.unique(beta)),
        "U_range": [float(np.min(U)), float(np.max(U))],
        "J_range": [float(np.min(J)), float(np.max(J))],
        "dc_schemes": sorted(set(str(row.get("dc_scheme", "")) for row in info)),
    }


if __name__ == "__main__":
    main()
