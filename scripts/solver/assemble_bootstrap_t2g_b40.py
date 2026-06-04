#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble SrVO3-warm beta=40 t2g CTHYB shards.")
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(Path(args.label_dir).glob("bootstrap_t2g_b40_*.npz"))
    if not files:
        raise FileNotFoundError(f"No bootstrap_t2g_b40_*.npz files found in {args.label_dir}")
    rows = {key: [] for key in ("delta", "g", "U", "mu", "beta", "J", "eps_d")}
    source: list[str] = []
    infos: list[str] = []
    elapsed: list[float] = []
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            for key in rows:
                rows[key].append(np.asarray(data[key]))
            source.extend([str(x) for x in data["source"]])
            infos.extend([str(x) for x in data["info_json"]])
            elapsed.append(float(np.asarray(data["elapsed_seconds"])[0]))

    assembled = {key: np.concatenate(value, axis=0) for key, value in rows.items()}
    finite = np.all(np.isfinite(assembled["delta"]), axis=(1, 2, 3)) & np.all(
        np.isfinite(assembled["g"]), axis=(1, 2, 3)
    )
    for key in assembled:
        assembled[key] = assembled[key][finite]
    source_arr = np.asarray(source)[finite]
    info_arr = np.asarray(infos)[finite]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **assembled, source=source_arr, info_json=info_arr)

    occ = []
    signs = []
    orders = []
    for raw in info_arr:
        info = json.loads(str(raw))
        cthyb = info.get("cthyb", {})
        if cthyb.get("orbital_occupations") is not None:
            occ.append(cthyb["orbital_occupations"])
        if cthyb.get("average_sign") is not None:
            signs.append(cthyb["average_sign"])
        if cthyb.get("average_order") is not None:
            orders.append(cthyb["average_order"])
    occ_arr = np.asarray(occ, dtype=float) if occ else np.empty((0, 3))
    summary = {
        "files": [str(path) for path in files],
        "n_rows": int(assembled["delta"].shape[0]),
        "n_dropped_nonfinite": int(len(finite) - int(np.sum(finite))),
        "shape_delta": list(assembled["delta"].shape),
        "shape_g": list(assembled["g"].shape),
        "source_counts": {x: int(np.sum(source_arr == x)) for x in sorted(set(source_arr))},
        "nonfinite_delta_entries": int(np.sum(~np.isfinite(assembled["delta"]))),
        "nonfinite_g_entries": int(np.sum(~np.isfinite(assembled["g"]))),
        "occupation_mean": None if occ_arr.size == 0 else np.mean(occ_arr, axis=0).tolist(),
        "occupation_min": None if occ_arr.size == 0 else np.min(occ_arr, axis=0).tolist(),
        "occupation_max": None if occ_arr.size == 0 else np.max(occ_arr, axis=0).tolist(),
        "average_sign_mean": None if not signs else float(np.mean(signs)),
        "average_order_mean": None if not orders else float(np.mean(orders)),
        "mean_task_elapsed_seconds": None if not elapsed else float(np.mean(elapsed)),
        "info_examples": [json.loads(str(x)) for x in info_arr[: min(3, len(info_arr))]],
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output} rows={summary['n_rows']}")


if __name__ == "__main__":
    main()
