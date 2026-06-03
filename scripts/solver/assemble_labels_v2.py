#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.solver.v2_common import metric_summary, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble v2 CTSEG label shards into one training dataset.")
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(Path(args.label_dir).glob("labels_v2_*.npz"))
    if not files:
        raise FileNotFoundError(f"No labels_v2_*.npz files found in {args.label_dir}")

    rows: dict[str, list[np.ndarray]] = defaultdict(list)
    sources = []
    rep_keys = []
    infos = []
    elapsed = []
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            for key in ("delta", "g", "U", "mu", "beta", "eps_d"):
                rows[key].append(np.asarray(data[key]))
            sources.extend([str(x) for x in data["source"]])
            rep_keys.extend([str(x) for x in data["replicate_key"]])
            infos.extend([str(x) for x in data["info_json"]])
            if "elapsed_seconds" in data:
                elapsed.append(float(np.asarray(data["elapsed_seconds"])[0]))

    assembled = {key: np.concatenate(value, axis=0) for key, value in rows.items()}
    finite = np.all(np.isfinite(assembled["delta"]), axis=1) & np.all(np.isfinite(assembled["g"]), axis=1)
    for key in assembled:
        assembled[key] = assembled[key][finite]
    sources_arr = np.asarray(sources)[finite]
    rep_arr = np.asarray(rep_keys)[finite]
    info_arr = np.asarray(infos)[finite]

    noise = []
    for key in sorted(set(rep_arr)):
        idx = np.flatnonzero(rep_arr == key)
        if len(idx) >= 2:
            noise.append(
                {
                    "replicate_key": key,
                    "n": int(len(idx)),
                    "g_vec": metric_summary(assembled["g"][idx[0]], assembled["g"][idx[1]]),
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        **assembled,
        source=sources_arr,
        replicate_key=rep_arr,
        info_json=info_arr,
    )
    source_counts = {source: int(np.sum(sources_arr == source)) for source in sorted(set(sources_arr))}
    summary = {
        "files": [str(path) for path in files],
        "n_rows": int(assembled["delta"].shape[0]),
        "n_finite": int(np.sum(finite)),
        "n_dropped_nonfinite": int(len(finite) - np.sum(finite)),
        "source_counts": source_counts,
        "noise_replicates": noise,
        "mean_task_elapsed_seconds": float(np.mean(elapsed)) if elapsed else None,
        "raw_task_elapsed_seconds": elapsed,
        "info_examples": [json.loads(info) for info in info_arr[: min(3, len(info_arr))]],
    }
    write_json(args.summary, summary)
    print(f"wrote {output} rows={summary['n_rows']}")


if __name__ == "__main__":
    main()
