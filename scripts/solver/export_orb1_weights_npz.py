#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Valenti orb1 PyTorch weights to NumPy.")
    parser.add_argument(
        "--checkpoint",
        default="/Users/li/dev/RA/dmft/ref/mlDMFT/mldmft/models/orb1/save_3000.pth",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")
    arrays = {key: value.detach().cpu().numpy() for key, value in state.items()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)
    print(f"wrote {output} ({len(arrays)} arrays)")


if __name__ == "__main__":
    main()
