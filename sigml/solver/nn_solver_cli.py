from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sigml.solver.net import FeedforwardNet
from sigml.solver.nn_solver_schema import (
    solver_input_features,
    unflatten_block_features,
    write_solver_output,
)


class LinearCheckpointNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPCheckpointNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int]):
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        layers: list[nn.Module] = []
        for left, right in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(left, right))
            layers.append(nn.GELU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a SigML torch NN solver sidecar.")
    parser.add_argument("--in", dest="input_path", required=True, help="Input NPZ path.")
    parser.add_argument("--out", dest="output_path", required=True, help="Output NPZ path.")
    parser.add_argument("--ckpt", dest="checkpoint_path", required=True, help="Torch checkpoint path.")
    args = parser.parse_args(argv)

    features, metadata = solver_input_features(args.input_path)
    model = load_model(
        args.checkpoint_path,
        input_dim=int(features.size),
        output_dim=int(features.size - 4),
    )
    model.eval()

    with torch.no_grad():
        x = torch.from_numpy(features[None, :].astype(np.float32))
        y = model(x)

    if y.ndim != 2 or y.shape != (1, features.size - 4):
        raise ValueError(
            f"model output must have shape (1, {features.size - 4}), got {tuple(y.shape)}"
        )
    g_dlr = unflatten_block_features(
        y.detach().cpu().numpy()[0],
        orbital_dim=metadata.orbital_dim,
        n_tau=metadata.n_tau,
    )
    write_solver_output(args.output_path, g_dlr=g_dlr)
    return 0


def load_model(checkpoint_path: str | Path, *, input_dim: int, output_dim: int) -> nn.Module:
    path = Path(checkpoint_path)
    try:
        scripted = torch.jit.load(str(path), map_location="cpu")
        return scripted
    except Exception:
        pass

    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, nn.Module):
        return checkpoint
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), nn.Module):
        return checkpoint["model"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"unsupported checkpoint type {type(checkpoint)!r}")

    architecture = str(checkpoint.get("architecture", "feedforward"))
    state_dict = _extract_state_dict(checkpoint)
    if architecture == "linear":
        model = LinearCheckpointNet(
            input_dim=int(checkpoint.get("input_dim", input_dim)),
            output_dim=int(checkpoint.get("output_dim", output_dim)),
        )
        _load_state_dict(model.linear, state_dict)
        return model
    if architecture == "mlp":
        model = MLPCheckpointNet(
            input_dim=int(checkpoint.get("input_dim", input_dim)),
            output_dim=int(checkpoint.get("output_dim", output_dim)),
            hidden_dims=[int(x) for x in checkpoint.get("hidden_dims", [256, 256])],
        )
        _load_state_dict(model, state_dict)
        return model

    model = FeedforwardNet()
    _load_state_dict(model, state_dict)
    return model


def _extract_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint state_dict must be a dictionary")
    return state_dict


def _load_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        prefixed = {key.removeprefix("module."): value for key, value in state_dict.items()}
        model.load_state_dict(prefixed, strict=True)


if __name__ == "__main__":
    raise SystemExit(main())
