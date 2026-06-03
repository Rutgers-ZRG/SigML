import os
import subprocess
import sys
import tempfile

import numpy as np
import torch


def test_train_cli_smoke_writes_checkpoint():
    n = 8
    with tempfile.TemporaryDirectory() as d:
        dataset = os.path.join(d, "data.npz")
        output = os.path.join(d, "ckpt.pth")
        U = np.linspace(1.0, 5.0, n)
        np.savez(
            dataset,
            delta=np.random.randn(n, 118).astype(np.float32),
            g=np.random.randn(n, 118).astype(np.float32),
            U=U.astype(np.float32),
            mu=(0.5 * U).astype(np.float32),
            beta=np.full(n, 70.0, dtype=np.float32),
            eps_d=np.zeros(n, dtype=np.float32),
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "sigml.solver.train",
                "--dataset",
                dataset,
                "--output",
                output,
                "--epochs",
                "1",
                "--batch-size",
                "4",
            ],
            check=True,
            cwd=os.getcwd(),
        )
        ckpt = torch.load(output, map_location="cpu")
        assert set(ckpt) == {"model_state_dict", "config", "loss_history"}
        assert len(ckpt["loss_history"]) == 1
