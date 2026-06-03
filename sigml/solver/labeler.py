from __future__ import annotations

import numpy as np

from sigml.solver.oracle import Orb1Oracle


class OracleLabeler:
    """Local oracle-distillation labeler using Valenti's released orb1 model."""

    def __init__(self, oracle: Orb1Oracle | None = None):
        self.oracle = oracle if oracle is not None else Orb1Oracle()
        self.grid = self.oracle.grid

    def solve(
        self,
        delta_vec: np.ndarray,
        U: float,
        mu: float,
        beta: float,
        eps_d: float = 0.0,
    ) -> np.ndarray:
        return self.oracle.solve(delta_vec, U=U, mu=mu, beta=beta, eps_d=eps_d)


class CtsegLabeler:
    """Phase-C CTSEG labeler stub.

    The verified TRIQS/CTSEG wiring, Slurm template, and the important
    near-atomic bath note are documented in:
    /Users/li/dev/RA/dmft/docs/amarel3-triqs-setup.md

    CTSEG rejects the strict Delta=0 atomic limit, so Phase C must use a
    near-atomic bath for atomic validation.
    """

    def solve(
        self,
        delta_vec: np.ndarray,
        U: float,
        mu: float,
        beta: float,
        eps_d: float = 0.0,
    ) -> np.ndarray:
        raise NotImplementedError(
            "CTSEG labels are Phase C; see "
            "/Users/li/dev/RA/dmft/docs/amarel3-triqs-setup.md. "
            "CTSEG rejects strict Delta=0 atomic validation; use a near-atomic bath."
        )
