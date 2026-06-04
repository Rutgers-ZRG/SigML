# Stage 2 Phase 13 SrVO3 NN Harness

Date: 2026-06-03

## Scope

This is the Phase 13 de-risking harness, not the final physics run. It runs a
small SrVO3 `SumkDFT` loop from the public/tutorial `svo.h5` archive and replaces
the impurity solve with the existing SigML torch sidecar contract:

```text
TRIQS/DFTTools loop
  -> BlockGf Weiss field
  -> block-DLR Delta tensor + scalars NPZ
  -> conda base torch sidecar CLI
  -> block-DLR G tensor NPZ
  -> BlockGf G/Sigma by Dyson
  -> next SumkDFT iteration
```

The default checkpoint is a deterministic untrained Phase-10 `BlockResNet` stub.
Its output is regularized only enough to keep the Dyson inversion moving. The
observables are useful for proving the mechanism, not for validating SrVO3
physics.

## Entry Point

Run in the TRIQS loop env:

```bash
source /home/lz432/miniconda3/etc/profile.d/conda.sh
conda activate solid_dmft
python scripts/solver/run_srvo3_nn_harness.py \
  --h5 /scratch/lz432/sigml_stage2/srvo3_reference/run_public_config_64c_refined_40M/out/svo.h5 \
  --out-dir /scratch/lz432/sigml_stage2/srvo3_nn_harness/stub_run \
  --repo-root /path/to/SigML \
  --iterations 3
```

The torch sidecar defaults to:

```text
conda run -n base python sigml/solver/nn_solver_cli.py
```

Use `--python-command ...` to override that prefix.

## Trained Net Slot

The trained solver replaces only the checkpoint argument:

```bash
--ckpt /scratch/lz432/sigml_stage2/models/selected_phase12_solver.pt
```

The selected checkpoint must be loadable by `sigml/solver/nn_solver_cli.py` and
must use the same sidecar schema: input `delta_dlr` shape `(3, 3, N_tau)` plus
scalars `(U, mu_over_U, beta, J)`, output `g_dlr` shape `(3, 3, N_tau)`.

## One-CTHYB Refinement Slot

After Phase 12 selects the trained NN, the one-CTHYB refinement plugs in after
the NN loop has produced the final `Delta/G/Sigma` state:

1. Reuse the final `delta` and `eps_d` arrays from
   `srvo3_nn_harness_observables.npz`.
2. Call the existing `CthybT2GLabeler` once in the `solid_dmft` env using the
   same beta-40 grid and SrVO3 `U=2.0`, `J=0.65`.
3. Replace the final NN `G` with the CTHYB-refined `G`, recompute
   `Sigma = G0^{-1} - G^{-1}`, then write the final observables for the Phase 14
   comparison.

That refinement is deliberately not part of the stub harness acceptance because
the Phase 13 harness goal is to prove the loop-to-sidecar mechanism without
spending CTHYB cycles.
