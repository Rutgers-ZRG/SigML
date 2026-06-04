# Stage 2 SrVO3 Result

Date: 2026-06-03

## Verdict

Stage-2 does **not** reproduce the SrVO3 DFT+DMFT reference with the current
148-label warm-region solver. The trained nets can fit held-out warm CTHYB
labels at `~3e-4` G(tau) MSE, but neither net provides a stable NN-driven
SrVO3 DMFT fixed point. Both NN loops fail on the second chemical-potential
solve, before a converged NN Delta exists. Therefore the documented
`NN + 1 CTHYB` refinement was **not submitted**: the required input is a
converged NN Delta, and refining a bracket-failing intermediate state would not
test the advertised workflow.

## Training Reconfirmation

Dataset: `data/warm_t2g_b40_v2.npz`, 148 beta-40 SrVO3-warm CTHYB labels.
Command: `conda run -n nequip python scripts/solver/train_compare_stage2_real.py
--dataset data/warm_t2g_b40_v2.npz --folds 5 --epochs 800 --train-fractions 1.0`.

| Net | 5-fold held-out G(tau) MSE | DLR MSE | Causality rate | Final checkpoint |
| --- | ---: | ---: | ---: | --- |
| BlockResNet + augmentation | `3.2617e-4 +/- 3.5712e-5` | `6.1126e-5` | `1.000` | `SAVED_MODELS/stage2_real_b40_v2_800e_5fold/block-resnet-aug_all_rows.pt` |
| e3nn irrep | `2.7290e-4 +/- 4.2854e-5` | `5.2837e-5` | `1.000` | `SAVED_MODELS/stage2_real_b40_v2_800e_5fold/e3nn-irrep_all_rows.pt` |

The label-CV metric alone would pick e3nn, contrary to the earlier 28-label
expectation. In the actual SrVO3 loop, e3nn is not usable: its first NN step is
far from the reference and iteration 2 fails `calc_mu`.

## Reference Target

Reference source:
`/scratch/lz432/sigml_stage2/srvo3_reference/run_public_config_64c_refined_40M`.
This is the public solid_dmft SrVO3 tutorial continued to iteration 12.

| Quantity | Reference |
| --- | ---: |
| Occupation per orbital, spin-summed | `[0.333300, 0.333923, 0.337404]` |
| Total occupation | `1.004627` |
| Z per orbital | `[0.860771, 0.860272, 0.864546]` |
| Reference DMFT iterations | `12` |

## SrVO3 NN Loop Runs

The harness used the public `svo.h5`, the selected all-row checkpoints, and the
torch sidecar in the remote `nequip` environment. The final harness fix passes
the same scalar convention used in training, `(mu - avg(eps_d)) / U`.

| Run | Outcome | Iterations completed | Failure |
| --- | --- | ---: | --- |
| e3nn irrep | Failed | `1` | iteration-2 `calc_mu` could not reach density 1.0 after 101 dichotomy iterations |
| BlockResNet + augmentation | Failed | `1` | iteration-2 `calc_mu` could not reach density 1.0 after 101 dichotomy iterations |

First-step NN prediction versus reference:

| Run | G(tau) diag MSE | G(tau) diag MAE | Occupation, spin-summed | Total occ error |
| --- | ---: | ---: | ---: | ---: |
| e3nn first step | `3.1907e-3` | `4.1542e-2` | `[0.210461, 0.210461, 0.210461]` | `-0.373244` |
| BlockResNet first step | `3.5937e-5` | `4.6806e-3` | `[0.351149, 0.349013, 0.332233]` | `+0.027768` |

BlockResNet is close on the first-step primary observables, but its
self-consistent Dyson feedback is not stable enough to continue the DFT+DMFT
loop. e3nn has better warm-label CV MSE but worse SrVO3 first-step behavior.

## NN + 1 CTHYB Refinement

No refinement job was submitted. The documented refinement slot requires the
converged NN-loop Delta and eps_d. Both NN loops failed before convergence, so
there is no valid Delta for the one-CTHYB refinement.

Slurm job IDs: none.

Core-hours for this headline refinement phase: `0` CTHYB core-hours.

## Evidence Paths

Remote first-step comparison:
`/scratch/lz432/sigml_stage2/srvo3_stage2_headline/first_step_comparison.json`.

Remote NN-loop attempts:

- `/scratch/lz432/sigml_stage2/srvo3_stage2_headline/e3nn_nn_loop`
- `/scratch/lz432/sigml_stage2/srvo3_stage2_headline/blockresnet_nn_loop_v3_scalarfix`

Local training results:
`SAVED_MODELS/stage2_real_b40_v2_800e_5fold/results.json`.

## Interpretation

This is a warm-region accelerator trained on 148 CTHYB labels, not a production
three-orbital solver. The current data/model are sufficient to learn the local
label distribution but insufficient to close a stable SrVO3 DFT+DMFT loop. The
primary failure is not low-frequency Dyson-Sigma mismatch; it is a direct
self-consistency failure visible in the chemical-potential solve and primary
occupation/G(tau) observables.
