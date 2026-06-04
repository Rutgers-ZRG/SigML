# Stage 2 SrVO3 Result

Date: 2026-06-03

## Verdict

Stage-2 still does **not** reproduce the SrVO3 DFT+DMFT reference with the
current 148-label warm-region solver, but the bottleneck is now sharp. The
BlockResNet+augmentation net is accurate in the first SrVO3 impurity step, but
the DLR/Dyson self-energy generated from the NN `G(tau)` has a catastrophically
wrong Matsubara tail. Cold feedback from that `Sigma(iw)` makes the lattice
density negative over the `calc_mu` search range, so the original loop dies at
iteration 2.

A stabilized loop can be made to converge by warm-starting from the
near-converged SrVO3 reference `Sigma`, applying very heavy `Sigma` damping
(`mix_sigma=0.02`), and projecting the NN/Dyson self-energy onto a bounded
causal high-frequency tail anchored to the reference. That produces a stable
three-iteration NN fixed point close to the reference in occupation and
`G(tau)`. However, the documented one-shot CTHYB refinement from that fixed
point does **not** improve the headline: it worsens the primary `G(tau)` error
and underfills the first orbital.

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

| Run | Outcome | Iterations completed | Failure / note |
| --- | --- | ---: | --- |
| e3nn irrep | Failed | `1` | iteration-2 `calc_mu` could not reach density 1.0 after 101 dichotomy iterations |
| BlockResNet + augmentation, cold | Failed | `1` | iteration-2 `calc_mu` could not reach density 1.0 after 101 dichotomy iterations |
| BlockResNet + augmentation, warm+damped+projected | Converged | `3` | `G`-delta MSE `9.93e-9`; this relies on reference warm-start and tail projection |

First-step NN prediction versus reference:

| Run | G(tau) diag MSE | G(tau) diag MAE | Occupation, spin-summed | Total occ error |
| --- | ---: | ---: | ---: | ---: |
| e3nn first step | `3.1907e-3` | `4.1542e-2` | `[0.210461, 0.210461, 0.210461]` | `-0.373244` |
| BlockResNet first step | `3.5937e-5` | `4.6806e-3` | `[0.351149, 0.349013, 0.332233]` | `+0.027768` |

BlockResNet is close on the first-step primary observables, but the direct
Dyson self-energy is unusable without projection. In the failed cold run,
all positive Matsubara diagonal self-energy points violated causality:
`Im Sigma_diag(iw_n>0) > 0` for `1025/1025` points per orbital, with high-frequency
mean imaginary tail about `+310i` and max diagonal magnitude about `322`. With
that self-energy, a direct `total_density(mu)` scan from `mu0-20` to `mu0+20`
returned densities from `-6.04` to `-0.048`, never crossing the required density
`1.0`; the subsequent `calc_mu` expansion up to `mu ~= 62.8` still stayed
negative (`-1.95` to `-0.013`). The failure is therefore the NN/DLR/Dyson
`Sigma(iw)` feedback, not the first-step `G(tau)` accuracy.

Stabilized BlockResNet loop versus reference:

| Stage | Iterations | G(tau) diag MSE | G(tau) diag MAE | Occupation, spin-summed | Total occ error | Z estimate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NN loop, warm+damped+projected | `3` | `2.9173e-5` | `3.9614e-3` | `[0.344331, 0.348689, 0.330981]` | `+0.019374` | `[-0.909243, -0.909160, -0.922044]` |

The `Z` estimate is not physically meaningful for the projected NN loop; it is
included only as a diagnostic of the remaining self-energy pathology.

## NN + 1 CTHYB Refinement

The documented one-shot CTHYB refinement was run from the stabilized NN-loop
Delta and eps_d with 300k cycles, 8000 warmup cycles, `length_cycle=120`, 64 MPI
ranks on Slurm `main-redhat`.

| Stage | G(tau) diag MSE | G(tau) diag MAE | Occupation, spin-summed | Total occ error | CTHYB average sign/order |
| --- | ---: | ---: | ---: | ---: | ---: |
| NN loop, warm+damped+projected | `2.9173e-5` | `3.9614e-3` | `[0.344331, 0.348689, 0.330981]` | `+0.019374` | n/a |
| NN + 1 CTHYB | `1.4706e-4` | `9.3691e-3` | `[0.285502, 0.352303, 0.341142]` | `-0.025679` | `1.0 / 32.04634` |

The one-shot CTHYB solve does not reproduce SrVO3 within the Stage-2 tolerance:
it degrades the primary `G(tau)` metric and shifts the first orbital occupation
too far below the reference.

Slurm job IDs:

- `55496277`: one-shot SrVO3 NN+1 CTHYB refinement, `COMPLETED`, elapsed
  `00:02:32`, 64 allocated CPUs, Slurm `TotalCPU=02:38:05`.

Core-hours for this headline refinement phase: about `2.70` allocated
core-hours (`64 * 152 s / 3600`), under the `~50` core-hour bound.

## Evidence Paths

Remote first-step comparison:
`/scratch/lz432/sigml_stage2/srvo3_stage2_headline/first_step_comparison.json`.

Remote NN-loop attempts:

- `/scratch/lz432/sigml_stage2/srvo3_stage2_headline/e3nn_nn_loop`
- `/scratch/lz432/sigml_stage2/srvo3_stage2_headline/blockresnet_nn_loop_v3_scalarfix`
- `/scratch/lz432/sigml_stage2/srvo3_stage2_headline/blockresnet_warm_tail_mix002`
- `/scratch/lz432/sigml_stage2/srvo3_stage2_headline/blockresnet_warm_tail_mix002_cthyb`

Remote comparison:
`/scratch/lz432/sigml_stage2/srvo3_stage2_headline/blockresnet_warm_tail_mix002_cthyb/srvo3_comparison_with_cthyb.json`.

Local training results:
`SAVED_MODELS/stage2_real_b40_v2_800e_5fold/results.json`.

## Interpretation

This is a warm-region accelerator trained on 148 CTHYB labels, not a production
three-orbital solver. The current data/model are sufficient to learn the local
`G(tau)` label distribution and can be held near a SrVO3 fixed point, but the
loop is not an honest standalone NN-DMFT accelerator yet. The remaining
bottleneck is the self-energy reconstruction used for lattice feedback:
tau-space accuracy does not imply a causal, well-tailed `Sigma(iw)` after DLR
projection and Dyson inversion. The one-CTHYB refinement result confirms that
stabilizing the loop by reference-anchored projection is not enough to recover
the advertised SrVO3 headline.
