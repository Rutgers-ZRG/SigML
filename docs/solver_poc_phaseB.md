# Solver PoC Phase B

Local oracle-distillation plumbing run for Tasks 6-9.

## Artifacts

- Dataset: `data/solver_poc_distill.npz`
- Checkpoint: `SAVED_MODELS/solver_poc_distill.pth`

## Data

- Base kept: 1000 / 1197 attempts
- Base skipped: 197
- Augmentation kept: 1000 / 1021 attempts
- Augmentation skipped: 21
- Total samples: 2000

## Training

- Epochs: 3000
- Batch size: 256
- Learning rate: 0.001
- Device: mps
- Final train loss: 3.1533837e-06
- Final validation loss: 3.8831715e-05

## Held-Out Validation

- Held-out samples: 400
- Student-vs-oracle `g_mse`: 3.8831713e-05
- Predicted-Sigma causality rate: 0.997500

## Notes

Labels are from the released Valenti `orb1` oracle, not CTSEG. This validates the local data and training pipeline only.
Augmentation uses damped one-to-two-step Bethe updates and skips any non-finite or acausal sample.
