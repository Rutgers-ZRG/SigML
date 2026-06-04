# Stage-2 Real-Label Net Training

## Data

First real SrVO3-warm t2g CTHYB bootstrap labels:

- Dataset: `data/bootstrap_t2g_b40.npz`
- Rows: 28 physically sane labels
- Grid: beta=40, pydlr lambda=600, eps=1e-10, rank=47
- Block schema: `delta/g` as complex DLR coefficients with shape `(N, 3, 3, 47)`
- Model schema: `x` shape `(N, 850)`, `y` shape `(N, 846)`, scalar channels `U, mu_over_U, beta, J`

The dataset is too small for architecture selection by raw MSE alone. Treat these numbers as a pipeline validation and first signal before retraining on the larger bootstrap set.

## Command

```bash
PYTHONWARNINGS='ignore::UserWarning' conda run -n nequip python scripts/solver/train_compare_stage2_real.py \
  --dataset data/bootstrap_t2g_b40.npz \
  --out-dir SAVED_MODELS/stage2_real_b40 \
  --results-json SAVED_MODELS/stage2_real_b40/results.json
```

Defaults used here: 7-fold CV, 500 epochs, batch size 8, AdamW lr=1e-3, weight_decay=1e-4. The MLP baseline is Phase-10 `BlockResNet` with mixed rotation/permutation augmentation. The equivariant model is Phase-11 `OrbitalIrrepNet`.

## Results

Primary metric is held-out G(tau) MSE after decoding predicted block-DLR coefficients through the beta=40 pydlr grid.

| Net | Train rows per fold | Held-out G(tau) MSE | Block-DLR MSE | Causality rate |
| --- | ---: | ---: | ---: | ---: |
| BlockResNet + aug | half | 3.1388e-4 +/- 8.7397e-5 | 6.7203e-5 +/- 4.6446e-6 | 1.000 +/- 0.000 |
| BlockResNet + aug | all | 3.0974e-4 +/- 6.9146e-5 | 6.5240e-5 +/- 4.0341e-6 | 1.000 +/- 0.000 |
| e3nn irrep | half | 3.7500e-4 +/- 1.8570e-4 | 5.9321e-5 +/- 6.2505e-6 | 1.000 +/- 0.000 |
| e3nn irrep | all | 3.8728e-4 +/- 1.5747e-4 | 5.9055e-5 +/- 1.0858e-5 | 1.000 +/- 0.000 |

On these 28 labels, BlockResNet+augmentation has the better raw held-out G(tau) MSE when using all available training rows in each fold: `3.10e-4` vs `3.87e-4` for e3nn.

Data-efficiency signal:

- BlockResNet+aug half/all G(tau) MSE ratio: `1.013`
- e3nn half/all G(tau) MSE ratio: `0.968`

The e3nn model degraded less under the half-data condition, but the 28-label spread is large and the half-data e3nn mean slightly beating its all-data mean is a noise warning, not a reliable monotonic learning curve.

## Checkpoints

Saved under `SAVED_MODELS/stage2_real_b40/`:

- All-row BlockResNet+aug checkpoint: `SAVED_MODELS/stage2_real_b40/block-resnet-aug_all_rows.pt`
- All-row e3nn checkpoint: `SAVED_MODELS/stage2_real_b40/e3nn-irrep_all_rows.pt`
- Fold checkpoints: `SAVED_MODELS/stage2_real_b40/{block-resnet-aug,e3nn-irrep}_frac{0.5,1}_fold*.pt`
- Full metrics JSON: `SAVED_MODELS/stage2_real_b40/results.json`

## Retrain On Larger Bootstrap

When the larger beta=40 SrVO3-warm set lands, run:

```bash
PYTHONWARNINGS='ignore::UserWarning' conda run -n nequip python scripts/solver/train_compare_stage2_real.py \
  --dataset data/bootstrap_t2g_b40_150.npz \
  --out-dir SAVED_MODELS/stage2_real_b40_150 \
  --results-json SAVED_MODELS/stage2_real_b40_150/results.json \
  --folds 5 \
  --epochs 800
```

Use the actual dataset path if the assembled larger file has a different name. If the larger campaign uses a different pydlr rank, pass the matching `--grid-lamb` and `--grid-eps`.

## Caveats

- N=28 is tiny; both nets can overfit and fold assignment can dominate the reported mean.
- These labels are warm-neighborhood SrVO3 labels, not a generic t2g impurity distribution.
- The real bootstrap uses stored `x/y` with `mu_over_U`; the dataset loader now preserves those stored features rather than recomputing `(mu - eps_d)/U`.
- Causality here is checked on predicted G(iw) eigenvalue imaginary parts at positive Matsubara frequencies.
