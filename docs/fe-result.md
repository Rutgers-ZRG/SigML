# Fe M=5 Warm-Start Result

Date: 2026-06-04

## Setup

Dataset: `data/fe_tscan5000k_b232.npz`

- Fe 3d, `M=5`, diagonal real-harmonic order: `z^2, x^2-y^2, xz, yz, xy`
- `N=64` labels from `tscan_5000k_v3`
- `beta=2.32`, `U=5.0`, `J=0.93`
- Input: block-DLR `Delta(tau)` with shape `(5, 5, 29)` plus scalars `[U, mu/U, beta, J]`
- Target: block-DLR `G(tau)` with shape `(5, 5, 29)`

Model: plain input-normalized `BlockMLP`, not the t2g `l=1` SO(3) augmentation path.
The primary run used no augmentation.  I also ran the O_h-compatible control
`d-shell-permutation`, which only permutes within eg `{z^2, x^2-y^2}` and t2g
`{xz, yz, xy}` subspaces; it did not improve held-out accuracy.

Primary checkpoint, gitignored:

```text
checkpoints/fe_m5_block_mlp_b232_noaug.pt
```

Tracked summaries:

```text
results/fe_m5_warmstart_noaug_summary.json
results/fe_m5_warmstart_summary.json
```

## Cross-Validation

Validation split: 4 grouped folds by eDMFT iteration suffix.  Each fold holds
out four eDMFT iteration suffixes across impurities, so validation is on held-out
Fe iterations rather than random rows from the same iteration.

Primary metric: diagonal `G(tau)` MSE on held-out rows.

| Run | Augmentation | Held-out diag `G(tau)` MSE | Held-out all-block `G(tau)` MSE | `G(iw)` causality rate | Max positive Im eig `G(iw)` |
|---|---:|---:|---:|---:|---:|
| Primary | none | `5.250989e5` | `1.050206e5` | `0.0` | `136.47` |
| Control | d-shell permutation | `5.280576e5` | `1.056121e5` | `0.0` | `191.64` |

Baselines for scale:

- zero prediction diagonal `G(tau)` MSE: `5.302332e5`
- mean-label prediction diagonal `G(tau)` MSE: `5.135838e5`

The primary no-augmentation model can memorize the small dataset in-sample:
final all-label checkpoint diagonal `G(tau)` MSE is about `5.66e2`, with
all-block `G(tau)` MSE about `2.30e2`.  Held-out performance remains near the
zero/mean baseline, so the 64-label model does not generalize across Fe eDMFT
iterations.

## Reproduce eDMFT Reference

Verdict: **No.**  This first Fe warm net does not reproduce the beta `2.32`
eDMFT reference on held-out iterations.

Evidence from the primary no-augmentation run:

- held-out diagonal `G(tau)` MSE: `5.250989e5`
- held-out all-block `G(tau)` MSE: `1.050206e5`
- held-out `G(iw)` causality rate: `0.0`
- held-out max positive imaginary eigenvalue for predicted `G(iw)`: `136.47`
- reference eDMFT `G(iw)` from stored DLR coefficients is causal; parser
  summary max imaginary eigenvalue is `-0.005899`

Occupations:

- eDMFT `Gf.out` header `nf` is available in the dataset: mean `6.4026`,
  range `6.3622` to `6.4151`
- a naive `-G(tau_last)` occupation estimate from the net is not reliable in
  this first cut because the held-out predicted DLR tau endpoint is nonphysical;
  the summary reports endpoint occupation MAE `360.0`

Self-energy:

- Dyson-implied Sigma from held-out predicted `G` is noncausal:
  `sigma_causality_rate = 0.0`
- held-out implied Sigma does not qualitatively match the eDMFT reference;
  the predicted positive-imaginary excursions and unstable endpoint occupations
  are already sufficient to reject the warm-net reproduction
- `sig.inp`/`Sig.out` convention comparisons are retained in the JSON summary,
  but the project metric remains `G(tau)`, not low-frequency Sigma

## Data Sufficiency

The 64 Fe labels are not enough for an `M=5` warm net.  The model has enough
capacity to fit the training labels but does not interpolate to held-out eDMFT
iterations.  This is an overfit / coverage failure, not evidence that the parser
or M-parametrized MLP path cannot work.

Recommended next label target: parse at least `500-1000` Fe labels before
claiming a warm Fe solver result.  That is roughly `8-16x` the current data and
is still modest relative to the input/output dimension (`1454 -> 1450`).  If the
available Fe runs have distinct double-counting or thermodynamic settings, keep
those runs separated or add explicit convention/condition channels before
mixing them.
