# Fe M=5 Warm-Start Result

Date: 2026-06-04

## Phase 5 Setup

Dataset: `data/fe_warm_v2.npz` (gitignored)

- Fe 3d, `M=5`, diagonal real-harmonic order: `z^2, x^2-y^2, xz, yz, xy`
- `N=148` accepted labels after physical quality gates
- Sources: `tscan_5000k_v3` (`64` labels) and `tscan_5000k_pert1` (`84` labels)
- Conditions: beta `2.32` (`~5002 K`), `U=5.0`, `J=0.93`, nominal double counting
- Input: block-DLR `Delta(tau)` with shape `(5, 5, 29)` plus scalars `[U, mu/U, beta, J]`
- Target: block-DLR `G(tau)` with shape `(5, 5, 29)`

The assembled scratch copy is:

```text
/scratch/lz432/sigml_fe_warmstart/fe_warm_v2.npz
```

The checkpoint is:

```text
checkpoints/fe_m5_block_mlp_warm_v2.pt
```

Tracked summaries:

```text
results/fe_m5_warm_v2_summary.json
data/fe_warm_v2_summary.json       # gitignored
data/fe_warm_v2_rejects.jsonl      # gitignored
```

## Data Scaling Attempt

I searched and staged additional existing eDMFT Fe roots on `an`, including
`Fe_fcc`, `Fe_fcc_disp_P1`, `tscan_5000k_pert1`, and the `fetest*` family.  The
only roots that passed the beta-`2.32`, M=5, causal-`G`, sane-occupation, and
sane-`G(tau)` gates were the two nominal tscan roots above.

Rejected beta-`2.32` exactd roots had causal `G(iw)` and reasonable header `nf`,
but their DLR-converted `G(tau)` labels were nonphysical, with target magnitudes
up to `1e8-1e9`.  Changing DLR `lambda` from `80` to `400` did not fix those
tau labels.  FLL roots were not mixed because the current model input does not
encode double-counting convention.

Result: the requested `500-1000` physically usable labels were not available
from the compatible existing roots under the current DLR conversion.  The usable
increase is `64 -> 148` labels.

## Cross-Validation

Validation split: four grouped folds by `source_iteration_group`, so held-out
iterations from the same source trajectory do not leak across train/test.

Model: plain input-normalized `BlockMLP`, no d-shell permutation augmentation.

| Dataset | Labels | Held-out diag `G(tau)` MSE | Held-out all-block `G(tau)` MSE | `G(iw)` causality rate | Max positive Im eig `G(iw)` |
|---|---:|---:|---:|---:|---:|
| Phase 3+4 | 64 | `5.250989e5` | `1.050206e5` | `0.0` | `136.47` |
| Phase 5 v2 | 148 | `5.528448e5` | `1.105692e5` | `0.0` | `179.40` |

The v2 final all-label training loss is `47.19`, so the model can still fit the
training labels.  Held-out performance remains near the bad 64-label result.

## Reproduce eDMFT Reference

Verdict: **still underfit / no**.  The scaled warm M=5 Fe net does not reproduce
the beta-`2.32` eDMFT reference on held-out iterations.

Evidence:

- held-out diagonal `G(tau)` MSE: `5.528448e5`
- held-out all-block `G(tau)` MSE: `1.105692e5`
- held-out `G(iw)` causality rate: `0.0`
- held-out max positive imaginary eigenvalue for predicted `G(iw)`: `179.40`
- occupation endpoint MAE remains large (`368.14`) and predicted endpoints are
  not physically reliable

## Data Sufficiency

The Fe M=5 warm net remains data-limited.  The added 84 usable labels did not
move the held-out metric below the previous 64-label result.  The concrete next
target is at least `1000` physically sane beta-`2.32`, nominal-convention labels
in this diagonal representation, or a separate parser/model fix that makes the
existing exactd `nom=100` roots produce sane `G(tau)` labels without DLR ringing.
