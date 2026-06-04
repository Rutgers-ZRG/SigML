# Fe eDMFT Parser

This parser converts the Haule eDMFT/Wien2k Fe run at
`/scratch/lz432/tscan_5000k_v3` into the Stage-2 block solver dataset schema.
The generated warm-label file is gitignored at `data/fe_tscan5000k_b232.npz`;
the copied scratch artifact is
`/scratch/lz432/sigml_fe_warmstart/fe_tscan5000k_b232.npz`.

## Source Conventions

The source is a beta `2.32` Fe 3d calculation with four inequivalent impurity
directories, `imp.0` through `imp.3`.  `params.dat` and `imp.N/PARAMS` provide
`U=5.0`, `J=0.93`, `beta=2.32`, and the impurity QMC chemical potential `mu`.
The double-counting scheme in `params.dat` is `nominal`.

`fetest7.indmfl` is treated as the correlated-subspace authority.  Each cix
block is a five-orbital real-harmonic d shell in this order:

```text
z^2, x^2-y^2, xz, yz, xy
```

The `Sigind` matrices are diagonal.  The parser therefore emits diagonal
`(5, 5, n_tau)` blocks and zeros all off-diagonal tau entries.

## Frequency And DLR Conversion

The eDMFT tables use positive fermionic Matsubara frequencies.  `Delta.inp*`,
`Gf.out*`, and `Sig.out*` contain frequency followed by real/imaginary pairs for
five diagonal components.  Outer `sig.inp*` files contain 20 components, five
per impurity, with the same impurity ordering as `indmfl`.

The assembler uses `PydlrGrid(beta=2.32, lamb=80.0, eps=1e-10)`, which has rank
`29`.  Dense eDMFT Matsubara rows are fit with pydlr's least-squares arbitrary
Matsubara fitter, then evaluated on the DLR tau nodes for `delta` and `g`.
Complex DLR coefficient arrays are also saved for diagnostics:

- `delta_dlr_coeffs`
- `g_dlr_coeffs`
- `sigma_dlr_coeffs` from matched `imp.N/Sig.out*`
- `siginp_dynamic_dlr_coeffs` from matched outer `sig.inp*`

The emitted training features are:

```text
x = flattened real/imag block-tau Delta + [U, mu_over_U, beta, J]
y = flattened real/imag block-tau G
```

`eps_d` stores the five `Ed` impurity levels from `PARAMS`, but the stored
feature scalar is `mu_over_U` to match the requested Fe warm-label schema.

## Self-Energy And Double Counting

`Sig.out*` is the solver self-energy stream in absolute eDMFT convention.  The
outer `sig.inp*` stream stores the dynamic part used by the eDMFT loop; its
header supplies:

- `s_oo`: high-frequency self-energy shift for each component.
- `Edc`: double-counting potential for each component.

For each label row, the parser slices the five components for that impurity and
saves them as `siginp_s_oo` and `siginp_Edc`.  For `tscan_5000k_v3`, `Edc` is
`25.175` for all 20 components and the double-counting scheme is `nominal`.

The later warm-anchor reconstruction should treat `sig.inp` as dynamic
self-energy plus `s_oo`, with `Edc` preserved separately.  Do not mix this
dataset with `exactd` or `FLL` Fe runs unless double counting is made explicit
in the model input or the datasets are separated.

## Current Artifact

The current assembled artifact contains 64 labels:

- four impurities
- 15 historical matched suffixes, `.1.1` through `.15.1`
- one current unversioned `Delta.inp`/`Gf.out` pair per impurity

Incomplete files are skipped by suffix intersection; for example
`imp.0/Delta.inp.16.1` has no matching `Gf.out.16.1` and is not included.

Validation metrics saved in `data/fe_tscan5000k_b232_summary.json`:

- `delta` and `g` shape: `(64, 5, 5, 29)`
- max positive-Matsubara causality imaginary eigenvalue for `G`: `-0.005899287981488611`
- max DLR fit residual for `Delta(iw)`: `4.121574635882797e-07`
- max DLR fit residual for `G(iw)`: `6.892515020047174e-09`

## Phase 5 Scaled Artifact

The scaled assembler script is `scripts/solver/assemble_fe_warm_v2.py`.  It can
merge multiple compatible eDMFT roots, records `source_iteration_group` for
grouped train/test splits, and writes a JSONL reject log.  The Phase 5 artifact
is gitignored at `data/fe_warm_v2.npz`, with a scratch copy on `an` at:

```text
/scratch/lz432/sigml_fe_warmstart/fe_warm_v2.npz
```

The accepted v2 dataset has `148` labels from `tscan_5000k_v3` and
`tscan_5000k_pert1`.  Both are beta `2.32`, `U=5.0`, `J=0.93`, nominal
double-counting, diagonal M=5 real-harmonic labels.  Additional beta-`2.32`
exactd roots were parsed but rejected after DLR conversion because their
`G(tau)` target magnitudes became nonphysical (`1e8-1e9`); FLL roots remain
excluded because double-counting convention is not encoded in the current model
input.
