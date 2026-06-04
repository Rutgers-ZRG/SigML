# Stage 2 Code Review

Date: 2026-06-03

Scope reviewed: `git diff main..HEAD` on `feat/sigml-stage2-srvo3`, focused on
`sigml/solver/`, `scripts/solver/`, and `tests/solver/`, plus the available
Stage-2 docs and `~/.paseo/plans/sigml-stage2-realmaterials.md`.

Requested context docs note: `docs/stage2-srvo3-reference.md` and
`docs/stage2-research-equivariant-arch.md` are not present in this checkout at
those paths. I used the available `docs/stage2-srvo3-result.md`,
`docs/stage2-srvo3-nn-harness.md`, `docs/stage2-net-training.md`, and the Paseo
plan.

## Findings

### MAJOR: The stabilized SrVO3 "reproduces reference" loop is reference-anchored and near-circular if presented as an independent solve

The warm-started run is not just initialized near the answer. The harness loads
the reference `Sigma_freq_0` into the active self-energy before the loop starts
and keeps a copy as `sigma_anchor` (`sigml/solver/srvo3_nn_harness.py:173`).
Each NN/Dyson self-energy is then regularized with that reference as the
high-frequency tail source (`sigml/solver/srvo3_nn_harness.py:222`), and with
the documented `mix_sigma=0.02` the first update is:

```text
sigma = 0.98 * sigma_anchor + 0.02 * projected_sigma_new
```

from `sigml/solver/srvo3_nn_harness.py:232`. Later updates remain heavily
damped against the existing, reference-seeded self-energy
(`sigml/solver/srvo3_nn_harness.py:229`).

Scientific-integrity assessment:

- This does demonstrate useful infrastructure and a warm-started accelerator
  mechanism: the loop can call the sidecar, produce an NN `G(tau)`, convert it
  through DLR/Dyson, keep the lattice loop numerically alive, and stay close in
  primary tau-space observables when held near a known fixed point.
- It does not demonstrate an independent SrVO3 NN-DMFT solve. The successful
  fixed point inherits essential information from the reference self-energy
  both at initialization and at the high-frequency tail. The occupation and
  `G(tau)` agreement cannot be cleanly attributed to the NN solver.
- The cold BlockResNet loop and the e3nn loop both fail after one NN step
  because the self-energy feedback is nonphysical, so the independent-loop
  headline is currently negative.

`docs/stage2-srvo3-result.md` is honest enough for merge: it explicitly says
Stage-2 does not reproduce the reference with the current solver, that the
successful run relies on reference warm-start and tail projection, and that the
one-shot CTHYB refinement worsens the primary metric
(`docs/stage2-srvo3-result.md:7`, `docs/stage2-srvo3-result.md:15`,
`docs/stage2-srvo3-result.md:102`, `docs/stage2-srvo3-result.md:134`). It should
not be summarized elsewhere as "SrVO3 reproduced" without the reference-anchor
caveat.

### MAJOR: The real bottleneck is correctly diagnosed as self-energy reconstruction, but the current "causality rate" can overstate model readiness

The result doc's diagnosis is technically coherent: first-step BlockResNet
`G(tau)` is close to the reference, while the DLR/Dyson `Sigma(iw)` tail is
catastrophically non-causal and breaks `calc_mu`
(`docs/stage2-srvo3-result.md:71`). The code path matches the diagnosis:
`G_freq` is rebuilt from NN DLR coefficients, then `Sigma = G0^-1 - G^-1`
(`sigml/solver/srvo3_nn_harness.py:218`). The implemented projection clamps
positive-frequency diagonal imaginary parts, bounds large values, and blends the
tail to the reference when available (`sigml/solver/srvo3_nn_harness.py:403`).

The fix direction named in the result doc, a causal-tail or PSD-Lehmann output
head, is the right next architectural fix. Tau-space MSE alone is not enough for
stable DMFT feedback because small DLR/tau errors can be amplified by inversion
and high-frequency extrapolation.

The caveat is that the training comparison's `causality_rate` is computed on
predicted `G(iw)` from DLR coefficients, not on the downstream Dyson
`Sigma(iw)` (`scripts/solver/train_compare_stage2_real.py:294`). That metric is
not wrong as a Green's-function sanity check, but it is not the causality
property that killed the loop. The docs do distinguish this later, but future
tables should name it `G(iw) causality rate` or add a `Dyson-Sigma causality`
metric to avoid false confidence.

### MAJOR: The e3nn model is equivariant only for the represented real-symmetric subspace

The numerical equivariance test is sound for what the model implements: it
rotates a symmetric real block, runs both paths, and checks
`net(Q Delta Q^T) ~= Q net(Delta) Q^T`. That matches the `0e+2e` symmetric tensor
representation in `OrbitalIrrepNet` (`sigml/solver/net.py:186`).

The limitation is also real: the forward path discards the imaginary component
of the input block (`sigml/solver/net.py:215`) and emits real symmetric Hermitian
outputs (`sigml/solver/net.py:219`). That is acceptable for the diagonal/real
SrVO3 label campaign, but it is not a general MxM Hermitian solver and should
not be carried to off-diagonal or spin-orbit cases without adding the omitted
antisymmetric/imaginary channel.

The CV-versus-loop tension is adequately explained in
`docs/stage2-srvo3-result.md`: e3nn wins 148-label held-out `G(tau)` MSE, but its
first actual SrVO3 step is much worse and the loop fails. That is plausible
because the held-out labels measure warm-label interpolation, while the loop
requires stable feedback under a very specific lattice trajectory and scalar
convention.

### MINOR: Matrix Dyson API layout differs from the Stage-2 DLR tensor layout

`PydlrGrid` and the datasets use a last-axis convention for tau/frequency nodes
(`sigml/solver/pydlr_grid.py:22`), e.g. `(M, M, N_tau)`. The generic matrix
Dyson helper recognizes matrix blocks only when the final two axes are
`(M, M)` and expects `iw.shape == g.shape[:-2]`
(`sigml/solver/dyson.py:18`). The tests cover frequency-first arrays, and the
SrVO3 harness uses separate conversion helpers, so this is not breaking the
current result. It is a future footgun: passing `(M, M, N_iw)` into
`sigma_from_g` or `is_causal` will not use matrix inversion/eigenvalue semantics.

### MINOR: CTHYB labeler convention is correct for `delta_interface=True`, but the API relies on callers knowing `eps_d` means `eff_atomic_levels`

The labeler uses rotationally invariant Kanamori with
`U_matrix_kanamori(..., U, J)` and `h_int_kanamori(..., off_diag=True)`
(`sigml/solver/labeler.py:138`). For the solid_dmft delta-interface path,
`eps_d` is correctly treated as already shifted effective atomic levels and
passed directly into `Hloc_0`; `mu` is intentionally ignored
(`sigml/solver/labeler.py:202`). That matches the Stage-2 fix.

The risk is naming/API clarity: `solve(..., mu, eps_d)` looks like it accepts a
bare crystal-field `eps_d`, but in this path it must be `eff_atomic_levels`.
The local comment is good; a public docstring or parameter rename would reduce
future misuse.

### MINOR: Two requested review-context docs are absent from the repo

The branch contains `docs/stage2-srvo3-result.md`,
`docs/stage2-srvo3-nn-harness.md`, and `docs/stage2-net-training.md`, but not the
requested `docs/stage2-srvo3-reference.md` or
`docs/stage2-research-equivariant-arch.md`. If those are intended merge
artifacts, they should be added or the references should be updated.

## Physics And Implementation Notes

The matrix Dyson equation and causality check are correct in the tested
frequency-first layout: `Sigma = (iw + mu)I - eps_d - Delta - G^-1`, and
positive-frequency matrix causality is checked through eigenvalue imaginary
parts. The beta-40 `PydlrGrid` is also coherent: rank follows pydlr, matrix
blocks round-trip with last-axis tau/frequency nodes, and the committed
`eval_at_tau` sign is validated by the single-pole and projection tests.

The t2g bath sampler produces Hermitian causal baths by constructing PSD
orbital weights and adapting the DLR coefficient sign. Warm mode is correctly
centered on supplied trajectory samples. The campaign later constrained to
diagonal baths, so off-diagonal sampler coverage is more infrastructure than a
validated CTHYB production path.

The SrVO3 harness is good mechanism code: sidecar invocation, scalar convention
`(mu - avg(eps_d)) / U`, DFTTools extraction, density diagnostics, and output
artifacts are all present. The stabilization code is intentionally not an
independent physics solver.

Stage-1 and the old position-only/orb1 path appear intact. The `ValentiOrb1Grid`
was generalized to last-axis arrays while preserving scalar behavior, and the
full solver test suite passed.

## Test Quality

Meaningful coverage:

- Matrix Dyson diagonal and coupled-orbital cases.
- Matrix causality through eigenvalues.
- beta-40 DLR round-trips and `eval_at_tau`.
- sidecar NPZ schema and subprocess error propagation.
- Hermitian output construction for BlockResNet.
- e3nn numerical equivariance for the represented subspace.
- CTHYB labeler import/shape/convention guards and optional TRIQS smoke tests.
- SrVO3 harness stabilization and reference-tail regularizer mechanism tests.

Tautological or limited coverage:

- The SrVO3 harness tests do not validate physical convergence; they validate
  plumbing and regularization mechanics.
- The e3nn equivariance test proves architectural equivariance, not accuracy or
  stable DMFT feedback.
- Training smoke tests use synthetic relationships, so they catch regressions in
  training mechanics but not scientific quality.
- The key failure mode, Dyson-Sigma tail causality from NN outputs, is diagnosed
  in result artifacts rather than enforced as a CI gate.

## Test Run

Command run:

```bash
conda run -n nequip python -m pytest tests/solver/ -q
```

Result:

```text
62 passed, 2 skipped, 61 warnings in 9.09s
```

## Overall Verdict

Mergeable as an honest Stage-2 deliverable, provided the merge summary preserves
the caveat: this branch delivers the multi-orbital infrastructure, beta-40
SrVO3-warm label/training pipeline, NN sidecar loop, a warm-started/reference
anchored accelerator demonstration, and a sharp bottleneck diagnosis. It does
not deliver an independent SrVO3 NN-DMFT reproduction.

Top 3 issues:

1. The successful SrVO3 loop is near-circular if advertised without the
   reference warm-start and reference-tail anchor.
2. The next required physics fix is a self-energy-aware causal-tail or
   PSD-Lehmann output head; held-out `G(tau)` MSE and `G(iw)` causality are not
   sufficient.
3. The e3nn implementation/test are valid only for the real symmetric
   `0e+2e` subspace; it is not yet a general Hermitian orbital-equivariant
   solver.

The result doc's honesty is adequate. It states the negative standalone result,
the reference-anchored nature of the stabilized loop, the failed cold/e3nn
loops, the unphysical `Z`, and the degraded one-CTHYB refinement. The only
thing to guard against is a shorter downstream summary that drops those caveats.
