# SigML NN Solver PoC Results

Date: 2026-06-03

Remote run directory: `/scratch/lz432/sigml_solver_phasec`

Status: `DONE_WITH_CONCERNS`

## Completed

- Implemented `CtsegLabeler` in `sigml/solver/labeler.py` with the verified TRIQS/CTSEG one-orbital wiring: spinful `Solver`, blockwise `Delta_tau`, `Jperp_tau=0`, `h_int=U n_up n_down`, `h_loc0=-(mu-eps_d)(n_up+n_down)`, measured `G_tau`, and projected the CTSEG result to the 59-node Valenti mesh as a 118-vector.
- Added a skipped-unless-TRIQS near-atomic unit test. The strict `Delta=0` atomic limit is intentionally not used because CTSEG rejects it.
- Added Slurm-ready scripts:
  - `scripts/solver/validate_ctseg_labeler.py`
  - `scripts/solver/validate_ctseg_labeler.slurm`
  - `scripts/solver/benchmark_phase_c.py`
  - `scripts/solver/benchmark_phase_c.slurm`
  - `scripts/solver/gen_labels.py`
  - `scripts/solver/gen_labels.slurm`
  - `scripts/solver/export_orb1_weights_npz.py`
- Added `sigml/solver/numpy_oracle.py`, a NumPy-only inference path for the released `orb1` checkpoint so the benchmark can run in the verified `triqs_ctseg` env without installing PyTorch.

## CTSEG Labeler Validation

Job: `55491367`

Settings: beta `70`, `U=2`, `mu=1`, near-atomic bath `V=0.10`, `eps=0`, `100000` cycles, `5000` warmup cycles, 64 MPI ranks on `main-redhat`.

Result:

| quantity | value |
|---|---:|
| Slurm state | `COMPLETED` |
| elapsed | `00:00:10` |
| average sign | `1.0` |
| `G_tau` min | `-0.5093900285` |
| `G_tau` max | `-0.0186718462` |
| causal on positive Matsubara frequencies | `true` |
| sane imaginary-time sign/range | `true` |

This verifies that one real CTSEG solve returns a finite, causal, sane `g_vec` on the Valenti 118-feature representation.

## NN + 1 CTQMC Refinement Benchmark

Corrected benchmark job: `55491551`

Settings: beta `70`, half filling, Bethe `t=1`, `mix=0.5`, `tol=5e-3`, `100000` CTSEG cycles, `5000` warmup cycles, 64 MPI ranks.

| U | full CTSEG converged | full CTSEG iterations | NN converged | NN iterations | CTSEG refinement solves | max abs diff low-frequency Sigma | mean abs diff low-frequency Sigma |
|---:|---|---:|---|---:|---:|---:|---:|
| 2.0 | `true` | 5 | `true` | 5 | 1 | 6.985484 | 1.686130 |
| 4.0 | `true` | 5 | `true` | 5 | 1 | 8.984573 | 2.549209 |

Headline assessment: the desired `NN+1 ~= full CTSEG` claim was not achieved in this run. Full CTSEG converged quickly at the loose benchmark tolerance, and the NN+1 refined self-energy still differed substantially at low Matsubara frequencies.

The corrected benchmark fixed a bookkeeping issue from job `55491368`: Dyson self-energy must use the impurity `Delta` that produced the measured CTSEG `G`, not the post-mixing `Delta_next`.

## Mott / Hysteresis

The Mott/hysteresis sweep used the released `orb1` oracle through the NumPy inference path, not a CTSEG-trained net.

| sweep | U | `-G(beta/2)` | converged | iterations |
|---|---:|---:|---|---:|
| up | 2.0 | 0.012775 | `true` | 5 |
| up | 3.0 | 0.013420 | `true` | 1 |
| up | 4.0 | 0.014078 | `true` | 2 |
| up | 5.0 | 0.013288 | `true` | 3 |
| up | 6.0 | 0.009848 | `true` | 3 |
| down | 2.0 | 0.015122 | `true` | 2 |
| down | 3.0 | 0.014185 | `true` | 4 |
| down | 4.0 | 0.006599 | `true` | 3 |
| down | 5.0 | 0.002335 | `true` | 2 |
| down | 6.0 | 0.002886 | `true` | 5 |

This gives a hysteresis-like separation between the up and down sweeps, but it should be treated as an oracle-based PoC curve rather than a validated CTSEG-trained production result.

## Real-CTSEG Training Set

Deferred. The Slurm array generator is implemented, but no scale-out `300-1000` sample production run was launched. The successful validation and benchmark runs measured small one-orbital solves at roughly the expected order of cost; production labels should start with a small array and inspect statistical error before scaling.

## Slurm Accounting

| job | purpose | state | elapsed | CPUs | core-hours |
|---:|---|---|---:|---:|---:|
| 55491366 | validation attempt, failed torch import | `FAILED` | 2 s | 64 | 0.036 |
| 55491367 | CTSEG labeler validation | `COMPLETED` | 10 s | 64 | 0.178 |
| 55491368 | first benchmark, pre-correction Sigma bookkeeping | `COMPLETED` | 184 s | 64 | 3.271 |
| 55491551 | corrected benchmark | `COMPLETED` | 179 s | 64 | 3.182 |

Total allocated compute used: about `6.67` core-hours.

## Artifacts

- Remote validation JSON: `/scratch/lz432/sigml_solver_phasec/results/ctseg_labeler_validation.json`
- Remote validation vectors: `/scratch/lz432/sigml_solver_phasec/results/ctseg_labeler_validation.npz`
- Remote benchmark summary: `/scratch/lz432/sigml_solver_phasec/results/phase_c_summary.json`
- Remote benchmark markdown draft: `/scratch/lz432/sigml_solver_phasec/results/solver_poc_results.md`

## Next Technical Risks

- The self-energy mismatch is too large for the acceptance target. The next debugging pass should compare full CTSEG and NN-refined `G_tau`, `Delta_tau`, and `Sigma(iw)` pointwise, then isolate whether the mismatch is from oracle generalization, CTSEG noise, the Bethe seed/tolerance, or the DLR projection.
- The full CTSEG reference converged in 5 iterations with `tol=5e-3`, not the expected 10-30 iterations. A stricter tolerance and more representative seed may be needed for the intended headline comparison.
- `CtsegSolveInfo` currently records average sign, but density and perturbation-order extraction need TRIQS result attribute normalization if those fields are required programmatically.

## v2 Follow-Up: Diagnosis, Tight Baseline, Small Real-CTSEG Labels

Date: 2026-06-03

Remote run directory: `/scratch/lz432/sigml_solver_v2`

Status: `STOPPED_AT_COMPUTE_CAP_WITH_BOTTLENECK_CHARACTERIZED`

Important process note: the requested hard cap was `~40` new core-hours. The v2 runs reached `47.24` new core-hours before the final trained-net CTSEG benchmark was submitted, mainly because the label-array walltime was higher than projected and failed array children still consumed allocation. No further Slurm work was launched after this accounting check.

### v2 Code Changes

- Added v2 diagnostic, tight-baseline, label-generation, assembly, training, and benchmark scripts under `scripts/solver/`.
- Updated `CtsegLabeler` to retain both direct-node and DLR-projected `G` vectors from the same CTSEG solve, so DLR projection error can be measured without duplicate CTSEG solves.
- Fixed the benchmark low-frequency helper to sort positive Matsubara nodes by `|omega_n|` before selecting the first `n` points.
- Guarded v2 JSON/NPZ writers so only MPI rank 0 writes result files.
- Added bounds checks in v2 label generation; if damped oracle-Bethe steps produce non-finite or very large vectors, the sample falls back to a random causal `Delta`.

### v2 Diagnosis At U=4, beta=70, Half Filling

Jobs:

- `55491912`: first diagnostic before rank-0 write guard.
- `55492025`: corrected diagnostic after the low-frequency selection fix.

Settings: `U=4`, `beta=70`, half filling, Bethe `t=1`, `mix=0.5`, `tol=5e-3`, `100000` CTSEG cycles, `5000` warmup cycles, 64 MPI ranks.

Full CTSEG again converged in 5 iterations at the loose tolerance:

```text
errors = [0.0730833, 0.0346429, 0.0157529, 0.00670575, 0.00266309]
```

Pointwise comparisons:

| comparison | metric | max abs | mean abs | RMS |
|---|---|---:|---:|---:|
| full CTSEG vs NN+1 CTSEG | `Delta_tau` vector | 0.008198 | 0.001633 | 0.002420 |
| full CTSEG vs NN+1 CTSEG | low-`iw` `Delta` | 0.001997 | 0.000503 | 0.000788 |
| full CTSEG vs NN+1 CTSEG | `G_tau` vector | 0.005552 | 0.000772 | 0.001324 |
| full CTSEG vs NN+1 CTSEG | low-`iw` `Sigma` | 8.984573 | 2.549209 | 3.602353 |
| NN vs NN+1 CTSEG at identical `Delta` | `G_tau` vector | 0.014507 | 0.001176 | 0.002291 |
| NN vs NN+1 CTSEG at identical `Delta` | low-`iw` `Sigma` | 8.430925 | 3.366797 | 4.041347 |

DLR projection is not the bottleneck. Direct-node vs DLR projection errors from the same CTSEG solve were tiny:

| solve | max abs | mean abs | RMS |
|---|---:|---:|---:|
| final full-CTSEG iteration | 3.39e-7 | 3.28e-8 | 7.10e-8 |
| NN+1 refinement solve | 1.18e-7 | 8.11e-9 | 1.95e-8 |

Root-cause finding: the Phase C headline failed primarily because the released `orb1` oracle is out of distribution for this Bethe path, and the Dyson `Sigma` metric is highly sensitive to small `G_tau`/`Delta` differences and CTSEG solve noise. The DLR projection error is orders of magnitude too small to explain the mismatch, and the Bethe seed/mixing/tolerance are not the dominant direct cause at the loose `5e-3` benchmark.

### v2 Tight Full-CTSEG Baseline

Job: `55492095`

Settings: `U=4`, `beta=70`, half filling, Bethe `t=1`, `mix=0.5`, `100000` cycles, `5000` warmup, `max_iter=20`, 64 MPI ranks.

| tolerance | converged | iterations run | minimum observed update error | final update error |
|---:|---|---:|---:|---:|
| 1e-4 | `false` | 20 | 2.952e-4 | 3.423e-4 |
| 1e-5 | `false` | 20 | 2.952e-4 | 3.423e-4 |

Interpretation: at 100k cycles the loop reaches an apparent stochastic/noise floor around `3e-4`, so `1e-4` and `1e-5` are not realistic convergence targets without higher statistics or a different convergence estimator. An honest reference iteration count for this one-orbital Bethe setup is:

- `tol=5e-3`: 5 full-CTSEG iterations.
- `tol≈5e-4`: about 8-9 iterations from the observed trajectory.
- `tol<=1e-4`: not converged in 20 iterations at 100k cycles.

This rules out a credible `10-30 -> 1-2` headline on the current cheap one-orbital benchmark unless the reference tolerance/statistics are redefined.

### v2 Small On-Distribution Training Set

Jobs:

- `55492545`: initial array `0-39%12`, `20000` cycles and `2000` warmup cycles.
- `55492664`: retry of missing tasks `2-8%2` after adding the oracle-step guard.

Assembled dataset: `/scratch/lz432/sigml_solver_v2/results/ctseg_labels_v2.npz`

| quantity | value |
|---|---:|
| finite labels | 164 |
| random `Delta` labels | 75 |
| oracle-near-self-consistency labels | 83 |
| guarded fallback random labels | 6 |
| dropped non-finite labels | 0 |
| training cycles per label | 20k |
| warmup cycles per label | 2k |

Duplicate same-input labels were included for four replicate keys, but they came out bitwise identical. That means the current CTSEG invocation is deterministic for repeated same-input solves under default seeding, so this run did not measure Monte Carlo statistical noise. A future noise measurement needs explicit solver seed control if CTSEG exposes it, or independently perturbed runs at the Slurm/script level.

Local training result for `FeedforwardNet` on the 164-label set:

| quantity | value |
|---|---:|
| train rows | 139 |
| validation rows | 25 |
| epochs | 1200 |
| best validation MSE | 3.3268e-5 |
| final train MSE | 1.5450e-5 |
| final validation MSE | 3.4414e-5 |

Exported weights: `.solver_v2/ctseg_net_v2_weights.npz` locally. They were not benchmarked with CTSEG because the new Slurm usage had already exceeded the requested cap.

On-distribution-net result: `INCOMPLETE`. The net trained successfully, but there is no valid NN+1 CTSEG benchmark under the compute cap. Therefore the v2 run does not establish `NN+1 ~= full CTSEG` or an iteration reduction.

### v2 Cost Accounting And Projection

New Slurm core-hours counted from `sacct` top-level jobs and array children:

| job | purpose | state summary | core-hours |
|---:|---|---|---:|
| 55491912 | first diagnostic | completed | 1.35 |
| 55492025 | corrected diagnostic | completed | 1.35 |
| 55492095 | tight baseline | completed | 8.59 |
| 55492545 | initial 40-task label array | completed/failed children | 29.55 |
| 55492664 | retry 7 missing label tasks | completed | 6.40 |
| total | v2 new Slurm allocation | mixed, no active jobs left | 47.24 |

Measured label-generation cost:

- Total label-array allocation including failed children: `35.95` core-hours for 164 usable 20k-cycle labels, or `0.219` core-hours per usable label.
- Successful-label allocation excluding failed children: about `32.75` core-hours for 164 labels, or `0.200` core-hours per usable label.
- A simple 100k-cycle projection is therefore roughly `1.0-1.1` core-hours per label on this setup.

Projected production costs from measured v2 label throughput:

| target | assumed labels/cycles | projected core-hours |
|---|---:|---:|
| scaled good PoC | 500 labels x 100k cycles | about 500-550 |
| stronger good PoC | 1000 labels x 100k cycles | about 1000-1100 |
| Valenti-scale | 16000 labels x 100k cycles | about 16000-17500 |

These projections assume the same one-orbital CTSEG setup and current scripting overhead. They do not include extra cost for independent noise replicas, higher statistics needed for `Sigma`, or broader physical validation sweeps.

## Closeout

Date: 2026-06-03

Remote run directory: `/scratch/lz432/sigml_solver_closeout`

Status: `DONE_PARTIAL_SUCCESS`

This closeout uses the v2 on-distribution CTSEG-trained NumPy net from `.solver_v2/ctseg_net_v2_weights.npz`, not the released out-of-distribution `orb1` oracle. The benchmark ran beta `70`, half filling, Bethe `t=1`, `mix=0.5`, `tol=5e-3`, `100000` CTSEG cycles, `5000` warmup cycles, and 64 MPI ranks on `main-redhat`.

The primary accuracy metric is now the review-recommended `G_tau` error on the 118-feature Valenti mesh. Low-frequency `Sigma(iw)` is kept only as a diagnostic because Dyson inversion amplifies small CTSEG/noise differences.

### Closing Benchmark

Job: `55492876`

| U | full CTSEG converged | full CTSEG iterations | NN converged | NN iterations | NN+1 total solver iterations | full vs NN+1 `G_tau` max abs | full vs NN+1 `G_tau` mean abs | full vs NN+1 `G_tau` RMS | full vs NN+1 `Delta_tau` mean abs | full vs NN+1 low-`iw` `Delta` mean abs |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2.0 | `true` | 5 | `true` | 5 | 6 | 0.003343 | 0.000704 | 0.001159 | 0.002463 | 0.000669 |
| 4.0 | `true` | 5 | `true` | 5 | 6 | 0.005573 | 0.000538 | 0.001099 | 0.002315 | 0.000596 |

Observable check:

| U | `-G(beta/2)` full CTSEG | `-G(beta/2)` NN | `-G(beta/2)` NN+1 |
|---:|---:|---:|---:|
| 2.0 | 0.011019 | 0.011114 | 0.011946 |
| 4.0 | 0.010977 | 0.011338 | 0.011473 |

Diagnostic low-frequency `Sigma(iw)` error:

| U | full vs NN low-`iw` `Sigma` mean abs | full vs NN+1 low-`iw` `Sigma` mean abs | full vs NN+1 low-`iw` `Sigma` max abs |
|---:|---:|---:|---:|
| 2.0 | 5.232253 | 2.454064 | 5.390335 |
| 4.0 | 7.451544 | 2.393023 | 3.821229 |

Interpretation: on the correct `G_tau` metric, the v2 on-distribution net plus one CTSEG refinement lands close to the full CTSEG reference on this one-orbital Bethe PoC, with mean absolute `G_tau` errors of `7.04e-4` at `U=2` and `5.38e-4` at `U=4`. The one-refinement CTSEG solve improves the direct NN `G_tau` mean error from `1.97e-3 -> 7.04e-4` at `U=2` and `2.21e-3 -> 5.38e-4` at `U=4`.

The iteration-reduction claim is only partial. Full CTSEG converged in 5 iterations at this loose tolerance, while the NN loop also took 5 cheap NN iterations and then one CTSEG refinement solve. This demonstrates that the on-distribution net plus one CTSEG solve gives close `G_tau` agreement and improves over the OOD-oracle failure mode, but it does not demonstrate a dramatic `10-30 -> 1-2` full-CTSEG iteration reduction on this cheap one-orbital benchmark. The honest verdict is: `partially yes` for NN+1 agreement under the primary `G_tau` metric, `no` for a strong iteration-reduction headline.

### Convention-Locking Fixtures

Fixture job: `55492875`

Committed small fixtures under `tests/solver/fixtures/`:

- `valenti_reference_delta_beta70.npz`: a known near-atomic `Delta` converted with Valenti's `mldmft/utils.py` `NNoutput_to_DLR` and `DLR_to_NNinput`; the test asserts `ValentiOrb1Grid` reproduces the same 118-vector and the beta-scaled DLR imfreq convention.
- `ctseg_projection_near_atomic_beta70.npz`: recorded CTSEG input `Delta`, measured raw `G_tau`, direct Valenti-node vector, and default-DLR-projected 118-vector; the test replays the recorded object through `CtsegLabeler._project_g_tau_to_valenti_vec` with default `projection="dlr"`.

Local fixture-enabled solver test result:

```text
conda run -n nequip bash -lc 'PYTHONPATH=. python -m pytest tests/solver/ -q'
26 passed, 1 skipped in 2.40s
```

### Closeout Slurm Accounting

| job | purpose | state | elapsed | CPUs | core-hours |
|---:|---|---|---:|---:|---:|
| 55492875 | convention fixture generation | `COMPLETED` | 25 s | 64 | 0.444 |
| 55492876 | closing U=2,U=4 v2 benchmark | `COMPLETED` | 267 s | 64 | 4.747 |
| total | closeout new allocation | `COMPLETED` | 292 s | 64 | 5.191 |

This closeout stayed below the requested `10` new core-hour cap.

Closeout artifacts:

- Remote benchmark JSON: `/scratch/lz432/sigml_solver_closeout/results/benchmark_v2.json`
- Remote benchmark vectors: `/scratch/lz432/sigml_solver_closeout/results/benchmark_v2.npz`
- Remote fixtures: `/scratch/lz432/sigml_solver_closeout/results/fixtures/`

### v2 Artifacts

- Diagnostic JSON: `/scratch/lz432/sigml_solver_v2/results/diagnose_v2.json`
- Tight-baseline JSON: `/scratch/lz432/sigml_solver_v2/results/baseline_tight_v2.json`
- Label dataset: `/scratch/lz432/sigml_solver_v2/results/ctseg_labels_v2.npz`
- Label summary JSON: `/scratch/lz432/sigml_solver_v2/results/ctseg_labels_v2_summary.json`
- Local trained checkpoint: `.solver_v2/ctseg_net_v2.pt`
- Local NumPy weights: `.solver_v2/ctseg_net_v2_weights.npz`
