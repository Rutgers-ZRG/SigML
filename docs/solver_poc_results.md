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
