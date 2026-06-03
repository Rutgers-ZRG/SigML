# Solver Phase A PoC

Date: 2026-06-03

This milestone drives the released Valenti `orb1` oracle through the SigML
Bethe-lattice DMFT loop at half filling (`mu=U/2`, `beta=70`, `t=1`).
The loop is run on the 118-real-feature tau-node vector with
`Delta_next=(1-mix)*t**2*G + mix*Delta_old`, `mix=0.6`, `tol=1e-4`,
and `max_iter=300`.

The oracle is sensitive to the initial hybridization. The production
milestone uses Valenti's provided example hybridization seed:

`/Users/li/dev/RA/dmft/ref/mlDMFT/mldmft/examples/1orbital/NN/example_inputs/inputs.npy`

With this seed, the U sweep gives:

| U | converged | iterations | -G(beta/2) |
|---:|:---:|---:|---:|
| 1.0 | yes | 17 | 0.0145016330644 |
| 2.0 | yes | 17 | 0.0147325325233 |
| 3.0 | yes | 17 | 0.0143371451413 |
| 4.0 | yes | 17 | 0.0140273905708 |
| 5.0 | yes | 22 | 0.0128696097527 |
| 6.0 | yes | 32 | 0.00129281892441 |

Phase-A physics result: the quasiparticle proxy collapses from
`0.0145016330644` at `U=1` to `0.00129281892441` at `U=6`, so the
metal-to-Mott direction is reproduced by the oracle-driven loop.

Concern: starting from the Task-4 zero hybridization seed, the same released
oracle diverges at low U (`U=1`, `beta=70`, `mix=0.6`) and reaches non-finite
G values by the end of `max_iter=300`. The analytic noninteracting solver test
still verifies the loop contract independently of the neural oracle.
