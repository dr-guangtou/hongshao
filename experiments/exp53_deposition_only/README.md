# exp53 — the deposition-only boundary test

Branch `exp53-deposition-only`. Plan:
`doc/plans/2026-08-20-deposition-only-test.md`.

## Why

exp52 found the adopted `1ch-mof` kernel grows galaxies by **redistribution,
not accretion**: 96.5% of its own `M*[50,100]` growth over z=0.7->0.4 is
material it already had, moved outward, and its own deposition supplies only
~11% of the z=2->0.4 mass growth (the rest is the `M(<500)` pinning). That is
incompatible with the established picture of massive-galaxy outskirt growth by
dry minor mergers, and it explains the exp47/exp48 compact-galaxy central
deficit — with deposition effectively off after z~1.5, filling an outskirt
requires emptying the centre.

exp53 runs the boundary condition: **remove transport entirely** (`q = 0`, a
clean nested special case) and see what the model becomes. It is run graded
rather than binary, with the efficiency window free, judged against the
measured added light rather than the loss, and with the Sersic family
reinstated behind the reparameterization that exp38 stage 1 named as the fix
for the degeneracy that had eliminated it.

## What is new here (machinery, all self-checked on real galaxies)

### The R50 reparameterization — `families.py`

Every deposit family is parameterized by its own **half-mass radius** rather
than its family-specific scale. exp38 stage 1 rejected `sersic (n free)` with
"LOWER rail 3/5 epochs (n-scale degeneracy; needs an R50 reparameterization)".
That degeneracy is now measured: **at a fixed raw scale `a = 1`, running the
index `n` from 0.5 to 8 moves the profile's half-mass radius by a factor of
1.2e6.** Scale and index were competing to set the same physical radius, so the
fit slid along the valley until one of them hit its box. In R50 coordinates the
shape parameter controls the wings alone.

Verified: `gauss == sersic n=0.5 ==` exp38's incumbent Gaussian and
`expo == sersic n=1` to 1e-12; both reparameterizations reproduce exp38's
raw-scale closed forms exactly; `R50` is the half-mass radius to 2e-8 for every
family and every shape value; closed forms match brute-force integration of
their own `Sigma`.

### The switchable kernel — `kernel.py`

Structurally the production kernel, with the family and the transport exponent
selectable. Verified against exp38's **adopted** `1ch-mof` on real galaxies:
CoGs agree to **8.9e-16** relative and the population loss to 1e-12. `q = 0`
nests exactly and is *proved* independent of the core-fraction clock (by
perturbing that clock and getting the identical answer, not by reading the
code).

Carries exp49's **300 kpc deposit-radius ceiling**, measured there at 1.5e-5 of
the loss after refitting — free — and it bites: uncapped, the adopted theta
deposits at half-mass radii up to **4088 kpc**, beyond any host halo's virial
radius. It matters more for a family shootout than it did for exp49, because
without it part of what a family comparison measures is which family is best at
hiding mass beyond the normalization radius.

One deliberate departure from the production box: `g` is allowed up to 6
instead of railing at 4. exp49 measured the interior optimum at
`g = 4.35 +/- 0.003`, so the production box is a wall, and exp48's family
comparison was criticized for fitting every family against one. Every exp53
cell shares the widened box, so the comparison is internally fair; the compact
central deficit is reported at every cell so the cost of the widening stays
visible.

### The judge — `score.py`

exp38 stage 0.2's added-light operator, applied to model CoGs: stacked median
added `Sigma` per logM* tercile x adjacent epoch pair, the same Sersic grid fit
over the same 4-120 kpc band. Verified to reproduce every recorded kernel cell
in `exp38/outputs/stage0_kernels.npz` exactly, and the data column is
model-independent by construction. Plus exp52's differential slopes and core
`dlogSigma`, the transport share, and the population-summed growth ratios that
carry exp53's falsifiable prediction.

## Status

Machinery complete and committed. Full-sample fits (14 cells x 3 starts,
n = 2397) running. Results below when they land.
