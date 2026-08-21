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

## Result 0 — a correction to exp52, found by failing to reproduce it

exp53 could not reproduce exp52's headline `M(<10)` model/truth growth ratio of
0.573 at the same theta, while reproducing its `M[50,100]` companion (0.989)
exactly. That split located a defect in `exp52/decompose.py`: the truth term
was `np.interp(r_out, R, d) - np.interp(r_in, R, d)` with `r_in = 0.0`, and
numpy's `interp` CLAMPS below its grid, returning the CoG at the innermost
aperture radius (2 kpc) rather than zero — while the model term used the
analytic `enclosed(0.0, ...) = 0`. Every `r_in = 0` row therefore divided the
model's APERTURE growth by the truth's ANNULUS growth.

It mattered because the truth's `M(<2 kpc)` FALLS over z=1.0->0.4 (-6.55e12
Msun summed, 62% of galaxies negative), so dropping it inflated the denominator
by 27%.

**`M(<10)` model/truth growth over z=1.0->0.4 is 0.726, not 0.573.** exp52's
qualitative finding survives — the adopted kernel does under-grow the centre —
at 27% rather than 43%. Its *mechanism* headline is untouched: transport shares
are ratios of model terms with no truth denominator.

Three model-side explanations were tested and cleared before the convention was
checked (the 300 kpc deposit ceiling: 0.726 uncapped vs 0.701 capped; an
analytic-vs-interpolated radial operator: 0.1 points; the choice of adopted
theta). Fixed in `exp52/decompose.py`, recorded in `doc/lessons.md`, and locked
out by assertions in `score.py`.

## Result 1 — the SLICE: removing transport DOES fix the central deficit, and
overshoots

The adopted theta with `q` forced along the grid and nothing re-optimized.
This is the form exp52's mechanism claim takes, and it is confirmed
emphatically. n = 2397, scored at all five epochs.

| `q` | M(<5) bias, all | M(<5) bias, **compact** | M(<10) growth vs truth | kernel dist [dex] | GATE diff (data 0.37) | dlog R50 (z=0.4) |
|---|---|---|---|---|---|---|
| 0.908 (adopted) | -13.4% | **-43.6%** | 0.701 | 0.077 | 0.41 | +0.028 |
| 0.80 | -1.6% | -36.2% | 1.279 | **0.063** | 0.37 | -0.019 |
| 0.60 | +21.3% | -22.4% | 2.326 | 0.088 | 0.29 | -0.105 |
| 0.40 | +43.8% | -9.3% | 3.317 | 0.139 | 0.23 | -0.190 |
| 0.20 | +65.7% | +2.9% | 4.230 | 0.198 | 0.18 | -0.273 |
| 0.00 | +86.3% | **+13.6%** | 5.054 | 0.260 | 0.12 | -0.355 |

**exp53's falsifiable prediction passes on the slice, and then some.** The
compact-galaxy `M*(<5 kpc)` deficit runs from -43.6% to +13.6%, crossing zero
near `q ~ 0.25`. But removing transport does not *correct* the centre so much
as *overrun* it: `M(<10)` ends up growing 5x faster than the truth, half-mass
radii collapse by 0.36 dex, and the differential-deposition gate fails
outright (0.12 against a measured 0.37).

Two readings that must not be taken from this table.

- **This is NOT "deposition-only fails".** The slice holds the efficiency
  window frozen at the adopted `sig = 0.237`, which is exactly the
  uninterpretable case design constraint 2 names: with a narrow early window
  the late deposits are nearly massless, so a no-transport model has no way to
  change shape at late times. The profiled fits are what settle it.
- **The added-light kernel does not want transport removed.** Its distance is
  minimized at `q = 0.80` (0.063 dex), not at `q = 0` (0.260) — the measured
  kernel prefers slightly LESS transport than the adopted 0.908, not none.

## Status

Machinery complete and committed; the slice is final. Full-sample profiled
fits (14 cells x 3 starts, n = 2397) running.

## Result 2 — stage A, the PROFILED transport exponent (n = 2397, 3 starts/cell)

Loss: 0.160001 (q=0) -> 0.158808 -> 0.158078 -> 0.156983 -> 0.153998 ->
**0.153577 at q = 0.908 free** -> 0.153872 (q=1.0) -> 0.156512 (q=1.2). A
single well-formed minimum near q ~ 0.9, rising on both sides; the profiled
curve brackets the free fit correctly. Under the production objective the data
genuinely wants transport, and removing it costs 4.2%.

| cell | loss | q | sig | g | log R50 | M(<10) growth | M(<5) compact | kernel [dex] | GATE (0.37) | dlog R50 |
|---|---|---|---|---|---|---|---|---|---|---|
| q free | 0.153577 | 0.91 | 0.223 | 4.28 | 3.27 | 0.722 | -43.4% | 0.081 | 0.42 | +0.026 |
| **q=0.6** | 0.156983 | 0.60 | 0.252 | 3.68 | 2.89 | 1.417 | **-39.0%** | 0.070 | **0.37** | **-0.009** |
| **q=0.8** | 0.153998 | 0.80 | 0.232 | 4.05 | 3.13 | **0.991** | -41.9% | **0.068** | 0.40 | +0.012 |
| q=0 | 0.160001 | 0.00 | 2.683 | 1.72 | 1.73 | 0.879 | -39.4% | 0.086 | 0.36 | +0.016 |
| q=1.2 | 0.156512 | 1.20 | 0.217 | 4.71 | 3.50 | -0.066 | -47.3% | 0.108 | 0.50 FAIL | +0.065 |

**(a) The compact-galaxy central deficit is TRANSPORT-INDEPENDENT.** It sits at
-39% to -47% at every q. Removing transport entirely buys 4 points out of 43.
**exp53's falsifiable prediction FAILS under profiling** — and it passes
spectacularly on the slice (-43.6% -> +13.6%), which is exactly the
slice-versus-profile disagreement exp49 warned about, now landing on the
central question. exp52's mechanism claim splits in two: *"the adopted kernel's
late outskirt growth is redistribution"* stands, but *"...and that causes the
compact central deficit"* does not survive re-optimization. The exp47/exp48
compact defect is still unexplained.

**(b) The profile has TWO BRANCHES and the fit switches at q ~ 0.5.** Below:
wide efficiency window (sig up to 2.68), small deposits (log R50 1.73), weak
size growth (g 1.7). Above: narrow early window (sig ~0.22), huge deposits
(log R50 3.3-3.5), steep growth (g 4.3-4.7). Physically different models, not
one model sliding; the adopted kernel lives on the high branch.

**(c) The best candidate is neither the adopted model nor the boundary.**
Judged against the measured added light, **q = 0.8 costs 0.27% of the loss**
and improves the kernel distance 0.081 -> 0.068, the M(<10) growth ratio
0.722 -> **0.991**, and the size offset +0.026 -> +0.012. q = 0.6 costs 2.2%
and matches the differential gate exactly (0.37 vs 0.37) with the best sizes.
**The answer to "remove transport?" is no — trim it by 20-35%**, which the
loss alone would never have indicated since it minimizes at 0.91.

## Result 3 — stage B, the family shootout (PROVISIONAL: rails, see below)

| family | q free | q = 0 | cost of removing transport |
|---|---|---|---|
| moffat (power-law tail) | 0.153577 | 0.160001 | +4.2% |
| sersic (**n = 2.29**) | 0.156708 | 0.162079 | +3.4% |
| expo (n = 1) | 0.158671 | 0.175218 | +10.4% |
| gauss (n = 0.5) | 0.160456 | 0.177770 | +10.8% |

**Sersic fits at n = 2.29, inside the measured kernel's 2.1-3.1**, unrailed and
stable across three starts. exp38 stage 1 rejected this family for railing `n`;
with the R50 reparameterization it lands exactly where the measured added light
says it should. The handover's point 4 is vindicated.

**Transport SUBSTITUTES for a heavy tail.** Removing it costs the two
heavy-tailed families 3-4% and the two light-tailed ones ~11% — the Gaussian
and exponential *need* transport because they have no other way to reach the
outskirts. That reframes the transport term as compensation for a too-light
deposit profile, which is more actionable than "the model redistributes".

### Rails, and what the rail check found

Six cells converged at a bound. `rail_check.py` refits each with that bound
widened, from the railed fit and from a point nudged past the old wall.

- **`mu` = 5.0 on the DEPOSITION-ONLY cells is BENIGN.** `sersic-q0` widened to
  `mu <= 12` gains 0.054% of the loss and one start settles at `mu` = 8.3 with
  the same loss. Design constraint 2 is satisfied: the efficiency window was
  effectively free.
- **`log_R50` = 3.5 on the light-tailed q-free cells is NOT benign, and it does
  not stop.** All three escape 3.5 and immediately rail at the NEW bound of
  5.0, gaining 0.4% (sersic), 1.6% (expo), 2.6% (gauss). **The deposit scale is
  UNIDENTIFIED for every family except the Moffat**, which is interior at 3.266
  — the exp36 Gaussian rail, alive in three new families.
  And with the box widened the three converge to nearly the same loss
  (0.156133 / 0.156156 / 0.156216): **once the scale is free to rail, the
  family comparison collapses**, because "every deposit at the 300 kpc ceiling"
  is the same model whatever profile shape you hang on it. That is a mechanism
  for exp48's "the profile does not matter", and it means a family shootout is
  only meaningful with the deposit scale pinned.
