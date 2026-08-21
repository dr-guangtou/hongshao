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

## Result 1 — the SLICE: removing transport DOES fix the central deficit, and overshoots

The adopted theta with transport switched off and NOTHING re-optimized. This
is the form exp52's mechanism claim takes. n = 2397, scored at all five epochs.

| | adopted (`q` = 0.908) | transport OFF (`q` = 0), nothing refitted |
|---|---|---|
| M(<5) bias, all | -13.4% | **+86.3%** |
| M(<5) bias, **compact** | **-43.6%** | **+13.6%** |
| M(<10) growth vs truth | 0.701 | 5.054 |
| added-light kernel [dex] | 0.077 | 0.260 |
| GATE differential (data 0.37) | 0.41 | 0.12 |
| dlog R50 at z=0.4 | +0.028 | -0.355 |

**On the slice the prediction passes emphatically**: the compact-galaxy
`M*(<5 kpc)` deficit goes from -43.6% to +13.6%. But it does not *correct* the
centre so much as *overrun* it — `M(<10)` ends up growing 5x faster than the
truth, half-mass radii collapse by 0.36 dex, and the differential gate fails
outright (0.12 against a measured 0.37).

**This is NOT "deposition-only fails".** The slice holds the efficiency window
frozen at the adopted `sig` = 0.237, which is exactly the uninterpretable case
design constraint 2 names: with a narrow early window the late deposits are
nearly massless, so a no-transport model has no way to change shape at late
times. Result 3 is the properly refitted answer, and it is very different.

## Result 3 — stage B, the family shootout (PROVISIONAL: rails, see below)

| family | q free | q = 0 | cost of removing transport |
|---|---|---|---|
| moffat (power-law tail) | 0.153577 | 0.160001 | +4.2% |
| sersic (**n = 2.29**) | 0.156708 | 0.162079 | +3.4% |
| expo (n = 1) | 0.158671 | 0.175218 | +10.4% |
| gauss (n = 0.5) | 0.160456 | 0.177770 | +10.8% |

**Sersic fits at n = 2.29, inside the measured kernel's 2.1-3.1**, unrailed and
stable across three starts.

**CORRECTION (2026-08-21): this is a CONFIRMATION of exp48, not a new result,
and the framing exp53 inherited was wrong.** The 2026-08-20 handover said
Sersic "was eliminated by a numerical artifact" that exp53 would fix. exp38
stage 1 did reject it for railing `n` -- but **exp48 step C had already
supplied the R50 reparameterization and already re-run Sersic inside the full
multi-epoch kernel**, where it came LAST on the judge (mean E/floor 2.44
against Moffat's 1.95) while tying best on the loss under exp48's winning
objective. exp53 reproduces that verdict under a DIFFERENT objective (Sersic
loses to Moffat on both loss, 0.1567 vs 0.1536, and kernel distance, 0.118 vs
0.081), which strengthens exp48's conclusion rather than overturning
anything. The genuinely new part is narrow: exp48 left the Moffat in raw `rc`
coordinates, so exp53 is the first run in which `log_R50` means the same
physical quantity for every candidate.

**Transport SUBSTITUTES for a heavy tail.** Removing it costs the two
heavy-tailed families 3-4% and the two light-tailed ones ~11% — the Gaussian
and exponential *need* transport because they have no other way to reach the
outskirts. That reframes the transport term as compensation for a too-light
deposit profile, which is more actionable than "the model redistributes".

### Rails, and what the rail check found

Six cells converged at a bound. `rail_check.py` refits each with that bound
widened, from the railed fit and from a point nudged past the old wall.

| cell | rail | widened to | loss gain | escapes? |
|---|---|---|---|---|
| moffat-q0 | `mu` 5.0 | 12 | +0.024% | rails again at 12 |
| sersic-q0 | `mu` 5.0 | 12 | +0.054% | rails again at 12 |
| moffat-q1.2 | `log_R50` 3.5 | 5 | +0.011% | **yes — settles at 3.626, INTERIOR** |
| sersic-qfree | `log_R50` 3.5 | 5 | +0.369% | no — rails again at 5.0 |
| expo-qfree | `log_R50` 3.5 | 5 | **+1.585%** | no — rails again at 5.0 |
| gauss-qfree | `log_R50` 3.5 | 5 | **+2.643%** | no — rails again at 5.0 |

**(a) The `mu` rail on the DEPOSITION-ONLY cells is benign** (0.02-0.05% of the
loss). Design constraint 2 is satisfied: the efficiency window was effectively
free, so the stage-A result stands as reported.

**(b) The Moffat is the only family whose deposit scale is IDENTIFIED.**
`moffat-qfree` is interior at `log_R50` = 3.266 and `moffat-q1.2` escapes its
rail to a finite 3.626. Every other family rails again at whatever bound it is
given.

**(c) The other three families' deposit scale is UNIDENTIFIED, and the loss
gain understates it.** The verdict threshold reads `sersic-qfree` as benign at
+0.369%, and in loss terms it is — but it still rails at the new bound, which
means the objective simply *does not constrain the deposit scale from above*
for a light-tailed deposit. That is the sharper statement: not "the box chose
the answer" but "nothing chooses the answer". (exp49 found the same thing from
the other side: the Mpc deposit tail is unconstrained slack.)

**(d) Once the scale is free to rail, the family comparison COLLAPSES.**
Widened, the three converge to 0.156131 (sersic) / 0.156156 (expo) / 0.156214
(gauss) — indistinguishable — while the Moffat sits clearly better at 0.153577
with an interior scale. "Every deposit at the 300 kpc ceiling" is the same
model whatever profile shape is hung on it. **This is a mechanism for exp48's
"the profile does not matter"**: it does not matter *because* the scale runs
away, and a family shootout is only meaningful with the deposit scale pinned.

**(e) The widened fits are physically absurd and should not be adopted.**
`log_R50` = 5.0 means a deposit half-mass radius of 100 Mpc before the 300 kpc
ceiling clips it — i.e. every deposit pinned at the ceiling. The loss prefers
this; no physical process produces it. The `R50_CAP` is doing all the work in
those cells, which is an argument for a *lower*, physically motivated ceiling
rather than a wider box.

Caveat: `gauss-qfree` and `sersic-qfree` each hit the 2500-evaluation cap on
their nudged start (flagged in the logs). Both starts agree to <= 2e-6 in each
case, so the conclusion is unaffected, but those two widened losses are not
fully converged.

## Result 4 — the judged verdicts (all five epochs; fitted on z <= 1.5)

### The DEPOSITION-ONLY result, ranked by the added-light kernel

Only the two in-scope columns: transport removed (`q` = 0) against the adopted
incumbent (`q` free). The intermediate grid is in the out-of-scope appendix.

| cell | kernel [dex] | kern R50 mod/dat | M(<10) growth | GATE diff (0.37) | GATE overshoot | dlog R50 | plane z=0.4 |
|---|---|---|---|---|---|---|---|
| moffat-qfree (adopted) | 0.081 | 24/26 | 0.722 | 0.42 | +0.023 | +0.026 | 2.1 |
| **moffat-q0** | 0.086 | 24/26 | **0.879** | 0.36 | +0.074 STRAINED | +0.016 | 2.7 |

Removing transport moves the added-light kernel slightly the WRONG way
(0.081 -> 0.086), improves the `M(<10)` growth ratio (0.722 -> 0.879), leaves
the differential gate comfortable (0.36 against a measured 0.37) and STRAINS
the outskirt-overshoot gate (+0.074 against a +/-0.06 band) that the adopted
model passes. On the tier 2b plane at z=0.4 it is worse (2.7 vs 2.1); on the
mass-size plane it is BETTER (see Result 5).

### Stage B, ranked by the added-light kernel

| cell | kernel [dex] | kern n mod/dat | kern R50 | M(<10) growth | M(<5) cmp | beyond 500 kpc |
|---|---|---|---|---|---|---|
| moffat-qfree | **0.081** | 2.7/2.7 | 24/26 | 0.722 | -43.4% | 4.9% |
| moffat-q0 | 0.086 | 2.4/2.7 | 24/26 | 0.879 | -39.4% | 6.8% |
| sersic-q0 | 0.102 | 2.3/2.7 | 34/26 | **1.015** | -40.7% | 0.7% |
| gauss-qfree | 0.114 | 1.4/2.7 | 32/26 | 0.755 | -40.3% | 0.5% |
| sersic-qfree | 0.118 | **2.7/2.7** | 32/26 | 0.533 | -44.6% | 1.4% |
| expo-qfree | 0.122 | 1.9/2.7 | 34/26 | 0.664 | -42.0% | 0.9% |
| expo-q0 | 0.286 | 1.9/2.7 | 39/26 | 1.875 | -35.2% | 0.8% |
| **gauss-q0** | **0.339** | 1.7/2.7 | 34/26 | 2.227 | -34.2% | 0.5% |

**The falsification control fails exactly as designed.** `gauss-q0` — wingless
deposit, no transport — is worst by a factor of four, with the wrong kernel
shape (n = 1.7 against a measured 2.7) and 5% sign disagreement. The ladder is
monotonic in tail weight at `q` = 0.

**The Moffat wins on the MEASURED KERNEL too, not only on the loss** — and
`moffat-q0` (0.086) beats `sersic-qfree` (0.118), so the family matters more
than the transport setting.

**The deposit-reach column localizes exp52's "too much reach" to the FAMILY.**
The Moffat throws 4.9-6.8% of its deposited mass beyond 500 kpc where the
`M(<500)` normalization discards it; every lighter-tailed family puts 0.5-1.4%
there. The power-law tail is itself a wasteful-reach mechanism — in genuine
tension with the Moffat also matching the measured added light best. A lower,
physically motivated deposit ceiling is the obvious lever, and exp49 already
measured a 300 kpc ceiling as costing 1.5e-5.

## What exp53 settles, and what it does not

**Settled.**
1. **Removing transport does NOT fix the compact-galaxy central deficit.**
   Refitted, `moffat-q0` gives -39.4% against the incumbent's -43.4% — four
   points out of forty-three. exp52's mechanism claim survives in its narrow
   form (late outskirt growth IS redistribution) and fails in its causal form.
   The exp47/exp48 defect is unexplained again and its leading hypothesis is
   retired.
2. **The slice and the refit disagree by 57 points on that number** (+13.6%
   against -39.4%). A mechanism that is real in a model's internal bookkeeping
   is not thereby the CAUSE of that model's error.
3. **A deposition-only kernel is more credible than the loss suggests.** It is
   0.3 points behind the incumbent on the program's standard profile mark
   (18.3% vs 18.0%), BETTER at z = 1.5 and z = 2.0, and it improves the
   mass-size plane (E/floor 4.4 vs 5.2 at z=0.4). The 4.2% loss gap is
   concentrated at low redshift.
4. **Among deposition-only families the ranking is monotonic in tail weight**:
   moffat (0.160001) < sersic (0.162079) < expo (0.175218) < gauss (0.177770),
   and the same order on the added-light kernel. The wingless control fails as
   designed and catastrophically at z=2 (`gauss-q0` mass-mass slope +4.11
   against a measured +1.43, E/floor 56.9).
5. **The deposit scale is unidentified for every family except the Moffat**,
   and the objective does not constrain it from above — a mechanism for
   exp48's "the profile does not matter".

**Not settled.**
- **exp53's fits are IN-SAMPLE.** exp38 stage 2 used 10-fold CV; 14 cells x 3
  starts x 10 folds was not affordable. Parameter counts differ by at most 2 of
  ~11 against ~220k residuals, so optimism cannot plausibly reorder families
  separated by more than a fraction of a per cent — but `moffat-q0` vs the
  incumbent on the PROFILE mark (0.3 points) is inside that margin, and that
  comparison specifically needs CV before it is quoted as a near-tie.
- **One objective only** (production CoG + fractional residuals). exp48
  measured that the objective REVERSES the profile ranking, so the family
  ordering is conditional. See the scope-limit section below.
- **The generalized-sigmoid family was not tested** — `gompertz_log`,
  `loglogistic` and `richards` (which nests the first two). exp48 ranked
  `gompertz_log` 2nd of six on the judge with the BEST compact central mass of
  any candidate. See the scope-limit section.
- **The deposit profile SHAPE is frozen in time.** See the design-asymmetry
  section below — this is the largest untested structural freedom.
- `gauss-qfree` / `sersic-qfree` hit the rail-check evaluation cap on one start
  each; both starts agree to <= 2e-6, but those widened losses are not fully
  converged.
- `gausswing` (exp38 stage 1 co-winner) was not tested.

## Result 5 — the STANDARD QA battery on the deposition-only set

exp53's own figures answer exp53's own question. They are not a substitute for
`hongshao.qa.evaluate`, which is what every promoted model in this program has
been shown on. Run for all four deposition-only families plus the adopted
baseline (`report.py qa`): 10 figures each, binned by halo mass, scored at all
five epochs with z = 2.0 an extrapolation test.

**It changed the picture in two ways the exp53-specific scorers could not see.**

### tier 3 — profile max|rel| over R > 5 kpc, epoch-averaged

| model | avg | per epoch (z = 0.4 ... 2.0) |
|---|---|---|
| moffat-qfree (adopted) | **18.0%** | 19.3 / 18.6 / 18.1 / 17.3 / 16.6 |
| **moffat-q0** | **18.3%** | 20.6 / 19.9 / 19.1 / 16.4 / 15.5 |
| sersic-q0 | 19.5% | 19.7 / 20.2 / 20.4 / 19.4 / 17.9 |
| expo-q0 | 20.4% | 19.5 / 21.6 / 24.3 / 21.3 / 15.4 |
| gauss-q0 | 21.2% | 19.3 / 21.2 / 24.7 / 22.7 / 18.0 |

**The deposition-only Moffat is 0.3 points behind the adopted model on the
program's standard profile mark** — far closer than the 4.2% loss gap implies,
and it is BETTER at z = 1.5 and z = 2.0 (16.4/15.5 against 17.3/16.6). The
loss gap is concentrated at low redshift. (In-sample; exp38's CV marks are not
directly comparable.)

### tier 2d — the MASS-SIZE plane, the program's standing failure

E/floor for R50 (lower is better; the truth's own split-half floor is 1.0):

| model | z=0.4 | z=1.0 | z=2.0 | R50 slope at z=0.4 (truth +0.50) |
|---|---|---|---|---|
| **gauss-q0** | **2.9** | 11.5 | **5.9** | **+0.44** |
| **expo-q0** | **3.2** | 11.0 | 6.0 | **+0.44** |
| moffat-q0 | 4.4 | 5.1 | 8.1 | +0.32 |
| moffat-qfree (adopted) | 5.2 | 5.2 | 10.4 | +0.41 |
| sersic-q0 | 7.2 | 8.2 | 13.8 | +0.34 |

**Every deposition-only model except Sersic beats the adopted model on the
mass-size plane at z = 0.4**, and the two wingless ones nearly halve it
(5.2 -> 2.9) while getting the slope closer to the truth (+0.44 against the
adopted +0.41, truth +0.50). This is the standing gap the user has flagged
repeatedly — "the adopted product does NOT pass the mass-size plane; the
relation is too FLAT at every epoch". A deposition-only kernel improves it.

The reason is visible in the scatter column of tier 2b: the heavy-tailed models
are MORE under-dispersed (R50 scatter 0.074-0.090 against a truth 0.173) than
the light-tailed ones (0.111-0.113). The known over-coherence defect
(exp45/exp46) is WORSE for the Moffat.

### ... and the wingless models pay for it catastrophically at z = 2

tier 2b `M(<30)` vs `M(30-50)` at z = 2.0, E/floor:

| model | E/floor | model slope (truth +1.43) | model scatter (truth 0.338) |
|---|---|---|---|
| sersic-q0 | 6.9 | +1.32 | 0.148 |
| moffat-q0 | 7.0 | +1.18 | 0.089 |
| moffat-qfree | 7.1 | +1.08 | 0.072 |
| expo-q0 | **28.3** | +2.08 | 0.352 |
| gauss-q0 | **56.9** | **+4.11** | **1.009** |

A wingless deposit with no transport cannot extrapolate: `gauss-q0`'s z=2
mass-mass slope is +4.11 against a measured +1.43, with three times the truth's
scatter. The falsification control fails in the predicted way and then some.

**Net verdict.** The deposition-only Moffat is a genuinely credible model —
0.3 points off the adopted profile mark, better at high z, better on the
mass-size plane — and it was NOT visible as such from the loss (+4.2%) or from
the added-light kernel (0.086 vs 0.081) alone. The light-tailed deposition-only
models trade a real mass-size-plane gain at z <= 0.4 for an unusable z = 2
extrapolation. Neither displaces the `q` ~ 0.6-0.8 recommendation, which wins
on the kernel and both gates, but the mass-size result means **deposition-only
should stay on the table rather than be closed**, and it should be re-tested
with CV.

## Result 6 — THE OUTSKIRT TEST: can deposition alone build the stellar halo?

**This is what the branch is actually for (user, 2026-08-21).** The central
region was never the target. The question is whether a deposition-only kernel
reproduces the extended stellar halo of massive galaxies and its evolution as
well as deposition + transport does.

Measured in `outskirt.py`: annuli inside the measured 2-148 kpc grid (nothing
extrapolated, and `M(<500)` deliberately excluded since the model matches it by
construction), 2000 galaxy bootstraps, split by stellar mass, and — the biggest
accuracy gain — a **PAIRED per-galaxy** comparison against the incumbent, which
removes the galaxy-to-galaxy variance that dominates an unpaired one.

All numbers below are the **massive tercile** (logM* >= 11.42, n = 799) and the
**`M[50,148] kpc`** annulus. CIs are 95%.

### A. Structure — how much halo mass is there

median dlog10(model/truth) [dex]:

| model | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| moffat-qfree (incumbent) | +0.007 | +0.001 | +0.008 | +0.010 | +0.030 |
| moffat-q0 | **-0.043** | **-0.058** | **-0.058** | -0.019 | +0.117 |
| **sersic-q0** | **+0.027** | **+0.018** | **+0.009** | **-0.006** | **+0.019** |
| expo-q0 | +0.020 | +0.073 | +0.053 | **-0.500** | **-2.268** |
| gauss-q0 | +0.003 | +0.065 | +0.089 | **-0.740** | **-6.199** |

### B. Evolution — how fast the halo GROWS (population-summed model/truth)

| model | z0.7->0.4 | z1.0->0.7 | z1.5->1.0 | z2.0->1.5 | **z2.0->0.4** |
|---|---|---|---|---|---|
| moffat-qfree (incumbent) | 1.071 | 1.016 | 1.133 | 0.992 | **1.062** |
| moffat-q0 | 1.020 | 0.920 | 0.820 | 0.702 | **0.885** |
| **sersic-q0** | 1.110 | 1.118 | 1.076 | 0.938 | **1.073** |
| expo-q0 | 0.739 | 1.285 | 1.951 | 0.710 | 1.212 |
| gauss-q0 | 0.626 | 0.981 | 2.266 | 0.543 | 1.165 |

### C. Head to head, PAIRED per galaxy

median(|dlog| deposition-only - |dlog| incumbent) [dex]; NEGATIVE = the
deposition-only model is CLOSER; `w` = fraction of galaxies it wins:

| model | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| moffat-q0 | +0.009 w43% | +0.014 w42% | +0.009 w45% | -0.009 w57% | +0.006 w47% |
| **sersic-q0** | **+0.002 w47%** | **+0.000 w50%** | **-0.003 w54%** | **-0.017 w59%** | **-0.053 w67%** |

### The answer

**YES — deposition alone can build the stellar halo, but only with the right
profile, and WHICH profile matters far more than whether transport is on.**

1. **`sersic-q0` matches the incumbent, and beats it at high z.** Its halo mass
   is within +/-0.03 dex of the truth at every epoch (incumbent: +0.00 to
   +0.03), it grows the halo at **1.073x** the truth's rate over the full
   z=2->0.4 baseline against the incumbent's 1.062x, and per galaxy it is
   statistically indistinguishable from the incumbent at z <= 0.7 and
   **significantly closer at z = 1.0, 1.5 and 2.0** (-0.053 dex, winning 67% of
   galaxies at z=2). **A deposition-only Sersic kernel reproduces the extended
   stellar halo and its evolution as well as deposition + transport, and
   extrapolates better.**
2. **`moffat-q0` does NOT.** It carries a systematic ~12% halo mass deficit at
   z <= 1 (-0.043 to -0.058 dex, CIs excluding zero) and under-grows the halo,
   worsening with lookback (0.820 at z1.5->1.0, 0.702 at z2.0->1.5) to
   **0.885x** over the full baseline. The Moffat genuinely needs its transport
   term to reach.
3. **Wings are ESSENTIAL.** Without transport the two light-tailed families
   collapse at z >= 1.5 — `expo-q0` reaches -2.27 dex and `gauss-q0` -6.20 dex
   at z=2, i.e. a halo one part in 10^2 to 10^6 of the truth's. This is the
   cleanest demonstration in the program that the deposit profile's TAIL is
   what builds the stellar halo.

### Why the Moffat loses the outskirt while winning the loss

`moffat-q0` beats `sersic-q0` on the loss (0.160001 vs 0.162079) and on the
added-light kernel (0.086 vs 0.102), and LOSES the outskirt decisively. The
deposit-reach measurement explains it: **`moffat-q0` puts 6.8% of its deposited
mass beyond 500 kpc, where the `M(<500)` normalization discards it; `sersic-q0`
puts 0.7% there.** The Moffat's power-law tail is genuinely heavier, but it
spends that weight OUTSIDE the aperture, so the mass is lost from the halo
budget rather than delivered to 50-148 kpc. The Sersic's n = 3.41 wings put
theirs inside.

That also sharpens exp52's "right source, ~10x too much reach": for a
deposition-only model the wasted reach is a property of the PROFILE FAMILY, and
choosing a family whose tail lands inside the aperture fixes it without any
ceiling.

### Caveat

`sersic-q0` overshoots the 30-60 kpc surface density (+0.040 to +0.175 dex,
rising with redshift) and undershoots at 100-148 kpc (-0.09 dex at z >= 1.5):
its halo has the right MASS with a slightly too-steep radial slope. The
incumbent is flatter in the far halo. So "as well as" holds for the mass budget
and its growth, not for every detail of the profile shape.

## Scope limit: exp53 ran ONE objective, and exp48 measured that the objective
## reverses the profile ranking

Every exp53 fit used the PRODUCTION objective — `Objective()`, i.e. the curve
of growth with FRACTIONAL residuals. It is not the log-density objective.
That matters more here than it would elsewhere, because exp48 step B measured
that **the objective REVERSES the profile ranking**: `sersic_r50` is worst
under the production objective (0.15767 vs Moffat 0.15362) and *tied best*
under exp48's winner (surface density with a LOG residual).

So exp53's family ranking — Moffat > Sersic > expo > gauss, on both loss and
added-light kernel — is **conditional on the production objective**. It agrees
with exp48's judged ranking under a different objective, which is reassuring,
but exp53 did not test the density-log objective and cannot claim the ordering
is objective-independent. `hongshao.objective.Objective(quantity="density",
residual="log")` makes the re-run a one-line change; the cost is another full
fit set.

Related: **`gompertz_log` was NOT tested in exp53, and it should have been.**
exp48 ranked it SECOND of six on the judge (2.05 against Moffat's 1.95) with
the **best compact-galaxy central mass of any candidate** (-27.7% against
Moffat's -31.4%) and the most comfortable differential gate (0.39/0.13). Since
exp53's central finding is that the compact deficit is transport-independent
and stuck near -39% to -47%, the one profile family that has ever moved that
number is exactly the one exp53 omitted. Its CoG is
`M(<R)/Mtot = exp(-b R^-c)`, which has a genuine power-law tail like the
Moffat but a super-exponentially emptied centre — so it is not obviously
suited to fixing a central DEFICIT, and exp48's gain may come from the
objective rather than the shape. Worth testing rather than assuming either way.

## A DESIGN ASYMMETRY exp53 inherited and did not test: scale is rich, shape is frozen

Raised by the user, 2026-08-21. Both Moffat and Sersic are TWO-parameter
families, and exp53 did fit both parameters of each — the answer to "did you
only vary one?" is no. But the two are given wildly unequal freedom, and that
inequality is inherited from exp35/exp38, never measured.

| | how many numbers control it | varies with halo? | varies with epoch? |
|---|---|---|---|
| **scale** `R50` | **5** — `log_R50`, the time exponent `g`, and 3 conditioning slopes on `[logMh, c200c, fz2]` | **yes** | **yes**, as `R50(t_i) = 10^log_R50 (t_i/t_obs)^g` |
| **shape** `gam` / `n` | **1** — a single population constant | no | **no** |

So the deposit's SIZE is a halo-conditioned power law in formation time, while
its SHAPE is one number shared by every galaxy, every halo mass and every
epoch. `expo` and `gauss` have no shape freedom at all by construction (n fixed
at 1 and 0.5); they are the controls.

**"Why ignore Re in Sersic?" — it is not ignored; it is `log_R50`+`g`+slopes,
the richest part of the model.** What is missing is the reverse: the shape has
none of that structure.

### Is a redshift-dependent shape worth building?

**Yes — and it deserves its own branch, not a footnote here (user, 2026-08-21).**
An earlier draft of this README leaned toward "the existing measurement does
not resolve it, so expect it to be weakly constrained". That conclusion is
withdrawn: what follows is a statement about the RESOLUTION of one coarse
measurement, not about the physics.

What the measured added-light kernel (exp38 stage 0.2) actually shows:

| epoch pair | n (T1) | n (T2) | n (T3) | mean | within-epoch spread |
|---|---|---|---|---|---|
| z0.7->0.4 | 2.87 | 2.27 | 2.08 | 2.41 | 0.79 |
| z1.0->0.7 | 2.67 | 2.27 | 3.06 | 2.67 | 0.79 |
| z1.5->1.0 | 2.67 | 2.67 | 2.87 | 2.74 | 0.20 |
| z2.0->1.5 | 2.67 | 2.67 | 2.87 | 2.74 | 0.20 |

Early-minus-late is +0.20 against a typical within-epoch tercile spread of 0.49
and an `n`-grid resolution of 0.197 — so exp38's stage 0.2 grid CANNOT resolve
a trend of this size. That is a reason to measure it better, not a reason to
assume it is absent. Note also that the terciles are by stellar mass, so part
of the "within-epoch spread" is a real mass dependence the model does not have
either — a shape conditioning row is the natural companion test.

Design sketch for the branch: add `shape(t_i) = shape_0 + shape_1 *
log10(t_i/t_obs)` (one parameter, slots into `Spec.unpack` exactly like the
existing rows), and separately a shape conditioning row on the standardized
halo vector. Nesting is clean — `shape_1 = 0` recovers exp53 exactly — so the
fitted `shape_1` IS the measurement. Pair it with a finer re-measurement of the
stage 0.2 kernel (a denser `n` grid and a bootstrap over galaxies) so the fit
has something to be checked against.

## The generalized-sigmoid FAMILY, not just Gompertz

Noted by the user, 2026-08-21: exp48 explored a whole family of growth-curve
sigmoids in log radius, and exp53 tested none of them. From
`exp48/gompertz.py` and `exp48/profiles.py`:

| family | CoG | params | note |
|---|---|---|---|
| `gompertz_log` | `exp(-b R^-c)` | 2 | Frechet; power-law tail like Moffat, super-exponentially emptied centre |
| `loglogistic` | `1/(1 + X^-k)` | 2 | |
| **`richards`** | generalized logistic in log R | **3** | **NESTS both above**: nu -> 0 gives `gompertz_log`, nu = 1 gives `loglogistic` |
| `gompertz_lin` | `exp(-b e^-cR)` | 2 | exponential-ish cutoff, NO power-law tail |
| `cuspy` | `(R/rc)^-alpha [1+(R/rc)^2]^-gamma` | 3 | **NESTS Moffat** at alpha = 0; the cusp the incumbent cannot make |

**Richards is the efficient entry point**: fitting it tests `gompertz_log` and
`loglogistic` simultaneously as nested special cases, so the fitted `nu` is
itself the measurement of which sigmoid the data want. `cuspy` plays the same
role for the Moffat.

Why this matters for exp53 specifically: exp48's judge ranked `gompertz_log`
2nd of six with the **best compact-galaxy central mass of any candidate**
(-27.7% against Moffat's -31.4%) and the most comfortable differential gate.
exp53's central finding is that the compact deficit is stuck near -39% to -47%
whatever the transport setting — so the one family that has ever moved that
number is exactly the one exp53 omitted. (exp48 also found `cuspy` fits
alpha = +0.814 and still scores worse, so the cusp hypothesis is separately
dead; that does not bear on the sigmoids.)

## Appendix (OUT OF SCOPE for this branch) — the profiled `q` grid

**Scoping decision (user, 2026-08-21): this is out of scope for exp53 and is
retained only as a record.** exp53's requirement is a DEPOSITION-ONLY test —
transport simply turned off — and the intermediate `q` grid answers a
different question (how a transport model should be tuned), complicating the
one exp53 was asked. **Nothing in this appendix should be read as an exp53
conclusion, and the "trim `q` to 0.6-0.8" recommendation it produced is
withdrawn from this branch's findings.** It belongs to a transport-tuning
experiment, and if it is wanted it should be re-run as one.

The `q = 0` and `q` free columns below are in scope and are carried into the
main results above; the intermediate points are not.

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
