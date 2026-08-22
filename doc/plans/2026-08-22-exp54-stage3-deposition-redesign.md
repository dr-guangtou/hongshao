# exp54 Stage 3 — redesigning the deposition model

**Status: PLANNED.** This is the model-building half of exp54. Stages 0-2 are
complete and shipped (`experiments/exp54_unpinned_amplitude/`); this document
covers Stage 3, which **replaces the physical kernel** rather than adjusting it.

Companion documents: `2026-08-22-exp54-unpinned-amplitude.md` (the parent plan,
revision 2) and `2026-08-22-exp54-evidence-probe.py` (reproduces its evidence).

---

## 1. Context — how we got here

exp53 closed with the adopted kernel classified as an **oracle-amplitude shape
model**: it pins `M_model(<500 kpc) == M_TNG(<500 kpc)` per galaxy per epoch and
normalizes its deposit weights to sum to 1, so it carries no absolute stellar
mass at all. exp54 was opened to remove that pin.

Removing it turned out to be the smaller half of the problem. Once the pin is
gone, every remaining ingredient has to justify itself as physics — and **four
of the five load-bearing ingredients cannot**. That is the motivation for a
redesign rather than a patch, and each item below is a measurement made in this
branch, not an opinion.

### 1.1 What the audit found

**(a) The efficiency window carries no mass, and its form is unmotivated.**
`walkthrough.py` scaled `w(z)` by 1000 and the predicted profile changed by
**3.3e-16** — the amplitude is destroyed twice over, by `dM = w/w.sum()` and
again by the pin. What `w` actually controls is the RADIAL distribution: moving
its centre `mu` from 1.0 to 2.0 runs the half-light radius from **41.8 to 2.0
kpc, a factor of 20**, while the pinned mass is exactly invariant. So `w` is a
radial mixing weight wearing the name of an efficiency.

Its lognormal form has no physical derivation. The record shows the original
(exp25/exp29) efficiency was a **broken power law in (1+z)** — monotone — which
was replaced around exp33/exp35. **exp36 already flagged the replacement as a
defect** ("the efficiency form is too rigid: the lognormal f(z) peak drifts
between z04-only and multi fits, peak z 2.7 vs 4.0 — the only ingredient that
does not land in one basin") **and already prescribed the fix** ("replace the
lognormal f(z) with a monotone low-order spline"). It was never done.

**(b) The deposit size law is arbitrary and badly posed.**
`R50_i = 10^log_R50 (t_i/t_obs)^g`. `t_obs` is a GLOBAL constant, so the
normalization changes nothing but the meaning of the intercept — which lands at
**1253 kpc**, extrapolated to a time when almost no deposits form, with the
regression pivot at the far EDGE of the deposit time range and `g` railed at
4.0. Measured over 42405 deposits, the fraction of the halo's own virial radius
`f = R50/R200c` runs **0.0013 at z=8-20 to 0.49 at z=0.5-1, spread 0.913 dex** —
about 3x steeper in log than virial growth. It places the earliest deposits at
11 pc.

**(c) The conditioning form fails its one physical check.**
`parameter = base + slopes . standardized halo properties` is a first-order
Taylor expansion, not a derivation. The only physical expectation available is
that galaxy size tracks virial radius, `R_vir ~ Mh^(1/3)`, so the `log R50` vs
`logMh` slope should be near **+0.333**. The adopted fit gives **-0.059 per
dex** — essentially zero, wrong sign. exp49 had already measured these six
slopes as the least-determined part of the model.

**(d) The profile family's tail is physically absurd.**
Stage 0 check 4b: the fitted Moffat has `gam = 1.3784`, an outer surface-density
slope of **-2.76**, so enclosed mass converges only as `R^-0.76`. **A deposit
with a 50 kpc half-mass radius needs 9.6 Mpc to enclose 99% of its own mass**
and 201 Mpc for 99.9%. This is why **18.0% of every galaxy's deposited stellar
mass lands beyond 500 kpc** (Stage 0 check 4) — silently discarded and restored
by the pin. Under the pin "the deposit's total mass" was never a well-posed
quantity, which is why nothing noticed.

**(e) The one ingredient that survives: transport does a real job.**
exp53's sharpest result is that deposition-only models carry a radial
distribution error that **flips sign with halo mass** (`M(<148)` -3% at low halo
mass, +6% at high, a 9-point tilt where the fiducial is flat), and concluded
that transport is what supplies the halo-mass-dependent concentration. The
redesign deletes transport, so **something else must supply that**, and §4.4
says what.

### 1.2 The quantitative case for a redesign

Stage 1's shuffled-MAH control is the decisive number. Permuting whole MAHs
within narrow final-mass bins and **refitting**, at z=0.4:

| model | real MAH | shuffled MAH | history information used |
|---|---|---|---|
| DiffMAH 4 params + linear regression | 0.1130 | 0.1228 | **0.0098 dex** |
| the 3-parameter deposition law | 0.1231 | 0.1247 | **0.0016 dex** |

**The physical law extracts about one sixth of the assembly-history information
that a plain linear regression on the same four numbers extracts.** A single
integral of a smooth efficiency against `dMh` cannot express what the MAH
supports. That is a statement about the functional form, and it is the
strongest single argument for rebuilding rather than tuning.

### 1.3 What Stage 2 says about the stakes

Re-baselining measured which conclusions actually depend on the pin.
**Population-level results are robust**: the tier 2b plane at z=0.4 moves
2.17 -> 2.20 under a realistic, cross-epoch-correlated amplitude error (the
parent plan predicted 2.1 -> 2.9; that estimate applied a paired operation to a
pairing-blind metric). **Per-galaxy results are not**: tier 3 profile max|rel|
degrades +38% at z=0.4 and +91% at z=2.0. And the pin's real hiding place was
the **cross-epoch growth tier**, where `Mtot` growth energy ratios sat at
0.27-0.33 — below the truth's own split-half floor, "indistinguishable from
reality" by construction — and are 1.22-1.60 once unpinned.

So: the redesign is not required to rescue the population-level science. It is
required to make the model **deployable** (it cannot run on an N-body halo at
all), to make its **mass growth** an honest prediction, and to replace four
ingredients that survive only because nothing ever scored them.

---

## 2. The model

The user's specification (2026-08-22), which this plan implements:

> Given a MAH and halo properties, between two epochs a fraction of the halo
> mass increase converts into stellar mass. That fraction can depend on
> redshift or more. The new stellar mass follows a functional form for its
> radial distribution, with a few parameters, some or all depending on halo
> mass and redshift. The integral of that profile equals the stellar mass the
> efficiency predicts. To predict the distribution at a given epoch, sum the
> profiles from all past epochs.

Formally, for galaxy `i` observed at epoch `t_k`:

```
Delta M*_j     =  eps(H_j) * Delta Mh_j                      [1]  the budget
Sigma_j(R)     =  Delta M*_j * rho( R ; theta_prof(H_j) )    [2]  the placement
M*(<R, t_k)    =  SUM_{t_j <= t_k}  Delta M*_j * F( R ; theta_prof(H_j) )  [3]
```

where `H_j` is the halo state **at the deposit's own formation time** `t_j`,
`F` is the unit-mass cumulative profile, and `rho` integrates to 1 by
construction so [2] conserves mass exactly. **No normalization, no rescaling,
no reference to any stellar quantity.** Stage 0's poisoned-data test is the
acceptance criterion: NaN every stellar array and the forward call must still
return finite masses in solar masses.

**Deleted from the incumbent**: the truth pin; `dM = w/w.sum()`; the transport
term (`q`, the core fraction `fc`); the `t_obs` normalization; the catalog
`logMh(z=0.4)` input; the 500 kpc reference radius.

---

## 3. Design axes and candidate forms

Each axis is built as a **switchable, nested family** in the exp53 `Spec`
pattern, so candidates are compared inside one harness and the simplest form is
always a special case of the richer one. Nesting matters: it makes "does this
term earn its place" a question the identifiability analysis can answer.

### 3.1 The efficiency `eps(H)`

Base coordinate: `x = ln(1+z_j)`, `m = logMh(t_j) - 13.5`.

| id | form | rationale | status |
|---|---|---|---|
| **E1** | `log10 eps = a0 + a_M m + a_z x` | the measured base; 3 params, CV-ties the halo-mass baseline, bias <=0.004 dex | **measured** |
| **E2** | E1 `+ a_Mz m x` | the mass dependence tilts with redshift; measured to help at every epoch | **measured** |
| **E3** | E1/E2 `+ a_c * dlogc(t_j)` | concentration excess over Diemer19; measured to help | **measured** |
| **E4** | E1 with `a_z x` -> free cubic spline in `x`, 3-5 knots | exp36's prescription (D), WITHOUT its monotonicity constraint — let the data say whether a peak exists | new |
| **E5** | double power law in mass: `eps = 2 eps0(z) / [ (M/M1)^-beta + (M/M1)^gamma ]` | the standard SHMR form (Moster/Behroozi); contains a peak in MASS | new |

**Recommendation: E1 as the base, E2/E3/E4 as switchable extensions; E5 built
but not required for this sample.** Scope statement to carry: this sample is
logMh = 13.01/13.29/14.36 (1/50/99th percentile), entirely above the SHMR peak
at ~10^12, so a single power law in mass is adequate **here** and E5 matters
only if the model is later applied below the peak.

**Do NOT impose monotonicity in redshift.** The measurement says only that a
peak is *unconstrained* by the mass budget, not that it is excluded — the free
lognormal degenerated to the power law rather than rejecting it. E4 plus the
identifiability report is the honest way to settle it.

**PARAMETERIZATION TRAP, measured**: never stack the `a_Mz` cross-term on a free
lognormal window. That combination reaches the same loss at `a0 = 26.8,
mu = 92.5, sig = 7.93` — meaningless values a loss check cannot catch. On the
monotone base the same term is well-behaved (`a_Mz = +0.268`).

### 3.2 The deposit profile `F`

**A new hard requirement, from Stage 0 check 4b.** Any candidate family must
satisfy a **convergence constraint**: the radius enclosing 99% of a single
deposit's mass must not exceed a fixed multiple of the halo's virial radius at
that time,

```
R99( theta_prof )  <=  C * R200c(t_j),     C a fixed constant (start at C = 3)
```

This is a physical statement — stars are not deposited a hundred virial radii
out — and it replaces the arbitrary 300 kpc ceiling with something dimensionally
sensible. **It immediately excludes the incumbent Moffat at its fitted `gam`**
(R99 = 192 R50 ~ 9.6 Mpc against R200c ~ 0.5 Mpc). Implement it as a penalty or
a bound, and report which candidates it binds.

| id | form | rationale |
|---|---|---|
| **P1** | Sérsic in R50 coordinates, `n` free | finite well-behaved tail; exp53 measured `sersic-q0` putting only 0.48% beyond 500 kpc against Moffat's 6.61% |
| **P2** | generalized sigmoid (`gompertz_log` / `richards`) | exp53's best added-light kernel of eleven (0.072 vs the fiducial 0.081), winning 10/13 metrics; `richards` fitted `nu = 0.345` |
| **P3** | Moffat with `gam` bounded so the convergence constraint holds | keeps the heavy tail the incumbent liked, but forces it to converge |
| **P4** | any of the above times a truncation at `C * R200c(t_j)` | enforces the constraint by construction rather than by penalty |

**Recommendation: P1 and P2 as the primary shootout, P3 as the incumbent's
honest descendant, P4 as the fallback if the penalty proves hard to fit.** All
are already implemented in `exp53/families.py` in a common `R50`
parameterization, which is what makes the comparison meaningful.

Two conventions to state and keep: the deposit is a **projected surface
density** (the data is a projected curve of growth, and exp48's objective
differences both sides rather than evaluating the model analytically); and the
profile is evaluated in `x = R/R50_j`, so the size law alone carries the units.

### 3.3 The deposit size law

| id | form | rationale |
|---|---|---|
| **S1** | `R50_j = f0 * R200c(t_j)` | one dimensionless number; "deposits form at a fixed fraction of the virial radius"; builds in `Mh^(1/3)` |
| **S2** | S1 `* (1+z_j)^b` | `b` measures the DEVIATION from virial scaling; **`b = 0` is a testable null**; the incumbent implies `b ~ -3` |
| **S3** | S2 with `log f_j = log f0 + slopes . cond(t_j)` | the conditioning layer, now acting on a physically normalized quantity |
| **S4** | fallback: keep the time power law, re-pivot to the mass-weighted median deposit time (~2 Gyr) | cosmetic but decorrelates intercept from slope and makes the base readable |

**Recommendation: S2 as the base, S3 for conditioning, S4 only if `R200c` turns
out to be unusable.** `Group_R_Crit200` is on disk at **all 73 snapshots**
(`exp46_highz_ridge/outputs/so_history.npz`, 96.4% finite), in ckpc/h; convert
with `/h/(1+z)`.

The gain from S1/S2 is not only physical readability: with `R200c` carrying the
mass scaling, the conditioning slope on `logMh` **measures the deviation from
virial scaling** instead of having to reproduce it, which is a far better-posed
estimation problem than the one that produced -0.059 per dex.

### 3.4 Conditioning, and the job transport used to do

`cond(t_j)` = standardized, epoch-local, evaluated at the deposit's own
formation time:

```
cond(t_j) = [ logMh(t_j),  dlogc(t_j),  f_form(t_j) ]
dlogc(t_j)  =  log10[ c200c(t_j) / c_Diemer19(Mh(t_j), z_j) ]      the EXCESS
f_form(t_j) =  a formation-time measure from the MAH TRUNCATED at t_j
```

All three are available: `logMh(t_j)` from the official DiffMAH curve;
`dlogc` from `exp54/outputs/halo_structure_history.npz` (16 snapshots,
interpolated in `ln(1+z)`, **0 outside their support** — "assume typical for its
mass and redshift", recorded rather than silently imputed, covering 57.8% of
deposits); `f_form` computed on the fly. The concentration excess rather than
raw `c200c` because Diemer19 is deterministic in `(M,z)` and therefore
redundant with `logMh(t_j)`, while the excess carries 100% of the measured
signal (Stage 1: 0.1102 either way).

**The transport job.** exp53 measured that transport supplies the
halo-mass-dependent concentration, and this design deletes transport. The
replacement is the `logMh(t_j)` slope in S3: the size law's mass dependence must
now generate the tilt that transport used to. **This is the redesign's principal
risk and it gets a dedicated test** — re-measure exp53's `M(<148)` tilt across
halo-mass terciles and require it to be flat, not merely small.

### 3.5 Optional: two channels (in-situ and accreted)

Massive galaxies are two-phase objects: a compact in-situ component formed
early, and an extended envelope accreted later. The incumbent's single channel
plus transport is a way of mimicking that with one population; **an explicit
split is the more physical statement and is tractable without merger
granularity**, because the in-situ fraction can be a function of `(Mh, z)`:

```
Delta M*_j  ->  f_in(Mh_j, z_j) * Delta M*_j   at  R50_in  = f0_in  * R200c(t_j)
            +  (1 - f_in)       * Delta M*_j   at  R50_ex  = f0_ex  * R200c(t_j)
```

exp36 tried a two-channel split under the pin and it did not decide the
question; under an absolute amplitude it is a different experiment. **Build it,
but only after the single-channel version passes Stage 0 and the objective is
settled** — it doubles the profile parameters and is the most likely source of
degeneracy.

**Known limitation to state, not solve**: DiffMAH is a smooth fit, so it removes
the merger granularity that actually controls ex-situ mass. Residual scatter is
expected even in a correct model, and `f_in` is a population-average statement.

---

## 4. The objective

Unchanged from the parent plan §4.3, restated because it now applies to a model
that supplies its own amplitude:

```
A        =  log10 M*(<100 kpc)                    the amplitude
F(<R)    =  M*(<R) / M*(<100 kpc)                 the shape
L        =  lambda_A * L_A  +  lambda_F * L_F
L_A      =  ( A_model - A_data )^2                dex^2, per galaxy per epoch
L_F      =  Objective() applied to F              the production loss, on shapes
```

Two things this plan adds.

**The amplitude floor is epoch-dependent, and it is now measured.** Use
`lambda_A(z_k) = 1/sigma_A,floor(z_k)^2` with the Stage 1 best halo-only row:
**0.1047 / 0.1359 / 0.1417 / 0.1456 / 0.1744** dex at z = 0.4 / 0.7 / 1.0 / 1.5
/ 2.0. Without this the high-z epochs dominate `L_A` merely because amplitude is
harder there.

**Translating the record.** `L_F` on `F` is the production loss with the pin
moved from 500 to 100 kpc, and Stage 0 measured that offset: **-0.027233
(-14.69%)**, production loss 0.185429 -> 0.158196. Every shape loss in
exp38/40/47/48/53 converts with it.

**Weight scan**: `lambda_A/lambda_F` over {1/4, 1, 4} around the floor-normalized
choice, reported. The weighting must be an evidenced choice.

**Mass growth needs no extra term** — scoring `L_A` at five epochs scores the
growth automatically. But it must be REPORTED separately, because Stage 2 showed
it is where the pin hid: `Mtot` growth energy ratio 0.27-0.33 pinned versus
1.22-1.60 honest.

---

## 5. Identifiability protocol — mandatory, every fit

Agreed with the user (2026-08-22): **do not cap the parameter count; measure
identifiability instead.** Overfitting is not the binding constraint —
2397 x 5 x 24 = 287,640 data points, and the 3-parameter law measured in-sample
0.1505 against CV 0.1506, a zero generalization gap. (Caveat: the 24 radii are
~93% correlated and the profile is 2-3 dimensional, so the effective constraint
count is nearer 36,000. Still far above any parameter count contemplated here.)

**But "the fit will decide" holds only for identifiable parameters, and the loss
cannot tell you which those are.** Three counterexamples from this project:
exp53's three families re-railing at a widened bound with losses converging to
within 1e-4 ("nothing chooses the answer"); exp49's six conditioning slopes
differing by up to 0.022 between fits agreeing to 0.015 on the base; and this
branch's cross-term reaching the *same loss* at `a0 = 26.8, mu = 92.5`.

Every fit ships:

1. **Hessian at the optimum**, eigendecomposed; report the eigenvalue spectrum.
2. **Effective parameter count** = eigenvalues above a noise-set threshold.
3. **Profile likelihood** for every parameter whose value will be quoted: scan
   it, re-optimizing the rest. A flat profile means unidentified.
4. **Rail check by ESCAPE, not by loss** — exp53's lesson: one cell gained 0.37%
   and was flagged benign while still sitting on the new bound.
5. **Eigenvector composition**, so the report states which COMBINATION is
   determined rather than which parameter.

**The rule**: a parameter that fails may remain in the model, but **its value is
never quoted or interpreted**. Practical corollary: flat directions make
Nelder-Mead wander, which is why the sigmoid family's multi-start spread is
1.7e-3 against 1e-8 elsewhere. Multi-start stays mandatory.

---

## 6. Validation sequence

**3a — single epoch, single channel.** Fit E1 x P1 x S2 at z=0.4 only. Pass
Stage 0's contract, mass conservation, and the convergence constraint. Ship the
identifiability report. This is the smallest thing that can be right.

**3b — the form shootout.** E1/E2/E3/E4 x P1/P2/P3 x S2/S3, single epoch,
ranked **by the judge, never by loss**. Report which combinations the
convergence constraint binds.

**3c — multi-epoch.** All five epochs, the split objective with epoch-dependent
amplitude floors, the weight scan. **The dedicated transport-replacement test**:
`M(<148)` tilt across halo-mass terciles must be flat.

**3d — two channels**, only if 3c leaves a structured residual that a second
component plausibly explains.

**3e — deployment scoring** (the parent plan's Stage 4): three different
problems scored separately — transfer to new haloes at a fitted epoch, redshift
extrapolation for tracked descendants, and epoch-local prediction for an
independent halo catalogue. Only the third is the stated use case.

---

## 7. Pre-registered predictions

1. **The convergence constraint will bind on the Moffat and not on the
   Sérsic.** `moffat-q0` puts 6.61% of deposits beyond 500 kpc against
   `sersic-q0`'s 0.48%; the constraint formalizes that difference. If it binds
   on every family, `C = 3` is too tight and must be re-derived, not relaxed.
2. **`b` in S2 will be significantly negative but far smaller in magnitude than
   the incumbent's implied -3.** The incumbent had to use the time exponent to
   do a job that `R200c(t_j)` now does directly.
3. **The `logMh` conditioning slope in S3 will be small**, because `R200c`
   already carries `Mh^(1/3)`. A slope near zero here is SUCCESS, the opposite
   of its meaning in the incumbent.
4. **The efficiency spline (E4) will not find a significant peak** in
   `ln(1+z)`, and its identifiability report will show the peak's location as a
   flat direction. If it DOES find one, that is a genuine discovery and the
   monotone forms are wrong.
5. **Deleting transport will re-open exp53's halo-mass tilt**, and the S3 mass
   slope will be what closes it. If the tilt survives a free S3, transport was
   doing something the size law cannot express, and the two-channel split (§3.5)
   becomes necessary rather than optional.
6. **The redesign will use more MAH-shape information than the incumbent law's
   0.0016 dex** under the shuffle control, but will not reach the linear
   regression's 0.0098 dex. Reaching it would mean the physical form has become
   as expressive as an unconstrained fit, which would be surprising.

---

## 8. Open decisions

These need the user before 3a starts.

1. **The convergence constant `C`.** `R99 <= C * R200c(t_j)` with `C = 3` is a
   proposal, not a measurement. Alternatives: derive `C` from the truth by
   measuring where TNG's own stellar mass ends relative to `R200c`, or drop the
   hard constraint and simply REPORT `R99/R200c` per candidate.
2. **Whether `f_form` earns its place** in `cond`, given `logMh(t_j)` and
   `dlogc(t_j)` are already there and Stage 1 showed `fz2` adding 0.004-0.015
   dex on top of concentration.
3. **Whether to build E5 (the double power law in mass) now or later**, given
   this sample never samples the SHMR peak.

---

## 9. What success looks like

Under the agreed criterion — **deployability, not victory** — Stage 3 succeeds
if it produces a model that:

- passes Stage 0's poisoned-data test, so it runs on an N-body halo catalogue;
- conserves mass exactly and keeps its deposits inside a physically sensible
  radius;
- predicts stellar mass with an amplitude error near the Stage 1 halo-only
  floor (0.1047 dex at z=0.4), with **honest** growth (`Mtot` growth energy
  ratio reported, not pinned to 0.3 by construction);
- has an identifiability report for every quoted parameter;
- and whose remaining defects are stated in physical terms rather than absorbed
  by a normalization.

It does **not** have to beat the statistical emulator. The emulator predicts
aperture masses; this predicts a galaxy, its profile, and its history, from a
halo — and can be wrong in ways that teach something.
