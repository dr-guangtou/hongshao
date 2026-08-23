# The deposition-only stellar mass model: setup, options, scoring, results

*Status as of 2026-08-23, after exp54 Stages 0–3.4. Supersedes the summaries in
`experiments/exp54_unpinned_amplitude/README.md`, which this document
consolidates and explains.*

This note is meant to be readable by someone who has not been in the room. Every
symbol, every code name (`E2`, `S3`, `gompertz_log`), and every metric is
defined where it first appears. Where a name is a piece of internal shorthand
rather than a standard term, that is said explicitly.

---

## 1. What the model is trying to do

**The question.** Given only the *dark matter halo's* history — how its mass
grew over cosmic time, how big it was, how concentrated — can we predict how
much stellar mass a massive galaxy has, and how that stellar mass is
distributed with radius, at five different cosmic epochs?

**The data.** 2397 galaxies from the TNG300 cosmological simulation, selected at
redshift *z* = 0.4 with stellar masses `log10 M*` between 10.66 and 12.36. For
each galaxy we have:

- its **mass accretion history** (MAH): the halo mass `Mh(t)` at 73 snapshots,
  from the smooth DiffMAH fit;
- its **halo structure** at 16 snapshots: `R200c`, the NFW concentration
  `c200c`;
- its **curve of growth** (CoG) at five epochs — see below.

**Curve of growth (CoG), defined.** `M*(<R)`, the total stellar mass inside a
sphere of radius `R`, measured on a fixed 24-point radial grid from 2 kpc to
148.22 kpc. It is cumulative, so it only ever rises with `R`. This is the thing
the model must reproduce.

**The five epochs.** *z* = 0.4, 0.7, 1.0, 1.5, 2.0 — snapshots 72, 59, 50, 40,
33. At the four higher redshifts we follow each galaxy's **main progenitor**,
the most massive branch of its merger tree.

**"Deposition-only", defined.** The model builds a galaxy by *depositing*
stellar mass at each step of the halo's growth and never moving it again. Each
deposit is a small blob of stars laid down with a fixed radial profile, and the
galaxy at any later time is the sum of all deposits laid down up to then. There
is no subsequent transport, no re-arrangement, no dynamical evolution. That is
a strong assumption, and it is the point: it is the simplest thing that could
work, and Section 8 reports where it breaks.

---

## 2. The model, in full

### 2.1 The three equations

For one galaxy, let `t_j` index the steps of its halo's mass accretion history
(one per simulation snapshot, ~72 steps). At step `j` the halo gains mass
`dMh_j` and has properties `H_j` = (mass, redshift, radius, concentration, …).

**(1) The budget** — how much stellar mass this step deposits:

```
dM*_j = eps_surv(H_j) * dMh_j
```

**(2) The boundary** — the outer edge of the deposit:

```
R_trunc,j = C * R200c(t_j)          with C = 3 throughout
```

**(3) The sum** — the galaxy's curve of growth at observation epoch `t_k`:

```
M*(<R, t_k) = SUM over all j with t_j <= t_k  of  dM*_j * F_j(<R)
```

`F_j(<R)` is a **unit-mass cumulative profile**: a function rising from 0 to 1
that says what fraction of deposit `j`'s mass lies inside radius `R`. It is
truncated at `R_trunc,j` and renormalised so that `F_j(R_trunc,j) = 1` exactly.

That is the whole model. Three lines.

### 2.2 What `eps_surv` is, and what it is not

`eps_surv` is internal shorthand for the **effective surviving-central mass
fraction**: the fraction of the halo mass gained in a step that ends up as
stellar mass still bound to the central galaxy at the epoch we observe it.

**It is not a star-formation efficiency**, and it must not be quoted as one. It
silently absorbs at least five distinct physical processes:

1. conversion of accreted baryons into stars,
2. stellar evolutionary mass loss (stars die and return gas),
3. tidal stripping of the stars that do form,
4. whether the stars end up in the central galaxy or in a satellite,
5. delivery of stars formed elsewhere and later accreted (*ex-situ* mass).

Because it is a single number standing in for all five, **no individual
physical interpretation of its value is legitimate.** What *is* legitimate is
its dependence on halo mass and redshift, since that dependence is measured.

### 2.3 Why there is a truncation, and why it matters more than it looks

Without a boundary, "the total mass of a deposit" is a mass at infinity, which
for some profile shapes is a badly behaved quantity. Concretely: the fitted
Moffat profile falls off as `R^-0.76`, so **a deposit with a 50 kpc half-mass
radius needs to extend to 9.6 Mpc to enclose 99% of its mass.** That is not a
galaxy. It caused 18% of the model's stellar mass to sit beyond 500 kpc in an
earlier version.

Truncating at `C * R200c(t_j)` fixes this: the deposit's "total" is the mass
inside a *halo-defined* radius, which is finite, physical, and epoch-local.

`C = 3` is **a working default, not a physical law.** It was chosen because
`R99 <= 3 R200c` holds for the fitted profiles, and it was scanned rather than
assumed.

**The subtlety that had to be handled carefully.** Cutting a profile off moves
its half-mass radius *inward* — half the remaining mass is now inside a smaller
radius than half the original mass was. So if you truncate a profile but keep
labelling it by its *untruncated* half-mass radius, you are no longer measuring
the same quantity, and comparing two differently-shaped families becomes
meaningless. Measured mislabelling: **+10% to +35%**.

The model therefore parameterises by the **true post-truncation half-mass
radius**. Writing `F0` for the untruncated unit profile in units of its own
half-mass radius (so `F0(1) = 0.5`) and `u = R_trunc / R50_untruncated`:

```
F_trunc(x) = F0(x) / F0(u)                    the renormalised profile
x_h(u)     = F0^-1( 0.5 * F0(u) )             its half-mass radius
h(u)       = x_h(u) / u  =  R50_true / R_trunc
```

`h` decreases monotonically, so it can be inverted: given the size law's
requested `R50_true`, the code solves `h(u) = R50_true / R_trunc` for `u` and
recovers the family's own internal scale. A numerical self-check asserts
`F(R50) = 0.5` and `F(R_trunc) = 1` for every family to better than 5e-3 and
1e-12 respectively.

---

## 3. The options we tried, explained

The model has three independent axes. Each option is *nested* inside the richer
one, so "does this extra term earn its place?" is a well-posed question.

Below, `m = log10 Mh(t_j) - 13.5` (halo mass at the deposit's own formation
time, measured relative to a pivot of `10^13.5` solar masses) and
`x = ln(1 + z_j)` (a redshift variable that is 0 today and grows with lookback
time).

### 3.1 The efficiency axis — how much mass a step deposits

These control `log10 eps_surv`. **The `E`-numbers are internal labels, not
standard nomenclature.**

| label | what it means in words | formula | free params |
|---|---|---|---|
| **E1** | Efficiency is a power law in halo mass and in `(1+z)` — the simplest thing that could work. | `a0 + a_M*m + a_z*x` | 3 |
| **E2** | E1 **plus a mass×redshift cross-term**: how efficiency depends on halo mass is itself allowed to change with redshift. | `a0 + a_M*m + a_z*x + a_Mz*m*x` | 4 |
| **E3** | An *add-on* to E1 or E2: efficiency also depends on how concentrated the halo is, relative to typical for its mass and redshift. | `... + a_c*dlogc` | +1 |
| **E4** | The redshift dependence is a **free spline** — a flexible wiggly curve in `ln(1+z)` with 4 knots, instead of a straight line. Built to test whether the efficiency peaks at some redshift. | `a0 + a_M*m + spline(x)` | 6 |
| **E5** | Efficiency is a **double power law in halo mass**: rising as `Mh^beta` below a characteristic mass `M1`, falling as `Mh^-gamma` above it, with a separate redshift power law. This is the shape abundance-matching studies find for the stellar-to-halo-mass relation. | `log_eps0 + log10(2) - log10(r^-beta + r^gamma) + a_z*x`, where `r = Mh/M1` | 5 |

**`dlogc`, defined** (used by E3 and S3): the **concentration excess**,
`log10[c200c_measured / c200c_Diemer19(Mh, z)]`. It measures how much *more*
concentrated a halo is than is typical for its mass and redshift. The reference
`c200c_Diemer19` is a published deterministic formula, and subtracting it is
what makes the variable informative: the raw concentration is almost entirely
predictable from mass and redshift, so all of its independent signal lives in
this residual.

### 3.2 The profile axis — what shape a single deposit has

Six functional forms for `F0`, the unit-mass cumulative profile of one deposit,
all written so that `R50` is genuinely the half-mass radius:

| label | form | shape parameter |
|---|---|---|
| **sersic** | `gammainc(2n, x^(1/n))` — the standard galaxy profile | `n` (Sérsic index) |
| **moffat** | `1 - (1 + x^2)^(1-gam)` — a power-law-tailed profile | `gam` |
| **gompertz_log** | `exp(-ln2 * (R/R50)^-c)` — a sigmoid in log radius | `c` |
| **loglogistic** | `X/(1+X)` with `X = (R/R50)^k` | `k` |
| **richards** | a generalisation containing both of the above | `k`, `nu` |
| **expo**, **gauss** | fixed-shape special cases | none |

The factorial in Section 4 used the first three.

### 3.3 The size axis — how big a deposit is

These control `R50`, the deposit's half-mass radius. `f0` and `b` are free.

| label | what it means in words | formula |
|---|---|---|
| **S2** | The deposit's size is a fixed fraction of the halo's virial radius *at the time the deposit is laid down*, with a free power-law redshift tilt on top. | `R50 = f0 * R200c(t_j) * (1+z_j)^b` |
| **S2a** | The same, but scaled to the halo's **NFW scale radius** `R200c/c200c` instead of `R200c` — i.e. the deposit tracks the halo's inner structure rather than its outer edge. | `R50 = f0 * [R200c(t_j)/c_eff] * (1+z_j)^b` |
| **S3** | An *add-on* to either: the size normalisation is also allowed to depend on halo mass, concentration excess, and how early the halo formed. | `log10 f = ... + s_M*m + s_c*dlogc + s_f*(f_form - 0.5)` |

**`f_form`, defined**: `t_half(t_j)/t_j`. In words — *of the mass this halo has
right now, how long ago did it acquire half of it, as a fraction of its current
age?* Near 0 means the halo assembled very early. Unlike other formation-time
measures it never divides by the halo's *final* mass, so it can be computed at
any epoch without knowing the future. That property is what makes it legal as a
model input.

**A naming trap worth flagging.** The code reads
`scale = R200c if size == "S2" else R200c/c_eff`. So **`S2a` takes the `else`
branch and does use concentration** — a fact that was missed once and cost a
wrong impact assessment (Section 7).

---

## 4. How models are scored

### 4.1 The split: amplitude and shape

Every predicted curve of growth is split losslessly into two independent parts:

```
A     = log10 M*(<100 kpc)                the AMPLITUDE — how much mass
F(<R) = M*(<R) / M*(<100 kpc)             the SHAPE — how it is distributed
```

They are scored separately because they fail independently: a model can have
the right total and the wrong distribution, or vice versa, and a single number
cannot tell those apart.

### 4.2 The two scores

```
score_A = rms over galaxies and epochs of [ (A_model - A_truth) / sigma_A(z) ]
score_F = Objective(F_model, F_truth, R) / L_F_ref
```

Both are **normalised by a measured benchmark, so that 1.0 means "as good as
what we already know how to do"**:

- **`sigma_A(z)`** is the scatter of the best *purely statistical* halo-only
  regression, refitted separately at each epoch — the best a regression can do
  using the same halo information. Its value is
  **0.1044 / 0.1120 / 0.1204 / 0.1279 / 0.1611 dex** at *z* = 0.4 / 0.7 / 1.0 /
  1.5 / 2.0. So `score_A = 1.0` means the physical model matches a regression;
  below 1 it beats one; above 1 it loses to one.

- **`L_F_ref = 0.158196`** is the previous-generation model's shape loss.

The `Objective` used for shape is: fractional residuals `(model - truth)/truth`
on the cumulative profile, root-mean-square over the 24 radii with equal
weight, arithmetic mean over epochs, arithmetic mean over galaxies.

The fitted loss is `L = score_A^2 + score_F^2`, plus a penalty for any galaxy
the model cannot represent at all (priced, never dropped — dropping failures
makes a failing model's loss silently improve).

### 4.3 The identifiability report — mandatory, and it has teeth

**A parameter can be useless with a sharp optimum, or genuinely useful and
completely undetermined, and the loss distinguishes neither.** Every fit
therefore ships:

- a scaled residual-Jacobian SVD, reporting **`ident` = smallest singular value
  / largest**. Small `ident` means some direction in parameter space barely
  changes the prediction — i.e. those parameters are not determined by the data.
- a profile scan per parameter, re-optimising the others;
- bootstrap stability over galaxies;
- profiles of *derived* quantities, which are often identified where their
  constituents are not.

**A parameter that fails may stay in the model; its VALUE is never quoted.**

### 4.4 The judge — and why we never rank by loss

Models are ranked by **mean rank across seven criteria**, not by the loss:

`score_A` at z=0.4 and z=2.0, `score_F` at z=0.4 and z=2.0, `ident`, an
over-reach diagnostic, and the **halo-mass tilt**.

**Halo-mass tilt, defined.** For a fixed radius and epoch, take each galaxy's
residual `y = log10[M*_model(<R)] - log10[M*_truth(<R)]`, and fit a straight
line of `y` against `log10(halo mass)` across the population. The tilt is that
slope, in **dex of stellar-mass error per dex of halo mass**. Zero means the
model is equally right for light and heavy haloes; non-zero means it has a
systematic mass-dependent bias that the population-averaged loss hides.

**This is not a stylistic preference.** Ranking the 45-cell factorial by loss
picks cells the judge places 2nd, 6th and 10th — and those loss-optimal cells
have `ident` around 8e-04 against the winner's **7.8e-03**, an order of
magnitude worse conditioned. **The loss rewards exactly the degenerate models.**

---

## 5. The main result

### 5.1 The adopted model

```
log10 eps_surv = a0 + a_M*m + a_z*x + a_Mz*m*x        (E2)
R50            = f0 * R200c(t_j) * (1+z_j)^b          (S2)
F0             = exp(-ln2 * (R/R50)^-c)               (gompertz_log)
truncated at 3*R200c(t_j) and renormalised
```

**Seven global parameters. No per-object freedom. No pinning to the truth.**
Fitted values on the full sample: `a0` = −2.549, `a_M` = −0.452, `a_z` = 0.488,
`a_Mz` = 0.233, `log f0` = −0.952, `b` = −0.735, `c` = 0.919.

It is the judge's rank-1 model out of 45, and it stayed rank 1 after two
independent corrections to the scoring (Section 7).

### 5.2 What it gets right

- **The radial profile.** `score_F` = 0.919 / 0.916 / 0.875 / 0.797 / 0.717 —
  better than the previous-generation model at every epoch, and improving with
  redshift. (These are from the corrected fit; the pre-correction fit gave
  0.919 / 0.916 / 0.875 / 0.797 / 0.716, i.e. the shape half of the problem was
  entirely unaffected by the benchmark error — see Section 7.2.)
- **Stellar mass growth.** The model's own `M*(<100 kpc)` growth from *z* = 2
  to *z* = 0.4 is right to **+0.030 dex** (corrected fit; +0.032 before). The
  previous generation's kernel needed +0.398 dex — a 13× improvement in a
  quantity that was previously supplied by a normalisation rather than
  predicted at all.
- **It generalises.** Held out on 1197 galaxies never used in the fit, the
  robust scatter of the amplitude residual is identical to in-sample at every
  epoch (0.1156/0.1202/0.1218/0.1305/0.1507 against
  0.1136/0.1178/0.1226/0.1310/0.1530). Seven global parameters do not overfit
  1199 galaxies, as one would hope.

### 5.3 What it gets wrong

**Against a purely statistical halo-only regression, it loses.**

| `score_A` (1.0 = ties the regression) | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| on the full 2397-galaxy sample | 1.131 | 1.100 | 1.059 | 1.058 | 1.050 |
| on a **complete** sample of massive haloes | 1.116 | **1.356** | **1.339** | **1.360** | **1.371** |

On the sample as selected, the model is 5–13% worse than a regression. On a
sample where the selection has been controlled (Section 6), it is **~36% worse
at z ≥ 0.7**. The regression barely notices the restriction — its scatter falls
only 3–15% — while the model degrades sharply.

**The plain reading: the model is weakest exactly where the data are best**, on
massive, well-resolved, completely-sampled haloes. That is the deployment case,
and it is currently the strongest argument against the model as it stands.

### 5.4 The standing structural defect: the model under-fills the centre

Per galaxy, `log10[M*_model(<R)] - log10[M*_truth(<R)]`, median over galaxies.
Negative means the model puts **less** stellar mass inside `R` than the
simulation does. Measured on the corrected fit:

**All 2397 galaxies:**

| R [kpc] | 2.8 | 4.9 | 10.2 | 27.5 | 79.8 | 148.2 |
|---|---|---|---|---|---|---|
| z = 0.4 | **−0.034** | −0.040 | −0.004 | +0.008 | −0.008 | −0.016 |
| z = 0.7 | −0.044 | −0.039 | +0.003 | +0.020 | +0.009 | +0.003 |
| z = 1.0 | −0.049 | −0.033 | +0.007 | +0.021 | +0.012 | +0.007 |
| z = 1.5 | −0.073 | −0.048 | −0.012 | +0.002 | +0.001 | −0.002 |
| z = 2.0 | −0.084 | −0.065 | −0.038 | −0.026 | −0.024 | −0.024 |

**Fair sample** (the completeness-controlled galaxies of Section 6):

| R [kpc] | 2.8 | 4.9 | 10.2 | 27.5 | 79.8 | 148.2 |
|---|---|---|---|---|---|---|
| z = 0.4 | −0.034 | −0.040 | −0.004 | +0.008 | −0.008 | −0.016 |
| z = 1.0 | −0.060 | −0.029 | +0.016 | +0.031 | +0.011 | +0.003 |
| z = 2.0 | **−0.133** | −0.081 | −0.039 | −0.030 | −0.040 | −0.043 |

Three things this says, none of which is the single number this defect used to
be quoted as:

1. **The deficit is inner and it is real**, but at low redshift it is modest:
   −0.034 dex at 2.8 kpc at *z* = 0.4, i.e. the model is ~7.5% too light there.
   (An earlier generation of this model was −0.39 to −0.47 dex; the previous
   internal summary of the present model quoted −0.10 dex, which the corrected
   fit does not reproduce — the measured value is −0.034.)

2. **It grows strongly with redshift**, to −0.084 dex on the full sample and
   **−0.133 dex (26% too light) on the fair sample at *z* = 2**. The model most
   badly under-fills the centres of high-redshift galaxies — which is exactly
   the population known independently to be unusually compact.

3. **At low redshift it is genuinely misplaced mass; at high redshift it is
   also missing mass.** At *z* = 0.4 the outer profile is within 0.016 dex, so
   the total is roughly right and the error is one of distribution. At *z* = 2
   on the fair sample the whole curve is low (−0.043 dex at 148 kpc) with the
   centre lower still, so there are two errors superposed: not enough mass, and
   too little of it in the middle.

No profile family fixes the inner part — `gompertz_log`, `moffat` and `sersic`
give the same curve — and it is not a resolution artifact in the direction that
would explain it away: gravitational softening makes the simulation's cores
*less* concentrated than reality, so the true deficit against nature is at
least this large.

---

## 6. The progenitor-selection confound, and why it mattered

### 6.1 The problem, stated plainly

Every galaxy in this project was chosen at *z* = 0.4. The high-redshift samples
are therefore **not high-redshift samples** — they are the main progenitors of
a low-redshift selection. At *z* = 2, a halo of a given mass is in our sample
only if it will grow fast enough afterwards to clear the *z* = 0.4 threshold.
At low halo mass we keep a small, growth-selected minority; at high halo mass
we keep essentially everything.

### 6.2 Measuring it

**Completeness, defined**: `N(our sample) / N(all TNG300 central haloes)` in a
bin of `log10 M200c` at one epoch. 1.0 means every halo of that mass is in our
sample. Computed against the *full* simulation catalogue — 346,000 to 460,000
central haloes per epoch.

The *z* = 0.4 ceiling is **0.781**, set by data-quality flags that are applied
to the *z* = 0.4 galaxy and therefore cost the same at every epoch. The
**fair-sample cut** is the halo mass at which completeness first reaches 60% of
that ceiling and stays there:

| | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| cut, `log10 M200c` | >13.00 | >13.00 | >13.00 | >12.90 | >12.80 |
| galaxies kept | 2397 | 1781 | 1436 | 1145 | 840 |

At *z* = 0.4 the cut keeps everything, which is the right sanity check: there
is no progenitor bias at the selection epoch.

### 6.3 The result: one of the two "defects" was the survey

The model's halo-mass tilt at *z* = 2, with parameters frozen:

| tilt at z=2 [dex/dex] | 5 kpc | 10 kpc | 75 kpc | 148 kpc |
|---|---|---|---|---|
| full sample | −0.135 | −0.075 | −0.083 | −0.094 |
| fair sample | −0.011 | **+0.066** | **+0.099** | **+0.089** |

It does not shrink — **it flips sign**, and lands on the same positive tilt the
fair sample shows at *z* = 0.7–1.5. Removing the incomplete regime makes *z* = 2
consistent with the other epochs instead of an outlier.

Meanwhile the *amplitude* deficit at *z* = 2 gets **worse**, from −0.0255 to
−0.0421 dex. So of the two things we thought were wrong with the model at high
redshift, one was the survey and the other was under-reported.

### 6.4 Why we believe it is the selection and not a range effect

A slope measured over a narrower mass range can differ from one measured over a
wider range for reasons unrelated to selection. Two further tests close that
gap.

**The mechanism test.** A selection can only bias a residual if the residual
depends on the variable the selection truncated. That variable is **future
growth**, `G_k = log10 Mh(z=0.4) - log10 Mh(z_k)` — how much the halo grows
*after* the epoch being scored, which is what the *z* = 0.4 selection cuts on
and what the model, reading only epoch-local quantities, cannot see. Measured:
partial correlation +0.19/+0.22/+0.23/+0.21 and slope
**dy/dG = +0.18/+0.16/+0.15/+0.15 dex per dex** at *z* = 0.7/1.0/1.5/2.0.
Significantly non-null at every epoch, in the direction required.

**The magnitude test.** Combining the measured completeness curve with the
measured `dy/dG` under a truncated-Gaussian model predicts the size of the tilt
shift, using no fitted model quantity:

| tilt shift at 148 kpc | observed | predicted (lower bound) | predicted (truncation-corrected) |
|---|---|---|---|
| z=0.7 | −0.0148 | −0.0170 | −0.0443 |
| z=1.0 | −0.0331 | −0.0296 | −0.0844 |
| z=1.5 | −0.0667 | −0.0429 | −0.1364 |
| z=2.0 | −0.1830 | −0.0542 | −0.1918 |

**The observation lies inside the bracket at every epoch.** The uncorrected
estimate matches where truncation is mildest, the corrected one matches *z* = 2
where truncation is most severe — which is how each should behave if the
mechanism is the right one. **The size of the tilt shift needs no physics.**

### 6.5 What is a limitation, not a bug

The framework describes the descendants of massive low-redshift galaxies. The
haloes it discards are ones whose growth stalls, whose centrals are typically
not massive ellipticals. Two consequences are accepted scope limits:

1. **The model cannot yet be applied to a halo sample selected at high redshift
   and evolved forward.** It has only ever been calibrated on backward-traced
   progenitors.
2. High-redshift performance must be *quoted* on the fair sample, not the full
   one, because the full one flatters (Section 5.3).

---

## 7. Two defects in the scoring, and what they changed

Both were found during Stage 3.4 and both changed conclusions.

### 7.1 A unit bug in the concentration excess

The halo-structure catalogue stores `M200c` as `log10(M/Msun)`. Two modules
converted it as if it were a *linear* mass in units of `1e10 Msun/h`, so the
reference concentration was evaluated at a mass **2.0006 dex too low**.

Effect on `dlogc`: a spurious 0.15 dex drift with redshift, and a halo-mass
slope that changes sign at *z* = 0.4 (−0.053 as coded against +0.047 correct).

**Scope**: 33 of 45 factorial cells read the variable — via `E3`, via `S2a`,
and via `S3`. After re-running all 45, `gompertz_log-E2-S2` was still rank 1,
but **the four largest upward rank moves in the factorial were all E1+E3
cells** (29→13, 34→18, 37→23, 36→22). The earlier verdict "concentration never
reaches the top 15" was partly an artifact of feeding it a corrupted variable.
Correcting it does not make concentration competitive with the cross-term, but
it had not been given a fair hearing.

### 7.2 One corrupt galaxy inflating the benchmark

Row 181's measured `M*(<100 kpc)` collapses from `10^11.712` at *z* = 0.4 to a
flat `~10^7.96` at all four higher redshifts — 2.5 dex below the sample's 0.1st
percentile. A main progenitor cannot be 3.76 dex lighter than its own
descendant; this is a broken cross-match, not a galaxy.

It sits in the 2397-galaxy set that defines `sigma_A`, and in **none** of the
model-fitting subsamples (300, 240, 1199 — all take every second galaxy, and
181 is odd). **So every `score_A` divided a model error measured on clean
galaxies by a benchmark charged for a corrupt one.**

| best halo-only regression | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| as published | 0.1047 | 0.1359 | 0.1417 | 0.1456 | 0.1744 |
| with row 181 rejected | 0.1044 | 0.1120 | 0.1204 | 0.1279 | 0.1611 |

| `gompertz_log-E2-S2` `score_A` | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| as published | 1.128 | 0.907 | 0.899 | 0.929 | 0.970 |
| corrected | 1.131 | 1.100 | 1.059 | 1.058 | 1.050 |

**This overturned the previous headline.** "The model beats the best halo-only
regression at four of five epochs" became "it is 5–13% worse at every epoch."

**The damage is bounded to the amplitude side.** Row 181 moves the population
*shape* loss by **+0.043%**, because the amplitude residual is
`log10(model/truth)` — unbounded, and its was 3.76 dex — while the shape
residual is computed on a profile normalised to 1, so it lives near unity and
no single galaxy can move a 2397-galaxy mean. Every `score_F` stands.

---

## 8. What we have learned

### 8.1 About the physics

1. **A three-line deposition law reproduces the radial profile well.** Seven
   global parameters, no per-object freedom, `score_F` from 0.92 down to 0.72
   with increasing redshift. The profile half of the problem is largely solved.

2. **The mass×redshift cross-term is the single most valuable term.** Without
   it (`E1`) the model has a halo-mass tilt of +0.024 to +0.041 dex/dex; with it
   (`E2`) the tilt collapses to +0.003 to +0.012. Physically: *how* efficiency
   depends on halo mass must itself change with redshift. This replaced what an
   earlier generation of the model achieved with an explicit transport term.

3. **Concentration is not competitive with that cross-term**, even after the
   unit bug was fixed. It improves, but reaches only rank 13 of 45.

4. **No peak in the efficiency's redshift dependence is identified.** `E4`, the
   free spline built precisely to find one, is the worst family in the
   factorial with `ident` ~1.5e-07 — its parameters are essentially undetermined
   — for a loss gain of ~0.007.

5. **The profile family barely matters.** Spread across families is 0.05
   against 0.08 across efficiency forms. What a deposit is shaped like matters
   less than how much mass it carries and where it is placed.

6. **The deposition-only assumption is not yet the binding constraint.** A
   floor computed from the simulation's own non-monotonic behaviour is 0.0456;
   the models sit at 0.113–0.145 — 2.5× to 3.2× headroom. **The functional
   forms are the limitation, not the deposition-only premise.**

### 8.2 About what the model still cannot do

7. **The inner deficit is unsolved and unexplained.** 21% too little stellar
   mass inside 3 kpc, ~5% too much at 20–70 kpc, total nearly right. Three
   generations of experiment have failed to remove it by varying the profile
   family, the objective, or the transport term. What has *not* been tried is a
   **second, compact deposit channel** — the profile family is the shape of one
   component, and a second component is a different move.

8. **The model degrades on complete, massive samples** — ~36% worse than a
   regression at *z* ≥ 0.7, against 5–10% on the selected sample. This is an
   amplitude failure at high halo mass, and it is distinct from the inner
   deficit. It is currently unexplained and deserves its own investigation.

### 8.3 About two terms that turned out to be fitting the survey

9. **The size-law conditioning (`S3`) absorbs selection structure, not
   physics.** Two independently-constructed fair samples both drive all three of
   its slopes to near zero from large full-sample values:

   | parameter | full sample | fair sample A | fair sample B |
   |---|---|---|---|
   | `s_M` (mass) | 0.0992 | 0.0032 | 0.0202 |
   | `s_c` (concentration) | −0.3380 | −0.0165 | −0.0242 |
   | `s_f` (formation time) | −0.3342 | +0.0043 | +0.0051 |

10. **And so, apparently, does `E5`.** Its whole advantage over `E2` was that it
    closes the *z* = 2 halo-mass tilt — and that tilt is the survey. **`E5`'s
    four extra parameters buy a fit to a selection artifact.** The case for the
    seven-parameter `E2-S2` is *stronger* after the selection control than
    before it, which was not the expected outcome.

### 8.4 About method — the lessons that generalise

11. **A benchmark and the thing it benchmarks must be scored on the same
    objects.** One corrupt galaxy in the denominator and not the numerator
    overturned an experiment's headline. When a metric is a ratio computed by
    two different scripts, assert the object sets match — it is one line.

12. **An unexpectedly good result deserves the same audit as a bad one.** A
    benchmark degraded by outliers flatters whatever it is compared against.

13. **Rank by a multi-criterion judge, never by the loss.** Demonstrated twice:
    the loss-optimal cells are precisely the degenerate ones.

14. **Check the completeness of your sample before naming a residual.** A
    delivery-delay kernel fitted to the *z* = 2 tilt would have absorbed a
    survey effect and acquired a physical name it had not earned.

15. **Classify which code a change invalidates from what the code *reads*, not
    from a list written by hand.** A hand-written list missed `S2a`; a control
    cell asserted not to move caught it within three cells.

---

## 9. Where the model stands, honestly

**What is solid.** A seven-parameter, no-pin, no-per-object-freedom deposition
law that predicts the radial profile well, gets stellar-mass growth right to
0.03 dex, generalises without a gap, and whose one important term (`E2`)
survives every control we have applied.

**What is not.** It does not beat a purely statistical halo-only regression at
any epoch, and on a completeness-controlled sample of massive haloes it is
about a third worse. Two of the extensions that appeared to help — `E5` and
`S3` — appear to be fitting the sample selection rather than the galaxies.

**What comes next**, in order:

1. Drop `S3`; treat `E5` as suspect for the same reason.
2. Build the **compact second channel** against the inner deficit, tied to
   `f_form` so it can fail rather than fit anything. Score it on the fair
   sample, since the full one is now known to flatter.
3. Investigate the high-mass amplitude failure — why the model degrades where
   the data are best. That, not the inner deficit, is what stands between this
   model and deployment.
4. A delivery-delay kernel remains a last resort. If built, it must carry an
   *empirical* halo-mass dependence: `tau = eta * t_dyn` cannot work, because
   for a 200×critical-density halo `V200c = 10 H(z) R200c` exactly, so
   `t_dyn = R200c/V200c = 1/(10 H(z))` is a function of redshift alone — the
   halo mass cancels identically, and a mass-independent clock cannot produce a
   mass-dependent effect.

---

## Appendix: where things live

| item | file |
|---|---|
| halo record (contains no stellar data by construction) | `experiments/exp54_unpinned_amplitude/halo.py` |
| the model, families, truncation | `model.py`, `experiments/exp53_deposition_only/families.py` |
| objective, scores, identifiability | `fit.py`, `hongshao/objective.py` |
| halo-only regression benchmark | `scoreboard.py` |
| the 45-cell factorial and its re-run | `stage32.py`, `stage32_rejudge.py` |
| five-epoch fits | `stage33.py`, `stage33_refit.py`, `stage33_perepoch.py` |
| the selection control | `selection.py` |
| QA figures | `qa_figures.py` |
