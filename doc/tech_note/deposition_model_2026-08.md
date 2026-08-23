# The deposition-only stellar mass model: setup, options, scoring, results

*Status as of 2026-08-23, after exp54 Stages 0–3.4 including the
representational ceiling (Section 8.5). Supersedes the summaries in
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

**The data.** 2397 galaxies from the TNG300 cosmological simulation. **The
selection is on HALO mass, not stellar mass**: the parent sample is massive
central haloes selected by peak halo mass at *z* = 0.4 (`logm0_halo` in
`hongshao.tng_data`), and 2397 is the usable cross-matched subset. The
resulting galaxies happen to span `log10 M*` from 10.66 to 12.36, which
describes the outcome and not the criterion. This matters for Section 6:
everything there is about a descendant-halo-selected progenitor sample. For
each galaxy we have:

- its **mass accretion history** (MAH): the halo mass `Mh(t)` at 73 snapshots,
  from the smooth DiffMAH fit;
- its **halo structure** at 16 snapshots: `R200c`, the NFW concentration
  `c200c`;
- its **curve of growth** (CoG) at five epochs — see below.

**Curve of growth (CoG), defined.** `M*(<R)`, the **projected** stellar mass
inside an elliptical isophotal aperture of semi-major axis `R`, measured on a
fixed 24-point grid from 2 kpc to 148.22 kpc. It is built by cumulatively
integrating the X-Y isophote surface density over elliptical annuli
(`hongshao.tng_data`), so it is a two-dimensional aperture mass and **not** a
spherical enclosed mass: the two have different kernels and are not
interchangeable. It is cumulative, so it only ever rises with `R`. This is the
thing the model must reproduce.

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

$$\Delta M_{*,j} \;=\; \varepsilon_{\rm surv}(H_j)\,\Delta M_{h,j}$$

**(2) The boundary** — the outer edge of the deposit:

$$R_{{\rm trunc},j} \;=\; C\,R_{200c}(t_j), \qquad C = 3 \;\text{throughout}$$

**(3) The sum** — the galaxy's curve of growth at observation epoch `t_k`:

$$M_*(<R,\,t_k) \;=\; \sum_{j\,:\,t_j \le t_k} \Delta M_{*,j}\; F_j(<R)$$

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

`C = 3` is **an empirical convention chosen by us, not a fitted or preferred
value.** Each deposit is truncated at three times the halo's epoch-local
`R200c` in order to retain an extended central-galaxy envelope while assigning
a finite, halo-scaled mass budget. It was not fitted to the TNG profiles and is
not claimed to be statistically preferred; no scan over `C` is persisted in
this experiment, and `R99 <= 3 R200c` follows automatically once the profile is
truncated, so it is not evidence for the value. **The model's total stellar
mass and its fitted efficiency are both defined partly by this choice and must
always be quoted together with it.** A sensitivity comparison over
`C = 1, 2, 3, 5` is on `doc/todo.md`.

**The subtlety that had to be handled carefully.** Cutting a profile off moves
its half-mass radius *inward* — half the remaining mass is now inside a smaller
radius than half the original mass was. So if you truncate a profile but keep
labelling it by its *untruncated* half-mass radius, you are no longer measuring
the same quantity, and comparing two differently-shaped families becomes
meaningless. Measured mislabelling: **+10% to +35%**.

The model therefore parameterises by the **true post-truncation half-mass
radius**. Writing `F0` for the untruncated unit profile in units of its own
half-mass radius (so `F0(1) = 0.5`) and `u = R_trunc / R50_untruncated`:

$$F_{\rm trunc}(x) = \frac{F_0(x)}{F_0(u)}, \qquad
x_h(u) = F_0^{-1}\!\left(\tfrac{1}{2}F_0(u)\right), \qquad
h(u) = \frac{x_h(u)}{u} = \frac{R_{50}^{\rm true}}{R_{\rm trunc}}$$

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

### 4.0 Which epochs are fitted

**All five.** The five-epoch fits (`stage33*.py`) minimise a single loss summed
over *z* = 0.4, 0.7, 1.0, 1.5 and 2.0 simultaneously — `epochs=(0,1,2,3,4)`.
There is no extrapolation in redshift: *z* = 2 is fitted like every other epoch.

The one place a reduced epoch set is used is the 45-cell **model-selection
factorial** (`stage32*.py`), which fits only `epochs=(0,4)` — *z* = 0.4 and
*z* = 2.0, the two **endpoints** — because it runs 45 models and needs to be
affordable. There the interpolated epochs are *z* = 0.7, 1.0 and 1.5, not
*z* = 2. So at no stage is *z* = 2 an extrapolation; in the factorial it is one
of only two epochs that constrain the fit at all.

**A test worth running that has not been run.** Fit only *z* ≤ 1.5 and hold
*z* = 2.0 out entirely, then score it. That converts *z* = 2 from a constraint
into a genuine **extrapolation in redshift**, which is a much stronger claim
about the model than fitting it: a physical law calibrated over 0.4 ≤ *z* ≤ 1.5
that also predicts *z* = 2 has demonstrated something a fitted curve has not.
It is cheap — one extra fit per model — and it is the natural way to advertise
the model's capability rather than its flexibility. Recorded in
`doc/todo.md`.

### 4.1 The split: amplitude and shape

Every predicted curve of growth is split losslessly into two independent parts:

$$A \;=\; \log_{10} M_*(<100\,{\rm kpc}) \quad\text{(amplitude: how much mass)}$$
$$F(<R) \;=\; \frac{M_*(<R)}{M_*(<100\,{\rm kpc})} \quad\text{(shape: how it is distributed)}$$

They are scored separately because they fail independently: a model can have
the right total and the wrong distribution, or vice versa, and a single number
cannot tell those apart.

### 4.2 The two scores

$$\mathrm{score}_A = \sqrt{\Big\langle \Big(\tfrac{A_{\rm model}-A_{\rm truth}}{\sigma_A(z)}\Big)^{\!2} \Big\rangle_{\rm galaxies,\,epochs}}
\qquad
\mathrm{score}_F = \frac{\mathcal{L}\big(F_{\rm model},F_{\rm truth},R\big)}{L_F^{\rm ref}}$$

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
- a **one-parameter conditional loss slice** per parameter. NOTE: the
  implementation in `fit.identifiability` displaces one parameter and holds the
  other six FIXED. That is conditional curvature, not a profiled fit: a genuine
  profile would re-optimise the remaining parameters at every grid point, and
  in a correlated model the conditional slice can rise steeply along a
  direction the profiled curve leaves nearly flat. Re-optimised profiles for
  the adopted model are on `doc/todo.md` and have not been run;
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

$$\log_{10}\varepsilon_{\rm surv} = a_0 + a_M\,m + a_z\,x + a_{Mz}\,m\,x
\qquad\text{(E2)}$$
$$R_{50} = f_0\,R_{200c}(t_j)\,(1+z_j)^{\,b}
\qquad\text{(S2)}$$
$$F_0(R) = \exp\!\big[-\ln 2\,(R/R_{50})^{-c}\big]
\qquad\text{(gompertz\_log)}$$
$$\text{truncated at } 3R_{200c}(t_j) \text{ and renormalised}$$

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
- **Stellar mass growth, in the median only.** The model reproduces the
  **median** `M*(<100 kpc)` growth from *z* = 2 to *z* = 0.4 to about **0.03
  dex**, against +0.398 dex for the previous generation's kernel — a quantity
  that used to be supplied by a normalisation rather than predicted. **But that
  is a population bias, not an individual-galaxy accuracy.** Measured on the
  cached 2397-galaxy predictions with the broken cross-match removed: median
  error **+0.034 dex**, per-galaxy standard deviation **0.208 dex**, NMAD
  **0.179 dex**, correlation with the true growth **0.71**, and a predicted
  growth distribution that is **about half as wide as TNG's** (0.145 against
  0.284 dex). So the model gets the average growth right, its individual
  growth errors are large, and it does not reproduce the diversity of growth
  histories at all. That last point is where a stochastic residual layer will
  eventually be required; it is not a defect of the mean law.
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

**Where in the profile that failure sits.** Median `100*(model-truth)/truth`
by tercile of `log10 Mh(z=0.4)`, for the adopted model:

| tercile | epoch | 2.8 kpc | 10.2 kpc | 27.5 kpc | 148.2 kpc |
|---|---|---|---|---|---|
| 13.00–13.18 | z = 0.4 | −4.0 | −2.0 | −1.6 | −4.1 |
| 13.18–13.46 | z = 0.4 | −6.5 | −2.2 | +0.3 | −3.9 |
| **13.46–15.14** | z = 0.4 | **−12.6** | +1.3 | **+6.3** | −3.5 |
| 13.00–13.18 | z = 2.0 | −12.6 | −9.3 | −7.0 | −5.1 |
| 13.18–13.46 | z = 2.0 | −16.6 | −10.0 | −7.9 | −6.6 |
| **13.46–15.14** | z = 2.0 | **−24.8** | −6.2 | −2.8 | −4.9 |

The failure in the most massive tercile is **not** uniform across radius. At
148 kpc the top tercile is no worse than the bottom (−3.5 against −4.1 at
*z* = 0.4). What grows is the **misplacement**: −12.6% in the centre together
with **+6.3% at 27.5 kpc**, against −4.0% and −1.6% in the lightest tercile.
All three short-listed models show it, so it is a property of the framework,
not of one profile family.

**Binned by halo mass at the epoch itself, inside the mh-complete sample, the
statement sharpens into the single worst thing the model does:**

| `log10 M200c(z=2)` | n | 2.8 | 4.9 | 10.2 | 27.5 | 79.8 | 148.2 kpc |
|---|---|---|---|---|---|---|---|
| 12.80–12.91 | 280 | −21.6 | −16.6 | −11.9 | −11.5 | −11.6 | −12.4 |
| 12.91–13.09 | 280 | −27.5 | −19.0 | −13.6 | −12.3 | −12.3 | −13.0 |
| **13.09–14.12** | 280 | **−29.2** | −16.1 | **−1.7** | **+1.5** | −2.5 | −4.9 |

**In the most massive *z* = 2 haloes the model is 29% too light inside 3 kpc
and essentially correct from 10 kpc outward.** That is a pure central failure,
not a global one — and the two lighter bins fail differently, being uniformly
~12% too light at every radius. The central deficit in the top mass bin grows
monotonically with redshift: **−12.6% at *z* = 0.4, −20.4% at *z* = 1.0,
−29.2% at *z* = 2.0.**

(The galaxy-to-galaxy scatter is wide — 16th–84th percentile width ~60% at
2.8 kpc — but with n = 280 per bin the median is determined to about ±2%.)

**Plainly: the model cannot build the dense centres of the most massive
galaxies at *z* = 2.** Those are exactly the compact, early-forming systems the
project set out to study, so this is not a peripheral failure — it is the
model failing hardest on its target population. It is also the strongest
motivation for the compact second channel of Section 9. **But see Section
8.5.2**: that population figure is dominated by the ~53% of galaxies whose
central mass later DECLINES, which no deposition-only model can follow. Among
those whose centre does not decline the model is only 11.5% low. The compact
channel is accordingly deprioritised rather than retired.

### 5.5 The same failure seen as a density profile: the outskirts are too shallow

The curve of growth is cumulative, so it is dominated by the inner regions and
hides what the outer profile is doing. Differencing it into a **surface density**
`Sigma(R) = dM/dA` and plotting `log10 Sigma` against `R^(1/4)` — the
coordinate in which a de Vaucouleurs (n = 4) profile is a straight line —
exposes a second, independent statement of the same defect.

Outer logarithmic slope `d log10 Sigma / d log10 R`, fitted over 50–148 kpc on
the median profile (more negative = steeper):

| epoch | truth | model | model − truth |
|---|---|---|---|
| z = 0.4 | −2.78 | −2.74 | +0.04 |
| z = 0.7 | −2.85 | −2.78 | +0.07 |
| z = 1.0 | −2.91 | −2.81 | +0.10 |
| z = 1.5 | −3.01 | −2.86 | +0.15 |
| **z = 2.0** | **−3.38** | **−2.90** | **+0.48** |

**CORRECTION, from Section 8.5.** This table is measured on the FULL
progenitor-selected sample. On the per-epoch mh-complete sample the truth
steepens only to **−2.95**, and on one fixed galaxy set it runs −2.56 to −2.95,
so the model's *z* = 2 error is **+0.21** dex/dex (`gompertz_log`) or **+0.06**
(`moffat`), not +0.48 — about half of the apparent failure is the changing
sample, exactly as in Section 6.4b. Section 8.5 also shows the remainder is
invisible to the production objective rather than unrepresentable.

**The simulation's galaxies get steeper with redshift; the model's barely
change.** By *z* = 2 the model's outskirt is 0.48 dex per dex too shallow. In
the cumulative view this is nearly invisible — the *z* = 2 curve of growth is
only −0.024 dex low at 148 kpc — because a shallow outer slope adds mass in the
outermost annuli while the total stays low.

**The two high-redshift failures are one statement.** Too little mass in the
centre *and* too shallow an outer slope both say: **the model's high-redshift
galaxies are too diffuse.** It is not that the model makes too little stellar
mass at *z* = 2; it makes roughly the right amount and spreads it out too far.
That is a much sharper target than "an inner deficit plus an amplitude
deficit", and it points squarely at the deposit size law
`R50 = f0 R200c(t_j) (1+z)^b`: the fitted `b = −0.735` shrinks deposits toward
high redshift, but evidently nowhere near enough.

(The truth's own slope is worth noting independently: these galaxies really are
close to de Vaucouleurs at *z* = 0.4 — the *z* = 0.4 truth tracks a straight
line in `R^(1/4)` across the whole 2–148 kpc range — and they steepen
substantially by *z* = 2.)

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

**Mh-complete sample** (the completeness-controlled galaxies of Section 6):

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
   **−0.133 dex (26% too light) on the mh-complete sample at *z* = 2**. The model most
   badly under-fills the centres of high-redshift galaxies — which is exactly
   the population known independently to be unusually compact.

3. **At low redshift it is genuinely misplaced mass; at high redshift it is
   also missing mass.** At *z* = 0.4 the outer profile is within 0.016 dex, so
   the total is roughly right and the error is one of distribution. At *z* = 2
   on the mh-complete sample the whole curve is low (−0.043 dex at 148 kpc) with the
   centre lower still, so there are two errors superposed: not enough mass, and
   too little of it in the middle.

No profile family fixes the inner part — `gompertz_log`, `moffat` and `sersic`
give the same curve, and Section 8.5 shows that no CHOICE OF WEIGHTS closes
more than about a third of it. Gravitational softening is unlikely to explain
it away, since it makes the simulation's cores *less* concentrated — but that
is a statement about the direction of a numerical bias inside TNG and **not** a
one-sided bound on the deficit against the real Universe, which this experiment
cannot measure.

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
**mh-complete cut** is the halo mass at which completeness first reaches 60% of
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
| mh-complete sample | −0.011 | **+0.066** | **+0.099** | **+0.089** |

It does not shrink — **it flips sign**, and lands on the same positive tilt the
mh-complete sample shows at *z* = 0.7–1.5. Removing the incomplete regime makes *z* = 2
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

### 6.4b A trap in reading fair-sample figures

**The mh-complete sample is a different set of galaxies at each epoch.** The cut is
applied to the halo mass *at that epoch*, so the surviving galaxies are 2397 /
1781 / 1436 / 1145 / 840, and they are progressively more massive:

| epoch | n | median `log10 Mh(z=0.4)` of that set |
|---|---|---|
| z = 0.4 | 2397 | 13.293 |
| z = 0.7 | 1781 | 13.415 |
| z = 1.0 | 1436 | 13.489 |
| z = 1.5 | 1145 | 13.541 |
| z = 2.0 | 840 | 13.623 |

So a figure that plots each epoch's mh-complete sample side by side is **not an
evolutionary sequence** — most of what changes between the curves is which
galaxies are in them. Quantified at 2.8 kpc: the apparent rise in central
stellar mass from *z* = 0.4 to *z* = 2 is **+0.247 dex as plotted**, but only
**+0.062 dex** when the same galaxies are followed. **Three quarters of it is
the changing sample.**

Following one fixed set (the 840 fair at *z* = 2) at every epoch gives the real
picture, median `log10 M*(<R)`:

| R [kpc] | 2.8 | 10.2 | 27.5 | 148.2 |
|---|---|---|---|---|
| z = 0.4 | 10.744 | 11.154 | 11.364 | 11.540 |
| z = 1.0 | 10.766 | 11.129 | 11.289 | 11.430 |
| z = 2.0 | 10.806 | 11.044 | 11.134 | 11.197 |

**This is inside-out growth, and it is the central physical fact the model has
to reproduce.** Between *z* = 2 and *z* = 0.4 the outskirt grows by 0.34 dex
while the central 2.8 kpc does not grow at all — it is very slightly *higher*
at *z* = 2. Galaxies of this mass finished their centres before *z* = 2 and
spent the next 9 Gyr adding mass at large radii.

(The small central *decrease* toward low redshift is not a measurement error:
TNG's enclosed mass genuinely decreases in 50–58% of these galaxies at ~5 kpc,
which is why the model carries a measured monotonicity floor rather than a
survival term.)

### 6.5 What is a limitation, not a bug

The framework describes the descendants of massive low-redshift galaxies. The
haloes it discards are ones whose growth stalls, whose centrals are typically
not massive ellipticals. Two consequences are accepted scope limits:

1. **The model cannot yet be applied to a halo sample selected at high redshift
   and evolved forward.** It has only ever been calibrated on backward-traced
   progenitors.
2. High-redshift performance must be *quoted* on the mh-complete sample, not the full
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
cells** (29 to 13, 34 to 18, 37 to 23, 36 to 22). The earlier verdict "concentration never
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

7. **The inner deficit is now localised, though not to a single cause.**
   21% too little stellar mass inside 3 kpc, ~5% too much at 20–70 kpc, total
   nearly right. Section 8.5.2 splits it: at 2 kpc and *z* = 2 the model is
   −37.4% for the 53% of cohort galaxies whose central mass DECLINED between
   *z* = 2 and *z* = 0.4, and only **−11.5%** for those it did not. No static
   non-negative deposition model can follow a decline at all, so the population
   figure is dominated by a failure mode a compact channel cannot address; and
   where the centre did not decline the ceiling recovers half of the remaining
   deficit, so the efficiency law is the culprit there. **The compact second
   deposit channel is deprioritised rather than retired**: it remains a
   conditional hypothesis for the non-declining galaxies.

8. **The model degrades on complete, massive samples** — ~36% worse than a
   regression at *z* ≥ 0.7, against 5–10% on the selected sample. This is an
   amplitude failure at high halo mass, and it is distinct from the inner
   deficit. It is currently unexplained and deserves its own investigation.

### 8.3 About two terms that turned out to be fitting the survey

9. **The size-law conditioning (`S3`) absorbs selection structure, not
   physics.** Two independently-constructed mh-complete samples both drive all three of
   its slopes to near zero from large full-sample values:

   | parameter | full sample | mh-complete sample A | mh-complete sample B |
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

16. **Bound the framework before enriching it, and price the bound's own
    freedom.** A ceiling computed with per-object freedom will always look
    reachable; what makes it interpretable is the CONTROL — fit the same
    machinery to a *different* object. Measured under the exact production
    objective, that control reaches `score_F` **0.319** against the true
    ceiling's **0.282** and the model's 0.898: the wrong galaxy costs 13 per
    cent, so roughly seven eighths of the apparent headroom is dictionary
    overcompleteness rather than information. Reporting the ceiling alone would
    have licensed exactly the extensions it was built to forestall.

16b. **A bound must be optimised on the same objects and the same functional it
    is reported on.** Three separate versions of that error appeared in one
    afternoon's work: a solve that used epochs its score excluded, an increment
    test weighted by one quantity and reported in another, and a surrogate loss
    quoted as if it were the production one — the last worth a factor of two
    (0.538 against the exact 0.282). Write the reported quantity down first,
    then optimise it.

16c. **Check whether a constraint binds the quantity you are reporting.** A
    monotone-in-time constraint on the ABSOLUTE profile places almost no floor
    under a shape loss that renormalises each epoch, because the later epochs
    can simply be inflated. Stage 3.0's "floor" of 0.29 in `score_F` was an
    artifact of the projection choosing to match the amplitude; the true
    shape-only bound is 0.009. A reference is not a bound until it is the
    argmin of the reported loss.

16d. **A control can be an identity.** Giving each galaxy one free amplitude
    per epoch interval cannot change its *z* = 2 normalised profile, because
    every pre-*z* = 2 deposit is in one interval — so that rung had to equal the
    fitted model at *z* = 2, and reading the equality as "no headroom" was a
    tautology. Before quoting a control, ask what it is *able* to change.

17. **A metric can be blind in a region and still look converged there.** The
    production loss is a fractional residual on the *cumulative* profile. At
    *z* = 2 only 6.3 per cent of the stellar mass lies outside 52 kpc, so a
    sub-per-cent error the loss cannot see is a thirty-per-cent error in that
    annulus. Before naming a defect, check whether the objective could have
    detected it at all.

18. **The completeness trap recurs in every new view.** The R^(1/4) figure's
    "the truth steepens to −3.38 by *z* = 2" is measured on the
    progenitor-selected sample; on the mh-complete one it steepens only to
    −2.95, and the model's error is +0.21 dex/dex rather than +0.48. Section
    6.4b said this about the inner profile in August; the same correction had
    to be made again for the outer slope. Apply the mask to a diagnostic the
    first time it is plotted, not after it has produced a headline.

---

## 8.5 The representational ceiling: what actually limits the model

Section 8 lists the failures. This section says which of them can be fixed
inside the framework, by bounding the framework instead of fitting it.

**The construction.** For each galaxy the deposits form a fixed **basis**: one
truncated unit-mass cumulative profile per MAH step, `R50` from the fitted size
law, truncated at `3 R200c(t_j)`. The model chooses their weights with seven
global parameters. Replace that choice with a free non-negative weight per
deposit, respecting causality — a deposit enters an epoch only if it was laid
down before it — and the residual bounds what any deposition model with this
basis could achieve given a perfect efficiency law.

**The sample.** Everything below is the **fixed z = 2-threshold cohort**: the
840 galaxies above the *z* = 2 mh-complete cut, followed at all five epochs.
The same objects at every epoch is what makes an evolution statement well
posed. It is a fixed massive-progenitor cohort and must not be read as a
general high-redshift population. The bound is also computed on the full
sample and on a per-galaxy epoch mask matching `MaskedProblem`; the ordering of
the ladder is the same in all three (Section 8.5.3).

**The objective.** Non-negative least squares minimises a fractional residual
on the *absolute* curve of growth, which is not the production shape loss. Both
`score_F` and `score_A^2` are means over galaxies of per-galaxy quantities, so
each can be minimised galaxy by galaxy and the population minimum obtained
EXACTLY; `stage34_objective.py` does that with analytic gradients (verified
against finite differences to 1e-8). Where the two differ the exact number is
quoted, and the difference is large.

### The ladder — `score_F`, cohort, lower is better

| bound | freedom per galaxy | mean over epochs |
|---|---|---|
| monotone in time, shape loss only | basis-free | **0.009** — *not a floor, see below* |
| monotone in time, amplitude + shape | basis-free | 0.195 |
| monotone in time, absolute-fractional loss (exact) | basis-free | 0.286 |
| each epoch fitted alone (not causal) | 5 x ~71 | ≤ 0.164 |
| **one causal weight vector, EXACT shape loss** | **~71** | **0.282** |
| the same under the NNLS surrogate | ~71 | 0.538 |
| *control*: the same fitted to ANOTHER galaxy, exact loss | ~71 | **0.319** |
| 8 deposit-group amplitudes (surrogate) | 8 | 0.614 |
| 5 epoch-interval amplitudes (surrogate) | 5 | 0.762 |
| **the fitted model** | 0 (7 global) | **0.898** |

Four things follow, and two of them contradict what an earlier draft of this
section claimed.

**1. The basis can draw the profiles.** With independent weights at each epoch
it reproduces every galaxy's curve of growth to a median **0.9–2.2 per cent**
RMS across the five epochs. The profile family, the size law and the truncation
are not the limitation, and a scan of the size law at the ceiling confirms it:
the fitted `(log_f0, b)` is already within 0.4 per cent of the ceiling-optimal
value.

**2. The limitation is that ONE causal weight vector must serve all five
epochs.** That is the whole gap between ≤0.164 and 0.282, with no change of
basis.

**3. But almost none of the model-to-ceiling gap is information a global law
could use.** The control that settles this is the **permuted** solve: one
galaxy's deposits fitted directly to a *different* galaxy's measured profile.
Under the exact production shape loss it reaches **0.319** against the true
ceiling's **0.282** — the wrong galaxy costs 13 per cent. **The dictionary is
overcomplete enough that halo-specific basis information contributes about an
eighth of the ceiling's advantage over the fitted model.** (What the permuted
control does *not* have is a matching halo basis; what it does have is full
access to the target's stellar profile. It measures how flexible the dictionary
is, not how much a predictor could learn.)

**4. Monotonicity does NOT put a floor under the shape loss.** This corrects
Stage 3.0's reference and an earlier draft of this section. The production
shape loss renormalises every epoch at 100 kpc, while monotonicity constrains
the *absolute* profile — and any set of shapes can be made non-decreasing by
inflating the later epochs' amplitudes. Optimised against the shape loss alone,
the basis-free monotone bound is **0.009**. Stage 3.0's isotonic projection
scored ~0.29 because it happened to match the amplitude, not because 0.29 was
unreachable. **The deposition-only premise binds only when amplitude and shape
are constrained together.**

### 8.5.1 Where the monotonicity conflict is real, and what it costs

It is real in the *absolute* profile, where it can be priced exactly. The causal
design is block-triangular: the deposits of one epoch interval are the only
things that can build that interval's rise in `M*(<R)`. Solving each interval in
the units it is reported in — an RMS error in the cumulative curve as a fraction
of `M*(<R)` at the later epoch — with a 400-sample galaxy bootstrap:

| interval | radii where the truth FALLS | cost of clipping that to zero | cost of the basis on the rise |
|---|---|---|---|
| z=2.0 to 1.5 | 32.2% | 3.68% [3.44, 3.83] | 3.17% [3.08, 3.26] |
| z=1.5 to 1.0 | 33.1% | 4.65% [4.43, 4.97] | 2.43% [2.27, 2.58] |
| z=1.0 to 0.7 | 38.2% | 2.35% [2.18, 2.57] | 1.93% [1.80, 1.98] |
| z=0.7 to 0.4 | 36.6% | 2.67% [2.53, 2.87] | 1.76% [1.69, 1.83] |

**The truth's own decline costs more than the basis does, at every interval.**
An earlier draft reported the opposite ordering; its solve minimised a
differently-weighted quantity from the one it printed.

### 8.5.2 The z = 2 central deficit: what is structural is the trade-off, not the number

An earlier draft claimed that two thirds of the *z* = 2 central deficit was
irreducible. **That claim is withdrawn.** Monotonicity guarantees that a
conflict exists between matching the *z* = 2 centre and matching the *z* = 0.4
centre; it does not fix which epoch pays. Re-solving the basis-free monotone
bound with the *z* = 2 epoch up-weighted traces the whole trade-off:

| z=2 epoch weight | `score_F` | z=2 residual at 2 kpc | z=0.4 residual at 2 kpc |
|---|---|---|---|
| 0.2 | 0.209 | −23.2% | +4.6% |
| 1.0 | 0.188 | **−17.2%** | +9.0% |
| 2.0 | 0.193 | −13.2% | +10.4% |
| 5.0 | 0.289 | −6.0% | +25.4% |
| 20.0 | 0.477 | −0.9% | +40.4% |

A deposition-only model can put the *z* = 2 central residual anywhere on that
curve. At the production weighting its optimum is **−17.2%**; the fitted model
sits at **−25.4%**, so about eight points are recoverable and the rest is the
price of the trade-off *at this weighting*.

**The mechanism split is sharper than any single number.** Preregistering
"declined" as a fall in `M*(<4.92 kpc)` between *z* = 2 and *z* = 0.4:

| group | n | model | ceiling | epoch-alone | monotone |
|---|---|---|---|---|---|
| centre declined | 443 (53%) | **−37.4%** | −22.4% | +3.6% | −25.9% |
| centre did not | 397 | **−11.5%** | −5.9% | +3.6% | −10.2% |

(median `100 (prediction − truth)/truth` at 2 kpc, *z* = 2.) **The population's
−25.4% is mostly the declining half**, where no static non-negative deposition
model can follow the data at all. Among the galaxies that did not decline the
model is only 11.5 per cent low and the ceiling recovers half of that, so a
compact channel is not the mechanism there either — the efficiency law is.

**The compact second channel is therefore DEPRIORITISED, not retired.** It
cannot address a temporal decline, which is the larger half of the effect. It
remains a conditional hypothesis for the non-declining galaxies, whose positive
interval increments are in fact the harder ones for the current basis (4.65 per
cent against 2.11 per cent for the decliners over *z* = 2 to 1.5).

### 8.5.3 Stability, and the freedom audit

**The ordering does not depend on the sample definition.** Mean `score_F` of the
surrogate ceiling is 0.559 solving on all five epochs, 0.503 solving only on
each galaxy's admitted epochs, and 0.538 on the fixed cohort; the model is
0.872, 0.872 and 0.898. The masked solve must be read with care — 542 of 2397
galaxies are admitted at one epoch only, where 24 radii against ~71 weights
bound nothing — which is why the cohort is the number quoted.

**The freedom audit.** The solve activates a median of **6 of ~71** weights, so
it is not using its nominal freedom evenly; it is finding a sparse, bursty
history. Capping `eps_surv` at the universal baryon fraction
`f_b = Omega_b/Omega_m` — no deposit may make more stars than it accreted
baryons — costs **+0.002** in mean `score_F`, so **the ceiling passes this
mass-budget check**. That is a necessary condition and not a sufficient one:
smoothness, merger delivery, stellar mass loss and time coherence are all
unconstrained.

`moffat-E2-S2` reproduces the whole ladder to within 0.03, so none of this
belongs to one profile family.

### 8.5.4 How much time resolution would a global law need?

The five-epoch-interval control used in an earlier draft has a hidden identity:
every deposit laid down before *z* = 2 sits in one interval, so rescaling its
single amplitude cannot change a profile renormalised at 100 kpc, and its
*z* = 2 score must equal the fitted model's exactly. Reading that identity as
"no headroom at *z* = 2" was wrong. Replacing it with `K` contiguous deposit
groups puts several independent components before *z* = 2:

| K | 1 | 2 | 3 | 5 | 8 | 12 | 20 | free |
|---|---|---|---|---|---|---|---|---|
| mean `score_F` | 0.898 | 0.817 | 0.661 | 0.645 | 0.614 | 0.598 | 0.570 | 0.538 |
| at z = 2 | 0.777 | 0.777 | **0.584** | 0.565 | 0.568 | 0.605 | 0.603 | 0.581 |

(K = 1 and K = 2 are the identity; the surrogate objective is used throughout,
so all of these are upper bounds.) **Most of the available improvement arrives
by K = 3–8**, i.e. at a temporal resolution a richer global redshift dependence
could plausibly express. That is the strongest positive result in this section
— and it must still be read against the permuted control, because these rungs
are oracles fitted to the stellar data.

### 8.5.5 A redshift-dependent deposit shape, tested at the bound and rejected

The extension that had been agreed as the next thing to build was one nested
parameter, `c = c0 + c_z ln(1+z)`, so that high-redshift deposits are
intrinsically more concentrated. It was tested where it cannot be rescued by
refitting something else: the *global* basis parameters were scanned on a 7x7
grid with the per-galaxy freedom held fixed, so a drop in the surface would be a
statement about the basis and not about degrees of freedom.

The optimum is at **`c_z` = 0** at both freedom levels scanned, on an interior
minimum, and re-measured on all 2397 galaxies and on the 840-galaxy cohort the
best non-zero `c_z` is **worse** by +0.001 and +0.019 in mean `score_F`. The
size law is likewise already at its ceiling optimum, worth +0.4 per cent.

This is strong negative evidence against the proposed mechanism rather than a
proof of impossibility: the scan minimises the ceiling's surrogate loss, and an
over-complete dictionary can be insensitive to a basis change a rigid global
law would still feel. **The extension is not built.**

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

**What the ceiling changed.** Section 8.5 bounds the framework instead of
fitting it, and it moved the diagnosis twice — once against the queued
extensions and once against an earlier draft of this very section. The basis
can draw any single epoch's profile to about one per cent, so neither the
profile family nor the size law is the limitation. What costs is that one
causal, non-negative weight vector must serve all five epochs. But the exact
production-objective ceiling is **0.282** against the model's **0.898**, and
the same machinery fitted to the WRONG galaxy reaches **0.319** — so most of
that gap is the dictionary being overcomplete, not physics a global law could
learn.

**What comes next**, in order:

1. Drop `S3`; treat `E5` as suspect for the same reason.
2. **Enrich the efficiency law's TIME dependence.** The temporal-resolution
   ladder (Section 8.5.4) shows most of the reachable improvement arriving by
   three to eight independent components in time — a resolution a richer global
   redshift dependence could plausibly express. This is the strongest positive
   lead the ceiling produced, and it replaces the compact channel at the top of
   the queue.
3. **Repair the objective before the profile family.** Section 8.5 shows the
   production loss is nearly blind to the outskirts at high redshift, and
   Section 8.5's monotone bound shows it is nearly blind to monotonicity as
   well once each epoch is renormalised. exp48 already measured a
   density-plus-log-residual objective that helps the compact-centre defect; it
   was not adopted for unrelated reasons and deserves re-examination now that
   the blindness is quantified.
4. Investigate the high-mass amplitude failure — why the model degrades where
   the data are best. Not explained by the shape ceiling, and still the
   deployment case.
5. ~~The compact second channel~~ — **deprioritised, not retired** (Section
   8.5.2). It cannot address a temporal decline, which is the larger half of
   the *z* = 2 central deficit; it remains a conditional hypothesis for the
   non-declining galaxies, where the current basis in fact struggles more with
   the positive increments.
6. ~~A redshift-dependent deposit shape `c = c0 + c_z ln(1+z)`~~ — **rejected
   at the bound** (Section 8.5.5), at two freedom levels and confirmed on the
   full sample.
7. A delivery-delay kernel remains a last resort. If built, it must carry an
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
| the representational ceiling, its ladder and its controls | `stage34_ceiling.py` |
| the EXACT production-objective bounds and the trade-off sweep | `stage34_objective.py` |
| the basis-parameter scan (size law, `c_z`) | `stage34_basis_scan.py` |
| QA figures | `qa_figures.py`, `stage34_figures.py` |
