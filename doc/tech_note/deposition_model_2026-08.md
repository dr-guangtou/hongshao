# Building a massive galaxy out of its dark-matter halo's growth history

**What this is.** A technical record of the exp54 experiment: an attempt to
predict, from the growth history of a dark-matter halo alone, how much stellar
mass a massive galaxy contains and how that mass is arranged with radius, at
five cosmic epochs between redshift 2 and redshift 0.4.

*Status: 2026-08-24, after Stages 0 to 3.4 and an independent internal review.
Supersedes `experiments/exp54_unpinned_amplitude/README.md`.*

**How to read the numbers in this note.** Every result is quoted three ways:
what was measured, what it means for the galaxy (which part of the galaxy, too
much mass or too little, by how much), and what it should be compared against.
Where a figure exists that shows the effect, its filename is given. Nothing is
referred to by a code name without being described first; the code names
themselves are collected in Appendix A so that this note can be matched to the
software.

---

## 1. The question, the data, and what would count as success

### 1.1 The question

Given only what a dark-matter halo does — how its mass grows with time, how
large it is, how centrally concentrated — can we predict the stellar mass of the
massive galaxy at its centre, and the way that stellar mass is distributed with
radius, at several epochs?

Nothing about the stars may enter the prediction. The model is handed halo
quantities and must produce a stellar profile. This is enforced in the code: the
per-galaxy record the model reads contains no stellar field at all, so there is
nothing to leak.

### 1.2 The galaxies

2397 galaxies from the TNG300 cosmological simulation.

**They were selected by halo mass, not by stellar mass.** The parent sample is
massive central haloes chosen by their peak halo mass at redshift 0.4; 2397 is
the subset with usable cross-matched profiles. Their stellar masses happen to
span `log10 M* = 10.66` to `12.36`, but that describes the outcome, not the
criterion. This distinction matters a great deal in Section 6.

For each galaxy we have:

- **How the halo grew**: halo mass at 73 simulation snapshots, taken from a
  smooth analytic fit (DiffMAH) to the main branch of the merger tree. Using a
  smooth fit rather than the raw history removes merger-scale fluctuations, and
  Section 5.2 notes where that may cost us.
- **How the halo was built**: its radius `R200c` — the radius inside which the
  mean density is 200 times the critical density of the Universe at that epoch,
  the standard working definition of a halo's edge — and how centrally
  concentrated it is, at 16 snapshots.
- **What the galaxy actually looks like**: the measured stellar profile at five
  epochs.

### 1.3 What we are trying to reproduce

The **measured stellar profile** is a *curve of growth*: the stellar mass lying
inside an aperture, as a function of the aperture's size, on a fixed 24-point
grid of semi-major axes running from 2 kpc to 148 kpc.

Two properties of it matter and are easy to get wrong:

- **It is a projected quantity, not a spherical one.** It is built by
  integrating the surface density measured on the simulated image, outward
  through elliptical isophotal annuli. It is the mass seen inside a
  two-dimensional aperture on the sky, and it is *not* the mass inside a sphere.
  The two are related by a projection integral and are not interchangeable.
- **It is cumulative**, so it can only rise as the aperture grows. It cannot
  tell you directly what the density is doing at a given radius; for that the
  curve has to be differenced, which Section 5.4 does.

**The five epochs** are redshift 0.4, 0.7, 1.0, 1.5 and 2.0. At the four higher
redshifts we follow each galaxy's *main progenitor*, the most massive branch of
its merger tree back through time.

### 1.4 The modelling assumption being tested

The model builds a galaxy by **laying down stellar mass and never moving it
again**. At each step of the halo's growth it deposits a small amount of stellar
mass with a fixed radial profile, and the galaxy at any later time is simply the
sum of everything deposited up to then. There is no subsequent migration, no
dynamical rearrangement, no stripping of what has already been placed.

That is a strong assumption, and testing it is the point of the experiment: it
is the simplest thing that could work. Section 8 measures how much it costs.

---

## 2. The model

### 2.1 Three equations

Index the steps of the halo's growth history by `j`. At step `j` the halo gains
mass `dMh_j`, and has a mass, a redshift, a radius and a concentration.

**(1) How much stellar mass this step contributes.** A fraction of the halo mass
gained becomes stellar mass in the central galaxy:

$$\Delta M_{*,j} \;=\; \varepsilon(H_j)\,\Delta M_{h,j}$$

**(2) How far out that stellar mass is spread.** Each deposit is cut off at
three times the halo's radius at the time it was laid down:

$$R_{{\rm trunc},j} \;=\; 3\,R_{200c}(t_j)$$

**(3) What the galaxy looks like at any epoch.** Add up everything deposited so
far:

$$M_*(<R,\,t_k) \;=\; \sum_{j\,:\,t_j \le t_k} \Delta M_{*,j}\; F_j(<R)$$

Here `F_j(<R)` is the fraction of deposit `j`'s mass lying inside radius `R`: a
curve rising from 0 to 1, cut off at `R_trunc,j` and rescaled so it reaches
exactly 1 there.

That is the entire model.

### 2.2 What the efficiency factor is, and what it is not

The fraction `epsilon` is **the stellar mass that ends up in the central galaxy
per unit of halo mass gained**, measured at the epoch we observe.

**It is not a star-formation efficiency and must never be quoted as one.** A
single number is standing in for at least five separate physical processes:

1. how many of the accreted baryons turn into stars;
2. how much of that stellar mass is returned to gas as stars die;
3. how much is tidally stripped away;
4. whether the stars end up in the central galaxy or in an orbiting satellite;
5. stars formed elsewhere and delivered later by mergers.

Because it absorbs all five, no physical reading of its *value* is legitimate.
What is legitimate is how it *depends* on halo mass and redshift, because that
dependence is what the fit actually measures.

### 2.3 Why each deposit is cut off, and why the cut-off is a choice we made

Without a cut-off, "the total mass of a deposit" is a mass at infinity, which
for a profile with a heavy tail is a badly behaved quantity. Concretely: one of
the profile shapes we tried falls off as `R^-0.76`, so a deposit with a 50 kpc
half-mass radius would have to extend to **9.6 Mpc** to contain 99 per cent of
its mass. An earlier version of this model ended up with **18 per cent of its
stellar mass beyond 500 kpc** for exactly this reason.

Cutting each deposit off at three halo radii fixes that: the deposit's total is
then the mass inside a radius the halo itself defines, which is finite and
changes with epoch as the halo does.

**The factor of three is an empirical convention we chose, not a fitted or
preferred value.** It was picked to keep an extended stellar envelope while
giving each deposit a finite, halo-scaled mass budget. It was never fitted to
the simulated profiles and no comparison across values of it has been run.
Because the model's total stellar mass and its fitted efficiency both depend on
this choice, **the two must always be quoted together with it**. A sensitivity
test at cut-offs of one, two, three and five halo radii is on the to-do list.

**One subtlety that had to be handled carefully.** Cutting a profile off moves
its half-mass radius inward, because half of what remains sits inside a smaller
radius than half of the original did. Labelling a truncated profile by its
*untruncated* half-mass radius therefore measures the wrong thing, and makes two
differently-shaped profiles incomparable. The mislabelling was measured at
**+10 to +35 per cent**. The model therefore works throughout in terms of the
*true, post-truncation* half-mass radius, and a numerical self-check verifies
for every profile shape that the curve really does reach one half at the
half-mass radius (to better than 5 parts in 1000) and exactly one at the
cut-off.

---

## 3. How a model is judged

### 3.1 Two separate questions

A predicted profile can be wrong in two independent ways — it can contain the
wrong total mass, or it can put the right mass in the wrong place — and a single
number cannot distinguish them. So every prediction is split:

- **the total**: the stellar mass inside 100 kpc;
- **the shape**: the profile divided by that total, so it runs from something
  small at 2 kpc to exactly 1 at 100 kpc.

### 3.2 The total-mass error

For each galaxy and epoch we take the error in `log10` of the mass inside
100 kpc, and quote its root-mean-square across the population **in dex**. A dex
is a factor of ten, so 0.1 dex is a 26 per cent error and 0.2 dex is a factor
of 1.6.

For comparison we also fit a **plain statistical regression** — no physics, just
a fit of the same halo properties to the stellar mass, refitted separately at
each epoch. That is the benchmark: it is the best that can be extracted from the
same information without any model of how galaxies form. Its scatter is

| | z = 0.4 | z = 0.7 | z = 1.0 | z = 1.5 | z = 2.0 |
|---|---|---|---|---|---|
| statistical benchmark, dex | 0.104 | 0.112 | 0.120 | 0.128 | 0.161 |

Throughout this note, total-mass errors are given **both** in dex and as a ratio
to this benchmark, so that "1.30" always means "30 per cent more scatter than a
straight statistical fit to the same halo properties".

### 3.3 The profile-shape error

For each galaxy and epoch we take the fractional error of the normalised
profile, `(model - truth)/truth`, at each of the 24 radii, average it in
quadrature across radii, then average over epochs and galaxies. The result is
**a typical per-cent error in the shape of the cumulative profile**.

The previous generation of this model scored **15.8 per cent** by this measure.
Throughout this note, shape errors are given **both** in per cent and as a ratio
to that 15.8 per cent, so "0.90" always means "10 per cent better than the
previous generation".

### 3.4 What is fitted, and on which epochs

All five epochs are fitted simultaneously, with one exception: the 45-model
comparison of Section 4 fits only the two endpoints (redshift 0.4 and 2.0) for
affordability, in which case the three intermediate epochs are the interpolated
ones. **At no stage is redshift 2 an extrapolation.** Holding it out entirely
and predicting it would be a much stronger claim than fitting it, and is on the
to-do list.

Galaxies the model cannot represent at all are **priced into the loss, never
dropped** — dropping them would make a failing model's score silently improve.

### 3.5 Whether the fitted parameters are actually determined

A parameter can be useless with a sharp optimum, or genuinely useful and
completely undetermined, and the loss distinguishes neither. Every fit therefore
also reports:

- how nearly singular the fit is, via the ratio of the smallest to the largest
  singular value of the scaled derivative matrix. A small ratio means some
  combination of parameters barely changes the prediction;
- a **one-parameter conditional slice** for each parameter: the loss as that
  parameter is displaced with all others **held fixed**. This is *not* a profile
  likelihood. A genuine profile would re-optimise the other parameters at every
  point, and where parameters compensate one another the conditional slice can
  rise steeply along a direction the profiled curve leaves nearly flat. **The
  re-optimised version has not been run** and is on the to-do list; until it is,
  the conditional slices should not be quoted as evidence that a parameter is
  determined;
- how stable the parameters are when galaxies are resampled.

**A parameter that fails these checks may stay in the model, but its value is
never quoted as a measurement.**

### 3.6 Why models are ranked by a panel and not by the loss

Models are ranked by their **mean rank across seven criteria**: the total-mass
error at the two endpoint epochs, the shape error at the two endpoint epochs,
how well-determined the parameters are, an outer-aperture bias diagnostic, and
the **halo-mass tilt** — the slope of the model's error against halo mass across
the population, in dex of stellar-mass error per dex of halo mass. A tilt of
zero means the model is equally right for light and heavy haloes; a non-zero
tilt is a systematic bias the population average hides.

**This is not a stylistic preference.** Ranking the 45-model comparison by loss
instead picks models the panel places 2nd, 6th and 10th, and those loss-optimal
models are an order of magnitude *worse* conditioned — their parameters are
close to undetermined. **The loss systematically rewards degenerate models.**

---

## 4. What was tried

Three parts of the model were varied independently, each nested inside the
richer option so that "does this extra term earn its place?" is a well-posed
question. 45 combinations were fitted and ranked.

**How much stellar mass a step deposits** was allowed to depend on:

1. halo mass and redshift, each as a simple power law (3 parameters);
2. the same, **plus a cross-term letting the halo-mass dependence itself change
   with redshift** (4 parameters);
3. either of the above plus how concentrated the halo is relative to what is
   typical for its mass and redshift (one extra parameter);
4. halo mass as a power law, with a free flexible curve in redshift instead of a
   power law — built specifically to look for a peak in efficiency at some
   epoch (6 parameters);
5. a double power law in halo mass, rising below a characteristic mass and
   falling above it — the shape that abundance-matching studies find for the
   stellar-to-halo-mass relation (5 parameters).

**What shape a single deposit has** was varied over six functional forms,
including the Sérsic profile, a power-law-tailed Moffat profile, and a
sigmoid-in-log-radius form. The comparison used the first three.

**How large a deposit is** was set by:

- a fixed fraction of the halo's radius at the time of deposition, with a free
  power-law dependence on `(1+z)` (2 parameters); or the same scaled to the
  halo's inner scale radius instead of its outer radius;
- either of those plus extra dependence of the size on halo mass, concentration
  and how early the halo assembled (3 extra parameters).

---

## 5. What the model does

### 5.1 The model that won

The best model out of 45, by the seven-criterion panel, and still the best after
two independent corrections to the scoring (Section 7):

- **Efficiency**: a power law in halo mass and in `(1+z)`, plus a cross-term
  allowing the halo-mass dependence to change with redshift.
- **Deposit size**: a fixed fraction of the halo's radius at deposition, with a
  free power-law redshift tilt.
- **Deposit shape**: a sigmoid in log radius, cut off at three halo radii.

**Seven parameters in total, shared by all 2397 galaxies. No per-galaxy freedom
whatsoever, and no use of the true stellar mass anywhere.**

### 5.2 What it gets right

**The radial profile is reproduced well, and better at higher redshift.** On
all 2397 galaxies the shape error is **14.5, 14.5, 13.8, 12.6 and 11.3 per
cent** at redshift 0.4, 0.7, 1.0, 1.5, 2.0 — that is 0.92, 0.92, 0.88, 0.80 and
0.72 times the previous generation's 15.8 per cent, so the model is 8 per cent
better at redshift 0.4 and 28 per cent better at redshift 2. See
`figures/qa_cog_gompertz_log-E2-S2_full.png`.

The fitted values, for reproducibility only — Section 3.5 explains why they are
not yet quotable as measurements: efficiency `a0 = -2.549`, halo-mass slope
`a_M = -0.452`, redshift slope `a_z = 0.488`, cross-term `a_Mz = 0.233`;
size law `log10 f0 = -0.952`, redshift tilt `b = -0.735`; deposit shape
`c = 0.919`.

**The average growth in stellar mass is right.** Between redshift 2 and redshift
0.4 the model's median stellar mass inside 100 kpc grows to within **0.03 dex**
(7 per cent) of the simulation's. The previous generation needed a hand-applied
correction of 0.398 dex — a factor of 2.5 — for the same quantity, which it did
not predict at all.

**But that is a population average, not a per-galaxy accuracy.** Measured over
the same 2397 galaxies: the median growth error is +0.034 dex, the **per-galaxy
scatter is 0.208 dex** (a factor of 1.6), the correlation between predicted and
true growth is 0.71, and the spread of predicted growth histories is **about
half as wide** as the simulation's (0.145 dex against 0.284 dex). So the model
gets the *typical* galaxy's growth right, is often substantially wrong about an
*individual* galaxy, and does not reproduce the diversity of growth histories at
all. That last point is a property of any single deterministic law, and is where
a stochastic layer will eventually be needed; it is not a defect of the mean
relation.

**It does not overfit.** On 1197 galaxies never used in the fit, the total-mass
scatter is the same as in-sample at every epoch (0.116/0.120/0.122/0.131/0.151
dex held out, against 0.114/0.118/0.123/0.131/0.153 dex in sample). Seven
parameters shared across 1199 galaxies do not memorise them.

### 5.3 What it gets wrong: the total stellar mass

**A plain statistical fit to the same halo properties predicts the total stellar
mass better than the model does, at every epoch.**

| error in `log10 M*(<100 kpc)` | z = 0.4 | z = 0.7 | z = 1.0 | z = 1.5 | z = 2.0 |
|---|---|---|---|---|---|
| statistical benchmark, dex | 0.104 | 0.112 | 0.120 | 0.128 | 0.161 |
| model on all 2397 galaxies, dex | 0.118 | 0.123 | 0.127 | 0.135 | 0.169 |
| ratio to the benchmark | 1.13 | 1.10 | 1.06 | 1.06 | 1.05 |
| model on the completeness-controlled sample, dex | 0.117 | 0.152 | 0.161 | 0.174 | 0.221 |
| ratio to the benchmark | 1.12 | **1.36** | **1.34** | **1.36** | **1.37** |

On the sample as originally selected, the model is 5 to 13 per cent worse than a
regression. On a sample where the selection has been controlled for
(Section 6), it is **about 36 per cent worse at redshift 0.7 and above** — a
scatter of 0.15 to 0.22 dex, meaning a typical galaxy's predicted total stellar
mass is off by 40 to 65 per cent. The regression barely notices the restriction
(its own scatter falls by only 3 to 15 per cent); the model degrades sharply.

**The plain reading: the model is weakest exactly where the data are best** — on
massive, well-resolved, completely sampled haloes. That is also the case we
would want to deploy it on, and it is currently the strongest argument against
the model as it stands.

### 5.4 What it gets wrong: where the mass is put

**In the most massive haloes at redshift 2, the model puts far too little
stellar mass in the centre and about the right amount further out.** Splitting
the completeness-controlled sample into three bins of halo mass at redshift 2,
and quoting the median fractional error `(model - truth)/truth` at each
aperture:

| halo mass at z = 2 | n | 2.8 kpc | 4.9 kpc | 10.2 kpc | 27.5 kpc | 79.8 kpc | 148 kpc |
|---|---|---|---|---|---|---|---|
| 12.80 to 12.91 | 280 | −22% | −17% | −12% | −12% | −12% | −12% |
| 12.91 to 13.09 | 280 | −28% | −19% | −14% | −12% | −12% | −13% |
| **13.09 to 14.12** | 280 | **−29%** | −16% | **−2%** | **+2%** | −3% | −5% |

The two lighter bins are uniformly about 12 per cent too light everywhere, which
is a total-mass problem. **The heaviest bin is different: it is 29 per cent too
light inside 3 kpc and essentially correct from 10 kpc outward.** That is a pure
central failure. It deepens steadily with redshift — the same bin is 13 per cent
too light in the centre at redshift 0.4, 20 per cent at redshift 1.0 and 29 per
cent at redshift 2.

Galaxy-to-galaxy scatter is wide (the 16th-to-84th percentile spread at 2.8 kpc
is about 60 per cent), but with 280 galaxies per bin the median is determined to
about 2 per cent. See `figures/qa_cogmass_gompertz_log-E2-S2.png` and
`figures/qa_resid_gompertz_log-E2-S2.png`.

**These are exactly the compact, early-forming systems the project set out to
study**, so this is not a peripheral failure. Section 8 shows, however, that
about half of it is a failure mode no model of this kind can fix, and that a
compact extra component would therefore be fitting the wrong thing.

### 5.5 What it gets wrong: the outskirts are too shallow — but half of that was the sample

Because the curve of growth is cumulative it is dominated by the inner regions
and hides what the outer profile is doing. Differencing it into a surface
density and measuring the outer logarithmic slope over 50 to 148 kpc exposes a
second defect. See `figures/qa_dens_gompertz_log-E2-S2_full.png`, which plots
the surface density against `R^(1/4)` — the coordinate in which a de Vaucouleurs
profile is a straight line.

| outer slope, `d log Sigma / d log R` | z = 0.4 | z = 0.7 | z = 1.0 | z = 1.5 | z = 2.0 |
|---|---|---|---|---|---|
| simulation, all 2397 galaxies | −2.78 | −2.85 | −2.91 | −3.01 | **−3.38** |
| model, all 2397 galaxies | −2.74 | −2.78 | −2.81 | −2.86 | −2.90 |
| difference (positive = model too shallow) | +0.04 | +0.07 | +0.10 | +0.15 | **+0.48** |

Read at face value, the simulated galaxies steepen sharply toward high redshift
while the model's barely change. **Most of that apparent steepening is the
sample changing, not the galaxies evolving.** On the completeness-controlled
sample the simulation's slope steepens only to **−2.95** at redshift 2, and
following one fixed set of galaxies it runs from −2.56 to −2.95. The model's
error at redshift 2 is then **+0.21** with the adopted deposit shape and
**+0.06** with a power-law-tailed one — not +0.48. About half of the headline
defect was the changing sample, the same trap as Section 6.4.

Section 8.4 shows the remainder is largely invisible to the loss we fit, rather
than impossible to represent.

### 5.6 The under-filled centre, at all masses

Across all galaxies, the median error in `log10 M*(<R)` — negative meaning the
model puts less mass inside `R` than the simulation:

| aperture | 2.8 kpc | 4.9 kpc | 10.2 kpc | 27.5 kpc | 79.8 kpc | 148 kpc |
|---|---|---|---|---|---|---|
| z = 0.4, all galaxies | −0.034 | −0.040 | −0.004 | +0.008 | −0.008 | −0.016 |
| z = 2.0, all galaxies | −0.084 | −0.065 | −0.038 | −0.026 | −0.024 | −0.024 |
| z = 2.0, completeness-controlled | **−0.133** | −0.081 | −0.039 | −0.030 | −0.040 | −0.043 |

Three readings:

1. **The central deficit is real but modest at low redshift**: −0.034 dex at
   2.8 kpc at redshift 0.4, i.e. the model is 7.5 per cent too light there. (An
   earlier generation of this model was 0.39 to 0.47 dex too light — a factor of
   two and a half.)
2. **It grows strongly with redshift**, reaching −0.133 dex — 26 per cent too
   light — on the completeness-controlled sample at redshift 2.
3. **At low redshift it is misplaced mass; at high redshift it is also missing
   mass.** At redshift 0.4 the outer profile is right to 0.016 dex, so the total
   is about right and only the distribution is wrong. At redshift 2 the whole
   curve is 0.043 dex low with the centre lower still — two errors superposed.

No profile shape fixes the inner part: the sigmoid, Moffat and Sérsic forms give
the same curve. Numerical resolution is unlikely to explain it away, since
gravitational softening makes the simulation's cores *less* concentrated than
reality — but that is a statement about a bias inside the simulation and **not**
a one-sided bound on the deficit against the real Universe, which this
experiment cannot measure.

---

## 6. Why the high-redshift sample is not what it looks like

### 6.1 The problem

Every galaxy was chosen at redshift 0.4. The higher-redshift samples are
therefore **not high-redshift samples**: they are the ancestors of a
low-redshift selection. At redshift 2, a halo of a given mass appears in our
sample only if it goes on to grow fast enough to clear the redshift-0.4
threshold. At low halo mass we keep only a fast-growing minority; at high halo
mass we keep essentially everything.

### 6.2 Measuring it

We counted our galaxies against the **full** simulation catalogue — 346,000 to
460,000 central haloes per epoch — in bins of halo mass, to get the fraction of
all haloes of a given mass that are in our sample.

At redshift 0.4 that fraction tops out at **0.781**, set by data-quality cuts
applied to the redshift-0.4 galaxy, which therefore cost the same at every
epoch. We then define a working threshold: the halo mass above which the
fraction first reaches 60 per cent of that maximum and stays there.

| | z = 0.4 | z = 0.7 | z = 1.0 | z = 1.5 | z = 2.0 |
|---|---|---|---|---|---|
| threshold, `log10 M200c` | > 13.00 | > 13.00 | > 13.00 | > 12.90 | > 12.80 |
| galaxies kept | 2397 | 1781 | 1436 | 1145 | 840 |

**This is a threshold, not completeness.** Above it we hold roughly 47 per cent
of all haloes of that mass, not all of them; the cut removes the regime where we
hold almost none. Everywhere below, "the completeness-controlled sample" means
exactly this cut. At redshift 0.4 it keeps everything, which is the right sanity
check — there is no ancestor bias at the selection epoch.

### 6.3 One of the model's two apparent defects was the survey

With the model's parameters frozen, its bias as a function of halo mass at
redshift 2 (in dex of stellar-mass error per dex of halo mass):

| bias slope at z = 2 | 5 kpc | 10 kpc | 75 kpc | 148 kpc |
|---|---|---|---|---|
| all galaxies | −0.135 | −0.075 | −0.083 | −0.094 |
| completeness-controlled | −0.011 | **+0.066** | **+0.099** | **+0.089** |

**It does not shrink — it changes sign**, and lands on the same positive value
the controlled sample shows at redshifts 0.7 to 1.5. Removing the badly sampled
regime makes redshift 2 consistent with the other epochs instead of an outlier.

Meanwhile the model's *total-mass* deficit at redshift 2 gets **worse**, from
−0.026 to −0.042 dex. So of the two things we believed were wrong at high
redshift, one was an artefact of the survey and the other had been
under-reported.

### 6.4 Why this is selection and not just a narrower mass range

Two further tests.

**The mechanism.** A selection can only bias a residual if the residual depends
on the quantity the selection cut on. Here that quantity is **future growth** —
how much the halo grows *after* the epoch being scored, which is what the
redshift-0.4 selection cuts on and which the model, reading only what is
available at the epoch, cannot see. The model's error does depend on it: partial
correlation +0.19 to +0.23, slope +0.15 to +0.18 dex per dex, significant at
every epoch and in the direction required.

**The size.** Feeding the measured sampling fraction and that measured slope
into a truncated-Gaussian calculation predicts how large the bias shift should
be, using no fitted model quantity at all:

| shift in the bias slope at 148 kpc | observed | predicted, lower estimate | predicted, truncation-corrected |
|---|---|---|---|
| z = 0.7 | −0.015 | −0.017 | −0.044 |
| z = 1.0 | −0.033 | −0.030 | −0.084 |
| z = 1.5 | −0.067 | −0.043 | −0.136 |
| z = 2.0 | −0.183 | −0.054 | −0.192 |

**The observation lies inside the predicted bracket at every epoch**, with the
lower estimate matching where the truncation is mildest and the corrected one
matching redshift 2 where it is most severe — which is how each should behave if
the mechanism is the right one. **The size of the effect needs no new
physics.**

### 6.5 A trap when reading figures of the controlled sample

**The controlled sample is a different set of galaxies at each epoch** — 2397
falling to 840 — and progressively more massive (median halo mass rising from
`log10 Mh = 13.29` to `13.62`). A figure that plots each epoch's controlled
sample side by side is therefore **not an evolutionary sequence**: most of what
changes between the curves is which galaxies are in them.

Quantified at 2.8 kpc: the apparent rise in central stellar mass from redshift
0.4 to redshift 2 is **+0.247 dex as plotted**, but only **+0.062 dex** when the
same galaxies are followed. **Three quarters of it is the changing sample.** The
figure `figures/qa_cog_gompertz_log-E2-S2_perepoch.png` shows all three views
side by side for exactly this reason.

Following one fixed set of galaxies gives the real picture, median
`log10 M*(<R)`:

| aperture | 2.8 kpc | 10.2 kpc | 27.5 kpc | 148 kpc |
|---|---|---|---|---|
| z = 0.4 | 10.744 | 11.154 | 11.364 | 11.540 |
| z = 1.0 | 10.766 | 11.129 | 11.289 | 11.430 |
| z = 2.0 | 10.806 | 11.044 | 11.134 | 11.197 |

**This is inside-out growth, and it is the central physical fact the model has
to reproduce.** Between redshift 2 and redshift 0.4 the outskirts grow by 0.34
dex — more than a factor of two — while the central 2.8 kpc does not grow at
all. It is very slightly *higher* at redshift 2. These galaxies finished their
centres before redshift 2 and spent the next 9 Gyr adding mass at large radii.

That small central *decrease* is not a measurement artefact: the simulation's
enclosed mass genuinely falls between redshift 2 and redshift 0.4 in **50 to 58
per cent** of these galaxies at about 5 kpc. Section 8.3 shows this is the
single most important constraint on the whole framework.

### 6.6 Accepted scope limits

1. **The model has not been validated on haloes selected at high redshift and
   evolved forward.** It has only ever been calibrated on ancestors traced
   backward from redshift 0.4.
2. **High-redshift performance must be quoted on the controlled sample**, since
   the full one flatters the model (Section 5.3).

---

## 7. Two arithmetic errors we found, and what they changed

### 7.1 A halo mass read in the wrong units

The halo-structure catalogue stores halo mass as `log10(M/Msun)`. Two modules
read it as a linear mass in units of `1e10 Msun/h`, so the reference
concentration was evaluated at a mass **two dex too low** — a factor of 100.

The consequence was a spurious 0.15 dex drift with redshift in the concentration
variable, and a halo-mass slope that changed sign at redshift 0.4. **33 of the
45 model variants read that variable.** After re-running all 45, the winning
model was unchanged, but the four largest improvements in rank were all variants
using concentration — so the earlier verdict that "concentration never reaches
the top 15" was partly an artefact of feeding it a corrupted input. Correcting
it does not make concentration competitive, but it had not previously been given
a fair hearing.

### 7.2 One corrupt galaxy inflating the benchmark

One galaxy's measured stellar mass inside 100 kpc collapses from `10^11.71` at
redshift 0.4 to a flat `~10^8` at all four higher redshifts — 2.5 dex below the
sample's 0.1st percentile. A main progenitor cannot be 3.76 dex lighter than its
own descendant; this is a broken cross-match, not a galaxy.

It sat in the 2397-galaxy set that defines the **statistical benchmark**, and in
none of the model-fitting subsets. **So every reported comparison divided a
model error measured on clean galaxies by a benchmark charged for a corrupt
one.**

| statistical benchmark, dex | z = 0.4 | z = 0.7 | z = 1.0 | z = 1.5 | z = 2.0 |
|---|---|---|---|---|---|
| as published | 0.105 | 0.136 | 0.142 | 0.146 | 0.174 |
| with that galaxy rejected | 0.104 | 0.112 | 0.120 | 0.128 | 0.161 |

**This overturned the previous headline.** "The model beats the best statistical
fit at four of five epochs" became "it is 5 to 13 per cent worse at every
epoch".

**The damage was confined to the total-mass side.** That galaxy moved the
population *shape* error by **0.04 per cent**. The reason is structural: the
total-mass error is a logarithm and is unbounded, and this galaxy's was 3.76
dex, so its square swamped 2396 others; the shape error is computed on a profile
normalised to 1, lives near unity, and no single galaxy can move a
2397-galaxy mean. Every shape result stands.

---

## 8. How good could a model of this kind ever be?

Sections 5 and 6 say what the model gets wrong. This section asks a different
question: **of those failures, which could be fixed by a better model of the
same kind, and which are built into the assumption itself?**

The method is to stop fitting the model and start bounding it. For each galaxy,
keep everything the model builds — the same deposits, the same sizes, the same
shapes, laid down at the same times — and throw away only the rule that decides
*how much mass each deposit carries*. Instead let every deposit's mass be chosen
freely and independently, per galaxy, subject only to being non-negative and to
causality: a deposit cannot contribute to an epoch before it was laid down.

Whatever error remains after that is error **no deposition model of this kind
can remove**, however clever its efficiency rule.

**The galaxies used.** All numbers in this section are for the **840 galaxies
above the redshift-2 threshold of Section 6.2, followed at all five epochs** —
the same objects throughout, which is what makes a statement about evolution
well posed. This is a fixed set of massive ancestors and must not be read as a
general high-redshift population. The same calculation on the full sample and on
a per-galaxy epoch mask gives the same ordering of everything below.

### 8.1 The result, in one table

Profile-shape error, as a typical per-cent error of the normalised cumulative
profile, and as a ratio to the previous generation's 15.8 per cent:

| what is allowed to vary | free numbers per galaxy | shape error | ratio |
|---|---|---|---|
| anything at all, as long as enclosed mass never falls with time | — | 0.1% | 0.009 |
| the same, but the total mass must also be right | — | 3.1% | 0.195 |
| every deposit's mass, fitted **separately at each epoch** | 5 x ~71 | 2.6% | ≤ 0.164 |
| **every deposit's mass, one set serving all five epochs** | **~71** | **4.5%** | **0.282** |
| *the same, fitted to a DIFFERENT galaxy's measured profile* | ~71 | **5.0%** | **0.319** |
| the mass in each of 8 broad time intervals | 8 | 9.7% | 0.614 |
| the mass in each of the 5 intervals between epochs | 5 | 12.1% | 0.762 |
| **the fitted model: 7 numbers shared by all galaxies** | **0** | **14.2%** | **0.898** |

The corresponding figure is `figures/stage34_ladder_gompertz_log-E2-S2.png`.

Four things follow.

**1. The deposits themselves are not the problem.** Allowed to refit their
masses separately at each epoch, they reproduce every galaxy's measured profile
to within about 1 to 2 per cent at every radius. Whatever is wrong, it is not
the profile shape, the deposit size law or the cut-off. A direct scan confirms
this: the fitted size law is already within 0.4 per cent of the best any size
law could do here.

**2. What costs is that one set of deposit masses must serve all five epochs.**
Imposing that raises the floor from 2.6 per cent to 4.5 per cent with no change
to the deposits at all.

**3. But almost none of the remaining gap is information a shared law could
use.** This is the most important caution in the section. Take one galaxy's
deposits and fit them directly to a **different** galaxy's measured profile: the
error is 5.0 per cent, against 4.5 per cent for fitting the right galaxy and
14.2 per cent for the fitted model. **Using the wrong galaxy's halo costs only
13 per cent.** The set of deposits is flexible enough to draw almost any massive
galaxy's profile, so roughly seven eighths of the apparent gap between 14.2 per
cent and 4.5 per cent is that flexibility, not physics a seven-parameter law
could learn. (This test has no matching halo, but it does have full access to
the target's stellar profile — it measures how flexible the deposits are, not
what a predictor could learn.)

**4. "Mass can never be removed" does not, by itself, constrain the profile
shape.** This corrects an earlier claim in this note. The shape error is
computed after dividing each epoch's profile by its own value at 100 kpc, while
the never-decreasing requirement applies to the *absolute* profile — and any set
of shapes can be made non-decreasing simply by inflating the later epochs.
Optimised against the shape error alone, the bound is **0.1 per cent**. An
earlier version of this note reported 29 per cent, which was an artefact of the
particular construction used: it happened to match the total mass as well. **The
assumption bites only when the total mass and the shape are constrained
together.**

### 8.2 Where "mass can never be removed" does bite, and what it costs

It bites in the absolute profile, where it can be priced exactly. Because a
deposit can only contribute to epochs after it was laid down, the *change* in
the profile between two consecutive epochs can only be built from the deposits
laid down in between. Two things can go wrong, and they call for opposite fixes.
Both are priced the same way — as a typical error in the cumulative profile, as
a percentage of the mass inside that radius at the later epoch — with a
400-sample bootstrap over galaxies:

| interval | apertures where the simulation's mass FALLS | cost of not being able to follow that | cost of the deposits not spanning the rise |
|---|---|---|---|
| z = 2.0 to 1.5 | 32% | 3.7% [3.4, 3.8] | 3.2% [3.1, 3.3] |
| z = 1.5 to 1.0 | 33% | 4.7% [4.4, 5.0] | 2.4% [2.3, 2.6] |
| z = 1.0 to 0.7 | 38% | 2.4% [2.2, 2.6] | 1.9% [1.8, 2.0] |
| z = 0.7 to 0.4 | 37% | 2.7% [2.5, 2.9] | 1.8% [1.7, 1.8] |

**At roughly a third of all apertures, in every interval, the simulation's
enclosed stellar mass goes down with time, and no model of this kind can follow
it.** That costs more than the deposits' inability to draw the rise, at every
interval. (An earlier version of this note reported the opposite ordering; its
calculation minimised a differently-weighted quantity from the one it printed.)

### 8.3 The central deficit at redshift 2 is two different failures added together

Section 5.4 reported that the model is about 25 per cent too light at 2 kpc at
redshift 2. **That single number conflates two populations with different
physics.**

Split the 840 galaxies by whether their measured stellar mass inside 4.9 kpc
*fell* between redshift 2 and redshift 0.4 — a criterion fixed in advance.
Median fractional error at 2 kpc, redshift 2:

| galaxies whose centre... | n | the fitted model | the best any such model could do | fitted separately at each epoch |
|---|---|---|---|---|
| **later declined** | 443 (53%) | **−37%** | −22% | +4% |
| **did not decline** | 397 (47%) | **−11%** | −6% | +4% |

**Read this as follows.** In the 53 per cent of these massive ancestors whose
central stellar mass later *decreases*, the model puts 37 per cent too little
mass inside 2 kpc at redshift 2 — and even a perfect efficiency rule with these
deposits would still be 22 per cent too light, because it is not allowed to
remove the mass again later. In the other 47 per cent the model is only 11 per
cent too light, and a perfect efficiency rule would close half of that.

**So the population figure of 25 per cent is dominated by a failure mode that
adding a compact component cannot address.** Where the centre does not decline
the deficit is modest and is an efficiency problem, not a missing-component
problem. See `figures/stage34_galaxies_gompertz_log-E2-S2.png`, which shows two
galaxies of each kind with their profiles at all five epochs.

**A compact second deposit component is therefore deprioritised, not ruled out.**
It remains a reasonable hypothesis for the non-declining galaxies — whose growth
between epochs is in fact the harder of the two for the current deposits to draw
(4.7 per cent against 2.1 per cent over redshift 2 to 1.5).

### 8.4 The number itself is not fixed; only the trade-off is

An earlier version of this note claimed that two thirds of the redshift-2
central deficit was irreducible. **That claim is withdrawn.**

"Enclosed mass never falls" guarantees that a *conflict* exists between matching
the redshift-2 centre and matching the redshift-0.4 centre. It does not say
which epoch pays for it. Re-solving the bound of Section 8.1 — the one that
allows any profiles at all, provided the enclosed mass never falls — while
progressively weighting redshift 2 more heavily traces the whole trade-off:

| weight given to z = 2 | 0.2 | 1.0 | 2.0 | 5.0 | 20.0 |
|---|---|---|---|---|---|
| shape error | 3.3% | **3.0%** | 3.1% | 4.6% | 7.5% |
| error at 2 kpc, z = 2 | −23% | **−17%** | −13% | −6% | −1% |
| error at 2 kpc, z = 0.4 | +5% | **+9%** | +10% | +25% | +40% |

A model of this kind can place the redshift-2 central error anywhere on that
curve; driving it to zero costs a 40 per cent central *over*-prediction at
redshift 0.4 and more than doubles the total shape error. At the weighting we
actually fit, the best achievable is −17 per cent; **the fitted model sits at
−25 per cent, so about 8 percentage points are recoverable and the rest is the
price of the trade-off at this weighting.** See
`figures/stage34_pareto.png`, where the fitted model is marked and sits outside
the achievable frontier — worse at redshift 2 than any point on the curve, and
buying nothing at redshift 0.4 in exchange.

### 8.5 How much time resolution a shared law would need

This is the one clearly positive result of the exercise, and it points at what
to build next.

Instead of freeing every deposit, free only the **total mass deposited in each
of K broad time intervals**, keeping the model's own distribution within each
interval. K = 1 is a single overall rescaling per galaxy; larger K is a
progressively finer description of *when* the stars arrive.

| K, number of free time intervals | 1 | 2 | 3 | 5 | 8 | 12 | 20 | free |
|---|---|---|---|---|---|---|---|---|
| shape error, all epochs | 14.2% | 12.9% | 10.5% | 10.2% | 9.7% | 9.5% | 9.0% | 8.5% |
| shape error at z = 2 | 12.3% | 12.3% | **9.2%** | 8.9% | 9.0% | 9.6% | 9.5% | 9.2% |

(These are computed with the convenient approximate loss the free-weight solve
minimises rather than the exact one, so all of them are upper bounds on what is
achievable.)

**Two readings.** First, the K = 1 and K = 2 columns at redshift 2 are an
identity, not a result: every deposit laid down before redshift 2 falls in a
single interval there, so rescaling it cannot change a profile that has been
normalised at 100 kpc, and the value *must* equal the fitted model's. An earlier
version of this note read that identity as "no room for improvement at redshift
2", which was wrong.

Second, and substantively: **most of the available improvement arrives by three
to eight independent components in time.** That is a resolution a richer
dependence of the efficiency on redshift could plausibly express — it does not
need per-galaxy freedom. See
`figures/stage34_temporal_gompertz_log-E2-S2.png`. It must still be read against
the wrong-galaxy test of Section 8.1, because these are oracles fitted to the
stellar data.

### 8.6 Two checks on the bound itself

**Is the bound physically achievable?** Capping each deposit so that it can
never turn more than the universal baryon fraction of its accreted halo mass
into stars changes the bound by 0.03 percentage points. **It passes that
mass-budget check** — a necessary condition, though not a sufficient one:
smoothness in time, merger delivery and stellar mass loss are all still
unconstrained, and the solution activates a median of only **6 of about 71**
deposits, which is a very bursty history.

**Does it depend on the deposit shape?** Repeating everything with the
power-law-tailed Moffat deposit reproduces every row above to within
0.5 percentage points. None of this belongs to one profile shape.

### 8.7 A redshift-dependent deposit shape, tested and rejected

The extension previously agreed as the next thing to build was to let the
deposit shape parameter vary with redshift, so that early deposits are
intrinsically more concentrated. It was tested **at the level of the bound**,
where it cannot be rescued by refitting something else: the deposit shape
parameters were scanned on a grid while the per-galaxy freedom was held fixed,
so any improvement would be a statement about the deposits rather than about
degrees of freedom.

**The best value of the redshift dependence is zero**, at both freedom levels
tested, on an interior minimum. Re-measured on all 2397 galaxies and on the 840
followed here, the best non-zero value is **worse** by 0.02 and 0.31 percentage
points of shape error. The best size law available is worth 0.4 per cent.
See `figures/stage34_cz_gompertz_log-E2-S2.png`.

This is strong negative evidence against the proposed mechanism rather than a
proof of impossibility — the scan minimises the free-weight solve's own loss,
and a very flexible set of deposits can be insensitive to a change a rigid
shared law would still feel. **The extension was not built.**

---

## 9. What we have learned

### 9.1 About the galaxies and the model

1. **A three-line deposition law reproduces the radial profile well.** Seven
   parameters shared by 2397 galaxies, no per-galaxy freedom, and a shape error
   falling from 14.5 per cent at redshift 0.4 to 11.3 per cent at redshift 2 —
   8 to 28 per cent better than the previous generation at every epoch.

2. **Letting the halo-mass dependence of the efficiency change with redshift is
   the single most valuable term in the model.** Without it, the model's bias
   varies with halo mass by +0.024 to +0.041 dex per dex; with it, that
   collapses to +0.003 to +0.012 — an almost complete removal of the
   mass-dependent bias. Physically: *how* efficiency depends on halo mass must
   itself change with time. An earlier generation of this model needed an
   explicit transport term to achieve the same thing.

3. **Halo concentration is not competitive with that cross-term**, even after
   the units bug of Section 7.1 was fixed. It helps, but reaches only rank 13 of
   45.

4. **No peak in the efficiency's redshift dependence is detected.** The flexible
   curve built specifically to find one is the worst family tried, with
   essentially undetermined parameters, in exchange for a negligible improvement
   in the loss.

5. **The shape of an individual deposit barely matters.** Across the 45-model
   comparison, changing the deposit's profile shape moves the loss about 40 per
   cent less than changing the efficiency law does. What a deposit looks like
   matters less than how much mass it carries and where it is placed.

6. **Half of the model's worst defect belongs to the assumption, not the fit.**
   In the majority of these massive ancestors, the simulation's central stellar
   mass *falls* with time, which a deposition-only model cannot reproduce at
   all (Sections 8.2 and 8.3).

### 9.2 About what it still cannot do

7. **The central deficit at redshift 2 is two failures added together**
   (Section 8.3): 37 per cent too light for the 53 per cent whose centres later
   decline, and 11 per cent for the rest. A compact extra component addresses
   only the smaller half.

8. **The model degrades on complete, massive samples** — about 36 per cent worse
   than a plain statistical fit at redshift 0.7 and above, against 5 to 10 per
   cent on the sample as originally selected. This is a total-mass failure at
   high halo mass, distinct from the central deficit, and it is unexplained.

9. **It predicts a typical galaxy, not the range of galaxies.** The spread of
   predicted growth histories is half the simulation's. That is inherent to any
   single deterministic law and will eventually need a stochastic layer.

### 9.3 About two terms that turned out to be fitting the survey

10. **Making the deposit size depend on halo properties absorbs selection
    structure, not physics.** All three of its coefficients collapse to near
    zero on two independently constructed controlled samples — from 0.099 to
    0.003 and 0.020 for the mass term, −0.338 to −0.017 and −0.024 for
    concentration, −0.334 to +0.004 and +0.005 for formation time.

11. **And so, apparently, does the double-power-law efficiency.** Its whole
    advantage was that it removed the redshift-2 mass-dependent bias — and that
    bias is the survey (Section 6.3). **Its four extra parameters buy a fit to a
    selection artefact.** The case for the simple seven-parameter model is
    *stronger* after the control than before it, which was not expected.

### 9.4 About method

12. **A benchmark and the thing it benchmarks must be measured on the same
    objects.** One corrupt galaxy in the denominator and not the numerator
    overturned an experiment's headline (Section 7.2).

13. **An unexpectedly good result deserves the same audit as a bad one.** A
    benchmark degraded by outliers flatters whatever is compared against it.

14. **Rank by a multi-criterion panel, never by the loss.** Demonstrated twice:
    the loss-optimal models are precisely the degenerate ones.

15. **Check the completeness of a sample before naming a residual.** A delivery
    delay fitted to the redshift-2 bias would have absorbed a survey effect and
    acquired a physical name it had not earned.

16. **Decide what a fix invalidates from what the code *reads*, not from a list
    written by hand.** A hand-written list missed one path; a control asserted
    not to move caught it within three tries.

17. **Bound a framework before enriching it — and price the bound's own
    freedom.** A bound computed with per-object freedom always looks reachable.
    What makes it interpretable is the control: fit the same machinery to a
    *different* object. Here that reaches 5.0 per cent against the true bound's
    4.5 and the model's 14.2, so about seven eighths of the apparent headroom is
    flexibility, not information. Reporting the bound alone would have licensed
    exactly the extensions it was built to forestall.

18. **A bound must be optimised on the same objects and the same quantity it is
    reported on.** Three versions of that error appeared in one afternoon: a
    solve that used epochs its score excluded; a test weighted by one quantity
    and reported in another, which reversed its conclusion; and a convenient
    approximate loss quoted as if it were the loss we actually fit, worth a
    factor of two (8.5 per cent against the true 4.5).

19. **Check whether a constraint actually binds the quantity being reported.**
    "Enclosed mass never falls" is a true theorem that constrains the shape
    error almost not at all, because each epoch is renormalised (Section 8.1).

20. **A control can be an identity.** One free amplitude per epoch interval
    cannot change a redshift-2 profile normalised at 100 kpc, so that test *had*
    to equal the fitted model there. Before quoting a control, ask what it is
    able to change.

21. **A loss can be blind in a region and still look converged there.** At
    redshift 2 only 6.3 per cent of the stellar mass lies beyond 52 kpc, so an
    error the loss cannot resolve is a 30 per cent error in that annulus.

22. **The completeness trap recurs in every new view.** The outer-slope headline
    of Section 5.5 was more than half survey, and the identical correction had
    already been made for the inner profile. Apply the control the first time a
    diagnostic is plotted, not after it has produced a headline.

23. **Scale variables before handing them to an optimiser.** A solver given
    parameters of order `1e10` and gradients of order `1e-11` reports success at
    its starting point. This happened twice in this experiment and is invisible
    unless you check that the answer moved.

---

## 10. Where this stands, and what to do next

### 10.1 Honest summary

**What is solid.** A seven-parameter law, shared by all galaxies, using no
stellar information and no per-galaxy tuning, that predicts the radial profile
8 to 28 per cent better than the previous generation at every epoch, gets the
median stellar-mass growth right to 0.03 dex, and does not overfit. Its one
important term — letting the halo-mass dependence change with redshift —
survives every control applied to it.

**What is not.** It does not beat a plain statistical fit to the same halo
properties at predicting total stellar mass at any epoch, and on a
completeness-controlled sample of massive haloes it is about a third worse. It
predicts a typical galaxy rather than a population. Two extensions that appeared
to help turned out to be fitting the sample selection.

**What we now know about the limit.** Half of the redshift-2 central deficit
is a failure mode no deposition-only model can address — the simulation's
central mass falls with time in the majority of these galaxies. Of the rest,
most of the apparent headroom is the flexibility of the deposits rather than
physics a shared law could learn.

### 10.2 Next steps, in order

**1. Give the efficiency law a richer dependence on time.** This is the one
clearly positive lead (Section 8.5): most of the reachable improvement in the
profile shape arrives once the amount of stellar mass deposited can vary
independently over three to eight broad time intervals, which a richer
dependence of efficiency on redshift could express with a handful of extra
shared parameters. Concretely: replace the single power law in `ln(1+z)` with a
low-order polynomial or a two-knot spline, fit it, and judge it with the
existing panel. **Score it against the wrong-galaxy control, not against the
bound.**

**2. Repair the loss before touching the profile family.** The loss we fit
cannot see the outskirts at high redshift — at redshift 2 only 6.3 per cent of
the stellar mass lies beyond 52 kpc, so a sub-per-cent error on the cumulative
curve is a 30 per cent error in that annulus (Section 8.1 and 5.5) — and it is
nearly blind to the never-decreasing constraint once each epoch is renormalised.
A previous experiment already measured an objective based on surface density
with a logarithmic residual, which improves the compact-centre defect; it was
not adopted for unrelated reasons and deserves re-examination now that the
blindness is quantified.

**3. Explain why the model degrades on complete, massive samples.** A 36 per
cent excess scatter in total stellar mass at redshift 0.7 and above, on exactly
the galaxies we would want to deploy on, and not explained by anything in
Section 8. This is the strongest remaining argument against the model, and it is
a total-mass problem rather than a profile problem.

**4. Only then, and only if it earns it, a compact second component** — and only
on the galaxies whose centres do not decline (Section 8.3), following a nested
sequence: first map the central residual against halo properties available at
the epoch, with a shuffled control; then try a fixed compact fraction; then let
it depend on mass and redshift; and let it depend on formation history only if
that beats its own shuffled control.

**5. Test a halo-mass-dependent size evolution.** The size law's dependence on
redshift is currently the same for all halo masses. Making that exponent depend
on halo mass is a single extra parameter that targets the massive high-redshift
galaxies directly, and it has never been tested — the size-law extension that
was tested varied the size *normalisation* with halo properties, not the
redshift *exponent*.

### 10.3 Owed to the record

- **Genuine profile likelihoods for the seven parameters** — fix one, re-optimise
  the other six, repeat across a grid. Until this is done, the identifiability
  claim in Section 3.5 is incomplete.
- **A sensitivity test on the deposit cut-off radius** at one, two, three and
  five halo radii, reporting how the total mass and the fitted efficiency move.
- **Sensitivity of the Section 6 conclusions** to the 60-per-cent threshold, the
  mass-bin width and the plateau rule.
- **Hold redshift 2 out of the fit entirely** and predict it, which would convert
  it from a constraint into a genuine extrapolation. The catch: the 45-model
  comparison fitted the two endpoints, so a clean hold-out must re-select as
  well as re-fit.
- **A stochastic layer**, eventually, to reproduce the population's spread
  rather than its median.

---

## Appendix A: code names

This note avoids the internal labels used in the software. For anyone reading
the code, the correspondence is:

| in this note | in the code |
|---|---|
| efficiency: power law in mass and redshift | `E1` |
| ... plus the mass-by-redshift cross-term (**adopted**) | `E2` |
| ... plus dependence on halo concentration | `E3` |
| efficiency: free flexible curve in redshift | `E4` |
| efficiency: double power law in halo mass | `E5` |
| deposit size: fraction of the halo radius (**adopted**) | `S2` |
| deposit size: fraction of the halo scale radius | `S2a` |
| ... plus dependence on mass, concentration, formation time | `S3` |
| deposit shape: sigmoid in log radius (**adopted**) | `gompertz_log` |
| deposit shape: power-law-tailed | `moffat` |
| the adopted model | `gompertz_log-E2-S2` |
| profile-shape error / 15.8 per cent | `score_F` |
| total-mass error / the statistical benchmark | `score_A` |
| the statistical benchmark's scatter | `SIGMA_A` |
| the previous generation's shape loss (15.8 per cent) | `L_F_REF` |
| the completeness-controlled sample | the "mh-complete" sample |
| the 840 galaxies followed at all five epochs | the "cohort" |
| the best any model of this kind could do | the "ceiling" |
| fitting one galaxy's deposits to another galaxy's profile | the "permuted control" |
| concentration relative to the mean relation | `dlogc` |
| fraction of the halo's current age at which it had half its current mass | `f_form` |

## Appendix B: where things live

| item | file |
|---|---|
| the per-galaxy halo record (contains no stellar data by construction) | `experiments/exp54_unpinned_amplitude/halo.py` |
| the model, deposit shapes, truncation | `model.py`, `experiments/exp53_deposition_only/families.py` |
| the loss, the scores, the identifiability report | `fit.py`, `hongshao/objective.py` |
| the statistical benchmark | `scoreboard.py` |
| the 45-model comparison and its re-run | `stage32.py`, `stage32_rejudge.py` |
| the five-epoch fits | `stage33.py`, `stage33_refit.py`, `stage33_perepoch.py` |
| the selection control | `selection.py` |
| how good a model of this kind could be, and the controls | `stage34_ceiling.py` |
| the exact-loss bounds and the trade-off sweep | `stage34_objective.py` |
| the deposit-shape and size-law scans | `stage34_basis_scan.py` |
| figures | `qa_figures.py`, `stage34_figures.py` |
