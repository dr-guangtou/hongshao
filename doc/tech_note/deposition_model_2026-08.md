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
half-mass radius would have to extend to **9.6 Mpc** to contain 99% of
its mass. An earlier version of this model ended up with **18% of its
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
**+10 to +35%**. The model therefore works throughout in terms of the
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
is a factor of ten, so 0.1 dex is a 26% error and 0.2 dex is a factor
of 1.6.

For comparison we also fit a **plain statistical regression** — no physics, just
a fit of the same halo properties to the stellar mass, refitted separately at
each epoch. That is the benchmark: it is the best that can be extracted from the
same information without any model of how galaxies form. Its scatter is

| | z = 0.4 | z = 0.7 | z = 1.0 | z = 1.5 | z = 2.0 |
|---|---|---|---|---|---|
| statistical benchmark, dex | 0.104 | 0.112 | 0.120 | 0.128 | 0.161 |

Throughout this note, total-mass errors are given **both** in dex and as a ratio
to this benchmark, so that "1.30" always means "30% more scatter than a
straight statistical fit to the same halo properties".

### 3.3 The profile-shape error

For each galaxy and epoch we take the fractional error of the normalised
profile, `(model - truth)/truth`, at each of the 24 radii, average it in
quadrature across radii, then average over epochs and galaxies. The result is
**a typical percentage error in the shape of the cumulative profile**.

The previous generation of this model scored **15.8%** by this measure.
Throughout this note, shape errors are given **both** as a % and as a ratio
to that 15.8%, so "0.90" always means "10% better than the
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
all 2397 galaxies the shape error is **14.5, 14.5, 13.8, 12.6 and 11.3%** at
redshift 0.4, 0.7, 1.0, 1.5, 2.0 — that is 0.92, 0.92, 0.88, 0.80 and
0.72 times the previous generation's 15.8%, so the model is 8%
better at redshift 0.4 and 28% better at redshift 2. See
`figures/qa_cog_gompertz_log-E2-S2_full.png`.

The fitted values, for reproducibility only — Section 3.5 explains why they are
not yet quotable as measurements: efficiency `a0 = -2.549`, halo-mass slope
`a_M = -0.452`, redshift slope `a_z = 0.488`, cross-term `a_Mz = 0.233`;
size law `log10 f0 = -0.952`, redshift tilt `b = -0.735`; deposit shape
`c = 0.919`.

**The average growth in stellar mass is right.** Between redshift 2 and redshift
0.4 the model's median stellar mass inside 100 kpc grows to within **0.03 dex**
(7%) of the simulation's. The previous generation needed a hand-applied
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
mass slightly better than the model does at most epochs — and slightly worse at
redshift 2.**

| error in `log10 M*(<100 kpc)` | z = 0.4 | z = 0.7 | z = 1.0 | z = 1.5 | z = 2.0 |
|---|---|---|---|---|---|
| statistical benchmark, dex | 0.104 | 0.112 | 0.120 | 0.128 | 0.161 |
| model on all galaxies, dex | 0.116 | 0.122 | 0.128 | 0.138 | 0.174 |
| ratio to the benchmark | 1.11 | 1.09 | 1.06 | 1.08 | 1.08 |
| model on the completeness-controlled sample, dex | 0.116 | 0.118 | 0.124 | 0.133 | 0.152 |
| ratio to the benchmark | 1.11 | 1.06 | 1.03 | 1.04 | **0.94** |

The model is 6 to 11% worse than a regression on the sample as selected, and 3
to 11% worse on a completeness-controlled one — except at redshift 2, where it
is **6% better**. Restricting to complete, massive haloes makes the model
BETTER, not worse.

### 5.3.1 A correction: an earlier version of this section said the opposite

**This section previously reported the model as "about 36% worse at redshift
0.7 and above" on completeness-controlled samples and called that "the
strongest argument against the model as it stands". That was one corrupt
galaxy.** The ratios it quoted were 1.36 / 1.34 / 1.36 / 1.37.

Galaxy row 181's measured `M*(<100 kpc)` falls **3.76 dex**, from 10^11.72 at
redshift 0.4 to a flat ~10^8 at all four earlier epochs — a broken cross-match,
not a progenitor. The next largest drop anywhere in the sample is 2.04 dex.

The project had identified this galaxy, and had a rule to reject it
(`MAX_BACKWARD_DEX = 3.0`), and had already recomputed the statistical
benchmark without it. **The rule was never applied on the model's side of the
comparison.** It was written for the selection diagnostics; the model-fitting
code built its sample directly from the profile catalogue. So the model was
charged for a galaxy the benchmark was not, from the moment the fitting sample
grew from a 1199-galaxy subsample to all 2397.

**Why it hid where the sample was smallest, which is what made it look like
physics.** The completeness cut shrinks the sample from 2397 galaxies to 840.
One galaxy's fixed contribution to a mean square therefore grows as a FRACTION
as the sample shrinks — which is exactly the signature the earlier text read as
"the regression barely notices the restriction; the model degrades sharply".
Removing that one galaxy accounts for **17.8% of the total fitting loss** and
inflated the amplitude score by 24%.

The general lesson, recorded in `doc/lessons.md`: when a defect grows as a
sample shrinks, suspect a fixed contaminant before a physical trend, and check
by removing one object. And a data-quality rule must be applied where the data
is loaded, not where the figure is drawn.

**What survives.** The profile results are almost untouched — the shape error
moves from 13.803% to 13.787% — because the corruption was in the total mass,
not the radial distribution. Every conclusion in Sections 8 and 9 about profile
shape stands. What does not survive is this section's headline and its place in
Section 10's priority list.

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

The two lighter bins are uniformly about 12% too light everywhere, which
is a total-mass problem. **The heaviest bin is different: it is 29% too
light inside 3 kpc and essentially correct from 10 kpc outward.** That is a pure
central failure. It deepens steadily with redshift — the same bin is 13%
too light in the centre at redshift 0.4, 20% at redshift 1.0 and 29% at
redshift 2.

Galaxy-to-galaxy scatter is wide (the 16th-to-84th percentile spread at 2.8 kpc
is about 60%), but with 280 galaxies per bin the median is determined to
about 2%. See `figures/qa_cogmass_gompertz_log-E2-S2.png` and
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
   2.8 kpc at redshift 0.4, i.e. the model is 7.5% too light there. (An
   earlier generation of this model was 0.39 to 0.47 dex too light — a factor of
   two and a half.)
2. **It grows strongly with redshift**, reaching −0.133 dex — 26% too
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
fraction first reaches 60% of that maximum and stays there.

| | z = 0.4 | z = 0.7 | z = 1.0 | z = 1.5 | z = 2.0 |
|---|---|---|---|---|---|
| threshold, `log10 M200c` | > 13.00 | > 13.00 | > 13.00 | > 12.90 | > 12.80 |
| galaxies kept | 2397 | 1781 | 1436 | 1145 | 840 |

**This is a threshold, not completeness.** Above it we hold roughly 47%
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
enclosed mass genuinely falls between redshift 2 and redshift 0.4 in **50 to
58%** of these galaxies at about 5 kpc. Section 8.3 shows this is the
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
fit at four of five epochs" became "it is 5 to 13% worse at every
epoch".

**The damage was confined to the total-mass side.** That galaxy moved the
population *shape* error by **0.04%**. The reason is structural: the
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

Profile-shape error, as a typical percentage error of the normalised cumulative
profile, and as a ratio to the previous generation's 15.8%:

| what is allowed to vary | free numbers per galaxy | shape error | ratio |
|---|---|---|---|
| anything at all, as long as enclosed mass never falls with time | — | 0.1% | 0.009 |
| the same, but the total mass must also be right | — | 3.1% | 0.195 |
| every deposit's mass, fitted **separately at each epoch** | 5 x ~71 | 2.6% | ≤ 0.164 |
| **every deposit's mass, one set serving all five epochs** | **~71** | **4.4%** | **0.281** |
| *the same, fitted to a DIFFERENT galaxy's measured profile* | ~71 | **5.0%** | **0.319** |
| the mass in each of 8 broad time intervals | 8 | 9.7% | 0.614 |
| the mass in each of the 5 intervals between epochs | 5 | 12.1% | 0.762 |
| **the fitted model: 7 numbers shared by all galaxies** | **0** | **14.2%** | **0.898** |

The corresponding figure is `figures/stage34_ladder_gompertz_log-E2-S2.png`.

**Re-derived 2026-08-25, on the cleaned sample.** Every number in this section
was originally computed with the broken cross-match at row 181 still inside the
sample, and with model parameters from a fit that also contained it
(Section 5.3.1). Both were corrected and the whole stage re-run. **Every rung
moved by between 0.01 and 0.03 percentage points**, and at the precision quoted
here only two cells change at all: the free-deposit bound from 4.5% to 4.4%,
and the three-interval rung of Section 8.5 from 10.5% to 10.4%. No ordering,
ratio or conclusion changes.

That the effect is this small is itself worth stating, because it was not
obvious in advance: row 181's measured profile is essentially all inside the
innermost aperture, which is the shape a deposit basis reproduces worst, so it
could have inflated a representational ceiling rather than merely perturbing
it. It does not, because the ceiling is a MEAN over 839 galaxies and one
object moves a mean by at most its own excess divided by 839. The contrast with
Section 5.3.1 — where the same galaxy moved a headline by a factor of 1.3 — is
the difference between a mean over a large sample and a mean over a sample the
completeness cut had already shrunk.

Four things follow.

**1. The deposits themselves are not the problem.** Allowed to refit their
masses separately at each epoch, they reproduce every galaxy's measured profile
to within about 1 to 2% at every radius. Whatever is wrong, it is not
the profile shape, the deposit size law or the cut-off. A direct scan confirms
this: the fitted size law is already within 0.4% of the best any size
law could do here.

**2. What costs is that one set of deposit masses must serve all five epochs.**
Imposing that raises the floor from 2.6% to 4.4% with no change
to the deposits at all.

**3. But almost none of the remaining gap is information a shared law could
use.** This is the most important caution in the section. Take one galaxy's
deposits and fit them directly to a **different** galaxy's measured profile: the
error is 5.0%, against 4.4% for fitting the right galaxy and
14.2% for the fitted model. **Using the wrong galaxy's halo costs only
13%.** The set of deposits is flexible enough to draw almost any massive
galaxy's profile, so roughly seven eighths of the apparent gap between 14.2%
and 4.4% is that flexibility, not physics a seven-parameter law
could learn. (This test has no matching halo, but it does have full access to
the target's stellar profile — it measures how flexible the deposits are, not
what a predictor could learn.)

**4. "Mass can never be removed" does not, by itself, constrain the profile
shape.** This corrects an earlier claim in this note. The shape error is
computed after dividing each epoch's profile by its own value at 100 kpc, while
the never-decreasing requirement applies to the *absolute* profile — and any set
of shapes can be made non-decreasing simply by inflating the later epochs.
Optimised against the shape error alone, the bound is **0.1%**. An
earlier version of this note reported 29%, which was an artefact of the
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

Section 5.4 reported that the model is about 25% too light at 2 kpc at
redshift 2. **That single number conflates two populations with different
physics.**

Split the 840 galaxies by whether their measured stellar mass inside 4.9 kpc
*fell* between redshift 2 and redshift 0.4 — a criterion fixed in advance.
Median fractional error at 2 kpc, redshift 2:

| galaxies whose centre... | n | the fitted model | the best any such model could do | fitted separately at each epoch |
|---|---|---|---|---|
| **later declined** | 443 (53%) | **−37%** | −22% | +4% |
| **did not decline** | 397 (47%) | **−11%** | −6% | +4% |

**Read this as follows.** In the 53% of these massive ancestors whose
central stellar mass later *decreases*, the model puts 37% too little
mass inside 2 kpc at redshift 2 — and even a perfect efficiency rule with these
deposits would still be 22% too light, because it is not allowed to
remove the mass again later. In the other 47% the model is only 11% too light,
and a perfect efficiency rule would close half of that.

**So the population figure of 25% is dominated by a failure mode that
adding a compact component cannot address.** Where the centre does not decline
the deficit is modest and is an efficiency problem, not a missing-component
problem. See `figures/stage34_galaxies_gompertz_log-E2-S2.png`, which shows two
galaxies of each kind with their profiles at all five epochs.

**A compact second deposit component is therefore deprioritised, not ruled out.**
It remains a reasonable hypothesis for the non-declining galaxies — whose growth
between epochs is in fact the harder of the two for the current deposits to draw
(4.7% against 2.1% over redshift 2 to 1.5).

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
curve; driving it to zero costs a 40% central *over*-prediction at
redshift 0.4 and more than doubles the total shape error. At the weighting we
actually fit, the best achievable is −17%; **the fitted model sits at
−25%, so about 8 percentage points are recoverable and the rest is the
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
| shape error, all epochs | 14.2% | 12.9% | 10.4% | 10.2% | 9.7% | 9.4% | 9.0% | 8.5% |
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

Second: **most of the available improvement arrives by three to eight
independent components in time.** See
`figures/stage34_temporal_gompertz_log-E2-S2.png`. It must be read against the
wrong-galaxy test of Section 8.1, because these are oracles fitted to the
stellar data.

**A correction, dated 2026-08-24.** An earlier version of this section added
"that is a resolution a richer dependence of the efficiency on redshift could
plausibly express — it does not need per-galaxy freedom", and Section 10 made
it the leading next step. **That inference was wrong.** Every column of the
table above is solved separately for each galaxy: at K = 3 each object gets its
own three numbers. A law shared by all galaxies has no per-galaxy freedom at
all, so this table never bounded one. The tell was already in the table — the
K = 1 column equals the fitted model everywhere, not only at redshift 2,
because one free amplitude per galaxy cancels out of a profile normalised at
100 kpc.

Section 8.8 measures the quantity this section should have measured. The answer
is 0.2 percentage points, not the 3.3 implied here.

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
points of shape error. The best size law available is worth 0.4%.
See `figures/stage34_cz_gompertz_log-E2-S2.png`.

This is strong negative evidence against the proposed mechanism rather than a
proof of impossibility — the scan minimises the free-weight solve's own loss,
and a very flexible set of deposits can be insensitive to a change a rigid
shared law would still feel. **The extension was not built.**

---

### 8.8 What a richer dependence on time actually buys

Section 8.5's table was the reason to try this, and Section 8.5 now records why
it was the wrong reason. The experiment was run anyway, properly bounded, and
the answer is clear.

The efficiency law's dependence on redshift was replaced by five richer forms,
each keeping the current law whole and adding a correction on top of it: a
polynomial of degree two, three or four, and a set of two or three hinges at
redshift 1, 3 and 6. Each added correction is built to be **orthogonal to a
constant and to the current redshift term** over the range of redshifts the
deposits actually span, so its coefficients cannot simply trade against the
parameters already there — an earlier free-curve attempt failed for exactly
that reason, and failed as algebra rather than as physics. Setting the new
coefficients to zero reproduces the current model exactly, to the last bit, and
every fit was started from the current model's own best-fit values so that no
enrichment could come out worse unless the optimiser failed.

| | extra parameters | profile-shape error |
|---|---|---|
| the current law | — | **13.80%** |
| polynomial, degree 2 | 1 | 13.72% |
| two hinges | 2 | 13.70% |
| polynomial, degrees 2-4 | 3 | 13.64% |
| three hinges | 3 | 13.64% |
| polynomial, degrees 2-3 | 2 | **13.62%** |
| **a completely FREE shared curve** | 4 | 13.64% |

The last row is the point. Giving the time dependence a free shape, rather than
two coefficients, buys nothing further; the 0.02-point difference is smaller
than the spread between repeated optimiser runs in that many dimensions. **A
shared law's dependence on time is worth about 0.2 percentage points of the
13.80%, and two extra parameters already collect all of it.** The target
implied by Section 8.5 was 10-11%. Nothing came within 2.6 percentage points.

Two further findings make this a mechanism rather than a null.

**The extra freedom is spent where there is almost no mass.** Every enriched
form leaves the efficiency essentially unchanged below redshift 4 and then
turns sharply upward, ending 1.4 dex above the current law by redshift 15. Only
7% of the stellar mass this model makes is made above redshift 4, and 1% above
redshift 7 (`figures/stage35_time_law.png`, panels a and b). The new
coefficients are describing a part of history that carries no weight.

**The new coefficients are not determined, and they damage the ones that are.**
Displacing any of the seven parameters the model already has raises the loss by
between 0.14 and 39; displacing any of the eleven new coefficients raises it by
between 0.0035 and 0.05 — three to four orders of magnitude less. Resampling
the galaxies moves the existing parameters by 0.9% to 8% of their own values
and the new ones by 11% to 201%; three of the eleven cannot be pinned to within
their own size at all. Adding them also **doubles the uncertainty on the
model's own redshift slope**, from 5% to 9-12%
(`figures/stage35_identifiability.png`).

**Nothing was adopted.** The judging panel ranks the current seven-parameter
law *second* out of six, ahead of three of the five enrichments, because what
they take off the profile error they give back in how well the parameters are
pinned down. The halo-mass tilt does not reopen — it moves by at most 0.008 dex
per dex below redshift 1.5 and improves at redshift 2 — so nothing was traded
away either. The conclusion is simply that the efficiency law's dependence on
time is not where the remaining error lives.

### 8.9 Where the centre's evolution can and cannot be fixed

Section 8.8 closed the efficiency law's dependence on time. Two further levers
were tested against the same defect, and together they locate it precisely.

**The defect.** The model's error in the stellar mass inside 2 kpc runs from
+0.020 dex at redshift 0.4 to **−0.126 at redshift 2** — correct in the centre
today, 25% too light in the centre early. That 0.145 dex swing is the defect;
a uniform offset would simply be absorbed by the efficiency normalisation.

It is not an artefact of the sample. Measured at fixed halo mass the swing is
*larger* (0.175 to 0.192 dex), and on the same 750 galaxies followed at all
five epochs it is 0.131 dex (`figures/stage37_precheck.png`).

**The loss cannot see it, and repairing the loss does not help.** The loss
divides each profile by its value at 100 kpc, which pins it there: an error in
the 132–148 kpc shell must be **175,000 times larger** than one inside 2 kpc to
cost the same. Five objectives were fitted, including two that see the centre
far better. Every one shifted the central error by a nearly CONSTANT offset —
varying by 0.006 dex across five epochs — and left the swing at 0.145 to 0.152
dex. (The obvious candidate, comparing surface densities, turns out to be
structurally blind to the mass inside the innermost radius: differencing a
cumulative profile gives one fewer number than it has radii, and the missing
one is the centre. That aperture holds 32% of the stellar mass at redshift 2.)

**The size law can fix most of it — but only by emptying the outskirts.** Every
deposit is placed at `R50 = 10^(log_f0 + b log10(1+z)) * R200c`, one exponent
for all galaxies and all halo masses. Fitted under the production loss, four
nested enrichments of that exponent reduce the swing from 0.145 to at best
0.132 dex. Fitted to reduce the swing DIRECTLY, holding the overall loss within
5% of the incumbent's, a free shared size law reaches **0.061 dex — a 58%
reduction**. The defect is reachable; the loss simply never asks for it.

What reaching it costs is the finding:

| | z = 0.4 | z = 0.7 | z = 1.0 | z = 1.5 | z = 2.0 |
|---|---|---|---|---|---|
| central error, adopted model | +0.020 | −0.003 | −0.021 | −0.062 | −0.126 |
| central error, size law fitted for the swing | +0.012 | +0.012 | −0.001 | −0.025 | **−0.048** |
| beyond 52 kpc, adopted model | −0.001 | −0.007 | −0.006 | −0.002 | −0.008 |
| beyond 52 kpc, the same fit | −0.039 | +0.003 | +0.023 | −0.035 | **−0.112** |

Closing 0.078 dex of central deficit at redshift 2 costs **0.104 dex of
outskirt mass at the same epoch**, taking the model from 2% to 23% too light
beyond 52 kpc. The outer *slope* improves at the same time, so the deposits are
the right shape in the wrong place rather than the wrong shape. The solution is
in any case not adoptable: one parameter sits at its bound, the parameters are
undetermined, and the halo-mass tilt changes sign.

**The conclusion.** With one radial scale per deposit, the central mass at
redshift 2 and the outer mass at redshift 2 cannot both be right — the same
parameter controls both, and it can only trade one for the other. This is a
quantitative argument for a second deposit component with its own scale: an
early, compact channel that fills the centre while an extended channel keeps
the outskirts. It agrees, independently, with what a parallel experiment finds
from the measured profiles alone — an exponential core plus an extended Moffat
reproduces curves of growth as well as a three-component decomposition — with
the caveat that the parallel result constrains a single epoch and this model
serves five.

### 8.10 The limit is the model class, not its parameterisation

Section 8.9 ended by pointing at a second deposit component with its own radial
scale. It was built and fitted, and it settles the question — in the other
direction.

**The test.** Each deposit's profile became a mixture of two blobs at two
sizes, weighted by *when* the mass was laid down: an extended one at the size
the model already used, and a compact one — an exponential — at a fraction of
that radius. The compact fraction was allowed to be constant, or to rise toward
early times. Setting that fraction to zero reproduces the single-component
model exactly, which the code asserts at random parameter values.

**The result.** In every fit, the compact fraction went to zero. Fitted under
the production loss, all three versions returned exactly the incumbent's loss
to six decimal places and every diagnostic was numerically identical. Fitted
instead to reduce the central swing directly, they reached 0.089 and 0.094 dex
— worse than the single-scale size law's 0.061 — and got there by changing the
size law, not by using the second component. The fit was free to choose any
mixing weight between zero and one, and chose zero.

**Why, and this is the whole point.** Split these galaxies on whether their
*measured* mass inside 4.9 kpc actually falls between redshift 2 and today:

| error inside 2 kpc, dex | z = 0.4 | z = 0.7 | z = 1.0 | z = 1.5 | z = 2.0 | swing |
|---|---|---|---|---|---|---|
| all | +0.020 | −0.003 | −0.021 | −0.062 | −0.126 | 0.145 |
| **centre declined (42%)** | +0.068 | +0.007 | −0.042 | −0.127 | −0.202 | **0.270** |
| **centre did not (58%)** | −0.008 | −0.008 | −0.012 | −0.034 | −0.053 | **0.045** |

The galaxy-to-galaxy scatter is 0.16 to 0.20 dex in both groups. So for the
majority whose centres do not decline, the systematic trend with redshift is
about a quarter of the scatter and is nearly a constant offset — the kind the
efficiency normalisation absorbs. For the others, the simulation's own central
mass really does fall, by a median of 14% between redshift 2 and today.

**A model that only adds mass cannot follow that.** Written out: the predicted
mass inside any radius is a sum of non-negative deposits over a set that only
grows with time, and each deposit's profile is fixed once it is made.
Therefore the predicted mass inside every radius is non-decreasing in time, at
every parameter value. Not "happens not to decrease" — *cannot*.

So for 42% of these galaxies the change the data asks for has a sign the model
cannot produce, and no enrichment of the efficiency law, the objective, the
size law or the profile family can alter that. Sections 8.8, 8.9 and this one
describe four experiments that failed against the same defect, and they failed
because the target lies outside the set of things the model can express.

It also explains Section 8.9's trade. To reduce the swing measured over the
whole population, the size law had to make every early deposit compact —
including for the majority whose centres were already right — and that is
exactly why it emptied the outskirts.

**This is reported as the honest limit of this kind of model.** No attempt was
made to rescue the numbers by fitting only the galaxies whose centres do not
decline: that would quote a better score for a narrower claim, and the claim
worth making is the one about the model class.

## 9. What we have learned

### 9.1 About the galaxies and the model

1. **A three-line deposition law reproduces the radial profile well.** Seven
   parameters shared by 2397 galaxies, no per-galaxy freedom, and a shape error
   falling from 14.5% at redshift 0.4 to 11.3% at redshift 2 —
   8 to 28% better than the previous generation at every epoch.

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
   comparison, changing the deposit's profile shape moves the loss about 40%
less than changing the efficiency law does. What a deposit looks like
   matters less than how much mass it carries and where it is placed.

6. **The model's worst remaining defect belongs to the assumption, not to the
   fit — and this is now proved rather than suspected.** In 42% of these
   massive ancestors the simulation's central stellar mass *falls* with time,
   by a median of 14% between redshift 2 and today. A model that only adds mass
   predicts an enclosed mass that never decreases, at every radius and every
   parameter value, so that change has a sign it cannot produce. Four separate
   enrichments were built and fitted against this defect and all four failed
   (Sections 8.8, 8.9, 8.10).

### 9.2 About what it still cannot do

7. **The central deficit at redshift 2 is two failures added together**
   (Sections 8.3 and 8.10): the galaxies whose centres later decline carry a
   0.270 dex swing in the model's central error between redshift 2 and today,
   against 0.045 dex for the rest. A compact extra component was built to
   address the smaller half and **earned nothing** — its mixing weight went to
   zero in every fit, because the half it could have helped is already fitted
   to within a quarter of its own scatter.

8. **WITHDRAWN 2026-08-25 — it was one corrupt galaxy; see Section 5.3.1.**
   ~~The model degrades on complete, massive samples, about 36% worse than a
   plain statistical fit at redshift 0.7 and above.~~ Corrected, the model is
   3 to 11% worse than the regression and **6% better at redshift 2**, and
   restricting to complete massive haloes makes it better rather than worse.

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
    *different* object. Here that reaches 5.0% against the true bound's
    4.5 and the model's 14.2, so about seven eighths of the apparent headroom is
    flexibility, not information. Reporting the bound alone would have licensed
    exactly the extensions it was built to forestall.

18. **A bound must be optimised on the same objects and the same quantity it is
    reported on.** Three versions of that error appeared in one afternoon: a
    solve that used epochs its score excluded; a test weighted by one quantity
    and reported in another, which reversed its conclusion; and a convenient
    approximate loss quoted as if it were the loss we actually fit, worth a
    factor of two (8.5% against the true 4.5).

19. **Check whether a constraint actually binds the quantity being reported.**
    "Enclosed mass never falls" is a true theorem that constrains the shape
    error almost not at all, because each epoch is renormalised (Section 8.1).

20. **A control can be an identity.** One free amplitude per epoch interval
    cannot change a redshift-2 profile normalised at 100 kpc, so that test *had*
    to equal the fitted model there. Before quoting a control, ask what it is
    able to change.

21. **A loss can be blind in a region and still look converged there.** At
    redshift 2 only 6.3% of the stellar mass lies beyond 52 kpc, so an
    error the loss cannot resolve is a 30% error in that annulus.

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
8 to 28% better than the previous generation at every epoch, gets the
median stellar-mass growth right to 0.03 dex, and does not overfit. Its one
important term — letting the halo-mass dependence change with redshift —
survives every control applied to it.

**What is not.** It is 3 to 11% worse than a plain statistical fit to the same
halo properties at predicting total stellar mass — though 6% *better* at
redshift 2, and restricting to complete, massive haloes improves it rather than
degrading it (Section 5.3, corrected). It predicts a typical galaxy rather than
a population: the spread of predicted growth histories is half the
simulation's. Two extensions that appeared to help turned out to be fitting the
sample selection.

**What we now know about the limit, and it is the main result of this round.**
The model's remaining central defect is not a defect of its parameterisation.
Four independent enrichments were built and fitted against it — a richer
dependence of the efficiency on time, five repaired objectives, a richer
dependence of the deposit size on epoch, and a second deposit component with
its own radial scale — and all four failed against the same thing.

They failed because a model that only adds mass predicts an enclosed mass that
never decreases, at any radius and at any parameter value, while in **42% of
these galaxies the simulation's own central mass falls** — by a median of 14%
between redshift 2 and today. For those galaxies the change the data asks for
has a sign this kind of model cannot produce. For the other 58% the model is
already right to within a quarter of the galaxy-to-galaxy scatter
(Section 8.10).

Two of the four also produced their own clean results along the way: a shared
law's dependence on time is worth 0.2 percentage points of a 13.8% profile
error however it is shaped (Section 8.8), and with one radial scale per deposit
the central mass and the outer mass at redshift 2 cannot both be right
(Section 8.9).

### 10.2 Next steps, in order

**0. Give the efficiency law a richer dependence on time — DONE, and it is a
dead end** (Section 8.8, 2026-08-24). Five richer forms and a completely free
shared curve all land between 13.62% and 13.66%, against the current law's
13.80% and a target of 10-11%. The whole budget for a shared dependence on time
is 0.2 percentage points, and two extra parameters already collect it. The new
coefficients are three to four orders of magnitude flatter than the existing
ones and cannot be pinned down by resampling the galaxies. Nothing adopted.
The lead that pointed here rested on a table of per-galaxy bounds, which never
constrained a shared law; Section 8.5 now says so.

**1. Repair the loss before touching the profile family.** Promoted, and now
the leading item for a reason this exercise supplied: a shared law cannot be
shown to be inadequate by a loss that cannot see where it fails. The loss we fit
cannot see the outskirts at high redshift — at redshift 2 only 6.3% of
the stellar mass lies beyond 52 kpc, so an error below 1% on the cumulative
curve is a 30% error in that annulus (Section 8.1 and 5.5) — and it is
nearly blind to the never-decreasing constraint once each epoch is renormalised.
A previous experiment already measured an objective based on surface density
with a logarithmic residual, which improves the compact-centre defect; it was
not adopted for unrelated reasons and deserves re-examination now that the
blindness is quantified.

**2. WITHDRAWN — there is no such degradation.** ~~Explain why the model
degrades on complete, massive samples.~~ The 36% excess scatter was one broken
cross-match inside the fitting sample (Section 5.3.1). On the corrected sample
the model is 3 to 11% worse than a plain regression and 6% BETTER at redshift 2,
and the completeness restriction improves it rather than degrading it. **This
was the strongest remaining argument against the model, and it is gone.**

**3. TESTED AND REJECTED — a compact second component** (Section 8.10). Built,
fitted five ways, and the mixing weight went to zero every time; the production
loss returned the incumbent's value to six decimal places. It cannot help,
because a component that adds central mass cannot reproduce a central mass that
falls, and the galaxies whose centres do not fall are already fitted to within
a quarter of their own scatter.

**3b. What the limit points at instead.** Reproducing these galaxies' centres
requires a mechanism in which stars already deposited move outward — the
observed "expansion" of the central stellar profiles of massive galaxies, which
is usually attributed to supermassive black hole evolution and is therefore
only indirectly related to halo assembly. An empirical description would be
enough: let each deposit expand after it is laid down, with both its half-mass
radius and its truncation radius growing by the same factor as a function of
the time elapsed since deposition. Mass is conserved and never removed, nothing
becomes negative, no stellar information enters, and a zero expansion rate
reproduces the present model exactly — but the mass inside a fixed small radius
can then decrease, because old deposits spread outward faster than new ones
arrive. **This changes the model class**, and two things follow: the Stage 0
contract currently asserts that predicted profiles never decrease with time,
and that assertion would have to become conditional; and the profile of each
deposit would start depending on the epoch it is viewed at as well as the epoch
it was made, which costs several times more per evaluation.

**4. Test a halo-mass-dependent size evolution.** The size law's dependence on
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
- **Sensitivity of the Section 6 conclusions** to the 60% threshold, the
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
| profile-shape error / 15.8% | `score_F` |
| total-mass error / the statistical benchmark | `score_A` |
| the statistical benchmark's scatter | `SIGMA_A` |
| the previous generation's shape loss (15.8%) | `L_F_REF` |
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
