# exp55 — Analytic curve-of-growth representation

Status: COMPLETE for profile representation, the initial redshift-0.4
halo-mapping test, and Stages 1--4 of the halo-map refinement.

## Question

Can a continuous analytic curve-of-growth (CoG) function with approximately
three to five galaxy-level parameters reproduce the TNG300 massive-galaxy CoGs
nearly as accurately as their low-dimensional PCA representation, while keeping
its fitted parameters sufficiently identifiable to serve as targets of a later
DiffMAH-to-profile map?

This experiment deliberately separates two questions:

1. Can a function represent an individual galaxy profile?
2. Can halo assembly predict the function's parameters?

The first stage tests the representation and its identifiability. The second
stage tests the halo map in held-out galaxies. A profile parameterization that
is degenerate on the measured CoG is not a suitable halo-map target, regardless
of how small its residuals are.

## Scope

- Single epoch: redshift 0.4.
- Population: the existing low-redshift massive-central sample used by Exp50.
- Observable: the CoG-derived stellar masses on the standard 2.0--148.2 kpc
  grid. Direct projected aperture masses remain a cross-check only.
- The amplitude is a fitted profile parameter, `log_mstar_148`; no per-galaxy
  truth normalization is part of a future prediction.
- “Compact” and “extended” are phenomenological component labels. They are not
  identified as in-situ and ex-situ stellar populations by this experiment.

## Candidate representations

The measured CoG is compared with:

1. PCA reconstructions with three and five shape components;
2. the Exp50 five-coordinate monotone decoder;
3. six one-component specifications: Sérsic, Moffat,
   Gompertz-in-log-radius, Richards, a five-parameter radial-slope sigmoid, and
   that sigmoid with its asymptotic outer slope fixed to zero;
4. three restricted two-Sérsic mixtures with globally fixed component indices;
5. a compact Sérsic component plus a power-law-tailed extended Moffat
   component on a 3-by-4 grid of globally fixed shape indices.

The full-sample analytic search therefore contains 21 specifications: six
one-component functions, three fixed-index two-Sérsic functions, and twelve
fixed-index Sérsic-plus-Moffat pairs. This is a deliberately limited search,
not an exhaustive comparison of all possible compact-plus-envelope functions.

The fixed-index mixtures are intentional. Beginning with two freely varying
components would allow component fraction, scale, and shape index to exchange
roles before the data have shown those freedoms are identifiable.

## Selected analytic representation

The selected family is a compact exponential component (Sérsic index
`n = 1`) plus an extended Moffat component with `gamma = 1.25`:

```text
M*(<R) = M*,148 [f_compact F_Sersic(R; R_compact, n=1) / F_Sersic(148; ...)
                 + (1-f_compact) F_Moffat(R; R_extended, gamma=1.25)
                                   / F_Moffat(148; ...)].
```

Here “Moffat” names the mathematical function, not a point-spread-function
origin for the stellar envelope. Its projected surface density is a cored
power law,

```text
Sigma(R) proportional to [1 + (R/r_core)^2]^(-gamma),
```

and therefore approaches `Sigma proportional to R^(-2 gamma)` at large
radius. The selected `gamma = 1.25` supplies an outer `R^-2.5` surface-density
tail. The experiment supports the need for a slowly declining power-law-like
envelope more directly than it supports the Moffat family uniquely. Alternative
cored or truncated power-law envelopes still need a matched comparison.

It has four galaxy-level parameters:

- `log_mstar_148`: logarithmic stellar mass enclosed within 148.2 kpc;
- `compact_fraction_148`: fraction of that enclosed mass assigned to the
  compact component;
- `log_r_compact`: logarithmic half-mass scale in kpc of the formally infinite
  compact Sérsic family, before aperture renormalization;
- `log_r_extended`: the analogous logarithmic half-mass scale of the formally
  infinite extended Moffat family, constrained to exceed `log_r_compact`.

The radius parameters are family scale coordinates; they are not the radii
enclosing half of each component's truncated mass inside 148.2 kpc. The
component normalizations and `compact_fraction_148` are defined within the
measured aperture, so `log_mstar_148` is an explicit parameter rather than a
hidden normalization to the true stellar mass. A future halo model must predict it.
The labels “compact” and “extended” describe radial structure only; no
in-situ/ex-situ interpretation has been established.

## Fitting and identifiability tests

- All candidates are fitted in `log10 M*(<R)` over the complete measured grid.
- Multiple widely separated starts test whether indistinguishable profiles
  lead to different parameters.
- The residual Jacobian singular values measure local weak directions.
- Profile scans vary one parameter while re-optimizing every other parameter.
- Synthetic recovery tests whether profiles generated inside each family return
  their known parameters over the actual radial grid.
- Boundary incidence and consistency across galaxy mass and compactness are
  reported.

The experiment judges cumulative-profile error, central aperture error,
annular/surface-density error, outer slope, parameter stability, and local
conditioning. Residual quality alone cannot select a winner.

## Controls and references

- PCA-3 is the established low-dimensional representation reference.
- PCA-5 checks whether analytic residuals are competing with profile information
  that is genuinely present beyond three modes.
- The Exp50 decoder separates the value of readable profile anchors from the
  value of a continuous analytic function.
- Nested one- and two-component functions test whether a second component earns
  its freedom.

## Execution plan

1. Run mechanical self-checks and synthetic recovery.
2. Benchmark a mass-stratified small sample in under one minute.
3. Inspect profile and degeneracy figures before full execution.
4. Run the full redshift-0.4 sample serially, saving after each candidate.
5. Record the decision here before any halo-to-parameter mapping is attempted.

## Decision rule

The preferred representation is the simplest candidate that is statistically
competitive with the PCA and Exp50 reconstruction references under the measured
uncertainty of the comparison and that also has reproducible fitted parameters.

- If a one-component family passes, do not add a second component.
- If a restricted two-component family passes, retain only phenomenological
  compact/extended labels.
- If a mixture fits well but remains degenerate, do not map its raw components
  from halo properties; use stable combined-profile coordinates instead.
- If no analytic family passes both gates, use the best analytic base plus the
  minimum necessary constrained residual modes in a later experiment.

## Results

All numbers below use the 2,545 usable redshift-0.4 TNG300 progenitors and the
24 measured radii from 2.0 to 148.2 kpc.

### Representation accuracy

PCA remains the most accurate compression. Its median per-galaxy RMS error in
`log10 M*(<R)` is 0.00338 dex with three components and 0.00146 dex with five
components. The Exp50 five-coordinate monotone decoder has a median error of
0.00437 dex.

For the analytic candidates, the best cumulative-profile result comes from
the selected Sérsic-plus-Moffat mixture. Selecting the global component shape
indices on the training galaxies of each of five folds gives a held-fold
median cumulative-profile RMS error of 0.00441 dex. The paired median
difference from the Exp50 five-coordinate decoder is -0.000061 dex, with a
16th--84th percentile bootstrap interval of -0.000113 to +0.000012 dex. The
two representations are therefore indistinguishable at useful precision in
this test; the analytic mixture earns its extra value by defining a continuous
monotone profile.

The analytic mixture is not as accurate as PCA-3: its paired median
cumulative-profile RMS error is 0.000975 dex larger. It also does not reproduce
the annular density profile as accurately as the five-coordinate decoder: its
median density RMS error is 0.0653 dex, 0.00719 dex larger in the paired
comparison. A smooth analytic curve cannot follow every radial fluctuation
that a sampled-coordinate decoder or PCA basis can retain.

The initially promising five-parameter radial-sigmoid family reaches a median
cumulative-profile RMS error of 0.00556 dex, but 89.6% of galaxies place at
least one parameter within 1% of a bound and its median scaled-Jacobian
smallest-to-largest singular-value ratio is only `10^-3.42`. The selected
mixture is both more accurate and substantially better conditioned.

### Why the mixed tail matters

The best restricted two-Sérsic model uses indices `n = 2` and `n = 1`. It is
well conditioned, with a median cumulative-profile RMS error of 0.00727 dex,
but its median annular-density RMS error is 0.145 dex because both components
have exponentially declining outer tails. Replacing the extended Sérsic with
a Moffat component supplies a power-law tail and lowers these errors to 0.00437
and 0.0660 dex, respectively, for the full-sample best global shape pair.

A fold-clean grid search over compact Sérsic index `n = 1, 2, 4` and extended
Moffat index `gamma = 1.1, 1.25, 1.4, 1.7` selects `n = 1`, `gamma = 1.25` in
four of five held-out folds and `n = 1`, `gamma = 1.4` in the fifth. This is
evidence for the functional tail shape, not a measurement of a physical
stellar component.

### Parameter identifiability

For the selected full-sample shape pair, only 2.55% of galaxies put any fitted
parameter within 1% of its allowed bound. The median scaled-Jacobian
smallest-to-largest singular-value ratio is `10^-1.70`, far healthier than the
radial sigmoid. Exact synthetic recovery returns the four known parameters to
numerical precision for 200 representative galaxies.

Refitting those galaxies after removing alternating radii changes the median
parameters by 0.00108 dex in `log_mstar_148`, 0.00389 in compact fraction,
0.00704 dex in compact radius, and 0.00590 dex in extended radius. The 95th
percentile changes are 0.00263 dex, 0.0176, 0.0200 dex, and 0.0272 dex. One
poorly conditioned galaxy has a 0.77-dex compact-radius change, so the local
condition diagnostic must accompany any downstream parameter catalogue.

The fitted population has median compact fraction 0.257, median compact radius
2.47 kpc, and median extended radius 35.7 kpc. The respective 16th--84th
percentile ranges are 0.172--0.380, 1.82--3.42 kpc, and 19.0--57.8 kpc. The
median extended-to-compact radius ratio is 14.0, with a 16th--84th percentile
range of 9.89--20.9. Only 2.55% of galaxies place any optimizer coordinate
within 1% of a bound; almost all of these cases make the two radii nearly equal.

Across galaxies, the strongest Spearman correlations among the four physical
coordinates are `rho = +0.594` between enclosed stellar mass and extended
radius, `rho = +0.433` between compact and extended radius, and
`rho = +0.334` between compact fraction and extended radius. Compact fraction
is nearly uncorrelated with enclosed mass (`rho = -0.059`) and compact radius
(`rho = +0.059`). These are population correlations between best-fit values;
they are not posterior correlations within one galaxy. The local Jacobian,
profiled scans with all other parameters re-optimized, and radial jackknife are
the relevant within-galaxy stability checks. These values summarize the
analytic decomposition; they do not license an in-situ/ex-situ interpretation.

### What is degenerate about the sigmoid candidates

The five-parameter radial-sigmoid candidate is a single CoG, not a two-component
mixture. It defines the local cumulative logarithmic slope as

```text
d ln M(<R) / d ln R = beta_out
                       + delta_beta / [1 + exp((ln R - ln Rc) / width)].
```

Thus `beta_out` is the asymptotic outer cumulative slope, `delta_beta` is the
extra inner slope, `Rc` is the transition radius, and `width` is its width in
log radius. On the finite 2.0--148.2 kpc measured interval, many galaxies do not
show both asymptotes. Moving the transition outside the well-constrained range
while compensating with `delta_beta` and `width` can then preserve nearly the
same integrated CoG.

The fitted symptom is quantitative: 49.0% of galaxies put `beta_out` at its
zero lower boundary, 34.4% put `delta_beta` at its upper boundary, and 26.0%
put `Rc` at its 0.3 kpc lower boundary. The population correlations also show
the trade, including `rho = -0.645` between `delta_beta` and `Rc` and
`rho = -0.499` between `Rc` and `width`. Fixing `beta_out = 0` reduces one
freedom but leaves 64.0% of galaxies at some boundary and a median scaled
Jacobian singular-value ratio of only `10^-3.10`.

This does not condemn every monotone sigmoid-like CoG. The four-parameter
Richards function is also poorly conditioned (`10^-3.34`; 55.3% at a bound),
whereas the more restricted three-parameter Gompertz-in-log-radius function is
much healthier (`10^-1.61`; 7.50% at a bound). The failure mode is too many
independent controls of the asymptotes and transition over a finite radial
interval, not sigmoid curvature by itself.

## Decision and limitations

The selected four-parameter Sérsic-plus-Moffat CoG passes the present
representation and typical-identifiability gates. It is the preferred target
for a separate, null-controlled DiffMAH-to-profile experiment. The original
hope for only three galaxy-level shape-and-amplitude parameters was too
restrictive, while five freely varying parameters were unnecessarily
degenerate.

This conclusion is limited to noiseless TNG300 CoGs at redshift 0.4 for
progenitors of the low-redshift massive-central selection. Every galaxy was
fitted to its own complete profile, so this experiment measures representation
capacity, not prediction from a halo. The fits do not include observational
radial covariance, PSF effects, sky uncertainty, or censored outer profiles.
The component shape grid is deliberately small. Before any physical reading,
the parameters must survive observational degradation, other epochs, and
comparison with independently tagged in-situ/ex-situ stellar populations.

## Halo-mapping stage

Map the four selected parameters from epoch-local DiffMAH and concurrent halo
properties using the same held-out folds as Exp50--Exp51. Compare against
final-mass-only and mass-conditioned shuffled-MAH controls, predict the four
parameters jointly with their residual covariance, reconstruct full profiles,
and judge the result on held-out cumulative and density profiles. Do not add
redshift evolution or free component indices until that single-epoch halo map
shows that the parameters are predictable.

The predeclared implementation is:

1. use the fixed global `n = 1`, `gamma = 1.25` representation and its four
   per-galaxy optimizer coordinates as the joint target;
2. predict their conditional distribution from the portable z=0.4 feature
   vector `[DiffMAH(4), c_200c]` in five held-out folds;
3. compare a linear mean with the established sparse degree-2 mean, final halo
   mass alone, DiffMAH without concentration, mass-conditioned shuffled MAH
   shape, and mass-conditioned shuffled concentration;
4. reconstruct every held-out mean and correlated parameter draw into a full
   CoG, then score the profile rather than only the intermediate parameters;
5. report parameter variance explained, cumulative- and density-profile error,
   calibrated profile draws, and the complete standard QA battery, including
   best and worst individual galaxies and population planes.

The assembly claim is gated on the correctly paired full model outperforming
the shuffled-MAH control. The analytic target is useful only if its reconstructed
held-out profiles remain competitive with the Exp50 readable-coordinate map;
good parameter scores alone are insufficient.

### Halo-mapping results

The halo map uses 2,539 galaxies with finite analytic-profile fits, DiffMAH
parameters, and redshift-0.4 concentration. Every number is evaluated on the
held-out member of the same five deterministic folds used by Exp50 and Exp51.
The selected model predicts the four optimizer coordinates jointly from the
four DiffMAH parameters and `c_200c`, using a degree-2 mean and a full,
input-dependent residual covariance. Thirty-two correlated parameter draws per
halo propagate the remaining scatter into full CoGs.

The conditional-mean profiles have a median per-galaxy RMS error of 0.08519 dex
in `log10 M*(<R)` relative to the TNG CoGs. On the identical galaxies, this is
0.000058 dex lower than the Exp50 readable-coordinate map, and the analytic map
has the lower error for 50.2% of galaxies: their mean-profile accuracy is
effectively tied. The analytic map's mean profile CRPS is 0.06421 dex, 0.00094
dex worse than Exp50's 0.06327 dex, and its median density-profile RMS error is
0.00371 dex worse. The continuous analytic profile therefore retains the
predictive information of Exp50 to useful precision, but it is not a more
accurate halo map.

The null controls establish a real assembly signal under this model. Relative
to final halo mass alone, the complete model lowers mean profile CRPS by 25.84%.
It lowers CRPS by 13.20% relative to a control that shuffles the three DiffMAH
shape parameters among galaxies of similar final halo mass, and by 4.44%
relative to the analogous shuffled-concentration control. The complete model
also improves by 3.84% over a linear mean using the same five halo inputs. In a
paired comparison, using the real rather than shuffled MAH shape lowers the
median galaxy's cumulative-profile RMS error by 0.01077 dex; the bootstrap
16th--84th percentile interval is 0.00992--0.01203 dex, and the real pairing is
better for 61.6% of galaxies. Thus halo assembly and concentration contain
profile information beyond final halo mass, although most individual galaxies
still retain large stochastic scatter.

The four analytic coordinates are not equally predictable. The held-out
variance explained is 86.5% for stellar mass within 148.2 kpc and 33.7% for the
extended-component radius, but only 6.0% for compact fraction and 6.8% for
compact-component radius. This does not invalidate the reconstructed profile:
different combinations of the weakly predicted coordinates can describe
similar profile variations, and their joint residual distribution carries
information that their conditional means do not. It does mean that the fitted
compact fraction and compact radius should not be treated as deterministic halo
outcomes or assigned an in-situ/ex-situ interpretation.

The standard QA exposes the difference between a useful probabilistic profile
prediction and a complete population model. For the conditional mean, the
median bias is +2.1% for `M*(<10 kpc)`, +0.6% for `M*(<30 kpc)`, and -0.5% for
`M*(<100 kpc)`; the corresponding per-galaxy scatters are 0.111, 0.101, and
0.097 dex. Differential masses are harder: the 50--100 kpc annulus has 0.166
dex scatter, while mass beyond 100 kpc has +15.8% median bias and 0.188 dex
scatter. These are halo-prediction errors, roughly twenty times the analytic
representation floor, not failures of the continuous profile formula alone.

The correlated draws give 68.3% pointwise coverage for a nominal 68% interval
and 87.4% for a nominal 90% interval. Pointwise coverage is not sufficient,
however. The first analysis incorrectly evaluated a generated population at
each TNG counterpart's true `Re`, obtaining 0.213 dex scatter in the
`M*(<2 R_e)` versus `M*(2--4 R_e)` plane. Measuring every draw at its own `Re`
gives the correct generative value, 0.188 dex, compared with 0.072 dex in TNG.
The qualitative conclusion survives but the original number and metric were
wrong. The conditional mean has the opposite failure in the fixed-kpc planes:
it is too narrow because it omits residual diversity. The Gaussian residual
model captures marginal uncertainty better than all derived joint population
structure.

### Decision after the halo map

The fixed-shape analytic CoG is a viable continuous target for a direct
DiffMAH-to-profile model at redshift 0.4. It reproduces held-out CoGs as well as
the Exp50 sampled-coordinate decoder and passes both assembly and concentration
shuffle controls. It has not demonstrated that its individual compact and
extended coordinates are physical halo variables, nor has its stochastic
population model passed all joint-distribution tests.

The next step should therefore refine the residual representation before
adding redshift evolution or freeing the global component shapes. A practical
test is to rotate the four fitted coordinates into data-constrained residual
modes, predict only the halo-dependent modes, and model any nearly halo-
independent modes as explicitly stochastic. The revised draws must be judged
on the standard fixed-kpc and size-scaled population planes, not only on
coordinate R2 or pointwise coverage. If that improves joint calibration without
degrading the held-out CoGs, the same fixed analytic family can then be tested
at the other Exp51 epochs.

## Halo-map refinement: predeclared Stages 1--4

The first halo map passes the assembly controls but leaves coherent radial
residuals with halo mass and a miscalibrated stochastic population. The next
work block changes neither the fixed `n = 1`, `gamma = 1.25` analytic profile
family nor the redshift scope. It addresses the map in four gated stages.

### Stage 1 — locate the radial systematics

Decompose the median profile residual in three equal-number `log M_peak` bins
into (a) analytic-representation error and (b) halo-mapping error. Linearize the
analytic decoder at each galaxy's fitted coordinates and project the mapping
error onto the four parameter derivatives. Report the non-linear closure error
so that a visually appealing decomposition is not mistaken for an exact one.

Decision: change the profile family only if the representation error is
comparable to the mapping error. Otherwise retain the family and target the
responsible parameter combinations in the halo map.

### Stage 2 — improve the conditional mean conservatively

Compare the existing sparse degree-2 basis with a fixed small set of analytic
extensions: the two missing final-mass interactions
`log M_peak × log t_c` and `log M_peak × c_200c`, a complete degree-2 basis,
and a complete degree-2 basis plus a cubic final-mass term. Use training-fold
standardization and the existing five held-out folds. Rank candidates by
profile CRPS, median cumulative- and density-profile RMS, and the maximum
absolute median shape residual across the three halo-mass bins.

Decision: accept extra terms only if they improve the reconstructed held-out
profiles and reduce the mass-binned radial structures. Parameter-coordinate
loss alone is not an acceptance criterion. Retain the final-mass-only and
mass-conditioned shuffled-MAH/concentration controls for the selected model.
Adopt a more complex mean only if it lowers profile CRPS by at least 0.5%,
reduces the worst mass-bin median shape residual by at least 0.5 percentage
point, and does not worsen median density RMS by more than 0.001 dex. If several
candidates pass, select the one with the fewest terms within 0.25% of the best
profile CRPS.

### Stage 3 — organize residuals by their profile effect

Construct a global parameter-space metric from the analytic decoder Jacobians,
then rotate held-out coordinate residuals into orthogonal modes ordered by
their contribution to CoG variance. Report each mode's profile variance,
held-out halo predictability, correlations with the halo inputs, and decoded
radial signature. This is a diagnostic reparameterization: because every target
uses the same linear design, an invertible rotation is not allowed to claim an
artificial improvement in the conditional mean.

Decision: retain modes with distinct radial effects as stochastic coordinates;
only modes with reproducible held-out halo information should receive a more
complex deterministic mean.

### Stage 4 — repair the population draws

Compare the incumbent heteroscedastic Gaussian with fixed residual correlation
against simple, measurable alternatives: residual modes with diagonal
heteroscedastic scatter, mass-binned mode correlations, and empirical
mass-neighbour residual resampling. No flexible density estimator is introduced
until these controls establish what fails. Evaluate complete population draws
in the standard fixed-kpc planes, the size-scaled plane, mass--R50/R80/R90,
pointwise coverage, and per-galaxy profile CRPS.

For the generated population, measure each draw's size-scaled apertures using
that draw's own predicted `Re`. The standard paired QA continues to use the TNG
`Re` for both truth and conditional mean because it answers a different
question: profile error at a fixed known physical scale. A generated galaxy
cannot use its TNG counterpart's true size.

If ordinary neighbour resampling undercovers, run one diagnostic follow-up in
which every residual in the training library is itself generated by an inner
held-out prediction. This tests the specific possibility that residuals from
the model's own training galaxies are artificially narrow; it does not use the
outer validation galaxies to tune a scale factor.

If empirical residuals reproduce the population geometry but miss marginal
coverage, permit one nested calibration: choose a single residual inflation
from `[1.00, 1.08, 1.16, 1.24, 1.32]` using only inner held-out profile
coverage inside each outer training fold. Freeze that scale before evaluating
the outer fold. This remains a one-parameter calibration of an empirical
distribution, not a new flexible likelihood.

Decision: the selected stochastic model must improve the size-scaled plane
without destroying the fixed-kpc planes or marginal coverage. If no candidate
does so, record the failure and stop before multi-epoch extension. Quantitatively,
require a lower size-scaled-plane energy distance, no more than a 10% increase
in the worse of the two fixed-kpc energy distances, nominal 68% pointwise
coverage within three percentage points, and profile CRPS no more than 1% worse
than the incumbent.

## Results of Stages 1--4

All results use 2,539 galaxies and the same five outer folds as the initial halo
map. Profile CRPS and interval coverage use 32 profile draws per galaxy. The
more expensive population-plane and size statistics are averages over four
complete population draws. Model selection and residual calibration use
training or inner-validation galaxies only; the quoted results are from the
untouched outer folds.

### Stage 1 result — the halo map dominates the coherent residual

The median RMS difference between the halo-predicted analytic profile and the
best analytic fit to the same galaxy is 0.08494 dex. The median analytic
representation error relative to TNG is only 0.00437 dex. Thus the halo map,
not the fixed Sérsic-plus-cored-power-law family, dominates the present radial
systematic by a factor of about 19 in RMS.

The amplitude-removed median residual reaches +3.73% near 8.1 kpc in the lowest
halo-mass tercile, -3.40% near 2.8 kpc in the middle tercile, and +5.45% at
2 kpc in the highest tercile. The Jacobian decomposition explains the signs.
At low halo mass, an underpredicted extended-to-compact radius coordinate adds
too much intermediate-radius mass and is partly cancelled by compact-fraction
and compact-radius errors. At high halo mass, excess compact fraction and a
slightly over-concentrated compact scale raise the centre, while the extended
radius combination suppresses the 10--30 kpc region. The median non-linear
closure RMS is 0.00886 dex, so this derivative decomposition is informative but
not exact.

### Stage 2 result — more polynomial terms do not solve the mean bias

No added mean basis passes the predeclared gate. The incumbent sparse degree-2
mean has profile CRPS 0.06429 dex, median density RMS 0.13287 dex, and a 5.45%
worst mass-bin shape residual. Adding the two missing mass interactions makes
CRPS slightly worse at 0.06432 dex and raises the worst residual to 5.61%.
The complete degree-2 basis improves CRPS by only 0.45%, below the 0.5% gate,
while worsening the radial residual to 6.49%. Adding a cubic mass term reduces
the worst residual to 4.98% but improves CRPS by only 0.38% and the residual by
only 0.47 percentage point, both below their gates.

The sparse degree-2 mean is therefore retained. Real MAH shape still lowers
CRPS by 13.0% relative to its mass-conditioned shuffle, real concentration
lowers it by 4.3% relative to its shuffle, and the complete halo vector lowers
it by 25.8% relative to final halo mass alone. The coherent radial bias is not
evidence that the assembly signal disappeared; it is evidence that adding
generic polynomial flexibility is the wrong correction.

### Stage 3 result — four profile-effect modes separate distinct roles

The profile-metric rotation orders residual-coordinate combinations by their
actual CoG effect. Modes 1--4 contain 62.5%, 24.4%, 10.2%, and 2.9% of the
residual profile variance. Their held-out absolute-coordinate R2 values are
0.776, 0.115, 0.765, and 0.155. Modes 1 and 3 therefore contain most of the
halo-predictable structure, while modes 2 and 4 are predominantly stochastic.

Mode 1 changes the whole profile with a broad maximum near 10 kpc; mode 2
transfers mass sharply from the centre to intermediate and outer radii; mode 3
adds mass near 10 kpc while removing it in the outskirts; mode 4 is a weak
central/intermediate adjustment. After fitting the existing mean, all four
mode residuals have Pearson correlations below 0.005 with every original halo
input. This is expected for the fitted linear directions and shows that a
simple residual-linear correction cannot remove the remaining structure.

As predeclared, the invertible rotation is not credited with a better mean.
Its value is diagnostic: it identifies which combinations should be modeled as
stochastic and supplies interpretable radial signatures for future targeted
features.

### Stage 4 result — nested empirical residuals improve the generator

The key procedural correction is to measure generated size-scaled apertures at
each generated galaxy's own `Re`. Under this definition, the raw Gaussian has
0.188 dex scatter in `M*(<2Re)` versus `M*(2--4Re)`, not the retired 0.213 dex
paired-scale value; TNG has 0.072 dex.

Ordinary nearest-neighbour residual resampling substantially improves all three
population planes but covers only 62.7% of profile points with its nominal 68%
interval. Rebuilding the residual library from inner held-out fits produces the
same undercoverage, showing that ordinary in-sample residual shrinkage is not
the cause. A single residual inflation was therefore selected independently
inside every outer training fold. Four folds select 1.16 and one selects 1.08,
so the calibration is stable rather than driven by one split.

The nested-calibrated neighbour model passes every predeclared gate:

- Nominal 68% pointwise coverage is 68.4%, compared with 67.6% for the raw
  Gaussian; nominal 90% coverage is 88.4%.
- Mean profile CRPS improves from 0.06429 to 0.06368 dex, a 0.95% reduction.
- In the `<30 kpc` versus `30--50 kpc` plane, generated scatter is 0.178 dex
  versus 0.172 dex in TNG; the energy-distance ratio to the TNG sampling floor
  improves from 1.75 to 0.71.
- In the `<30 kpc` versus `50--100 kpc` plane, generated scatter is 0.197 dex
  versus 0.206 dex in TNG; the energy ratio changes from 1.19 to 1.27, remaining
  inside the allowed fixed-kpc gate while its centered distribution improves.
- In the self-consistent `<2Re` versus `2--4Re` plane, generated scatter is
  0.084 dex versus 0.072 dex in TNG; the energy ratio improves from 1.73 to
  1.38.

The remaining population limitation is the far-outer size distribution. The
R50 and R80 energy-distance ratios are 1.28 and 1.68 times the TNG sampling
floor, while R90 remains 3.63 times the floor. The generated R90 scatter is
0.099 dex, close to TNG's 0.104 dex, but its mean relation is too shallow and
its median radius is high by 0.024 dex. This agrees with the conditional mean's
outer-density and mass-beyond-100-kpc defects: the model now generates realistic
overall diversity but has not learned the outermost radial dependence fully.

### Decision after Stages 1--4

Retain the sparse analytic mean and replace the raw-coordinate Gaussian draws
with nested-calibrated nearest-neighbour residual draws for the experimental
redshift-0.4 generator. The calibration procedure is fold-local and portable:
it stores the training residual catalogue, standardized halo features, the
neighbour count, and one inner-selected scalar inflation.

Do not add generic polynomial terms, free the global component indices, or
extend to other epochs yet. The next focused question is why the halo map has
opposite central/intermediate residual signatures at low and high halo mass,
and why R90 retains a distribution offset. Candidate explanations should be
tested through targeted halo information or a profile-relevant mean correction,
not through unrestricted algebraic flexibility.

## Files and commands

The implementation lives entirely in this folder. Figures and numerical
outputs are regenerable and gitignored.

The complete standard QA battery, explanations of each panel, and a detailed
account of why small cumulative errors produce larger outer-density errors are
in [`FIGURES.md`](FIGURES.md).

```bash
PYTHONPATH=. uv run python experiments/exp55_analytic_cog/run.py demo
EXP55_NMAX=60 PYTHONPATH=. uv run python \
  experiments/exp55_analytic_cog/run.py fit
PYTHONPATH=. uv run python experiments/exp55_analytic_cog/run.py fit
PYTHONPATH=. uv run python experiments/exp55_analytic_cog/mixed.py demo
PYTHONPATH=. uv run python experiments/exp55_analytic_cog/mixed.py run
PYTHONPATH=. uv run python experiments/exp55_analytic_cog/diagnostics.py
PYTHONPATH=. uv run python experiments/exp55_analytic_cog/qa_figures.py
EXP55_MAP_NMAX=60 EXP55_MAP_N_DRAW=4 PYTHONPATH=. uv run python \
  experiments/exp55_analytic_cog/halo_map.py demo
PYTHONPATH=. uv run python experiments/exp55_analytic_cog/halo_map.py fit
EXP55_REFINE_NMAX=90 EXP55_REFINE_N_DRAW=4 PYTHONPATH=. uv run python \
  experiments/exp55_analytic_cog/refine_halo_map.py demo
PYTHONPATH=. uv run python experiments/exp55_analytic_cog/refine_halo_map.py run
PYTHONPATH=. uv run python \
  experiments/exp55_analytic_cog/refine_halo_map.py figures
```
