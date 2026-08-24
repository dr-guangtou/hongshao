# exp55 — Analytic curve-of-growth representation

Status: COMPLETE for the profile-representation and identifiability question.

## Question

Can a continuous analytic curve-of-growth (CoG) function with approximately
three to five galaxy-level parameters reproduce the TNG300 massive-galaxy CoGs
nearly as accurately as their low-dimensional PCA representation, while keeping
its fitted parameters sufficiently identifiable to serve as targets of a later
DiffMAH-to-profile map?

This experiment deliberately separates two questions:

1. Can a function represent an individual galaxy profile?
2. Can halo assembly predict the function's parameters?

Only the first is tested here. A profile parameterization that is degenerate on
the measured CoG is not a suitable halo-map target, regardless of how small its
residuals are.

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

## Next experiment

Map the four selected parameters from epoch-local DiffMAH and concurrent halo
properties using the same held-out folds as Exp50--Exp51. Compare against
final-mass-only and mass-conditioned shuffled-MAH controls, predict the four
parameters jointly with their residual covariance, reconstruct full profiles,
and judge the result on held-out cumulative and density profiles. Do not add
redshift evolution or free component indices until that single-epoch halo map
shows that the parameters are predictable.

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
```
