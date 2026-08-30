# Exp66 — Expanded constrained symbolic-density search

Status: COMPLETE NULL AT THE PREDECLARED ROUND-A BROAD GATE. The search did not
advance to Round B, selection, or final validation.

## Question

Does a substantially broader library of transition and trigonometric functions
contain a compact, globally physical projected-density family that approaches
unrestricted double Sersic in CoG and density accuracy while retaining
cubic-logit-level conditioning? Exp65 tested only 15 unique teacher galaxies
and one symbolic atom at a time. Exp66 removes that early teacher stop, uses all
3,380 galaxies with complete five-epoch CoGs, and allows bounded compositions,
free-frequency harmonics, localized wave packets, and two-atom expressions.

This remains a representation experiment. Expression structure is shared, but
every galaxy--epoch profile receives its own coordinates. No halo-to-profile
relation or physical component interpretation is inferred.

## Population split

The Exp62 full atlas contains 3,380 unique galaxies with valid CoGs at all five
epochs, or 16,900 galaxy--epoch profiles. A deterministic split with seed 66 is
made within 20 equal-count bins of z=0.4 halo mass, keeping all five epochs of a
galaxy together:

- expression discovery: 1,000 galaxies = 5,000 profiles;
- expression selection: 1,000 galaxies = 5,000 profiles;
- untouched final validation: 1,380 galaxies = 6,900 profiles.

The atlas contains finite z=0.4 halo masses for 2,395 of these galaxies and
missing halo masses for 985. The implemented sort places the missing-mass rows
after the finite-mass rows before making the equal-count blocks. The resulting
discovery, selection, and validation sets contain 708, 708, and 979 galaxies
with finite halo mass, respectively. This is adequate for a profile-family
search, which does not use halo mass in the fit, but it is not the clean
finite-mass stratification implied by the original wording. A future search
must declare the missing-mass stratum separately.

The discovery population has a frozen 250-galaxy broad-screen subset. Every
Round-A expression is fitted to those 1,250 profiles; the best 12 eligible
expressions are then fitted to all 5,000 discovery profiles. Mechanically
generated Round-B compositions are also fitted to all 5,000 discovery
profiles. The best five eligible expressions proceed to the 5,000-profile
selection population. One finalist is frozen before the 6,900-profile final
validation. No galaxy used for expression choice appears in final validation.

## Hard projected-density construction

Let

```text
u = ln(1 + R / R_c),
Sigma(R) = Sigma_0 exp[-Q(u)],
q(u) = dQ/du
     = (1 - exp(-u)) {2 + epsilon + delta [1 + rho T(u)]},
epsilon = 0.02,
delta > 0,
-1 < rho < 1,
|T(u)| <= 1.
```

For two-atom expressions, `T` is constructed as a convex or norm-bounded
combination whose absolute value cannot exceed one. These bounds must hold
algebraically, not through sampled penalties. Consequently every allowed
profile has:

1. finite positive `Sigma(0) = Sigma_0`;
2. zero central radial derivative and a quadratic central density core;
3. positive non-increasing projected density;
4. strictly increasing cumulative projected mass for positive radius;
5. density everywhere asymptotically steeper than `R^-(2 + epsilon)`, hence
   finite infinite projected mass;
6. nonnegative spherical Abel deprojection.

The fitted amplitude is `Mstar(<148.2 kpc)`. The CoG is formed from positive
annular quadrature and normalized exactly at that aperture. Infinite total
mass is labeled as an extrapolation.

## Relaxed Round-A grammar

All candidates include aperture mass, `R_c`, `delta`, and a bounded modulation
amplitude. At most six total per-profile coordinates are allowed in Round A.
The grammar contains:

### Non-trigonometric atoms

```text
constant:             T(u) = 0,
sigmoid transition:   T(u) = tanh[(u - t) / (2 w)],
rational transition:  T(u) = (u - s) / (u + s),
arctangent transition:T(u) = (2/pi) atan[(u - t) / w],
localized shoulder:   T(u) = 2 sech[(u - t) / w] - 1.
```

Locations `t` or scales `s` may be fitted. The fixed global width grid is
`w = 0.35, 0.7, 1.4`.

### Trigonometric atoms

```text
harmonic:       T(u) = sin(omega u + phi),
damped harmonic:T(u) = exp(-lambda u) sin(omega u + phi),
wave packet:    T(u) = sech[(u - t)/w] sin[omega (u - t) + phi],
chirp:          T(u) = sin(omega u + kappa u^2 + phi),
log harmonic:   T(u) = sin[omega ln(1 + u/s) + phi].
```

The fixed grids are

```text
omega = 0.5, 1, 2, 4,
lambda = 0.25, 0.75, 1.5,
phi = 0, pi/2,
kappa = -0.25, 0.25,
w = 0.5, 1.
```

The relaxed library additionally includes a fitted-frequency harmonic, a
fitted-frequency-and-phase harmonic, and fitted-frequency damped sine and
cosine forms. Their frequency range is `0.25 <= omega <= 8`; phase is confined
to `[-pi, pi]`; damping is `0.05 <= lambda <= 3`. Periodic phase and
frequency--scale degeneracies are explicit diagnostics.

## Round-B expression construction

Round B is generated mechanically from discovery results, without selection
data:

1. norm-bounded sums of each of the two best non-trigonometric atoms with each
   of the four best trigonometric atoms;
2. norm-bounded sums of the four best linearly independent trigonometric pairs;
3. products of each of the two best transition atoms with each of the two best
   trigonometric atoms;
4. odd cubic transforms `T^3` and centered-square transforms `2 T^2 - 1` of
   the two best atoms in each class.

Duplicate algebraic forms are removed. Round-B expressions may have at most
seven total fitted coordinates. They are ranked with their actual coordinate
count and cannot win merely through extra flexibility.

## Analytic-property audit

Every expression receives a symbolic property record before fitting:

- exact central density and first nonzero central expansion term;
- analytic logarithmic density slope and its hard global bounds;
- limiting outer slope, or a rigorous lower bound for undamped oscillatory
  slopes;
- whether `Q(u)` is elementary, expressible through a named standard special
  function, or requires numerical quadrature;
- whether projected density is consequently closed form;
- whether cumulative mass, enclosed radii, radial moments, Abel deprojection,
  Fourier transform, and Gaussian/Moffat PSF convolution are elementary,
  standard-special-function, or numerical;
- convergence of all numerical routes at doubled resolution.

For fixed-frequency damped and undamped harmonics, `Q(u)` must use the exact
elementary integral of exponentials times sine/cosine. A numerical integral is
not accepted for those forms. The CoG may still require positive numerical
quadrature. The audit is reported alongside accuracy; mathematical
tractability is not silently reduced to one scalar score.

## Objectives, references, and expression ranking

The paired references are unrestricted double Sersic for analytic accuracy,
cubic-logit for conditioning and resolved CoG accuracy, PCA(3) for numerical
compression, and the unmodulated Exp66 density for nested simplicity.

Expression search uses uniform-radius logarithmic CoG RMS. The frozen finalist
is also fitted under aperture-normalized linear CoG and logarithmic shell
density, including the central 0--2 kpc aperture with the established measured
zero-shell convention. Common evaluation includes full and 5--30 kpc CoG,
shell density, aperture and 100--148 kpc shell masses, and R20/R50/R80/R90.

An expression is eligible at a stage only if:

- at least 99.9% of fits succeed;
- fewer than 5% of fits place a coordinate within 1% of a bound during broad
  discovery, tightened to 1% for selection and final validation;
- at least four epoch medians of its scaled-Jacobian ratio exceed one quarter
  of cubic-logit's paired value during discovery, tightened to one half for
  selection and final validation;
- its worst decoded CoG spread among solutions within 1% of the best loss is
  below 0.003 dex during discovery and 0.001 dex thereafter.

Eligible expressions are ranked lexicographically by the worst-epoch ratio of
median full-CoG RMS to double Sersic, then median shell-density RMS, total
coordinate count, and expression-tree size. The analytic-property record is
shown but does not override a failed physical or numerical gate.

## Degeneracy and robustness tests

- four separated starts for selection and final fits;
- exact synthetic recovery for every atom and generated composition;
- scaled residual-Jacobian singular values, boundary incidence, and all
  solutions within 1% of the best loss;
- doubled quadrature and dense-radius checks;
- alternating-radius and `R >= 5 kpc` refits;
- weakest-direction displacement and profiled scans of every finalist
  coordinate;
- alternate optimizer on 50 mass-, compactness-, and epoch-stratified final
  profiles;
- frequency/damping neighbors, fitted-frequency multimodality, phase wrapping,
  and radial-grid aliasing;
- removal of each atom from a selected composition, reporting the resolved CoG
  change and whether its amplitude collapses to zero;
- structure stability across five bootstrap resamples of discovery galaxies.

## Staged runtime gates

Before any large search, the complete mechanics, synthetic-recovery, all
Round-A expression fits, serialization, metrics, and direct/frontier figures
must run on 25 galaxies at all five epochs--125 profiles--in under 60 measured
wall-clock seconds. This is an operational gate only and cannot support a
scientific family decision.

The 1,250-profile broad screen and each later stage have their wall time
measured rather than estimated. Results are checkpointed by expression so an
interruption does not repeat completed fits. No selection or validation stage
starts until the preceding stage has completed and its fixed survivor list is
serialized.

## Final replacement and visual gates

On the untouched 1,380-galaxy, 6,900-profile validation population, a
replacement must satisfy at every epoch:

- median full-CoG, 5--30 kpc CoG, shell-density, and R50/R80/R90 errors no more
  than 10% larger than unrestricted double Sersic;
- scaled-Jacobian median at least half cubic-logit's value at four epochs;
- boundary incidence below 1%;
- median near-optimal raw-coordinate spread below 1% of an allowed range and
  worst decoded spread below 0.001 dex;
- no failed analytic-property, doubled-grid, radial-jackknife, frequency, or
  alternate-optimizer test.

A numerical pass is not called promising until average CoGs and residuals in
halo- and stellar-mass bins and the stellar-mass planes pass direct inspection.
Only then is the complete aperture, annular, outskirt, density, size,
population-distribution, and representative-object QA battery generated.

Exp66 stops after final validation. No post-result grammar expansion or manual
repair is allowed inside this experiment.

## Result

The complete 125-profile operational path searched all 88 Round-A expressions,
ran the mechanics and doubled-resolution checks, serialized the fits, computed
the common metrics, and generated both declared figures in 57.76 seconds. It
therefore passed the sub-minute development gate. The mechanics check recovered
representatives of every atom class to `2.27e-7--2.16e-6 dex` CoG RMS. Changing
the density quadrature from 512 to 2,048 points changed their CoGs by at most
`1.61e-4 dex`.

The broad screen then fit all 88 expressions to 250 discovery galaxies at all
five epochs, or 1,250 measured profiles, in 623.17 seconds. None passed all of
the predeclared Round-A eligibility rules, so the experiment stopped without
using the remaining 3,130 galaxies for expression choice or validation and
without constructing Round B.

The most accurate expression by the frozen ranking was the six-coordinate
fitted-frequency-and-phase harmonic. Its median full-CoG RMS is
`0.00238--0.00349 dex` across epochs, compared with
`0.00187--0.00280 dex` for unrestricted double Sersic on exactly the same
profiles; its worst epoch is therefore 29.75% worse than double Sersic. Only
95.36% of its fits report optimizer convergence, 65.36% lie within 1% of a
coordinate bound, and its median scaled-Jacobian strength is only
0.62--1.94% of cubic-logit's paired value. The added free phase improves raw
accuracy but creates a severe frequency--phase degeneracy.

The most interesting accuracy--conditioning compromise is the five-coordinate
log-periodic cosine

```text
T(u) = sin[4 ln(1 + u/s) + pi/2].
```

Its median full-CoG RMS is `0.00250--0.00292 dex`, compared with
`0.00187--0.00280 dex` for unrestricted double Sersic; its worst epoch is
33.35% worse. Its median scaled-Jacobian strength is 26.17--40.28% of
cubic-logit's paired value, so it clears the discovery conditioning threshold
at every epoch. It nevertheless has only 96.24% optimizer convergence and
28.40% boundary incidence. The frequency-2 log-periodic cosine improves
optimizer convergence to 99.92% and has 27.62--32.79% of cubic-logit's
conditioning, but remains 47.95% worse than double Sersic at its worst epoch
and has 22.72% boundary incidence. For that form, 17.12% of all fits pin the
positive modulation coordinate at its upper limit and 4.08% pin it at its
lower limit. The family is asking for stronger slope contrast than the
`|rho| < 1` wrapper permits.

Ordinary fixed-frequency harmonics, damped harmonics, and chirps do not offer a
competitive compromise. Damping usually makes both accuracy and conditioning
worse because a rapidly vanishing atom becomes interchangeable with the core
scale and slope amplitude. Several log-periodic forms and the broad
frequency-2 wave packet are genuinely better in raw CoG accuracy than the
other new atoms, but no one-atom family combines double-Sersic accuracy,
cubic-logit conditioning, and non-saturated coordinates.

**Decision:** Exp66 is a null for the declared bounded-linear slope modulation,
not a null for all symbolic or trigonometric density families. Preserve the
log-periodic atom as the most interesting new functional form. Do not relax an
Exp66 gate after seeing the result. A separate experiment may replace the
saturated linear modulation with an algebraically positive squared-slope
construction, which permits stronger contrast and retains elementary density
exponents for fixed harmonics.

## Required figures

Every figure uses `hongshao.plotting`, is saved as PNG and PDF, and is displayed
for direct review.

1. Accuracy--conditioning--complexity frontier with mathematical-property
   classes and trigonometric/non-trigonometric expressions distinguished.
2. Direct CoG, CoG residual, projected density, logarithmic density slope, and
   density residual panels for best/typical/worst and compact/extended cases.
3. Frequency, damping, phase, atom-amplitude, boundary, and near-solution
   diagnostics for surviving trigonometric forms.
4. Minimum promising-model gate: average CoGs in halo- and stellar-mass bins
   and stellar-mass planes on untouched validation galaxies.
5. Remaining standard QA only after the minimum visual gate passes.

## Review checklist

- [x] Confirm predeclaration predates evaluator and driver code.
- [x] Verify exact 1,000/1,000/1,380 galaxy split and five-epoch containment.
- [x] Report all searched expressions and their mathematical-property records.
- [x] Record every measured stage time and survivor list (the survivor list is
  empty).
- [x] Keep discovery, selection, and untouched validation results separate.
- [x] Display and inspect every generated figure in the generating turn.
- [x] End without changing a gate or adding a post-result expression.
