# Exp67 — Squared-slope symbolic-density search

Status: COMPLETE NULL AT THE PREDECLARED DISCOVERY GATE. Round B, selection,
and final validation were not triggered.

## Motivation and question

Exp66 searched 88 bounded transition and trigonometric atoms on 250 galaxies at
all five epochs. Its best accuracy--conditioning compromise, a log-periodic
cosine, frequently drove the relative modulation to `|rho| = 1`. That is a
failure of the bounded-linear slope wrapper as well as a limitation of the
atom. Exp67 asks whether the same relaxed atom library becomes competitive
when the positive logarithmic density slope can have arbitrarily strong but
still globally physical contrast.

This is a new family rather than a repair applied to individual profiles. One
expression tree and one parameterization are shared by every fitted profile;
only its numerical coordinates vary. No halo information enters the fits.

## Frozen population accounting

The Exp62 full atlas has 3,380 galaxies with valid measured CoGs at all five
epochs, or 16,900 profiles. At z=0.4, 2,395 galaxies have finite aligned halo
mass and 985 have missing halo mass. Seed 67 defines a galaxy-level split with
all epochs kept together:

- discovery: 1,200 galaxies = 6,000 profiles;
- selection: 1,000 galaxies = 5,000 profiles;
- untouched validation: 1,180 galaxies = 5,900 profiles.

Finite-mass galaxies are randomized within 20 equal-count halo-mass bins. The
missing-mass galaxies form an explicit twenty-first stratum. Integer allocation
uses fixed largest-remainder counts so the three population totals are exact.
No small scientific broad screen precedes discovery: every Round-A expression
is evaluated on all 1,200 discovery galaxies. This directly addresses the
small-sample limitation of Exp65 and Exp66.

## Hard squared-slope family

Define

```text
u = ln(1 + R / R_c),
Sigma(R) = Sigma_0 exp[-Q(u)],
q(u) = dQ/du
     = (1 - exp(-u)) {2 + epsilon + [a + b T(u)]^2},
epsilon = 0.02,
a > 0,
|T(u)| <= 1.
```

The optimizer coordinates are `ln(a)` and signed `b`; `a > 0` removes the
exact simultaneous sign symmetry `(a,b) -> (-a,-b)`. There is no relative
modulation ceiling analogous to Exp66's `|rho| < 1`. The square supplies the
positivity constraint algebraically.

Every allowed profile therefore has finite positive central density, a zero
central radial derivative and quadratic core, positive non-increasing
projected density, strictly increasing cumulative mass, asymptotic density no
shallower than `R^-(2.02)`, finite infinite projected mass, and nonnegative
spherical Abel deprojection. The fitted amplitude remains
`Mstar(<148.2 kpc)`, normalized through positive annular quadrature.

The unmodulated nested reference fixes `b = 0`. A fitted `b` statistically
consistent with zero removes the atom; its atom-specific location, frequency,
or phase is then declared unidentified rather than interpreted.

## Round-A grammar

Exp67 repeats the complete Exp66 atom library without choosing atoms from the
Exp66 result:

```text
tanh transition: T = tanh[(u-t)/(2w)]                 w=0.35,0.7,1.4
rational:        T = (u-s)/(u+s)
arctangent:      T = (2/pi) atan[(u-t)/w]             w=0.35,0.7,1.4
shoulder:        T = 2 sech[(u-t)/w]-1                w=0.35,0.7,1.4
harmonic:        T = sin(omega u+phi)
damped harmonic: T = exp(-lambda u) sin(omega u+phi)
wave packet:     T = sech[(u-t)/w] sin[omega(u-t)+phi]
chirp:           T = sin(omega u+kappa u^2+phi)
log harmonic:    T = sin[omega ln(1+u/s)+phi]
```

The fixed grids remain `omega = 0.5,1,2,4`, `phi = 0,pi/2`,
`lambda = 0.25,0.75,1.5`, `kappa = -0.25,0.25`, transition width
`w = 0.35,0.7,1.4`, and packet width `w = 0.5,1`. Fitted-frequency sine and
cosine, fitted-frequency-and-phase harmonic, and fitted-frequency-and-damping
sine and cosine are included with the Exp66 ranges. Round A has at most six
coordinates including aperture mass.

Round B is generated only from discovery results: bounded means and products
of the best two transition and four linearly independent trigonometric atoms,
plus `T^3` and `2T^2-1` transforms. It has at most seven coordinates. Duplicate
trees are removed mechanically.

## Analytic-property audit

Before fitting, every family records the exact central expansion, analytic
logarithmic density slope and lower bound, outer behavior, and the analytic or
numerical status of `Q`, projected density, cumulative mass, enclosed radii,
moments, Abel deprojection, Fourier transform, and Gaussian/Moffat PSF
convolution.

For a fixed harmonic `T = sin(omega u + phi)`, expand

```text
[a+bT]^2 = a^2 + b^2/2 + 2ab sin(omega u+phi)
            - (b^2/2) cos(2 omega u+2 phi).
```

Consequently `Q(u)` and `Sigma(R)` must use the exact elementary integrals of
exponentials times sine/cosine. Numerical integration is forbidden for fixed
or fitted single harmonics and damped single harmonics where the finite
exponential--trigonometric expansion is elementary. Other atoms and
compositions may use checked numerical quadrature. Every numerical route is
repeated at doubled resolution.

## Fitting, references, and frozen gates

The primary search objective is uniform-radius logarithmic CoG RMS. References
are unrestricted double Sersic for accuracy, cubic-logit for conditioning and
resolved accuracy, PCA(3) for numerical compression, and `b=0` for nested
simplicity. Common reporting includes full and 5--30 kpc CoG RMS, independent
shell-density RMS, aperture and 100--148 kpc shell masses, and R20/R50/R80/R90.

Round A uses two separated starts on all 6,000 discovery profiles. An expression
may enter the top 12 only if at least 99.9% of fits converge, fewer than 5% of
fits place a coordinate within 1% of a finite bound, at least four epoch
medians of its scaled-Jacobian ratio exceed 0.25 of cubic-logit's paired value,
its worst decoded CoG spread among solutions within 1% of the best loss is
below 0.003 dex, and its worst-epoch median full-CoG RMS is less than 1.5 times
double Sersic. The final accuracy condition prevents an extremely stable but
inaccurate form from consuming the composition round.

Round-B expressions and surviving Round-A expressions are ranked
lexicographically by worst-epoch full-CoG RMS ratio to double Sersic, median
shell-density RMS, coordinate count, and tree size. The best six eligible
expressions are fit to all 5,000 selection profiles with four starts. Selection
tightens boundary incidence to 1%, requires at least four epoch Jacobian ratios
above 0.5 of cubic-logit, and limits near-optimal decoded spread to 0.001 dex.
One finalist is frozen before validation.

The finalist is fit under logarithmic CoG, aperture-normalized linear CoG, and
logarithmic shell-density objectives on all 5,900 untouched validation profiles.
It replaces double Sersic only if, at every epoch, median full-CoG, 5--30 kpc
CoG, shell-density, and R50/R80/R90 errors are no more than 10% larger; the
selection conditioning, boundary, convergence, and degeneracy gates also must
hold. The mathematical audit cannot be traded against a failed numerical gate.

## Degeneracy tests

- exact synthetic recovery for every atom and generated composition;
- the explicit absence of the Exp66 `(delta,rho)` saturation direction;
- four separated starts at selection and validation;
- scaled residual-Jacobian singular values and all solutions within 1% of the
  best loss;
- profile scans along the weakest singular direction;
- atom-removal fits with `b=0` and likelihood-free CoG loss differences;
- frequency, phase wrapping, damping, and radial-grid alias tests;
- alternating-radius and `R >= 5 kpc` refits;
- an alternate optimizer on 50 stratified validation profiles;
- five galaxy-bootstrap discovery rerankings;
- doubled quadrature resolution and dense-radius physical checks.

## Operational gate and checkpointing

Before discovery, every Round-A expression, mechanics and synthetic recovery,
serialization, common metrics, and direct/frontier figures must complete on 25
galaxies at all five epochs in under 60 measured seconds. This is an
operational gate only. Discovery is checkpointed by expression. Measured wall
times are reported; no full-stage duration is estimated. A failed gate stops
the experiment without modifying the grammar.

## Visual gates

Every generated figure is saved as PNG and PDF with `hongshao.plotting` and
displayed in the generating turn. The operational and discovery stages show an
accuracy--conditioning frontier and direct CoG/residual/density/slope panels.
For a selection-eligible family, frequency/phase/amplitude and near-solution
diagnostics are added. A validation result cannot be called promising until
average CoGs and residuals in halo- and stellar-mass bins and the stellar-mass
planes have been inspected. Only a promising validation result receives the
full standard aperture, annular, outskirt, density, size, CDF, and
representative-object QA battery.

## Result

The complete operational path fit all 88 Round-A expressions to 25 galaxies at
all five epochs, ran mechanics and synthetic recovery, serialized results,
computed the common metrics, and generated the direct and frontier figures in
42.88 seconds. It passed the sub-minute gate. Representative synthetic CoGs
were recovered to `3.03e-7--2.09e-6 dex` RMS, and increasing density quadrature
from 512 to 2,048 points changed CoGs by at most `1.64e-4 dex`.

The scientific discovery stage then fit every Round-A expression to all 1,200
discovery galaxies at all five epochs with two separated starts: 6,000 profiles
and 12,000 optimizations per expression. The 88-expression stage took 3,905.30
seconds. No expression passed every frozen eligibility rule, so the experiment
stopped without constructing Round B or inspecting the 1,000 selection and
1,180 validation galaxies.

### Accurate but non-identifiable elementary family

The lowest raw error comes from the fitted-frequency and fitted-damping cosine

```text
T(u) = exp(-lambda u) cos(omega u),
q(u) = (1-exp(-u)) {2.02 + [a+b exp(-lambda u) cos(omega u)]^2}.
```

Its median full-CoG RMS is `0.00167--0.00178 dex` across epochs, compared with
`0.00197--0.00275 dex` for unrestricted double Sersic on exactly the same
profiles. Its worst epoch is still 13.67% better than double Sersic. This is the
first tested non-central-hole density family to beat double Sersic in the
median CoG statistic at every epoch.

The accuracy is not usable as a stable coordinate system. Only 99.50% of fits
converge, 17.83% lie within 1% of a finite coordinate bound, and its median
scaled-Jacobian strength is only 2.15--3.10% of cubic-logit's paired value. The
damping coordinate is within 1% of its lower bound in 12.52% of all profiles,
showing that damping and the remaining slope coordinates are often
interchangeable. Three of 6,000 profiles have two solutions within 1% of the
best loss whose decoded CoGs differ by more than 0.003 dex; the maximum is
0.00443 dex.

This family nevertheless has attractive mathematics. Expanding the square
gives a constant, an exponentially damped first harmonic, an exponentially
damped constant term, and an exponentially damped second harmonic. Therefore
`Q(u)` is an exact finite sum of elementary exponential--sine/cosine integrals,
and `Sigma(R)=Sigma_0 exp[-Q(u)]` is closed form. Its central density is finite
with zero first derivative, its outer logarithmic density slope tends exactly
to `-(2.02+a^2)`, and its infinite projected mass is finite. Cumulative mass,
enclosed radii, Abel deprojection, Hankel transform, and PSF convolution remain
numerical.

The fitted-frequency and fitted-damping sine behaves similarly: its median
full-CoG RMS is `0.00171--0.00191 dex`, but its Jacobian strength is only
1.38--3.60% of cubic-logit's paired value, its boundary incidence is 21.35%,
and only 99.05% of fits converge. Free damping buys raw accuracy by creating a
nearly null direction.

### Best accuracy--conditioning compromise

The strongest stable compromise is the fixed localized sine packet

```text
T(u) = sech[(u-t)/0.5] sin[0.5(u-t)],
q(u) = (1-exp(-u)) {2.02 + [a+b T(u)]^2}.
```

Its median full-CoG RMS is `0.00234--0.00255 dex`, compared with
`0.00197--0.00275 dex` for double Sersic; its worst epoch is 14.78% worse than
double Sersic. It has 99.95% convergence, 2.72% boundary incidence, and median
scaled-Jacobian strength equal to 46.87--62.80% of cubic-logit's paired value.
Unlike the free damped harmonic, this is a reasonably conditioned
five-coordinate family.

It fails the frozen worst-case multi-start test. Twelve of 6,000 profiles have
two solutions within 1% of the best loss whose decoded CoGs differ by more than
0.003 dex; the maximum is 0.00695 dex. Only 0.2% of the population crosses that
limit, but the rule was deliberately a worst-profile replacement gate. The
packet's `Q(u)` and density require numerical quadrature because the
`sech`-localized trigonometric square has no elementary antiderivative in the
required exponential-weighted form. Its outer slope tends exactly to
`-(2.02+a^2)` because the packet vanishes at large `u`; cumulative mass and all
transform/convolution routes are numerical.

The frequency-1, width-0.5 sine packet is slightly closer to double Sersic at
its worst epoch (12.57% worse) and has 55.01--71.30% of cubic-logit's Jacobian
strength, but its convergence rate is 99.85% and nine profiles exceed the
0.003 dex near-solution spread limit. Other low-frequency packet phases and
widths reproduce the same accuracy cluster, whereas ordinary harmonics and
chirps are substantially worse. The packet class is a real result rather than
one tuned phase choice.

### Log-periodic family

The fixed frequency-4 log-periodic sine

```text
T(u) = sin[4 ln(1+u/s)]
```

has median full-CoG RMS `0.00222--0.00290 dex`, 99.983% convergence, 4.88%
boundary incidence, and 25.59--48.26% of cubic-logit's Jacobian strength. It
passes the broad accuracy, convergence, boundary, and conditioning screens,
but a near-equal solution pair differs by 0.01499 dex. Its stretched
oscillation is more useful than ordinary frequency-4 ringing, but both `Q` and
the projected density require numerical quadrature.

### Decision

The squared-slope construction shows that Exp66's bounded modulation was
prematurely restrictive: new trigonometric densities can match or beat double
Sersic without a central hole. The remaining trade-off is structural, not a
failure to search enough galaxies. Free damping attains the best accuracy but
is badly non-identifiable; fixed packets approach double-Sersic accuracy with
cubic-logit-like local conditioning but retain rare multi-start branch
ambiguities and lose a closed-form density exponent.

Do not promote any Exp67 expression as the production replacement and do not
relax a worst-case gate after seeing that only a few profiles fail it. Retain
the fixed low-frequency packet as the most interesting numerically stable new
family and the free damped squared harmonic as the most interesting analytic
family. Further broad grammar expansion is unlikely to resolve their opposing
trade-off. A future investigation, if desired, should be narrowly targeted at
the packet's discrete sign/location branches rather than another open-ended
symbolic search.

## Review checklist

- [x] Confirm this predeclaration predates evaluator and driver code.
- [x] Verify the 1,200/1,000/1,180 galaxy split and explicit missing-mass stratum.
- [x] Run the focused checks and the measured operational gate.
- [x] Fit all Round-A expressions on 6,000 discovery profiles without a small
  scientific pre-screen.
- [x] Report every expression, analytic-property class, and measured stage time.
- [x] Keep selection and validation untouched until their frozen stage.
- [x] Display and inspect every generated figure in the generating turn.
- [x] Stop without changing a gate or adding a post-result expression.
