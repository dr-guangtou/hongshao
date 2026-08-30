# Exp70 — Population-wide damped cosine

Status: PREDECLARED. No Exp70 evaluator, driver, or fit exists yet. Exp67
selection and validation profiles have not been loaded by this experiment.

## Production question

Exp67 established that the six-coordinate free damped-cosine density family is
more accurate than unrestricted double Sersic but has a nearly unidentified
per-profile damping coordinate. Exp69 showed on its 25-galaxy operational
sample that changing coordinate labels or choosing a canonical near-optimal
branch does not remove that weak direction, while fixing damping materially
strengthens the remaining five coordinates.

Exp70 asks the production question left open by those experiments:

> Can one damping value, shared by every galaxy and every epoch, retain the free
> damped-cosine family's CoG and shell-density accuracy while providing stable
> five-coordinate profile fits?

This is a new experiment rather than a revision of Exp69's failed runtime gate.
Its narrower scope follows the measured Exp69 result. It does not reopen pivot
coordinates, finite-window orthogonalization, canonical branch selection, a
broad damping search, or any symbolic grammar.

## Frozen physical family and parameter accounting

Every fitted profile retains

```text
u = ln(1 + R/R_c),
Sigma(R) = Sigma_0 exp[-Q(u)],
Q'(u) = (1-exp(-u)) {2.02 +
         [a + b exp(-lambda u) cos(omega u)]^2}.
```

The free reference has six per-profile coordinates:

```text
(log Mstar_148, log R_c, log a, b, log omega, log lambda).
```

Each fixed-damping candidate has five per-profile coordinates:

```text
(log Mstar_148, log R_c, log a, b, log omega),
```

plus one `lambda` shared by the complete population. The amplitude is the
primary CoG-derived stellar mass inside 148.2 kpc in h-free
`log10(M / Msun)`. Direct two-dimensional projected apertures remain
cross-checks only.

The mathematical guarantees are unchanged for every declared value: finite
positive central projected density, zero central derivative, non-increasing
projected density, strictly increasing cumulative mass, asymptotic slope
steeper than `R^-2.02`, finite projected mass, and nonnegative spherical Abel
deprojection. `Q(u)` and `Sigma(R)` remain elementary finite sums; cumulative
mass and enclosed radii remain numerical.

## Frozen shared-damping grid

The only fixed candidates are

```text
lambda = 0.75, 1.00, 1.25, 1.50.
```

The endpoints are the two Exp69 operational cases that bracketed the observed
accuracy--conditioning trade. The two interior values divide that already
measured interval into equal steps. One value must be chosen globally on
discovery; `lambda` may never be selected separately by galaxy, epoch, halo
mass, stellar mass, or fitting objective. The grid cannot be extended or
refined after any Exp70 result is seen.

The original six-coordinate free-damping family is newly fit with the same
starts, numerical resolution, and loss as the fixed candidates and is the
functional reference. Unrestricted double Sersic is the paired raw-accuracy
reference, cubic-logit is the paired conditioning reference, and PCA(3) is the
numerical compression reference. Those three established references are read,
not refit. The atom-removed `b=0` model is the nested simplicity reference; if
validation is reached, it is fit only on 50 predeclared stratified validation
profiles to measure whether modulation is needed, not to choose `lambda`.

## Blind population roles

Exp70 inherits Exp67's seed-67 galaxy-level partition exactly, with all five
epochs of a galaxy kept together:

- discovery: 1,200 galaxies and 6,000 profiles; fit all four shared-damping
  values and the free reference, apply every discovery gate, and freeze exactly
  one fixed value if any is eligible;
- selection: 1,000 galaxies and 5,000 profiles; fit only the frozen fixed value
  and the free reference, with no damping retuning, and require every stricter
  selection gate to pass;
- untouched validation: 1,180 galaxies and 5,900 profiles; loaded only after
  selection passes, then fit the frozen candidate and free reference once under
  all three objectives for the final production decision.

The 25-galaxy operational sample is the exact Exp67/Exp69 discovery-only sample
of 20 finite-halo-mass quantiles and five missing-halo-mass galaxies, with all
five epochs retained. It tests execution only and cannot choose `lambda`.

The preserved Exp67 discovery archive is accessed only through
`HONGSHAO_EXP67_ROOT`. Every stored archive must carry indices exactly equal to
the authorized stage. Operational and discovery code must reject any fit
archive containing an Exp67 selection or validation index. Selection and
validation are never loaded speculatively.

## Fitting objectives and starts

The three deterministic objectives are losses, not likelihoods:

1. primary: uniform-radius RMS in `log10 Mstar(<R)` at the 24 measured radii;
2. secondary: RMS of linear enclosed-mass difference divided by measured
   `Mstar(<148.2 kpc)`;
3. local: RMS in logarithmic mean shell surface density, including 0--2 kpc and
   using a 1 Msun floor only for an exactly zero measured shell.

Operational fitting uses four separated deterministic starts. Discovery,
selection, and validation use eight starts. Discovery and selection use only
the primary objective. Validation independently refits the candidate and free
reference under all three objectives.

Every start stores its coordinates, objective, convergence flag, CoG,
residual-Jacobian singular values, weakest right-singular direction, and common
physical descriptors. Every solution within 1% of the best mean squared loss
plus `1e-14` is retained. The reported fit is always the lowest-loss solution;
Exp69 showed that a conditioned branch tie-breaker was not effective, so it is
not reopened here.

## Mechanics and synthetic recovery

Before measured-profile fitting, all five systems must pass:

- exact aperture normalization and strictly increasing numerical CoGs;
- the dense-radius squared-slope physical audit;
- analytic `Q(u)` versus a sufficiently resolved numerical integral to maximum
  absolute difference below `3e-7`;
- CoGs evaluated at 512 and 2,048 integration points differing by less than
  `4e-4 dex`;
- noiseless recovery for both contrast signs at every declared fixed damping
  value and four matched free-family cases, from four displaced starts, with
  full-CoG RMS below `3e-5 dex` and maximum common physical-descriptor error
  below 0.03 of its declared range.

## Common measurements and continuity

Every fitted system reports, by epoch:

- median full and 5--30 kpc CoG RMS relative to the measured CoGs;
- median shell-density RMS relative to the measured annular densities;
- fixed-kpc aperture, annular, and 100--148.2 kpc shell-mass errors;
- R20/R50/R80/R90 errors;
- convergence fraction, finite-bound incidence, scaled minimum-Jacobian
  strength relative to the paired cubic-logit value, and worst near-optimal CoG
  spread.

Continuity uses the measured-profile pairs frozen independently of every model.
Within each epoch, amplitude-pin every measured CoG at 148.2 kpc and pair each
profile with its nearest other measured shape. Separately pair each galaxy at
its four adjacent epochs. For fixed damping, use the common five descriptors
`(log R_c, a^2, c_p, psi)` plus signed contrast normalized by its declared
range; damping itself is absent because it is global. Report median and 95th-
percentile descriptor separations for both pair classes and retain the ten
largest jumps with their measured and fitted profiles.

## Frozen operational gate

The complete 25-galaxy, 125-profile path must finish in less than 60 measured
seconds. It includes all mechanics, 20 synthetic recoveries, four-start fitting
of the free reference and four fixed candidates, serialization, common metrics,
continuity, and these five PNG/PDF figure classes:

1. synthetic CoG and physical-descriptor recovery;
2. the one-dimensional shared-`lambda` trade among CoG accuracy, shell-density
   accuracy, convergence, boundary incidence, Jacobian strength, near-solution
   spread, and continuity;
3. all multi-start CoGs and residuals for common best, typical, and worst
   profiles;
4. decoded CoGs along the weakest scaled-Jacobian direction for the free
   reference and every fixed candidate;
5. nearest-profile and adjacent-epoch coordinate-continuity distributions with
   the largest jumps shown directly.

This is an operational gate only. Failure serializes the result and stops Exp70
without removing a start, candidate, measurement, or figure.

## Frozen discovery gates and global choice

A fixed value is discovery-eligible on all 6,000 discovery profiles only if:

- at least 99.9% of fits converge;
- fewer than 5% of fits place a native coordinate within 1% of a finite bound;
- at every epoch its median scaled minimum-Jacobian strength is at least 0.15
  of cubic-logit's paired value, and at least four epochs reach 0.25;
- at every epoch its median scaled minimum-Jacobian strength is at least four
  times the newly refit free-damping reference;
- its maximum CoG separation among solutions within 1% of the best loss is
  below `0.003 dex`;
- at every epoch its median full-CoG, 5--30 kpc CoG, and shell-density RMS are
  no more than 10% larger than the newly refit free-damping reference;
- its nearest-profile and adjacent-epoch 95th-percentile coordinate
  separations are each at most 0.75 times the free-family values, and neither
  median separation is larger than the free-family median.

Eligible fixed values are ranked by the largest, across epochs, full-CoG RMS
ratio to the free family; then by the largest shell-density RMS ratio; then by
the worse of the two continuity ratios; then by smaller `lambda`. Exactly one
value is frozen. No result-dependent interpolation between grid values is
allowed.

Before calling the frozen value scientifically interesting, inspect average
measured and fitted CoGs and residuals in z=0.4 halo- and stellar-mass terciles,
plus all three standard z=0.4 inner--outer stellar-mass planes. Summary metrics
alone cannot advance a candidate.

## Frozen selection gates

Selection fits only the frozen fixed value and the free reference on all 5,000
selection profiles. The fixed value advances only if:

- at least 99.9% of fits converge;
- fewer than 1% of fits lie within 1% of a finite bound;
- every epoch reaches 0.15 of cubic-logit's paired Jacobian strength and at
  least four epochs reach 0.25;
- every epoch remains at least four times stronger than the free reference;
- maximum near-optimal CoG separation is below `0.001 dex`;
- the same 10% per-epoch full-CoG, 5--30 kpc CoG, and shell-density allowances
  relative to the free reference hold;
- the same 0.75 continuity improvement repeats.

Nothing is retuned after selection. Passing selection triggers the complete
standard QA battery: aperture, annular, and outskirt masses; average CoGs and
residuals in halo- and stellar-mass bins; density profiles; observational and
size planes; cumulative residual distributions; and common best, typical, and
worst individual profiles. Every figure states the population, measured
reference, and main interpretation.

## Frozen validation and production decision

Validation independently refits the frozen candidate and free reference under
all three objectives on all 5,900 untouched profiles. Production promotion
requires:

- every primary-objective selection gate to repeat;
- under every objective and at every epoch, median full-CoG, 5--30 kpc CoG,
  shell-density, and R50/R80/R90 errors no more than 10% larger than the free
  family fitted under that same objective;
- under every objective and at every epoch, those same errors no more than 10%
  larger than unrestricted double Sersic's paired values;
- no new systematic trend in residuals with halo mass, stellar mass, size, or
  redshift in the complete visual QA;
- five-epoch stellar-growth diagnostics showing no branch-like coordinate jump
  or coherent bias in total and radial stellar growth.

If every requirement passes, the family is worth production integration as a
five-coordinate profile system with one immutable library-level damping
constant. Any failed accuracy, convergence, boundary, conditioning,
near-solution, continuity, or visual gate rejects production promotion; a gain
in another metric cannot compensate.

## Checkpointing and figures

Discovery is checkpointed separately for the free reference and each fixed
value. Selection and validation are checkpointed by family and objective. A
failed stage writes its numerical summary, manifest, fit archive, and figures
before stopping. Measured wall times are reported; full-stage durations are
never estimated.

All figures use `hongshao.plotting.set_style()`, the Okabe--Ito palette,
redundant line styles/markers, and `cividis` for same-sign sequential maps. They
are saved as PNG and PDF with `hongshao.plotting.save_fig()`, displayed in the
generating turn, and listed with full paths and one-line descriptions.

## Review checklist

- [x] Freeze the family, four-value global damping grid, references, blind
  sample roles, objectives, starts, mechanics, metrics, continuity definition,
  stage gates, production QA, checkpointing, and sub-minute path before writing
  Exp70 code.
- [ ] Write focused checks first and observe failure because Exp70 code is
  absent.
- [ ] Pass the complete operational path in less than 60 measured seconds.
- [ ] Fit all four fixed values and the free reference on discovery with no
  grid change; freeze at most one eligible value.
- [ ] If discovery passes, confirm the frozen value on untouched selection with
  no retuning and render full QA.
- [ ] If selection passes, use untouched validation once under all three
  objectives and make the production decision.
- [ ] Display and inspect every generated figure in its generating turn.
- [ ] Record the measured result without changing a gate or adding a damping
  value after seeing results.
