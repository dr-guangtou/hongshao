# Exp65 — Constrained symbolic projected-density search

Status: COMPLETE. The hard-constrained grammar fails the predeclared synthetic
teacher feasibility gate, so measured-profile expression search and population
validation were not run. Nothing is promoted to the HongShao library.

## Question

Can a compact expression for the logarithmic slope of a projected stellar
density match unrestricted double Sersic in resolved curve-of-growth (CoG)
accuracy and cubic-logit in conditioning, while guaranteeing a finite,
strictly positive central density? The search includes localized trigonometric
terms, as requested, but they are not allowed to weaken the physical
constraints or create an oscillatory asymptotic tail.

This is a representation experiment. It does not infer physical stellar
components or a halo-to-profile relation. A negative result closes the present
analytic-family search; it does not prove that no other profile law exists.

## Data and separation of search from validation

- The primary observable is the 24-point CoG-derived stellar mass from
  2.0--148.2 kpc at redshifts 0.4, 0.7, 1.0, 1.5, and 2.0 used by Exp62.
- Exp62's established 90-profile development sample is retained. Galaxy folds
  0--2 form the expression-discovery subset and folds 3--4 form the
  development holdout. All epochs of a galaxy remain on one side.
- Only the three discovery-selected expressions are evaluated on the
  development holdout. One expression is then frozen and refitted to all 90
  profiles under the three declared objectives.
- If the development gates pass, the final population validation excludes all
  90 development galaxy--epoch pairs from the 16,900-profile Exp62 record.
- Exp62's stored predictions supply paired unrestricted-double-Sersic,
  cubic-logit, and PCA(3) references. They are never refitted here.
- Masses are h-free `log10(M / Msun)`. The amplitude is the stellar mass inside
  the measured 148.2 kpc aperture, not an inferred infinite total.

## Hard mathematical wrapper

For a positive scale radius `R_c`, define

```text
u = ln(1 + R / R_c),
Sigma(R) = Sigma_0 exp[-Q(u)],
Q(u) = integral_0^u q(v) dv,
q(u) = (1 - exp(-u)) [2 + epsilon + softplus(g(u))],
epsilon = 0.02.
```

Symbolic search acts only on the bounded expression `g(u)`. For every allowed
expression and parameter vector, the wrapper guarantees:

1. `Sigma(0) = Sigma_0` is finite and strictly positive;
2. the central radial derivative is zero because `q(u) = O(u)`;
3. `Sigma(R)` is positive and non-increasing for every `R >= 0`;
4. cumulative projected mass is nonnegative and strictly increasing for
   positive radius;
5. the asymptotic projected-density slope is everywhere steeper than
   `R^-(2 + epsilon)`, so the infinite projected mass is finite;
6. the spherical Abel deprojection is nonnegative, without implying that a
   spherical galaxy interpretation is physically correct.

The CoG is evaluated by positive annular quadrature and normalized exactly to
the fitted mass inside 148.2 kpc. Infinite total mass is a reported
extrapolation only.

## Restricted symbolic grammar

The independent terminal is `u`. Per-profile scalar coefficients, positive
locations, and positive scales are allowed subject to a maximum of six total
fitted coordinates including aperture mass and `R_c`. The expression tree may
use addition, subtraction, multiplication, and complements of bounded atoms.
It may not use unrestricted division, arbitrary powers, raw exponentials or
logarithms, or a polynomial in `u`.

The non-trigonometric atoms are

```text
S(u; t, w) = sigmoid[(u - t) / w],
H(u; s) = u / (u + s),
```

with positive `t`, `w`, and `s`. The discovery grid fixes sigmoid width to
`w = 0.5, 1, 2`; the rational scale remains a fitted coordinate.

The relaxed grammar also contains bounded, localized trigonometric atoms

```text
T(u; lambda, omega, phi)
  = exp(-lambda u) sin(omega u + phi),
```

with the finite global grid

```text
lambda = 0.5, 1,
omega = 1, 2,
phi = 0, pi/2.
```

Thus both sine- and cosine-phase behavior are searched. Frequency, damping,
and phase are global expression choices rather than per-galaxy coordinates;
this prevents frequency--phase--scale exchange from becoming an unmeasured
per-profile degeneracy. Damping makes the trig contribution vanish in the
asymptotic tail. Trigonometric structure modulates the positive logarithmic
density slope; the density itself can never oscillate upward.

### Frozen search rounds

Round A fits these one-atom structures on the discovery subset:

```text
g(u) = beta_0,
g(u) = beta_0 + beta_1 S(u; t, w),
g(u) = beta_0 + beta_1 H(u; s),
g(u) = beta_0 + beta_1 T(u; lambda, omega, phi).
```

Round B is generated mechanically, without inspecting the holdout: combine
the best discovery non-trigonometric atom with the best discovery
trigonometric atom, and combine the two best linearly independent
trigonometric atoms. Global atom choices remain those selected in Round A.
Expressions exceeding six total fitted coordinates are not generated.

Within each round an expression is eligible only if every fit is finite, its
discovery boundary incidence is below 5%, its median scaled-Jacobian ratio is
at least one quarter of cubic-logit's paired value at four epochs, and its
worst decoded spread among solutions within 1% of the best loss is below
0.003 dex. Eligible expressions are ranked by the worst, across epochs, ratio
of their median full-CoG RMS to unrestricted double Sersic; parameter count is
the tie-breaker. The best three structures across both rounds proceed to the
development holdout. The same ranking freezes one finalist there.

This is finite symbolic grammar enumeration with per-galaxy nonlinear fitting,
not regression of one population-average curve through all galaxies.

## Synthetic teacher feasibility gate

Before fitting measured CoGs, every Round-A structure is tested on 15
unrestricted-double-Sersic predictions selected as three compactness-stratified
profiles per epoch from the discovery subset. The grammar is viable only if at
least one expression reaches both a median candidate-to-teacher full-CoG RMS
below 0.003 dex and a 90th-percentile RMS below 0.010 dex, with no failed fit.
Failure closes Exp65 before a galaxy-data structure search.

Synthetic profiles generated exactly from every expression type must also be
recovered from deliberately displaced starts to below 2e-5 dex full-CoG RMS.

## References and fitting objectives

All comparisons are paired on the same profiles.

1. Unrestricted double Sersic is the six-parameter analytic accuracy ceiling.
2. Cubic-logit is the five-parameter conditioning reference and resolved-CoG
   accuracy reference; its exact-center density hole is disallowed here.
3. PCA with three shape modes plus amplitude is the four-coordinate numerical
   compression reference.
4. The constant-`g` wrapper is the nested simplicity control.

The objectives are deterministic losses, not likelihoods:

1. primary: uniform-radius RMS in `log10 Mstar(<R)`;
2. secondary: RMS of linear enclosed-mass difference divided by measured
   `Mstar(<148.2 kpc)`;
3. local structure: RMS in logarithmic shell surface density, including the
   central 0--2 kpc aperture and using the established 1 Msun floor only for
   an exactly zero measured shell.

Expression discovery uses only the primary objective. The frozen finalist is
fitted under all three objectives and judged by full and 5--30 kpc CoG RMS,
shell-density RMS, aperture and 100--148 kpc shell masses, and R20/R50/R80/R90.

## Degeneracy and numerical tests

- at least four separated deterministic starts for the finalist;
- scaled residual-Jacobian singular values and boundary incidence;
- all solutions within 1% of the best loss, reporting raw-coordinate and
  decoded-CoG spread;
- exact synthetic recovery from displaced starts;
- doubled quadrature resolution and positive-annulus monotonicity;
- dense-grid density, logarithmic-slope, tail, and Abel-sign checks;
- alternating-radius and `R >= 5 kpc` refits from the incumbent solution;
- profiled scans of every coordinate and weakest-Jacobian-direction motion;
- a second optimizer on a mass-, compactness-, and epoch-stratified subset;
- fold stability of the selected symbolic structure and global atom constants;
- fitted trig-amplitude incidence near zero and comparison with the selected
  expression after removing its trig atom;
- sensitivity to the adjacent declared frequencies and damping constants, so
  radial-grid aliasing cannot masquerade as a preferred harmonic.

No raw coordinate receives physical interpretation unless these tests pass.

## Sub-minute development gate

Before any population validation, one measured complete path over the 90
development profiles must finish in less than 60 seconds. It includes

1. mathematical and synthetic-recovery checks;
2. the 15-profile double-Sersic teacher feasibility screen;
3. Round-A and Round-B discovery fits and holdout selection;
4. all three objective fits for the frozen finalist;
5. common reference metrics, conditioning and boundary diagnostics;
6. result serialization and two figures: an expression/accuracy/conditioning
   comparison and direct measured/model CoG plus density profiles.

The time includes PNG and PDF rendering under the actual plotting
configuration. If it exceeds 60 measured seconds, the implementation or
grammar must be narrowed and the complete gate rerun; timing is never
extrapolated.

## Decision and stop gates

The development result may trigger full validation only if the finalist
satisfies all hard mathematical checks and, at every epoch:

- median full-CoG, 5--30 kpc CoG, shell-density, and R50/R80/R90 errors are no
  more than 10% larger than paired unrestricted double Sersic;
- at least four epoch medians of the scaled-Jacobian ratio are at least half
  the paired cubic-logit value;
- fewer than 1% of fits place any coordinate within 1% of a bound;
- median near-optimal raw-coordinate spread is below 1% of its allowed range
  and worst decoded near-optimal CoG spread is below 0.001 dex.

The same gates are applied to the 16,810-profile population validation after
excluding development pairs. A passing numerical candidate is not called
promising until direct figures show no coherent failure in average CoGs in
halo- and stellar-mass bins and the stellar-mass planes. Only then is the full
standard QA battery generated.

If the teacher gate fails, no expression passes the 90-profile gate, or the
frozen finalist fails population validation, Exp65 closes. No expression is
manually repaired and no third search round is added after seeing results.

## Required figures

Every generated figure uses `hongshao.plotting`, is saved as PNG and PDF, and
is inspected directly.

1. Expression frontier: discovery and holdout CoG accuracy, conditioning,
   boundary incidence, and parameter count, with trig expressions explicitly
   identified.
2. Direct profiles: measured and modeled CoGs, CoG residuals, projected
   densities, and density residuals for compactness-stratified plus
   best/typical/worst profiles.
3. For a numerically passing population candidate, average CoGs in halo- and
   stellar-mass bins and the stellar-mass planes form the minimum visual gate;
   the remaining standard QA follows only if those panels pass.

## Review checklist

- [x] Confirm the predeclaration predates evaluator and driver code.
- [x] Record measured mechanics and teacher-stop timings. The complete
      measured-profile path was correctly not entered.
- [x] Report every Round-A structure searched. Round B and fitted trig-amplitude
      diagnostics were not triggered by the teacher gate.
- [x] Keep discovery, development-holdout, and population-validation results
      distinct.
- [x] Inspect and surface every generated figure in the generating turn.
- [x] Record the final decision without loosening a gate.

## Results

### Mathematical and numerical mechanics pass

All 13 predeclared Round-A expressions pass the finite positive center,
non-increasing density, positive cumulative mass, asymptotic tail, doubled-grid
agreement, and displaced-start synthetic-recovery checks. The complete
mechanics path took 1.590 measured wall-clock seconds. The focused experiment
suite has nine passing checks.

### The grammar fails before measured-profile structure selection

The teacher bank contains 15 unrestricted-double-Sersic CoGs: three
compactness-stratified discovery profiles at each of five epochs. Every
expression fit succeeds, but none meets both declared approximation limits of
less than 0.003 dex median candidate-to-teacher full-CoG RMS and less than
0.010 dex at the 90th percentile.

The strongest non-trigonometric expression is the sigmoid transition with
fixed width 0.5. Its median candidate-to-double-Sersic CoG RMS is 0.00535 dex
and its 90th-percentile RMS is 0.01177 dex, so it fails both limits.

The trigonometric relaxation materially improves the frontier:

- the best joint compromise, a damped sine with damping 0.5 and frequency 2,
  has 0.00321 dex median candidate-to-double-Sersic CoG RMS, 6.88% above the
  0.00300 dex limit, while its 0.00973 dex 90th-percentile RMS passes the
  0.01000 dex limit;
- the damped sine with damping 1 and frequency 2 passes the median limit at
  0.00263 dex, but its 0.01328 dex 90th-percentile RMS is 32.8% above the
  0.01000 dex limit;
- the damped cosine with damping 0.5 and frequency 1 passes the tail limit at
  0.00562 dex, but its 0.00398 dex median RMS is 32.8% above the 0.00300 dex
  limit.

Repeating these three closest cases with four separated starts, a 512-point
quadrature grid, and twice the original evaluation budget gives respectively
`(median, p90) = (0.00321, 0.00973)`, `(0.00264, 0.01328)`, and
`(0.00398, 0.00562)` dex. The gate failures are therefore stable to the
declared numerical refinement rather than artifacts of the two-start screen.

The mechanics, teacher fits, serialization, and PNG+PDF failure figure reached
the declared stop in 7.565 measured wall-clock seconds. No measured TNG CoG was
used to select an expression, Round B was not generated, and neither the
90-profile measured path nor the 16,810-profile population validation was
run.

## Visual review

`figures/teacher/exp65_teacher_failure.{png,pdf}` shows all 13 expressions
against the two teacher thresholds and the best, typical, and worst direct
double-Sersic approximation by the best compromise harmonic. The trigonometric
expressions clearly occupy the improved part of the frontier, but the worst
direct residual has a coherent change of sign and reaches approximately
0.04 dex at 148 kpc. The failure is structural rather than hidden by the
aggregate statistics.

## Decision

Close Exp65 at the predeclared teacher gate. Bounded trigonometric terms are a
useful addition--they improve substantially over the transition-only grammar--
but one bounded atom cannot reproduce both the typical and difficult
double-Sersic shapes at the accuracy required of a replacement. The experiment
does not authorize adding more atoms after seeing this result.

Together with Exp64, this is a second bounded null after the successful but
centrally unphysical cubic-logit compression. The practical conclusion is to
stop the present analytic-family search: retain cubic-logit only on its
validated 2--148 kpc domain and unrestricted double Sersic as the analytic
accuracy reference. A future search would need a newly motivated constraint
or observable, not another post-result expansion of this grammar.

## Reproduction

```text
uv run python experiments/exp65_constrained_symbolic_density/search.py mechanics
uv run python experiments/exp65_constrained_symbolic_density/search.py demo --force
uv run pytest -q experiments/exp65_constrained_symbolic_density/test_grammar.py \
  experiments/exp65_constrained_symbolic_density/test_search.py
uv run ruff format --check experiments/exp65_constrained_symbolic_density
uv run ruff check experiments/exp65_constrained_symbolic_density
```
