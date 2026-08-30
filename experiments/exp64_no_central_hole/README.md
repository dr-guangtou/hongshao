# exp64 — Centrally finite analytic projected profiles

Status: COMPLETE. No tested family passes the predeclared combined accuracy,
conditioning, boundary, and robustness gates. Nothing is promoted to the
HongShao library.

## Question

Can a compact analytic projected-density family reproduce the measured
2.0--148.2 kpc curves of growth (CoGs) as accurately as unrestricted double
Sersic and as stably as cubic-logit, while guaranteeing a finite, strictly
positive projected density at the exact center?

This is a representation experiment. It does not infer a physical stellar
component decomposition, fit a halo-to-profile relation, or promote a new
library interface. Any surviving density law remains an empirical profile
family until a separate promotion decision.

## Data and sample

- Primary observable: the same 24-point CoG-derived stellar masses from 2.0 to
  148.2 kpc at redshifts 0.4, 0.7, 1.0, 1.5, and 2.0 used by Exp62.
- Development sample: Exp62's 90 unique galaxy--epoch profiles, with 18 at
  each epoch and concurrent-halo-mass stratification.
- Validation sample: every finite, positive, monotonic CoG at each epoch,
  excluding the 90 development pairs when a decision is reported.
- Mass convention: h-free `log10(M / Msun)`. The fitted amplitude is
  `log_mstar_148`, the stellar mass inside the measured 148.2 kpc aperture.
- Exp62's stored, regenerable prediction record supplies paired reference
  predictions so this experiment does not refit or alter Exp62.

## Hard mathematical constraints

Every eligible family must satisfy these properties for every allowed
parameter vector, not merely for the fitted population:

1. `Sigma(R)` is finite and strictly positive at `R = 0`.
2. `Sigma(R) > 0` and `d Sigma / dR <= 0` for every `R >= 0`.
3. `M(<R) = 2 pi integral_0^R Sigma(r) r dr` is finite, nonnegative, and
   strictly increasing for positive radius.
4. The infinite projected mass is finite. The outer projected-density slope
   must therefore be steeper than `R^-2`.
5. The primary interface is normalized by `Mstar(<148.2 kpc)`; an inferred
   infinite total is always reported as extrapolation, never as observed mass.
6. Ordered break radii are enforced in the optimizer coordinates. A profile
   with merged breaks remains mathematically allowed but is diagnosed as an
   effectively simpler and potentially degenerate fit.
7. Because an everywhere non-increasing projected density has a nonnegative
   spherical Abel deprojection, the sign of that deprojection is checked
   numerically on representative allowed parameters. No claim of spherical
   galaxy structure follows from this mathematical check.

## Candidate families

### One-break control

The one-break cored power law is

```text
Sigma(R) = Sigma_0 [1 + (R / r_1)^k_1]^(-gamma_out / k_1),
```

with `r_1 > 0` and `gamma_out > 2`. Its galaxy-level coordinates are
`log_mstar_148`, `log_r_1`, and `gamma_out`. The transition sharpness is a
training-selected global value from `k_1 = (1, 2, 4)`. This control measures
whether the second break earns its freedom.

### Two-break candidate

The primary family is

```text
Sigma(R) = Sigma_0
           [1 + (R / r_1)^k_1]^(-gamma_mid / k_1)
           [1 + (R / r_2)^k_2]^(-(gamma_out - gamma_mid) / k_2),
```

with

```text
0 < r_1 < r_2,
0 < gamma_mid < gamma_out,
gamma_out > 2.
```

It has five galaxy-level coordinates including `log_mstar_148`: two ordered
break radii, the intermediate projected-density slope, and the asymptotic
outer slope. Its exact central density is `Sigma_0 > 0`; its central logarithmic
slope is zero; its intermediate and outer slopes are `-gamma_mid` and
`-gamma_out` when the two breaks are separated.

The deliberately finite global sharpness grid is

```text
(k_1, k_2) = (1,1), (1,2), (2,1), (2,2), (2,4), (4,2), (4,4).
```

Sharpness pairs are selected using training galaxies only within five fixed
galaxy folds and separately at each epoch. The held-out profile never selects
its own sharpness pair.

### Degeneracy stress test

A six-parameter version fits one common positive sharpness `k_1 = k_2` per
galaxy. It may diagnose missing flexibility, but it cannot be selected as the
preferred compact representation unless its extra coordinate is identifiable
under every declared degeneracy test.

No component labels are attached to either break. In particular, the two
slope changes are not identified as in-situ and ex-situ stellar populations.

## Reference representations

All comparisons are paired on the same galaxy--epoch profiles.

1. Unrestricted double Sersic is the six-parameter analytic accuracy
   reference. Exp62 measured a 0.00263 dex median full-CoG RMS across epoch
   medians, but found unstable raw component shapes and a 14.44% boundary
   incidence.
2. Cubic-logit is the five-parameter accuracy-and-conditioning reference.
   Exp62 measured a 0.00249 dex median full-CoG RMS for its log-CoG fit, no
   fitted boundary hits, and approximately twelve times the local scaled-
   Jacobian ratio of unrestricted double Sersic. Its forced `Sigma(0) = 0` is
   the defect Exp64 must remove.
3. PCA with three shape modes plus amplitude is the four-coordinate numerical
   compression reference. It is not an analytic density law.
4. The one-break cored power law is the nested simplicity control.

## Fitting objectives

These are deterministic fitting losses, not likelihoods; no validated radial
covariance is available for the stored CoGs.

1. Primary: uniform-radius RMS in `log10 Mstar(<R)` over all 24 radii.
2. Secondary: RMS of the linear enclosed-mass difference divided by measured
   `Mstar(<148.2 kpc)`.
3. Local-structure sensitivity: RMS in logarithmic shell surface density,
   obtained by differencing adjacent CoG apertures. The central 0--2 kpc
   aperture is included as one finite-area shell; it must not be discarded as
   in a density-only difference objective. A measured shell with exactly zero
mass is floored at 1 Msun before taking its logarithm, matching Exp62; a
   negative candidate-family shell remains an error. For the paired PCA(3)
   reference only, a nonpositive reconstructed shell is floored at 1 Msun in
   the common diagnostic metric because PCA does not enforce monotonicity.

The global sharpness grid is selected under the primary objective. The
selected two-break family is then refitted under all three objectives. Every
fit is judged under the same complete metric set regardless of its fitting
objective: full and 5--30 kpc CoG RMS, shell-density RMS, errors in the 2, 5,
10, 30, 50, 100, and 148 kpc aperture masses, errors in R20/R50/R80/R90, and
the 100--148 kpc shell-mass error.

## Degeneracy and robustness tests

Before interpreting any raw coordinate, run all of the following:

- exact synthetic recovery from deliberately displaced starts;
- at least four widely separated deterministic starts on every real fit;
- boundary incidence and agreement of all solutions within 1% of the best
  objective value;
- the smallest-to-largest singular-value ratio of the residual Jacobian after
  scaling each coordinate by its allowed range;
- refits after removing alternating radii and after removing every radius
  below 5 kpc;
- profiled scans of each coordinate with every other coordinate reoptimized;
- a second optimizer on a mass-, compactness-, and epoch-stratified subset;
- explicit rates of merged breaks, vanishing intermediate-slope intervals,
  extreme outer slopes, and large infinite-total corrections;
- fold-to-fold stability of the globally selected sharpness pair;
- perturbation along the weakest local Jacobian direction, comparing raw
  coordinate motion with the decoded CoG motion.

An accurate decoded profile may be retained while its raw coordinates are
barred from physical interpretation.

## Sub-minute development gate

The complete development path must run on all 90 predeclared profiles in less
than 60 measured wall-clock seconds before any validation-population fit. It
must include:

1. every mathematical mechanics check;
2. the one-break controls and complete two-break sharpness grid under the
   primary objective;
3. fold-clean sharpness selection;
4. all three objective fits for the selected two-break family;
5. paired reference metrics, degeneracy diagnostics, result serialization,
   and the development figures.

If the measured path exceeds 60 seconds, improve or narrow the implementation
and rerun the entire gate. Do not extrapolate timing and do not start the full
sample until the gate passes.

## Decision gates

A family is a successful no-hole replacement only if all hard mathematical
constraints pass and, on validation galaxies excluded from development:

- at every epoch its median full-CoG, 5--30 kpc CoG, shell-density, and
  R50/R80/R90 errors are no more than 10% larger than the paired unrestricted
  double-Sersic values;
- at least four of five epoch medians of its scaled-Jacobian ratio are at least
  half the paired cubic-logit value;
- fewer than 1% of fits place any coordinate within 1% of an allowed bound;
- its median near-optimal raw-coordinate spread is below 1% of an allowed
  coordinate range, while its worst decoded near-optimal CoG spread remains
  below 0.001 dex;
- synthetic recovery, alternating-radius refits, the second optimizer, and
  profiled scans do not reveal an unreported weak coordinate;
- direct visual QA shows no coherent failure in average CoGs binned by halo or
  stellar mass, the stellar-mass planes, or representative best, typical, and
  worst objects.

If no family passes, Exp64 closes with a measured trade space rather than
loosening these gates after seeing the result.

## Stage 2 predeclaration — monotone-density envelope

The first development gate is allowed to trigger one mechanically distinct
follow-up, but its form and gates must be frozen before its evaluator or driver
is written. The trigger is now met: the fixed-sharpness two-break density fails
the declared accuracy and conditioning comparison on the 90-profile sample,
while its free-sharpness stress test places a coordinate within 1% of a bound
for the median profile at four of five epochs.

Let `Sigma_cubic(R; theta)` be the projected density of the five-coordinate
cubic-logit profile. Define its least non-increasing radial majorant by

```text
Sigma_envelope(R; theta) = sup_{s >= R} Sigma_cubic(s; theta).
```

This is the smallest non-increasing projected density that is nowhere below
the original density. It adds no per-galaxy coordinate. The CoG is obtained
from `2 pi integral Sigma_envelope(R) R dR` and normalized by the measured
mass inside 148.2 kpc.

For every allowed cubic-logit shape, the envelope must be finite and strictly
positive at the center, positive and non-increasing at every radius, monotonic
in cumulative mass, and finite in total mass. Its spherical Abel deprojection
must therefore be nonnegative. The envelope may have slope corners where a
future density maximum overtakes the local density; those locations and the
enclosed mass affected by the envelope are explicit diagnostics, not hidden
smoothness.

The existing cubic-logit parameter bounds, aperture-mass convention, and four
separated starts remain fixed. The primary, secondary, and local-structure
losses remain log CoG, aperture-normalized linear CoG, and logarithmic shell
density with the same measured-zero-shell convention. Paired unrestricted
double Sersic, original cubic-logit, and PCA(3) remain the references. The
envelope is not allowed to inherit the original cubic-logit's conditioning
score without measurement: its finite-difference residual Jacobian is
recomputed after the envelope transform.

Additional degeneracy tests are:

- convergence of the radial grid and cumulative quadrature;
- agreement between a direct reverse-maximum construction and an independent
  stationary-point construction on representative shapes;
- fraction of the measured 2--148 kpc density and enclosed mass changed by the
  envelope;
- number and locations of slope corners, plus sensitivity when a corner moves
  across a measured radius;
- coordinate motion from the original cubic-logit fit after reoptimization;
- the same displaced-start, bounds, near-optimal spread, radial-jackknife,
  weakest-direction, profiled-scan, and alternate-optimizer checks declared
  for Stage 1.

Before any validation-population calculation, the complete envelope path must
fit all three objectives on the same 90 profiles, recompute conditioning and
all common metrics, serialize results, and generate direct CoG/density plus
accuracy/conditioning figures in less than 60 measured wall-clock seconds.
The successful no-hole replacement gates are unchanged. If the envelope fails
them, Exp64 closes rather than adding another post-result family.

## Required figures

Every figure is saved as PNG and PDF with `hongshao.plotting`.

1. Candidate/reference accuracy, boundary incidence, and scaled-Jacobian
   conditioning by epoch.
2. Measured and fitted CoGs, residual CoGs, projected densities, and density
   residuals for mass-stratified plus best/typical/worst galaxies.
3. Parameter-recovery, weakest-direction, profiled-scan, and radial-jackknife
   diagnostics.
4. Average CoGs and residuals in concurrent halo-mass and stellar-mass bins.
5. Aperture, annular, outskirt, size, observational, and stellar-mass planes;
   cumulative residual distributions; and 100--148 kpc shell-mass behavior.

The first two development figures are part of the sub-minute gate. A candidate
that clears the numerical gate receives the complete population QA battery
before it is called promising.

## Execution order

1. Write this predeclaration and its focused mechanics checks.
2. Write the experiment driver only after the predeclaration exists.
3. Run the focused checks, then the complete 90-profile gate.
4. Only after a measured sub-minute pass, run the validation population.
5. Record measured results and the decision here; do not promote library code
   or merge the branch without explicit permission.

Stage 2 follows the same order: this added predeclaration precedes its focused
checks, evaluator, and separate development driver.

## Results

### Mechanics and measured development gates

All 37 focused Exp64 checks pass. They cover the central and radial limits,
positive annular integration, aperture normalization, quadrature convergence,
nonnegative spherical Abel deprojection, independent stationary-point
construction of the envelope, and displaced-start synthetic recovery under
all three fitting losses.

The complete first-stage path over 90 development galaxy--epoch profiles took
30.35 seconds, below the predeclared 60-second limit. The complete monotone-
density-envelope path over the same profiles took 9.43 seconds after the final
reporting corrections, also below 60 seconds. Both timings include mechanics,
fits, metrics, serialization, and two PNG+PDF figures.

### The direct two-break density is a null result

The fold-selected fixed-sharpness two-break density reaches median full-CoG
RMS errors of 0.00216--0.00501 dex across the five development epochs, where
the paired unrestricted double-Sersic reference reaches 0.00181--0.00342 dex;
smaller is better. Its median scaled-Jacobian ratio is
0.000123--0.00251, where the better-conditioned cubic-logit reference reaches
0.00168--0.01336; larger is better. Thus the family misses both the declared
accuracy and conditioning targets.

Allowing one common transition sharpness per galaxy improves the development
full-CoG error to 0.00165--0.00387 dex, but the fitted sharpness lies within 1%
of an allowed bound for the median galaxy at four of five epochs and its local
conditioning is worse. The extra coordinate is absorbing residual structure
without becoming identifiable. The one-break control is much less accurate,
with 0.00720--0.01037 dex median full-CoG RMS. Therefore neither cored
power-law family proceeds to the full population.

### The monotone-density envelope fixes the center but fails replacement gates

The full calculation fits 16,900 galaxy--epoch profiles under each of the three
losses and takes 712.96 seconds. Every decision below excludes the 90
development pairs and uses 16,810 validation profiles.

For the primary log-CoG fit, the monotone-density envelope has median full-CoG
RMS errors of 0.00227--0.00263 dex across redshifts 0.4--2, compared with
0.00198--0.00275 dex for unrestricted double Sersic and
0.00227--0.00255 dex for original cubic-logit. The envelope is more accurate
than double Sersic at redshifts 0.4, 0.7, and 1, 4% worse at redshift 1.5, and
15% worse at redshift 2. It therefore fails the requirement of staying within
10% of double Sersic at every epoch.

The same log-CoG fit has median shell-density RMS errors of
0.0571--0.2036 dex, compared with 0.0497--0.1762 dex for double Sersic. It
exceeds the allowed double-Sersic error at redshifts 0.4, 1.5, and 2. Its
R50/R80/R90 errors also fail at least one declared epoch comparison. This
means the no-hole transform retains cubic-logit's cumulative accuracy but does
not acquire double-Sersic local-density and size accuracy.

The envelope's median scaled-Jacobian ratio is 0.00462--0.01174 across epochs,
compared with 0.00462--0.01179 for original cubic-logit. It passes the
conditioning comparison at every epoch. Reoptimization moves the largest raw
coordinate by a median 0.414% of its allowed range, so the envelope has not
created a new broad weak direction for a typical fit.

Conditioning alone is insufficient. Across validation profiles, 4.94% place a
coordinate within 1% of a bound, above the declared 1% maximum. The positive
derivative-margin coordinate accounts for 3.10% of all validation fits, the
scale radius for 1.23%, and the cubic-curvature coordinate for 0.625%.
Near-optimal starts differ by more than the allowed 0.001 dex in decoded CoG
for one validation profile, reaching 0.00171 dex. The envelope therefore also
fails the boundary and near-solution robustness gates.

Mechanically, the central repair adds a median 0.718% of the envelope model's
mass inside 148.2 kpc relative to the original cubic-logit density. It changes
no sampled density point between 2 and 148 kpc for the median galaxy, and only
2.50% of validation profiles change any sampled density point by more than
0.2%. The maximum affected mass fraction is nevertheless 24.1%. Thus the
repair is usually an unresolved extrapolation change but is not uniformly
small enough to treat as harmless.

Changing the fitting loss reproduces the Exp62 trade-off rather than finding a
universal solution. The aperture-normalized linear-CoG loss improves density
and outer-size errors but worsens full-CoG RMS to 0.00244--0.00390 dex. The
log-shell loss improves shell-density RMS to 0.0339--0.1151 dex but worsens
full-CoG RMS to 0.00502--0.00940 dex and permits nearly identical loss values
whose decoded CoGs differ by as much as 0.419 dex at redshift 2. These losses
therefore do not rescue the family.

## Decision

Exp64 closes as a measured null for a replacement family. A direct cored
two-break density cannot simultaneously reproduce the CoG and remain well
conditioned. The least non-increasing density envelope is a mathematically
clean way to remove cubic-logit's central hole without adding a coordinate,
and it preserves cubic-logit conditioning and resolved cumulative accuracy,
but it does not meet the predeclared double-Sersic accuracy, boundary, or
near-solution robustness gates. It remains experiment-local and is not a new
HongShao profile.

The result means the central-hole defect can be repaired as an unresolved
extrapolation, but that repair does not solve the separate finite-resolution
trade-off between cumulative accuracy, local-density accuracy, and parameter
identifiability. No additional post-result family is added to Exp64.

## Review

- The experiment followed the predeclared order: README, failing focused
  checks, family implementation, sub-minute gate, then full validation.
- The two failed development runs exposed numerical shell-mass edge cases;
  both are recorded in `doc/lessons.md` and covered by regression checks.
- The full standard population QA battery was not run because no candidate
  passed the numerical gate or was called promising. The required accuracy,
  conditioning, and representative direct-profile figures were generated and
  visually inspected at development and full scale.
- Repository-wide pytest collection remains intentionally unrun because the
  unrelated gitignored Exp07 input is absent. Focused Exp64 and Exp62 reference
  checks are the verification target.

## Commands

```bash
uv run python experiments/exp64_no_central_hole/run.py mechanics
uv run python experiments/exp64_no_central_hole/run.py demo --force
uv run python experiments/exp64_no_central_hole/stage2_envelope.py mechanics
uv run python experiments/exp64_no_central_hole/stage2_envelope.py demo --force
uv run python experiments/exp64_no_central_hole/stage2_envelope.py full --force
uv run pytest -q experiments/exp64_no_central_hole/test_exp64.py \
  experiments/exp64_no_central_hole/test_envelope.py
```
